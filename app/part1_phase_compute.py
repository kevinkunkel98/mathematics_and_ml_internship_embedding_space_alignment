"""Per-supercategory cross-modal CKA between vision and language activations,
compared across Part 1 training phases (phase_1 baseline vs. full_train_set).

Only these two phases have a matched vision+language snapshot — cka_phase_2
has no corresponding vision extraction, so it's excluded from this comparison.
"""

import pickle
from pathlib import Path

import h5py
import numpy as np

from app.coco_categories import LABEL_TO_SUPERCATEGORY, SUPERCATEGORIES
from app.vision_compute import linear_cka

CACHE_DIR = Path("data/cache")

_MIN_SAMPLES_PER_CATEGORY = 20


def _load_layer(path: str, layer: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        activations = f[layer][:].astype(np.float32)
        labels = f["labels"][:]
    return activations, labels


def compute_supercategory_alignment(
    vision_phase1_path: str,
    vision_full_path: str,
    language_phase1_path: str,
    language_full_path: str,
    vision_layer: str = "layer_19",
    language_layer: str = "layer_28",
    cache_key: str = "part1_supercategory",
) -> dict:
    """Per-supercategory CKA(vision, language) at phase_1 and full_train_set.

    Returns dict with keys "categories", "phase1", "full_train", "n_samples".
    Categories with fewer than _MIN_SAMPLES_PER_CATEGORY matched samples are
    dropped (too few points for a stable CKA estimate).
    """
    cache = CACHE_DIR / f"{cache_key}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    vis_p1, labels_p1 = _load_layer(vision_phase1_path, vision_layer)
    vis_full, labels_full_v = _load_layer(vision_full_path, vision_layer)
    lang_p1, labels_p1_l = _load_layer(language_phase1_path, language_layer)
    lang_full, labels_full_l = _load_layer(language_full_path, language_layer)

    assert np.array_equal(labels_p1, labels_p1_l), "phase_1 vision/language label mismatch"
    assert np.array_equal(labels_full_v, labels_full_l), "full_train_set vision/language label mismatch"
    assert np.array_equal(labels_p1, labels_full_v), "phase_1/full_train_set label mismatch"

    labels = labels_p1
    supercats = np.array([LABEL_TO_SUPERCATEGORY[int(l)] for l in labels])

    categories, phase1_scores, full_scores, n_samples = [], [], [], []
    for name in SUPERCATEGORIES:
        mask = supercats == name
        n = int(mask.sum())
        if n < _MIN_SAMPLES_PER_CATEGORY:
            continue
        categories.append(name)
        n_samples.append(n)
        phase1_scores.append(linear_cka(vis_p1[mask], lang_p1[mask]))
        full_scores.append(linear_cka(vis_full[mask], lang_full[mask]))

    result = {
        "categories": categories,
        "phase1": phase1_scores,
        "full_train": full_scores,
        "n_samples": n_samples,
    }

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(result, f)

    return result


_N_VISION_LAYERS = 25
_N_LANGUAGE_LAYERS = 29
_N_SAMPLES_FOR_MATRIX = 2000


def _load_all_layers(path: str, n_layers: int, n_samples: int) -> dict[int, np.ndarray]:
    with h5py.File(path, "r") as f:
        n = min(n_samples, f["labels"].shape[0])
        return {i: f[f"layer_{i:02d}"][:n].astype(np.float32) for i in range(n_layers)}


def _finite_layers(*layer_dicts: dict[int, np.ndarray]) -> list[int]:
    """Layer ids that are entirely finite (no float16 inf/nan) in every given dict."""
    common = set(layer_dicts[0].keys())
    for d in layer_dicts[1:]:
        common &= set(d.keys())
    return sorted(i for i in common if all(np.isfinite(d[i]).all() for d in layer_dicts))


def compute_phase_cka_matrices(
    vision_phase1_path: str,
    vision_full_path: str,
    language_phase1_path: str,
    language_full_path: str,
    cache_key: str = "part1_phase_matrices",
) -> dict:
    """Full vision-layer x language-layer CKA matrices, phase_1 vs. full_train_set.

    Restricted to the layers that are float16-clean (no inf/nan) in both
    compared phases, and subsampled to _N_SAMPLES_FOR_MATRIX rows (matmul
    cost scales linearly in n; 2000 rows keeps this a few minutes instead
    of 15+). Which layers overflow float16 differs by phase and modality —
    detected here from the actual loaded data rather than assumed.
    """
    cache = CACHE_DIR / f"{cache_key}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    vis_p1_all = _load_all_layers(vision_phase1_path, _N_VISION_LAYERS, _N_SAMPLES_FOR_MATRIX)
    vis_full_all = _load_all_layers(vision_full_path, _N_VISION_LAYERS, _N_SAMPLES_FOR_MATRIX)
    lang_p1_all = _load_all_layers(language_phase1_path, _N_LANGUAGE_LAYERS, _N_SAMPLES_FOR_MATRIX)
    lang_full_all = _load_all_layers(language_full_path, _N_LANGUAGE_LAYERS, _N_SAMPLES_FOR_MATRIX)

    vision_clean = _finite_layers(vis_p1_all, vis_full_all)
    language_clean = _finite_layers(lang_p1_all, lang_full_all)
    print(f"  Part 1 matrix: {len(vision_clean)}/{ _N_VISION_LAYERS} vision layers, "
          f"{len(language_clean)}/{_N_LANGUAGE_LAYERS} language layers are float16-clean")

    vis_p1 = {i: vis_p1_all[i] for i in vision_clean}
    vis_full = {i: vis_full_all[i] for i in vision_clean}
    lang_p1 = {i: lang_p1_all[i] for i in language_clean}
    lang_full = {i: lang_full_all[i] for i in language_clean}

    def _matrix(vision_layers: dict, language_layers: dict) -> np.ndarray:
        vis_keys = sorted(vision_layers.keys())
        lang_keys = sorted(language_layers.keys())
        matrix = np.zeros((len(vis_keys), len(lang_keys)), dtype=np.float32)
        for i, vi in enumerate(vis_keys):
            for j, lj in enumerate(lang_keys):
                matrix[i, j] = linear_cka(vision_layers[vi], language_layers[lj])
        return matrix

    result = {
        "phase1": _matrix(vis_p1, lang_p1),
        "full_train": _matrix(vis_full, lang_full),
        "vision_layer_names": [f"layer {i}" for i in vision_clean],
        "language_layer_names": [f"layer {i}" for i in language_clean],
    }

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(result, f)

    return result


def mean_cka(matrix: np.ndarray) -> float:
    return float(matrix.mean())


def max_cka(matrix: np.ndarray) -> float:
    return float(matrix.max())
