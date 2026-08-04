"""Per-layer, per-phase UMAP projections of Part 1 vision and language activations.

Lets the dashboard show how the 2D layout of same-category points shifts across
layers and between the phase_1 baseline and the full_train_set checkpoint.
"""

import pickle
from pathlib import Path

import h5py
import numpy as np
import umap as umap_lib
from joblib import Parallel, delayed

from app.part1_phase_compute import _load_all_layers, _finite_layers, _N_VISION_LAYERS, _N_LANGUAGE_LAYERS

CACHE_DIR = Path("data/cache")


def _fit_umap_layer(
    key: tuple[str, str, int], X: np.ndarray
) -> tuple[tuple[str, str, int], np.ndarray]:
    X = X.astype(np.float32)
    reducer = umap_lib.UMAP(
        n_neighbors=min(15, X.shape[0] - 1),
        min_dist=0.1,
        n_components=2,
        random_state=42,
        low_memory=True,
    )
    Z = reducer.fit_transform(X).astype(np.float32)
    modality, phase, layer_idx = key
    print(f"  part1 umap [{modality}/{phase}] layer {layer_idx} done")
    return key, Z


def fit_part1_umap(
    vision_phase1_path: str,
    vision_full_path: str,
    language_phase1_path: str,
    language_full_path: str,
    language_phase2_path: str,
    max_samples: int = 1500,
    cache_key: str = "part1_umap",
) -> dict:
    """UMAP per layer per phase for vision and language, colored by shared labels.

    Vision only has a matched snapshot at phase_1 and full_train_set. Language
    additionally has cka_phase_2 (no vision counterpart), so its layer set is
    the finite intersection across all three phases while vision's is the
    intersection across its two.

    Returns {"labels": (N,), "vision": {"phase1": {layer: (N,2)}, "full_train": {...}},
    "language": {"phase1": {...}, "phase2": {...}, "full_train": {...}}}.
    """
    cache = CACHE_DIR / f"{cache_key}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    vis_p1_all = _load_all_layers(vision_phase1_path, _N_VISION_LAYERS, max_samples)
    vis_full_all = _load_all_layers(vision_full_path, _N_VISION_LAYERS, max_samples)
    lang_p1_all = _load_all_layers(language_phase1_path, _N_LANGUAGE_LAYERS, max_samples)
    lang_full_all = _load_all_layers(language_full_path, _N_LANGUAGE_LAYERS, max_samples)
    lang_p2_all = _load_all_layers(language_phase2_path, _N_LANGUAGE_LAYERS, max_samples)

    with h5py.File(vision_phase1_path, "r") as f:
        labels = f["labels"][:max_samples]

    vision_clean = _finite_layers(vis_p1_all, vis_full_all)
    language_clean = _finite_layers(lang_p1_all, lang_p2_all, lang_full_all)
    print(f"  Part 1 UMAP: {len(vision_clean)}/{_N_VISION_LAYERS} vision layers, "
          f"{len(language_clean)}/{_N_LANGUAGE_LAYERS} language layers are float16-clean")

    tasks = (
        [(("vision", "phase1", i), vis_p1_all[i]) for i in vision_clean]
        + [(("vision", "full_train", i), vis_full_all[i]) for i in vision_clean]
        + [(("language", "phase1", i), lang_p1_all[i]) for i in language_clean]
        + [(("language", "phase2", i), lang_p2_all[i]) for i in language_clean]
        + [(("language", "full_train", i), lang_full_all[i]) for i in language_clean]
    )

    fitted = Parallel(n_jobs=-1, verbose=10)(
        delayed(_fit_umap_layer)(key, X) for key, X in tasks
    )

    result: dict = {
        "labels": labels,
        "vision": {"phase1": {}, "full_train": {}},
        "language": {"phase1": {}, "phase2": {}, "full_train": {}},
    }
    for (modality, phase, layer_idx), Z in fitted:
        result[modality][phase][layer_idx] = Z

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(result, f)

    return result
