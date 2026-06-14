from __future__ import annotations
"""Cross-modal CKA between vision and language model layer activations.

Computes pairwise CKA matrices for:
  - vision model (e.g. ViT-B/32 CLIP or ResNet) layers  vs.  LLM layers
  - one matrix per LLM (base, instruct, clip-text upper bound)

All inputs must be matched: row i in vision_layers[k] corresponds to the
same image/caption pair as row i in language_layers[k].
"""

import pickle
import numpy as np
from pathlib import Path

from app.vision_compute import linear_cka

CACHE_DIR = Path("data/cache")


def _linear_cka_samples(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA via sample Gram matrices — O(n^2 * d) instead of O(n * d^2).

    Equivalent to linear_cka but efficient when n << d (e.g. n=500, d=4096).
    """
    X = (X - X.mean(axis=0)).astype(np.float64)
    Y = (Y - Y.mean(axis=0)).astype(np.float64)
    K = X @ X.T
    L = Y @ Y.T
    numerator = float(np.sum(K * L))
    denominator = float(np.linalg.norm(K, "fro") * np.linalg.norm(L, "fro"))
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def compute_crossmodal_cka_matrix(
    vision_layers: dict[int, np.ndarray],
    language_layers: dict[int, np.ndarray],
) -> np.ndarray:
    """Pairwise CKA: shape (n_vision_layers, n_language_layers).

    vision_layers[i]:   (n_samples, vision_dim)
    language_layers[j]: (n_samples, language_dim)
    """
    vis_keys = sorted(vision_layers.keys())
    lang_keys = sorted(language_layers.keys())
    matrix = np.zeros((len(vis_keys), len(lang_keys)), dtype=np.float32)
    for i, vi in enumerate(vis_keys):
        for j, lj in enumerate(lang_keys):
            matrix[i, j] = _linear_cka_samples(
                vision_layers[vi].astype(np.float32),
                language_layers[lj].astype(np.float32),
            )
    return matrix


def fit_crossmodal(
    vision_layers: dict[int, np.ndarray],
    language_layers_base: dict[int, np.ndarray],
    language_layers_instruct: dict[int, np.ndarray],
    language_layers_clip: dict[int, np.ndarray] | None = None,
    cache_key: str = "default",
) -> dict:
    """Compute cross-modal CKA matrices for base, instruct, and optionally CLIP.

    Returns dict with keys:
      "base":       (n_vision_layers, n_lang_layers) float32
      "instruct":   (n_vision_layers, n_lang_layers) float32
      "clip":       (n_vision_layers, n_lang_layers) float32 | None
      "vision_layer_names":  list[str]
      "language_layer_names": list[str]
    """
    cache = CACHE_DIR / "crossmodal" / f"{cache_key}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    print("Computing cross-modal CKA matrices...")
    base_matrix = compute_crossmodal_cka_matrix(vision_layers, language_layers_base)
    instruct_matrix = compute_crossmodal_cka_matrix(
        vision_layers, language_layers_instruct
    )
    clip_matrix = (
        compute_crossmodal_cka_matrix(vision_layers, language_layers_clip)
        if language_layers_clip is not None
        else None
    )

    result = {
        "base": base_matrix,
        "instruct": instruct_matrix,
        "clip": clip_matrix,
        "vision_layer_names": [f"layer {i}" for i in sorted(vision_layers.keys())],
        "language_layer_names": [
            f"layer {j}" for j in sorted(language_layers_base.keys())
        ],
    }

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(result, f)

    print(f"Cross-modal CKA matrix shape: {base_matrix.shape}")
    return result


def mean_crossmodal_cka(matrix: np.ndarray) -> float:
    """Scalar summary: mean CKA across all layer pairs."""
    return float(matrix.mean())


def max_crossmodal_cka(matrix: np.ndarray) -> float:
    """Scalar summary: peak CKA across all layer pairs."""
    return float(matrix.max())
