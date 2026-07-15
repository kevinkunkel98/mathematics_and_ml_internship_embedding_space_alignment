import pickle
from pathlib import Path

from app.vision_compute import linear_cka, compute_cka_matrix
from scripts.io import load_embeddings

CACHE_PATH = Path("data/cache/rlhf_drift/cka.pkl")
MATRIX_CACHE_PATH = Path("data/cache/rlhf_drift/cka_matrix.pkl")


def fit_rlhf_drift(sft_path: Path, dpo_path: Path, rlhf_path: Path) -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    sft, _ = load_embeddings(sft_path)
    dpo, _ = load_embeddings(dpo_path)
    rlhf, _ = load_embeddings(rlhf_path)

    layers = sorted(sft.keys())
    result: dict = {"layers": layers, "sft_dpo": [], "dpo_rlhf": [], "sft_rlhf": []}
    for layer_idx in layers:
        result["sft_dpo"].append(linear_cka(sft[layer_idx], dpo[layer_idx]))
        result["dpo_rlhf"].append(linear_cka(dpo[layer_idx], rlhf[layer_idx]))
        result["sft_rlhf"].append(linear_cka(sft[layer_idx], rlhf[layer_idx]))

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(result, f)

    return result


def fit_rlhf_cka_matrices(sft_path: Path, dpo_path: Path, rlhf_path: Path) -> dict:
    """Full layer-by-layer CKA matrices between checkpoints (not just matched-index drift)."""
    if MATRIX_CACHE_PATH.exists():
        with open(MATRIX_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    sft, _ = load_embeddings(sft_path)
    dpo, _ = load_embeddings(dpo_path)
    rlhf, _ = load_embeddings(rlhf_path)

    layer_names = [f"layer {i}" for i in sorted(sft.keys())]
    result: dict = {
        "layer_names": layer_names,
        "sft_dpo": compute_cka_matrix(sft, dpo),
        "dpo_rlhf": compute_cka_matrix(dpo, rlhf),
        "sft_rlhf": compute_cka_matrix(sft, rlhf),
    }

    MATRIX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MATRIX_CACHE_PATH, "wb") as f:
        pickle.dump(result, f)

    return result
