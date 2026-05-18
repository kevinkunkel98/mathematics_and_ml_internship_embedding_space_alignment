import pickle
import numpy as np
from pathlib import Path
import umap as umap_lib

CACHE_DIR = Path("data/cache")

_CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA similarity between two activation matrices (n_samples x dim).

    Centers each matrix before computing to remove mean effects.
    Returns value in [0, 1] where 1 = identical representations.
    """
    X = (X - X.mean(axis=0)).astype(np.float64)
    Y = (Y - Y.mean(axis=0)).astype(np.float64)
    xty = X.T @ Y
    xtx = X.T @ X
    yty = Y.T @ Y
    numerator = float(np.linalg.norm(xty, "fro") ** 2)
    denominator = float(np.linalg.norm(xtx, "fro") * np.linalg.norm(yty, "fro"))
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def compute_cka_matrix(
    cnn_activations: dict[int, np.ndarray],
    vit_activations: dict[int, np.ndarray],
) -> np.ndarray:
    """Compute pairwise CKA matrix: shape (n_cnn_layers, n_vit_layers)."""
    cnn_layers = sorted(cnn_activations.keys())
    vit_layers = sorted(vit_activations.keys())
    matrix = np.zeros((len(cnn_layers), len(vit_layers)), dtype=np.float32)
    for i, ci in enumerate(cnn_layers):
        for j, vi in enumerate(vit_layers):
            matrix[i, j] = linear_cka(
                cnn_activations[ci].astype(np.float32),
                vit_activations[vi].astype(np.float32),
            )
    return matrix


def fit_vision(
    cnn_activations: dict[int, np.ndarray],
    vit_activations: dict[int, np.ndarray],
) -> dict:
    cache = CACHE_DIR / "vision" / "cka.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    print("Computing CKA matrix...")
    cka_matrix = compute_cka_matrix(cnn_activations, vit_activations)
    cnn_layer_names = [f"layer {i}" for i in sorted(cnn_activations.keys())]
    vit_layer_names = [f"layer {i}" for i in sorted(vit_activations.keys())]

    result = {
        "cka_matrix": cka_matrix,
        "cnn_layer_names": cnn_layer_names,
        "vit_layer_names": vit_layer_names,
    }

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(result, f)

    print(f"CKA matrix shape: {cka_matrix.shape}")
    return result


def fit_vision_umap(
    cnn_activations: dict[int, np.ndarray],
    vit_activations: dict[int, np.ndarray],
    max_samples: int = 1000,
) -> dict:
    """Fit UMAP on each layer's activations for CNN and ViT.

    Returns {"resnet18": {layer_idx: (N, 2)}, "vit_b16": {layer_idx: (N, 2)}}.
    """
    cache = CACHE_DIR / "vision" / "umap.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    print("Fitting vision UMAP...")
    result: dict[str, dict[int, np.ndarray]] = {}
    for slug, acts in [("resnet18", cnn_activations), ("vit_b16", vit_activations)]:
        result[slug] = {}
        for layer_idx, X in sorted(acts.items()):
            X = X[:max_samples].astype(np.float32)
            reducer = umap_lib.UMAP(
                n_neighbors=min(15, X.shape[0] - 1),
                min_dist=0.1,
                n_components=2,
                random_state=42,
                low_memory=True,
            )
            result[slug][layer_idx] = reducer.fit_transform(X).astype(np.float32)
            print(f"  [{slug}] layer {layer_idx} done")

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(result, f)

    return result
