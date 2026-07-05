import pickle
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from sklearn.manifold import TSNE

CACHE_DIR = Path("data/cache")


def _cache_path(model_slug: str) -> Path:
    return CACHE_DIR / model_slug / "projections.pkl"


def _fit_layer(layer_idx: int, X: np.ndarray, labels: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, float]:
    X32 = X.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X32)

    umap_proj = UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, random_state=42
    ).fit_transform(X_scaled)

    tsne_proj = TSNE(
        perplexity=min(30, len(X_scaled) - 1), n_components=2, random_state=42
    ).fit_transform(X_scaled)

    svc = LinearSVC(max_iter=2000, C=1.0)
    scores = cross_val_score(svc, X_scaled, labels, cv=5)
    svc_accuracy = float(scores.mean())

    print(f"  layer {layer_idx:02d} done (svc acc={svc_accuracy:.3f})")
    return layer_idx, umap_proj, tsne_proj, svc_accuracy


def fit_all(model_slug: str, layers: dict[int, np.ndarray], labels: np.ndarray) -> dict:
    cache = _cache_path(model_slug)
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    result: dict = {"umap": {}, "tsne": {}, "svc_accuracy": {}}

    layer_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_fit_layer)(layer_idx, X, labels) for layer_idx, X in layers.items()
    )

    for layer_idx, umap_proj, tsne_proj, svc_accuracy in layer_results:
        result["umap"][layer_idx] = umap_proj
        result["tsne"][layer_idx] = tsne_proj
        result["svc_accuracy"][layer_idx] = svc_accuracy

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(result, f)

    return result
