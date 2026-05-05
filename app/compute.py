import pickle
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from sklearn.manifold import TSNE

CACHE_DIR = Path("data/cache")


def _cache_path(model_slug: str) -> Path:
    return CACHE_DIR / model_slug / "projections.pkl"


def fit_all(model_slug: str, layers: dict[int, np.ndarray], labels: np.ndarray) -> dict:
    cache = _cache_path(model_slug)
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    result: dict = {"umap": {}, "tsne": {}, "svc_accuracy": {}}

    for layer_idx, X in layers.items():
        X32 = X.astype(np.float32)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X32)

        result["umap"][layer_idx] = UMAP(
            n_neighbors=15, min_dist=0.1, n_components=2, random_state=42
        ).fit_transform(X_scaled)

        result["tsne"][layer_idx] = TSNE(
            perplexity=min(30, len(X_scaled) - 1), n_components=2, random_state=42
        ).fit_transform(X_scaled)

        svc = LinearSVC(max_iter=2000, C=1.0)
        scores = cross_val_score(svc, X_scaled, labels, cv=5)
        result["svc_accuracy"][layer_idx] = float(scores.mean())

        print(f"  layer {layer_idx:02d} done (svc acc={result['svc_accuracy'][layer_idx]:.3f})")

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(result, f)

    return result
