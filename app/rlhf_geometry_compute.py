import pickle
from pathlib import Path

import numpy as np

CACHE_DIR = Path("data/cache/rlhf_metrics")


def _cache_path(model_slug: str) -> Path:
    return CACHE_DIR / model_slug / "metrics.pkl"


def compute_anisotropy(X: np.ndarray, n_pairs: int = 1000, seed: int = 0) -> float:
    """Average cosine similarity between random pairs of rows.

    Near 1.0 = representations collapse into a narrow cone (anisotropic).
    Near 0.0 = representations spread evenly (isotropic).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    a = X[idx_a].astype(np.float64)
    b = X[idx_b].astype(np.float64)
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    cos_sims = np.sum(a_norm * b_norm, axis=1)
    return float(cos_sims.mean())


def compute_cohens_d(X: np.ndarray, labels: np.ndarray) -> float:
    """Effect size between the two label groups, projected onto their mean-difference direction.

    This detects separation along the single best linear direction even when a
    regularized classifier (LinearSVC) fails to find it in the full high-dim space.
    """
    chosen = X[labels == 1].astype(np.float64)
    rejected = X[labels == 0].astype(np.float64)

    diff = chosen.mean(axis=0) - rejected.mean(axis=0)
    norm = np.linalg.norm(diff)
    if norm == 0.0:
        return 0.0
    direction = diff / norm

    proj_chosen = chosen @ direction
    proj_rejected = rejected @ direction
    pooled_std = np.sqrt((proj_chosen.var(ddof=1) + proj_rejected.var(ddof=1)) / 2)
    if pooled_std == 0.0:
        return 0.0
    return float((proj_chosen.mean() - proj_rejected.mean()) / pooled_std)


def compute_effective_rank(X: np.ndarray, n_sub: int = 600, seed: int = 0) -> float:
    """Participation ratio of the covariance spectrum: (sum(eig))^2 / sum(eig^2).

    Interpretable as "how many dimensions are effectively in use." Subsamples
    rows to n_sub since the full SVD cost scales with n^2 * d.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n > n_sub:
        idx = rng.choice(n, size=n_sub, replace=False)
        X = X[idx]

    Xc = (X - X.mean(axis=0)).astype(np.float64)
    s = np.linalg.svd(Xc, compute_uv=False)
    eigenvalues = s ** 2
    total = eigenvalues.sum()
    if total == 0.0:
        return 0.0
    return float((total ** 2) / np.sum(eigenvalues ** 2))


def fit_rlhf_geometry(model_slug: str, layers: dict[int, np.ndarray], labels: np.ndarray) -> dict:
    cache = _cache_path(model_slug)
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    result: dict = {"anisotropy": {}, "cohens_d": {}, "effective_rank": {}}
    for layer_idx, X in layers.items():
        X32 = X.astype(np.float32)
        result["anisotropy"][layer_idx] = compute_anisotropy(X32)
        result["cohens_d"][layer_idx] = compute_cohens_d(X32, labels)
        result["effective_rank"][layer_idx] = compute_effective_rank(X32)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(result, f)

    return result
