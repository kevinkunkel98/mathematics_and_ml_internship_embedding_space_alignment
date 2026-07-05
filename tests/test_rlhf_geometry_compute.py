import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch


def test_compute_anisotropy_identical_vectors_is_one():
    from app.rlhf_geometry_compute import compute_anisotropy

    X = np.ones((50, 8), dtype=np.float32)
    score = compute_anisotropy(X, n_pairs=100, seed=0)
    assert abs(score - 1.0) < 1e-5


def test_compute_anisotropy_orthogonal_vectors_is_zero():
    from app.rlhf_geometry_compute import compute_anisotropy

    # 4 mutually orthonormal rows, no duplicates: any distinct-index pair is orthogonal.
    X = np.eye(4, dtype=np.float32)
    score = compute_anisotropy(X, n_pairs=200, seed=0)
    assert abs(score) < 1e-5


def test_compute_cohens_d_separated_classes_is_large():
    from app.rlhf_geometry_compute import compute_cohens_d

    rng = np.random.default_rng(0)
    chosen = rng.normal(loc=10.0, scale=0.1, size=(50, 8)).astype(np.float32)
    rejected = rng.normal(loc=0.0, scale=0.1, size=(50, 8)).astype(np.float32)
    X = np.concatenate([chosen, rejected])
    labels = np.array([1] * 50 + [0] * 50, dtype=np.int8)
    d = compute_cohens_d(X, labels)
    assert d > 5.0


def test_compute_cohens_d_identical_classes_is_near_zero():
    from app.rlhf_geometry_compute import compute_cohens_d

    rng = np.random.default_rng(0)
    X = rng.normal(loc=0.0, scale=1.0, size=(100, 8)).astype(np.float32)
    labels = np.array([0, 1] * 50, dtype=np.int8)
    d = compute_cohens_d(X, labels)
    assert abs(d) < 1.0


def test_compute_effective_rank_single_direction_is_one():
    from app.rlhf_geometry_compute import compute_effective_rank

    rng = np.random.default_rng(0)
    direction = rng.normal(size=(1, 8))
    X = rng.normal(size=(100, 1)) @ direction
    rank = compute_effective_rank(X.astype(np.float32), n_sub=100)
    assert abs(rank - 1.0) < 0.2


def test_compute_effective_rank_full_rank_is_high():
    from app.rlhf_geometry_compute import compute_effective_rank

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8)).astype(np.float32)
    rank = compute_effective_rank(X, n_sub=200)
    assert rank > 4.0


def test_fit_rlhf_geometry_returns_expected_keys():
    from app.rlhf_geometry_compute import fit_rlhf_geometry

    rng = np.random.default_rng(0)
    layers = {i: rng.standard_normal((40, 8)).astype(np.float32) for i in range(3)}
    labels = np.array([0, 1] * 20, dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("app.rlhf_geometry_compute.CACHE_DIR", Path(tmpdir)):
            result = fit_rlhf_geometry("test-model", layers, labels)

    assert set(result.keys()) == {"anisotropy", "cohens_d", "effective_rank"}
    assert set(result["anisotropy"].keys()) == {0, 1, 2}
    assert isinstance(result["anisotropy"][0], float)
    assert isinstance(result["cohens_d"][0], float)
    assert isinstance(result["effective_rank"][0], float)


def test_fit_rlhf_geometry_uses_cache_on_second_call():
    from app.rlhf_geometry_compute import fit_rlhf_geometry

    rng = np.random.default_rng(0)
    layers = {0: rng.standard_normal((40, 8)).astype(np.float32)}
    labels = np.array([0, 1] * 20, dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        with patch("app.rlhf_geometry_compute.CACHE_DIR", cache_dir):
            result1 = fit_rlhf_geometry("test-model", layers, labels)
            layers[0] = np.zeros((40, 8), dtype=np.float32)
            result2 = fit_rlhf_geometry("test-model", layers, labels)

    assert result1 == result2
