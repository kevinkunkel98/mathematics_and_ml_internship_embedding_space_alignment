import numpy as np
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch


def test_fit_all_returns_expected_keys():
    from app.compute import fit_all

    rng = np.random.default_rng(0)
    layers = {i: rng.standard_normal((20, 8)).astype(np.float32) for i in range(3)}
    labels = np.array([0, 1] * 10, dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("app.compute.CACHE_DIR", Path(tmpdir)):
            result = fit_all("test-model", layers, labels)

    assert set(result.keys()) == {"umap", "tsne", "svc_accuracy"}
    assert set(result["umap"].keys()) == {0, 1, 2}
    assert result["umap"][0].shape == (20, 2)
    assert result["tsne"][0].shape == (20, 2)
    assert 0.0 <= result["svc_accuracy"][0] <= 1.0


def test_fit_all_uses_cache_on_second_call():
    from app.compute import fit_all

    rng = np.random.default_rng(0)
    layers = {0: rng.standard_normal((20, 8)).astype(np.float32)}
    labels = np.array([0, 1] * 10, dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        with patch("app.compute.CACHE_DIR", cache_dir):
            result1 = fit_all("test-model", layers, labels)
            # Corrupt the layers to ensure second call reads cache, not recomputes
            layers[0] = np.zeros((20, 8), dtype=np.float32)
            result2 = fit_all("test-model", layers, labels)

    np.testing.assert_array_equal(result1["umap"][0], result2["umap"][0])
