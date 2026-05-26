import numpy as np
import pytest
import pickle
from pathlib import Path

from app.crossmodal_compute import (
    compute_crossmodal_cka_matrix,
    fit_crossmodal,
    mean_crossmodal_cka,
    max_crossmodal_cka,
)


def _rand_layers(
    n_layers: int, n_samples: int, dim: int, seed: int = 0
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        i: rng.standard_normal((n_samples, dim)).astype(np.float32)
        for i in range(n_layers)
    }


def test_compute_crossmodal_cka_matrix_shape():
    vis = _rand_layers(5, 100, 64)
    lang = _rand_layers(8, 100, 128)
    matrix = compute_crossmodal_cka_matrix(vis, lang)
    assert matrix.shape == (5, 8)
    assert matrix.dtype == np.float32


def test_compute_crossmodal_cka_matrix_range():
    vis = _rand_layers(3, 100, 64)
    lang = _rand_layers(4, 100, 64)
    matrix = compute_crossmodal_cka_matrix(vis, lang)
    assert float(matrix.min()) >= 0.0
    assert float(matrix.max()) <= 1.0


def test_compute_crossmodal_cka_identical_representations():
    layers = _rand_layers(3, 100, 64)
    matrix = compute_crossmodal_cka_matrix(layers, layers)
    assert matrix.shape == (3, 3)
    for i in range(3):
        assert matrix[i, i] == pytest.approx(1.0, abs=1e-4)


def test_fit_crossmodal_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("app.crossmodal_compute.CACHE_DIR", tmp_path)
    vis = _rand_layers(4, 50, 32)
    base = _rand_layers(6, 50, 64)
    instruct = _rand_layers(6, 50, 64)

    result = fit_crossmodal(vis, base, instruct, cache_key="test")
    assert set(result.keys()) == {
        "base",
        "instruct",
        "clip",
        "vision_layer_names",
        "language_layer_names",
    }
    assert result["base"].shape == (4, 6)
    assert result["instruct"].shape == (4, 6)
    assert result["clip"] is None
    assert len(result["vision_layer_names"]) == 4
    assert len(result["language_layer_names"]) == 6


def test_fit_crossmodal_with_clip(tmp_path, monkeypatch):
    monkeypatch.setattr("app.crossmodal_compute.CACHE_DIR", tmp_path)
    vis = _rand_layers(4, 50, 32)
    base = _rand_layers(6, 50, 64)
    instruct = _rand_layers(6, 50, 64)
    clip = _rand_layers(4, 50, 32)

    result = fit_crossmodal(vis, base, instruct, clip, cache_key="test_clip")
    assert result["clip"] is not None
    assert result["clip"].shape == (4, 4)


def test_fit_crossmodal_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("app.crossmodal_compute.CACHE_DIR", tmp_path)
    vis = _rand_layers(3, 50, 32)
    base = _rand_layers(4, 50, 64)
    instruct = _rand_layers(4, 50, 64)

    r1 = fit_crossmodal(vis, base, instruct, cache_key="cache_test")
    # overwrite inputs — cache should return original result
    r2 = fit_crossmodal(
        _rand_layers(3, 50, 32, seed=99),
        _rand_layers(4, 50, 64, seed=99),
        _rand_layers(4, 50, 64, seed=99),
        cache_key="cache_test",
    )
    np.testing.assert_array_equal(r1["base"], r2["base"])


def test_mean_and_max_crossmodal_cka():
    matrix = np.array([[0.1, 0.3], [0.5, 0.7]], dtype=np.float32)
    assert mean_crossmodal_cka(matrix) == pytest.approx(0.4, abs=1e-5)
    assert max_crossmodal_cka(matrix) == pytest.approx(0.7, abs=1e-5)
