import numpy as np
import pytest
from app.vision_compute import linear_cka, compute_cka_matrix, fit_vision


def test_linear_cka_identical():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 64)).astype(np.float32)
    assert abs(linear_cka(X, X) - 1.0) < 1e-4


def test_linear_cka_range():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 32)).astype(np.float32)
    Y = rng.standard_normal((50, 48)).astype(np.float32)
    val = linear_cka(X, Y)
    assert 0.0 <= val <= 1.0


def test_compute_cka_matrix_shape():
    rng = np.random.default_rng(2)
    cnn = {i: rng.standard_normal((30, 16)).astype(np.float32) for i in range(4)}
    vit = {i: rng.standard_normal((30, 24)).astype(np.float32) for i in range(6)}
    mat = compute_cka_matrix(cnn, vit)
    assert mat.shape == (4, 6)


def test_compute_cka_matrix_diagonal_high():
    rng = np.random.default_rng(3)
    acts = {i: rng.standard_normal((50, 32)).astype(np.float32) for i in range(3)}
    mat = compute_cka_matrix(acts, acts)
    for i in range(3):
        assert mat[i, i] > 0.99


def test_fit_vision_returns_expected_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("app.vision_compute.CACHE_DIR", tmp_path)
    rng = np.random.default_rng(4)
    cnn = {i: rng.standard_normal((20, 16)).astype(np.float32) for i in range(3)}
    vit = {i: rng.standard_normal((20, 24)).astype(np.float32) for i in range(4)}
    result = fit_vision(cnn, vit)
    assert "cka_matrix" in result
    assert "cnn_layer_names" in result
    assert "vit_layer_names" in result
    assert result["cka_matrix"].shape == (3, 4)


def test_fit_vision_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("app.vision_compute.CACHE_DIR", tmp_path)
    rng = np.random.default_rng(5)
    cnn = {i: rng.standard_normal((20, 16)).astype(np.float32) for i in range(2)}
    vit = {i: rng.standard_normal((20, 16)).astype(np.float32) for i in range(2)}
    r1 = fit_vision(cnn, vit)
    # Second call should load from cache — pass different data to confirm cache is used
    cnn2 = {i: rng.standard_normal((20, 16)).astype(np.float32) for i in range(2)}
    r2 = fit_vision(cnn2, vit)
    np.testing.assert_array_equal(r1["cka_matrix"], r2["cka_matrix"])
