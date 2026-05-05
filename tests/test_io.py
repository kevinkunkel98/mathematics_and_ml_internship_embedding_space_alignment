import numpy as np
import pytest
import tempfile
from pathlib import Path

from scripts.io import save_embeddings, load_embeddings


def test_roundtrip_shapes_and_values():
    rng = np.random.default_rng(0)
    layers = {
        0: rng.standard_normal((10, 8)).astype(np.float32),
        1: rng.standard_normal((10, 8)).astype(np.float32),
    }
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sub" / "layers.h5"
        save_embeddings(path, layers, labels)
        loaded_layers, loaded_labels = load_embeddings(path)

    assert set(loaded_layers.keys()) == {0, 1}
    assert loaded_layers[0].shape == (10, 8)
    assert loaded_layers[1].shape == (10, 8)
    np.testing.assert_array_equal(loaded_labels, labels)


def test_save_creates_parent_dirs():
    rng = np.random.default_rng(1)
    layers = {0: rng.standard_normal((4, 4)).astype(np.float32)}
    labels = np.zeros(4, dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "a" / "b" / "c" / "layers.h5"
        save_embeddings(path, layers, labels)
        assert path.exists()
