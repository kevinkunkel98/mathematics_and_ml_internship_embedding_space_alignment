import numpy as np
import pytest
import tempfile
from pathlib import Path

from scripts.io import (
    save_embeddings,
    load_embeddings,
    save_vision_data,
    load_vision_data,
)


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


def test_vision_roundtrip():
    rng = np.random.default_rng(2)
    activations = {i: rng.standard_normal((8, 16)).astype(np.float32) for i in range(3)}
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int8)
    images = rng.uniform(0, 1, (4, 32, 32, 3)).astype(np.float32)
    cams = rng.uniform(0, 1, (4, 32, 32)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.h5"
        save_vision_data(path, activations, labels, images, cams)
        loaded_acts, loaded_labels, loaded_images, loaded_cams, loaded_class_names = (
            load_vision_data(path)
        )

    assert set(loaded_acts.keys()) == {0, 1, 2}
    assert loaded_acts[0].shape == (8, 16)
    np.testing.assert_array_equal(loaded_labels, labels)
    assert loaded_images.shape == (4, 32, 32, 3)
    assert loaded_cams.shape == (4, 32, 32)
    assert loaded_class_names is None


def test_vision_roundtrip_with_class_names():
    rng = np.random.default_rng(3)
    activations = {0: rng.standard_normal((4, 8)).astype(np.float32)}
    labels = np.array([0, 1, 2, 3], dtype=np.int8)
    images = rng.uniform(0, 1, (4, 8, 8, 3)).astype(np.float32)
    cams = rng.uniform(0, 1, (4, 8, 8)).astype(np.float32)
    class_names = ["cat", "dog", "bird"]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.h5"
        save_vision_data(
            path, activations, labels, images, cams, class_names=class_names
        )
        _, _, _, _, loaded_names = load_vision_data(path)

    assert loaded_names == class_names
