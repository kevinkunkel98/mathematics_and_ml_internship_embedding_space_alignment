import numpy as np
import pytest
from app.vision_figures import build_cka_heatmap, build_cam_comparison


def _mock_data(n=20, n_cam=8):
    rng = np.random.default_rng(42)
    cka_matrix = rng.uniform(0, 1, (5, 7)).astype(np.float32)
    cnn_names = [f"layer {i}" for i in range(5)]
    vit_names = [f"layer {i}" for i in range(7)]
    images = rng.uniform(0, 1, (n_cam, 32, 32, 3)).astype(np.float32)
    cnn_cams = rng.uniform(0, 1, (n_cam, 32, 32)).astype(np.float32)
    vit_cams = rng.uniform(0, 1, (n_cam, 32, 32)).astype(np.float32)
    labels = (np.arange(n_cam) % 10).astype(np.int8)
    return cka_matrix, cnn_names, vit_names, images, cnn_cams, vit_cams, labels


def test_cka_heatmap_has_one_trace():
    cka, cnn_n, vit_n, *_ = _mock_data()
    fig = build_cka_heatmap(cka, cnn_n, vit_n)
    assert len(fig.data) == 1


def test_cka_heatmap_dimensions():
    cka, cnn_n, vit_n, *_ = _mock_data()
    fig = build_cka_heatmap(cka, cnn_n, vit_n)
    z = fig.data[0].z
    assert np.array(z).shape == (5, 7)


def test_cam_comparison_traces():
    cka, cnn_n, vit_n, images, cnn_cams, vit_cams, labels = _mock_data(n_cam=8)
    fig = build_cam_comparison(
        images, cnn_cams, vit_cams, labels, selected_class=None, n_show=2
    )
    # 3 traces per row (image, cnn cam, vit cam)
    assert len(fig.data) == 6


def test_cam_comparison_class_filter():
    cka, cnn_n, vit_n, images, cnn_cams, vit_cams, labels = _mock_data(n_cam=8)
    fig = build_cam_comparison(
        images, cnn_cams, vit_cams, labels, selected_class=0, n_show=4
    )
    n_class0 = int((labels == 0).sum())
    expected_rows = min(n_class0, 4)
    assert len(fig.data) == expected_rows * 3


def test_cam_comparison_empty_class():
    _, _, _, images, cnn_cams, vit_cams, labels = _mock_data(n_cam=8)
    labels[:] = 1  # no class 9 samples
    fig = build_cam_comparison(
        images, cnn_cams, vit_cams, labels, selected_class=9, n_show=4
    )
    assert len(fig.data) == 0
