import numpy as np
import plotly.graph_objects as go
import pytest

from app.crossmodal_figures import (
    build_crossmodal_heatmaps,
    build_crossmodal_scalar_bar,
)


def _make_result(with_clip: bool = True) -> dict:
    rng = np.random.default_rng(0)
    base = rng.random((5, 8)).astype(np.float32)
    instruct = rng.random((5, 8)).astype(np.float32)
    clip = rng.random((5, 5)).astype(np.float32) if with_clip else None
    return {
        "base": base,
        "instruct": instruct,
        "clip": clip,
        "vision_layer_names": [f"layer {i}" for i in range(5)],
        "language_layer_names": [f"layer {j}" for j in range(8)],
    }


def test_build_crossmodal_heatmaps_two_models_without_clip():
    result = _make_result(with_clip=False)
    fig = build_crossmodal_heatmaps(result, show_clip=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_build_crossmodal_heatmaps_three_models_with_clip():
    result = _make_result(with_clip=True)
    fig = build_crossmodal_heatmaps(result, show_clip=True)
    assert len(fig.data) == 3


def test_build_crossmodal_heatmaps_clip_none_shows_two():
    result = _make_result(with_clip=False)
    fig = build_crossmodal_heatmaps(result, show_clip=True)
    assert len(fig.data) == 2


def test_build_crossmodal_scalar_bar_has_two_bar_traces():
    result = _make_result(with_clip=False)
    fig = build_crossmodal_scalar_bar(result)
    bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
    assert len(bar_traces) == 2


def test_build_crossmodal_scalar_bar_with_clip_three_x_values():
    result = _make_result(with_clip=True)
    fig = build_crossmodal_scalar_bar(result)
    assert len(fig.data[0].x) == 3


def test_build_crossmodal_scalar_bar_yrange():
    result = _make_result()
    fig = build_crossmodal_scalar_bar(result)
    assert list(fig.layout.yaxis.range) == [0, 1]
