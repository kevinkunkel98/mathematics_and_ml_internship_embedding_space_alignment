import numpy as np
import plotly.graph_objects as go

from app.figures import build_scatter, build_metric_line


def test_build_scatter_has_two_traces():
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((20, 2))
    labels = np.array([0, 1] * 10)
    fig = build_scatter(coords, labels, "test title")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_build_scatter_trace_names():
    coords = np.zeros((4, 2))
    labels = np.array([0, 0, 1, 1])
    fig = build_scatter(coords, labels, "t")
    names = {trace.name for trace in fig.data}
    assert names == {"Chosen", "Rejected"}


def test_build_metric_line_has_correct_trace_count():
    svc_scores = {
        "model-a": {i: 0.5 + i * 0.01 for i in range(5)},
        "model-b": {i: 0.6 + i * 0.01 for i in range(5)},
    }
    fig = build_metric_line(svc_scores, current_layer=2)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_build_metric_line_has_vline():
    svc_scores = {"m": {0: 0.7, 1: 0.8}}
    fig = build_metric_line(svc_scores, current_layer=1)
    # vline is added as a layout shape
    assert any(s.get("x0") == 1 for s in fig.to_dict().get("layout", {}).get("shapes", []))
