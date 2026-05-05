import numpy as np
import plotly.graph_objects as go


def _make_mock_app_data():
    rng = np.random.default_rng(0)
    cache = {
        "umap": {i: rng.standard_normal((20, 2)) for i in range(3)},
        "tsne": {i: rng.standard_normal((20, 2)) for i in range(3)},
        "svc_accuracy": {i: 0.7 + i * 0.05 for i in range(3)},
    }
    labels = np.array([0, 1] * 10, dtype=np.int8)
    return {
        "meta-llama--Meta-Llama-3-8B": {"cache": cache, "labels": labels},
        "meta-llama--Meta-Llama-3-8B-Instruct": {"cache": cache, "labels": labels},
    }


def test_compute_update_returns_figures_and_label():
    from app.callbacks import compute_update

    app_data = _make_mock_app_data()
    scatter, metric, label_text = compute_update(
        app_data, "meta-llama--Meta-Llama-3-8B", "umap", 1
    )

    assert isinstance(scatter, go.Figure)
    assert isinstance(metric, go.Figure)
    assert label_text == "Layer: 1"


def test_compute_update_tsne_title():
    from app.callbacks import compute_update

    app_data = _make_mock_app_data()
    scatter, _, _ = compute_update(app_data, "meta-llama--Meta-Llama-3-8B", "tsne", 0)

    assert "TSNE" in scatter.layout.title.text
