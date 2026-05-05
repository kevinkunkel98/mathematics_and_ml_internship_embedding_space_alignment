import numpy as np
import plotly.graph_objects as go
from app.figures import build_scatter, build_metric_line


def compute_update(
    app_data: dict, model_slug: str, projection: str, layer: int
) -> tuple[go.Figure, go.Figure, str]:
    coords = app_data[model_slug]["cache"][projection][layer]
    labels = app_data[model_slug]["labels"]
    scatter = build_scatter(coords, labels, f"{projection.upper()} — Layer {layer}")
    svc_scores = {slug: d["cache"]["svc_accuracy"] for slug, d in app_data.items()}
    metric = build_metric_line(svc_scores, layer)
    return scatter, metric, f"Layer: {layer}"


def register(app, app_data: dict) -> None:
    from dash import Input, Output

    @app.callback(
        Output("scatter", "figure"),
        Output("metric-line", "figure"),
        Output("layer-label", "children"),
        Input("model-selector", "value"),
        Input("projection-selector", "value"),
        Input("layer-slider", "value"),
    )
    def update(model_slug: str, projection: str, layer: int):
        return compute_update(app_data, model_slug, projection, layer)
