import numpy as np
import plotly.graph_objects as go
from app.vision_figures import build_cka_heatmap, build_cam_comparison

_CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def compute_vision_update(
    vision_data: dict,
    selected_class: int | None,
) -> tuple[go.Figure, go.Figure]:
    cka_fig = build_cka_heatmap(
        vision_data["cka"]["cka_matrix"],
        vision_data["cka"]["cnn_layer_names"],
        vision_data["cka"]["vit_layer_names"],
    )
    cam_fig = build_cam_comparison(
        vision_data["images"],
        vision_data["cnn_cams"],
        vision_data["vit_cams"],
        vision_data["labels"],
        selected_class,
    )
    return cka_fig, cam_fig


def register(app, vision_data: dict) -> None:
    from dash import Input, Output

    @app.callback(
        Output("cka-heatmap", "figure"),
        Output("cam-comparison", "figure"),
        Input("class-selector", "value"),
    )
    def update(selected_class):
        cls = None if selected_class == "all" else selected_class
        return compute_vision_update(vision_data, cls)
