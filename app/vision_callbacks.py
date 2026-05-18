import numpy as np
import plotly.graph_objects as go
from app.vision_figures import build_cka_heatmap, build_cam_comparison, build_vision_umap

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

    @app.callback(
        Output("vision-umap", "figure"),
        Output("vision-layer-label", "children"),
        Input("vision-model-selector", "value"),
        Input("vision-layer-slider", "value"),
    )
    def update_vision_umap(model_slug: str, layer_idx: int):
        umap_by_layer = vision_data["umap"][model_slug]
        # clamp to available layers for this model
        max_layer = max(umap_by_layer.keys())
        layer_idx = min(layer_idx, max_layer)

        Z = umap_by_layer[layer_idx]
        labels = vision_data["labels"][: len(Z)]

        model_label = "ResNet-18" if model_slug == "resnet18" else "ViT-B/16"
        title = f"{model_label} — layer {layer_idx} activations (UMAP)"
        slider_label = f"Layer: {layer_idx} / {max_layer}"

        return build_vision_umap(Z, labels, title), slider_label
