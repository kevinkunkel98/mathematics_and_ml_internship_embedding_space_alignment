from __future__ import annotations

from app.part1_phase_figures import build_part1_umap

_PHASE_LABELS = {
    "phase1": "phase_1 (baseline)",
    "phase2": "cka_phase_2",
    "full_train": "full_train_set (final)",
}


def _nearest_layer(available: list[int], layer_idx: int) -> int:
    return min(available, key=lambda i: abs(i - layer_idx))


def register(app, part1_umap_data: dict) -> None:
    from dash import Input, Output

    @app.callback(
        Output("part1-vision-umap", "figure"),
        Output("part1-vision-layer-label", "children"),
        Input("part1-vision-phase-selector", "value"),
        Input("part1-vision-layer-slider", "value"),
    )
    def update_vision(phase: str, layer_idx: int):
        layers = part1_umap_data["vision"][phase]
        layer_idx = _nearest_layer(sorted(layers.keys()), layer_idx)
        title = f"Vision (DINOv2) — layer {layer_idx}, {_PHASE_LABELS[phase]}"
        fig = build_part1_umap(layers[layer_idx], part1_umap_data["labels"], title)
        return fig, f"Vision layer: {layer_idx}"

    @app.callback(
        Output("part1-language-umap", "figure"),
        Output("part1-language-layer-label", "children"),
        Input("part1-language-phase-selector", "value"),
        Input("part1-language-layer-slider", "value"),
    )
    def update_language(phase: str, layer_idx: int):
        layers = part1_umap_data["language"][phase]
        layer_idx = _nearest_layer(sorted(layers.keys()), layer_idx)
        title = f"Language (Llama-3B) — layer {layer_idx}, {_PHASE_LABELS[phase]}"
        fig = build_part1_umap(layers[layer_idx], part1_umap_data["labels"], title)
        return fig, f"Language layer: {layer_idx}"
