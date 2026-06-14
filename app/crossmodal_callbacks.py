import plotly.graph_objects as go
from app.crossmodal_figures import (
    build_crossmodal_heatmaps,
    build_crossmodal_scalar_bar,
)


def compute_crossmodal_update(
    crossmodal_data: dict,
    show_clip: bool,
) -> tuple[go.Figure, go.Figure]:
    heatmap_fig = build_crossmodal_heatmaps(crossmodal_data, show_clip=show_clip)
    bar_fig = build_crossmodal_scalar_bar(crossmodal_data)
    return heatmap_fig, bar_fig


def register(app, crossmodal_data: dict) -> None:
    from dash import Input, Output

    @app.callback(
        Output("crossmodal-heatmaps", "figure"),
        Output("crossmodal-scalar-bar", "figure"),
        Input("crossmodal-clip-toggle", "value"),
    )
    def update(clip_toggle):
        show_clip = "clip" in (clip_toggle or [])
        return compute_crossmodal_update(crossmodal_data, show_clip=show_clip)
