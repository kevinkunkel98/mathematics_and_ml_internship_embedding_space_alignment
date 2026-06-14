import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.crossmodal_compute import mean_crossmodal_cka, max_crossmodal_cka

_MODEL_LABELS = {
    "base": "Llama-3-8B (base)",
    "instruct": "Llama-3-8B-Instruct",
    "clip": "CLIP (upper bound)",
}

_MODEL_COLORS = {
    "base": "#636EFA",
    "instruct": "#00CC96",
    "clip": "#FFA15A",
}


def build_crossmodal_heatmaps(
    result: dict,
    show_clip: bool = True,
) -> go.Figure:
    """Side-by-side CKA heatmaps: base | instruct | CLIP (optional).

    result: output of fit_crossmodal()
    """
    models = ["base", "instruct"]
    if show_clip and result["clip"] is not None:
        models.append("clip")

    n_cols = len(models)
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=[_MODEL_LABELS[m] for m in models],
        horizontal_spacing=0.08,
    )

    for col, model in enumerate(models, start=1):
        matrix = result[model]
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=result["language_layer_names"],
                y=result["vision_layer_names"],
                colorscale="Viridis",
                zmin=0.0,
                zmax=1.0,
                showscale=(col == n_cols),
                colorbar=dict(title="CKA", thickness=12, x=1.02),
                hovertemplate=(
                    "Vision %{y}<br>Language %{x}<br>CKA: %{z:.3f}<extra></extra>"
                ),
                name=_MODEL_LABELS[model],
            ),
            row=1,
            col=col,
        )

    fig.update_layout(
        title="Cross-modal CKA: Vision layers × Language layers",
        template="plotly_dark",
        margin=dict(l=80, r=80, t=80, b=60),
        height=420,
    )
    fig.update_xaxes(title_text="Language layer", tickangle=45)
    fig.update_yaxes(title_text="Vision layer", col=1)
    return fig


def build_crossmodal_scalar_bar(result: dict) -> go.Figure:
    """Bar chart comparing mean and max CKA across models.

    Shows base, instruct, and CLIP (if available) as grouped bars.
    """
    models = ["base", "instruct"]
    if result["clip"] is not None:
        models.append("clip")

    means = [mean_crossmodal_cka(result[m]) for m in models]
    maxes = [max_crossmodal_cka(result[m]) for m in models]
    labels = [_MODEL_LABELS[m] for m in models]
    colors = [_MODEL_COLORS[m] for m in models]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Mean CKA",
            x=labels,
            y=means,
            marker_color=colors,
            opacity=0.7,
            hovertemplate="%{x}<br>Mean CKA: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Max CKA",
            x=labels,
            y=maxes,
            marker_color=colors,
            opacity=1.0,
            marker_pattern_shape="/",
            hovertemplate="%{x}<br>Max CKA: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Cross-modal Alignment Summary",
        yaxis_title="CKA score",
        yaxis=dict(range=[0, 1]),
        barmode="group",
        template="plotly_dark",
        margin=dict(l=60, r=20, t=60, b=60),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
