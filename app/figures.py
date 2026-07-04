import numpy as np
import plotly.graph_objects as go

_LABEL_COLORS = {0: "#f87171", 1: "#60a5fa"}
_LABEL_NAMES = {0: "Rejected", 1: "Chosen"}
_MODEL_COLORS = {
    "allenai--Llama-3.1-Tulu-3-8B-SFT": "#3b82f6",
    "allenai--Llama-3.1-Tulu-3-8B-DPO": "#f59e0b",
    "allenai--Llama-3.1-Tulu-3-8B":      "#7c3aed",
}
_MODEL_DISPLAY = {
    "allenai--Llama-3.1-Tulu-3-8B-SFT": "Tulu-3-8B SFT",
    "allenai--Llama-3.1-Tulu-3-8B-DPO": "Tulu-3-8B DPO",
    "allenai--Llama-3.1-Tulu-3-8B":      "Tulu-3-8B RLHF",
}


def build_scatter(coords: np.ndarray, labels: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    for label_val in [1, 0]:
        mask = labels == label_val
        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            marker=dict(size=4, color=_LABEL_COLORS[label_val], opacity=0.6),
            name=_LABEL_NAMES[label_val],
        ))
    fig.update_layout(
        title=title,
        xaxis_title="dim 1",
        yaxis_title="dim 2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_dark",
    )
    return fig


def build_metric_line(
    svc_scores: dict[str, dict[int, float]], current_layer: int
) -> go.Figure:
    fig = go.Figure()
    for slug, scores in svc_scores.items():
        layers = sorted(scores.keys())
        accuracies = [scores[l] for l in layers]
        fig.add_trace(go.Scatter(
            x=layers,
            y=accuracies,
            mode="lines",
            name=_MODEL_DISPLAY.get(slug, slug),
            line=dict(
                color=_MODEL_COLORS.get(slug, "#aaaaaa"),
                dash="solid",
                width=2,
            ),
        ))
    fig.add_vline(x=current_layer, line_dash="dot", line_color="#4f46e5", opacity=0.7)
    fig.update_layout(
        title="LinearSVC separation score per layer",
        xaxis_title="Layer",
        yaxis_title="CV accuracy",
        yaxis=dict(range=[0.4, 1.0]),
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
