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


def build_drift_line(drift: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drift["layers"], y=drift["sft_dpo"], mode="lines",
        name="SFT vs. DPO", line=dict(color="#3b82f6", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=drift["layers"], y=drift["dpo_rlhf"], mode="lines",
        name="DPO vs. RLHF", line=dict(color="#f59e0b", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=drift["layers"], y=drift["sft_rlhf"], mode="lines",
        name="SFT vs. RLHF", line=dict(color="#7c3aed", width=2),
    ))
    fig.update_layout(
        title="Representational drift across alignment stages (Linear CKA)",
        xaxis_title="Layer",
        yaxis_title="Linear CKA",
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_rlhf_cka_heatmap(
    matrix: np.ndarray, layer_names: list[str], title: str, y_label: str, x_label: str
) -> go.Figure:
    # RLHF checkpoints are all highly similar (CKA ~0.85-1.0) — unlike the Part 1
    # cross-modal heatmap, a fixed [0, 1] scale would flatten all the real structure
    # into a thin sliver at the top of the colorscale. Autoscale to the data instead.
    zmin = float(np.floor(matrix.min() * 20) / 20)  # round down to nearest 0.05
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=layer_names,
            y=layer_names,
            colorscale="Viridis",
            zmin=zmin,
            zmax=1.0,
            colorbar=dict(title="CKA", thickness=15),
            hovertemplate=f"{y_label} %{{y}}<br>{x_label} %{{x}}<br>CKA: %{{z:.3f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=80, r=20, t=60, b=80),
        template="plotly_dark",
    )
    return fig


def build_geometry_line(
    scores: dict[str, dict[int, float]], title: str, yaxis_title: str
) -> go.Figure:
    fig = go.Figure()
    for slug, values in scores.items():
        layers = sorted(values.keys())
        ys = [values[l] for l in layers]
        fig.add_trace(go.Scatter(
            x=layers,
            y=ys,
            mode="lines",
            name=_MODEL_DISPLAY.get(slug, slug),
            line=dict(color=_MODEL_COLORS.get(slug, "#aaaaaa"), width=2),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title=yaxis_title,
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
