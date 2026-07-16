import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PHASE_LABELS = {"phase1": "phase_1 (baseline)", "full_train": "full_train_set (final)"}
_PHASE_COLORS = {"phase1": "#636EFA", "full_train": "#00CC96"}


def build_phase_cka_heatmaps(result: dict) -> go.Figure:
    """Side-by-side vision-layer x language-layer CKA heatmaps: phase_1 | full_train_set."""
    phases = ["phase1", "full_train"]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[_PHASE_LABELS[p] for p in phases],
        horizontal_spacing=0.1,
    )

    for col, phase in enumerate(phases, start=1):
        fig.add_trace(
            go.Heatmap(
                z=result[phase],
                x=result["language_layer_names"],
                y=result["vision_layer_names"],
                colorscale="Viridis",
                zmin=0.0,
                zmax=float(max(result["phase1"].max(), result["full_train"].max())),
                showscale=(col == 2),
                colorbar=dict(title="CKA", thickness=12, x=1.02),
                hovertemplate="Vision %{y}<br>Language %{x}<br>CKA: %{z:.3f}<extra></extra>",
                name=_PHASE_LABELS[phase],
            ),
            row=1,
            col=col,
        )

    fig.update_layout(
        title="Cross-modal CKA: DINOv2 layers x Llama-3B layers",
        template="plotly_dark",
        margin=dict(l=80, r=80, t=80, b=60),
        height=460,
    )
    fig.update_xaxes(title_text="Language layer", tickangle=45)
    fig.update_yaxes(title_text="Vision layer", col=1)
    return fig


def build_phase_scalar_bar(result: dict) -> go.Figure:
    """Bar chart: mean and max cross-modal CKA, phase_1 vs. full_train_set."""
    phases = ["phase1", "full_train"]
    means = [float(result[p].mean()) for p in phases]
    maxes = [float(result[p].max()) for p in phases]
    labels = [_PHASE_LABELS[p] for p in phases]
    colors = [_PHASE_COLORS[p] for p in phases]

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
        barmode="group",
        template="plotly_dark",
        margin=dict(l=60, r=20, t=60, b=60),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_supercategory_dumbbell(result: dict) -> go.Figure:
    """Dumbbell chart: cross-modal CKA per COCO supercategory, phase_1 vs. full_train_set.

    Categories sorted by the size of the shift (largest convergence on top).
    """
    categories = result["categories"]
    phase1 = result["phase1"]
    full_train = result["full_train"]

    order = sorted(
        range(len(categories)), key=lambda i: full_train[i] - phase1[i], reverse=True
    )
    categories = [categories[i] for i in order]
    phase1 = [phase1[i] for i in order]
    full_train = [full_train[i] for i in order]

    fig = go.Figure()

    for cat, p1, pf in zip(categories, phase1, full_train):
        fig.add_trace(
            go.Scatter(
                x=[p1, pf],
                y=[cat, cat],
                mode="lines",
                line=dict(color="#9aa5b1", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=phase1,
            y=categories,
            mode="markers",
            name="phase_1 (baseline)",
            marker=dict(color="#636EFA", size=11),
            hovertemplate="%{y}<br>phase_1 CKA: %{x:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=full_train,
            y=categories,
            mode="markers",
            name="full_train_set (final)",
            marker=dict(color="#00CC96", size=11),
            hovertemplate="%{y}<br>full_train_set CKA: %{x:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Cross-modal alignment by COCO supercategory: phase_1 vs. full_train_set",
        xaxis_title="Linear CKA (vision vs. language)",
        template="plotly_dark",
        margin=dict(l=120, r=40, t=60, b=50),
        height=90 + 40 * len(categories),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(range=[0, 1])
    return fig
