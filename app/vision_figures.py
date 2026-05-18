import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def build_cka_heatmap(
    cka_matrix: np.ndarray,
    cnn_layer_names: list[str],
    vit_layer_names: list[str],
) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=cka_matrix,
            x=vit_layer_names,
            y=cnn_layer_names,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="CKA", thickness=15),
            hovertemplate="CNN %{y}<br>ViT %{x}<br>CKA: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Linear CKA: ResNet-18 layers vs ViT-B/16 layers",
        xaxis_title="ViT-B/16 layer",
        yaxis_title="ResNet-18 layer",
        margin=dict(l=80, r=20, t=60, b=80),
        template="plotly_dark",
    )
    return fig


def build_cam_comparison(
    images: np.ndarray,
    cnn_cams: np.ndarray,
    vit_cams: np.ndarray,
    labels: np.ndarray,
    selected_class: int | None,
    n_show: int = 4,
) -> go.Figure:
    """Side-by-side CAM overlay for CNN vs ViT for up to n_show images.

    images:   (N, 32, 32, 3) float32 [0,1]
    cnn_cams: (N, 32, 32) float32
    vit_cams: (N, 32, 32) float32
    labels:   (N,) int
    """
    if selected_class is not None:
        idxs = np.where(labels == selected_class)[0][:n_show]
    else:
        idxs = np.arange(min(n_show, len(labels)))

    n = len(idxs)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="No samples for selected class")
        return fig

    fig = make_subplots(
        rows=n,
        cols=3,
        column_titles=["Image", "ResNet-18 CAM", "ViT-B/16 CAM"],
        row_titles=[_CIFAR10_CLASSES[labels[i]] for i in idxs],
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    for row, idx in enumerate(idxs, start=1):
        img_rgb = (images[idx] * 255).clip(0, 255).astype(np.uint8)
        cnn_cam = cnn_cams[idx]
        vit_cam = vit_cams[idx]

        fig.add_trace(go.Image(z=img_rgb), row=row, col=1)
        fig.add_trace(_cam_heatmap_trace(img_rgb, cnn_cam), row=row, col=2)
        fig.add_trace(_cam_heatmap_trace(img_rgb, vit_cam), row=row, col=3)

    height = max(300, n * 140)
    fig.update_layout(
        title="Class Activation Maps: ResNet-18 vs ViT-B/16",
        template="plotly_dark",
        margin=dict(l=60, r=20, t=80, b=40),
        height=height,
        showlegend=False,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


_CIFAR10_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def build_vision_umap(
    Z: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> go.Figure:
    """Scatter plot of UMAP-projected activations, coloured by CIFAR-10 class.

    Z:      (N, 2) float32
    labels: (N,)  int  — CIFAR-10 class indices
    """
    fig = go.Figure()
    n = min(len(Z), len(labels))
    Z, labels = Z[:n], labels[:n]

    for cls_idx, cls_name in enumerate(_CIFAR10_CLASSES):
        mask = labels == cls_idx
        if not mask.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=Z[mask, 0],
                y=Z[mask, 1],
                mode="markers",
                name=cls_name,
                marker=dict(
                    size=5,
                    color=_CIFAR10_COLORS[cls_idx],
                    opacity=0.8,
                ),
                hovertemplate=f"{cls_name}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis=dict(showticklabels=False, title="UMAP 1"),
        yaxis=dict(showticklabels=False, title="UMAP 2"),
        legend=dict(itemsizing="constant", title="Class"),
        margin=dict(l=40, r=20, t=60, b=40),
        height=460,
    )
    return fig


def _cam_heatmap_trace(img_rgb: np.ndarray, cam: np.ndarray) -> go.Heatmap:
    """Overlay CAM as a semi-transparent heatmap on top of image dims."""
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return go.Heatmap(
        z=cam_norm,
        colorscale="Hot",
        opacity=0.6,
        showscale=False,
        hovertemplate="CAM: %{z:.2f}<extra></extra>",
    )
