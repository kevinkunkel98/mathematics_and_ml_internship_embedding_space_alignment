"""
Export static PNG figures for slides from mock data.
Uses system python3 (which has kaleido + plotly).
Run with: python3 scripts/export_slide_figures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h5py
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT = Path("assets/slides")
OUT.mkdir(parents=True, exist_ok=True)

_DARK = "plotly_dark"

# ── helpers ───────────────────────────────────────────────────────────────────


def load_vision(slug: str):
    path = Path("data/vision") / f"{slug}.h5"
    activations, labels, images, cams = {}, None, None, None
    with h5py.File(path, "r") as f:
        for key in f["activations"].keys():
            idx = int(key.split("_")[1])
            activations[idx] = f["activations"][key][:].astype(np.float32)
        labels = f["labels"][:]
        images = f["images"][:]
        cams = f["cams"][:]
    return activations, labels, images, cams


def load_llm(slug: str):
    path = Path("data/embeddings") / slug / "layers.h5"
    layers = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if key.startswith("layer_"):
                idx = int(key.split("_")[1])
                layers[idx] = f[key][:].astype(np.float32)
        labels = f["labels"][:]
    return layers, labels


def linear_cka(X, Y):
    X = (X - X.mean(0)).astype(np.float64)
    Y = (Y - Y.mean(0)).astype(np.float64)
    n = np.linalg.norm(X.T @ Y, "fro") ** 2
    d = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(n / d) if d else 0.0


# ── Figure 1: CKA heatmap ─────────────────────────────────────────────────────


def export_cka():
    cnn_acts, *_ = load_vision("resnet18")
    vit_acts, *_ = load_vision("vit_b16")

    cnn_layers = sorted(cnn_acts.keys())
    vit_layers = sorted(vit_acts.keys())
    matrix = np.array(
        [
            [linear_cka(cnn_acts[ci], vit_acts[vi]) for vi in vit_layers]
            for ci in cnn_layers
        ],
        dtype=np.float32,
    )

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[f"ViT L{i}" for i in vit_layers],
            y=[f"CNN L{i}" for i in cnn_layers],
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar=dict(title="CKA", thickness=14, len=0.85),
            hovertemplate="CNN %{y}<br>ViT %{x}<br>CKA: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Linear CKA: ResNet-18 vs ViT-B/16 layer activations (CIFAR-10)",
        xaxis_title="ViT-B/16 layer",
        yaxis_title="ResNet-18 layer",
        margin=dict(l=80, r=20, t=60, b=70),
        template=_DARK,
        width=820,
        height=480,
        font=dict(size=13),
    )
    fig.write_image(OUT / "cka_heatmap.png", scale=2)
    print("✓ cka_heatmap.png")


# ── Figure 2: CAM comparison (4 images) ──────────────────────────────────────


def export_cam():
    _, labels, images, cnn_cams = load_vision("resnet18")
    _, _, _, vit_cams = load_vision("vit_b16")

    _CLASSES = [
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

    idxs = np.arange(4)
    fig = make_subplots(
        rows=4,
        cols=3,
        column_titles=["Image", "ResNet-18 CAM", "ViT-B/16 CAM"],
        row_titles=[_CLASSES[labels[i]] for i in idxs],
        horizontal_spacing=0.04,
        vertical_spacing=0.04,
    )
    for row, idx in enumerate(idxs, 1):
        img_rgb = (images[idx] * 255).clip(0, 255).astype(np.uint8)
        for col, cam in [(2, cnn_cams[idx]), (3, vit_cams[idx])]:
            cam_n = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            fig.add_trace(
                go.Heatmap(
                    z=cam_n,
                    colorscale="Hot",
                    opacity=0.75,
                    showscale=False,
                    hovertemplate="CAM: %{z:.2f}<extra></extra>",
                ),
                row=row,
                col=col,
            )
        fig.add_trace(go.Image(z=img_rgb), row=row, col=1)

    fig.update_layout(
        title="Class Activation Maps — ResNet-18 vs ViT-B/16",
        template=_DARK,
        showlegend=False,
        margin=dict(l=70, r=20, t=70, b=20),
        width=820,
        height=600,
        font=dict(size=12),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.write_image(OUT / "cam_comparison.png", scale=2)
    print("✓ cam_comparison.png")


# ── Figure 3: UMAP scatter (instruct, layer 24) ───────────────────────────────


def export_umap():
    cache_path = Path("data/cache/meta-llama--Meta-Llama-3-8B-Instruct/projections.pkl")
    if not cache_path.exists():
        print("! UMAP cache missing — run app/app.py first to build cache")
        return
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    _, labels = load_llm("meta-llama--Meta-Llama-3-8B-Instruct")
    coords = cache["umap"][24]

    colors = {0: "#f87171", 1: "#60a5fa"}
    names = {0: "Rejected", 1: "Chosen"}
    fig = go.Figure()
    for lv in [1, 0]:
        mask = labels == lv
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(size=4, color=colors[lv], opacity=0.6),
                name=names[lv],
            )
        )
    fig.update_layout(
        title="UMAP — Llama-3-8B-Instruct, Layer 24 (chosen vs. rejected)",
        xaxis_title="UMAP dim 1",
        yaxis_title="UMAP dim 2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
        template=_DARK,
        width=700,
        height=460,
        font=dict(size=13),
    )
    fig.write_image(OUT / "umap_scatter.png", scale=2)
    print("✓ umap_scatter.png")


# ── Figure 4: LinearSVC separation score across layers ────────────────────────


def export_svc_line():
    slugs = {
        "meta-llama--Meta-Llama-3-8B": ("Llama-3-8B (base)", "#6b7280", "dash"),
        "meta-llama--Meta-Llama-3-8B-Instruct": (
            "Llama-3-8B-Instruct",
            "#7c3aed",
            "solid",
        ),
    }
    fig = go.Figure()
    for slug, (label, color, dash) in slugs.items():
        cache_path = Path("data/cache") / slug / "projections.pkl"
        if not cache_path.exists():
            print(f"! Cache missing for {slug}")
            continue
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        scores = cache["svc_accuracy"]
        layers = sorted(scores.keys())
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=[scores[l] for l in layers],
                mode="lines",
                name=label,
                line=dict(color=color, dash=dash, width=2.5),
            )
        )
    fig.update_layout(
        title="LinearSVC separation score per layer — base vs. RLHF-aligned",
        xaxis_title="Layer",
        yaxis_title="5-fold CV accuracy",
        yaxis=dict(range=[0.4, 1.0]),
        margin=dict(l=50, r=20, t=60, b=50),
        template=_DARK,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=700,
        height=300,
        font=dict(size=13),
    )
    fig.write_image(OUT / "svc_line.png", scale=2)
    print("✓ svc_line.png")


if __name__ == "__main__":
    export_cka()
    export_cam()
    export_umap()
    export_svc_line()
    print(f"\nAll figures written to {OUT}/")
