from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dash import Dash, dcc, html

from scripts.io import load_embeddings
from app.compute import fit_all
from app.part1_phase_compute import compute_supercategory_alignment, compute_phase_cka_matrices
from app.part1_umap_compute import fit_part1_umap
from app.part1_phase_figures import (
    build_supercategory_dumbbell,
    build_phase_cka_heatmaps,
    build_phase_scalar_bar,
)
from app.rlhf_drift_compute import fit_rlhf_drift, fit_rlhf_cka_matrices
from app.rlhf_geometry_compute import fit_rlhf_geometry
from app.figures import build_drift_line, build_geometry_line, build_rlhf_cka_heatmap

_LLM_MODELS = {
    "allenai--Llama-3.1-Tulu-3-8B-SFT": "Tulu-3-8B SFT",
    "allenai--Llama-3.1-Tulu-3-8B-DPO": "Tulu-3-8B DPO",
    "allenai--Llama-3.1-Tulu-3-8B":      "Tulu-3-8B RLHF",
}


def _load_llm_data() -> dict:
    data = {}
    for slug in _LLM_MODELS:
        path = Path("data/embeddings") / slug / "layers.h5"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}\n"
                f"Run: python scripts/extract_embeddings.py --trajectory tulu3\n"
                f"Or:  python scripts/generate_mock_data.py"
            )
        layers, labels = load_embeddings(path)
        print(f"Fitting projections for {slug} ...")
        cache = fit_all(slug, layers, labels)
        print(f"Fitting geometry metrics for {slug} ...")
        geometry = fit_rlhf_geometry(slug, layers, labels)
        data[slug] = {"cache": cache, "labels": labels, "geometry": geometry}
    return data


APP_DATA = _load_llm_data()

RLHF_DRIFT = fit_rlhf_drift(
    Path("data/embeddings/allenai--Llama-3.1-Tulu-3-8B-SFT/layers.h5"),
    Path("data/embeddings/allenai--Llama-3.1-Tulu-3-8B-DPO/layers.h5"),
    Path("data/embeddings/allenai--Llama-3.1-Tulu-3-8B/layers.h5"),
)

RLHF_CKA_MATRICES = fit_rlhf_cka_matrices(
    Path("data/embeddings/allenai--Llama-3.1-Tulu-3-8B-SFT/layers.h5"),
    Path("data/embeddings/allenai--Llama-3.1-Tulu-3-8B-DPO/layers.h5"),
    Path("data/embeddings/allenai--Llama-3.1-Tulu-3-8B/layers.h5"),
)


_PART1_DATA_DIR = Path("data/embeddings part 1")
_PART1_VISION_PHASE1 = _PART1_DATA_DIR / "vision/llama_3B_coco_multilabel_phase_1.h5"
_PART1_VISION_FULL = _PART1_DATA_DIR / "vision/llama_3B_coco_multilabel_full_train_set.h5"
_PART1_LANGUAGE_PHASE1 = _PART1_DATA_DIR / "language/llama_3B_coco_multilabel_phase_1/layers.h5"
_PART1_LANGUAGE_FULL = _PART1_DATA_DIR / "language/llama_3B_coco_multilabel_full_train_set/layers.h5"
_PART1_LANGUAGE_PHASE2 = _PART1_DATA_DIR / "language/llama_3B_coco_multilabel_cka_phase_2/layers.h5"


def _load_part1_data() -> dict | None:
    try:
        for p in (_PART1_VISION_PHASE1, _PART1_VISION_FULL, _PART1_LANGUAGE_PHASE1, _PART1_LANGUAGE_FULL, _PART1_LANGUAGE_PHASE2):
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}")

        print("Fitting Part 1 cross-modal CKA matrices...")
        matrices = compute_phase_cka_matrices(
            vision_phase1_path=str(_PART1_VISION_PHASE1),
            vision_full_path=str(_PART1_VISION_FULL),
            language_phase1_path=str(_PART1_LANGUAGE_PHASE1),
            language_full_path=str(_PART1_LANGUAGE_FULL),
        )

        print("Fitting Part 1 supercategory alignment...")
        supercategory = compute_supercategory_alignment(
            vision_phase1_path=str(_PART1_VISION_PHASE1),
            vision_full_path=str(_PART1_VISION_FULL),
            language_phase1_path=str(_PART1_LANGUAGE_PHASE1),
            language_full_path=str(_PART1_LANGUAGE_FULL),
        )

        print("Fitting Part 1 UMAP projections...")
        umap = fit_part1_umap(
            vision_phase1_path=str(_PART1_VISION_PHASE1),
            vision_full_path=str(_PART1_VISION_FULL),
            language_phase1_path=str(_PART1_LANGUAGE_PHASE1),
            language_full_path=str(_PART1_LANGUAGE_FULL),
            language_phase2_path=str(_PART1_LANGUAGE_PHASE2),
        )

        return {"matrices": matrices, "supercategory": supercategory, "umap": umap}
    except FileNotFoundError as e:
        print(f"[part1] Skipping Part 1 — {e}")
        return None


PART1_DATA = _load_part1_data()

N_LAYERS = max(next(iter(APP_DATA.values()))["cache"]["umap"].keys())

app = Dash(__name__, title="Representational Geometry in Neural Networks")

_tab_style = {"padding": "6px 16px"}
_selected_tab_style = {
    "padding": "6px 16px",
    "fontWeight": "bold",
    "borderTop": "3px solid #6366f1",
}

# ── Part 2 layout ─────────────────────────────────────────────────────────────

_part2_layout = html.Div(
    [
        html.P(
            "Layer-wise UMAP and LinearSVC separation across the Tulu-3 alignment pipeline (SFT → DPO → RLHF). "
            "Does human preference become increasingly linearly separable as alignment progresses — and at which layer?",
            style={"color": "#6b7280", "fontSize": "13px", "marginBottom": "12px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Model", style={"fontWeight": "bold"}),
                        dcc.RadioItems(
                            id="model-selector",
                            options=[
                                {"label": v, "value": k} for k, v in _LLM_MODELS.items()
                            ],
                            value=list(_LLM_MODELS.keys())[0],
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "16px"},
                        ),
                    ],
                    style={"marginRight": "32px"},
                ),
                html.Div(
                    [
                        html.Label("Projection", style={"fontWeight": "bold"}),
                        dcc.RadioItems(
                            id="projection-selector",
                            options=[
                                {"label": "UMAP", "value": "umap"},
                                {"label": "t-SNE", "value": "tsne"},
                            ],
                            value="umap",
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "16px"},
                        ),
                    ]
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "flex-start",
                "marginBottom": "16px",
            },
        ),
        html.Div(
            [
                html.Label(id="layer-label", style={"fontWeight": "bold"}),
                dcc.Slider(
                    id="layer-slider",
                    min=0,
                    max=N_LAYERS,
                    step=1,
                    value=16,
                    marks={0: "0", 8: "8", 16: "16", 24: "24", N_LAYERS: str(N_LAYERS)},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
            style={"marginBottom": "16px"},
        ),
        dcc.Graph(id="scatter", style={"height": "460px"}),
        dcc.Graph(id="metric-line", style={"height": "220px"}),
        html.P(
            "Negative finding: LinearSVC separation of chosen/rejected stays near chance "
            "(~0.51–0.55) across every layer and every checkpoint above — last-token "
            "pooled representations show no strong linearly separable preference signal, "
            "in any of SFT, DPO, or RLHF.",
            style={"color": "#f87171", "fontSize": "12px", "marginTop": "8px", "marginBottom": "20px"},
        ),
        html.Hr(style={"margin": "8px 0 20px 0", "borderColor": "#333"}),
        html.H4("Representational Drift Across Alignment Stages", style={"marginBottom": "4px"}),
        html.P(
            "Linear CKA between checkpoints on the same inputs, layer by layer. "
            "Most of the shift happens at the SFT→DPO step; the subsequent RLHF step "
            "changes the geometry only marginally (DPO vs. RLHF CKA stays near 0.999, "
            "min 0.9990, through layer 32).",
            style={"color": "#6b7280", "fontSize": "13px", "marginBottom": "12px"},
        ),
        dcc.Graph(id="drift-line", figure=build_drift_line(RLHF_DRIFT), style={"height": "320px"}),
        html.Hr(style={"margin": "8px 0 20px 0", "borderColor": "#333"}),
        html.H4("Layer-wise CKA Heatmaps", style={"marginBottom": "4px"}),
        html.P(
            "Full layer-by-layer CKA, not just matched layer indices — shows whether "
            "representations shift to a different layer depth, not only how much they drift.",
            style={"color": "#6b7280", "fontSize": "13px", "marginBottom": "12px"},
        ),
        dcc.Graph(
            id="rlhf-cka-heatmap-sft-dpo",
            figure=build_rlhf_cka_heatmap(
                RLHF_CKA_MATRICES["sft_dpo"], RLHF_CKA_MATRICES["layer_names"],
                "Linear CKA: SFT layers vs. DPO layers", "SFT layer", "DPO layer",
            ),
            style={"height": "480px"},
        ),
        dcc.Graph(
            id="rlhf-cka-heatmap-dpo-rlhf",
            figure=build_rlhf_cka_heatmap(
                RLHF_CKA_MATRICES["dpo_rlhf"], RLHF_CKA_MATRICES["layer_names"],
                "Linear CKA: DPO layers vs. RLHF layers", "DPO layer", "RLHF layer",
            ),
            style={"height": "480px"},
        ),
        dcc.Graph(
            id="rlhf-cka-heatmap-sft-rlhf",
            figure=build_rlhf_cka_heatmap(
                RLHF_CKA_MATRICES["sft_rlhf"], RLHF_CKA_MATRICES["layer_names"],
                "Linear CKA: SFT layers vs. RLHF layers", "SFT layer", "RLHF layer",
            ),
            style={"height": "480px"},
        ),
        html.Hr(style={"margin": "8px 0 20px 0", "borderColor": "#333"}),
        html.H4("Geometry Diagnostics", style={"marginBottom": "4px"}),
        html.P(
            "Anisotropy: average cosine similarity between random embedding pairs — near 1 means "
            "representations collapse into a narrow cone. Cohen's d: chosen/rejected separation "
            "projected onto the single best linear direction — detects a subtle effect even where "
            "LinearSVC (regularized, full-dimensional) found none. Effective rank: how many "
            "dimensions the representation actually uses (participation ratio of the covariance "
            "spectrum).",
            style={"color": "#6b7280", "fontSize": "13px", "marginBottom": "12px"},
        ),
        dcc.Graph(
            id="anisotropy-line",
            figure=build_geometry_line(
                {slug: d["geometry"]["anisotropy"] for slug, d in APP_DATA.items()},
                title="Anisotropy per layer",
                yaxis_title="Avg. cosine similarity",
            ),
            style={"height": "280px"},
        ),
        dcc.Graph(
            id="cohens-d-line",
            figure=build_geometry_line(
                {slug: d["geometry"]["cohens_d"] for slug, d in APP_DATA.items()},
                title="Chosen vs. rejected effect size (Cohen's d) per layer",
                yaxis_title="Cohen's d",
            ),
            style={"height": "280px"},
        ),
        dcc.Graph(
            id="effective-rank-line",
            figure=build_geometry_line(
                {slug: d["geometry"]["effective_rank"] for slug, d in APP_DATA.items()},
                title="Effective rank per layer",
                yaxis_title="Effective rank",
            ),
            style={"height": "280px"},
        ),
    ]
)

# ── Part 1 layout ─────────────────────────────────────────────────────────────

if PART1_DATA is not None:
    _PART1_VISION_UMAP_LAYERS = sorted(PART1_DATA["umap"]["vision"]["phase1"].keys())
    _PART1_LANGUAGE_UMAP_LAYERS = sorted(PART1_DATA["umap"]["language"]["phase1"].keys())

    _part1_layout = html.Div(
        [
            html.P(
                "Cross-modal CKA between DINOv2 (vision teacher) and Llama-3B (language student) "
                "on matched MS-COCO (image, caption) pairs, comparing the phase_1 baseline against "
                "the full_train_set checkpoint after teacher-student CKA fine-tuning. "
                "Does a shared representational structure emerge — and does it strengthen with training?",
                style={"color": "#6b7280", "fontSize": "13px", "marginBottom": "12px"},
            ),
            html.H4("UMAP: Structure by Layer and Training Phase", style={"marginBottom": "4px"}),
            html.P(
                "Per-layer 2D UMAP of vision and language activations, colored by COCO "
                "supercategory. Move the layer slider or switch phase to see whether "
                "same-category points in the two modalities start occupying similar "
                "regions of the embedding space as training progresses.",
                style={"color": "#6b7280", "fontSize": "13px", "marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Phase", style={"fontWeight": "bold"}),
                            dcc.RadioItems(
                                id="part1-vision-phase-selector",
                                options=[
                                    {"label": "phase_1 (baseline)", "value": "phase1"},
                                    {"label": "full_train_set (final)", "value": "full_train"},
                                ],
                                value="phase1",
                                inline=True,
                                inputStyle={"marginRight": "4px"},
                                labelStyle={"marginRight": "16px"},
                            ),
                            html.Label(id="part1-vision-layer-label", style={"fontWeight": "bold"}),
                            dcc.Slider(
                                id="part1-vision-layer-slider",
                                min=min(_PART1_VISION_UMAP_LAYERS),
                                max=max(_PART1_VISION_UMAP_LAYERS),
                                step=1,
                                value=_PART1_VISION_UMAP_LAYERS[len(_PART1_VISION_UMAP_LAYERS) // 2],
                                marks={i: str(i) for i in _PART1_VISION_UMAP_LAYERS[::4]},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            dcc.Graph(id="part1-vision-umap", style={"height": "420px"}),
                        ],
                        style={"flex": 1, "marginRight": "16px"},
                    ),
                    html.Div(
                        [
                            html.Label("Phase", style={"fontWeight": "bold"}),
                            dcc.RadioItems(
                                id="part1-language-phase-selector",
                                options=[
                                    {"label": "phase_1 (baseline)", "value": "phase1"},
                                    {"label": "cka_phase_2", "value": "phase2"},
                                    {"label": "full_train_set (final)", "value": "full_train"},
                                ],
                                value="phase1",
                                inline=True,
                                inputStyle={"marginRight": "4px"},
                                labelStyle={"marginRight": "16px"},
                            ),
                            html.Label(id="part1-language-layer-label", style={"fontWeight": "bold"}),
                            dcc.Slider(
                                id="part1-language-layer-slider",
                                min=min(_PART1_LANGUAGE_UMAP_LAYERS),
                                max=max(_PART1_LANGUAGE_UMAP_LAYERS),
                                step=1,
                                value=_PART1_LANGUAGE_UMAP_LAYERS[len(_PART1_LANGUAGE_UMAP_LAYERS) // 2],
                                marks={i: str(i) for i in _PART1_LANGUAGE_UMAP_LAYERS[::4]},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            dcc.Graph(id="part1-language-umap", style={"height": "420px"}),
                        ],
                        style={"flex": 1},
                    ),
                ],
                style={"display": "flex"},
            ),
            html.Hr(style={"margin": "24px 0", "borderColor": "#333"}),
            dcc.Graph(
                id="part1-cka-heatmaps",
                figure=build_phase_cka_heatmaps(PART1_DATA["matrices"]),
                style={"height": "460px"},
            ),
            dcc.Graph(
                id="part1-scalar-bar",
                figure=build_phase_scalar_bar(PART1_DATA["matrices"]),
                style={"height": "320px"},
            ),
            html.Hr(style={"margin": "24px 0", "borderColor": "#333"}),
            html.H4("Cross-modal Alignment by COCO Supercategory", style={"marginBottom": "4px"}),
            html.P(
                "Linear CKA between vision and language activations (last clean layer of each), "
                "grouped into the 12 COCO supercategories, comparing the phase_1 baseline against "
                "the full_train_set checkpoint. Shows whether each category's cross-modal alignment "
                "moved closer together or stayed the same over training.",
                style={"color": "#6b7280", "fontSize": "13px", "marginBottom": "12px"},
            ),
            dcc.Graph(
                id="part1-supercategory-dumbbell",
                figure=build_supercategory_dumbbell(PART1_DATA["supercategory"]),
            ),
        ]
    )
else:
    _part1_layout = html.Div(
        [
            html.P(
                "Part 1 embeddings not found. Expected under 'data/embeddings part 1/'.",
                style={"color": "#f87171", "padding": "32px"},
            )
        ]
    )

# ── App layout ────────────────────────────────────────────────────────────────

app.layout = html.Div(
    [
        html.H2(
            "Representational Geometry in Neural Networks",
            style={"marginBottom": "4px"},
        ),
        html.P(
            "Does a shared representation emerge when vision and language models train on the same task?",
            style={"color": "#6b7280", "marginTop": "0", "marginBottom": "20px", "fontSize": "14px"},
        ),
        dcc.Tabs(
            id="main-tabs",
            value="tab-part1",
            children=[
                dcc.Tab(
                    label="Part 1 — Cross-modal Alignment · DINOv2 (Teacher) × Llama-3B (Student) · MS-COCO",
                    value="tab-part1",
                    style=_tab_style,
                    selected_style=_selected_tab_style,
                    children=_part1_layout,
                ),
                dcc.Tab(
                    label="Part 2 — RLHF Geometry · Tulu-3-8B SFT → DPO → RLHF",
                    value="tab-llm",
                    style=_tab_style,
                    selected_style=_selected_tab_style,
                    children=_part2_layout,
                ),
            ],
            style={"marginBottom": "24px"},
        ),
    ],
    style={
        "maxWidth": "1100px",
        "margin": "0 auto",
        "padding": "24px",
        "fontFamily": "sans-serif",
    },
)

from app.callbacks import register  # noqa: E402

register(app, APP_DATA)

if PART1_DATA is not None:
    from app.part1_callbacks import register as register_part1  # noqa: E402

    register_part1(app, PART1_DATA["umap"])

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
