from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dash import Dash, dcc, html

from scripts.io import load_embeddings, load_vision_data
from app.compute import fit_all
from app.vision_compute import fit_vision, fit_vision_umap
from app.crossmodal_compute import fit_crossmodal

_LLM_MODELS = {
    "allenai--Llama-3.1-Tulu-3-8B-SFT": "Tulu-3-8B SFT",
    "allenai--Llama-3.1-Tulu-3-8B-DPO": "Tulu-3-8B DPO",
    "allenai--Llama-3.1-Tulu-3-8B":      "Tulu-3-8B RLHF",
}

_VISION_MODELS = {
    "resnet18": "ResNet-18",
    "vit_b16": "ViT-B/16",
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
        data[slug] = {"cache": cache, "labels": labels}
    return data


def _load_vision_data() -> dict | None:
    try:
        vision = {}
        for slug in _VISION_MODELS:
            path = Path("data/vision") / f"{slug}.h5"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}\nRun: python scripts/generate_mock_vision.py"
                )
            activations, labels, images, cams, class_names = load_vision_data(path)
            vision[slug] = {
                "activations": activations,
                "labels": labels,
                "images": images,
                "cams": cams,
                "class_names": class_names,
            }
            print(
                f"Loaded vision {slug}: {len(activations)} layers, {len(labels)} samples"
            )

        print("Fitting CKA...")
        cka = fit_vision(
            vision["resnet18"]["activations"], vision["vit_b16"]["activations"]
        )

        print("Fitting vision UMAP...")
        umap_data = fit_vision_umap(
            vision["resnet18"]["activations"], vision["vit_b16"]["activations"]
        )

        return {
            "cka": cka,
            "labels": vision["resnet18"]["labels"],
            "images": vision["resnet18"]["images"],
            "cnn_cams": vision["resnet18"]["cams"],
            "vit_cams": vision["vit_b16"]["cams"],
            "umap": umap_data,
            "class_names": vision["resnet18"]["class_names"],
            "model_labels": _VISION_MODELS,
            "n_layers": {
                slug: max(vision[slug]["activations"].keys()) for slug in _VISION_MODELS
            },
        }
    except FileNotFoundError as e:
        print(f"[vision] Skipping Part 1 — {e}")
        return None


APP_DATA = _load_llm_data()
VISION_DATA = _load_vision_data()


def _load_crossmodal_data() -> dict | None:
    try:
        base_path = Path("data/embeddings/crossmodal/llama-base.h5")
        instruct_path = Path("data/embeddings/crossmodal/llama-instruct.h5")
        vision_path = Path("data/embeddings/crossmodal/vision.h5")
        clip_path = Path("data/embeddings/crossmodal/clip-text.h5")

        for p in (base_path, instruct_path, vision_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing {p}\nRun: python scripts/generate_mock_crossmodal.py"
                )

        base_layers, _ = load_embeddings(base_path)
        instruct_layers, _ = load_embeddings(instruct_path)
        vision_layers, _ = load_embeddings(vision_path)
        clip_layers = load_embeddings(clip_path)[0] if clip_path.exists() else None

        print("Fitting cross-modal CKA...")
        return fit_crossmodal(vision_layers, base_layers, instruct_layers, clip_layers)
    except FileNotFoundError as e:
        print(f"[crossmodal] Skipping Part 3 — {e}")
        return None


CROSSMODAL_DATA = _load_crossmodal_data()

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
    ]
)

# ── Part 1 layout ─────────────────────────────────────────────────────────────

if VISION_DATA is not None:
    _vision_max_layer = max(VISION_DATA["n_layers"].values())
    _vision_layer_marks = {
        i: str(i)
        for i in range(0, _vision_max_layer + 1, max(1, _vision_max_layer // 6))
    }
    _vision_layer_marks[_vision_max_layer] = str(_vision_max_layer)

    _vision_class_names = VISION_DATA.get("class_names") or [
        str(i) for i in range(int(VISION_DATA["labels"].max()) + 1)
    ]

    _part1_layout = html.Div(
        [
            # ── CKA heatmap ───────────────────────────────────────────────────
            dcc.Graph(id="cka-heatmap", style={"height": "480px"}),
            # ── Vision UMAP ───────────────────────────────────────────────────
            html.Hr(style={"margin": "24px 0", "borderColor": "#333"}),
            html.H4(
                "Layer Activation Space — UMAP",
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Model", style={"fontWeight": "bold"}),
                            dcc.RadioItems(
                                id="vision-model-selector",
                                options=[
                                    {"label": "ResNet-18", "value": "resnet18"},
                                    {"label": "ViT-B/16", "value": "vit_b16"},
                                ],
                                value="resnet18",
                                inline=True,
                                inputStyle={"marginRight": "4px"},
                                labelStyle={"marginRight": "16px"},
                            ),
                        ],
                        style={"marginBottom": "12px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                id="vision-layer-label",
                                style={"fontWeight": "bold"},
                            ),
                            dcc.Slider(
                                id="vision-layer-slider",
                                min=0,
                                max=_vision_max_layer,
                                step=1,
                                value=0,
                                marks=_vision_layer_marks,
                                tooltip={
                                    "placement": "bottom",
                                    "always_visible": False,
                                },
                            ),
                        ],
                        style={"marginBottom": "8px"},
                    ),
                ]
            ),
            dcc.Graph(id="vision-umap", style={"height": "460px"}),
            # ── CAM comparison ────────────────────────────────────────────────
            html.Hr(style={"margin": "24px 0", "borderColor": "#333"}),
            html.Div(
                [
                    html.Label("Filter CAMs by class", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="class-selector",
                        options=[{"label": "All classes", "value": "all"}]
                        + [
                            {"label": c, "value": i}
                            for i, c in enumerate(_vision_class_names)
                        ],
                        value="all",
                        clearable=False,
                        style={"width": "240px", "color": "#111"},
                    ),
                ],
                style={"marginBottom": "16px"},
            ),
            dcc.Graph(id="cam-comparison"),
        ]
    )
else:
    _part1_layout = html.Div(
        [
            html.P(
                "Vision data not found. Run: python scripts/generate_mock_vision.py",
                style={"color": "#f87171", "padding": "32px"},
            )
        ]
    )

# ── Part 3 layout ─────────────────────────────────────────────────────────────

if CROSSMODAL_DATA is not None:
    _part3_layout = html.Div(
        [
            html.P(
                "CKA between DINOv2 (vision) and Llama-3-8B (language) on matched MS-COCO (image, caption) pairs. "
                "Does a shared representational structure emerge? CLIP sets the upper bound.",
                style={"color": "#6b7280", "fontSize": "13px", "marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Label("Show CLIP upper bound", style={"fontWeight": "bold"}),
                    dcc.Checklist(
                        id="crossmodal-clip-toggle",
                        options=[{"label": " CLIP ViT-B/32 (explicitly trained for cross-modal alignment)", "value": "clip"}],
                        value=["clip"] if CROSSMODAL_DATA["clip"] is not None else [],
                        inputStyle={"marginRight": "6px"},
                    ),
                ],
                style={"marginBottom": "16px"},
            ),
            dcc.Graph(id="crossmodal-heatmaps", style={"height": "440px"}),
            dcc.Graph(id="crossmodal-scalar-bar", style={"height": "320px"}),
        ]
    )
else:
    _part3_layout = html.Div(
        [
            html.P(
                "Cross-modal data not found. Run: python scripts/generate_mock_crossmodal.py",
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
            value="tab-crossmodal",
            children=[
                dcc.Tab(
                    label="Part 1 — Cross-modal Alignment · DINOv2 × Llama · MS-COCO",
                    value="tab-crossmodal",
                    style=_tab_style,
                    selected_style=_selected_tab_style,
                    children=_part3_layout,
                ),
                dcc.Tab(
                    label="Part 2 — RLHF Geometry · Tulu-3-8B SFT → DPO → RLHF",
                    value="tab-llm",
                    style=_tab_style,
                    selected_style=_selected_tab_style,
                    children=_part2_layout,
                ),
                dcc.Tab(
                    label="Vision Layer Explorer · ResNet-18 vs ViT-B/16",
                    value="tab-vision",
                    style=_tab_style,
                    selected_style=_selected_tab_style,
                    children=_part1_layout,
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
from app.vision_callbacks import register as register_vision  # noqa: E402
from app.crossmodal_callbacks import register as register_crossmodal  # noqa: E402

register(app, APP_DATA)
if VISION_DATA is not None:
    register_vision(app, VISION_DATA)
if CROSSMODAL_DATA is not None:
    register_crossmodal(app, CROSSMODAL_DATA)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
