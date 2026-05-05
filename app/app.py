from pathlib import Path
from dash import Dash, dcc, html

from scripts.io import load_embeddings
from app.compute import fit_all

_MODELS = {
    "meta-llama--Meta-Llama-3-8B": "Llama-3-8B (base)",
    "meta-llama--Meta-Llama-3-8B-Instruct": "Llama-3-8B-Instruct",
}


def _load_app_data() -> dict:
    data = {}
    for slug in _MODELS:
        path = Path("data/embeddings") / slug / "layers.h5"
        if not path.exists():
            model_id = slug.replace("--", "/")
            raise FileNotFoundError(
                f"Missing {path}\n"
                f"Run: python scripts/extract_embeddings.py --model {model_id}"
            )
        layers, labels = load_embeddings(path)
        print(f"Fitting projections for {slug} ...")
        cache = fit_all(slug, layers, labels)
        data[slug] = {"cache": cache, "labels": labels}
    return data


APP_DATA = _load_app_data()
N_LAYERS = max(next(iter(APP_DATA.values()))["cache"]["umap"].keys())

app = Dash(__name__, title="RLHF Embedding Visualizer")

app.layout = html.Div(
    [
        html.H2("RLHF Embedding Space Visualizer"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Model", style={"fontWeight": "bold"}),
                        dcc.RadioItems(
                            id="model-selector",
                            options=[{"label": v, "value": k} for k, v in _MODELS.items()],
                            value=list(_MODELS.keys())[0],
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
            style={"display": "flex", "alignItems": "flex-start", "marginBottom": "16px"},
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
    ],
    style={"maxWidth": "960px", "margin": "0 auto", "padding": "24px", "fontFamily": "sans-serif"},
)

from app.callbacks import register  # noqa: E402
register(app, APP_DATA)

if __name__ == "__main__":
    app.run(debug=True)
