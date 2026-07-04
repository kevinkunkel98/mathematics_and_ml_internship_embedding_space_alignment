"""
Generate synthetic embeddings for dashboard testing without a GPU or HF token.

Produces HDF5 files in the same format as extract_embeddings.py:
  data/embeddings/allenai--Llama-3.1-Tulu-3-8B-SFT/layers.h5
  data/embeddings/allenai--Llama-3.1-Tulu-3-8B-DPO/layers.h5
  data/embeddings/allenai--Llama-3.1-Tulu-3-8B/layers.h5

Each checkpoint shows progressively stronger chosen/rejected separation,
mimicking the geometric effect of the SFT → DPO → RLHF alignment pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scripts.io import save_embeddings

N_SAMPLES = 1000   # 500 chosen + 500 rejected
N_LAYERS = 33      # embedding layer + 32 transformer layers
HIDDEN_DIM = 4096
SEED = 42


def _make_labels() -> np.ndarray:
    labels = np.zeros(N_SAMPLES, dtype=np.int8)
    labels[::2] = 1   # even indices = chosen
    return labels


def _make_layers(
    rng: np.random.Generator,
    labels: np.ndarray,
    max_separation: float,
) -> dict[int, np.ndarray]:
    """Random embeddings with chosen/rejected separation ramping up across layers."""
    layers = {}
    for i in range(N_LAYERS):
        X = rng.standard_normal((N_SAMPLES, HIDDEN_DIM)).astype(np.float32)
        sep = (i / (N_LAYERS - 1)) * max_separation
        signal = np.zeros(HIDDEN_DIM, dtype=np.float32)
        signal[:64] = sep
        X[labels == 1] += signal
        X[labels == 0] -= signal
        layers[i] = X
    return layers


def main() -> None:
    rng = np.random.default_rng(SEED)
    labels = _make_labels()

    # Increasing separation mirrors the SFT → DPO → RLHF alignment progression
    models = {
        "allenai--Llama-3.1-Tulu-3-8B-SFT": 0.8,
        "allenai--Llama-3.1-Tulu-3-8B-DPO": 1.5,
        "allenai--Llama-3.1-Tulu-3-8B":      2.5,
    }

    for slug, max_sep in models.items():
        layers = _make_layers(rng, labels, max_sep)
        path = Path("data/embeddings") / slug / "layers.h5"
        save_embeddings(path, layers, labels)
        print(f"Saved {N_LAYERS} layers × {N_SAMPLES} samples → {path}")

    print("\nMock data ready. Launch the dashboard with:")
    print("  python app/app.py")


if __name__ == "__main__":
    main()
