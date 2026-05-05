"""
Generate synthetic embeddings for dashboard testing without a GPU or HF token.

Produces HDF5 files in the same format as extract_embeddings.py:
  data/embeddings/meta-llama--Meta-Llama-3-8B/layers.h5
  data/embeddings/meta-llama--Meta-Llama-3-8B-Instruct/layers.h5

The instruct model embeddings are nudged so that chosen/rejected separation
visibly increases in later layers, mimicking the real RLHF geometry shift.
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


def _make_base_layers(rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Random embeddings — no systematic chosen/rejected separation."""
    return {
        i: rng.standard_normal((N_SAMPLES, HIDDEN_DIM)).astype(np.float32)
        for i in range(N_LAYERS)
    }


def _make_instruct_layers(rng: np.random.Generator, labels: np.ndarray) -> dict[int, np.ndarray]:
    """
    Embeddings where chosen/rejected separation grows across layers,
    simulating the geometric effect of RLHF alignment.
    """
    layers = {}
    for i in range(N_LAYERS):
        X = rng.standard_normal((N_SAMPLES, HIDDEN_DIM)).astype(np.float32)
        # Linearly ramp up the separation signal with layer depth
        separation = (i / (N_LAYERS - 1)) * 2.0
        signal = np.zeros(HIDDEN_DIM, dtype=np.float32)
        signal[:64] = separation
        X[labels == 1] += signal
        X[labels == 0] -= signal
        layers[i] = X
    return layers


def main() -> None:
    rng = np.random.default_rng(SEED)
    labels = _make_labels()

    models = {
        "meta-llama--Meta-Llama-3-8B": _make_base_layers(rng),
        "meta-llama--Meta-Llama-3-8B-Instruct": _make_instruct_layers(rng, labels),
    }

    for slug, layers in models.items():
        path = Path("data/embeddings") / slug / "layers.h5"
        save_embeddings(path, layers, labels)
        print(f"Saved {N_LAYERS} layers × {N_SAMPLES} samples → {path}")

    print("\nMock data ready. Launch the dashboard with:")
    print("  python app/app.py")


if __name__ == "__main__":
    main()
