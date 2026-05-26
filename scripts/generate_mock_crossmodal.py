"""
Generate synthetic cross-modal embeddings for dashboard testing without GPU.

Simulates matched (image, caption) pairs for N_PAIRS samples with:
  - vision_layers:             ViT-like activations (13 layers, dim=768)
  - language_layers_base:      Llama-base-like activations (33 layers, dim=4096)
  - language_layers_instruct:  RLHF-shifted — later layers more aligned to vision
  - language_layers_clip:      CLIP text encoder upper bound — highly aligned at all layers

Saves to:
  data/embeddings/crossmodal/vision.h5
  data/embeddings/crossmodal/llama-base.h5
  data/embeddings/crossmodal/llama-instruct.h5
  data/embeddings/crossmodal/clip-text.h5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scripts.io import save_embeddings

N_PAIRS = 500
N_VISION_LAYERS = 13  # ViT-B/32: patch embed + 12 transformer blocks
N_LANG_LAYERS = 33  # Llama-3-8B: embed + 32 transformer blocks
VISION_DIM = 768
LANG_DIM = 4096
CLIP_DIM = 512
SEED = 0


def _shared_semantic(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """Low-dim semantic signal projected to `dim`."""
    latent = rng.standard_normal((n, 32)).astype(np.float32)
    proj = rng.standard_normal((32, dim)).astype(np.float32) * 0.1
    return latent @ proj


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = Path("data/embeddings/crossmodal")

    semantic = _shared_semantic(rng, N_PAIRS, max(VISION_DIM, LANG_DIM))
    # dummy labels (0 = no category distinction for cross-modal)
    labels = np.zeros(N_PAIRS, dtype=np.int8)

    # --- Vision layers (ViT-like): alignment grows in later layers ---
    vision_layers: dict[int, np.ndarray] = {}
    for i in range(N_VISION_LAYERS):
        noise = rng.standard_normal((N_PAIRS, VISION_DIM)).astype(np.float32)
        alpha = (i / max(N_VISION_LAYERS - 1, 1)) * 0.8
        vision_layers[i] = noise + alpha * semantic[:, :VISION_DIM]
    save_embeddings(out / "vision.h5", vision_layers, labels)
    print(
        f"Saved vision: {N_VISION_LAYERS} layers × {N_PAIRS} samples → {out}/vision.h5"
    )

    # --- Llama base: minimal alignment at all layers ---
    base_layers: dict[int, np.ndarray] = {}
    for j in range(N_LANG_LAYERS):
        noise = rng.standard_normal((N_PAIRS, LANG_DIM)).astype(np.float32)
        alpha = 0.05
        base_layers[j] = noise + alpha * semantic[:, :LANG_DIM]
    save_embeddings(out / "llama-base.h5", base_layers, labels)
    print(
        f"Saved llama-base: {N_LANG_LAYERS} layers × {N_PAIRS} samples → {out}/llama-base.h5"
    )

    # --- Llama instruct (RLHF): alignment increases in later layers ---
    instruct_layers: dict[int, np.ndarray] = {}
    for j in range(N_LANG_LAYERS):
        noise = rng.standard_normal((N_PAIRS, LANG_DIM)).astype(np.float32)
        alpha = 0.05 + (j / max(N_LANG_LAYERS - 1, 1)) * 0.4
        instruct_layers[j] = noise + alpha * semantic[:, :LANG_DIM]
    save_embeddings(out / "llama-instruct.h5", instruct_layers, labels)
    print(
        f"Saved llama-instruct: {N_LANG_LAYERS} layers × {N_PAIRS} samples → {out}/llama-instruct.h5"
    )

    # --- CLIP text upper bound: strong alignment throughout ---
    clip_layers: dict[int, np.ndarray] = {}
    clip_vis_dim = min(CLIP_DIM, VISION_DIM)
    for j in range(N_VISION_LAYERS):
        noise = rng.standard_normal((N_PAIRS, CLIP_DIM)).astype(np.float32) * 0.2
        clip_layers[j] = noise + semantic[:, :CLIP_DIM]
    save_embeddings(out / "clip-text.h5", clip_layers, labels)
    print(
        f"Saved clip-text: {N_VISION_LAYERS} layers × {N_PAIRS} samples → {out}/clip-text.h5"
    )

    print("\nMock cross-modal data ready.")


if __name__ == "__main__":
    main()
