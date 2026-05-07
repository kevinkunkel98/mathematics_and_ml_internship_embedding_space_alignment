"""
Generate synthetic vision data for dashboard testing without a GPU.

Produces HDF5 files in the same format as extract_vision_embeddings.py:
  data/vision/resnet18.h5
  data/vision/vit_b16.h5

ResNet-18: activations that cluster by class in later layers (typical CNN behaviour).
ViT-B/16:  activations with a different cluster structure, showing partial CKA alignment
           — later ViT layers align with later CNN layers, early layers are uncorrelated.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scripts.io import save_vision_data

N_SAMPLES = 500
N_CAM_SAMPLES = 40  # subset stored as images+CAMs
N_CIFAR_CLASSES = 10
SEED = 0

# ResNet-18: 4 block groups × 2 layers + stem + fc-pre = 10 layers
CNN_LAYERS = 10
CNN_DIM = 512

# ViT-B/16: embedding + 12 transformer layers = 13 layers
VIT_LAYERS = 13
VIT_DIM = 768


def _class_labels(rng: np.random.Generator) -> np.ndarray:
    labels = np.repeat(np.arange(N_CIFAR_CLASSES), N_SAMPLES // N_CIFAR_CLASSES)
    rng.shuffle(labels)
    return labels.astype(np.int8)


def _make_cnn_activations(
    rng: np.random.Generator, labels: np.ndarray
) -> dict[int, np.ndarray]:
    """Early layers: random. Later layers: class clusters emerge progressively."""
    acts = {}
    for i in range(CNN_LAYERS):
        X = rng.standard_normal((N_SAMPLES, CNN_DIM)).astype(np.float32)
        cluster_strength = (i / (CNN_LAYERS - 1)) * 3.0
        for c in range(N_CIFAR_CLASSES):
            center = rng.standard_normal(CNN_DIM).astype(np.float32) * cluster_strength
            X[labels == c] += center
        acts[i] = X
    return acts


def _make_vit_activations(
    rng: np.random.Generator, labels: np.ndarray, cnn_acts: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """ViT activations: correlated with corresponding-depth CNN layers in later layers,
    but with different geometry (different projection)."""
    acts = {}
    for i in range(VIT_LAYERS):
        X = rng.standard_normal((N_SAMPLES, VIT_DIM)).astype(np.float32)
        cnn_depth_frac = i / (VIT_LAYERS - 1)
        cnn_idx = int(cnn_depth_frac * (CNN_LAYERS - 1))
        # Align later ViT layers with corresponding CNN layers via shared class signal
        align_strength = cnn_depth_frac * 2.0
        cnn_signal = (
            cnn_acts[cnn_idx][:, :VIT_DIM]
            if CNN_DIM >= VIT_DIM
            else np.pad(cnn_acts[cnn_idx], ((0, 0), (0, VIT_DIM - CNN_DIM)))
        )
        X += cnn_signal * align_strength * 0.3
        acts[i] = X
    return acts


def _make_mock_images(rng: np.random.Generator, labels: np.ndarray) -> np.ndarray:
    """Synthetic 32×32 RGB images — coloured noise, tinted by class."""
    class_tints = rng.uniform(0.2, 0.8, (N_CIFAR_CLASSES, 3)).astype(np.float32)
    imgs = rng.uniform(0, 0.4, (N_CAM_SAMPLES, 32, 32, 3)).astype(np.float32)
    for i in range(N_CAM_SAMPLES):
        imgs[i] += class_tints[labels[i]] * 0.6
    return imgs.clip(0, 1)


def _make_mock_cams(
    rng: np.random.Generator, labels: np.ndarray, strong: bool
) -> np.ndarray:
    """CAMs with a focal bright region. strong=True mimics a more focused model (ViT)."""
    cams = rng.uniform(0, 0.3, (N_CAM_SAMPLES, 32, 32)).astype(np.float32)
    spread = 4 if strong else 8
    for i in range(N_CAM_SAMPLES):
        cx, cy = rng.integers(8, 24, size=2)
        yy, xx = np.ogrid[:32, :32]
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * spread**2)).astype(
            np.float32
        )
        cams[i] += blob
    return cams.clip(0, 1)


def main() -> None:
    rng = np.random.default_rng(SEED)
    labels = _class_labels(rng)
    cam_labels = labels[:N_CAM_SAMPLES]

    cnn_acts = _make_cnn_activations(rng, labels)
    vit_acts = _make_vit_activations(rng, labels, cnn_acts)

    images = _make_mock_images(rng, cam_labels)
    cnn_cams = _make_mock_cams(rng, cam_labels, strong=False)
    vit_cams = _make_mock_cams(rng, cam_labels, strong=True)

    out_dir = Path("data/vision")
    save_vision_data(out_dir / "resnet18.h5", cnn_acts, labels, images, cnn_cams)
    print(
        f"Saved ResNet-18: {CNN_LAYERS} layers × {N_SAMPLES} samples → {out_dir}/resnet18.h5"
    )

    save_vision_data(out_dir / "vit_b16.h5", vit_acts, labels, images, vit_cams)
    print(
        f"Saved ViT-B/16:  {VIT_LAYERS} layers × {N_SAMPLES} samples → {out_dir}/vit_b16.h5"
    )

    print("\nMock data ready. Launch the dashboard with:")
    print("  python app/vision_app.py")


if __name__ == "__main__":
    main()
