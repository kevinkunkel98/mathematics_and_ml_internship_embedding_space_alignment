import h5py
import numpy as np
from pathlib import Path


def save_embeddings(
    path: str | Path, layers: dict[int, np.ndarray], labels: np.ndarray
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for layer_idx, arr in layers.items():
            f.create_dataset(f"layer_{layer_idx:02d}", data=arr.astype(np.float16))
        f.create_dataset("labels", data=labels.astype(np.int8))


def load_embeddings(path: str | Path) -> tuple[dict[int, np.ndarray], np.ndarray]:
    path = Path(path)
    layers: dict[int, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if key.startswith("layer_"):
                idx = int(key.split("_")[1])
                layers[idx] = f[key][:]
        labels = f["labels"][:]
    return layers, labels


def save_vision_data(
    path: str | Path,
    activations: dict[int, np.ndarray],
    labels: np.ndarray,
    images: np.ndarray,
    cams: np.ndarray,
) -> None:
    """Save vision model data: layer activations, class labels, sample images, CAM heatmaps.

    activations: layer_idx -> (n_samples, dim)
    labels:      (n_samples,) int8 — CIFAR-10 class indices
    images:      (n_cam_samples, 32, 32, 3) float32 in [0, 1]
    cams:        (n_cam_samples, 32, 32) float32 — per-image CAM heatmaps
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        acts_grp = f.create_group("activations")
        for layer_idx, arr in activations.items():
            acts_grp.create_dataset(
                f"layer_{layer_idx:02d}", data=arr.astype(np.float16)
            )
        f.create_dataset("labels", data=labels.astype(np.int8))
        f.create_dataset("images", data=images.astype(np.float32))
        f.create_dataset("cams", data=cams.astype(np.float32))


def load_vision_data(
    path: str | Path,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Returns activations, labels, images, cams."""
    path = Path(path)
    activations: dict[int, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in f["activations"].keys():
            idx = int(key.split("_")[1])
            activations[idx] = f["activations"][key][:]
        labels = f["labels"][:]
        images = f["images"][:]
        cams = f["cams"][:]
    return activations, labels, images, cams
