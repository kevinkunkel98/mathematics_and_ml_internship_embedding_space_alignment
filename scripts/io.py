import h5py
import numpy as np
from pathlib import Path


def save_embeddings(path: str | Path, layers: dict[int, np.ndarray], labels: np.ndarray) -> None:
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
