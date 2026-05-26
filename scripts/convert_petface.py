"""Convert Sarah & Marla's PetFace experiment outputs to dashboard-compatible HDF5.

Expected experiment folder layout (as created by OutputManager + extract_and_save_activations):

  <experiment_root>/
    activations/
      cnn/
        pool3.pt    # torch.Tensor (N, D) float32
        pool4.pt
        gap.pt
        ...
      vit/
        block0.pt
        block1.pt
        ...
    metrics/
      labels.pt     # torch.Tensor (N,) int64 — breed indices
    config.yaml     # optional; breed names read from class_names.txt if present

Usage:
  python scripts/convert_petface.py \\
    --cnn  experiments/cnn_petfaces/20260215_145230 \\
    --vit  experiments/vit_petfaces/20260215_190413 \\
    --class-names data/cat/class_names.txt \\
    --out  data/vision

Outputs:
  data/vision/resnet18.h5   (cnn activations, labels, placeholder images/cams)
  data/vision/vit_b16.h5   (vit activations, labels, placeholder images/cams)

Images and CAMs cannot be recovered from saved activations alone; dummy arrays are
written so the dashboard loads without error.  Replace them by re-running extraction
with save_images=True if image-level visualisation is needed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.io import save_vision_data


# Layer name → integer index, deterministic sort-order for each model type
def _layer_order(names: list[str]) -> list[str]:
    """Sort layer names: numeric suffix first (block0<block1), then alphabetically."""

    def _key(n):
        digits = "".join(c for c in n if c.isdigit())
        return (int(digits) if digits else 9999, n)

    return sorted(names, key=_key)


def _load_pt_activations(folder: Path) -> dict[int, np.ndarray]:
    """Load all *.pt tensors in folder; return {layer_idx: array (N, D)}."""
    pt_files = sorted(folder.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {folder}")

    import torch

    names = [p.stem for p in pt_files]
    ordered = _layer_order(names)
    activations: dict[int, np.ndarray] = {}
    for idx, name in enumerate(ordered):
        tensor = torch.load(
            folder / f"{name}.pt", map_location="cpu", weights_only=True
        )
        activations[idx] = tensor.float().numpy()
    return activations


def _load_labels(experiment_root: Path) -> np.ndarray:
    metrics_dir = experiment_root / "metrics"
    labels_path = metrics_dir / "labels.pt"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}\n"
            "Expected a torch.Tensor (N,) saved with torch.save."
        )
    import torch

    labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    return labels.numpy().astype(np.int8)


def _dummy_images_cams(
    n: int, h: int = 224, w: int = 224
) -> tuple[np.ndarray, np.ndarray]:
    """Placeholder images and CAMs — grey images, uniform CAMs."""
    images = np.full((n, h, w, 3), 0.5, dtype=np.float32)
    cams = np.ones((n, h, w), dtype=np.float32)
    return images, cams


def _read_class_names(path: Path | None) -> list[str] | None:
    if path is None or not path.exists():
        return None
    lines = path.read_text().splitlines()
    return [l.strip() for l in lines if l.strip()]


def convert(
    cnn_root: Path,
    vit_root: Path,
    class_names_file: Path | None,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = _read_class_names(class_names_file)

    for slug, experiment_root, subfolder in [
        ("resnet18", cnn_root, "cnn"),
        ("vit_b16", vit_root, "vit"),
    ]:
        print(f"Loading {slug} activations from {experiment_root} ...")
        acts_dir = experiment_root / "activations" / subfolder
        activations = _load_pt_activations(acts_dir)
        labels = _load_labels(experiment_root)

        n = len(labels)
        images, cams = _dummy_images_cams(n)

        out_path = out_dir / f"{slug}.h5"
        save_vision_data(
            out_path, activations, labels, images, cams, class_names=class_names
        )
        print(
            f"  Wrote {out_path}: {len(activations)} layers, {n} samples"
            + (f", {len(class_names)} classes" if class_names else "")
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cnn",
        required=True,
        type=Path,
        help="CNN experiment root (contains activations/cnn/*.pt and metrics/labels.pt)",
    )
    parser.add_argument(
        "--vit",
        required=True,
        type=Path,
        help="ViT experiment root (contains activations/vit/*.pt and metrics/labels.pt)",
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        default=None,
        help="Text file with one breed name per line (optional)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/vision"),
        help="Output directory (default: data/vision)",
    )
    args = parser.parse_args()

    convert(args.cnn, args.vit, args.class_names, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
