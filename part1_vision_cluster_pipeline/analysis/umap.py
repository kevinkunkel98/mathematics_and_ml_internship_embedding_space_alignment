from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import umap
import torch

def run_umap(
    act_dir: Path,
    labels,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> Dict[str, plt.Figure]:
    """
    Run UMAP on saved activations.
    Expects:
        activations/
            cnn/*.pt
            vit/*.pt

    Args:
        act_dir: path to activations/ (contains cnn/ and vit/)
        labels: tensor or array of shape (B,)
        n_neighbors, min_dist: UMAP hyperparameters

    Returns:
        dict: {layer_name: matplotlib Figure}
    """
    labels = labels.cpu().numpy() if hasattr(labels, "cpu") else labels

    figures: Dict[str, plt.Figure] = {}

    for model_type in ["cnn", "vit"]:
        model_dir = act_dir / model_type
        if not model_dir.exists():
            print(f"[UMAP] Missing directory: {model_dir}")
            continue
        
        act_files = list(model_dir.glob("*.pt"))
        if len(act_files) == 0:
            print(f"[UMAP] No activations found in {model_dir}")
            continue

        for act_file in act_files:
            layer_name = act_file.stem
            X = torch.load(act_file)          # (B, D) or (B, ...)
            X = X.reshape(X.shape[0], -1)     # flatten
            X = X.cpu().numpy()

            k = min(n_neighbors, X.shape[0] - 1) # Prefer 500–2000 samples for stable UMAP

            reducer = umap.UMAP(
                n_neighbors=k,
                min_dist=min_dist,
                n_components=2,
                random_state=random_state,
            )

            Z = reducer.fit_transform(X)  # (B, 2)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(
                Z[:, 0],
                Z[:, 1],
                c=labels,
                cmap="tab20",
                s=40,
                alpha=0.9,
            )

            ax.set_title(f"UMAP – {model_type.upper()} – {layer_name}")
            ax.set_xticks([])
            ax.set_yticks([])

            figures[f"{model_type}_{layer_name}"] = fig

    return figures
