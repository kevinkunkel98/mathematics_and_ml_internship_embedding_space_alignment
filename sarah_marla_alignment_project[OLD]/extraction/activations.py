import torch
from pathlib import Path
from typing import Dict, Callable


@torch.no_grad()
def extract_and_save_activations(
    model: torch.nn.Module,
    layers: Dict[str, torch.nn.Module],
    images: torch.Tensor,
    save_dir: Path,
    postprocess_fn: Callable[[torch.Tensor], torch.Tensor],
    model_type: str,
):
    """
    Runs a single forward pass, extracts activations from selected layers,
    post-processes them into (B, D) representations, and saves them to disk.

    Saved format:
        save_dir/
            layer_name.pt   # tensor of shape (B, D)

    Args:
        model: trained CNN or ViT (already on correct device)
        layers: dict {layer_name: module}
        images: input tensor (B, C, H, W), already preprocessed
        save_dir: directory to save activations
        postprocess_fn: function mapping raw activations -> (B, D)
        model_type: "cnn" or "vit" (for logging / sanity checks)
    """

    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    activations = {}
    handles = []


    # Hook registration
    def make_hook(name):
        def hook(_, __, output):
            reps = postprocess_fn(output)

            if reps.ndim != 2:
                raise ValueError(
                    f"[{model_type}] Layer '{name}' produced invalid shape "
                    f"{tuple(reps.shape)} after postprocessing (expected 2D - (B, D))."
                )

            activations[name] = reps.cpu()
        return hook

    for name, layer in layers.items():
        handles.append(layer.register_forward_hook(make_hook(name)))

    # Forward pass (single pass is enough)
    _ = model(images)

    # Cleanup hooks
    for h in handles:
        h.remove()

    # Save activations
    for name, tensor in activations.items():
        path = save_dir / f"{name}.pt"
        torch.save(tensor, path)
        print(
        f"[{model_type}] {name}: "
        f"B={tensor.shape[0]}, D={tensor.shape[1]}"
        )

    # Logging
    print(
        f"[Activation Extraction] Saved {len(activations)} layers "
        f"for {model_type.upper()}: {save_dir}"
    )
