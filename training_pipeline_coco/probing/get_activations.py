import torch
from pathlib import Path
from typing import Dict, Callable
import random
from torch.utils.data import Dataset, DataLoader

def postprocess_vision(h, inputs):
    return h[:, 0, :]                              # CLS-Token

def postprocess_language(h, inputs):
    last_idx = inputs["attention_mask"].sum(1) - 1
    return h[torch.arange(h.size(0)), last_idx]    # letztes echtes Token

### get random batch 
def random_batch_from_loader(loader, device, drop_keys=("labels",)):
    n = random.randint(0, len(loader) - 1)             
    for i, batch in enumerate(loader):
        if i == n:
            return {k: v.to(device) for k, v in batch.items() if k not in drop_keys}, n
        
### get layer names 
def find_layers(model, keyword: str) -> Dict[str, torch.nn.Module]:
    return {name: module
            for name, module in model.named_modules()
            if keyword in name}

@torch.no_grad()
def extract_and_save_activations(
    model: torch.nn.Module,
    save_dir: Path,
    postprocess_fn: Callable[[torch.Tensor], torch.Tensor],
    model_type: str,
    test_loader: DataLoader, 
    device
):
    """
    Runs a single forward pass, extracts activations from selected layers,
    post-processes them into (B, D) representations, and saves them to disk.

    Saved format:
        save_dir/
            layer_name.pt   # tensor of shape (B, D)

    Args:
        model: trained CNN or ViT (already on correct device)
        save_dir: directory to save activations
        postprocess_fn: function mapping raw activations -> (B, D)
        model_type: "cnn" or "vit" (for logging / sanity checks)
    """

        
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
        
    ### get input batch for activation s
    inputs, batch_idx = random_batch_from_loader(test_loader, device)

    # Forward pass (single pass is enough)
    outputs = model.backbone(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    
    for i, h in enumerate(hidden_states):          # h: (B, seq, hidden)
        reps = postprocess_fn(h, inputs)           # -> (B, D)
        if reps.ndim != 2:
            raise ValueError(f"[{model_type}] layer {i}: shape {tuple(reps.shape)} (expected 2D)")
        reps = reps.detach().cpu()
        torch.save(reps, save_dir / f"layer_{i}.pt")
        print(f"[{model_type}] layer_{i}: B={reps.shape[0]}, D={reps.shape[1]}")

    # Logging
    print(
        f"[Activation Extraction] Saved layers "
        f"for {model_type.upper()}: {save_dir}"
    )
