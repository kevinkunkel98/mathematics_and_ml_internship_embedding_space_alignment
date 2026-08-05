import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Callable
from collections import defaultdict
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


@torch.no_grad()
def extract_activations_to_h5(
    model: torch.nn.Module,
    out_path: Path,
    postprocess_fn: Callable[[torch.Tensor, dict], torch.Tensor],
    model_type: str,
    test_loader: DataLoader,
    device,
    max_batches: int | None = None,
    label_key: str = "labels",
):
    """
    Run forward passes over the test set, extract per-layer (B, D) representations,
    and save a single .h5 file in the format the dashboard expects
    (see scripts/io.py::load_embeddings):

        layer_00, layer_01, ...   float16  (n_samples, D)
        labels                    int8     (n_samples,)

    Unlike extract_and_save_activations (single random batch, no labels), this
    iterates the whole loader and keeps labels, so the dashboard has enough
    points to visualise and can colour them by class.

    Args:
        out_path:     path to the .h5 file to write (e.g. data/vision/dinov2.h5)
        max_batches:  optionally cap the number of batches (None = full loader)
        label_key:    key in each batch holding the class labels
    """
    model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_layer: Dict[int, list] = defaultdict(list)
    labels_all: list = []

    for b, batch in enumerate(test_loader):
        if max_batches is not None and b >= max_batches:
            break

        if label_key in batch:
            labels_all.append(batch[label_key].detach().cpu())

        inputs = {k: v.to(device) for k, v in batch.items() if k != label_key}

        outputs = model.backbone(**inputs, output_hidden_states=True)

        for i, h in enumerate(outputs.hidden_states):     # h: (B, seq, hidden)
            reps = postprocess_fn(h, inputs)              # -> (B, D)
            if reps.ndim != 2:
                raise ValueError(
                    f"[{model_type}] layer {i}: shape {tuple(reps.shape)} (expected 2D)"
                )
            per_layer[i].append(reps.detach().cpu())

    if not per_layer:
        raise RuntimeError("No batches processed — is test_loader empty?")

    # Concatenate across batches -> (n_samples, D) per layer
    layers = {i: torch.cat(chunks, dim=0).float().numpy() for i, chunks in per_layer.items()}
    n_samples = next(iter(layers.values())).shape[0]

    if labels_all:
        labels = torch.cat(labels_all, dim=0).numpy()
        # Dashboard expects one integer class per sample (1D). COCO batches carry
        # multi-hot vectors (n_samples, n_classes); reduce to a single "primary"
        # class = first/most prominent present category (argmax of the multi-hot).
        if labels.ndim == 2:
            print(
                f"[{model_type}] labels are multi-hot {labels.shape} -> "
                f"reducing to primary class via argmax (1D)."
            )
            labels = labels.argmax(axis=1)
        labels = labels.astype(np.int8)
    else:
        # Dashboard requires a labels dataset; fall back to zeros if none provided.
        print(f"[{model_type}] WARNING: no '{label_key}' in batches — writing zeros.")
        labels = np.zeros(n_samples, dtype=np.int8)

    # Write in the exact format scripts/io.py::save_embeddings uses.
    with h5py.File(out_path, "w") as f:
        for layer_idx, arr in layers.items():
            f.create_dataset(f"layer_{layer_idx:02d}", data=arr.astype(np.float16))
        f.create_dataset("labels", data=labels)

    print(
        f"[Activation Extraction] {model_type.upper()}: "
        f"{len(layers)} layers x {n_samples} samples -> {out_path}"
    )
