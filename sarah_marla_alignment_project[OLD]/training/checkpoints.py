import torch
import os
from datetime import datetime
class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float("inf")
        
        print(f"[CheckpointManager] Saving checkpoints to: {os.path.abspath(self.checkpoint_dir)}")


    def save_epoch(self, model, optimizer, epoch):
        path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, path)

    def save_best(self, model, optimizer, epoch, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "val_loss": val_loss,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, path)

    def save_last(self, model, optimizer, epoch):
        path = os.path.join(self.checkpoint_dir, "last.pt")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, path)
    
    def save(self, name: str, model, optimizer, epoch, val_loss=None):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss
        }, path)

    def maybe_save_best(self, model, optimizer, epoch, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save("best.pt", model, optimizer, epoch, val_loss)

def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
):
    """
    Load model weights from a checkpoint into an existing model instance.

    Args:
        model: Initialized model (CNN, ViT, etc.)
        checkpoint_path: Path to .pt checkpoint
        device: torch.device
        strict: Whether to enforce exact key matching

    Returns:
        model: Model with loaded weights
        metadata: dict with epoch, val_loss (if present)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both raw state_dict and full checkpoint dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
        }
    else:
        state_dict = checkpoint
        metadata = {}

    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()

    return model, metadata
