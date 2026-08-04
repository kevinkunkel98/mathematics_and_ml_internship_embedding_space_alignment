import torch
import torch.nn as nn
import yaml
import os
from training.cnn import fit
from training.checkpoints import CheckpointManager
from utilities.load_data import load_petface_data
from architectures.factory import build_model
from utilities.output_manager import OutputManager

def main():
    # 1. Load config
    Current_dir = os.path.dirname(os.path.realpath(__file__))
    
    with open(os.path.join(Current_dir, "config", "cnn.yaml")) as f: 
        config = yaml.safe_load(f)

    if config["training"]["device"] == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device specified but not available.")
        else:
            print("CUDA device is available. Using GPU for training.")
            
    device = torch.device(config["training"]["device"])
    print("Using device:", device)
    torch.manual_seed(config["training"]["seed"])
   
    # 2. Load data
    train_ldr, val_ldr, _, n_breeds = load_petface_data(
        batch_size=config["training"]["batch_size"],
        label_path="./data/cat/cat/cat.csv", ### path different local and on cluster (this is cluster path)
        strip_percent=config["data"]["strip_percent"],
        visualize_breed_distribution=False,
    )
   
    output_manager = OutputManager(
        experiment_name="cnn_petfaces",
        config=config
    )


    checkpoint_manager = CheckpointManager(
        checkpoint_dir=output_manager.get_checkpoint_dir()
    )

    # 3. Build model
    model = build_model(config, num_classes=n_breeds).to(device)

    # 4. Optimizer / loss

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["wd"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"]["factor"],
        patience=config["training"]["patience_lr"],
    )
    loss_fn = nn.CrossEntropyLoss()

    # 5. Training

    fit(
        train_ldr=train_ldr,
        val_ldr=val_ldr,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=None,
        patience_earlystopping=config["training"]["patience_earlystopping"],
        n_epochs=config["training"]["epochs"],
        device=device,
        checkpoint_manager=checkpoint_manager,
    )


if __name__ == "__main__":
    main()
