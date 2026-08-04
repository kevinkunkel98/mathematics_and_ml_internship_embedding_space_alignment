import torch 
import yaml
import os

from architectures.vit import Transformer
from utilities.load_data import load_petface_data
from training.vit import fit
from training.checkpoints import CheckpointManager
from architectures.factory import build_model
from utilities.output_manager import OutputManager
def main(): 
    Current_dir = os.path.dirname(os.path.realpath(__file__))
        
    with open(os.path.join(Current_dir, "config", "vit_train.yaml")) as f: 
        train_cfg = yaml.safe_load(f)
        
    if train_cfg["training"]["device"] == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device specified but not available.")
        else:
            print("CUDA device is available. Using GPU for training.")
            
    device = torch.device(train_cfg["training"]["device"])
    print("Using device:", device)
    torch.manual_seed(train_cfg["training"]["seed"])

    # 2. Load data
    train_ldr, val_ldr, test_ldr, n_breeds = load_petface_data(
        train_cfg["training"]["batch_size"],
        os.path.join(Current_dir, "data","cat", "cat.csv",),
        strip_percent=train_cfg["data"]["strip_percent"],
        visualize_breed_distribution=False
    ) 

    #base_dir=os.path.join("checkpoints","vit","model")
    #checkpoint_manager = CheckpointManager(base_dir)

    output_manager = OutputManager(
        experiment_name="vit_petfaces",
        config=train_cfg
    )


    checkpoint_manager = CheckpointManager(
        checkpoint_dir=output_manager.get_checkpoint_dir()
    )

    # 3. Build model
    model = build_model(train_cfg, num_classes=n_breeds).to(device)
    
    """
    if not isinstance(model, Transformer):
        print(f"False model used: {type(model)}")
        return -1
    """
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad , model.parameters()) , 
                                 lr= train_cfg["training"]["lr"], 
                                 weight_decay=train_cfg["training"]["wd"])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=train_cfg["training"]["factor"],
        patience=train_cfg["training"]["patience_lr"]
    )
    
    
    fit(
        model=model, 
        optimizer=optimizer, 
        train_loader=train_ldr, 
        val_loader=val_ldr, 
        num_epochs=train_cfg["training"]["num_epochs"], 
        device=device,
        scheduler=scheduler, 
        patience_earlystopping=train_cfg["training"]["patience_earlystopping"],
        checkpoint_manager=checkpoint_manager,
    )
    
    
if __name__ == "__main__":
    main()
        