import torch 
import yaml
import torch.nn.functional as F 
from architectures.vit import Transformer
from utilities.load_data import load_petface_data, plot_breed_distribution
from training.vit import fit, train, val, visualize_predictions  
from utilities.common import OutputManager
import os
import matplotlib.pyplot as plt 

def main(): 
    Current_dir = os.path.dirname(os.path.realpath(__file__))
 
    with open(os.path.join(Current_dir, "config", "vit.yaml")) as f: 
        cfg = yaml.safe_load(f)
      
        
    if cfg["training"]["device"] == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device specified but not available.")
        else:
            print("CUDA device is available. Using GPU for training.")
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
    print(cfg["training"]["batch_size"])      
    ### load data 
    train_ldr, val_ldr, test_ldr, n_breeds = load_petface_data(cfg["training"]["batch_size"],
                                                              os.path.join(Current_dir, "data","cat","cat.csv",),
                                                              strip_percent=cfg["data"]["strip_percent"],
                                                              visualize_breed_distribution=True) 
    
    print(cfg["model"]["embedding_dim"])
    ### create model 
    model = Transformer(images_size=cfg["data"]["images_size"], 
                        batch_size=cfg["training"]["batch_size"], 
                        patches_size=cfg["model"]["patches_size"], 
                        embedding_dim=cfg["model"]["embedding_dim"],
                        num_encoder_blocks=cfg["model"]["num_encoder_blocks"],
                        num_attention_heads=cfg["model"]["num_attention_heads"],
                        factor_hidden_size_encoder=cfg["model"]["factor_hidden_size_encoder"],
                        dropout_en=cfg["model"]["dropout_en"],
                        num_classes=n_breeds)
    model = model.to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters() , lr= cfg["training"]["lr"], weight_decay=cfg["training"]["wd"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg["training"]["factor"],
        patience=cfg["training"]["patience_lr"]
    )
    
    model, train_losses, val_losses = fit(model=model, 
                                          optimizer=optimizer, 
                                          train_loader=train_ldr,
                                          val_loader=val_ldr,
                                          num_epochs=cfg["training"]["num_epochs"],
                                          device=device,
                                          scheduler=scheduler, 
                                          patience_earlystopping=cfg["training"]["patience_earlystopping"])
    
    test_loss, test_acc = val(data_loader=test_ldr, device=device, model=model)
    print(f"Test error: \n accuracy: {(100*test_acc):>0.1f}%, avg loss: {test_loss:>8f} \n")
    
    ### TO DO: INITIALIZE OUTPUT MANAGER and visualize loss, accuracy and tests 
    output_manager = OutputManager(base_dir=os.path.join("data","vit","model"), config=cfg)
    
    # Save loss visualization
    output_manager.save_loss_plot(train_losses, val_losses, test_loss)
    
    # Save metrics
    output_manager.save_metrics(test_acc, test_loss)
    
    # Save model 
    output_manager.dump_model(model)
    model = model.to(device)

    # Save breed distribution plots
    print("\n" + "="*50)
    print("Saving breed distribution plots...")
    print("="*50)
    
    # Get label encoder and data frames from dataloaders
    label_encoder = test_ldr.dataset.le
    train_df = train_ldr.dataset.df
    val_df = val_ldr.dataset.df
    test_df = test_ldr.dataset.df
    
    # Create and save breed distribution plots for each split
    fig_train_breeds = plot_breed_distribution(train_df, label_encoder, split_name="Train")
    output_manager.save_breed_distribution_plot(fig_train_breeds, filename="breed_distribution_train.png")
    plt.close(fig_train_breeds)
    
    fig_val_breeds = plot_breed_distribution(val_df, label_encoder, split_name="Validation")
    output_manager.save_breed_distribution_plot(fig_val_breeds, filename="breed_distribution_val.png")
    plt.close(fig_val_breeds)
    
    fig_test_breeds = plot_breed_distribution(test_df, label_encoder, split_name="Test")
    output_manager.save_breed_distribution_plot(fig_test_breeds, filename="breed_distribution_test.png")
    plt.close(fig_test_breeds)

    # Visualize predictions on test set
    print("\n" + "="*50)
    print("Visualizing test predictions...")
    print("="*50)
    
    # Get label encoder from test loader
    label_encoder = test_ldr.dataset.le
    
    # Get predictions figure and save it
    fig_predictions = visualize_predictions(
        dataloader=test_ldr,
        model=model,
        device=device,
        label_encoder=label_encoder,
        num_samples=12,
        figsize=(15, 10),
        show=True
    )
    
    # Save predictions figure
    output_manager.save_predictions_plot(fig_predictions)
    output_manager.dump_config()
    print("\n" + "="*50)
    print(f"All outputs saved to: {output_manager.get_output_dir()}")
    print("="*50)
    
if __name__ == "__main__":
    main()
    
    

        