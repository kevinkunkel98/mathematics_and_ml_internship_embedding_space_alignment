import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import yaml
from utilities.load_data import load_petface_data, plot_breed_distribution
from architectures.cnn import CNNModel
from training.cnn import train, test, fit, visualize_predictions
from architectures.cnn import create_resnet
from utilities.common import OutputManager
import os 

#from vit_saliency import load_vit_config, oneImage, saliency, visualizeMap

def main_saliency(model, dataloader, device, num_images=1):
    model.eval()
    
    # Get one image from dataloader
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    image = images[0:num_images].to(device)
    
    saliency_val = saliency(model, image)    
    
    # Move image and saliency map to CPU and convert to numpy
    image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
    saliency_np = saliency_val.cpu().squeeze().numpy()
    
    visualizeMap(image_np, saliency_np)   

    return None

def main():
    with open("config/cnn.yaml", "r") as f:
        config = yaml.safe_load(f)

    requested_device = config["training"]["device"]
    if isinstance(requested_device, str) and requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested in config but no CUDA device is available. Falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    # load data
    train_ldr, val_ldr, test_ldr, n_breeds = load_petface_data(
        batch_size=config["training"]["batch_size"],
        label_path="./data/cat/cat.csv",
        strip_percent=config["data"]["strip_percent"],
        visualize_breed_distribution=True,
    )
    
    """Input: RGB-Bilder (224×224 Pixel nach Preprocessing)"""
    print("Using {} device".format(device))

    #fit and evaluate
    if config["model"]["architecture"] == "resnet18":
        model = create_resnet(num_classes=n_breeds).to(device)
    elif config["model"]["architecture"] == "custom_cnn":
        model = CNNModel(input_channels=config["data"]["input_channels"], num_classes=n_breeds).to(device)
    else:
        raise ValueError(f"Unknown architecture: {config['model']['architecture']}")
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["wd"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config["training"]["factor"],
        patience=config["training"]["patience_lr"]
    )
    model, train_losses, val_losses = fit(train_ldr=train_ldr, val_ldr=val_ldr, model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, patience_earlystopping=config["training"]["patience_earlystopping"], n_epochs=config["training"]["epochs"], device=device)
    test_loss, accuracy = test(test_ldr, model, device, loss_fn)
    print(f"Test error: \n accuracy: {(100*accuracy):>0.1f}%, avg loss: {test_loss:>8f} \n")

    # Initialize OutputManager for saving visualizations
    output_manager = OutputManager(base_dir= os.path.join("data","cnn","model"), config=config)
    
    # Save loss visualization
    output_manager.save_loss_plot(train_losses, val_losses, test_loss)
    
    # Save metrics
    output_manager.save_metrics(accuracy, test_loss)
    
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
    label_encoder_saliency = test_ldr.dataset.le
    saliency_fig = visualize_predictions(
        dataloader=test_ldr,
        model=model,
        device=device,
        label_encoder=label_encoder_saliency,
        num_samples=12,
        figsize=(15, 10),
        show=True
    )

    output_manager.save_predictions_plot(saliency_fig, filename="saliency_predictions.png")


    # Save predictions figure
    output_manager.save_predictions_plot(fig_predictions)
    output_manager.dump_config()
    print("\n" + "="*50)
    print(f"All outputs saved to: {output_manager.get_output_dir()}")
    print("="*50)

if __name__ == "__main__":
    main()


# TODO für unterschiedliche farben, representational alignment vergelich zwischen modell auswerten