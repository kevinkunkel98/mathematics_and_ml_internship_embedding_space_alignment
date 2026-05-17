import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from training.early_stopping import EarlyStopping

def train(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:

    model.train()
    size = len(dataloader.dataset)

    train_loss, correct = 0, 0
    # train loss is calcualted as sum over all samples divided by number of samples for epoch
    
    for X, y, _ in dataloader:
        X, y = X.to(device), y.to(device)
        #print(f"input shape: {X.shape}, target shape: {y.shape}")
        """input shape: torch.Size([64, 3, 224, 224]), target shape: torch.Size([64])"""
        #Compute prediction error
        pred = model(X)
        #print(f"prediction shape: {pred.shape}, target shape: {y.shape}")
        """
        prediction shape: torch.Size([3136, 10]), target shape: torch.Size([64])
        """

        #print(f"prediction sample: {pred[0]}, target sample: {y[0]}")
        loss = loss_fn(pred, y)

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #accumulate loss and accuracy
        train_loss += loss.item() * len(y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()


    train_loss /= size
    correct /= size
    return train_loss, correct


def test(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    loss_fn: nn.Module,
) -> Tuple[float, float]:

    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y, _ in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()*len(y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    return test_loss, correct


def fit(train_ldr, val_ldr, model, loss_fn, optimizer, scheduler, patience_earlystopping, n_epochs, device, checkpoint_manager=None,):
    train_losses = []
    val_losses = []

    # initialize early stoppping
    early_stopping = EarlyStopping(patience=patience_earlystopping)

    for t in range(1, n_epochs+1):
        train_loss, train_acc = train(model=model, dataloader=train_ldr, loss_fn=loss_fn, optimizer=optimizer, device=device)
        val_loss, val_acc = test(dataloader=val_ldr, model=model, device=device, loss_fn=loss_fn)
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        info(t, n_epochs, train_loss, val_loss, train_acc, val_acc)
        
        """early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Terminate by early stopping.')
            break"""
        
        if checkpoint_manager is not None:
            checkpoint_manager.save(
                name=f"epoch_{t:03d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=t,
                val_loss=val_loss,
            )

            checkpoint_manager.maybe_save_best(
                model=model,
                optimizer=optimizer,
                epoch=t,
                val_loss=val_loss,
            )

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Terminate by early stopping.")
            break

    return model, train_losses, val_losses

# prints info about progress
def info(t, n_epochs, train_loss, val_loss, train_acc, val_acc):
    digits = len(str(n_epochs))
    msg = (f'[{t:>{digits}}/{n_epochs:>{digits}}] ' +
         f'train loss: {train_loss:.5f}     ' +
         f'val loss: {val_loss:.5f}      ' +
         f'train acc: {train_acc:.5f}     ' +
         f'val acc: {val_acc:.5f}')
    print(msg)


def visualize_predictions(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    label_encoder,
    num_samples: int = 12,
    figsize: Tuple[int, int] = (15, 10),
    show: bool = True
) -> plt.Figure:
    """
    Visualize test predictions with actual and predicted labels and image paths.
    
    Args:
        dataloader: DataLoader with test data
        model: Trained model
        device: torch device (cpu or cuda)
        label_encoder: LabelEncoder to convert class indices to breed names
        num_samples: Number of samples to visualize (default: 12)
        figsize: Figure size for matplotlib
        show: Whether to display the plot (default: True)
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    model.eval()
    
    samples_shown = 0
    predictions = []
    actuals = []
    images = []
    image_paths = []
    folder_paths = []
    with torch.no_grad():
        for X, y, paths in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred_classes = pred.argmax(1)
            
            # Collect data
            for img, actual, pred_class, path in zip(X, y, pred_classes, paths):
                if samples_shown >= num_samples:
                    break
                
                # Denormalize image
                img_np = img.cpu().numpy()
                # Reverse normalization: x = (x * std) + mean
                img_np = (img_np * 0.5) + 0.5  # Reverse the normalize from load_data.py
                img_np = np.transpose(img_np, (1, 2, 0))
                img_np = np.clip(img_np, 0, 1)
                
                images.append(img_np)
                actuals.append(actual.item())
                predictions.append(pred_class.item())
                image_paths.append(path)
                samples_shown += 1
            
            if samples_shown >= num_samples:
                break
    
    # Create grid for visualization
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    # Decode labels
    actual_breeds = label_encoder.inverse_transform(actuals)
    predicted_breeds = label_encoder.inverse_transform(predictions)
    
    for idx, (ax, img, actual_breed, pred_breed, img_path) in enumerate(
        zip(axes, images, actual_breeds, predicted_breeds, image_paths)
    ):
        ax.imshow(img)
        
        # Color: green if correct, red if wrong
        is_correct = actual_breed == pred_breed
        color = 'green' if is_correct else 'red'
        
        # Extract filename from path
        filename = img_path.split('/')[-1] if '/' in img_path else img_path
        
        title = f"True: {actual_breed}\nPred: {pred_breed}\nPath: {img_path}"
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig