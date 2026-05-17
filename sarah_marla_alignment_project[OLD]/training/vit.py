import torch 
import yaml
import torch.nn.functional as F 
from architectures.vit import Transformer
from training.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from training.checkpoints import CheckpointManager
import matplotlib.pyplot as plt 
from typing import Tuple
import numpy as np 
import os 



# prints info about progress
def info(t, n_epochs, train_loss, val_loss, train_acc, val_acc):
    digits = len(str(n_epochs))
    msg = (f'[{t:>{digits}}/{n_epochs:>{digits}}] ' +
         f'train loss: {train_loss:.5f}     ' +
         f'val loss: {val_loss:.5f}      ' +
         f'train acc: {train_acc:.5f}     ' +
         f'val acc: {val_acc:.5f}')
    print(msg)


### Cross Entropy with one correct class  
# predict_labels: (BS,NUM_CLASSES)
# labels: (BS)
def loss_func(labels, predict_labels): 
    loss = F.cross_entropy(predict_labels, labels)
    return loss 


### TO DO TEST FUNCTION 
def test(model, images, labels): 
    predict_labels = model(images)
    loss = loss_func(labels, predict_labels)
    return loss.item(), predict_labels  

def val(data_loader: DataLoader, device: torch.device, model: torch.nn.Module): 
    model.eval()
    len_dataset = len(data_loader.dataset)
    val_loss, val_acc = 0.0, 0.0 
    with torch.no_grad(): 
        for images, labels, _ in data_loader: 
                images = images.to(device)
                labels = labels.to(device) 
                
                loss, pred = test(model, images, labels)
                val_loss += loss * len(labels)
                val_acc += (pred.argmax(1) == labels).type(torch.float).sum().item()
    
    val_loss /= len_dataset
    val_acc /= len_dataset      
        
    return val_loss, val_acc 

def train_step(model, optimizer, images, labels): 
    #print("Made it into training step.")
    optimizer.zero_grad()
    predict_labels = model(images)
    #print("Predicting also kind of works.")
    loss = loss_func(labels, predict_labels)
    #print("Calculating the loss too.")
    loss.backward() 
    optimizer.step()
    #print("And now we did a step.") 
    return loss.item(), predict_labels

def train(train_loader: DataLoader, device: torch.device,
          model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    
    model.train()
    len_dataset = len(train_loader.dataset)
    train_loss, train_acc = 0.0, 0.0
    i = 0 
    for images, labels, _ in train_loader:
        #print(f"Made it to batch: {i}") 
        #print(f"Device is: {device}")
        images = images.to(device)
        labels = labels.to(device)
        #print("Put images and label on device.")
            
        loss, pred = train_step(model, optimizer, images, labels)
        train_loss += loss * len(labels)
        
        train_acc += (pred.argmax(1) == labels).type(torch.float).sum().item()
        i += 1 
        
    train_loss /= len_dataset
    train_acc /= len_dataset
    return train_loss, train_acc  

def fit(model:Transformer, optimizer, train_loader, val_loader, 
          num_epochs, device, scheduler, patience_earlystopping, checkpoint_manager=None):
    
    train_losses = []
    val_losses = []
    
    early_stopping = EarlyStopping(patience=patience_earlystopping)#, path=os.path.join("models", "vit", "model.pt"))
    
    for t in range(1,num_epochs+1): 
        
        train_loss, train_acc = train(train_loader=train_loader, device=device, model=model, optimizer=optimizer)
        val_loss, val_acc = val(data_loader=val_loader, model=model, device=device)
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        info(t, num_epochs, train_loss, val_loss, train_acc, val_acc)
        #print(checkpoint_manager)
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
        ### TODO: Look at early stopping and if it works for Transformer model 
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print('Terminate by early stopping.')
            break

    return model, train_losses, val_losses



def visualize_predictions(
    dataloader: DataLoader,
    model: torch.nn.Module,
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
        
        
            
    
        
