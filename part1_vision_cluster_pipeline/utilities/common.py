# Standard library imports
from typing import Tuple, List, Dict, Any, Optional, Union
from datetime import datetime
import os
import yaml
# Third-party imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

# Data processing
import numpy as np
import pandas as pd

class EarlyStoppingDeprecated:
    def __init__(self, patience=5, path='model.pt'):
        '''
        Parameters:
        patience (int): maximum number of epochs without improvement of the validation loss
            Default: 5
        path (str): path of the model to be saved to. 
            Default: 'model.pt'
        '''
        self.path = path
        self.patience = patience
        self.counter = 0
        self.min_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.patience is None: return
        if val_loss < self.min_loss: 
            self.save_model(val_loss, model)
            self.min_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            self.early_stop = self.patience < self.counter
      
    def save_model(self, val_loss, model):
        if self.patience is None: return
        torch.save(model.state_dict(), self.path)
        print(f'\t Validation loss decreased: {self.min_loss:>.6f} to {val_loss:>.6f}. Model saved')

    def get_model(self, model):
        if self.patience is None: return model
        model.load_state_dict(torch.load(self.path, weights_only=True))
        return model


class OutputManager:
    """
    Manages saving of visualizations and model outputs with timestamps.
    Creates timestamped directories for organizing outputs.
    """
    
    def __init__(self, base_dir: os.path = os.path.join("data","cnn","model"), config: dict = None):
        """
        Args:
            base_dir (path): Base directory for saving outputs (default: "data/cnn/model")
        """
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(base_dir, self.timestamp)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"OutputManager initialized: {self.output_dir}")

        self.config = config
        if self.config is not None:
            self.dump_config()

    def dump_config(self, filename: str = "config.yaml") -> str:

        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            yaml.safe_dump(self.config, f, sort_keys=False)
    
    def save_loss_plot(self, train_losses: List[float], val_losses: List[float], 
                      test_loss: float, filename: str = "loss_plot.png") -> str:
        """
        Save training/validation/test loss visualization.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            test_loss: Final test loss
            filename: Output filename (default: "loss_plot.png")
            
        Returns:
            str: Path to saved figure
        """
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(10, 6))
        t = range(1, len(train_losses) + 1)
        plt.plot(t, train_losses, label='train loss')
        plt.plot(t, val_losses, label='val loss')
        plt.plot(t, test_loss * np.ones(len(t)), label='test loss', linestyle='--')
        
        # Early stopping checkpoint
        minpos_val = np.argmin(val_losses) + 1
        plt.axvline(minpos_val, linestyle='-.', color='r', label='early stopping')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training, Validation and Test Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss plot saved: {output_path}")
        return output_path
    
    def save_predictions_plot(self, fig_obj, filename: str = "predictions.png") -> str:
        """
        Save predictions visualization (cat breed predictions).
        
        Args:
            fig_obj: Matplotlib figure object
            filename: Output filename (default: "predictions.png")
            
        Returns:
            str: Path to saved figure
        """
        output_path = os.path.join(self.output_dir, filename)
        fig_obj.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved: {output_path}")
        return output_path
    
    def save_breed_distribution_plot(self, fig_obj, filename: str = "breed_distribution.png") -> str:
        """
        Save breed distribution visualization.
        
        Args:
            fig_obj: Matplotlib figure object
            filename: Output filename (default: "breed_distribution.png")
            
        Returns:
            str: Path to saved figure
        """
        output_path = os.path.join(self.output_dir, filename)
        fig_obj.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Breed distribution plot saved: {output_path}")
        return output_path
    
    def save_metrics(self, accuracy: float, test_loss: float,
                    filename: str = "metrics.txt") -> str:
        """
        Save model metrics to text file.
        
        Args:
            accuracy: Test accuracy
            test_loss: Test loss
            filename: Output filename (default: "metrics.txt")
            
        Returns:
            str: Path to saved metrics file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w') as f:
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write(f"Test Loss: {test_loss:.6f}\n")
        
        print(f"Metrics saved: {output_path}")
        return output_path
    
    def get_output_dir(self) -> str:
        """Get the output directory path."""
        return self.output_dir
    
    def get_timestamp(self) -> str:
        """Get the timestamp."""
        return self.timestamp