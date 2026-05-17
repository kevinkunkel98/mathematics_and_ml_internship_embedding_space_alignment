import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from utilities.common import *

def create_resnet(num_classes):
    """
    Create a ResNet model for image classification.

    Args:
        num_classes (int): Number of output classes for classification.
        pretrained (bool): Whether to use a pretrained model.
    Returns:
        model (nn.Module): ResNet model.
    """
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    for param in model.parameters(): 
        param.requires_grad = False 
        
    for param in model.fc.parameters(): 
        param.requires_grad = True
    
    return model

class CNNModel(nn.Module):
    """
    Optimized CNN architecture for 224x224 RGB image classification.
    Designed for cat breed classification on PetFace dataset.
    """
    
    def __init__(self, input_channels: int = 3, num_classes: int = 10) -> None:
        """
        Args:
            input_channels (int): Number of input channels (3 for RGB images)
            num_classes (int): Number of output classes for classification
        """
        super(CNNModel, self).__init__()
        
        # Convolutional Block 1: 224x224 -> 112x112
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2: 112x112 -> 56x56
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 3: 56x56 -> 28x28
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 4: 28x28 -> 14x14
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling: 14x14x512 -> 512
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Block 1: 224x224 -> 112x112
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2: 112x112 -> 56x56
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3: 56x56 -> 28x28
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4: 28x28 -> 14x14
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Global Average Pooling: 14x14x512 -> 1x1x512
        x = self.global_avg_pool(x)
        
        # Flatten: 1x1x512 -> 512
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers with Batch Norm and Dropout
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        
        # Output Layer (no activation - CrossEntropyLoss applies softmax)
        x = self.fc3(x)

        # TODO: last should be linear layer without softmax for CrossEntropyLoss
        # zahl ohne wahrscheinlichkeit, ohne aktivierungsfunktion
        
        return x



