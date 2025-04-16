import numpy as np
import torch
import torch.nn as nn
from mmdet.models.builder import BACKBONES

@BACKBONES.register_module()
class RewardConvNet(nn.Module):
    def __init__(self, input_channels: int = 512, conv1_out_channels: int = 256, conv2_out_channels: int = 256):
        """
        Initialize RewardConvNet.

        Args:
            input_channels (int): Number of channels in the input feature map. Default is 512.
            conv1_out_channels (int): Number of output channels for the first convolution. Default is 256.
            conv2_out_channels (int): Number of output channels for the second convolution. Default is 256.
        """
        super(RewardConvNet, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=conv1_out_channels, 
            kernel_size=3, 
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(conv1_out_channels)  # Batch normalization for the first layer
        self.relu1 = nn.ReLU()
        
        # Second convolution
        self.conv2 = nn.Conv2d(
            in_channels=conv1_out_channels, 
            out_channels=conv2_out_channels, 
            kernel_size=3, 
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(conv2_out_channels)  # Batch normalization for the second layer
        self.relu2 = nn.ReLU()
        
        # Adaptive average pooling to reduce spatial dimensions to 1x1
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            x (torch.Tensor): Input feature map, shape [batch_size*num_traj, 512, 4, 8]=[batch_size*num_traj, C1 + C2, H, W]

        Returns:
            torch.Tensor: Scoring features, shape [batch_size*num_traj, 128, 1, 1]
        """
        
        x = self.conv1(x)         # [batch_size*num_traj, 256, 4, 8]
        x = self.bn1(x)           # Batch normalization
        x = self.relu1(x)         # ReLU activation
        x = self.conv2(x)         # [batch_size*num_traj, 128, 4, 8]
        x = self.bn2(x)           # Batch normalization
        x = self.relu2(x)         # ReLU activation
        x = self.pool(x)          # [batch_size*num_traj, 128, 1, 1]
        return x