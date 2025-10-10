import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.logging_setup import logger

class ConvBlock(nn.Module):
    '''A Convolutional Block consisting of two convolutional layers each followed by
    Batch Normalization and ReLU activation.
    '''
    def __init__(self, in_channels: int, out_channels: int):
        '''
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        '''
        logger.info(f"Initializing ConvBlock with in_channels={in_channels}, out_channels={out_channels}")
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        logger.info("ConvBlock initialized successfully.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.info(f"ConvBlock input shape: {x.shape}")
        output = self.conv(x)
        logger.info(f"ConvBlock output shape: {output.shape}")
        return output