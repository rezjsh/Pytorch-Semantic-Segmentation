import torch
import torch.nn as nn
from Semantic_Segmentation.src.config.configuration import ConfigurationManager
from src.entity.config_entity import SimpleFCNConfig
from src.utils.logging_setup import logger
from torchvision import models

class SimpleFCN(nn.Module):
    '''A simple Fully Convolutional Network (FCN) for semantic segmentation based on a pre-trained VGG16 backbone.
    The model replaces the fully connected layers with convolutional layers and upsamples the output to match the input size.
    '''
    def __init__(self, config: SimpleFCNConfig, num_classes: int) -> None:
        '''
        Args:
            num_classes: Number of output classes for segmentation.
        '''
        config_manager = ConfigurationManager()
        self.config: SimpleFCNConfig = config_manager.get_simple_fcn_config()
        self.num_classes = num_classes
        logger.info(f"Initializing SimpleFCN with num_classes={self.num_classes}")
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = vgg.features

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, self.num_classes, kernel_size=1)
        )

        # Upsampling by 32x (5 maxpool layers)
        self.upsample = nn.ConvTranspose2d(
            self.num_classes, self.num_classes,
            kernel_size=64,
            stride=32,
            padding=16,
            bias=False
        )

        logger.info("SimpleFCN initialized successfully.")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.info(f"SimpleFCN input shape: {x.shape}")
        x = self.features(x)
        x = self.classifier(x)
        x = self.upsample(x)
        logger.info(f"SimpleFCN output shape: {x.shape}")
        return x

