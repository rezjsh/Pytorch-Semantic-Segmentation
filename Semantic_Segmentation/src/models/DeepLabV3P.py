import torch
import torch.nn as nn
from Semantic_Segmentation.src.config.configuration import ConfigurationManager
from src.entity.config_entity import DeepLabV3PConfig
from src.utils.logging_setup import logger
from torchvision import models


class DeepLabV3P(nn.Module):
    '''DeepLabV3+ architecture for semantic segmentation.'''
    def __init__(self, config: DeepLabV3PConfig, num_classes: int) -> None:
        super().__init__()
        logger.info(f"Initializing DeepLabV3P with num_classes={self.num_classes}")
        config_manager = ConfigurationManager()
        self.config: DeepLabV3PConfig = config_manager.get_deeplabv3p_config()
        self.num_classes = num_classes
        self.model = models.segmentation.deeplabv3_resnet101(
            weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT,
            progress=True
        )
        # Replace final classifier conv layer for required number of classes
        self.model.classifier[-1] = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1)
        logger.info("DeepLabV3P initialized successfully.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.info(f"DeepLabV3P input shape: {x.shape}")
        x = self.model(x)['out']
        logger.info(f"DeepLabV3P output shape: {x.shape}")
        return x