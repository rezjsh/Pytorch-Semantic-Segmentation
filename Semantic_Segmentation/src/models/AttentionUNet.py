import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config.configuration import ConfigurationManager
from src.entity.config_entity import AttentionUNetConfig
from src.utils.logging_setup import logger
from src.models.AttentionGate import AttentionGate
from src.models.ConvBlock import ConvBlock


class AttentionUNet(nn.Module):
    '''Attention U-Net architecture for semantic segmentation.
    This model incorporates attention gates in the skip connections to focus on relevant features.
    '''
    def __init__(
        self,
        num_classes: int
    ):
        '''
        Args:
            config: Configuration object containing model parameters.
            num_classes: Number of output segmentation classes.
        '''
        config_manager = ConfigurationManager()
        self.config: AttentionUNetConfig = config_manager.get_attention_unet_config()

        self.num_classes = num_classes
        logger.info(f"Initializing AttentionUNet with in_channels={self.config.in_channels}, num_classes={self.num_classes}, base_channels={self.config.base_channels}, upsample_size={self.config.upsample_size}")
        super().__init__()
        self.upsample_size = self.config.upsample_size

        # Encoder blocks
        self.e1 = ConvBlock(self.config.in_channels, self.config.base_channels)           # 64
        self.e2 = ConvBlock(self.config.base_channels, self.config.base_channels * 2)     # 128
        self.e3 = ConvBlock(self.config.base_channels * 2, self.config.base_channels * 4) # 256

        # Bottleneck block
        self.b = ConvBlock(self.config.base_channels * 4, self.config.base_channels * 8)  # 512

        # Decoder blocks with ConvTranspose for upsampling
        self.up_c3 = nn.ConvTranspose2d(self.config.base_channels * 8, self.config.base_channels * 4, kernel_size=2, stride=2)
        self.d3 = ConvBlock(self.config.base_channels * 4 + self.config.base_channels * 4, self.config.base_channels * 4)

        self.up_c2 = nn.ConvTranspose2d(self.config.base_channels * 4, self.config.base_channels * 2, kernel_size=2, stride=2)
        self.d2 = ConvBlock(self.config.base_channels * 2 + self.config.base_channels * 2, self.config.base_channels * 2)

        # Attention gates
        self.att3 = AttentionGate(F_g=self.config.base_channels * 4, F_l=self.config.base_channels * 4, F_int=self.config.base_channels * 2)
        self.att2 = AttentionGate(F_g=self.config.base_channels * 2, F_l=self.config.base_channels * 2, F_int=self.config.base_channels)

        # Output convolution
        self.out_conv = nn.Conv2d(self.config.base_channels * 2, self.num_classes, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

        logger.info("AttentionUNet initialized successfully.")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.info(f"AttentionUNet input shape: {x.shape}")
        # Encoder pathway
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))

        # Bottleneck
        b = self.b(self.pool(x3))

        # Decoder pathway with attention gates
        d3_up = self.up_c3(b)
        x3_att = self.att3(x=x3, g=d3_up)
        d3_cat = torch.cat([d3_up, x3_att], dim=1)
        d3 = self.d3(d3_cat)

        d2_up = self.up_c2(d3)
        x2_att = self.att2(x=x2, g=d2_up)
        d2_cat = torch.cat([d2_up, x2_att], dim=1)
        d2 = self.d2(d2_cat)

        # Output logits
        out = self.out_conv(d2)

        # Upsample output logits to target size (default 256x512)
        if self.config.upsample_size is not None:
            out = F.interpolate(out, size=self.config.upsample_size, mode='bilinear', align_corners=False)
        logger.info(f"AttentionUNet output shape: {out.shape}")
        return out
