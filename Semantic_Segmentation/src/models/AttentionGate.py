import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.logging_setup import logger

class AttentionGate(nn.Module):
    '''Attention Gate as described in "Attention U-Net: Learning Where to Look for the Pancreas"
    '''
    def __init__(self, F_g: int, F_l: int, F_int: int):
        '''
        Args:
            F_g: Number of gating features
            F_l: Number of input features
            F_int: Number of intermediate features
        '''
        logger.info(f"Initializing AttentionGate with F_g={F_g}, F_l={F_l}, F_int={F_int}")
        super().__init__()
        # Transform the gating and input feature maps to the intermediate channels
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Psi to generate attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

        logger.info("AttentionGate initialized successfully.")

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        logger.info(f"AttentionGate input shape: x={x.shape}, g={g.shape}")
        g1 = self.W_g(g)  # Decoder features
        x1 = self.W_x(x)  # Encoder features
        # Upsample gating signal to encoder feature size for additive attention
        g1_upsampled = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1_upsampled + x1)
        psi = self.psi(psi)  # Attention coefficients
        logger.info(f"AttentionGate output shape: {x.shape}, attention mask shape: {psi.shape}")
        return x * psi  # Apply attention mask

