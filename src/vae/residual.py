import torch
from torch import nn
from torch.nn import functional as F

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else: 
            self.residual_layer = nn.Conv2d(in_channels, out_channels, 1, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.groupnorm_1(x)
        x = self.conv1(F.silu(x))

        x = self.groupnorm_2(x)
        x = self.conv2(F.silu(x))

        return x + self.residual_layer(residual)
