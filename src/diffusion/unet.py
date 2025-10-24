import torch
from torch import nn
from torch.nn import functional as F

from blocks import *
from switch_sequential import SwitchSequential

class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            # (batch_size, out_channels: 4, height / 8, width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=0)),
            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8, 40)),

            # (batch_size, out_channels: 320, height / 16, width / 16)            
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320,640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640,640), UNET_AttentionBlock(8, 80)),
            
            # (batch_size, out_channels: 640, height / 32, width / 32)            
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640,1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280,1280), UNET_AttentionBlock(8, 160)),
            
            # (batch_size, out_channels: 1280, height / 64, width / 64)            
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280,1280)),
            SwitchSequential(UNET_ResidualBlock(1280,1280)),
    
        ])

        self.bottleneck = nn.ModuleList([
            SwitchSequential([
                UNET_ResidualBlock(1280,1280), 
                UNET_AttentionBlock(8, 160),
                UNET_ResidualBlock(1280,1280), 
            ])
        ])

        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560,1280)),
            SwitchSequential(UNET_ResidualBlock(2560,1280)),
            SwitchSequential(UNET_ResidualBlock(2560,1280), UpSample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560,1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560,1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920,1280), UNET_AttentionBlock(8, 160), UpSample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920,640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280,640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960,640), UNET_AttentionBlock(8, 160), UpSample(640)),

            SwitchSequential(UNET_ResidualBlock(960,320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640,320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640,320), UNET_AttentionBlock(8, 40))
        ])

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 4, num_groups: int = 32, kernel_size: int = 3):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, H/8, W/8)
        return: (B, out_channels, H/8, W/8)
        """
        return self.conv(self.act(self.norm(x)))