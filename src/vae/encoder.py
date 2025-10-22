import torch
from torch import nn
from torch.nn import functional as F

from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential): # inheritence sequential class
    def __init__(self): # constructor 
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),
            nn.SiLU(),

            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (batch_size, out_channels: 8, height / 8, width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x : input image (batch_size, channel, height, width)
        # noise : adding noise (batch_size, out_channels, height / 8, width / 8)
        
        for module in self:
            
            if getattr(module, "stride", None) == (2, 2):
                # for applying asymmetric padding, only padding right and bottom
                x = F.pad(x, (0,1,0,1))

            x = module(x)

        # for VAE latent space 
        # (batch_size, out_channels / 2, height / 8, width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        variance = torch.clamp(log_variance, -30, 20).exp() # clamping and transform log_var to var

        standart_dev = variance.sqrt()
          
        # Transform N(0, 1) -> N(mean, stdev) 
        # X = mean, + standart_dev * Z (noise)
        x = mean + standart_dev + noise

        # scale the output by a constant : from the original repository
        x *= 0.18125

        return x