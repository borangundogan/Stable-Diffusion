import torch
from torch import nn
import torch.nn.functional as F

from vae_attention import VAE_AttentionBlock
from residual import VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
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
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        noise: (B, 4, H/8, W/8)
        Returns: latent (B, 4, H/8, W/8)
        """
        for module in self:
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                # Asymmetric padding before downsampling conv
                x = F.pad(x, (0, 1, 0, 1)).contiguous()
            x = module(x)

        # Split latent channels into mean & log variance
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # Safe variance computation
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        standard_dev = torch.sqrt(variance + 1e-8)

        # Reparameterization trick: z = μ + σ * ε
        x = mean + standard_dev * noise.to(mean.dtype)

        # Scale as in Stable Diffusion
        x *= 0.18125

        return x
