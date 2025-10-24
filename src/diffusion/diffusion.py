import torch
from torch import nn
from torch.nn import functional as F

from utils import TimeEmbedding
from unet import UNET, UNET_OutputLayer

class Diffusion(nn.Module):
    def __init__(self, base_channels=320, out_channels=4):
        super().__init__()
        self.time_embedding = TimeEmbedding(base_channels)
        self.unet = UNET()
        self.final = UNET_OutputLayer(base_channels, out_channels)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, 4, H/8, W/8)
        context: (B, seq_len, dim)
        time: (B, 1)
        """
        time = self.time_embedding(time)
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
                            dtype=torch.float16):
            x = self.unet(latent, context, time)
            return self.final(x)