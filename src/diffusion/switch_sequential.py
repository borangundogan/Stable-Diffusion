import torch
from torch import nn, functional as F
from blocks import UNET_ResidualBlock, UNET_AttentionBlock

class SwitchSequential(nn.Sequential):
    
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        for layer in self.layers:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x