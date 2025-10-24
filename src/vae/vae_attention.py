import torch
from torch import nn
from torch.nn import functional as F

from src.attention.self_attention import SelfAttention 

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(n_heads=1, d_embed=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        residual = x

        # Normalize in float32 for FP16 stability
        x = self.groupnorm(x.float())
        
        B, C, H, W = x.shape

        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        # Apply self-attention (no causal mask)
        x = self.attention(x)

        # Restore spatial dimensions: (B, H*W, C) -> (B, C, H, W)
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # Add residual and restore original dtype
        return (x + residual).to(residual.dtype)
