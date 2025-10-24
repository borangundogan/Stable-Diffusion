import torch
from torch import nn
from torch.nn import functional  as F

from src.attention.self_attention import SelfAttention
from src.attention.cross_attention import CrossAttention

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int = 1280):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        t: (B, time_dim)
        return: (B, C_out, H, W)
        """
        residual = self.skip(x)

        # First path
        x = self.conv1(self.act(self.norm1(x)))

        # Inject time
        t = self.time_proj(t).unsqueeze(-1).unsqueeze(-1)
        x = x + t

        # Second path
        x = self.conv2(self.act(self.norm2(x)))

        return x + residual

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context: int = 768):
        super().__init__()
        self.channels = n_head * n_embd

        self.norm_in = nn.GroupNorm(32, self.channels, eps=1e-6)
        self.proj_in = nn.Conv2d(self.channels, self.channels, 1)

        self.norm1 = nn.LayerNorm(self.channels)
        self.self_attn = SelfAttention(n_head, self.channels, in_proj_bias=False)

        self.norm2 = nn.LayerNorm(self.channels)
        self.cross_attn = CrossAttention(n_head, self.channels, d_context, bias=False)

        self.norm3 = nn.LayerNorm(self.channels)
        self.ff_geglu = nn.Sequential(
            nn.Linear(self.channels, 8 * n_embd),   # 4x + 4x gate
            nn.GELU(),
            nn.Linear(4 * n_embd, self.channels)
        )

        self.proj_out = nn.Conv2d(self.channels, self.channels, 1)
    
    def forward(self, x, context):
        B, C, H, W = x.shape
        residual_long = x

        # Normalize + Flatten
        x = self.proj_in(self.norm_in(x)).view(B, C, H * W).transpose(1, 2)

        # Self-Attention
        residual = x
        x = self.self_attn(self.norm1(x))
        x = x + residual

        # Cross-Attention
        residual = x
        x = self.cross_attn(self.norm2(x), context)
        x = x + residual

        # FeedForward (GeGLU approximation)
        residual = x
        x = self.ff_geglu(self.norm3(x))
        x = x + residual

        # Reshape back
        x = x.transpose(1, 2).view(B, C, H, W)
        return self.proj_out(x) + residual_long

class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))
