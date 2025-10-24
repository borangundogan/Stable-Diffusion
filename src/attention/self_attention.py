import torch
from torch import nn
from torch.nn import functional as F

import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        # Combine Q, K, V projections into one linear for efficiency
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

    def forward(self, x: torch.Tensor, causal_mask=False):
        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_head

        # Compute Q, K, V
        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        def split_heads(t):
            return t.view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)

        q, k, v = map(split_heads, (q, k, v))

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)

        if causal_mask:
            mask = torch.ones_like(attn, dtype=torch.bool).triu(1)
            attn.masked_fill_(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, L, Dh)

        # Recombine heads
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        # Final linear projection
        return self.out_proj(out)