import torch
from torch import nn
from torch.nn import functional as F

import math

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        # Combined projections to reduce kernel calls
        self.q_proj = nn.Linear(d_embed, d_embed, bias=bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=bias)

    def forward(self, x, context):
            B, Lq, Dq = x.shape
            _, Lk, _ = context.shape
            H, Dh = self.n_heads, self.d_head

            # Project
            q = self.q_proj(x)
            k = self.k_proj(context)
            v = self.v_proj(context)

            # Reshape for multi-head
            def reshape_heads(t):
                return t.view(B, -1, H, Dh).transpose(1, 2)  # (B, H, L, Dh)

            q, k, v = map(reshape_heads, (q, k, v))

            # Attention weights
            attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)
            attn = F.softmax(attn, dim=-1)

            # Apply attention
            out = torch.matmul(attn, v)                     # (B, H, Lq, Dh)
            out = out.transpose(1, 2).contiguous().view(B, Lq, Dq)
            return self.out_proj(out)
