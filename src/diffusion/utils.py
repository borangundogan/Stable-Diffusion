import torch
from torch import nn
from torch.nn import functional as F

def get_timestep_embedding(t, dim):
    half_dim = dim // 2
    freqs = torch.exp(-torch.arange(half_dim, device=t.device) * (torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
    emb = t[:, None] * freqs[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    return emb

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, 4 * dim)
        self.linear2 = nn.Linear(4 * dim, 4 * dim)

    def forward(self, t):
        t = get_timestep_embedding(t, self.linear1.in_features)
        t = F.silu(self.linear1(t))
        t = self.linear2(t)
        return t
