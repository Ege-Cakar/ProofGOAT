from typing import Optional

import torch
import torch.nn as nn
import numpy as np


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)
        self.embedding_size = embedding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ResidualBlock(nn.Module):
    """ResNet block with time injection."""

    def __init__(self, dim: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        
        # Time injection
        self.time_proj = nn.Linear(time_dim, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        # Add time embedding
        h = h + self.time_proj(t_emb)
        h = self.linear1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.norm2(h)
        h = self.linear2(h)
        return x + h


class VelocityField(nn.Module):
    """Velocity field v_theta(h, t) mapping (d + time_emb_dim) -> d using ResNet."""

    def __init__(
        self,
        hidden_dim: int,
        time_embed_dim: int = 256,
        num_layers: int = 3,
        mlp_width: int = 2048, # Kept for compatibility, but used as inner dim if needed or ignored
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        
        # Gaussian Fourier Features
        self.time_embed = GaussianFourierProjection(time_embed_dim, scale=30.0)
        
        # Input projection if needed, but we assume input is already hidden_dim
        # We'll use a sequence of ResNet blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_embed_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Final output projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute velocity.

        h: [B, L, d] or [N, d]
        t: scalar in [0,1], or tensor broadcastable to leading dims of h
        returns: same shape as h
        """
        orig_shape = h.shape
        d = h.shape[-1]

        # flatten h to [B*L, d]
        h_flat = h.view(-1, d)

        # prepare t aligned with token dimension
        if torch.is_tensor(t):
            t_tensor = t
        else:
            t_tensor = torch.tensor(float(t), device=h.device, dtype=h.dtype)

        # Flatten t to [B*L]
        if t_tensor.dim() == 0:
            t_flat = t_tensor.expand(h_flat.shape[0])
        elif t_tensor.dim() == 1:
            # assume shape [B] -> expand to [B, L]
            B = orig_shape[0]
            L = orig_shape[1] if len(orig_shape) > 1 else 1
            if t_tensor.shape[0] == B:
                t_exp = t_tensor.unsqueeze(-1).expand(B, L)
                t_flat = t_exp.reshape(-1)
            else:
                t_flat = t_tensor.reshape(-1).expand(h_flat.shape[0])
        else:
            t_flat = t_tensor.reshape(-1)

        time_emb = self.time_embed(t_flat) # [B*L, time_embed_dim]

        x = h_flat
        for block in self.blocks:
            x = block(x, time_emb)
            
        x = self.final_norm(x)
        v_flat = self.head(x)
        
        v = v_flat.view(*orig_shape)
        return v
