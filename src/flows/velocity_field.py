from typing import Optional

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """Simple time embedding: sinusoidal + linear projection."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: shape [..., 1] or [...]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        # create sinusoidal embedding
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=device, dtype=torch.float32) * (torch.log(torch.tensor(1e4)) / (half - 1))
        )
        args = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            pad = torch.zeros(*emb.shape[:-1], self.dim - emb.shape[-1], device=device)
            emb = torch.cat([emb, pad], dim=-1)
        return self.proj(emb)


class VelocityField(nn.Module):
    """Velocity field v_theta(h, t) mapping (d + time_emb_dim) -> d"""

    def __init__(
        self,
        hidden_dim: int,
        time_embed_dim: int = 128,
        num_layers: int = 3,
        mlp_width: int = 2048,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.time_embed = TimeEmbedding(time_embed_dim)

        if activation is None:
            activation = nn.GELU()

        layers = []
        in_dim = hidden_dim + time_embed_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else mlp_width, mlp_width))
            layers.append(activation)
        layers.append(nn.Linear(mlp_width if num_layers > 1 else in_dim, hidden_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute velocity.

        h: [B, L, d] or [N, d]
        t: scalar in [0,1], or tensor broadcastable to leading dims of h with shape [B, L] or [B]
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

        # If t is scalar, expand to [B*L, 1]
        if t_tensor.dim() == 0:
            t_flat = t_tensor.unsqueeze(0).expand(h_flat.shape[0], 1)
        elif t_tensor.dim() == 1:
            # assume shape [B] -> expand to [B, L]
            B = orig_shape[0]
            L = orig_shape[1] if len(orig_shape) > 1 else 1
            if t_tensor.shape[0] == B:
                t_exp = t_tensor.unsqueeze(-1).expand(B, L)
                t_flat = t_exp.reshape(-1, 1)
            else:
                # fallback: repeat or truncate to match
                t_flat = t_tensor.reshape(-1, 1).expand(h_flat.shape[0], 1)
        elif t_tensor.dim() == 2:
            # assume [B, L]
            t_flat = t_tensor.reshape(-1, 1)
        else:
            t_flat = t_tensor.reshape(-1, 1)

        time_emb = self.time_embed(t_flat)

        x = torch.cat([h_flat, time_emb], dim=-1)
        v_flat = self.net(x)
        v = v_flat.view(*orig_shape)
        return v
