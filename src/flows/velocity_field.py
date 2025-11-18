from typing import Optional
import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings as in 'Attention is All You Need'.

    These embeddings encode the position of each token in the sequence.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the div term for sinusoidal functions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int, batch_size: int = 1) -> torch.Tensor:
        """
        Returns positional embeddings for sequences.

        Args:
            seq_len: Length of the sequence
            batch_size: Batch size

        Returns:
            Positional embeddings of shape (batch_size, seq_len, d_model)
        """
        pos_emb = self.pe[:seq_len].unsqueeze(0).expand(batch_size, seq_len, self.d_model)
        return pos_emb


class TimeEmbedding(nn.Module):
    """Time embedding with non-overlapping sinusoidals + MLP for smoothing.

    Uses higher frequency range than position embeddings to avoid overlap.
    """

    def __init__(self, dim: int = 128, hidden_mult: int = 4):
        super().__init__()
        self.dim = dim
        hidden_dim = dim * hidden_mult

        # Use higher frequencies for time (different range than positional embeddings)
        # Position uses 1-10000, we use 10-100000 for non-overlap
        half = dim // 2
        self.register_buffer(
            'frequencies',
            torch.exp(
                torch.linspace(
                    math.log(10.0),      # Start at higher frequency
                    math.log(100000.0),  # Go to even higher frequency
                    half
                )
            )
        )

        # MLP to smooth and project time embeddings
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values, shape [B] or [B, 1] or [B, L]
               Values should be in [0, 1]

        Returns:
            Time embeddings of shape [B, dim] or [B, L, dim]
        """
        # Handle different input shapes
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B] -> [B, 1]
            squeeze_output = True
        elif t.dim() == 2 and t.shape[-1] == 1:
            squeeze_output = False
        elif t.dim() == 2:
            # [B, L] -> [B, L, 1]
            t = t.unsqueeze(-1)
            squeeze_output = False
        else:
            squeeze_output = False

        # Create sinusoidal embeddings
        # t: [..., 1], frequencies: [half]
        angles = t * self.frequencies.unsqueeze(0)  # [..., half]

        # Concatenate sin and cos
        sin_emb = torch.sin(angles)
        cos_emb = torch.cos(angles)
        time_emb = torch.cat([sin_emb, cos_emb], dim=-1)  # [..., dim]

        # Apply MLP for smoothing
        time_emb = self.proj(time_emb)

        if squeeze_output and time_emb.shape[-2] == 1:
            time_emb = time_emb.squeeze(-2)

        return time_emb


class VelocityField(nn.Module):
    """Velocity field v_theta(x, p, t) taking token embeddings x, positional embeddings p, and time t.

    The velocity field is parameterized as:
        v_theta(x(t), p, t) -> dx/dt

    where:
        x(t): token embeddings at time t, shape [B, L, d]
        p: positional embeddings (sinusoidal), shape [B, L, d]
        t: time in [0, 1], shape [B] or [B, L]
    """

    def __init__(
        self,
        hidden_dim: int,
        time_embed_dim: int = 128,
        num_layers: int = 3,
        mlp_width: int = 2048,
        max_seq_len: int = 5000,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim

        # Positional embedding (sinusoidal, as in transformers)
        self.pos_embed = SinusoidalPositionalEmbedding(hidden_dim, max_len=max_seq_len)

        # Time embedding (non-overlapping sinusoidals + MLP)
        self.time_embed = TimeEmbedding(time_embed_dim)

        if activation is None:
            activation = nn.GELU()

        # Network takes concatenation of: x(t) + p + time_emb
        # Input dimension: hidden_dim (x) + hidden_dim (p) + time_embed_dim (t)
        layers = []
        in_dim = hidden_dim + hidden_dim + time_embed_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else mlp_width, mlp_width))
            layers.append(activation)
        layers.append(nn.Linear(mlp_width if num_layers > 1 else in_dim, hidden_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, p: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute velocity v_theta(x(t), p, t).

        Args:
            x: Token embeddings at time t, shape [B, L, d]
            t: Time values in [0, 1], shape [B] or [B, L] or scalar
            p: Positional embeddings, shape [B, L, d]. If None, will be computed.

        Returns:
            Velocity field v, shape [B, L, d]
        """
        B, L, d = x.shape
        device = x.device

        # Get or compute positional embeddings
        if p is None:
            p = self.pos_embed(L, B).to(device)

        # Ensure positional embeddings match hidden_dim
        if p.shape[-1] != self.hidden_dim:
            raise ValueError(f"Positional embedding dim {p.shape[-1]} != hidden_dim {self.hidden_dim}")

        # Prepare time embeddings
        if torch.is_tensor(t):
            t_tensor = t
        else:
            t_tensor = torch.tensor(float(t), device=device, dtype=x.dtype)

        # Expand time to match sequence dimension if needed
        if t_tensor.dim() == 0:
            # Scalar time -> [B]
            t_tensor = t_tensor.unsqueeze(0).expand(B)

        if t_tensor.dim() == 1 and t_tensor.shape[0] == B:
            # [B] -> [B, L] (broadcast same time to all tokens)
            t_tensor = t_tensor.unsqueeze(1).expand(B, L)
        elif t_tensor.dim() == 2 and t_tensor.shape == (B, L):
            # Already [B, L]
            pass
        else:
            raise ValueError(f"Invalid time tensor shape: {t_tensor.shape}, expected [B] or [B, L] where B={B}, L={L}")

        # Compute time embeddings [B, L, time_embed_dim]
        time_emb = self.time_embed(t_tensor)

        # Concatenate: x(t) + p + time_emb
        # All shapes: [B, L, d], [B, L, d], [B, L, time_embed_dim]
        inputs = torch.cat([x, p, time_emb], dim=-1)  # [B, L, 2*d + time_embed_dim]

        # Debug: check dimensions
        expected_dim = 2 * self.hidden_dim + self.time_embed_dim
        if inputs.shape[-1] != expected_dim:
            raise ValueError(
                f"Input dimension mismatch: got {inputs.shape[-1]}, expected {expected_dim}\n"
                f"x: {x.shape}, p: {p.shape}, time_emb: {time_emb.shape}\n"
                f"hidden_dim: {self.hidden_dim}, time_embed_dim: {self.time_embed_dim}"
            )

        # Flatten for processing
        inputs_flat = inputs.view(B * L, -1)
        v_flat = self.net(inputs_flat)
        v = v_flat.view(B, L, d)

        return v
