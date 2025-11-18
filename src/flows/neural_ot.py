from typing import Optional

import torch
import torch.nn as nn

from .velocity_field import VelocityField
from .integration import integrate_flow
from .losses import flow_matching_loss


class NeuralOTFlow(nn.Module):
    """High-level wrapper for the neural OT flow with positional embeddings.

    The flow transports between token embeddings (x, p) where:
        x: token embeddings [B, L, d]
        p: positional embeddings [B, L, d] (sinusoidal)

    The velocity field is parameterized as v_theta(x(t), p, t).
    """

    def __init__(
        self,
        hidden_dim: int,
        time_embed_dim: int = 128,
        num_layers: int = 3,
        mlp_width: int = 2048,
        max_seq_len: int = 5000,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.v_theta = VelocityField(
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            num_layers=num_layers,
            mlp_width=mlp_width,
            max_seq_len=max_seq_len,
        )

    def forward(
        self,
        x: torch.Tensor,
        num_steps: int = 10,
        direction: str = "forward",
        p: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Integrate the flow from x.

        Args:
            x: Token embeddings [B, L, d]
            num_steps: Number of integration steps
            direction: 'forward' or 'backward'
            p: Positional embeddings [B, L, d]. If None, computed automatically.

        Returns:
            Transported token embeddings [B, L, d]
        """
        return integrate_flow(self.v_theta, x, num_steps=num_steps, direction=direction, p=p)

    def transport_nl_to_lean(
        self,
        x_nl: torch.Tensor,
        num_steps: int = 10,
        p: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transport NL embeddings to Lean space.

        Args:
            x_nl: NL token embeddings [B, L, d]
            num_steps: Number of integration steps
            p: Positional embeddings [B, L, d]

        Returns:
            Lean token embeddings [B, L, d]
        """
        return self.forward(x_nl, num_steps=num_steps, direction="forward", p=p)

    def transport_lean_to_nl(
        self,
        x_lean: torch.Tensor,
        num_steps: int = 10,
        p: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transport Lean embeddings to NL space.

        Args:
            x_lean: Lean token embeddings [B, L, d]
            num_steps: Number of integration steps
            p: Positional embeddings [B, L, d]

        Returns:
            NL token embeddings [B, L, d]
        """
        return self.forward(x_lean, num_steps=num_steps, direction="backward", p=p)

    def compute_flow_matching_loss(
        self,
        x_nl: torch.Tensor,
        x_lean: torch.Tensor,
        t_sample: str = "per_token",
        p: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute flow matching loss between NL and Lean embeddings.

        Args:
            x_nl: NL token embeddings [B, L, d]
            x_lean: Lean token embeddings [B, L, d]
            t_sample: 'per_token' or 'per_batch'
            p: Positional embeddings [B, L, d]

        Returns:
            Flow matching loss (scalar)
        """
        return flow_matching_loss(self.v_theta, x_nl, x_lean, t_sample=t_sample, p=p)
