from typing import Optional

import torch
import torch.nn as nn

from .velocity_field import VelocityField
from .integration import integrate_flow
from .losses import flow_matching_loss


class NeuralOTFlow(nn.Module):
    """High-level wrapper for the neural OT flow with data normalization.

    The flow transports between token embeddings x where:
        x: token embeddings [B, L, d]

    The velocity field is parameterized as v_theta(x(t), t).
    """

    def __init__(
        self,
        hidden_dim: int,
        time_embed_dim: int = 128,
        num_layers: int = 3,
        mlp_width: int = 2048,
        dropout: float = 0.0,
        src_mean: Optional[torch.Tensor] = None,
        src_std: Optional[torch.Tensor] = None,
        tgt_mean: Optional[torch.Tensor] = None,
        tgt_std: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.v_theta = VelocityField(
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            num_layers=num_layers,
            mlp_width=mlp_width,
            dropout=dropout,
        )

        # Register normalization statistics as buffers
        self.register_buffer("src_mean", src_mean if src_mean is not None else torch.zeros(hidden_dim))
        self.register_buffer("src_std", src_std if src_std is not None else torch.ones(hidden_dim))
        self.register_buffer("tgt_mean", tgt_mean if tgt_mean is not None else torch.zeros(hidden_dim))
        self.register_buffer("tgt_std", tgt_std if tgt_std is not None else torch.ones(hidden_dim))

    def _normalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (x - mean) / (std + 1e-6)

    def _denormalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * (std + 1e-6) + mean

    def forward(
        self,
        x: torch.Tensor,
        num_steps: int = 10,
        direction: str = "forward",
    ) -> torch.Tensor:
        """Integrate the flow from x (assumed normalized)."""
        return integrate_flow(self.v_theta, x, num_steps=num_steps, direction=direction)

    def transport_nl_to_lean(
        self,
        x_nl: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Transport NL embeddings to Lean space (handles normalization)."""
        # Normalize input
        x_norm = self._normalize(x_nl, self.src_mean, self.src_std)
        
        # Integrate in normalized space
        x_lean_norm = self.forward(x_norm, num_steps=num_steps, direction="forward")
        
        # Denormalize output
        return self._denormalize(x_lean_norm, self.tgt_mean, self.tgt_std)

    def transport_lean_to_nl(
        self,
        x_lean: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Transport Lean embeddings to NL space (handles normalization)."""
        # Normalize input
        x_norm = self._normalize(x_lean, self.tgt_mean, self.tgt_std)
        
        # Integrate backward in normalized space
        x_nl_norm = self.forward(x_norm, num_steps=num_steps, direction="backward")
        
        # Denormalize output
        return self._denormalize(x_nl_norm, self.src_mean, self.src_std)

    def compute_flow_matching_loss(
        self,
        x_nl: torch.Tensor,
        x_lean: torch.Tensor,
        t_sample: str = "per_token",
    ) -> torch.Tensor:
        """Compute flow matching loss between NL and Lean embeddings (handles normalization)."""
        # Normalize inputs
        x_nl_norm = self._normalize(x_nl, self.src_mean, self.src_std)
        x_lean_norm = self._normalize(x_lean, self.tgt_mean, self.tgt_std)
        
        # Compute loss in normalized space
        return flow_matching_loss(self.v_theta, x_nl_norm, x_lean_norm, t_sample=t_sample)
