from typing import Optional

import torch
import torch.nn as nn

from .velocity_field import VelocityField
from .integration import integrate_flow
from .losses import flow_matching_loss


class NeuralOTFlow(nn.Module):
    """High-level wrapper for the neural OT flow."""

    def __init__(
        self,
        hidden_dim: int,
        time_embed_dim: int = 128,
        num_layers: int = 3,
        mlp_width: int = 2048,
    ) -> None:
        super().__init__()
        self.v_theta = VelocityField(
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            num_layers=num_layers,
            mlp_width=mlp_width,
        )

    def forward(self, h: torch.Tensor, num_steps: int = 10, direction: str = "forward") -> torch.Tensor:
        return integrate_flow(self.v_theta, h, num_steps=num_steps, direction=direction)

    def transport_nl_to_lean(self, h_nl: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        return self.forward(h_nl, num_steps=num_steps, direction="forward")

    def transport_lean_to_nl(self, h_lean: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        return self.forward(h_lean, num_steps=num_steps, direction="backward")

    def compute_flow_matching_loss(self, h_nl: torch.Tensor, h_lean: torch.Tensor):
        return flow_matching_loss(self.v_theta, h_nl, h_lean)
