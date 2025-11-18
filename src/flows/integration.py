from typing import Literal

import torch


def integrate_flow(v_theta, h0: torch.Tensor, num_steps: int = 10, direction: Literal['forward','backward']='forward') -> torch.Tensor:
    """Integrate velocity field using Euler method.

    h0: [B, L, d]
    Returns h_final: [B, L, d]
    """
    dt = 1.0 / float(num_steps)
    h = h0
    device = h.device

    if direction == 'forward':
        for k in range(num_steps):
            t = torch.full((h.shape[0],), float(k) * dt, device=device)
            v = v_theta(h, t)
            h = h + dt * v
        return h
    else:
        # backward integration from t=1 down to 0
        for k in range(num_steps, 0, -1):
            t = torch.full((h.shape[0],), float(k) * dt, device=device)
            v = v_theta(h, t)
            h = h - dt * v
        return h
