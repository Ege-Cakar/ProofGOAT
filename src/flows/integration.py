from typing import Literal, Optional

import torch


def integrate_flow(
    v_theta,
    x0: torch.Tensor,
    num_steps: int = 10,
    direction: Literal['forward','backward']='forward',
    p: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Integrate velocity field using Euler method.

    Args:
        v_theta: Velocity field model with signature v_theta(x, t, p)
        x0: Initial token embeddings [B, L, d]
        num_steps: Number of integration steps
        direction: 'forward' (t=0 to t=1) or 'backward' (t=1 to t=0)
        p: Positional embeddings [B, L, d]. If None, computed by v_theta.

    Returns:
        x_final: Final token embeddings [B, L, d]
    """
    dt = 1.0 / float(num_steps)
    x = x0
    device = x.device

    if direction == 'forward':
        for k in range(num_steps):
            t = torch.full((x.shape[0],), float(k) * dt, device=device)
            v = v_theta(x, t, p)
            x = x + dt * v
        return x
    else:
        # backward integration from t=1 down to 0
        for k in range(num_steps, 0, -1):
            t = torch.full((x.shape[0],), float(k) * dt, device=device)
            v = v_theta(x, t, p)
            x = x - dt * v
        return x
