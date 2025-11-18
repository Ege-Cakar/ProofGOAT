from typing import Optional
import torch


def flow_matching_loss(
    v_theta,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t_sample: str = "per_token",
    p: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Flow matching loss with positional embeddings.

    Args:
        v_theta: Velocity field model with signature v_theta(x, t, p)
        x0: Source embeddings [B, L, d] (e.g., NL)
        x1: Target embeddings [B, L, d] (e.g., Lean)
        t_sample: 'per_token' or 'per_batch' controlling sampling granularity
        p: Positional embeddings [B, L, d]. If None, computed by v_theta.

    Returns:
        Flow matching loss (MSE between predicted and target velocity)
    """
    B, L, d = x0.shape

    if t_sample == "per_token":
        t = torch.rand(B, L, device=x0.device, dtype=x0.dtype)
    else:
        t = torch.rand(B, 1, device=x0.device, dtype=x0.dtype).expand(B, L)

    # Linear interpolation: x(t) = (1-t) * x0 + t * x1
    t_exp = t.unsqueeze(-1)
    x_t = (1 - t_exp) * x0 + t_exp * x1

    # Target velocity is the constant vector field from x0 to x1
    v_target = x1 - x0

    # Predicted velocity from the model
    v_pred = v_theta(x_t, t, p)

    # MSE loss
    loss = torch.mean((v_pred - v_target) ** 2)
    return loss


def cycle_consistency_loss(
    neural_ot,
    x: torch.Tensor,
    num_steps: int = 8,
    p: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Forward then backward cycle loss: MSE(x, back(forward(x))).

    Args:
        neural_ot: NeuralOTFlow model
        x: Token embeddings [B, L, d]
        num_steps: Number of integration steps
        p: Positional embeddings [B, L, d]

    Returns:
        Cycle consistency loss
    """
    x_fwd = neural_ot.transport_nl_to_lean(x, num_steps=num_steps, p=p)
    x_rec = neural_ot.transport_lean_to_nl(x_fwd, num_steps=num_steps, p=p)
    return torch.mean((x - x_rec) ** 2)
