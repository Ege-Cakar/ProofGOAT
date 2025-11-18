import torch


def flow_matching_loss(v_theta, h0: torch.Tensor, h1: torch.Tensor, t_sample: str = "per_token") -> torch.Tensor:
    """Flow matching loss.

    h0, h1: [B, L, d]
    t_sample: 'per_token' or 'per_batch' controlling sampling granularity
    """
    B, L, d = h0.shape

    if t_sample == "per_token":
        t = torch.rand(B, L, device=h0.device, dtype=h0.dtype)
    else:
        t = torch.rand(B, 1, device=h0.device, dtype=h0.dtype)

    # linear interpolation
    t_exp = t.unsqueeze(-1)
    h_t = (1 - t_exp) * h0 + t_exp * h1

    v_target = h1 - h0

    v_pred = v_theta(h_t, t)

    loss = torch.mean((v_pred - v_target) ** 2)
    return loss


def cycle_consistency_loss(neural_ot, h: torch.Tensor, num_steps: int = 8) -> torch.Tensor:
    """Forward then backward cycle loss: MSE(h, back(forward(h)))."""
    h_fwd = neural_ot.transport_nl_to_lean(h, num_steps=num_steps)
    h_rec = neural_ot.transport_lean_to_nl(h_fwd, num_steps=num_steps)
    return torch.mean((h - h_rec) ** 2)
