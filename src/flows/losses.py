from typing import Optional
import torch


def flow_matching_loss(
    v_theta,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t_sample: str = "per_token",
) -> torch.Tensor:
import torch.nn.functional as F
import ot
from typing import Optional


def flow_matching_loss(model, x_0, x_1, t=None):
    """
    Conditional Flow Matching loss.
    L_CFM = E_t,p(x_1) [ || v_t(psi_t(x_0)) - u_t(x_0|x_1) ||^2 ]
    
    We use the optimal transport path (straight line):
    psi_t(x_0) = (1-t)x_0 + t*x_1
    u_t(x_0|x_1) = x_1 - x_0
    """
    batch_size = x_0.shape[0]
    
    if t is None:
        # Sample t uniformly from [0, 1]
        t = torch.rand(batch_size, device=x_0.device).type_as(x_0)
    
    # Reshape t for broadcasting: [B, 1]
    t_reshaped = t.view(-1, 1)
    
    # Compute target velocity (straight line)
    u_t = x_1 - x_0
    
    # Compute interpolated state
    # x_t = (1 - t) * x_0 + t * x_1
    x_t = (1 - t_reshaped) * x_0 + t_reshaped * x_1
    
    # Predict velocity field
    v_t = model(x_t, t)
    
    # MSE Loss
    loss = F.mse_loss(v_t, u_t)
    
    return loss

def cycle_consistency_loss(model, x, num_steps=10):
    """
    Computes cycle consistency loss: x -> Lean -> x_rec
    """
    # 1. Transport x (NL) -> x_lean (Lean domain)
    # We use the model's transport method if available, or integrate manually
    # Assuming model has a method or we use the helper
    # For loss computation, we need differentiable integration. 
    # The 'neural_ot.transport_nl_to_lean' uses 'integrate_flow' which is differentiable.
    
    # Import here to avoid circular import if possible, or assume model has the method
    from .neural_ot import NeuralOTFlow
    
    if isinstance(model, NeuralOTFlow):
        x_fwd = model.transport_nl_to_lean(x, num_steps=num_steps)
        x_rec = model.transport_lean_to_nl(x_fwd, num_steps=num_steps)
    else:
        # Fallback if model is just the velocity field
        # This part is tricky without the full class context, but let's assume standard usage
        raise NotImplementedError("Cycle loss requires NeuralOTFlow instance")

    loss = F.mse_loss(x_rec, x)
    return loss

def sinkhorn_loss(x_pred, x_target, reg=0.1, num_iter=100):
    """
    Computes the Sinkhorn distance between two batches of samples.
    x_pred: [B, D]
    x_target: [B, D]
    reg: Entropic regularization parameter
    """
    # Compute pairwise distance matrix
    M = ot.dist(x_pred, x_target, metric='euclidean')
    
    # Compute Sinkhorn distance
    # ot.sinkhorn2 returns the loss (scalar or vector depending on inputs)
    # We assume uniform weights for the samples
    batch_size = x_pred.shape[0]
    a = torch.ones(batch_size, device=x_pred.device) / batch_size
    b = torch.ones(batch_size, device=x_pred.device) / batch_size
    
    loss = ot.sinkhorn2(a, b, M, reg, numItermax=num_iter)
    
    return loss
