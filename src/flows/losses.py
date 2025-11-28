"""Flow matching losses with POT-based optimal transport alignment."""

from typing import Optional, List, Tuple
import torch
import numpy as np
import ot
from tqdm import tqdm


def compute_ot_coupling_pot(
    x0: np.ndarray,
    x1: np.ndarray,
    cost: str = "euclidean",
    reg: float = 0.05,
    method: str = "sinkhorn"
) -> np.ndarray:
    """Compute OT coupling between two point clouds using POT.
    
    Args:
        x0: Source embeddings [L0, d]
        x1: Target embeddings [L1, d]
        cost: Cost function - "euclidean" or "cosine"
        reg: Entropy regularization (for sinkhorn)
        method: "sinkhorn" (entropic regularized) or "emd" (exact)
        
    Returns:
        Coupling matrix P [L0, L1]
    """
    L0, L1 = x0.shape[0], x1.shape[0]
    
    # Uniform marginals
    a = np.ones(L0, dtype=np.float64) / L0
    b = np.ones(L1, dtype=np.float64) / L1
    
    # Compute cost matrix
    if cost == "euclidean":
        C = ot.dist(x0.astype(np.float64), x1.astype(np.float64), metric='sqeuclidean')
    elif cost == "cosine":
        x0_norm = x0 / (np.linalg.norm(x0, axis=1, keepdims=True) + 1e-8)
        x1_norm = x1 / (np.linalg.norm(x1, axis=1, keepdims=True) + 1e-8)
        C = (1 - x0_norm @ x1_norm.T).astype(np.float64)
    else:
        raise ValueError(f"Unknown cost: {cost}")
    
    # Normalize cost matrix for numerical stability
    C = C / (C.max() + 1e-8)
    
    # Compute OT plan
    if method == "sinkhorn":
        P = ot.sinkhorn(a, b, C, reg=reg, numItermax=100)
    elif method == "emd":
        P = ot.emd(a, b, C)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return P.astype(np.float32)


def precompute_ot_couplings(
    nl_embeddings: List[np.ndarray],
    lean_embeddings: List[np.ndarray],
    cost: str = "euclidean",
    reg: float = 0.05,
    method: str = "sinkhorn",
    show_progress: bool = True
) -> List[np.ndarray]:
    """Precompute OT couplings for all pairs in the dataset.
    
    Args:
        nl_embeddings: List of NL embeddings, each [L_i, d]
        lean_embeddings: List of Lean embeddings, each [L_i, d]
        cost: Cost function
        reg: Sinkhorn regularization
        method: OT method
        show_progress: Whether to show progress bar
        
    Returns:
        List of coupling matrices
    """
    assert len(nl_embeddings) == len(lean_embeddings), "Mismatched number of pairs"
    
    couplings = []
    iterator = zip(nl_embeddings, lean_embeddings)
    
    if show_progress:
        iterator = tqdm(
            list(iterator), 
            desc="Computing OT couplings",
            unit="pair"
        )
    
    for nl_emb, lean_emb in iterator:
        P = compute_ot_coupling_pot(nl_emb, lean_emb, cost=cost, reg=reg, method=method)
        couplings.append(P)
    
    return couplings


def compute_aligned_targets(
    x0: torch.Tensor,
    x1: torch.Tensor,
    coupling: torch.Tensor
) -> torch.Tensor:
    """Compute OT-aligned targets using precomputed coupling.
    
    For each source token, compute the expected target as weighted average
    according to the OT coupling (barycentric projection).
    
    Args:
        x0: Source embeddings [B, L0, d]
        x1: Target embeddings [B, L1, d]
        coupling: OT coupling [B, L0, L1]
        
    Returns:
        Aligned targets [B, L0, d]
    """
    # Normalize coupling to get conditional distribution P(j|i)
    coupling_normalized = coupling / (coupling.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Compute expected target for each source token
    x1_aligned = torch.bmm(coupling_normalized, x1)
    
    return x1_aligned


def ot_flow_matching_loss(
    v_theta,
    x0: torch.Tensor,
    x1: torch.Tensor,
    coupling: torch.Tensor,
    t_sample: str = "per_token",
    p: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """OT-aligned Flow Matching loss with precomputed coupling.
    
    Args:
        v_theta: Velocity field model with signature v_theta(x, t, p)
        x0: Source embeddings [B, L0, d] (NL)
        x1: Target embeddings [B, L1, d] (Lean)
        coupling: Precomputed OT coupling [B, L0, L1]
        t_sample: 'per_token' or 'per_batch' controlling time sampling
        p: Positional embeddings [B, L0, d]. If None, computed by v_theta.
        
    Returns:
        Flow matching loss (MSE between predicted and OT-aligned target velocity)
    """
    B, L0, d = x0.shape
    device = x0.device
    dtype = x0.dtype
    
    # Compute aligned targets using OT coupling
    x1_aligned = compute_aligned_targets(x0, x1, coupling)
    
    # Sample time
    if t_sample == "per_token":
        t = torch.rand(B, L0, device=device, dtype=dtype)
    else:
        t = torch.rand(B, 1, device=device, dtype=dtype).expand(B, L0)
    
    # Linear interpolation along OT-aligned paths
    t_exp = t.unsqueeze(-1)
    x_t = (1 - t_exp) * x0 + t_exp * x1_aligned
    
    # Target velocity: direction from source to aligned target
    v_target = x1_aligned - x0
    
    # Predicted velocity
    v_pred = v_theta(x_t, t, p)
    
    # MSE loss
    loss = torch.mean((v_pred - v_target) ** 2)
    return loss


def flow_matching_loss(
    v_theta,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t_sample: str = "per_token",
    p: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Flow matching loss with positional embeddings (element-wise aligned version).
    
    WARNING: This assumes x0 and x1 are already element-wise aligned.
    For unaligned sequences, use ot_flow_matching_loss with precomputed couplings.

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
