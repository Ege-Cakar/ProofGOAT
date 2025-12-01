"""Evaluate Augmented OT-CFM model on transport quality.

This script:
1. Loads a trained A-OT-CFM model checkpoint
2. Loads NL and Lean embeddings (last 500 samples from parquet files)
3. Computes OT pairings using POT (GPU backend)
4. Evaluates transport quality via:
   a) Cosine similarity between transported and target tokens
   b) Wasserstein-2 distance between transported and target distributions
   c) Round-trip (circle) accuracy: transport A→B→A and measure same metrics

Usage:
  python -m scripts.eval_augmented_ot --model outputs/neural_ot/augmented_online_run_XXX/checkpoints/best_model.pt \
    --nl-path outputs/kimina17_all_nl_embeddings.parquet \
    --lean-path outputs/kimina17_all_lean_embeddings.parquet \
    --num-samples 500
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
from tqdm import tqdm

# POT for optimal transport
import ot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

# =============================================================================
# Device
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Model Definition (same as train_augmented_ot_online.py)
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class AugmentedVelocityField(nn.Module):
    """Transformer-based velocity field for A-OT-CFM."""
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        time_embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.aug_dim = hidden_dim + 1
        
        self.transformer_dim = ((self.aug_dim + num_heads - 1) // num_heads) * num_heads
        
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, self.transformer_dim)
        
        self.input_proj = nn.Linear(self.aug_dim, self.transformer_dim)
        self.input_norm = nn.LayerNorm(self.transformer_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=num_heads,
            dim_feedforward=int(self.transformer_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_norm = nn.LayerNorm(self.transformer_dim)
        self.output_proj = nn.Linear(self.transformer_dim, self.aug_dim)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        h = self.input_proj(x)
        t_emb = self.time_embed(t)
        t_emb = self.time_proj(t_emb)
        t_emb = t_emb.unsqueeze(1)
        h = self.input_norm(h) + t_emb
        h = self.transformer(h)
        h = self.output_norm(h)
        v = self.output_proj(h)
        return v


# =============================================================================
# Sinusoidal Positional Embeddings
# =============================================================================

def get_sinusoidal_embedding(positions: np.ndarray, d: int) -> np.ndarray:
    """Generate sinusoidal positional embeddings."""
    N = len(positions)
    pe = np.zeros((N, d), dtype=np.float32)
    div_term = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))
    
    for i, p in enumerate(positions):
        pe[i, 0::2] = np.sin(p * np.pi * div_term)
        pe[i, 1::2] = np.cos(p * np.pi * div_term[:d//2] if d % 2 == 0 else div_term[:(d+1)//2])
    
    return pe


# =============================================================================
# Augmented State Creation
# =============================================================================

def create_augmented_state(
    embeddings: np.ndarray,
    N_max: int,
    d: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create augmented state with existence channel and distributed voids."""
    L = min(embeddings.shape[0], N_max)
    embeddings = embeddings[:L]
    n_void = N_max - L
    
    active_positions = np.linspace(1/L, 1.0, L) if L > 0 else np.array([])
    
    if n_void > 0:
        void_positions = np.linspace(1/(n_void+1), n_void/(n_void+1), n_void)
    else:
        void_positions = np.array([])
    
    all_positions = np.concatenate([active_positions, void_positions])
    all_indices = np.argsort(all_positions)
    
    state = np.zeros((N_max, d + 1), dtype=np.float32)
    positions = np.zeros(N_max, dtype=np.float32)
    is_active = np.zeros(N_max, dtype=bool)
    
    active_pe = get_sinusoidal_embedding(active_positions, d) if L > 0 else np.zeros((0, d))
    void_pe = get_sinusoidal_embedding(void_positions, d) if n_void > 0 else np.zeros((0, d))
    
    active_idx = 0
    void_idx = 0
    
    for out_idx, orig_idx in enumerate(all_indices):
        if orig_idx < L:
            state[out_idx, :d] = embeddings[active_idx] + active_pe[active_idx]
            state[out_idx, d] = 1.0
            positions[out_idx] = active_positions[active_idx]
            is_active[out_idx] = True
            active_idx += 1
        else:
            state[out_idx, :d] = void_pe[void_idx]
            state[out_idx, d] = 0.0
            positions[out_idx] = void_positions[void_idx]
            is_active[out_idx] = False
            void_idx += 1
    
    return state, positions, is_active


# =============================================================================
# OT Cost Matrix (GPU)
# =============================================================================

def compute_augmented_ot_cost_gpu(
    src_positions: torch.Tensor,
    tgt_positions: torch.Tensor,
    src_active: torch.Tensor,
    tgt_active: torch.Tensor,
    src_embeddings: torch.Tensor,
    tgt_embeddings: torch.Tensor,
    lambda_pos: float = 1.0,
    alpha_delete: float = 1.0,
    alpha_create: float = 1.0,
    beta_create: float = 0.1,
) -> torch.Tensor:
    """Compute 4-block augmented OT cost matrix on GPU."""
    N = len(src_positions)
    
    pos_diff = src_positions[:, None] - tgt_positions[None, :]
    pos_cost = lambda_pos * (pos_diff ** 2)
    
    src_norm_sq = torch.sum(src_embeddings ** 2, dim=1)
    tgt_norm_sq = torch.sum(tgt_embeddings ** 2, dim=1)
    semantic_cost = src_norm_sq[:, None] + tgt_norm_sq[None, :] - 2 * (src_embeddings @ tgt_embeddings.T)
    semantic_cost = torch.clamp(semantic_cost, min=0)
    
    src_active_2d = src_active[:, None]
    tgt_active_2d = tgt_active[None, :]
    
    mask_aa = src_active_2d & tgt_active_2d
    mask_av = src_active_2d & ~tgt_active_2d
    mask_va = ~src_active_2d & tgt_active_2d
    mask_vv = ~src_active_2d & ~tgt_active_2d
    
    C = torch.zeros((N, N), dtype=torch.float64, device=src_positions.device)
    
    C = C + mask_aa * (semantic_cost + pos_cost)
    C = C + mask_av * (alpha_delete + pos_cost)
    
    creation_cost = alpha_create + beta_create * tgt_norm_sq[None, :]
    C = C + mask_va * (creation_cost + pos_cost)
    C = C + mask_vv * (0.01 * pos_cost)
    
    return C


def compute_augmented_ot_coupling_gpu(
    src_state: np.ndarray,
    tgt_state: np.ndarray,
    src_positions: np.ndarray,
    tgt_positions: np.ndarray,
    src_active: np.ndarray,
    tgt_active: np.ndarray,
    reg: float = 0.1,
    lambda_pos: float = 1.0,
    alpha_delete: float = 1.0,
    alpha_create: float = 1.0,
    beta_create: float = 0.1,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute augmented OT coupling on GPU."""
    if device is None:
        device = DEVICE
    
    N = src_state.shape[0]
    d = src_state.shape[1] - 1
    
    src_emb = torch.tensor(src_state[:, :d], dtype=torch.float64, device=device)
    tgt_emb = torch.tensor(tgt_state[:, :d], dtype=torch.float64, device=device)
    src_pos_t = torch.tensor(src_positions, dtype=torch.float64, device=device)
    tgt_pos_t = torch.tensor(tgt_positions, dtype=torch.float64, device=device)
    src_act_t = torch.tensor(src_active, dtype=torch.bool, device=device)
    tgt_act_t = torch.tensor(tgt_active, dtype=torch.bool, device=device)
    
    C = compute_augmented_ot_cost_gpu(
        src_pos_t, tgt_pos_t,
        src_act_t, tgt_act_t,
        src_emb, tgt_emb,
        lambda_pos=lambda_pos,
        alpha_delete=alpha_delete,
        alpha_create=alpha_create,
        beta_create=beta_create,
    )
    
    C = C / (C.max() + 1e-8)
    
    a = torch.ones(N, dtype=torch.float64, device=device) / N
    b = torch.ones(N, dtype=torch.float64, device=device) / N
    
    coupling = ot.bregman.sinkhorn_log(a, b, C, reg=reg, numItermax=200)
    perm = torch.argmax(coupling, dim=1)
    
    return coupling.float().cpu().numpy(), perm.cpu().numpy()


# =============================================================================
# ODE Integration
# =============================================================================

def integrate_augmented_flow(
    model: AugmentedVelocityField,
    x0: torch.Tensor,
    num_steps: int = 20,
    direction: str = 'forward',
) -> torch.Tensor:
    """Integrate the augmented velocity field using Euler method.
    
    Args:
        model: Velocity field network
        x0: Initial state [B, N, d+1]
        num_steps: Number of integration steps
        direction: 'forward' (t=0→1) or 'backward' (t=1→0)
    
    Returns:
        Final state [B, N, d+1]
    """
    dt = 1.0 / float(num_steps)
    x = x0.clone()
    device = x.device
    B = x.shape[0]
    
    if direction == 'forward':
        for k in range(num_steps):
            t = torch.full((B,), float(k) * dt, device=device)
            with torch.no_grad():
                v = model(x, t)
            x = x + dt * v
    else:
        for k in range(num_steps, 0, -1):
            t = torch.full((B,), float(k) * dt, device=device)
            with torch.no_grad():
                v = model(x, t)
            x = x - dt * v
    
    return x


# =============================================================================
# Metrics Computation
# =============================================================================

def extract_active_embeddings(
    state: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract active token embeddings from augmented state.
    
    Args:
        state: [B, N, d+1] augmented state
        threshold: Existence threshold
    
    Returns:
        embeddings: [B, N, d] content embeddings (all tokens)
        mask: [B, N] boolean mask for active tokens
    """
    existence = state[:, :, -1]  # [B, N]
    mask = existence > threshold
    embeddings = state[:, :, :-1]  # [B, N, d]
    return embeddings, mask


def compute_cosine_similarity_batch(
    transported: torch.Tensor,
    target: torch.Tensor,
    trans_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute OT-aligned cosine similarity between transported and target.
    
    Computes fresh OT coupling between transported and target active tokens,
    then measures cosine similarity between matched pairs.
    
    Args:
        transported: [B, N, d] transported embeddings
        target: [B, N, d] target embeddings
        trans_mask: [B, N] mask for transported active tokens
        tgt_mask: [B, N] mask for target active tokens
    
    Returns:
        Dictionary with mean, std, median cosine similarities
    """
    B, N, d = transported.shape
    
    all_sims = []
    
    for b in range(B):
        # Get active indices
        trans_active_idx = trans_mask[b].nonzero(as_tuple=True)[0]
        tgt_active_idx = tgt_mask[b].nonzero(as_tuple=True)[0]
        
        if len(trans_active_idx) == 0 or len(tgt_active_idx) == 0:
            continue
        
        # Get active embeddings
        trans_active = transported[b, trans_active_idx].cpu().numpy()  # [n1, d]
        tgt_active = target[b, tgt_active_idx].cpu().numpy()  # [n2, d]
        
        n1, n2 = len(trans_active), len(tgt_active)
        
        # Compute OT coupling between transported and target
        # Use squared Euclidean cost
        M = ot.dist(trans_active, tgt_active, metric='sqeuclidean')
        
        # Normalize cost matrix for numerical stability
        M = M / (M.max() + 1e-8)
        
        # Uniform weights
        a = np.ones(n1) / n1
        b_weights = np.ones(n2) / n2
        
        # Compute OT coupling (log-stabilized sinkhorn for numerical stability)
        try:
            coupling = ot.bregman.sinkhorn_log(a, b_weights, M, reg=0.1, numItermax=100)
        except Exception:
            continue
        
        # Get hard assignment: for each transported token, find its matched target
        # Use argmax of coupling rows
        assignments = np.argmax(coupling, axis=1)  # [n1]
        
        # Compute cosine similarity for each matched pair
        trans_norm = trans_active / (np.linalg.norm(trans_active, axis=1, keepdims=True) + 1e-8)
        tgt_norm = tgt_active / (np.linalg.norm(tgt_active, axis=1, keepdims=True) + 1e-8)
        
        for i in range(n1):
            j = assignments[i]
            sim = np.dot(trans_norm[i], tgt_norm[j])
            all_sims.append(sim)
    
    if len(all_sims) == 0:
        return {'mean': 0.0, 'std': 0.0, 'median': 0.0}
    
    all_sims = np.array(all_sims)
    return {
        'mean': float(np.mean(all_sims)),
        'std': float(np.std(all_sims)),
        'median': float(np.median(all_sims)),
    }


def compute_wasserstein2_batch(
    transported: torch.Tensor,
    target: torch.Tensor,
    trans_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute Wasserstein-2 distance between transported and target distributions.
    
    Args:
        transported: [B, N, d] transported embeddings
        target: [B, N, d] target embeddings
        trans_mask: [B, N] mask for transported active tokens
        tgt_mask: [B, N] mask for target active tokens
    
    Returns:
        Dictionary with mean, std W2 distances
    """
    B = transported.shape[0]
    w2_distances = []
    
    for b in range(B):
        trans_active_idx = trans_mask[b].nonzero(as_tuple=True)[0]
        tgt_active_idx = tgt_mask[b].nonzero(as_tuple=True)[0]
        
        if len(trans_active_idx) == 0 or len(tgt_active_idx) == 0:
            continue
        
        trans_active = transported[b, trans_active_idx].cpu().numpy()  # [n1, d]
        tgt_active = target[b, tgt_active_idx].cpu().numpy()  # [n2, d]
        
        n1, n2 = len(trans_active), len(tgt_active)
        
        # Uniform weights
        a = np.ones(n1) / n1
        b_weights = np.ones(n2) / n2
        
        # Euclidean cost matrix
        M = ot.dist(trans_active, tgt_active, metric='sqeuclidean')
        
        # Compute W2 (square root of OT cost with squared distances)
        try:
            w2_sq = ot.emd2(a, b_weights, M)
            w2 = np.sqrt(max(0, w2_sq))
            w2_distances.append(w2)
        except Exception:
            continue
    
    if len(w2_distances) == 0:
        return {'mean': 0.0, 'std': 0.0}
    
    w2_distances = np.array(w2_distances)
    return {
        'mean': float(np.mean(w2_distances)),
        'std': float(np.std(w2_distances)),
    }


# =============================================================================
# Data Loading
# =============================================================================

def load_embeddings_from_end(
    parquet_path: str,
    num_samples: int = 500,
    max_seq_len: int = 256,
) -> List[np.ndarray]:
    """Load embeddings from the last N samples of a parquet file."""
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    
    # Calculate which row groups contain the last N samples
    start_idx = max(0, total_rows - num_samples)
    
    embeddings = []
    current_row = 0
    
    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(rg_idx)
        rg_start = current_row
        rg_end = current_row + rg.num_rows
        
        if rg_end > start_idx:
            # This row group contains some of our samples
            table = pf.read_row_group(rg_idx, columns=['hidden'])
            hidden_list = table['hidden'].to_pylist()
            
            for i, hidden in enumerate(hidden_list):
                row_idx = rg_start + i
                if row_idx >= start_idx:
                    # Parse embedding
                    if isinstance(hidden, np.ndarray):
                        emb = hidden.astype(np.float32)
                    else:
                        emb = np.array(hidden, dtype=np.float32)
                    
                    if emb.ndim == 1:
                        emb = emb.reshape(1, -1)
                    
                    embeddings.append(emb[:max_seq_len])
        
        current_row = rg_end
        
        if len(embeddings) >= num_samples:
            break
    
    return embeddings[:num_samples]


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate(
    model: AugmentedVelocityField,
    nl_embeddings: List[np.ndarray],
    lean_embeddings: List[np.ndarray],
    N_max: int,
    d: int,
    num_integration_steps: int = 20,
    ot_params: Dict = None,
    batch_size: int = 16,
) -> Dict:
    """Run full evaluation.
    
    Returns:
        Dictionary with all metrics
    """
    if ot_params is None:
        ot_params = {
            'reg': 0.05,
            'lambda_pos': 1.0,
            'alpha_delete': 1.0,
            'alpha_create': 1.0,
            'beta_create': 0.1,
        }
    
    model.eval()
    
    num_samples = min(len(nl_embeddings), len(lean_embeddings))
    print(f"Evaluating on {num_samples} samples...")
    
    # Metrics accumulators
    metrics = {
        'nl_to_lean': {'cos_sim': [], 'w2': []},
        'lean_to_nl': {'cos_sim': [], 'w2': []},
        'circle_nl': {'cos_sim': [], 'w2': []},  # NL → Lean → NL
        'circle_lean': {'cos_sim': [], 'w2': []},  # Lean → NL → Lean
    }
    
    # Process in batches
    for batch_start in tqdm(range(0, num_samples, batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, num_samples)
        
        # Create augmented states for this batch
        nl_states = []
        lean_states = []
        nl_pos_list, lean_pos_list = [], []
        nl_active_list, lean_active_list = [], []
        
        for i in range(batch_start, batch_end):
            nl_state, nl_pos, nl_active = create_augmented_state(nl_embeddings[i], N_max, d)
            lean_state, lean_pos, lean_active = create_augmented_state(lean_embeddings[i], N_max, d)
            
            nl_states.append(nl_state)
            lean_states.append(lean_state)
            nl_pos_list.append(nl_pos)
            lean_pos_list.append(lean_pos)
            nl_active_list.append(nl_active)
            lean_active_list.append(lean_active)
        
        # Convert to tensors
        nl_batch = torch.tensor(np.stack(nl_states), dtype=torch.float32, device=DEVICE)
        lean_batch = torch.tensor(np.stack(lean_states), dtype=torch.float32, device=DEVICE)
        
        B = nl_batch.shape[0]
        
        # Get masks for active tokens
        nl_mask = nl_batch[:, :, -1] > 0.5
        lean_mask = lean_batch[:, :, -1] > 0.5
        
        # =================================================================
        # 1. NL → Lean transport
        # =================================================================
        transported_nl_to_lean = integrate_augmented_flow(
            model, nl_batch, num_steps=num_integration_steps, direction='forward'
        )
        
        # Extract embeddings
        trans_emb, trans_mask = extract_active_embeddings(transported_nl_to_lean)
        tgt_emb = lean_batch[:, :, :-1]
        
        # Compute metrics
        cos_metrics = compute_cosine_similarity_batch(trans_emb, tgt_emb, trans_mask, lean_mask)
        w2_metrics = compute_wasserstein2_batch(trans_emb, tgt_emb, trans_mask, lean_mask)
        
        metrics['nl_to_lean']['cos_sim'].append(cos_metrics)
        metrics['nl_to_lean']['w2'].append(w2_metrics)
        
        # =================================================================
        # 2. Lean → NL transport
        # =================================================================
        transported_lean_to_nl = integrate_augmented_flow(
            model, lean_batch, num_steps=num_integration_steps, direction='backward'
        )
        
        trans_emb, trans_mask = extract_active_embeddings(transported_lean_to_nl)
        tgt_emb = nl_batch[:, :, :-1]
        
        cos_metrics = compute_cosine_similarity_batch(trans_emb, tgt_emb, trans_mask, nl_mask)
        w2_metrics = compute_wasserstein2_batch(trans_emb, tgt_emb, trans_mask, nl_mask)
        
        metrics['lean_to_nl']['cos_sim'].append(cos_metrics)
        metrics['lean_to_nl']['w2'].append(w2_metrics)
        
        # =================================================================
        # 3. Circle: NL → Lean → NL
        # =================================================================
        circle_nl = integrate_augmented_flow(
            model, transported_nl_to_lean, num_steps=num_integration_steps, direction='backward'
        )
        
        trans_emb, trans_mask = extract_active_embeddings(circle_nl)
        tgt_emb = nl_batch[:, :, :-1]
        
        cos_metrics = compute_cosine_similarity_batch(trans_emb, tgt_emb, trans_mask, nl_mask)
        w2_metrics = compute_wasserstein2_batch(trans_emb, tgt_emb, trans_mask, nl_mask)
        
        metrics['circle_nl']['cos_sim'].append(cos_metrics)
        metrics['circle_nl']['w2'].append(w2_metrics)
        
        # =================================================================
        # 4. Circle: Lean → NL → Lean
        # =================================================================
        circle_lean = integrate_augmented_flow(
            model, transported_lean_to_nl, num_steps=num_integration_steps, direction='forward'
        )
        
        trans_emb, trans_mask = extract_active_embeddings(circle_lean)
        tgt_emb = lean_batch[:, :, :-1]
        
        cos_metrics = compute_cosine_similarity_batch(trans_emb, tgt_emb, trans_mask, lean_mask)
        w2_metrics = compute_wasserstein2_batch(trans_emb, tgt_emb, trans_mask, lean_mask)
        
        metrics['circle_lean']['cos_sim'].append(cos_metrics)
        metrics['circle_lean']['w2'].append(w2_metrics)
    
    # Aggregate metrics
    results = {}
    for direction in ['nl_to_lean', 'lean_to_nl', 'circle_nl', 'circle_lean']:
        cos_means = [m['mean'] for m in metrics[direction]['cos_sim']]
        w2_means = [m['mean'] for m in metrics[direction]['w2']]
        
        results[direction] = {
            'cosine_similarity': {
                'mean': float(np.mean(cos_means)),
                'std': float(np.std(cos_means)),
            },
            'wasserstein_2': {
                'mean': float(np.mean(w2_means)),
                'std': float(np.std(w2_means)),
            },
        }
    
    return results


def load_model(checkpoint_path: str, config: Dict = None) -> AugmentedVelocityField:
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    state = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Get config from checkpoint or use provided config
    if config is None:
        # Try to load config from same directory
        config_path = Path(checkpoint_path).parent.parent / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            # Default config
            config = {
                'hidden_dim': 2048,
                'time_embed_dim': 256,
                'num_layers': 6,
                'num_heads': 8,
            }
    
    model = AugmentedVelocityField(
        hidden_dim=config.get('hidden_dim', 2048),
        time_embed_dim=config.get('time_embed_dim', 256),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
    )
    
    # Load state dict
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    
    model.to(DEVICE)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params:,} parameters")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Evaluate Augmented OT-CFM Model")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--nl-path", type=str, 
                        default="outputs/kimina17_all_nl_embeddings.parquet",
                        help="Path to NL embeddings parquet")
    parser.add_argument("--lean-path", type=str,
                        default="outputs/kimina17_all_lean_embeddings.parquet",
                        help="Path to Lean embeddings parquet")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of samples to evaluate (from end of files)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Number of ODE integration steps")
    parser.add_argument("--n-max", type=int, default=256,
                        help="Fixed container size")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    
    # OT parameters
    parser.add_argument("--ot-reg", type=float, default=0.05)
    parser.add_argument("--lambda-pos", type=float, default=1.0)
    parser.add_argument("--alpha-delete", type=float, default=1.0)
    parser.add_argument("--alpha-create", type=float, default=1.0)
    parser.add_argument("--beta-create", type=float, default=0.1)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AUGMENTED OT-CFM EVALUATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"NL embeddings: {args.nl_path}")
    print(f"Lean embeddings: {args.lean_path}")
    print(f"Samples: {args.num_samples} (from end)")
    print(f"Integration steps: {args.num_steps}")
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)
    
    # Load model
    model, config = load_model(args.model)
    
    # Get embedding dimension from config
    d = config.get('hidden_dim', 2048)
    
    # Load embeddings
    print(f"\nLoading last {args.num_samples} NL embeddings...")
    nl_embeddings = load_embeddings_from_end(args.nl_path, args.num_samples, args.n_max)
    print(f"Loaded {len(nl_embeddings)} NL samples")
    
    print(f"\nLoading last {args.num_samples} Lean embeddings...")
    lean_embeddings = load_embeddings_from_end(args.lean_path, args.num_samples, args.n_max)
    print(f"Loaded {len(lean_embeddings)} Lean samples")
    
    # OT parameters
    ot_params = {
        'reg': args.ot_reg,
        'lambda_pos': args.lambda_pos,
        'alpha_delete': args.alpha_delete,
        'alpha_create': args.alpha_create,
        'beta_create': args.beta_create,
    }
    
    # Run evaluation
    print("\n" + "=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)
    
    results = evaluate(
        model=model,
        nl_embeddings=nl_embeddings,
        lean_embeddings=lean_embeddings,
        N_max=args.n_max,
        d=d,
        num_integration_steps=args.num_steps,
        ot_params=ot_params,
        batch_size=args.batch_size,
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for direction, metrics in results.items():
        print(f"\n{direction.upper().replace('_', ' ')}:")
        print(f"  Cosine Similarity: {metrics['cosine_similarity']['mean']:.4f} ± {metrics['cosine_similarity']['std']:.4f}")
        print(f"  Wasserstein-2:     {metrics['wasserstein_2']['mean']:.4f} ± {metrics['wasserstein_2']['std']:.4f}")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        # Save next to model checkpoint
        model_dir = Path(args.model).parent.parent
        output_path = model_dir / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    results['config'] = {
        'model': args.model,
        'nl_path': args.nl_path,
        'lean_path': args.lean_path,
        'num_samples': args.num_samples,
        'num_steps': args.num_steps,
        'n_max': args.n_max,
        'ot_params': ot_params,
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
