"""Online Augmented OT-CFM Training - NO PREPROCESSING REQUIRED.

This script streams parquet files and computes augmented OT on-the-fly.
Implements the FULL Augmented OT-CFM framework:

1. EXISTENCE CHANNEL: Each token has e_i ∈ {0, 1}
   - Real tokens: e_i = 1
   - Void tokens: e_i = 0

2. DISTRIBUTED VOID POSITIONING: Voids uniformly distributed in position space
   - Active tokens at positions 1/L, 2/L, ..., 1
   - Voids fill gaps uniformly

3. CONTENT AUGMENTATION: a_i = h_i + P(p_i)
   - Content = hidden state + positional embedding

4. 4-BLOCK COST MATRIX:
   - Active↔Active: ||a_i - b_j||^2 + λ(p_i - p_j)^2
   - Active↔Void: α_delete + λ(p_i - p_j)^2
   - Void↔Active: α_create + β||b_j||^2 + λ(p_i - p_j)^2
   - Void↔Void: ≈ 0

5. TRANSFORMER VELOCITY FIELD:
   - Input: [B, N_max, d+1] (content + existence)
   - Time conditioning via embedding addition
   - Output: [B, N_max, d+1] velocity

Usage:
  python -m scripts.train_augmented_ot_online --config project_config.yaml --max-samples 5000 --no-wandb
"""

import argparse
import os
import gc
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Iterator, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import ot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(ROOT)

from src.io_utils import ensure_dir

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Global device
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Sinusoidal Embeddings (from compute_augmented_ot.py)
# =============================================================================

def get_sinusoidal_embedding(positions: np.ndarray, d: int) -> np.ndarray:
    """Generate sinusoidal positional embeddings.
    
    Args:
        positions: [N] array of positions in [0, 1]
        d: embedding dimension
    
    Returns:
        [N, d] positional embeddings
    """
    N = len(positions)
    pe = np.zeros((N, d), dtype=np.float32)
    
    div_term = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))
    
    for i, p in enumerate(positions):
        pe[i, 0::2] = np.sin(p * np.pi * div_term)
        pe[i, 1::2] = np.cos(p * np.pi * div_term[:d//2] if d % 2 == 0 else div_term[:(d+1)//2])
    
    return pe


# =============================================================================
# Augmented State Creation (from compute_augmented_ot.py)
# =============================================================================

def create_augmented_state(
    embeddings: np.ndarray,  # [L, d] active embeddings
    N_max: int,
    d: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create augmented state with existence channel and distributed voids.
    
    Returns:
        state: [N_max, d+1] augmented state (content + existence)
        positions: [N_max] positions in [0, 1]
        is_active: [N_max] boolean mask
    """
    L = min(embeddings.shape[0], N_max)
    embeddings = embeddings[:L]
    n_void = N_max - L
    
    # Distributed positioning
    # Active tokens: positions 1/L, 2/L, ..., 1
    active_positions = np.linspace(1/L, 1.0, L) if L > 0 else np.array([])
    
    # Void tokens: distributed uniformly in [0, 1]
    if n_void > 0:
        void_positions = np.linspace(1/(n_void+1), n_void/(n_void+1), n_void)
    else:
        void_positions = np.array([])
    
    # Interleave by sorting all positions
    all_positions = np.concatenate([active_positions, void_positions])
    all_indices = np.argsort(all_positions)
    
    # Build state array
    state = np.zeros((N_max, d + 1), dtype=np.float32)
    positions = np.zeros(N_max, dtype=np.float32)
    is_active = np.zeros(N_max, dtype=bool)
    
    # Get positional embeddings
    active_pe = get_sinusoidal_embedding(active_positions, d) if L > 0 else np.zeros((0, d))
    void_pe = get_sinusoidal_embedding(void_positions, d) if n_void > 0 else np.zeros((0, d))
    
    active_idx = 0
    void_idx = 0
    
    for out_idx, orig_idx in enumerate(all_indices):
        if orig_idx < L:
            # Active token: content = hidden + positional
            state[out_idx, :d] = embeddings[active_idx] + active_pe[active_idx]
            state[out_idx, d] = 1.0  # existence = 1
            positions[out_idx] = active_positions[active_idx]
            is_active[out_idx] = True
            active_idx += 1
        else:
            # Void token: content = positional only
            state[out_idx, :d] = void_pe[void_idx]  # zero content + position
            state[out_idx, d] = 0.0  # existence = 0
            positions[out_idx] = void_positions[void_idx]
            is_active[out_idx] = False
            void_idx += 1
    
    return state, positions, is_active


# =============================================================================
# 4-Block Cost Matrix (from compute_augmented_ot.py - GPU version)
# =============================================================================

def compute_augmented_ot_cost_gpu(
    src_positions: torch.Tensor,  # [N_max]
    tgt_positions: torch.Tensor,  # [N_max]
    src_active: torch.Tensor,     # [N_max] bool
    tgt_active: torch.Tensor,     # [N_max] bool
    src_embeddings: torch.Tensor, # [N_max, d] (without existence channel)
    tgt_embeddings: torch.Tensor, # [N_max, d]
    lambda_pos: float = 1.0,    # positional penalty weight
    alpha_delete: float = 1.0,  # deletion cost
    alpha_create: float = 1.0,  # creation cost
    beta_create: float = 0.1,   # creation norm penalty
) -> torch.Tensor:
    """Compute 4-block augmented OT cost matrix on GPU.
    
    Blocks:
    - Active→Active: ||a_i - b_j||^2 + λ(p_i - p_j)^2
    - Active→Void: α_delete + λ(p_i - p_j)^2
    - Void→Active: α_create + β||b_j||^2 + λ(p_i - p_j)^2
    - Void→Void: ≈ 0
    """
    N = len(src_positions)
    
    # Positional cost matrix [N, N]
    pos_diff = src_positions[:, None] - tgt_positions[None, :]
    pos_cost = lambda_pos * (pos_diff ** 2)
    
    # Semantic cost matrix [N, N] (squared Euclidean)
    src_norm_sq = torch.sum(src_embeddings ** 2, dim=1)  # [N]
    tgt_norm_sq = torch.sum(tgt_embeddings ** 2, dim=1)  # [N]
    semantic_cost = src_norm_sq[:, None] + tgt_norm_sq[None, :] - 2 * (src_embeddings @ tgt_embeddings.T)
    semantic_cost = torch.clamp(semantic_cost, min=0)  # numerical stability
    
    # Block masks [N, N]
    src_active_2d = src_active[:, None]  # [N, 1]
    tgt_active_2d = tgt_active[None, :]  # [1, N]
    
    mask_aa = src_active_2d & tgt_active_2d      # Active → Active
    mask_av = src_active_2d & ~tgt_active_2d     # Active → Void
    mask_va = ~src_active_2d & tgt_active_2d     # Void → Active
    mask_vv = ~src_active_2d & ~tgt_active_2d    # Void → Void
    
    # Build cost matrix
    C = torch.zeros((N, N), dtype=torch.float64, device=src_positions.device)
    
    # Block 1: Active → Active
    C = C + mask_aa * (semantic_cost + pos_cost)
    
    # Block 2: Active → Void (deletion)
    C = C + mask_av * (alpha_delete + pos_cost)
    
    # Block 3: Void → Active (creation)
    creation_cost = alpha_create + beta_create * tgt_norm_sq[None, :]
    C = C + mask_va * (creation_cost + pos_cost)
    
    # Block 4: Void → Void (near zero)
    C = C + mask_vv * (0.01 * pos_cost)
    
    return C


def compute_augmented_ot_coupling_gpu(
    src_state: np.ndarray,      # [N_max, d+1]
    tgt_state: np.ndarray,      # [N_max, d+1]
    src_positions: np.ndarray,  # [N_max]
    tgt_positions: np.ndarray,  # [N_max]
    src_active: np.ndarray,     # [N_max] bool
    tgt_active: np.ndarray,     # [N_max] bool
    reg: float = 0.1,
    lambda_pos: float = 1.0,
    alpha_delete: float = 1.0,
    alpha_create: float = 1.0,
    beta_create: float = 0.1,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute augmented OT coupling on GPU.
    
    Returns:
        coupling: [N_max, N_max] transport plan
        perm: [N_max] permutation
    """
    if device is None:
        device = DEVICE
    
    N = src_state.shape[0]
    d = src_state.shape[1] - 1
    
    # Move to GPU
    src_emb = torch.tensor(src_state[:, :d], dtype=torch.float64, device=device)
    tgt_emb = torch.tensor(tgt_state[:, :d], dtype=torch.float64, device=device)
    src_pos_t = torch.tensor(src_positions, dtype=torch.float64, device=device)
    tgt_pos_t = torch.tensor(tgt_positions, dtype=torch.float64, device=device)
    src_act_t = torch.tensor(src_active, dtype=torch.bool, device=device)
    tgt_act_t = torch.tensor(tgt_active, dtype=torch.bool, device=device)
    
    # Compute cost matrix
    C = compute_augmented_ot_cost_gpu(
        src_pos_t, tgt_pos_t,
        src_act_t, tgt_act_t,
        src_emb, tgt_emb,
        lambda_pos=lambda_pos,
        alpha_delete=alpha_delete,
        alpha_create=alpha_create,
        beta_create=beta_create,
    )
    
    # Normalize
    C = C / (C.max() + 1e-8)
    
    # Uniform marginals
    a = torch.ones(N, dtype=torch.float64, device=device) / N
    b = torch.ones(N, dtype=torch.float64, device=device) / N
    
    # Sinkhorn
    coupling = ot.bregman.sinkhorn_log(a, b, C, reg=reg, numItermax=200)
    
    # Permutation
    perm = torch.argmax(coupling, dim=1)
    
    return coupling.float().cpu().numpy(), perm.cpu().numpy()


# =============================================================================
# Transformer Velocity Field (from train_augmented_ot.py)
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class AugmentedVelocityField(nn.Module):
    """Transformer-based velocity field for A-OT-CFM.
    
    Input: [B, N, d+1] (content + existence)
    Output: [B, N, d+1] (velocity for content and existence)
    
    We project d+1 to a transformer_dim that is divisible by num_heads.
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,      # d (content dimension)
        time_embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.aug_dim = hidden_dim + 1  # +1 for existence channel
        
        # Transformer dimension (must be divisible by num_heads)
        # Round up to nearest multiple of num_heads
        self.transformer_dim = ((self.aug_dim + num_heads - 1) // num_heads) * num_heads
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, self.transformer_dim)
        
        # Input projection (aug_dim -> transformer_dim)
        self.input_proj = nn.Linear(self.aug_dim, self.transformer_dim)
        self.input_norm = nn.LayerNorm(self.transformer_dim)
        
        # Transformer
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
        
        # Output projection (transformer_dim -> aug_dim)
        self.output_norm = nn.LayerNorm(self.transformer_dim)
        self.output_proj = nn.Linear(self.transformer_dim, self.aug_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, d+1] augmented state
            t: [B] time values in [0, 1]
        
        Returns:
            v: [B, N, d+1] velocity
        """
        B, N, _ = x.shape
        
        # Project input to transformer dim
        h = self.input_proj(x)  # [B, N, transformer_dim]
        
        # Time conditioning
        t_emb = self.time_embed(t)  # [B, time_embed_dim]
        t_emb = self.time_proj(t_emb)  # [B, transformer_dim]
        t_emb = t_emb.unsqueeze(1)  # [B, 1, transformer_dim]
        
        # Add time to all positions
        h = self.input_norm(h) + t_emb
        
        # Transformer
        h = self.transformer(h)
        
        # Output
        h = self.output_norm(h)
        v = self.output_proj(h)  # [B, N, aug_dim]
        
        return v


# =============================================================================
# Streaming Dataset
# =============================================================================

def parse_hidden_field(hidden, max_len: int = 256) -> np.ndarray:
    """Parse hidden field from parquet row."""
    if isinstance(hidden, np.ndarray):
        if hidden.dtype == object:
            emb = np.stack([np.asarray(x, dtype=np.float32) for x in hidden])
        else:
            emb = hidden.astype(np.float32)
    else:
        emb = np.asarray(hidden, dtype=np.float32)
    
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    
    return emb[:max_len]


class StreamingAugmentedDataset:
    """Streaming dataset that computes augmented OT on-the-fly."""
    
    def __init__(
        self,
        nl_path: str,
        lean_path: str,
        N_max: int = 256,
        batch_size: int = 32,
        max_samples: Optional[int] = None,
        # OT parameters
        ot_reg: float = 0.1,
        lambda_pos: float = 1.0,
        alpha_delete: float = 1.0,
        alpha_create: float = 1.0,
        beta_create: float = 0.1,
    ):
        self.nl_path = nl_path
        self.lean_path = lean_path
        self.N_max = N_max
        self.batch_size = batch_size
        self.max_samples = max_samples
        
        # OT params
        self.ot_reg = ot_reg
        self.lambda_pos = lambda_pos
        self.alpha_delete = alpha_delete
        self.alpha_create = alpha_create
        self.beta_create = beta_create
        
        # Open files
        self.nl_pf = pq.ParquetFile(nl_path)
        self.lean_pf = pq.ParquetFile(lean_path)
        
        self.num_row_groups = self.nl_pf.metadata.num_row_groups
        self.total_rows = self.nl_pf.metadata.num_rows
        
        if max_samples and max_samples < self.total_rows:
            self.effective_rows = max_samples
            self.effective_row_groups = min(self.num_row_groups, (max_samples + 127) // 128)
        else:
            self.effective_rows = self.total_rows
            self.effective_row_groups = self.num_row_groups
        
        # Get embedding dimension
        sample = self.nl_pf.read_row_group(0, columns=['hidden'])['hidden'][0].as_py()
        self.d = len(sample[0])
        
        print(f"StreamingAugmentedDataset:")
        print(f"  Total rows: {self.total_rows}, using: {self.effective_rows}")
        print(f"  N_max: {N_max}, d: {self.d}")
        print(f"  OT: reg={ot_reg}, λ_pos={lambda_pos}, α_del={alpha_delete}, α_cre={alpha_create}")
    
    def _process_row_group(self, rg_idx: int) -> List[Dict[str, np.ndarray]]:
        """Load and augment a row group."""
        nl_table = self.nl_pf.read_row_group(rg_idx)
        lean_table = self.lean_pf.read_row_group(rg_idx)
        
        nl_hidden_list = nl_table['hidden'].to_pylist()
        lean_hidden_list = lean_table['hidden'].to_pylist()
        
        samples = []
        for nl_hidden_raw, lean_hidden_raw in zip(nl_hidden_list, lean_hidden_list):
            nl_emb = parse_hidden_field(nl_hidden_raw, self.N_max)
            lean_emb = parse_hidden_field(lean_hidden_raw, self.N_max)
            
            # Create augmented states
            src_state, src_pos, src_active = create_augmented_state(nl_emb, self.N_max, self.d)
            tgt_state, tgt_pos, tgt_active = create_augmented_state(lean_emb, self.N_max, self.d)
            
            samples.append({
                'src_state': src_state,
                'tgt_state': tgt_state,
                'src_pos': src_pos,
                'tgt_pos': tgt_pos,
                'src_active': src_active,
                'tgt_active': tgt_active,
            })
        
        return samples
    
    def iterate_epoch(self, shuffle: bool = True) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Iterate through one epoch."""
        rg_indices = list(range(self.effective_row_groups))
        if shuffle:
            np.random.shuffle(rg_indices)
        
        current_samples = []
        total_yielded = 0
        
        for rg_idx in rg_indices:
            if self.max_samples and total_yielded >= self.max_samples:
                break
            
            samples = self._process_row_group(rg_idx)
            if shuffle:
                np.random.shuffle(samples)
            current_samples.extend(samples)
            
            while len(current_samples) >= self.batch_size:
                if self.max_samples and total_yielded >= self.max_samples:
                    break
                
                batch = current_samples[:self.batch_size]
                current_samples = current_samples[self.batch_size:]
                total_yielded += len(batch)
                
                yield self._collate_and_compute_ot(batch)
            
            del samples
            gc.collect()
        
        # Remaining samples
        if current_samples and (not self.max_samples or total_yielded < self.max_samples):
            yield self._collate_and_compute_ot(current_samples)
    
    def _collate_and_compute_ot(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate batch and compute OT couplings.
        
        Returns:
            src_states: [B, N, d+1]
            tgt_states: [B, N, d+1]
            couplings: [B, N, N] - full transport plan for barycentric projection
        """
        B = len(batch)
        
        src_states = []
        tgt_states = []
        couplings = []
        
        for sample in batch:
            # Compute OT coupling for this sample
            coupling, _ = compute_augmented_ot_coupling_gpu(
                sample['src_state'],
                sample['tgt_state'],
                sample['src_pos'],
                sample['tgt_pos'],
                sample['src_active'],
                sample['tgt_active'],
                reg=self.ot_reg,
                lambda_pos=self.lambda_pos,
                alpha_delete=self.alpha_delete,
                alpha_create=self.alpha_create,
                beta_create=self.beta_create,
            )
            
            src_states.append(sample['src_state'])
            tgt_states.append(sample['tgt_state'])
            couplings.append(coupling)
        
        return (
            torch.tensor(np.stack(src_states), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(tgt_states), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(couplings), dtype=torch.float32, device=DEVICE),
        )


# =============================================================================
# Training
# =============================================================================

def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


class TrainingLogger:
    """Logger for metrics and checkpoints."""
    
    def __init__(self, output_dir: str, config: Dict[str, Any], use_wandb: bool = False):
        self.output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"augmented_online_run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.metrics_history = []
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        if self.use_wandb:
            wandb.init(project="augmented-ot-cfm", config=config)
        
        print(f"Results: {self.run_dir}")
    
    def log(self, metrics: Dict[str, float], step: int, epoch: int):
        metrics['step'] = step
        metrics['epoch'] = epoch
        self.metrics_history.append(metrics)
        
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, step, is_best=False):
        ckpt = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        }
        
        torch.save(ckpt, self.checkpoints_dir / f"checkpoint_epoch{epoch}_step{step}.pt")
        torch.save(ckpt, self.checkpoints_dir / "latest_model.pt")
        
        if is_best:
            torch.save(ckpt, self.checkpoints_dir / "best_model.pt")
    
    def save_metrics(self):
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def finish(self):
        self.save_metrics()
        if self.use_wandb:
            wandb.finish()


def train(
    model: AugmentedVelocityField,
    dataset: StreamingAugmentedDataset,
    num_epochs: int,
    learning_rate: float,
    logger: TrainingLogger,
    log_every: int = 10,
    save_every: int = 500,
    grad_clip: float = 1.0,
):
    """Main training loop."""
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    steps_per_epoch = dataset.effective_rows // dataset.batch_size
    total_steps = num_epochs * steps_per_epoch
    
    warmup_steps = max(1, total_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    global_step = 0
    best_loss = float('inf')
    
    print(f"\nTraining:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Steps/epoch: ~{steps_per_epoch}")
    print(f"  Total steps: ~{total_steps}")
    print(f"  LR: {learning_rate}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_start = time.time()
        
        pbar = tqdm(dataset.iterate_epoch(shuffle=True), 
                   desc=f"Epoch {epoch+1}/{num_epochs}",
                   total=steps_per_epoch)
        
        for batch_idx, (src_state, tgt_state, coupling) in enumerate(pbar):
            B, N, aug_dim = src_state.shape
            
            optimizer.zero_grad()
            
            # Barycentric projection: compute expected target for each source token
            # tgt_aligned[b, i] = sum_j (coupling[b,i,j] / sum_k coupling[b,i,k]) * tgt_state[b, j]
            # This properly handles soft matching, creation, and deletion
            coupling_row_sum = coupling.sum(dim=-1, keepdim=True) + 1e-8  # [B, N, 1]
            weights = coupling / coupling_row_sum  # [B, N, N] normalized row-wise
            tgt_aligned = torch.bmm(weights, tgt_state)  # [B, N, d+1]
            
            # Sample time
            t = torch.rand(B, device=DEVICE)
            t_exp = t.view(B, 1, 1)
            
            # Interpolate: x_t = (1-t) * src + t * tgt_aligned
            x_t = (1 - t_exp) * src_state + t_exp * tgt_aligned
            
            # Target velocity: v = tgt_aligned - src
            v_target = tgt_aligned - src_state
            
            # Predict velocity
            v_pred = model(x_t, t)
            
            # MSE loss
            loss = F.mse_loss(v_pred, v_target)
            
            loss.backward()
            
            grad_norm = compute_grad_norm(model)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            global_step += 1
            
            avg_loss = np.mean(epoch_losses[-100:]) if epoch_losses else loss.item()
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.1e}',
            })
            
            if global_step % log_every == 0:
                logger.log({
                    'train/loss': loss.item(),
                    'train/avg_loss': avg_loss,
                    'train/grad_norm': grad_norm,
                    'train/lr': scheduler.get_last_lr()[0],
                }, step=global_step, epoch=epoch)
            
            if global_step % save_every == 0:
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                logger.save_checkpoint(model, optimizer, scheduler, epoch, global_step, is_best)
        
        epoch_time = time.time() - epoch_start
        epoch_avg = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} complete. Loss: {epoch_avg:.4f}, Time: {epoch_time/60:.1f}min")
        
        is_best = epoch_avg < best_loss
        if is_best:
            best_loss = epoch_avg
        logger.save_checkpoint(model, optimizer, scheduler, epoch, global_step, is_best)
    
    logger.finish()
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Results: {logger.run_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Online Augmented OT-CFM Training")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--nl-path", type=str, help="NL embeddings parquet")
    parser.add_argument("--lean-path", type=str, help="Lean embeddings parquet")
    parser.add_argument("--output-dir", type=str, default="outputs/neural_ot")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    
    # Model
    parser.add_argument("--n-max", type=int, default=256, help="Fixed container size")
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--time-embed-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    
    # OT parameters
    parser.add_argument("--ot-reg", type=float, default=0.1, help="Sinkhorn regularization")
    parser.add_argument("--lambda-pos", type=float, default=1.0, help="Positional penalty")
    parser.add_argument("--alpha-delete", type=float, default=1.0, help="Deletion cost")
    parser.add_argument("--alpha-create", type=float, default=1.0, help="Creation cost")
    parser.add_argument("--beta-create", type=float, default=0.1, help="Creation norm penalty")
    
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    
    # Load config
    if args.config and YAML_AVAILABLE:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        ot_cfg = cfg.get('neural_ot', {})
        
        if not args.nl_path:
            args.nl_path = ot_cfg.get('nl_embeddings')
        if not args.lean_path:
            args.lean_path = ot_cfg.get('lean_embeddings')
        if args.batch_size == 32:
            args.batch_size = ot_cfg.get('batch_size', 32)
        if args.num_epochs == 10:
            args.num_epochs = ot_cfg.get('num_epochs', 10)
        if args.learning_rate == 1e-4:
            args.learning_rate = float(ot_cfg.get('learning_rate', 1e-4))
        if args.hidden_dim == 2048:
            args.hidden_dim = ot_cfg.get('hidden_dim', 2048)
        if args.output_dir == "outputs/neural_ot":
            args.output_dir = ot_cfg.get('output_dir', 'outputs/neural_ot')
    
    print("=" * 70)
    print("AUGMENTED OT-CFM ONLINE TRAINING")
    print("=" * 70)
    print(f"NL: {args.nl_path}")
    print(f"Lean: {args.lean_path}")
    print(f"N_max: {args.n_max}, d: {args.hidden_dim}")
    print(f"Batch: {args.batch_size}, Epochs: {args.num_epochs}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"OT: reg={args.ot_reg}, λ_pos={args.lambda_pos}")
    print(f"    α_del={args.alpha_delete}, α_cre={args.alpha_create}, β_cre={args.beta_create}")
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)
    
    # Create model
    model = AugmentedVelocityField(
        hidden_dim=args.hidden_dim,
        time_embed_dim=args.time_embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Create dataset
    dataset = StreamingAugmentedDataset(
        nl_path=args.nl_path,
        lean_path=args.lean_path,
        N_max=args.n_max,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        ot_reg=args.ot_reg,
        lambda_pos=args.lambda_pos,
        alpha_delete=args.alpha_delete,
        alpha_create=args.alpha_create,
        beta_create=args.beta_create,
    )
    
    # Create logger
    config = vars(args)
    config['num_params'] = num_params
    logger = TrainingLogger(args.output_dir, config, use_wandb=not args.no_wandb)
    
    # Train
    train(
        model=model,
        dataset=dataset,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logger=logger,
        log_every=args.log_every,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
