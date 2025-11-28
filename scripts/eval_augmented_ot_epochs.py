"""Evaluate Augmented OT-CFM model across multiple epoch checkpoints.

This script:
1. Loads checkpoints from epoch 0 to epoch N
2. Runs evaluation on each checkpoint
3. Plots metrics across epochs

Usage:
  python -m scripts.eval_augmented_ot_epochs \
    --run-dir outputs/neural_ot/augmented_online_run_20251128_064541 \
    --nl-path outputs/kimina17_all_nl_embeddings.parquet \
    --lean-path outputs/kimina17_all_lean_embeddings.parquet \
    --num-samples 100 \
    --max-epoch 7
"""

import argparse
import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
from tqdm import tqdm
import matplotlib.pyplot as plt

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
# Helper Functions (from eval_augmented_ot.py)
# =============================================================================

def get_sinusoidal_embedding(positions: np.ndarray, d: int) -> np.ndarray:
    N = len(positions)
    pe = np.zeros((N, d), dtype=np.float32)
    div_term = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))
    for i, p in enumerate(positions):
        pe[i, 0::2] = np.sin(p * np.pi * div_term)
        pe[i, 1::2] = np.cos(p * np.pi * div_term[:d//2] if d % 2 == 0 else div_term[:(d+1)//2])
    return pe


def create_augmented_state(
    embeddings: np.ndarray,
    N_max: int,
    d: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def integrate_augmented_flow(
    model: AugmentedVelocityField,
    x0: torch.Tensor,
    num_steps: int = 20,
    direction: str = 'forward',
) -> torch.Tensor:
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


def extract_active_embeddings(state: torch.Tensor, threshold: float = 0.5):
    existence = state[:, :, -1]
    mask = existence > threshold
    embeddings = state[:, :, :-1]
    return embeddings, mask


def compute_cosine_similarity_batch(
    transported: torch.Tensor,
    target: torch.Tensor,
    trans_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
) -> Dict[str, float]:
    B, N, d = transported.shape
    all_sims = []
    
    for b in range(B):
        trans_active_idx = trans_mask[b].nonzero(as_tuple=True)[0]
        tgt_active_idx = tgt_mask[b].nonzero(as_tuple=True)[0]
        
        if len(trans_active_idx) == 0 or len(tgt_active_idx) == 0:
            continue
        
        trans_active = transported[b, trans_active_idx].cpu().numpy()
        tgt_active = target[b, tgt_active_idx].cpu().numpy()
        
        n1, n2 = len(trans_active), len(tgt_active)
        M = ot.dist(trans_active, tgt_active, metric='sqeuclidean')
        M = M / (M.max() + 1e-8)
        
        a = np.ones(n1) / n1
        b_weights = np.ones(n2) / n2
        
        try:
            coupling = ot.bregman.sinkhorn_log(a, b_weights, M, reg=0.1, numItermax=100)
        except Exception:
            continue
        
        assignments = np.argmax(coupling, axis=1)
        
        trans_norm = trans_active / (np.linalg.norm(trans_active, axis=1, keepdims=True) + 1e-8)
        tgt_norm = tgt_active / (np.linalg.norm(tgt_active, axis=1, keepdims=True) + 1e-8)
        
        for i in range(n1):
            j = assignments[i]
            sim = np.dot(trans_norm[i], tgt_norm[j])
            all_sims.append(sim)
    
    if len(all_sims) == 0:
        return {'mean': 0.0, 'std': 0.0}
    
    all_sims = np.array(all_sims)
    return {'mean': float(np.mean(all_sims)), 'std': float(np.std(all_sims))}


def compute_wasserstein2_batch(
    transported: torch.Tensor,
    target: torch.Tensor,
    trans_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
) -> Dict[str, float]:
    B = transported.shape[0]
    w2_distances = []
    
    for b in range(B):
        trans_active_idx = trans_mask[b].nonzero(as_tuple=True)[0]
        tgt_active_idx = tgt_mask[b].nonzero(as_tuple=True)[0]
        
        if len(trans_active_idx) == 0 or len(tgt_active_idx) == 0:
            continue
        
        trans_active = transported[b, trans_active_idx].cpu().numpy()
        tgt_active = target[b, tgt_active_idx].cpu().numpy()
        
        n1, n2 = len(trans_active), len(tgt_active)
        a = np.ones(n1) / n1
        b_weights = np.ones(n2) / n2
        
        M = ot.dist(trans_active, tgt_active, metric='sqeuclidean')
        
        try:
            w2_sq = ot.emd2(a, b_weights, M)
            w2 = np.sqrt(max(0, w2_sq))
            w2_distances.append(w2)
        except Exception:
            continue
    
    if len(w2_distances) == 0:
        return {'mean': 0.0, 'std': 0.0}
    
    w2_distances = np.array(w2_distances)
    return {'mean': float(np.mean(w2_distances)), 'std': float(np.std(w2_distances))}


# =============================================================================
# Data Loading
# =============================================================================

def load_embeddings_from_end(
    parquet_path: str,
    num_samples: int = 500,
    max_seq_len: int = 256,
) -> List[np.ndarray]:
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    start_idx = max(0, total_rows - num_samples)
    
    embeddings = []
    current_row = 0
    
    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(rg_idx)
        rg_start = current_row
        rg_end = current_row + rg.num_rows
        
        if rg_end > start_idx:
            table = pf.read_row_group(rg_idx, columns=['hidden'])
            hidden_list = table['hidden'].to_pylist()
            
            for i, hidden in enumerate(hidden_list):
                row_idx = rg_start + i
                if row_idx >= start_idx:
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
# Evaluation for single checkpoint
# =============================================================================

def evaluate_checkpoint(
    model: AugmentedVelocityField,
    nl_embeddings: List[np.ndarray],
    lean_embeddings: List[np.ndarray],
    N_max: int,
    d: int,
    num_integration_steps: int = 20,
    batch_size: int = 16,
) -> Dict:
    model.eval()
    num_samples = min(len(nl_embeddings), len(lean_embeddings))
    
    metrics = {
        'nl_to_lean': {'cos_sim': [], 'w2': []},
        'lean_to_nl': {'cos_sim': [], 'w2': []},
        'circle_nl': {'cos_sim': [], 'w2': []},
        'circle_lean': {'cos_sim': [], 'w2': []},
    }
    
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        
        nl_states = []
        lean_states = []
        
        for i in range(batch_start, batch_end):
            nl_state, _, _ = create_augmented_state(nl_embeddings[i], N_max, d)
            lean_state, _, _ = create_augmented_state(lean_embeddings[i], N_max, d)
            nl_states.append(nl_state)
            lean_states.append(lean_state)
        
        nl_batch = torch.tensor(np.stack(nl_states), dtype=torch.float32, device=DEVICE)
        lean_batch = torch.tensor(np.stack(lean_states), dtype=torch.float32, device=DEVICE)
        
        nl_mask = nl_batch[:, :, -1] > 0.5
        lean_mask = lean_batch[:, :, -1] > 0.5
        
        # NL → Lean
        transported_nl_to_lean = integrate_augmented_flow(model, nl_batch, num_steps=num_integration_steps, direction='forward')
        trans_emb, trans_mask = extract_active_embeddings(transported_nl_to_lean)
        tgt_emb = lean_batch[:, :, :-1]
        
        metrics['nl_to_lean']['cos_sim'].append(compute_cosine_similarity_batch(trans_emb, tgt_emb, trans_mask, lean_mask))
        metrics['nl_to_lean']['w2'].append(compute_wasserstein2_batch(trans_emb, tgt_emb, trans_mask, lean_mask))
        
        # Lean → NL
        transported_lean_to_nl = integrate_augmented_flow(model, lean_batch, num_steps=num_integration_steps, direction='backward')
        trans_emb, trans_mask = extract_active_embeddings(transported_lean_to_nl)
        tgt_emb = nl_batch[:, :, :-1]
        
        metrics['lean_to_nl']['cos_sim'].append(compute_cosine_similarity_batch(trans_emb, tgt_emb, trans_mask, nl_mask))
        metrics['lean_to_nl']['w2'].append(compute_wasserstein2_batch(trans_emb, tgt_emb, trans_mask, nl_mask))
        
        # Circle NL
        circle_nl = integrate_augmented_flow(model, transported_nl_to_lean, num_steps=num_integration_steps, direction='backward')
        trans_emb, trans_mask = extract_active_embeddings(circle_nl)
        tgt_emb = nl_batch[:, :, :-1]
        
        metrics['circle_nl']['cos_sim'].append(compute_cosine_similarity_batch(trans_emb, tgt_emb, trans_mask, nl_mask))
        metrics['circle_nl']['w2'].append(compute_wasserstein2_batch(trans_emb, tgt_emb, trans_mask, nl_mask))
        
        # Circle Lean
        circle_lean = integrate_augmented_flow(model, transported_lean_to_nl, num_steps=num_integration_steps, direction='forward')
        trans_emb, trans_mask = extract_active_embeddings(circle_lean)
        tgt_emb = lean_batch[:, :, :-1]
        
        metrics['circle_lean']['cos_sim'].append(compute_cosine_similarity_batch(trans_emb, tgt_emb, trans_mask, lean_mask))
        metrics['circle_lean']['w2'].append(compute_wasserstein2_batch(trans_emb, tgt_emb, trans_mask, lean_mask))
    
    # Aggregate
    results = {}
    for direction in ['nl_to_lean', 'lean_to_nl', 'circle_nl', 'circle_lean']:
        cos_means = [m['mean'] for m in metrics[direction]['cos_sim']]
        w2_means = [m['mean'] for m in metrics[direction]['w2']]
        
        results[direction] = {
            'cos_sim': float(np.mean(cos_means)),
            'w2': float(np.mean(w2_means)),
        }
    
    return results


def load_model(checkpoint_path: str, config: Dict) -> AugmentedVelocityField:
    state = torch.load(checkpoint_path, map_location=DEVICE)
    
    model = AugmentedVelocityField(
        hidden_dim=config.get('hidden_dim', 2048),
        time_embed_dim=config.get('time_embed_dim', 256),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
    )
    
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    
    model.to(DEVICE)
    model.eval()
    return model


def find_epoch_checkpoints(checkpoints_dir: Path, max_epoch: int) -> List[Tuple[int, Path]]:
    """Find checkpoint files for each epoch."""
    checkpoints = []
    
    for f in checkpoints_dir.glob("checkpoint_epoch*.pt"):
        match = re.search(r'checkpoint_epoch(\d+)_step\d+\.pt', f.name)
        if match:
            epoch = int(match.group(1))
            if epoch <= max_epoch:
                # Keep the latest step for each epoch
                checkpoints.append((epoch, f))
    
    # Group by epoch and take the one with highest step
    epoch_to_ckpt = {}
    for epoch, path in checkpoints:
        step_match = re.search(r'step(\d+)', path.name)
        step = int(step_match.group(1)) if step_match else 0
        
        if epoch not in epoch_to_ckpt or step > epoch_to_ckpt[epoch][1]:
            epoch_to_ckpt[epoch] = (path, step)
    
    result = [(epoch, path) for epoch, (path, _) in sorted(epoch_to_ckpt.items())]
    return result


def plot_metrics(all_results: Dict[int, Dict], output_path: Path):
    """Plot metrics across epochs."""
    epochs = sorted(all_results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('A-OT-CFM Evaluation Metrics Across Epochs', fontsize=14, fontweight='bold')
    
    directions = ['nl_to_lean', 'lean_to_nl', 'circle_nl', 'circle_lean']
    titles = ['NL → Lean', 'Lean → NL', 'Circle (NL→Lean→NL)', 'Circle (Lean→NL→Lean)']
    
    colors_cos = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    colors_w2 = ['#27ae60', '#2980b9', '#8e44ad', '#c0392b']
    
    # Plot 1: Cosine Similarity
    ax1 = axes[0, 0]
    for i, (direction, title) in enumerate(zip(directions, titles)):
        cos_vals = [all_results[e][direction]['cos_sim'] for e in epochs]
        ax1.plot(epochs, cos_vals, 'o-', label=title, color=colors_cos[i], linewidth=2, markersize=6)
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('OT-Aligned Cosine Similarity', fontsize=11)
    ax1.set_title('Cosine Similarity (higher is better)', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Plot 2: Wasserstein-2
    ax2 = axes[0, 1]
    for i, (direction, title) in enumerate(zip(directions, titles)):
        w2_vals = [all_results[e][direction]['w2'] for e in epochs]
        ax2.plot(epochs, w2_vals, 's-', label=title, color=colors_w2[i], linewidth=2, markersize=6)
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Wasserstein-2 Distance', fontsize=11)
    ax2.set_title('Wasserstein-2 Distance (lower is better)', fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    
    # Plot 3: Forward transport only
    ax3 = axes[1, 0]
    nl_cos = [all_results[e]['nl_to_lean']['cos_sim'] for e in epochs]
    lean_cos = [all_results[e]['lean_to_nl']['cos_sim'] for e in epochs]
    nl_w2 = [all_results[e]['nl_to_lean']['w2'] for e in epochs]
    lean_w2 = [all_results[e]['lean_to_nl']['w2'] for e in epochs]
    
    ax3.plot(epochs, nl_cos, 'o-', label='NL→Lean (cos)', color='#2ecc71', linewidth=2)
    ax3.plot(epochs, lean_cos, 's-', label='Lean→NL (cos)', color='#3498db', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Cosine Similarity', fontsize=11)
    ax3.set_title('Forward Transport: Cosine Similarity', fontsize=12)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(epochs)
    
    # Plot 4: Circle accuracy
    ax4 = axes[1, 1]
    circle_nl_cos = [all_results[e]['circle_nl']['cos_sim'] for e in epochs]
    circle_lean_cos = [all_results[e]['circle_lean']['cos_sim'] for e in epochs]
    
    ax4.plot(epochs, circle_nl_cos, 'o-', label='NL→Lean→NL', color='#9b59b6', linewidth=2)
    ax4.plot(epochs, circle_lean_cos, 's-', label='Lean→NL→Lean', color='#e74c3c', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Cosine Similarity', fontsize=11)
    ax4.set_title('Round-Trip (Circle) Accuracy', fontsize=12)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(epochs)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate A-OT-CFM across epochs")
    
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to run directory (contains checkpoints/ and config.json)")
    parser.add_argument("--nl-path", type=str, 
                        default="outputs/kimina17_all_nl_embeddings.parquet")
    parser.add_argument("--lean-path", type=str,
                        default="outputs/kimina17_all_lean_embeddings.parquet")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=20,
                        help="ODE integration steps")
    parser.add_argument("--n-max", type=int, default=256)
    parser.add_argument("--max-epoch", type=int, default=7,
                        help="Maximum epoch to evaluate (inclusive)")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    
    print("=" * 70)
    print("A-OT-CFM MULTI-EPOCH EVALUATION")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print(f"Max epoch: {args.max_epoch}")
    print(f"Samples: {args.num_samples}")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    
    # Load config
    config_path = run_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    d = config.get('hidden_dim', 2048)
    
    # Find checkpoints
    epoch_checkpoints = find_epoch_checkpoints(checkpoints_dir, args.max_epoch)
    print(f"\nFound {len(epoch_checkpoints)} epoch checkpoints:")
    for epoch, path in epoch_checkpoints:
        print(f"  Epoch {epoch}: {path.name}")
    
    # Load embeddings once
    print(f"\nLoading embeddings...")
    nl_embeddings = load_embeddings_from_end(args.nl_path, args.num_samples, args.n_max)
    lean_embeddings = load_embeddings_from_end(args.lean_path, args.num_samples, args.n_max)
    print(f"Loaded {len(nl_embeddings)} NL, {len(lean_embeddings)} Lean samples")
    
    # Evaluate each checkpoint
    all_results = {}
    
    for epoch, ckpt_path in tqdm(epoch_checkpoints, desc="Evaluating epochs"):
        print(f"\n--- Epoch {epoch} ---")
        
        model = load_model(str(ckpt_path), config)
        
        results = evaluate_checkpoint(
            model=model,
            nl_embeddings=nl_embeddings,
            lean_embeddings=lean_embeddings,
            N_max=args.n_max,
            d=d,
            num_integration_steps=args.num_steps,
            batch_size=args.batch_size,
        )
        
        all_results[epoch] = results
        
        print(f"  NL→Lean:  cos={results['nl_to_lean']['cos_sim']:.4f}, W2={results['nl_to_lean']['w2']:.2f}")
        print(f"  Lean→NL:  cos={results['lean_to_nl']['cos_sim']:.4f}, W2={results['lean_to_nl']['w2']:.2f}")
        print(f"  Circle NL:   cos={results['circle_nl']['cos_sim']:.4f}, W2={results['circle_nl']['w2']:.2f}")
        print(f"  Circle Lean: cos={results['circle_lean']['cos_sim']:.4f}, W2={results['circle_lean']['w2']:.2f}")
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Save results
    results_path = run_dir / f"epoch_eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Plot
    plot_path = run_dir / "epoch_metrics_plot.png"
    plot_metrics(all_results, plot_path)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Epoch':<8} {'NL→Lean cos':<14} {'Lean→NL cos':<14} {'Circle NL':<14} {'Circle Lean':<14}")
    print("-" * 70)
    for epoch in sorted(all_results.keys()):
        r = all_results[epoch]
        print(f"{epoch:<8} {r['nl_to_lean']['cos_sim']:<14.4f} {r['lean_to_nl']['cos_sim']:<14.4f} "
              f"{r['circle_nl']['cos_sim']:<14.4f} {r['circle_lean']['cos_sim']:<14.4f}")


if __name__ == "__main__":
    main()
