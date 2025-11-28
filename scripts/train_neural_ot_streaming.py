"""Train Neural OT with online OT coupling computation.

This script streams parquet files row-group by row-group, computes OT
couplings on-the-fly, and trains the model without precomputing all couplings.

Memory-efficient approach:
1. Load a batch of row groups (e.g., 2-4 row groups)
2. Compute OT couplings for that batch
3. Train model on those samples
4. Release memory and move to next batch

Usage with config file:
  python -m scripts.train_neural_ot_streaming --config project_config.yaml

Usage with explicit arguments:
  python -m scripts.train_neural_ot_streaming \
      --nl-path outputs/kimina17_all_nl_embeddings.parquet \
      --lean-path outputs/kimina17_all_lean_embeddings.parquet \
      --output-dir outputs/neural_ot/streaming_run \
      --batch-size 32 \
      --num-epochs 3
"""
import argparse
import os
import gc
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import ot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(ROOT)

from src.flows.neural_ot import NeuralOTFlow
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
# OT Computation Utilities
# =============================================================================

# Global device for GPU OT
_OT_DEVICE = None

def set_ot_device(use_gpu: bool = False):
    """Set device for OT computation."""
    global _OT_DEVICE
    if use_gpu and torch.cuda.is_available():
        _OT_DEVICE = torch.device('cuda')
        print("  🚀 Using GPU for OT computation")
    else:
        _OT_DEVICE = None
        print("  💻 Using CPU for OT computation (use --use-gpu-ot for faster)")


def compute_single_ot_coupling(
    x0: np.ndarray,
    x1: np.ndarray,
    cost: str = "euclidean",
    reg: float = 0.05,
    method: str = "sinkhorn"
) -> np.ndarray:
    """Compute OT coupling between two point clouds using POT."""
    L0, L1 = x0.shape[0], x1.shape[0]
    
    # Use GPU if available
    if _OT_DEVICE is not None and method == "sinkhorn":
        return _compute_ot_gpu(x0, x1, cost, reg)
    
    # CPU fallback
    a = np.ones(L0, dtype=np.float64) / L0
    b = np.ones(L1, dtype=np.float64) / L1
    
    if cost == "euclidean":
        C = ot.dist(x0.astype(np.float64), x1.astype(np.float64), metric='sqeuclidean')
    elif cost == "cosine":
        x0_norm = x0 / (np.linalg.norm(x0, axis=1, keepdims=True) + 1e-8)
        x1_norm = x1 / (np.linalg.norm(x1, axis=1, keepdims=True) + 1e-8)
        C = (1 - x0_norm @ x1_norm.T).astype(np.float64)
    else:
        raise ValueError(f"Unknown cost: {cost}")
    
    C = C / (C.max() + 1e-8)
    
    if method == "sinkhorn":
        P = ot.sinkhorn(a, b, C, reg=reg, numItermax=200, warn=False)
    elif method == "emd":
        P = ot.emd(a, b, C)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return P.astype(np.float32)


def _compute_ot_gpu(x0: np.ndarray, x1: np.ndarray, cost: str, reg: float) -> np.ndarray:
    """GPU-accelerated OT using PyTorch tensors."""
    L0, L1 = x0.shape[0], x1.shape[0]
    
    x0_t = torch.tensor(x0, dtype=torch.float32, device=_OT_DEVICE)
    x1_t = torch.tensor(x1, dtype=torch.float32, device=_OT_DEVICE)
    
    a = torch.ones(L0, dtype=torch.float64, device=_OT_DEVICE) / L0
    b = torch.ones(L1, dtype=torch.float64, device=_OT_DEVICE) / L1
    
    if cost == "euclidean":
        C = torch.cdist(x0_t.unsqueeze(0), x1_t.unsqueeze(0), p=2).squeeze(0) ** 2
    elif cost == "cosine":
        x0_norm = x0_t / (x0_t.norm(dim=1, keepdim=True) + 1e-8)
        x1_norm = x1_t / (x1_t.norm(dim=1, keepdim=True) + 1e-8)
        C = 1 - x0_norm @ x1_norm.T
    else:
        raise ValueError(f"Unknown cost: {cost}")
    
    C = C.double() / (C.max() + 1e-8)
    
    P = ot.sinkhorn(a, b, C, reg=reg, numItermax=200, warn=False)
    return P.cpu().numpy().astype(np.float32)


def compute_batch_ot_couplings_gpu(
    nl_batch: np.ndarray,
    lean_batch: np.ndarray, 
    nl_lens: np.ndarray,
    lean_lens: np.ndarray,
    cost: str = "euclidean",
    reg: float = 0.05,
) -> np.ndarray:
    """Compute OT couplings for a batch using GPU-accelerated Sinkhorn.
    
    Args:
        nl_batch: [B, L, D] padded NL embeddings
        lean_batch: [B, L, D] padded Lean embeddings
        nl_lens: [B] actual lengths for NL
        lean_lens: [B] actual lengths for Lean
        cost: "euclidean" or "cosine"
        reg: Sinkhorn regularization
        
    Returns:
        couplings: [B, L, L] OT coupling matrices
    """
    B, L, D = nl_batch.shape
    device = _OT_DEVICE if _OT_DEVICE is not None else torch.device('cpu')
    
    nl_t = torch.tensor(nl_batch, dtype=torch.float32, device=device)
    lean_t = torch.tensor(lean_batch, dtype=torch.float32, device=device)
    
    # Compute cost matrices for entire batch: [B, L, L]
    if cost == "euclidean":
        C = torch.cdist(nl_t, lean_t, p=2) ** 2
    elif cost == "cosine":
        nl_norm = nl_t / (nl_t.norm(dim=2, keepdim=True) + 1e-8)
        lean_norm = lean_t / (lean_t.norm(dim=2, keepdim=True) + 1e-8)
        C = 1 - torch.bmm(nl_norm, lean_norm.transpose(1, 2))
    else:
        raise ValueError(f"Unknown cost: {cost}")
    
    # Normalize each cost matrix
    C = C / (C.amax(dim=(1, 2), keepdim=True) + 1e-8)
    
    # Create masked uniform distributions based on actual lengths
    # For padded positions, set mass to 0
    a = torch.zeros(B, L, dtype=torch.float64, device=device)
    b = torch.zeros(B, L, dtype=torch.float64, device=device)
    
    for i in range(B):
        a[i, :nl_lens[i]] = 1.0 / nl_lens[i]
        b[i, :lean_lens[i]] = 1.0 / lean_lens[i]
    
    # Batched Sinkhorn
    C = C.double()
    P = _batched_sinkhorn(a, b, C, reg=reg, numItermax=200)
    
    return P.cpu().numpy().astype(np.float32)


def _batched_sinkhorn(a, b, C, reg, numItermax=200, tol=1e-9):
    """Batched Sinkhorn algorithm on GPU.
    
    Args:
        a: [B, L] source distribution
        b: [B, L] target distribution  
        C: [B, L, L] cost matrix
        reg: regularization
        numItermax: max iterations
        tol: convergence tolerance
        
    Returns:
        P: [B, L, L] transport plans
    """
    B, L = a.shape
    
    # Initialize dual variables
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    
    # Gibbs kernel
    K = torch.exp(-C / reg)
    
    for _ in range(numItermax):
        u_prev = u.clone()
        
        # Sinkhorn iterations
        v = b / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-16)
        u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-16)
        
        # Check convergence
        if torch.max(torch.abs(u - u_prev)) < tol:
            break
    
    # Compute transport plan
    P = u.unsqueeze(-1) * K * v.unsqueeze(1)
    
    return P


def _ot_worker(args):
    """Worker function for parallel OT computation (CPU only)."""
    idx, nl_emb, lean_emb, nl_len, lean_len, cost, reg, method = args
    coupling = compute_single_ot_coupling(nl_emb[:nl_len], lean_emb[:lean_len], cost=cost, reg=reg, method=method)
    # Pad coupling back to full size
    full_coupling = np.zeros((nl_emb.shape[0], lean_emb.shape[0]), dtype=np.float32)
    full_coupling[:nl_len, :lean_len] = coupling
    return idx, full_coupling


def parse_hidden_field(hidden, max_len: int = 256) -> Tuple[np.ndarray, int]:
    """Parse hidden field from parquet row into numpy array.
    
    Returns:
        (padded_embedding, actual_length)
    """
    if isinstance(hidden, np.ndarray):
        if hidden.dtype == object:
            emb = np.stack([np.asarray(x, dtype=np.float32) for x in hidden])
        else:
            emb = hidden.astype(np.float32)
    else:
        emb = np.asarray(hidden, dtype=np.float32)
    
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    
    actual_len = min(emb.shape[0], max_len)
    
    # Pad or truncate to max_len
    if emb.shape[0] > max_len:
        emb = emb[:max_len]
    elif emb.shape[0] < max_len:
        pad_len = max_len - emb.shape[0]
        emb = np.pad(emb, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
    
    return emb, actual_len


# =============================================================================
# Streaming Dataset for Row-Group-Based Loading
# =============================================================================

class StreamingParquetDataset:
    """Dataset that streams from parquet files row-group by row-group.
    
    This class does NOT implement __len__ since it streams data.
    Instead, use it as an iterator.
    """
    
    def __init__(
        self,
        nl_path: str,
        lean_path: str,
        max_len: int = 256,
        ot_cost: str = "euclidean",
        ot_reg: float = 0.05,
        ot_method: str = "sinkhorn",
        batch_size: int = 32,
        shuffle_within_rg: bool = True,
        rng_seed: int = 42,
        max_samples: int = None,  # Limit total samples
    ):
        self.nl_path = nl_path
        self.lean_path = lean_path
        self.max_len = max_len
        self.ot_cost = ot_cost
        self.ot_reg = ot_reg
        self.ot_method = ot_method
        self.batch_size = batch_size
        self.shuffle_within_rg = shuffle_within_rg
        self.rng = np.random.default_rng(rng_seed)
        self.max_samples = max_samples
        
        # Open files and get metadata
        self.nl_pf = pq.ParquetFile(nl_path)
        self.lean_pf = pq.ParquetFile(lean_path)
        
        self.num_row_groups = self.nl_pf.metadata.num_row_groups
        self.total_rows = self.nl_pf.metadata.num_rows
        
        # Limit samples if specified
        if max_samples and max_samples < self.total_rows:
            self.effective_rows = max_samples
            # Calculate how many row groups we need (each has ~128 samples)
            self.effective_row_groups = min(
                self.num_row_groups,
                (max_samples + 127) // 128  # Round up
            )
        else:
            self.effective_rows = self.total_rows
            self.effective_row_groups = self.num_row_groups
        
        # Verify alignment
        self._verify_alignment()
        
        print(f"StreamingParquetDataset initialized:")
        print(f"  Total rows in file: {self.total_rows}")
        print(f"  Using rows: {self.effective_rows}")
        print(f"  Using row groups: {self.effective_row_groups}/{self.num_row_groups}")
        print(f"  Batch size: {batch_size}")
        print(f"  Approx batches per epoch: {self.effective_rows // batch_size}")
    
    def _verify_alignment(self):
        """Verify that NL and Lean files are aligned."""
        assert self.nl_pf.metadata.num_row_groups == self.lean_pf.metadata.num_row_groups
        assert self.nl_pf.metadata.num_rows == self.lean_pf.metadata.num_rows
        
        # Check first and last row group IDs
        for rg_idx in [0, self.num_row_groups - 1]:
            nl_ids = self.nl_pf.read_row_group(rg_idx, columns=['id'])['id'].to_pylist()
            lean_ids = self.lean_pf.read_row_group(rg_idx, columns=['id'])['id'].to_pylist()
            assert nl_ids == lean_ids, f"ID mismatch in row group {rg_idx}"
    
    def _load_and_process_row_group(self, rg_idx: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
        """Load a row group and compute OT couplings.
        
        Returns list of (nl_emb, lean_emb, coupling, nl_len, lean_len) tuples.
        """
        # Read row groups
        nl_table = self.nl_pf.read_row_group(rg_idx)
        lean_table = self.lean_pf.read_row_group(rg_idx)
        
        nl_hidden = nl_table['hidden'].to_pylist()
        lean_hidden = lean_table['hidden'].to_pylist()
        
        # Parse embeddings (now returns padded arrays + lengths)
        nl_parsed = [parse_hidden_field(h, self.max_len) for h in nl_hidden]
        lean_parsed = [parse_hidden_field(h, self.max_len) for h in lean_hidden]
        
        nl_embeddings = np.stack([x[0] for x in nl_parsed])  # [N, L, D]
        nl_lens = np.array([x[1] for x in nl_parsed])  # [N]
        lean_embeddings = np.stack([x[0] for x in lean_parsed])  # [N, L, D]
        lean_lens = np.array([x[1] for x in lean_parsed])  # [N]
        
        # Compute OT couplings in batch (GPU if available)
        if _OT_DEVICE is not None and self.ot_method == "sinkhorn":
            couplings = compute_batch_ot_couplings_gpu(
                nl_embeddings, lean_embeddings, nl_lens, lean_lens,
                cost=self.ot_cost, reg=self.ot_reg
            )
        else:
            # CPU fallback - sequential
            couplings = []
            for i in range(len(nl_embeddings)):
                coupling = compute_single_ot_coupling(
                    nl_embeddings[i, :nl_lens[i]], lean_embeddings[i, :lean_lens[i]],
                    cost=self.ot_cost, reg=self.ot_reg, method=self.ot_method
                )
                # Pad coupling to full size
                full_coupling = np.zeros((self.max_len, self.max_len), dtype=np.float32)
                full_coupling[:nl_lens[i], :lean_lens[i]] = coupling
                couplings.append(full_coupling)
            couplings = np.stack(couplings)
        
        # Build samples
        samples = [
            (nl_embeddings[i], lean_embeddings[i], couplings[i], nl_lens[i], lean_lens[i])
            for i in range(len(nl_embeddings))
        ]
        
        return samples
    
    def iterate_epoch(self, shuffle_row_groups: bool = True) -> Iterator[Tuple]:
        """Iterate through one epoch, yielding batches.
        
        Yields:
            (nl_batch, lean_batch, coupling_batch, nl_lens, lean_lens) tensors
        """
        # Shuffle row group order (only use effective_row_groups)
        rg_indices = list(range(self.effective_row_groups))
        if shuffle_row_groups:
            self.rng.shuffle(rg_indices)
        
        current_samples = []
        total_samples_yielded = 0
        
        for rg_idx in rg_indices:
            # Check if we've hit the sample limit
            if self.max_samples and total_samples_yielded >= self.max_samples:
                break
            
            # Load and process row group
            samples = self._load_and_process_row_group(rg_idx)
            
            # Optionally shuffle within row group
            if self.shuffle_within_rg:
                self.rng.shuffle(samples)
            
            current_samples.extend(samples)
            
            # Yield batches
            while len(current_samples) >= self.batch_size:
                # Check sample limit before yielding
                if self.max_samples and total_samples_yielded >= self.max_samples:
                    break
                    
                batch = current_samples[:self.batch_size]
                current_samples = current_samples[self.batch_size:]
                total_samples_yielded += len(batch)
                yield self._collate_batch(batch)
            
            # Free memory
            del samples
            gc.collect()
        
        # Yield remaining samples (if under limit)
        if current_samples and (not self.max_samples or total_samples_yielded < self.max_samples):
            yield self._collate_batch(current_samples)
    
    def _collate_batch(self, batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
        """Collate fixed-length sequences with couplings.
        
        Each sample is (nl_emb, lean_emb, coupling, nl_len, lean_len).
        All arrays are already padded to max_len.
        """
        nl_list = [x[0] for x in batch]
        lean_list = [x[1] for x in batch]
        coupling_list = [x[2] for x in batch]
        nl_lens = [x[3] for x in batch]
        lean_lens = [x[4] for x in batch]
        
        # Stack - all already same shape (max_len, D) and (max_len, max_len)
        nl_batch = np.stack(nl_list)
        lean_batch = np.stack(lean_list)
        coupling_batch = np.stack(coupling_list)
        
        return (
            torch.from_numpy(nl_batch),
            torch.from_numpy(lean_batch),
            torch.from_numpy(coupling_batch),
            torch.tensor(nl_lens, dtype=torch.long),
            torch.tensor(lean_lens, dtype=torch.long),
        )
    
    def get_hidden_dim(self) -> int:
        """Get the embedding dimension from the first sample."""
        nl_table = self.nl_pf.read_row_group(0, columns=['hidden'])
        first_hidden = nl_table['hidden'][0].as_py()
        return len(first_hidden[0])


# =============================================================================
# Training Utilities
# =============================================================================

def compute_grad_norm(model):
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


class ResultsLogger:
    """Logger for metrics, checkpoints, and plots."""
    
    def __init__(self, output_dir, use_wandb=False, wandb_project=None, 
                 wandb_entity=None, config=None):
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.metrics_history = {}
        self.config = config or {}
        
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        
        if self.use_wandb:
            wandb.init(
                project=wandb_project or "neural-ot",
                entity=wandb_entity,
                config=self.config,
                dir=str(self.run_dir)
            )
        
        print(f"Results will be saved to: {self.run_dir}")
    
    def log_metrics(self, metrics, step, epoch=None):
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
        
        if self.use_wandb:
            log_dict = dict(metrics)
            if epoch is not None:
                log_dict["epoch"] = epoch
            wandb.log(log_dict, step=step)
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, step, metrics, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
        }
        
        # Save latest
        latest_path = self.checkpoints_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.checkpoints_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (loss: {metrics.get('train_loss', 'N/A'):.6f})")
    
    def generate_plots(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plots")
            return
        
        if "train/loss" in self.metrics_history:
            plt.figure(figsize=(10, 6))
            losses = np.array(self.metrics_history["train/loss"])
            plt.plot(losses, alpha=0.3, label="loss")
            if len(losses) > 50:
                window = min(50, len(losses) // 5)
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(losses)), smoothed, label=f"smoothed")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.plots_dir / "loss_curve.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to: {self.plots_dir}")
    
    def finish(self):
        with open(self.run_dir / "metrics_history.json", "w") as f:
            history = {k: [float(x) for x in v] for k, v in self.metrics_history.items()}
            json.dump(history, f, indent=2)
        
        self.generate_plots()
        
        loss_key = "train/loss"
        summary = {
            "total_steps": len(self.metrics_history.get(loss_key, [])),
            "final_loss": self.metrics_history[loss_key][-1] if loss_key in self.metrics_history else None,
            "best_loss": min(self.metrics_history[loss_key]) if loss_key in self.metrics_history else None,
            "run_dir": str(self.run_dir),
        }
        
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        if self.use_wandb:
            wandb.finish()
        
        print(f"\nTraining complete. Results saved to: {self.run_dir}")
        return summary


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Neural OT with streaming data")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (overrides other args)")
    parser.add_argument("--nl-path", type=str, default=None, help="Path to NL embeddings parquet")
    parser.add_argument("--lean-path", type=str, default=None, help="Path to Lean embeddings parquet")
    parser.add_argument("--output-dir", type=str, default="outputs/neural_ot", help="Output directory")
    
    # Training config
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to use (for faster training)")
    
    # Model config
    parser.add_argument("--hidden-dim", type=int, default=None, help="Hidden dim (auto-detect if not set)")
    parser.add_argument("--time-embed-dim", type=int, default=64, help="Time embedding dim")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--mlp-width", type=int, default=4096, help="MLP width")
    
    # OT config
    parser.add_argument("--ot-method", type=str, default="sinkhorn", choices=["sinkhorn", "emd"])
    parser.add_argument("--ot-cost", type=str, default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--ot-reg", type=float, default=0.1, help="Sinkhorn regularization")
    parser.add_argument("--use-gpu-ot", action="store_true", help="Use GPU for OT computation (much faster)")
    
    # Logging
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--wandb-project", type=str, default="neural-ot-streaming")
    parser.add_argument("--wandb-entity", type=str, default=None)
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        if not YAML_AVAILABLE:
            raise RuntimeError("YAML support required. Install with: pip install pyyaml")
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        neo_cfg = cfg.get("neural_ot", {})
        
        # Override args with config values
        args.nl_path = args.nl_path or neo_cfg.get("nl_embeddings")
        args.lean_path = args.lean_path or neo_cfg.get("lean_embeddings")
        args.output_dir = neo_cfg.get("output_dir", args.output_dir)
        args.batch_size = neo_cfg.get("batch_size", args.batch_size)
        args.num_epochs = neo_cfg.get("num_epochs", args.num_epochs)
        args.learning_rate = neo_cfg.get("learning_rate", args.learning_rate)
        args.max_len = neo_cfg.get("max_len", args.max_len)
        args.max_samples = args.max_samples or neo_cfg.get("max_samples")
        args.hidden_dim = args.hidden_dim or neo_cfg.get("hidden_dim")
        args.time_embed_dim = neo_cfg.get("time_embed_dim", args.time_embed_dim)
        args.num_layers = neo_cfg.get("num_layers", args.num_layers)
        args.mlp_width = neo_cfg.get("mlp_width", args.mlp_width)
        args.ot_method = neo_cfg.get("ot_method", args.ot_method)
        args.ot_cost = neo_cfg.get("ot_cost", args.ot_cost)
        args.ot_reg = neo_cfg.get("ot_reg", args.ot_reg)
        args.log_every = neo_cfg.get("log_every", args.log_every)
        args.save_every = neo_cfg.get("save_every", args.save_every)
    
    # Validate required paths
    if not args.nl_path or not args.lean_path:
        parser.error("--nl-path and --lean-path are required (or use --config)")
    
    # Ensure numeric types (config may load strings)
    args.batch_size = int(args.batch_size)
    args.num_epochs = int(args.num_epochs)
    args.learning_rate = float(args.learning_rate)
    args.max_len = int(args.max_len)
    args.time_embed_dim = int(args.time_embed_dim)
    args.num_layers = int(args.num_layers)
    args.mlp_width = int(args.mlp_width)
    args.ot_reg = float(args.ot_reg)
    args.log_every = int(args.log_every)
    args.save_every = int(args.save_every)
    if args.hidden_dim:
        args.hidden_dim = int(args.hidden_dim)
    if args.max_samples:
        args.max_samples = int(args.max_samples)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Neural OT Training (Streaming Mode)")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"NL embeddings: {args.nl_path}")
    print(f"Lean embeddings: {args.lean_path}")
    print(f"Output dir: {args.output_dir}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print("=" * 80)
    
    # Initialize GPU OT if requested
    set_ot_device(args.use_gpu_ot and device.type == "cuda")
    
    # Initialize streaming dataset
    print("\n📊 Initializing streaming dataset...")
    dataset = StreamingParquetDataset(
        nl_path=args.nl_path,
        lean_path=args.lean_path,
        max_len=args.max_len,
        ot_cost=args.ot_cost,
        ot_reg=args.ot_reg,
        ot_method=args.ot_method,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    
    # Auto-detect hidden dim
    hidden_dim = args.hidden_dim or dataset.get_hidden_dim()
    print(f"\n🔧 Model config:")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Time embed dim: {args.time_embed_dim}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  MLP width: {args.mlp_width}")
    
    # Initialize model
    model = NeuralOTFlow(
        hidden_dim=hidden_dim,
        time_embed_dim=args.time_embed_dim,
        num_layers=args.num_layers,
        mlp_width=args.mlp_width,
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Estimate total steps for scheduler
    approx_batches_per_epoch = dataset.effective_rows // args.batch_size
    total_steps = args.num_epochs * approx_batches_per_epoch
    warmup_steps = max(1, total_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    
    if args.resume:
        print(f"\n📥 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["step"]
        best_loss = checkpoint["metrics"].get("train_loss", float("inf"))
        print(f"  Resuming from epoch {start_epoch}, step {global_step}")
    
    # Initialize logger
    config = {
        "nl_path": args.nl_path,
        "lean_path": args.lean_path,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "max_len": args.max_len,
        "hidden_dim": hidden_dim,
        "time_embed_dim": args.time_embed_dim,
        "num_layers": args.num_layers,
        "mlp_width": args.mlp_width,
        "ot_method": args.ot_method,
        "ot_cost": args.ot_cost,
        "ot_reg": args.ot_reg,
    }
    
    logger = ResultsLogger(
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        config=config,
    )
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Approx steps per epoch: {approx_batches_per_epoch}")
    print(f"  Total steps: ~{total_steps}")
    
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0
        epoch_start = time.time()
        
        pbar = tqdm(
            dataset.iterate_epoch(shuffle_row_groups=True),
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            total=approx_batches_per_epoch,
            dynamic_ncols=True,
        )
        
        for batch_idx, (h_nl, h_lean, coupling, nl_lens, lean_lens) in enumerate(pbar):
            h_nl = h_nl.to(device)
            h_lean = h_lean.to(device)
            coupling = coupling.to(device)
            nl_lens = nl_lens.to(device)
            
            B, L_nl, d = h_nl.shape
            
            optimizer.zero_grad()
            
            # Compute aligned targets via barycentric projection
            coupling_norm = coupling / (coupling.sum(dim=-1, keepdim=True) + 1e-8)
            h_lean_aligned = torch.bmm(coupling_norm, h_lean)
            
            # Sample time and interpolate
            t = torch.rand(B, L_nl, device=device, dtype=h_nl.dtype)
            t_exp = t.unsqueeze(-1)
            x_t = (1 - t_exp) * h_nl + t_exp * h_lean_aligned
            v_target = h_lean_aligned - h_nl
            
            # Forward pass
            v_pred = model.v_theta(x_t, t, None)
            
            # Masked MSE loss
            mask = torch.zeros(B, L_nl, device=device, dtype=h_nl.dtype)
            for i in range(B):
                mask[i, :nl_lens[i]] = 1.0
            
            diff = (v_pred - v_target) ** 2
            diff = diff * mask.unsqueeze(-1)
            loss = diff.sum() / (mask.sum() * d + 1e-8)
            
            loss.backward()
            
            grad_norm = compute_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            epoch_loss_sum += loss.item()
            epoch_count += 1
            global_step += 1
            
            avg_loss = epoch_loss_sum / epoch_count
            current_lr = optimizer.param_groups[0]['lr']
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.1e}',
            })
            
            # Log metrics
            if (batch_idx + 1) % args.log_every == 0:
                logger.log_metrics({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/grad_norm": grad_norm,
                    "train/learning_rate": current_lr,
                }, step=global_step, epoch=epoch)
            
            # Save checkpoint
            if (batch_idx + 1) % args.save_every == 0:
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                logger.save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step,
                    {"train_loss": avg_loss},
                    is_best=is_best
                )
        
        # End of epoch
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = epoch_loss_sum / max(epoch_count, 1)
        print(f"\n✅ Epoch {epoch+1} complete. Avg loss: {epoch_avg_loss:.6f}, Time: {epoch_time/60:.1f}min")
        
        # Save checkpoint
        is_best = epoch_avg_loss < best_loss
        if is_best:
            best_loss = epoch_avg_loss
        logger.save_checkpoint(
            model, optimizer, scheduler, epoch, global_step,
            {"train_loss": epoch_avg_loss},
            is_best=is_best
        )
    
    # Finalize
    summary = logger.finish()
    
    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Results saved to: {logger.run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
