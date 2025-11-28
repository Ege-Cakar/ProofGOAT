"""Train Augmented OT-CFM (A-OT-CFM) for cardinality-fluid flow matching.

This trains a Transformer-based velocity field that learns:
- Translation (Active → Active)
- Deletion (Active → Void)
- Creation (Void → Active)

With distributed void positioning and existence channel.

Usage:
  # First, precompute augmented OT:
  python -m scripts.compute_augmented_ot --config project_config.yaml

  # Then train:
  python -m scripts.train_augmented_ot --config project_config.yaml [--no-wandb]
"""
import argparse
import os
import yaml
import math
import json
from datetime import datetime
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(ROOT)

from src.io_utils import ensure_dir

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Augmented OT-CFM Model (Transformer-based)
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
    
    Input: [B, N_max, d+1] (content + existence)
    Output: [B, N_max, d+1] (velocity for content and existence)
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
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, self.aug_dim)
        
        # Input projection (aug_dim -> aug_dim for residual)
        self.input_norm = nn.LayerNorm(self.aug_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.aug_dim,
            nhead=num_heads,
            dim_feedforward=int(self.aug_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_norm = nn.LayerNorm(self.aug_dim)
        self.output_proj = nn.Linear(self.aug_dim, self.aug_dim)
        
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
        
        # Time conditioning
        t_emb = self.time_embed(t)  # [B, time_embed_dim]
        t_emb = self.time_proj(t_emb)  # [B, aug_dim]
        t_emb = t_emb.unsqueeze(1)  # [B, 1, aug_dim]
        
        # Add time to all positions
        h = self.input_norm(x) + t_emb
        
        # Transformer (self-attention across sequence)
        h = self.transformer(h)
        
        # Output
        h = self.output_norm(h)
        v = self.output_proj(h)
        
        return v


# =============================================================================
# Dataset
# =============================================================================

class AugmentedOTDataset(Dataset):
    """Dataset for augmented OT-CFM training."""
    
    def __init__(self, shard_dir: str):
        self.shard_dir = Path(shard_dir)
        
        # Load metadata
        metadata_path = self.shard_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        self.num_shards = self.metadata['num_shards']
        self.total_samples = self.metadata['total_samples']
        self.config = self.metadata.get('config', {})
        
        # Build index
        self.index = []
        for shard_idx in range(self.num_shards):
            shard_path = self.shard_dir / f"shard_{shard_idx:05d}.npz"
            if shard_path.exists():
                data = np.load(shard_path, allow_pickle=True)
                n_samples = len(data['ids'])
                for local_idx in range(n_samples):
                    self.index.append((shard_idx, local_idx))
                data.close()
        
        # Shard cache
        self._cache = {}
        self._cache_order = []
        self._max_cache = 3
        
        print(f"AugmentedOTDataset: {len(self.index)} samples from {self.num_shards} shards")
        print(f"  Config: {self.config}")
    
    def _load_shard(self, shard_idx: int):
        if shard_idx in self._cache:
            return self._cache[shard_idx]
        
        shard_path = self.shard_dir / f"shard_{shard_idx:05d}.npz"
        data = np.load(shard_path, allow_pickle=True)
        
        shard = {
            'src_states': data['src_states'],
            'tgt_states': data['tgt_states'],
            'permutations': data['permutations'],
        }
        
        self._cache[shard_idx] = shard
        self._cache_order.append(shard_idx)
        
        while len(self._cache_order) > self._max_cache:
            old_idx = self._cache_order.pop(0)
            if old_idx in self._cache:
                del self._cache[old_idx]
        
        return shard
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        shard_idx, local_idx = self.index[idx]
        shard = self._load_shard(shard_idx)
        
        src_state = torch.from_numpy(shard['src_states'][local_idx].astype(np.float32))
        tgt_state = torch.from_numpy(shard['tgt_states'][local_idx].astype(np.float32))
        perm = torch.from_numpy(shard['permutations'][local_idx].astype(np.int64))
        
        return src_state, tgt_state, perm


def collate_augmented(batch):
    """Collate augmented OT samples."""
    src_states, tgt_states, perms = zip(*batch)
    return (
        torch.stack(src_states),
        torch.stack(tgt_states),
        torch.stack(perms),
    )


# =============================================================================
# Training utilities
# =============================================================================

def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


class ResultsLogger:
    """Logger for metrics and checkpoints."""
    
    def __init__(self, output_dir, use_wandb=False, wandb_project=None, config=None):
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.metrics_history = {}
        self.config = config or {}
        
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        
        if self.use_wandb:
            wandb.init(project=wandb_project or "augmented-ot-cfm", config=self.config)
        
        print(f"Results: {self.run_dir}")
    
    def log_metrics(self, metrics, step, epoch=None):
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
        
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def save_checkpoint(self, model, optimizer, epoch, step, metrics, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        
        path = self.checkpoints_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
        torch.save(checkpoint, path)
        
        if is_best:
            torch.save(checkpoint, self.checkpoints_dir / "best_model.pt")
            print(f"  Saved best model (loss: {metrics.get('train_loss', 'N/A'):.6f})")
    
    def finish(self):
        with open(self.run_dir / "metrics_history.json", "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in self.metrics_history.items()}, f)
        
        if self.use_wandb:
            wandb.finish()
        
        print(f"\nResults saved to: {self.run_dir}")


# =============================================================================
# Main training
# =============================================================================

def main(cfg, use_wandb=False, wandb_project=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    neo_cfg = cfg.get("neural_ot", {})
    out_dir = neo_cfg.get("output_dir", "outputs/neural_ot")
    
    # Model config
    hidden_dim = int(neo_cfg.get("hidden_dim", 2048))
    time_embed_dim = int(neo_cfg.get("time_embed_dim", 256))
    num_layers = int(neo_cfg.get("num_layers", 6))
    
    # Training config
    batch_size = int(neo_cfg.get("batch_size", 64))
    num_epochs = int(neo_cfg.get("num_epochs", 10))
    lr = float(neo_cfg.get("learning_rate", 1e-4))
    log_every = int(neo_cfg.get("log_every", 10))
    save_every = int(neo_cfg.get("save_every", 500))
    
    # Data path
    shard_dir = os.path.join(out_dir, "augmented_ot_shards")
    
    print("\n" + "="*80)
    print("Loading Augmented OT Dataset")
    print("="*80)
    
    if not os.path.exists(shard_dir):
        print(f"\nERROR: Shards not found: {shard_dir}")
        print("\nRun preprocessing first:")
        print("  python -m scripts.compute_augmented_ot --config project_config.yaml")
        return
    
    dataset = AugmentedOTDataset(shard_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_augmented,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
    )
    
    # Get dimensions from data
    sample = dataset[0]
    aug_dim = sample[0].shape[-1]  # d+1
    n_max = sample[0].shape[0]
    data_hidden_dim = aug_dim - 1
    
    print(f"N_max: {n_max}, aug_dim: {aug_dim} (hidden: {data_hidden_dim})")
    print(f"Samples: {len(dataset)}, Batches/epoch: {len(dataloader)}")
    
    if data_hidden_dim != hidden_dim:
        print(f"WARNING: config hidden_dim={hidden_dim}, data has {data_hidden_dim}. Using data dim.")
        hidden_dim = data_hidden_dim
    
    # Initialize logger
    logger = ResultsLogger(
        output_dir=out_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        config={"neural_ot": neo_cfg}
    )
    
    print("\n" + "="*80)
    print("Initializing Model")
    print("="*80)
    
    model = AugmentedVelocityField(
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        num_layers=num_layers,
        num_heads=8,
        mlp_ratio=4.0,
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Scheduler
    total_steps = num_epochs * len(dataloader)
    warmup_steps = max(1, total_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print("\n" + "="*80)
    print("Training")
    print("="*80 + "\n")
    
    best_loss = float("inf")
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0
        epoch_start = time.time()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (src_state, tgt_state, perm) in enumerate(pbar):
            src_state = src_state.to(device)  # [B, N, d+1]
            tgt_state = tgt_state.to(device)
            perm = perm.to(device)  # [B, N]
            
            B, N, aug_d = src_state.shape
            
            optimizer.zero_grad()
            
            # Apply permutation to get aligned targets
            # tgt_aligned[b, i] = tgt_state[b, perm[b, i]]
            batch_idx_expanded = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
            tgt_aligned = tgt_state[batch_idx_expanded, perm]  # [B, N, d+1]
            
            # Sample time
            t = torch.rand(B, device=device)
            t_exp = t.view(B, 1, 1)  # [B, 1, 1]
            
            # Interpolate: x_t = (1-t) * src + t * tgt_aligned
            x_t = (1 - t_exp) * src_state + t_exp * tgt_aligned
            
            # Target velocity: u = tgt_aligned - src
            v_target = tgt_aligned - src_state
            
            # Predict velocity
            v_pred = model(x_t, t)
            
            # MSE loss
            loss = ((v_pred - v_target) ** 2).mean()
            
            loss.backward()
            
            grad_norm = compute_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss_sum += loss.item()
            epoch_count += 1
            global_step += 1
            
            avg_loss = epoch_loss_sum / epoch_count
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.1e}',
            })
            
            if (batch_idx + 1) % log_every == 0:
                logger.log_metrics({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/grad_norm": grad_norm,
                    "train/lr": optimizer.param_groups[0]['lr'],
                }, step=global_step, epoch=epoch)
            
            if (batch_idx + 1) % save_every == 0:
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                logger.save_checkpoint(model, optimizer, epoch, global_step, {"train_loss": avg_loss}, is_best)
        
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = epoch_loss_sum / epoch_count
        print(f"Epoch {epoch+1} complete. Loss: {epoch_avg_loss:.6f}, Time: {epoch_time/60:.1f}min")
        
        is_best = epoch_avg_loss < best_loss
        if is_best:
            best_loss = epoch_avg_loss
        logger.save_checkpoint(model, optimizer, epoch, global_step, {"train_loss": epoch_avg_loss}, is_best)
    
    logger.finish()
    
    print("\n" + "="*80)
    print(f"Training complete! Best loss: {best_loss:.6f}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="project_config.yaml")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    main(cfg, use_wandb=not args.no_wandb, wandb_project=args.wandb_project)
