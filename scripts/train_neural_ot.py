"""Train a Neural OT velocity field from precomputed OT shards.

This script loads precomputed OT couplings from shards and trains
a neural velocity field to learn the OT-aligned transport.

Usage:
  # First, precompute OT couplings (saves in shards):
  python -m scripts.compute_ot_couplings --config project_config.yaml

  # Then train:
  python -m scripts.train_neural_ot --config project_config.yaml [--no-wandb]
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
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(ROOT)

from src.flows.neural_ot import NeuralOTFlow
from src.io_utils import ensure_dir

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Dataset that loads from precomputed OT shards
# =============================================================================

class ShardedOTDataset(Dataset):
    """Dataset that loads precomputed OT couplings from shards.
    
    Each shard is a .npz file containing:
      - ids: sample IDs
      - nl_embeddings: list of [L, d] arrays
      - lean_embeddings: list of [L, d] arrays  
      - couplings: list of [L_nl, L_lean] coupling matrices
    """
    
    def __init__(self, shard_dir: str, max_len: int = 256):
        self.shard_dir = Path(shard_dir)
        self.max_len = max_len
        
        # Load metadata
        metadata_path = self.shard_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        self.num_shards = self.metadata['num_shards']
        self.total_samples = self.metadata['total_samples']
        self.shard_size = self.metadata['shard_size']
        
        # Build index: global_idx -> (shard_idx, local_idx)
        self.index = []
        for shard_idx in range(self.num_shards):
            shard_path = self.shard_dir / f"shard_{shard_idx:05d}.npz"
            if shard_path.exists():
                # Peek at shard to get count
                data = np.load(shard_path, allow_pickle=True)
                n_samples = len(data['ids'])
                for local_idx in range(n_samples):
                    self.index.append((shard_idx, local_idx))
                data.close()
        
        # Cache for loaded shards
        self._cache = {}
        self._cache_order = []
        self._max_cache = 3  # Keep up to 3 shards in memory
        
        print(f"ShardedOTDataset: {len(self.index)} samples from {self.num_shards} shards")
        print(f"  Config: {self.metadata.get('config', {})}")
    
    def _load_shard(self, shard_idx: int):
        """Load a shard into cache."""
        if shard_idx in self._cache:
            return self._cache[shard_idx]
        
        shard_path = self.shard_dir / f"shard_{shard_idx:05d}.npz"
        data = np.load(shard_path, allow_pickle=True)
        
        shard = {
            'ids': data['ids'],
            'nl_embeddings': data['nl_embeddings'],
            'lean_embeddings': data['lean_embeddings'],
            'couplings': data['couplings'],
        }
        
        # Add to cache
        self._cache[shard_idx] = shard
        self._cache_order.append(shard_idx)
        
        # Evict old shards
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
        
        nl_emb = shard['nl_embeddings'][local_idx].astype(np.float32)
        lean_emb = shard['lean_embeddings'][local_idx].astype(np.float32)
        coupling = shard['couplings'][local_idx].astype(np.float32)
        
        # Truncate to max_len
        if nl_emb.shape[0] > self.max_len:
            nl_emb = nl_emb[:self.max_len]
            coupling = coupling[:self.max_len, :]
        if lean_emb.shape[0] > self.max_len:
            lean_emb = lean_emb[:self.max_len]
            coupling = coupling[:, :self.max_len]
        
        return (
            torch.from_numpy(nl_emb),
            torch.from_numpy(lean_emb),
            torch.from_numpy(coupling),
        )


def collate_ot_batch(batch):
    """Collate variable-length sequences with couplings."""
    nl_list, lean_list, coupling_list = zip(*batch)
    
    B = len(batch)
    max_nl = max(x.shape[0] for x in nl_list)
    max_lean = max(x.shape[0] for x in lean_list)
    d = nl_list[0].shape[1]
    
    # Pad sequences
    nl_padded = torch.zeros(B, max_nl, d)
    lean_padded = torch.zeros(B, max_lean, d)
    coupling_padded = torch.zeros(B, max_nl, max_lean)
    nl_lens = torch.zeros(B, dtype=torch.long)
    lean_lens = torch.zeros(B, dtype=torch.long)
    
    for i, (nl, lean, coup) in enumerate(batch):
        L_nl, L_lean = nl.shape[0], lean.shape[0]
        nl_padded[i, :L_nl] = nl
        lean_padded[i, :L_lean] = lean
        coupling_padded[i, :L_nl, :L_lean] = coup
        nl_lens[i] = L_nl
        lean_lens[i] = L_lean
    
    return nl_padded, lean_padded, coupling_padded, nl_lens, lean_lens


# =============================================================================
# Training utilities
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
            best_path = self.checkpoints_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (loss: {metrics.get('train_loss', 'N/A'):.6f})")
    
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
        
        if "train/learning_rate" in self.metrics_history:
            plt.figure(figsize=(10, 4))
            plt.plot(self.metrics_history["train/learning_rate"])
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True, alpha=0.3)
            plt.savefig(self.plots_dir / "lr_schedule.png", dpi=150, bbox_inches='tight')
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
# Main training function
# =============================================================================

def main(cfg, use_wandb=False, wandb_project=None, wandb_entity=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    neo_cfg = cfg.get("neural_ot", {})
    out_dir = neo_cfg.get("output_dir", os.path.join(cfg["data"]["out_dir"], "neural_ot"))
    ensure_dir(out_dir)
    
    # Model config
    max_len = int(neo_cfg.get("max_len", 256))
    batch_size = int(neo_cfg.get("batch_size", 64))
    num_epochs = int(neo_cfg.get("num_epochs", 5))
    lr = float(neo_cfg.get("learning_rate", 1e-4))
    hidden_dim = int(neo_cfg.get("hidden_dim", 2048))
    time_embed_dim = int(neo_cfg.get("time_embed_dim", 256))
    num_layers = int(neo_cfg.get("num_layers", 5))
    mlp_width = int(neo_cfg.get("mlp_width", 9216))
    
    # Training config
    log_every = int(neo_cfg.get("log_every", 10))
    save_every = int(neo_cfg.get("save_every", 500))
    
    # Data paths
    shard_dir = neo_cfg.get("ot_shards_dir", os.path.join(out_dir, "ot_shards"))
    
    # =========================================================================
    # PHASE 1: Load dataset from shards
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 1: Loading precomputed OT shards")
    print("="*80)
    
    if not os.path.exists(shard_dir):
        print(f"\nERROR: OT shards not found: {shard_dir}")
        print("\nPlease run the OT precomputation first:")
        print("  python -m scripts.compute_ot_couplings --config project_config.yaml")
        return
    
    dataset = ShardedOTDataset(shard_dir, max_len=max_len)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ot_batch,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=2,
    )
    
    # Get embedding dimension from first sample
    sample = dataset[0]
    data_hidden_dim = sample[0].shape[-1]
    
    if data_hidden_dim != hidden_dim:
        print(f"\nWARNING: Config hidden_dim={hidden_dim} but data has dim={data_hidden_dim}")
        print(f"Using data dimension: {data_hidden_dim}")
        hidden_dim = data_hidden_dim
    
    print(f"\nDataset: {len(dataset)} samples")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Embedding dim: {hidden_dim}")
    print(f"Max sequence length: {max_len}")
    
    # Initialize logger
    logger = ResultsLogger(
        output_dir=out_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        config={"neural_ot": neo_cfg}
    )
    
    # =========================================================================
    # PHASE 2: Initialize model
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: Initializing model")
    print("="*80)
    
    model = NeuralOTFlow(
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        num_layers=num_layers,
        mlp_width=mlp_width
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: hidden={hidden_dim}, time_embed={time_embed_dim}, layers={num_layers}, mlp={mlp_width}")
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
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
    
    print(f"Optimizer: AdamW, lr={lr}")
    print(f"Scheduler: {warmup_steps} warmup steps, cosine decay")
    
    # =========================================================================
    # PHASE 3: Training loop
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 3: Training")
    print("="*80 + "\n")
    
    best_loss = float("inf")
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0
        epoch_start = time.time()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True)
        
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
            if (batch_idx + 1) % log_every == 0:
                logger.log_metrics({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/grad_norm": grad_norm,
                    "train/learning_rate": current_lr,
                }, step=global_step, epoch=epoch)
            
            # Save checkpoint
            if (batch_idx + 1) % save_every == 0:
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                logger.save_checkpoint(
                    model, optimizer, epoch, global_step,
                    {"train_loss": avg_loss},
                    is_best=is_best
                )
        
        # End of epoch
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = epoch_loss_sum / epoch_count
        print(f"Epoch {epoch+1} complete. Avg loss: {epoch_avg_loss:.6f}, Time: {epoch_time/60:.1f}min")
        
        # Save checkpoint
        is_best = epoch_avg_loss < best_loss
        if is_best:
            best_loss = epoch_avg_loss
        logger.save_checkpoint(
            model, optimizer, epoch, global_step,
            {"train_loss": epoch_avg_loss},
            is_best=is_best
        )
    
    # Finalize
    summary = logger.finish()
    
    print("\n" + "="*80)
    print("Training complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Results saved to: {logger.run_dir}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neural OT from precomputed shards")
    parser.add_argument("--config", type=str, default="project_config.yaml")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    main(
        cfg,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
