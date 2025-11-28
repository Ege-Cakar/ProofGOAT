"""Train a Neural OT velocity field on precomputed embeddings with comprehensive logging.

Usage:
  python -m scripts.train_neural_ot --config project_config.yaml [--wandb-project PROJECT] [--wandb-entity ENTITY] [--no-wandb]
"""
import argparse
import os
import yaml
import math
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(ROOT)

from src.flows.dataset import EmbeddingPairDataset, StreamingEmbeddingPairDataset, collate_fn
from src.flows.neural_ot import NeuralOTFlow
from src.flows.losses import cycle_consistency_loss
from src.io_utils import ensure_dir

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


class ResultsLogger:
    """Logger that saves results locally and optionally to wandb."""

    def __init__(self, output_dir, use_wandb=False, wandb_project=None, wandb_entity=None, config=None):
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.plots_dir = self.run_dir / "plots"
        self.metrics_dir = self.run_dir / "metrics"
        self.embeddings_dir = self.run_dir / "embeddings"

        for d in [self.checkpoints_dir, self.plots_dir, self.metrics_dir, self.embeddings_dir]:
            d.mkdir(exist_ok=True)

        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=wandb_project or "hermes-neural-ot",
                entity=wandb_entity,
                config=config,
                dir=str(self.run_dir)
            )
            print(f"Initialized wandb run: {wandb.run.name}")

        # Initialize metrics storage
        self.metrics_history = {
            "train_loss": [],
            "flow_matching_loss": [],
            "cycle_loss": [],
            "learning_rate": [],
            "grad_norm": [],
            "velocity_magnitude": [],
            "epoch": [],
            "step": []
        }

        # Save config
        if config:
            with open(self.run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

        print(f"Results will be saved to: {self.run_dir}")

    def log_metrics(self, metrics, step=None, epoch=None, commit=True):
        """Log metrics to wandb and local storage."""
        # Add to history
        if step is not None:
            if "step" not in self.metrics_history:
                self.metrics_history["step"] = []
            self.metrics_history["step"].append(step)
        if epoch is not None:
            if "epoch" not in self.metrics_history:
                self.metrics_history["epoch"] = []
            self.metrics_history["epoch"].append(epoch)

        for key, value in metrics.items():
            # Normalize key (remove namespace prefix for storage)
            storage_key = key.replace("train/", "").replace("velocity_field/", "").replace("transport/", "")
            if storage_key not in self.metrics_history:
                self.metrics_history[storage_key] = []
            self.metrics_history[storage_key].append(float(value) if isinstance(value, (int, float, torch.Tensor)) else value)

        # Log to wandb
        if self.use_wandb:
            log_dict = metrics.copy()
            if step is not None:
                log_dict["step"] = step
            if epoch is not None:
                log_dict["epoch"] = epoch
            wandb.log(log_dict, commit=commit)

        # Save metrics to JSON periodically
        if step is not None and step % 100 == 0:
            self._save_metrics()

    def _save_metrics(self):
        """Save metrics history to JSON file."""
        metrics_file = self.metrics_dir / "metrics_history.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def save_checkpoint(self, model, optimizer, epoch, step, metrics, is_best=False):
        """Save model checkpoint with metadata."""
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics
        }

        # Save regular checkpoint
        ckpt_path = self.checkpoints_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoints_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

            if self.use_wandb:
                wandb.run.summary["best_epoch"] = epoch
                wandb.run.summary["best_step"] = step
                wandb.run.summary["best_loss"] = metrics.get("train_loss", float("inf"))

        # Save latest checkpoint (for easy resuming)
        latest_path = self.checkpoints_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)

    def save_embeddings(self, embeddings_dict, filename):
        """Save embeddings as numpy arrays."""
        save_path = self.embeddings_dir / filename
        np.savez(save_path, **embeddings_dict)
        print(f"Saved embeddings: {save_path}")

        if self.use_wandb:
            wandb.save(str(save_path))

    def log_velocity_field_stats(self, model, x_samples, epoch, num_time_points=10):
        """Analyze and log velocity field statistics."""
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            x_samples = x_samples.to(device)
            B, L, d = x_samples.shape

            # Sample at different time points
            velocity_norms = []
            for t_val in np.linspace(0, 1, num_time_points):
                t = torch.full((B, L), t_val, device=device, dtype=x_samples.dtype)
                v = model.v_theta(x_samples, t, p=None)
                v_norm = torch.norm(v, dim=-1).mean().item()
                velocity_norms.append(v_norm)

            # Log statistics
            stats = {
                f"velocity_field/norm_t{i}": norm
                for i, norm in enumerate(velocity_norms)
            }
            stats["velocity_field/mean_norm"] = np.mean(velocity_norms)
            stats["velocity_field/std_norm"] = np.std(velocity_norms)
            stats["velocity_field/max_norm"] = np.max(velocity_norms)

            self.log_metrics(stats, epoch=epoch)

            # Save velocity profile
            profile_data = {
                "time_points": np.linspace(0, 1, num_time_points),
                "velocity_norms": velocity_norms,
                "epoch": epoch
            }
            np.savez(
                self.metrics_dir / f"velocity_profile_epoch{epoch}.npz",
                **profile_data
            )

    def log_transport_analysis(self, model, x_nl, x_lean, epoch, num_steps=8):
        """Analyze transport quality and log metrics."""
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            x_nl = x_nl.to(device)
            x_lean = x_lean.to(device)

            # Forward transport: NL -> Lean
            x_nl_transported = model.transport_nl_to_lean(x_nl, num_steps=num_steps)

            # Backward transport: Lean -> NL
            x_lean_transported = model.transport_lean_to_nl(x_lean, num_steps=num_steps)

            # Cycle consistency: NL -> Lean -> NL
            x_nl_cycle = model.transport_lean_to_nl(x_nl_transported, num_steps=num_steps)

            # Compute distances
            dist_nl_to_lean = torch.norm(x_nl_transported - x_lean, dim=-1).mean().item()
            dist_lean_to_nl = torch.norm(x_lean_transported - x_nl, dim=-1).mean().item()
            cycle_error = torch.norm(x_nl - x_nl_cycle, dim=-1).mean().item()

            # Compute cosine similarities
            x_nl_flat = x_nl.reshape(-1, x_nl.shape[-1])
            x_lean_flat = x_lean.reshape(-1, x_lean.shape[-1])
            x_nl_transported_flat = x_nl_transported.reshape(-1, x_nl_transported.shape[-1])

            # Original similarity
            original_sim = torch.nn.functional.cosine_similarity(
                x_nl_flat, x_lean_flat, dim=-1
            ).mean().item()

            # Transported similarity
            transported_sim = torch.nn.functional.cosine_similarity(
                x_nl_transported_flat, x_lean_flat, dim=-1
            ).mean().item()

            metrics = {
                "transport/nl_to_lean_distance": dist_nl_to_lean,
                "transport/lean_to_nl_distance": dist_lean_to_nl,
                "transport/cycle_error": cycle_error,
                "transport/original_similarity": original_sim,
                "transport/transported_similarity": transported_sim,
                "transport/similarity_improvement": transported_sim - original_sim
            }

            self.log_metrics(metrics, epoch=epoch)

            # Save transport analysis
            analysis_data = {
                "epoch": epoch,
                "metrics": metrics,
                "x_nl_sample": x_nl[:2].cpu().numpy(),
                "x_lean_sample": x_lean[:2].cpu().numpy(),
                "x_nl_transported_sample": x_nl_transported[:2].cpu().numpy(),
                "x_lean_transported_sample": x_lean_transported[:2].cpu().numpy()
            }
            np.savez(
                self.embeddings_dir / f"transport_analysis_epoch{epoch}.npz",
                **analysis_data
            )

    def finish(self):
        """Finalize logging and save final results."""
        self._save_metrics()

        # Create summary
        loss_key = "loss" if "loss" in self.metrics_history else "train_loss"
        summary = {
            "total_steps": len(self.metrics_history.get("step", [])),
            "total_epochs": len(set(self.metrics_history.get("epoch", []))) if self.metrics_history.get("epoch") else 0,
            "final_loss": self.metrics_history[loss_key][-1] if loss_key in self.metrics_history and self.metrics_history[loss_key] else None,
            "best_loss": min(self.metrics_history[loss_key]) if loss_key in self.metrics_history and self.metrics_history[loss_key] else None,
            "run_dir": str(self.run_dir)
        }

        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if self.use_wandb:
            wandb.finish()

        print(f"\nTraining complete. Results saved to: {self.run_dir}")
        print(f"Summary: {json.dumps(summary, indent=2)}")


def compute_grad_norm(model):
    """Compute the global gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def main(cfg, use_wandb=True, wandb_project=None, wandb_entity=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    neo_cfg = cfg.get("neural_ot", {})
    nl_path = neo_cfg.get("nl_embeddings", os.path.join(cfg["data"]["out_dir"], "nl_embeddings.parquet"))
    lean_path = neo_cfg.get("lean_embeddings", os.path.join(cfg["data"]["out_dir"], "lean_embeddings.parquet"))
    out_dir = neo_cfg.get("output_dir", os.path.join(cfg["data"]["out_dir"], "neural_ot"))
    ensure_dir(out_dir)

    # Explicit type conversions to prevent type errors
    max_len = int(neo_cfg.get("max_len", 256))
    batch_size = int(neo_cfg.get("batch_size", 8))
    num_epochs = int(neo_cfg.get("num_epochs", 5))
    lr = float(neo_cfg.get("learning_rate", 1e-4))
    hidden_dim = int(neo_cfg.get("hidden_dim", 4096))
    time_embed_dim = int(neo_cfg.get("time_embed_dim", 128))
    num_layers = int(neo_cfg.get("num_layers", 3))
    mlp_width = int(neo_cfg.get("mlp_width", 2048))
    num_steps = int(neo_cfg.get("num_steps", 8))
    lambda_cycle = float(neo_cfg.get("lambda_cycle", 0.0))

    # Scheduler configuration
    warmup_steps_config = neo_cfg.get("warmup_steps", None)
    min_lr_ratio = float(neo_cfg.get("min_lr_ratio", 0.1))

    # Logging configuration
    log_every = int(neo_cfg.get("log_every", 10))
    eval_every = int(neo_cfg.get("eval_every", 100))
    save_every = int(neo_cfg.get("save_every", 500))

    # Initialize logger
    logger = ResultsLogger(
        output_dir=out_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        config={
            "neural_ot": neo_cfg,
            "data": cfg.get("data", {}),
            "models": cfg.get("models", {})
        }
    )

    # Load dataset
    print(f"Loading dataset from {nl_path} and {lean_path}")
    
    # Use streaming dataset for large parquet files (>100MB)
    nl_size = os.path.getsize(nl_path) if os.path.exists(nl_path) else 0
    lean_size = os.path.getsize(lean_path) if os.path.exists(lean_path) else 0
    use_streaming = (nl_size > 100 * 1024 * 1024) or (lean_size > 100 * 1024 * 1024)
    
    if use_streaming and nl_path.endswith('.parquet') and lean_path.endswith('.parquet'):
        print(f"Using streaming dataset (NL: {nl_size/1e9:.2f}GB, Lean: {lean_size/1e9:.2f}GB)")
        ds = StreamingEmbeddingPairDataset(nl_path, lean_path, max_len=max_len, cache_size=4)
    else:
        ds = EmbeddingPairDataset(nl_path, lean_path, max_len=max_len)
    
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    print(f"Dataset size: {len(ds)} pairs")

    # Instantiate model
    print(f"Initializing model with hidden_dim={hidden_dim}, num_layers={num_layers}")
    model = NeuralOTFlow(
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        num_layers=num_layers,
        mlp_width=mlp_width
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    logger.log_metrics({
        "model/total_parameters": total_params,
        "model/trainable_parameters": trainable_params
    })

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Learning rate scheduler: warmup + cosine decay
    total_steps = num_epochs * len(dl)
    if warmup_steps_config is None:
        warmup_steps = max(1, total_steps // 10)  # Default: 10% of training
    else:
        warmup_steps = int(warmup_steps_config)

    print(f"Scheduler: {warmup_steps} warmup steps, decay to {min_lr_ratio:.1%} of initial LR")

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine decay to min_lr_ratio of initial LR
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float("inf")
    global_step = 0

    # Sample batch for analysis
    sample_nl, sample_lean = next(iter(dl))
    sample_nl = sample_nl.to(device)
    sample_lean = sample_lean.to(device)

    print("\n" + "="*80)
    print("Starting training")
    print("="*80 + "\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_metrics = {
            "loss_sum": 0.0,
            "fm_loss_sum": 0.0,
            "cycle_loss_sum": 0.0,
            "count": 0
        }

        # Progress bar for this epoch
        pbar = tqdm(enumerate(dl), total=len(dl), desc=f"Epoch {epoch+1}/{num_epochs}", 
                    dynamic_ncols=True, leave=True)
        
        for i, (h_nl, h_lean) in pbar:
            h_nl = h_nl.to(device)
            h_lean = h_lean.to(device)

            optimizer.zero_grad()

            # Flow matching loss
            l_fm = model.compute_flow_matching_loss(h_nl, h_lean)
            loss = l_fm

            # Cycle consistency loss (optional)
            l_cycle = 0.0
            if lambda_cycle > 0.0:
                l_cycle_tensor = cycle_consistency_loss(model, h_nl, num_steps=num_steps)
                loss = loss + lambda_cycle * l_cycle_tensor
                l_cycle = l_cycle_tensor.item()

            loss.backward()

            # Compute gradient norm before clipping
            grad_norm = compute_grad_norm(model)

            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Update metrics
            epoch_metrics["loss_sum"] += loss.item()
            epoch_metrics["fm_loss_sum"] += l_fm.item()
            epoch_metrics["cycle_loss_sum"] += l_cycle if isinstance(l_cycle, float) else l_cycle.item()
            epoch_metrics["count"] += 1
            global_step += 1

            # Update progress bar with current metrics
            avg_loss = epoch_metrics["loss_sum"] / epoch_metrics["count"]
            avg_fm = epoch_metrics["fm_loss_sum"] / epoch_metrics["count"]
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'fm': f'{avg_fm:.4f}',
                'lr': f'{current_lr:.1e}',
                'gnorm': f'{grad_norm:.2f}'
            })

            # Log every N steps
            if (i + 1) % log_every == 0:
                avg_cycle = epoch_metrics["cycle_loss_sum"] / epoch_metrics["count"]

                metrics = {
                    "train/loss": loss.item(),
                    "train/flow_matching_loss": l_fm.item(),
                    "train/cycle_loss": l_cycle,
                    "train/avg_loss": avg_loss,
                    "train/avg_fm_loss": avg_fm,
                    "train/avg_cycle_loss": avg_cycle,
                    "train/grad_norm": grad_norm,
                    "train/learning_rate": current_lr
                }

                logger.log_metrics(metrics, step=global_step, epoch=epoch)

            # Evaluation every N steps
            if (i + 1) % eval_every == 0:
                logger.log_velocity_field_stats(model, sample_nl[:4], epoch=epoch)
                logger.log_transport_analysis(model, sample_nl[:4], sample_lean[:4], epoch=epoch, num_steps=num_steps)
                model.train()

            # Save checkpoint every N steps
            if (i + 1) % save_every == 0:
                checkpoint_metrics = {
                    "train_loss": epoch_metrics["loss_sum"] / epoch_metrics["count"],
                    "fm_loss": epoch_metrics["fm_loss_sum"] / epoch_metrics["count"],
                    "cycle_loss": epoch_metrics["cycle_loss_sum"] / epoch_metrics["count"]
                }
                is_best = checkpoint_metrics["train_loss"] < best_loss
                if is_best:
                    best_loss = checkpoint_metrics["train_loss"]

                logger.save_checkpoint(model, optimizer, epoch, global_step, checkpoint_metrics, is_best=is_best)

        # End of epoch
        epoch_loss = epoch_metrics["loss_sum"] / max(1, epoch_metrics["count"])
        epoch_fm = epoch_metrics["fm_loss_sum"] / max(1, epoch_metrics["count"])
        epoch_cycle = epoch_metrics["cycle_loss_sum"] / max(1, epoch_metrics["count"])

        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs} Complete")
        print(f"Average Loss: {epoch_loss:.6f} | FM Loss: {epoch_fm:.6f} | Cycle Loss: {epoch_cycle:.6f}")
        print(f"{'='*80}\n")

        # Log epoch-level metrics
        logger.log_metrics({
            "train/epoch_loss": epoch_loss,
            "train/epoch_fm_loss": epoch_fm,
            "train/epoch_cycle_loss": epoch_cycle
        }, step=global_step, epoch=epoch+1)

        # End-of-epoch evaluation
        logger.log_velocity_field_stats(model, sample_nl, epoch=epoch+1)
        logger.log_transport_analysis(model, sample_nl, sample_lean, epoch=epoch+1, num_steps=num_steps)

        # Save end-of-epoch checkpoint
        checkpoint_metrics = {
            "train_loss": epoch_loss,
            "fm_loss": epoch_fm,
            "cycle_loss": epoch_cycle
        }
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss

        logger.save_checkpoint(model, optimizer, epoch+1, global_step, checkpoint_metrics, is_best=is_best)

    print("\n" + "="*80)
    print("Training complete!")
    print(f"Best loss: {best_loss:.6f}")
    print("="*80)

    # Final evaluation and save
    logger.log_velocity_field_stats(model, sample_nl, epoch=num_epochs)
    logger.log_transport_analysis(model, sample_nl, sample_lean, epoch=num_epochs, num_steps=num_steps)

    # Generate plots before finishing
    print("\nGenerating analysis plots...")
    try:
        import matplotlib
        import seaborn
        import sklearn
        from scripts.analyze_results import ResultsAnalyzer
        analyzer = ResultsAnalyzer(str(logger.run_dir))
        analyzer.plot_training_curves()
        analyzer.plot_velocity_field_analysis()
        analyzer.plot_transport_analysis()
        analyzer.create_comprehensive_dashboard()
        analyzer.generate_summary_report()
        print("✓ Plots generated successfully")
    except ImportError as e:
        print(f"\nNote: Visualization libraries not installed. Skipping plot generation.")
        print(f"To generate plots, install: pip install matplotlib seaborn scikit-learn")
        print(f"Then run: python -m scripts.analyze_results --run-dir {logger.run_dir}")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    logger.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="project_config.yaml", help="Path to config YAML file")
    ap.add_argument("--wandb-project", default=None, help="Wandb project name")
    ap.add_argument("--wandb-entity", default=None, help="Wandb entity/username")
    ap.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg, use_wandb=not args.no_wandb, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity)
