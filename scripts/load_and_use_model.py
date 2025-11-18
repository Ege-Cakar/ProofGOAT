"""Example script for loading trained Neural OT models and using them for inference.

This script demonstrates:
1. Loading a trained model from checkpoint
2. Transporting embeddings between NL and Lean spaces
3. Evaluating transport quality
4. Batch processing

Usage:
    python -m scripts.load_and_use_model --checkpoint outputs/neural_ot/run_XX/checkpoints/best_model.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.flows.neural_ot import NeuralOTFlow
from src.flows.dataset import EmbeddingPairDataset, collate_fn
from torch.utils.data import DataLoader


def load_model(checkpoint_path: str, device: str = "cuda") -> NeuralOTFlow:
    """Load a trained Neural OT model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        device: Device to load model on

    Returns:
        Loaded NeuralOTFlow model in eval mode
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract configuration
    if "cfg" in checkpoint:
        cfg = checkpoint["cfg"]
    elif "metrics" in checkpoint:
        # Try to infer from checkpoint structure
        cfg = {}
    else:
        # Use defaults
        cfg = {}

    # Get model hyperparameters (with defaults)
    hidden_dim = cfg.get("hidden_dim", 4096)
    time_embed_dim = cfg.get("time_embed_dim", 128)
    num_layers = cfg.get("num_layers", 3)
    mlp_width = cfg.get("mlp_width", 2048)
    max_seq_len = cfg.get("max_len", 5000)

    print(f"Model configuration:")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Time embed dim: {time_embed_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  MLP width: {mlp_width}")

    # Initialize model
    model = NeuralOTFlow(
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        num_layers=num_layers,
        mlp_width=mlp_width,
        max_seq_len=max_seq_len
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Print checkpoint info
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if "step" in checkpoint:
        print(f"  Step: {checkpoint['step']}")
    if "metrics" in checkpoint:
        print(f"  Metrics: {checkpoint['metrics']}")

    print("Model loaded successfully!\n")
    return model


def transport_embeddings(
    model: NeuralOTFlow,
    embeddings: torch.Tensor,
    direction: str = "nl_to_lean",
    num_steps: int = 16,
    device: str = "cuda"
) -> torch.Tensor:
    """Transport embeddings using the trained model.

    Args:
        model: Trained NeuralOTFlow model
        embeddings: Input embeddings [B, L, d]
        direction: 'nl_to_lean' or 'lean_to_nl'
        num_steps: Number of ODE integration steps
        device: Device for computation

    Returns:
        Transported embeddings [B, L, d]
    """
    embeddings = embeddings.to(device)

    with torch.no_grad():
        if direction == "nl_to_lean":
            transported = model.transport_nl_to_lean(embeddings, num_steps=num_steps)
        elif direction == "lean_to_nl":
            transported = model.transport_lean_to_nl(embeddings, num_steps=num_steps)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    return transported


def evaluate_transport_quality(
    model: NeuralOTFlow,
    nl_embeddings: torch.Tensor,
    lean_embeddings: torch.Tensor,
    num_steps: int = 16,
    device: str = "cuda"
) -> dict:
    """Evaluate transport quality on a batch of embeddings.

    Args:
        model: Trained NeuralOTFlow model
        nl_embeddings: NL embeddings [B, L, d]
        lean_embeddings: Lean embeddings [B, L, d]
        num_steps: Number of ODE integration steps
        device: Device for computation

    Returns:
        Dictionary of quality metrics
    """
    nl_embeddings = nl_embeddings.to(device)
    lean_embeddings = lean_embeddings.to(device)

    with torch.no_grad():
        # Forward transport: NL -> Lean
        nl_transported = model.transport_nl_to_lean(nl_embeddings, num_steps=num_steps)

        # Backward transport: Lean -> NL
        lean_transported = model.transport_lean_to_nl(lean_embeddings, num_steps=num_steps)

        # Cycle consistency: NL -> Lean -> NL
        nl_cycle = model.transport_lean_to_nl(nl_transported, num_steps=num_steps)

        # Compute metrics
        nl_to_lean_dist = torch.norm(nl_transported - lean_embeddings, dim=-1).mean().item()
        lean_to_nl_dist = torch.norm(lean_transported - nl_embeddings, dim=-1).mean().item()
        cycle_error = torch.norm(nl_embeddings - nl_cycle, dim=-1).mean().item()

        # Cosine similarities
        nl_flat = nl_embeddings.reshape(-1, nl_embeddings.shape[-1])
        lean_flat = lean_embeddings.reshape(-1, lean_embeddings.shape[-1])
        nl_transported_flat = nl_transported.reshape(-1, nl_transported.shape[-1])

        original_sim = torch.nn.functional.cosine_similarity(
            nl_flat, lean_flat, dim=-1
        ).mean().item()

        transported_sim = torch.nn.functional.cosine_similarity(
            nl_transported_flat, lean_flat, dim=-1
        ).mean().item()

        metrics = {
            "nl_to_lean_distance": nl_to_lean_dist,
            "lean_to_nl_distance": lean_to_nl_dist,
            "cycle_error": cycle_error,
            "original_similarity": original_sim,
            "transported_similarity": transported_sim,
            "similarity_improvement": transported_sim - original_sim
        }

    return metrics


def batch_process_dataset(
    model: NeuralOTFlow,
    nl_path: str,
    lean_path: str,
    output_path: str,
    max_len: int = 256,
    batch_size: int = 8,
    num_steps: int = 16,
    device: str = "cuda"
):
    """Process entire dataset and save transported embeddings.

    Args:
        model: Trained NeuralOTFlow model
        nl_path: Path to NL embeddings file
        lean_path: Path to Lean embeddings file
        output_path: Path to save transported embeddings
        max_len: Maximum sequence length
        batch_size: Batch size for processing
        num_steps: Number of ODE integration steps
        device: Device for computation
    """
    print(f"Loading dataset from {nl_path} and {lean_path}")
    dataset = EmbeddingPairDataset(nl_path, lean_path, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Processing {len(dataset)} pairs...")

    all_nl_transported = []
    all_lean_transported = []
    all_metrics = []

    for i, (nl_emb, lean_emb) in enumerate(dataloader):
        nl_emb = nl_emb.to(device)
        lean_emb = lean_emb.to(device)

        # Transport
        with torch.no_grad():
            nl_transported = model.transport_nl_to_lean(nl_emb, num_steps=num_steps)
            lean_transported = model.transport_lean_to_nl(lean_emb, num_steps=num_steps)

        # Evaluate this batch
        batch_metrics = evaluate_transport_quality(model, nl_emb, lean_emb, num_steps=num_steps, device=device)
        all_metrics.append(batch_metrics)

        # Store results
        all_nl_transported.append(nl_transported.cpu().numpy())
        all_lean_transported.append(lean_transported.cpu().numpy())

        if (i + 1) % 10 == 0:
            print(f"  Processed {(i+1)*batch_size}/{len(dataset)} pairs")

    # Concatenate all results
    all_nl_transported = np.concatenate(all_nl_transported, axis=0)
    all_lean_transported = np.concatenate(all_lean_transported, axis=0)

    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }

    # Save results
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path / "transported_embeddings.npz",
        nl_to_lean=all_nl_transported,
        lean_to_nl=all_lean_transported
    )

    with open(output_path / "transport_metrics.json", "w") as f:
        json.dump(avg_metrics, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"\nAverage metrics across dataset:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Load and use trained Neural OT models")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["evaluate", "batch_process", "demo"],
        default="demo",
        help="Operation mode"
    )
    parser.add_argument(
        "--nl-embeddings",
        type=str,
        default="outputs/nl_embeddings.parquet",
        help="Path to NL embeddings"
    )
    parser.add_argument(
        "--lean-embeddings",
        type=str,
        default="outputs/lean_embeddings.parquet",
        help="Path to Lean embeddings"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/transported",
        help="Output directory for batch processing"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=16,
        help="Number of ODE integration steps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )

    args = parser.parse_args()

    print("="*80)
    print("HERMES Neural OT - Model Inference")
    print("="*80 + "\n")

    # Load model
    model = load_model(args.checkpoint, device=args.device)

    if args.mode == "demo":
        print("Running demo with random embeddings...\n")

        # Create random embeddings for demo
        batch_size = 2
        seq_len = 64
        hidden_dim = model.hidden_dim

        nl_emb = torch.randn(batch_size, seq_len, hidden_dim)
        lean_emb = torch.randn(batch_size, seq_len, hidden_dim)

        print(f"Input shapes: NL={nl_emb.shape}, Lean={lean_emb.shape}")

        # Transport NL -> Lean
        print(f"\nTransporting NL -> Lean with {args.num_steps} steps...")
        nl_transported = transport_embeddings(
            model, nl_emb, direction="nl_to_lean",
            num_steps=args.num_steps, device=args.device
        )
        print(f"Output shape: {nl_transported.shape}")

        # Transport Lean -> NL
        print(f"\nTransporting Lean -> NL with {args.num_steps} steps...")
        lean_transported = transport_embeddings(
            model, lean_emb, direction="lean_to_nl",
            num_steps=args.num_steps, device=args.device
        )
        print(f"Output shape: {lean_transported.shape}")

        # Evaluate quality
        print(f"\nEvaluating transport quality...")
        metrics = evaluate_transport_quality(
            model, nl_emb, lean_emb,
            num_steps=args.num_steps, device=args.device
        )

        print("\nTransport Quality Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        print("\n✅ Demo complete!")

    elif args.mode == "evaluate":
        print("Evaluating on dataset...\n")

        # Load dataset
        dataset = EmbeddingPairDataset(args.nl_embeddings, args.lean_embeddings, max_len=256)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        # Evaluate on first batch
        nl_emb, lean_emb = next(iter(dataloader))

        metrics = evaluate_transport_quality(
            model, nl_emb, lean_emb,
            num_steps=args.num_steps, device=args.device
        )

        print("Transport Quality Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        print("\n✅ Evaluation complete!")

    elif args.mode == "batch_process":
        print("Batch processing dataset...\n")

        batch_process_dataset(
            model,
            args.nl_embeddings,
            args.lean_embeddings,
            args.output,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            device=args.device
        )

        print("\n✅ Batch processing complete!")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
