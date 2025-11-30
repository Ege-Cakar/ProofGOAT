"""Visualize OT coupling between NL and Lean tokens.

This script:
1. Loads a sample from the dataset.
2. Computes embeddings using the specified model (or loads them).
3. Computes the Optimal Transport (OT) coupling matrix.
4. Generates a heatmap visualization of the token alignments.

Usage:
    python -m scripts.visualize_ot_coupling --example-idx 0 --compute-embeddings
"""
import argparse
import os
import sys
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Resolve project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.io_utils import read_jsonl, ensure_dir
from src.model_utils import load_model_and_tokenizer
from src.alignment import extract_hidden_states
from scripts.compute_ot_couplings import compute_single_ot_coupling


def visualize_coupling(
    coupling: np.ndarray,
    nl_tokens: list[str],
    lean_tokens: list[str],
    save_path: str,
    title: str = "Token Alignment (OT Coupling)"
):
    """Generate and save a heatmap of the coupling matrix."""
    plt.figure(figsize=(12, 10))
    
    # Normalize for better visualization if needed, but raw coupling is usually fine
    # Log scale can help see small values: np.log(coupling + 1e-8)
    
    sns.heatmap(
        coupling,
        xticklabels=lean_tokens,
        yticklabels=nl_tokens,
        cmap="viridis",
        cbar_kws={'label': 'Coupling Mass'}
    )
    
    plt.title(title)
    plt.xlabel("Lean Tokens")
    plt.ylabel("NL Tokens")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize OT coupling")
    parser.add_argument("--config", type=str, default="project_config.yaml")
    parser.add_argument("--example-idx", type=int, default=0, help="Index of example to visualize")
    parser.add_argument("--compute-embeddings", action="store_true", help="Compute embeddings on the fly")
    parser.add_argument("--max-len", type=int, default=32, help="Max sequence length for visualization")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Load Data
    data_path = cfg["data"]["input_jsonl"]
    print(f"Loading data from {data_path}...")
    pairs = read_jsonl(data_path)
    
    if args.example_idx >= len(pairs):
        print(f"Error: Example index {args.example_idx} out of range (max {len(pairs)-1})")
        return

    example = pairs[args.example_idx]
    nl_text = example["nl_proof"]
    lean_text = example["lean_proof"]
    
    print(f"\nExample {args.example_idx}:")
    print(f"NL (first 50 chars): {nl_text[:50]}...")
    print(f"Lean (first 50 chars): {lean_text[:50]}...")

    # 2. Get Embeddings & Tokens
    if args.compute_embeddings:
        print("\nLoading model to compute embeddings...")
        # Assuming same model for both for simplicity, or use config
        model_name = cfg["models"]["nl_model"]
        model, tokenizer = load_model_and_tokenizer(model_name, fp16=True)
        
        # Helper to get tokens and embeddings
        def get_data(text, layer):
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=args.max_len
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            # Get hidden states
            h = outputs.hidden_states[layer].squeeze(0).cpu().numpy().astype(np.float32)
            
            # Get tokens
            tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            # Clean up tokens (remove GPT-style Ġ)
            tokens = [t.replace('Ġ', '') for t in tokens]
            
            return h, tokens

        print("Computing NL embeddings...")
        nl_emb, nl_tokens = get_data(nl_text, cfg["extract"]["nl_layer"])
        
        print("Computing Lean embeddings...")
        lean_emb, lean_tokens = get_data(lean_text, cfg["extract"]["lean_layer"])
        
    else:
        # TODO: Implement loading from parquet if needed, but on-the-fly is safer for single example
        print("Error: Please use --compute-embeddings for now as parquet lookup is complex.")
        return

    print(f"\nEmbeddings shape: NL {nl_emb.shape}, Lean {lean_emb.shape}")

    # 3. Compute OT Coupling
    print("\nComputing OT coupling...")
    ot_method = cfg["neural_ot"].get("ot_method", "sinkhorn")
    ot_cost = cfg["neural_ot"].get("ot_cost", "euclidean")
    ot_reg = float(cfg["neural_ot"].get("ot_reg", 0.05))
    
    coupling = compute_single_ot_coupling(
        nl_emb, 
        lean_emb, 
        cost=ot_cost, 
        reg=ot_reg, 
        method=ot_method
    )
    
    print(f"Coupling shape: {coupling.shape}")

    # 4. Visualize
    out_dir = Path("outputs/visualizations")
    ensure_dir(out_dir)
    save_path = out_dir / f"coupling_ex{args.example_idx}.png"
    
    visualize_coupling(
        coupling,
        nl_tokens,
        lean_tokens,
        str(save_path),
        title=f"Token Alignment (Example {args.example_idx})"
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
