"""Visualize OT coupling between NL and Lean tokens - Colab Version.

This version works with pre-computed embeddings stored in parquet files.

Usage in Colab:
    # 1. Upload this script and your data files to Colab
    # 2. Install dependencies
    !pip install pandas pyarrow POT matplotlib seaborn transformers
    
    # 3. Run the script
    !python visualize_ot_coupling_colab.py \
        --nl-embeddings /content/kimina17_all_nl_embeddings.parquet \
        --lean-embeddings /content/kimina17_all_lean_embeddings.parquet \
        --data-jsonl /content/herald_pairs.jsonl \
        --example-idx 0
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ot
from pathlib import Path
from transformers import AutoTokenizer
import json


def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_ot_coupling(x0, x1, cost="euclidean", reg=0.05, method="sinkhorn"):
    """Compute OT coupling between two point clouds using POT."""
    L0, L1 = x0.shape[0], x1.shape[0]
    
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
        P = ot.sinkhorn(a, b, C, reg=reg, numItermax=100)
    elif method == "emd":
        P = ot.emd(a, b, C)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return P.astype(np.float32)


def visualize_coupling(
    coupling,
    nl_tokens,
    lean_tokens,
    save_path,
    title="Token Alignment (OT Coupling)",
    max_tokens=50
):
    """Generate and save a heatmap of the coupling matrix."""
    # Truncate if too many tokens
    if len(nl_tokens) > max_tokens:
        coupling = coupling[:max_tokens, :]
        nl_tokens = nl_tokens[:max_tokens]
    if len(lean_tokens) > max_tokens:
        coupling = coupling[:, :max_tokens]
        lean_tokens = lean_tokens[:max_tokens]
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    sns.heatmap(
        coupling,
        xticklabels=lean_tokens,
        yticklabels=nl_tokens,
        cmap="YlOrRd",
        cbar_kws={'label': 'Transport Mass'},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Lean Tokens", fontsize=12)
    ax.set_ylabel("NL Tokens", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize OT coupling (Colab version)")
    parser.add_argument("--nl-embeddings", type=str, required=True, help="Path to NL embeddings parquet")
    parser.add_argument("--lean-embeddings", type=str, required=True, help="Path to Lean embeddings parquet")
    parser.add_argument("--data-jsonl", type=str, required=True, help="Path to herald_pairs.jsonl")
    parser.add_argument("--example-idx", type=int, default=0, help="Index of example to visualize")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to show in visualization")
    parser.add_argument("--ot-method", type=str, default="sinkhorn", choices=["sinkhorn", "emd"])
    parser.add_argument("--ot-cost", type=str, default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--ot-reg", type=float, default=0.05, help="Sinkhorn regularization")
    parser.add_argument("--model-name", type=str, default="AI-MO/Kimina-Prover-Distill-8B", help="Tokenizer model")
    args = parser.parse_args()

    print("="*80)
    print("Neural OT Token Visualization (Colab)")
    print("="*80)

    # 1. Load data
    print(f"\n📂 Loading data from {args.data_jsonl}...")
    pairs = load_jsonl(args.data_jsonl)
    
    if args.example_idx >= len(pairs):
        print(f"❌ Error: Example index {args.example_idx} out of range (max {len(pairs)-1})")
        return

    example = pairs[args.example_idx]
    example_id = example.get("id", f"example_{args.example_idx}")
    nl_text = example["nl_proof"]
    lean_text = example["lean_proof"]
    
    print(f"\n📝 Example {args.example_idx} (ID: {example_id}):")
    print(f"  NL:   {nl_text[:100]}...")
    print(f"  Lean: {lean_text[:100]}...")

    # 2. Load tokenizer (just for tokens, not the full model!)
    print(f"\n🔤 Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Tokenize to get token strings
    nl_tokens = tokenizer.tokenize(nl_text)
    lean_tokens = tokenizer.tokenize(lean_text)
    
    print(f"  NL tokens:   {len(nl_tokens)} tokens")
    print(f"  Lean tokens: {len(lean_tokens)} tokens")

    # 3. Load pre-computed embeddings
    print(f"\n📊 Loading embeddings...")
    print(f"  NL:   {args.nl_embeddings}")
    print(f"  Lean: {args.lean_embeddings}")
    
    nl_df = pd.read_parquet(args.nl_embeddings)
    lean_df = pd.read_parquet(args.lean_embeddings)
    
    # Find the example by ID
    nl_row = nl_df[nl_df['id'] == example_id]
    lean_row = lean_df[lean_df['id'] == example_id]
    
    if len(nl_row) == 0 or len(lean_row) == 0:
        print(f"❌ Error: Example ID '{example_id}' not found in embeddings")
        print(f"   Available IDs (first 5): {nl_df['id'].head().tolist()}")
        return
    
    # Extract embeddings
    nl_emb = np.array(nl_row.iloc[0]['hidden'], dtype=np.float32)
    lean_emb = np.array(lean_row.iloc[0]['hidden'], dtype=np.float32)
    
    # Handle different formats
    if nl_emb.ndim == 1:
        nl_emb = nl_emb.reshape(1, -1)
    if lean_emb.ndim == 1:
        lean_emb = lean_emb.reshape(1, -1)
    
    print(f"  NL embedding shape:   {nl_emb.shape}")
    print(f"  Lean embedding shape: {lean_emb.shape}")
    
    # Sanity check: token count should match embedding length
    if nl_emb.shape[0] != len(nl_tokens):
        print(f"⚠️  Warning: NL token count mismatch ({len(nl_tokens)} tokens vs {nl_emb.shape[0]} embeddings)")
        # Truncate to min
        min_len = min(len(nl_tokens), nl_emb.shape[0])
        nl_tokens = nl_tokens[:min_len]
        nl_emb = nl_emb[:min_len]
    
    if lean_emb.shape[0] != len(lean_tokens):
        print(f"⚠️  Warning: Lean token count mismatch ({len(lean_tokens)} tokens vs {lean_emb.shape[0]} embeddings)")
        min_len = min(len(lean_tokens), lean_emb.shape[0])
        lean_tokens = lean_tokens[:min_len]
        lean_emb = lean_emb[:min_len]

    # 4. Compute OT coupling
    print(f"\n🔄 Computing OT coupling...")
    print(f"  Method: {args.ot_method}, Cost: {args.ot_cost}, Reg: {args.ot_reg}")
    
    coupling = compute_ot_coupling(
        nl_emb,
        lean_emb,
        cost=args.ot_cost,
        reg=args.ot_reg,
        method=args.ot_method
    )
    
    print(f"  Coupling shape: {coupling.shape}")
    print(f"  Coupling sum: {coupling.sum():.4f} (should be ~1.0)")

    # 5. Visualize
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(args.output_dir) / f"coupling_{example_id}.png"
    
    print(f"\n🎨 Creating visualization...")
    visualize_coupling(
        coupling,
        nl_tokens,
        lean_tokens,
        str(save_path),
        title=f"Token Alignment - Example {example_id}",
        max_tokens=args.max_tokens
    )
    
    # 6. Print top matchings
    print(f"\n🔍 Top 10 Token Matchings:")
    flat_indices = np.argsort(coupling.flatten())[::-1][:10]
    for rank, flat_idx in enumerate(flat_indices, 1):
        i, j = np.unravel_index(flat_idx, coupling.shape)
        mass = coupling[i, j]
        if i < len(nl_tokens) and j < len(lean_tokens):
            print(f"  {rank}. '{nl_tokens[i]}' → '{lean_tokens[j]}' (mass: {mass:.4f})")
    
    print("\n✅ Done!")
    print("="*80)


if __name__ == "__main__":
    main()
