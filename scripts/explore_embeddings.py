"""
Look at outputs/lean_embeddings.parquet and print metadata on it (such as number of them and their corresponding lean/natural text)
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

# Resolve project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.io_utils import read_jsonl


def explore_embeddings(embeddings_path: str, pairs_path: str = None):
    """
    Explore embeddings parquet file and display metadata.

    Args:
        embeddings_path: Path to the embeddings parquet file
        pairs_path: Optional path to pairs.jsonl to show corresponding text
    """

    # Check if file exists
    if not os.path.exists(embeddings_path):
        print(f"Error: File not found: {embeddings_path}")
        return
    # Read parquet file
    print("\n📊 Reading parquet file...")
    try:
        df = pd.read_parquet(embeddings_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    # Basic metadata
    print("\n" + "=" * 80)
    print("BASIC METADATA")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Column information
    print("\n" + "=" * 80)
    print("COLUMN INFORMATION")
    print("=" * 80)
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:")
    for col in df.columns:
        dtype = df[col].dtype
        print(f"  {col}: {dtype}")
        # Check if column contains arrays/lists
        if dtype == 'object':
            sample_val = df[col].iloc[0] if len(df) > 0 else None
            if sample_val is not None:
                if isinstance(sample_val, (list, np.ndarray)):
                    print(f"    -> Contains arrays of length {len(sample_val)}")
                elif isinstance(sample_val, (np.ndarray,)):
                    print(f"    -> Contains numpy arrays of shape {sample_val.shape}")

    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    # Only show statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("No numeric columns found for statistics.")

    # Sample rows
    print("\n" + "=" * 80)
    print("SAMPLE ROWS")
    print("=" * 80)
    print(df.head())

    # If pairs file is provided, show corresponding text
    if pairs_path and os.path.exists(pairs_path):
        print("\n" + "=" * 80)
        print("CORRESPONDING TEXT (from pairs.jsonl)")
        print("=" * 80)

        pairs = read_jsonl(pairs_path)

        # Get the first few pairs
        num_samples = min(5, len(df), len(pairs))
        print(f"\nShowing first {num_samples} embeddings with corresponding text:\n")

        for i in range(num_samples):
            print("-" * 80)
            print(f"Embedding {i+1}:")

            if i < len(pairs):
                pair = pairs[i]
                print(f"  ID: {pair.get('id', 'N/A')}")
                nl_text = pair.get('nl_text', 'N/A') if pair.get('nl_text') else 'N/A'
                print(f"  NL Text: {nl_text[:200]}")
                nl_proof = pair.get('nl_proof', 'N/A')
                print(f"  NL Proof: {nl_proof[:200]}")
                lean_text = pair.get('lean_text', 'N/A')
                print(f"  Lean Text: {lean_text[:200]}")

            # Show embedding row info
            if i < len(df):
                row = df.iloc[i]

                # Check if embeddings are stored as arrays in a single column
                first_col = df.columns[0] if len(df.columns) > 0 else None
                if first_col is not None and df[first_col].dtype == 'object':
                    embedding = row[first_col]
                    if isinstance(embedding, (list, np.ndarray)):
                        embedding = np.array(embedding)
                        if embedding.ndim > 0:
                            print(f"  Embedding dimension: {len(embedding)}")
                            print(f"  Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}, std={embedding.std():.4f}")
                # Check if embeddings are stored as multiple columns (one per dimension)
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        embedding = row[numeric_cols].values
                        print(f"  Embedding dimension: {len(embedding)}")
                        print(f"  Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}, std={embedding.std():.4f}")
                    else:
                        # Show column names and sample values
                        print(f"  Columns: {list(df.columns)}")
                        print(f"  Sample row values: {dict(list(row.items())[:3])}")
            print()

    # Additional analysis if embeddings are stored as columns
    if len(df.columns) > 0:
        print("\n" + "=" * 80)
        print("EMBEDDING DIMENSION ANALYSIS")
        print("=" * 80)

        # Check if columns look like embedding dimensions (numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"Number of numeric columns (potential embedding dimensions): {len(numeric_cols)}")
            if len(numeric_cols) > 0:
                print(f"Sample column names: {list(numeric_cols[:5])}")
                print(f"Embedding dimension per row: {len(numeric_cols)}")
        else:
            print("No numeric columns found. Embeddings might be stored differently.")

    print("\n" + "=" * 80)
    print("Exploration complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Explore embeddings parquet file")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="outputs/lean_embeddings.parquet",
        help="Path to embeddings parquet file (default: outputs/lean_embeddings.parquet)"
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="data/pairs.jsonl",
        help="Path to pairs.jsonl file (default: data/pairs.jsonl)"
    )
    parser.add_argument(
        "--nl-embeddings",
        action="store_true",
        help="Explore NL embeddings instead of Lean embeddings"
    )
    args = parser.parse_args()
    # Determine embeddings path
    if args.nl_embeddings:
        embeddings_path = "outputs/nl_embeddings.parquet"
    else:
        embeddings_path = args.embeddings
    # Resolve paths relative to project root
    embeddings_path = os.path.join(ROOT, embeddings_path)
    pairs_path = os.path.join(ROOT, args.pairs)
    explore_embeddings(embeddings_path, pairs_path)


if __name__ == "__main__":
    main()
