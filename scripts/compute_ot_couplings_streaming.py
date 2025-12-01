"""Streaming OT coupling computation for large parquet files.

This script processes parquet files row-group by row-group to avoid
loading all data into memory. Since both NL and Lean parquet files
have the same row groups with matching IDs, we can process them
in lockstep.

Memory usage: ~2-3 row groups at a time (configurable)

Usage:
  python -m scripts.compute_ot_couplings_streaming \
      --nl-path outputs/kimina17_all_nl_embeddings.parquet \
      --lean-path outputs/kimina17_all_lean_embeddings.parquet \
      --output-dir outputs/neural_ot/ot_shards \
      --shard-size 500
"""
import argparse
import os
import gc
import json
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import ot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(ROOT)

from src.io_utils import ensure_dir


def compute_single_ot_coupling(
    x0: np.ndarray,
    x1: np.ndarray,
    cost: str = "euclidean",
    reg: float = 0.05,
    method: str = "sinkhorn"
) -> np.ndarray:
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
    
    # Normalize cost matrix
    C = C / (C.max() + 1e-8)
    
    if method == "sinkhorn":
        P = ot.sinkhorn(a, b, C, reg=reg, numItermax=100)
    elif method == "emd":
        P = ot.emd(a, b, C)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return P.astype(np.float32)


def _ot_worker(args):
    """Worker function for parallel OT computation."""
    idx, nl_emb, lean_emb, cost, reg, method = args
    coupling = compute_single_ot_coupling(nl_emb, lean_emb, cost=cost, reg=reg, method=method)
    return idx, coupling


def parse_hidden_field(hidden, max_len: int = 256) -> np.ndarray:
    """Parse hidden field from parquet row into numpy array."""
    if isinstance(hidden, np.ndarray):
        if hidden.dtype == object:
            emb = np.stack([np.asarray(x, dtype=np.float32) for x in hidden])
        else:
            emb = hidden.astype(np.float32)
    else:
        # Could be a list of lists
        emb = np.asarray(hidden, dtype=np.float32)
    
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    
    if max_len and emb.shape[0] > max_len:
        emb = emb[:max_len]
    
    return emb


class StreamingOTWriter:
    """Incrementally save OT couplings in shards to avoid memory issues."""
    
    def __init__(self, output_dir: str, shard_size: int = 500, config: dict = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.config = config or {}
        
        # Current shard data
        self.current_shard = 0
        self.shard_data = {
            'ids': [],
            'nl_embeddings': [],
            'lean_embeddings': [],
            'couplings': [],
        }
        self.total_saved = 0
        
    def add(self, sample_id: str, nl_emb: np.ndarray, lean_emb: np.ndarray, coupling: np.ndarray):
        """Add a sample to the current shard."""
        self.shard_data['ids'].append(sample_id)
        self.shard_data['nl_embeddings'].append(nl_emb)
        self.shard_data['lean_embeddings'].append(lean_emb)
        self.shard_data['couplings'].append(coupling)
        
        if len(self.shard_data['ids']) >= self.shard_size:
            self._flush_shard()
    
    def _flush_shard(self):
        """Save current shard to disk and clear memory."""
        if len(self.shard_data['ids']) == 0:
            return
        
        shard_path = self.output_dir / f"shard_{self.current_shard:05d}.npz"
        
        np.savez_compressed(
            shard_path,
            ids=np.array(self.shard_data['ids'], dtype=object),
            nl_embeddings=np.array(self.shard_data['nl_embeddings'], dtype=object),
            lean_embeddings=np.array(self.shard_data['lean_embeddings'], dtype=object),
            couplings=np.array(self.shard_data['couplings'], dtype=object),
        )
        
        self.total_saved += len(self.shard_data['ids'])
        print(f"  💾 Saved shard {self.current_shard} ({len(self.shard_data['ids'])} samples, total: {self.total_saved})")
        
        # Clear memory
        self.shard_data = {
            'ids': [],
            'nl_embeddings': [],
            'lean_embeddings': [],
            'couplings': [],
        }
        self.current_shard += 1
        gc.collect()
    
    def finish(self):
        """Flush remaining data and save metadata."""
        self._flush_shard()
        
        # Save metadata
        metadata = {
            'num_shards': self.current_shard,
            'total_samples': self.total_saved,
            'shard_size': self.shard_size,
            'config': self.config,
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Saved {self.total_saved} samples in {self.current_shard} shards")
        print(f"📁 Output directory: {self.output_dir}")
        
        return metadata


def verify_alignment(nl_pf: pq.ParquetFile, lean_pf: pq.ParquetFile, sample_row_groups: int = 5):
    """Verify that NL and Lean parquet files have matching row groups."""
    print("\n🔍 Verifying parquet file alignment...")
    
    if nl_pf.metadata.num_row_groups != lean_pf.metadata.num_row_groups:
        raise ValueError(
            f"Row group count mismatch: NL has {nl_pf.metadata.num_row_groups}, "
            f"Lean has {lean_pf.metadata.num_row_groups}"
        )
    
    if nl_pf.metadata.num_rows != lean_pf.metadata.num_rows:
        raise ValueError(
            f"Total row count mismatch: NL has {nl_pf.metadata.num_rows}, "
            f"Lean has {lean_pf.metadata.num_rows}"
        )
    
    # Sample a few row groups to verify IDs match
    num_rgs = nl_pf.metadata.num_row_groups
    sample_indices = [0, num_rgs // 4, num_rgs // 2, 3 * num_rgs // 4, num_rgs - 1]
    sample_indices = list(set(min(i, num_rgs - 1) for i in sample_indices))[:sample_row_groups]
    
    for rg_idx in sample_indices:
        nl_ids = nl_pf.read_row_group(rg_idx, columns=['id'])['id'].to_pylist()
        lean_ids = lean_pf.read_row_group(rg_idx, columns=['id'])['id'].to_pylist()
        
        if nl_ids != lean_ids:
            raise ValueError(f"ID mismatch in row group {rg_idx}")
    
    print(f"  ✅ Verified {len(sample_indices)} row groups - all IDs match!")
    print(f"  📊 Total: {nl_pf.metadata.num_rows} rows in {num_rgs} row groups")
    return True


def process_row_group(
    rg_idx: int,
    nl_pf: pq.ParquetFile,
    lean_pf: pq.ParquetFile,
    max_len: int,
    ot_cost: str,
    ot_reg: float,
    ot_method: str,
    num_workers: int,
) -> list:
    """Process a single row group and return list of (id, nl_emb, lean_emb, coupling)."""
    # Read row group from both files
    nl_table = nl_pf.read_row_group(rg_idx)
    lean_table = lean_pf.read_row_group(rg_idx)
    
    nl_ids = nl_table['id'].to_pylist()
    lean_ids = lean_table['id'].to_pylist()
    
    # Verify alignment (should always match if verify_alignment passed)
    assert nl_ids == lean_ids, f"ID mismatch in row group {rg_idx}"
    
    nl_hidden = nl_table['hidden'].to_pylist()
    lean_hidden = lean_table['hidden'].to_pylist()
    
    # Parse embeddings
    nl_embeddings = [parse_hidden_field(h, max_len) for h in nl_hidden]
    lean_embeddings = [parse_hidden_field(h, max_len) for h in lean_hidden]
    
    # Prepare OT computation arguments
    ot_args = [
        (i, nl_embeddings[i], lean_embeddings[i], ot_cost, ot_reg, ot_method)
        for i in range(len(nl_ids))
    ]
    
    # Compute OT couplings in parallel
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_ot_worker, ot_args))
    else:
        results = [_ot_worker(args) for args in ot_args]
    
    # Sort by index
    results.sort(key=lambda x: x[0])
    
    # Build output
    output = []
    for i, sample_id in enumerate(nl_ids):
        _, coupling = results[i]
        output.append((sample_id, nl_embeddings[i], lean_embeddings[i], coupling))
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Streaming OT coupling computation")
    parser.add_argument("--nl-path", type=str, required=True, help="Path to NL embeddings parquet")
    parser.add_argument("--lean-path", type=str, required=True, help="Path to Lean embeddings parquet")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for shards")
    parser.add_argument("--shard-size", type=int, default=500, help="Samples per shard")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel workers for OT computation")
    parser.add_argument("--max-len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--ot-method", type=str, default="sinkhorn", choices=["sinkhorn", "emd"])
    parser.add_argument("--ot-cost", type=str, default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--ot-reg", type=float, default=0.05, help="Sinkhorn regularization")
    parser.add_argument("--start-rg", type=int, default=0, help="Starting row group (for resuming)")
    parser.add_argument("--max-rgs", type=int, default=None, help="Max row groups to process")
    args = parser.parse_args()

    print("=" * 80)
    print("Streaming OT Coupling Computation")
    print("=" * 80)
    print(f"NL embeddings:  {args.nl_path}")
    print(f"Lean embeddings: {args.lean_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Shard size: {args.shard_size}")
    print(f"OT config: method={args.ot_method}, cost={args.ot_cost}, reg={args.ot_reg}")
    print(f"Max sequence length: {args.max_len}")
    print(f"Parallel workers: {args.num_workers}")
    print("=" * 80)

    # Open parquet files
    nl_pf = pq.ParquetFile(args.nl_path)
    lean_pf = pq.ParquetFile(args.lean_path)
    
    # Verify alignment
    verify_alignment(nl_pf, lean_pf)
    
    # Setup writer
    ot_config = {
        'ot_method': args.ot_method,
        'ot_cost': args.ot_cost,
        'ot_reg': args.ot_reg,
        'max_len': args.max_len,
        'nl_path': args.nl_path,
        'lean_path': args.lean_path,
    }
    writer = StreamingOTWriter(args.output_dir, shard_size=args.shard_size, config=ot_config)
    
    # Process row groups
    num_rgs = nl_pf.metadata.num_row_groups
    end_rg = num_rgs if args.max_rgs is None else min(args.start_rg + args.max_rgs, num_rgs)
    
    print(f"\n🚀 Processing row groups {args.start_rg} to {end_rg - 1}...")
    
    pbar = tqdm(range(args.start_rg, end_rg), desc="Row groups")
    for rg_idx in pbar:
        # Process this row group
        results = process_row_group(
            rg_idx, nl_pf, lean_pf,
            max_len=args.max_len,
            ot_cost=args.ot_cost,
            ot_reg=args.ot_reg,
            ot_method=args.ot_method,
            num_workers=args.num_workers,
        )
        
        # Add to writer
        for sample_id, nl_emb, lean_emb, coupling in results:
            writer.add(sample_id, nl_emb, lean_emb, coupling)
        
        # Free memory
        del results
        gc.collect()
        
        pbar.set_postfix({'saved': writer.total_saved})
    
    # Finish
    metadata = writer.finish()
    
    print("\n" + "=" * 80)
    print("✅ Done!")
    print(f"Total samples: {metadata['total_samples']}")
    print(f"Total shards: {metadata['num_shards']}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
