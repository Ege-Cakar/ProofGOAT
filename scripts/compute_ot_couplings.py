"""Precompute OT couplings between NL and Lean embeddings.

This script:
1. Loads NL and Lean embeddings from parquet files (streaming)
2. Computes OT couplings between each aligned pair using POT
3. Saves results incrementally in shards to avoid memory issues

Usage:
  python -m scripts.compute_ot_couplings --config project_config.yaml
"""
import argparse
import os
import yaml
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import ot
import json
import gc
from concurrent.futures import ThreadPoolExecutor

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


class ShardedOTWriter:
    """Incrementally save OT couplings in shards to avoid memory issues."""
    
    def __init__(self, output_dir: str, shard_size: int = 1000, config: dict = None):
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
        print(f"  Saved shard {self.current_shard} ({len(self.shard_data['ids'])} samples, total: {self.total_saved})")
        
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
        
        print(f"\nSaved {self.total_saved} samples in {self.current_shard} shards")
        print(f"Output directory: {self.output_dir}")
        
        return metadata


def load_and_align_embeddings(
    nl_path: str,
    lean_path: str,
    max_len: int = 256,
    max_samples: int = None,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load embeddings and return aligned data by ID."""
    
    # Load NL embeddings into dict by ID
    print("Loading NL embeddings...")
    nl_pf = pq.ParquetFile(nl_path)
    nl_by_id = {}
    
    pbar = tqdm(range(nl_pf.metadata.num_row_groups), desc="NL")
    for rg_idx in pbar:
        if max_samples and len(nl_by_id) >= max_samples:
            break
        
        df = nl_pf.read_row_group(rg_idx).to_pandas()
        for _, row in df.iterrows():
            if max_samples and len(nl_by_id) >= max_samples:
                break
            
            sample_id = row.get('id', str(len(nl_by_id)))
            
            if 'hidden' in row:
                hidden = row['hidden']
            elif 'embedding' in row:
                hidden = row['embedding']
            else:
                hidden = row.iloc[0]
            
            if isinstance(hidden, np.ndarray):
                if hidden.dtype == object:
                    emb = np.stack([np.asarray(x, dtype=np.float32) for x in hidden])
                else:
                    emb = hidden.astype(np.float32)
            else:
                emb = np.asarray(hidden, dtype=np.float32)
            
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            
            if max_len and emb.shape[0] > max_len:
                emb = emb[:max_len]
            
            nl_by_id[sample_id] = emb
        
        pbar.set_postfix({'loaded': len(nl_by_id)})
    
    print(f"  Loaded {len(nl_by_id)} NL embeddings")
    
    # Load Lean embeddings into dict by ID
    print("\nLoading Lean embeddings...")
    lean_pf = pq.ParquetFile(lean_path)
    lean_by_id = {}
    
    pbar = tqdm(range(lean_pf.metadata.num_row_groups), desc="Lean")
    for rg_idx in pbar:
        if max_samples and len(lean_by_id) >= max_samples:
            break
        
        df = lean_pf.read_row_group(rg_idx).to_pandas()
        for _, row in df.iterrows():
            if max_samples and len(lean_by_id) >= max_samples:
                break
            
            sample_id = row.get('id', str(len(lean_by_id)))
            
            if 'hidden' in row:
                hidden = row['hidden']
            elif 'embedding' in row:
                hidden = row['embedding']
            else:
                hidden = row.iloc[0]
            
            if isinstance(hidden, np.ndarray):
                if hidden.dtype == object:
                    emb = np.stack([np.asarray(x, dtype=np.float32) for x in hidden])
                else:
                    emb = hidden.astype(np.float32)
            else:
                emb = np.asarray(hidden, dtype=np.float32)
            
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            
            if max_len and emb.shape[0] > max_len:
                emb = emb[:max_len]
            
            lean_by_id[sample_id] = emb
        
        pbar.set_postfix({'loaded': len(lean_by_id)})
    
    print(f"  Loaded {len(lean_by_id)} Lean embeddings")
    
    # Find common IDs
    common_ids = sorted(set(nl_by_id.keys()) & set(lean_by_id.keys()))
    print(f"\nFound {len(common_ids)} aligned pairs")
    
    return common_ids, nl_by_id, lean_by_id


def main():
    parser = argparse.ArgumentParser(description="Precompute OT couplings (sharded)")
    parser.add_argument("--config", type=str, default="project_config.yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for shards")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--shard-size", type=int, default=1000, help="Samples per shard")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel workers for OT")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    neo_cfg = cfg.get("neural_ot", {})
    nl_path = neo_cfg.get("nl_embeddings")
    lean_path = neo_cfg.get("lean_embeddings")
    
    max_len = int(neo_cfg.get("max_len", 256))
    ot_reg = float(neo_cfg.get("ot_reg", 0.05))
    ot_method = neo_cfg.get("ot_method", "sinkhorn")
    ot_cost = neo_cfg.get("ot_cost", "euclidean")
    max_samples = args.max_samples or neo_cfg.get("max_samples", None)
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(neo_cfg.get("output_dir", "outputs/neural_ot"), "ot_shards")

    print("="*80)
    print("OT Coupling Precomputation (Sharded)")
    print("="*80)
    print(f"NL embeddings: {nl_path}")
    print(f"Lean embeddings: {lean_path}")
    print(f"Output directory: {output_dir}")
    print(f"Shard size: {args.shard_size}")
    print(f"OT method: {ot_method}, cost: {ot_cost}, reg: {ot_reg}")
    print(f"Max sequence length: {max_len}")
    print(f"Parallel workers: {args.num_workers}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print("="*80 + "\n")

    # Load and align embeddings
    common_ids, nl_by_id, lean_by_id = load_and_align_embeddings(
        nl_path, lean_path, max_len=max_len, max_samples=max_samples
    )
    
    if len(common_ids) == 0:
        print("ERROR: No aligned pairs found!")
        return
    
    # Initialize sharded writer
    ot_config = {
        'ot_method': ot_method,
        'ot_cost': ot_cost,
        'ot_reg': ot_reg,
        'max_len': max_len,
    }
    writer = ShardedOTWriter(output_dir, shard_size=args.shard_size, config=ot_config)
    
    # Compute OT couplings with progress bar
    print(f"\nComputing OT couplings for {len(common_ids)} pairs...")
    print(f"  (Saving every {args.shard_size} samples)\n")
    
    # Process in batches for parallel computation
    batch_size = args.num_workers * 4
    
    for batch_start in tqdm(range(0, len(common_ids), batch_size), desc="OT batches"):
        batch_end = min(batch_start + batch_size, len(common_ids))
        batch_ids = common_ids[batch_start:batch_end]
        
        # Prepare batch
        batch_args = []
        for i, sample_id in enumerate(batch_ids):
            nl_emb = nl_by_id[sample_id]
            lean_emb = lean_by_id[sample_id]
            batch_args.append((i, nl_emb, lean_emb, ot_cost, ot_reg, ot_method))
        
        # Compute OT in parallel
        if args.num_workers > 1:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                results = list(executor.map(_ot_worker, batch_args))
        else:
            results = [_ot_worker(args) for args in batch_args]
        
        # Sort by original index and add to writer
        results.sort(key=lambda x: x[0])
        for (idx, coupling), sample_id in zip(results, batch_ids):
            nl_emb = nl_by_id[sample_id]
            lean_emb = lean_by_id[sample_id]
            writer.add(sample_id, nl_emb, lean_emb, coupling)
        
        # Clear processed embeddings from memory to save RAM
        for sample_id in batch_ids:
            del nl_by_id[sample_id]
            del lean_by_id[sample_id]
        
        if batch_start % (batch_size * 10) == 0:
            gc.collect()
    
    # Finish and save metadata
    metadata = writer.finish()
    
    print("\n" + "="*80)
    print("Done!")
    print(f"Total samples: {metadata['total_samples']}")
    print(f"Total shards: {metadata['num_shards']}")
    print(f"Output: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
