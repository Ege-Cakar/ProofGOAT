"""Compute Augmented OT couplings for A-OT-CFM.

This implements the full Augmented Optimal Transport Flow Matching framework:
1. Fixed container N_max with existence channel
2. Distributed void positioning
3. 4-block cost matrix (Active↔Active, Active↔Void, Void↔Active, Void↔Void)
4. Soft ordering via positional penalty

GPU-accelerated using PyTorch backend for POT's Sinkhorn.

Usage:
  python -m scripts.compute_augmented_ot --config project_config.yaml
"""
import argparse
import os
import yaml
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path
import ot
import json
import gc
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(ROOT)

from src.io_utils import ensure_dir

# Check for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def get_sinusoidal_embedding(positions: np.ndarray, d: int) -> np.ndarray:
    """Generate sinusoidal positional embeddings.
    
    Args:
        positions: [N] array of positions in [0, 1]
        d: embedding dimension
    
    Returns:
        [N, d] positional embeddings
    """
    N = len(positions)
    pe = np.zeros((N, d), dtype=np.float32)
    
    div_term = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))
    
    for i, p in enumerate(positions):
        pe[i, 0::2] = np.sin(p * np.pi * div_term)
        pe[i, 1::2] = np.cos(p * np.pi * div_term[:d//2] if d % 2 == 0 else div_term[:(d+1)//2])
    
    return pe


def create_augmented_state(
    embeddings: np.ndarray,  # [L, d] active embeddings
    N_max: int,
    d: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create augmented state with existence channel and distributed voids.
    
    Returns:
        state: [N_max, d+1] augmented state (content + existence)
        positions: [N_max] positions in [0, 1]
        is_active: [N_max] boolean mask
    """
    L = embeddings.shape[0]
    n_void = N_max - L
    
    # Distributed positioning
    # Active tokens: positions 1/L, 2/L, ..., 1
    active_positions = np.linspace(1/L, 1.0, L) if L > 0 else np.array([])
    
    # Void tokens: distributed uniformly in [0, 1]
    if n_void > 0:
        void_positions = np.linspace(1/(n_void+1), n_void/(n_void+1), n_void)
    else:
        void_positions = np.array([])
    
    # Interleave by sorting all positions
    all_positions = np.concatenate([active_positions, void_positions])
    all_indices = np.argsort(all_positions)
    
    # Build state array
    state = np.zeros((N_max, d + 1), dtype=np.float32)
    positions = np.zeros(N_max, dtype=np.float32)
    is_active = np.zeros(N_max, dtype=bool)
    
    # Get positional embeddings
    active_pe = get_sinusoidal_embedding(active_positions, d) if L > 0 else np.zeros((0, d))
    void_pe = get_sinusoidal_embedding(void_positions, d) if n_void > 0 else np.zeros((0, d))
    
    active_idx = 0
    void_idx = 0
    
    for out_idx, orig_idx in enumerate(all_indices):
        if orig_idx < L:
            # Active token
            state[out_idx, :d] = embeddings[active_idx] + active_pe[active_idx]
            state[out_idx, d] = 1.0  # existence = 1
            positions[out_idx] = active_positions[active_idx]
            is_active[out_idx] = True
            active_idx += 1
        else:
            # Void token
            state[out_idx, :d] = void_pe[void_idx]  # zero content + position
            state[out_idx, d] = 0.0  # existence = 0
            positions[out_idx] = void_positions[void_idx]
            is_active[out_idx] = False
            void_idx += 1
    
    return state, positions, is_active


def compute_augmented_ot_cost_torch(
    src_positions: torch.Tensor,  # [N_max]
    tgt_positions: torch.Tensor,  # [N_max]
    src_active: torch.Tensor,     # [N_max] bool
    tgt_active: torch.Tensor,     # [N_max] bool
    src_embeddings: torch.Tensor, # [N_max, d] (without existence channel)
    tgt_embeddings: torch.Tensor, # [N_max, d]
    lambda_pos: float = 1.0,    # positional penalty weight
    alpha_delete: float = 1.0,  # deletion cost
    alpha_create: float = 1.0,  # creation cost
    beta_create: float = 0.1,   # creation norm penalty
) -> torch.Tensor:
    """Compute 4-block augmented OT cost matrix (GPU VECTORIZED).
    
    Blocks:
    - Active→Active: ||a_i - b_j||^2 + λ(p_i - p_j)^2
    - Active→Void: α_delete + λ(p_i - p_j)^2
    - Void→Active: α_create + β||b_j||^2 + λ(p_i - p_j)^2
    - Void→Void: ≈ 0
    """
    N = len(src_positions)
    
    # Positional cost matrix [N, N]
    pos_diff = src_positions[:, None] - tgt_positions[None, :]
    pos_cost = lambda_pos * (pos_diff ** 2)
    
    # Semantic cost matrix [N, N] (squared Euclidean)
    src_norm_sq = torch.sum(src_embeddings ** 2, dim=1)  # [N]
    tgt_norm_sq = torch.sum(tgt_embeddings ** 2, dim=1)  # [N]
    semantic_cost = src_norm_sq[:, None] + tgt_norm_sq[None, :] - 2 * (src_embeddings @ tgt_embeddings.T)
    semantic_cost = torch.clamp(semantic_cost, min=0)  # numerical stability
    
    # Block masks [N, N]
    src_active_2d = src_active[:, None]  # [N, 1]
    tgt_active_2d = tgt_active[None, :]  # [1, N]
    
    mask_aa = src_active_2d & tgt_active_2d      # Active → Active
    mask_av = src_active_2d & ~tgt_active_2d     # Active → Void
    mask_va = ~src_active_2d & tgt_active_2d     # Void → Active
    mask_vv = ~src_active_2d & ~tgt_active_2d    # Void → Void
    
    # Build cost matrix
    C = torch.zeros((N, N), dtype=torch.float64, device=src_positions.device)
    
    # Block 1: Active → Active
    C = C + mask_aa * (semantic_cost + pos_cost)
    
    # Block 2: Active → Void (deletion)
    C = C + mask_av * (alpha_delete + pos_cost)
    
    # Block 3: Void → Active (creation)
    creation_cost = alpha_create + beta_create * tgt_norm_sq[None, :]
    C = C + mask_va * (creation_cost + pos_cost)
    
    # Block 4: Void → Void (near zero)
    C = C + mask_vv * (0.01 * pos_cost)
    
    return C


def compute_augmented_ot_coupling_gpu(
    src_state: np.ndarray,      # [N_max, d+1]
    tgt_state: np.ndarray,      # [N_max, d+1]
    src_positions: np.ndarray,  # [N_max]
    tgt_positions: np.ndarray,  # [N_max]
    src_active: np.ndarray,     # [N_max] bool
    tgt_active: np.ndarray,     # [N_max] bool
    reg: float = 0.05,
    lambda_pos: float = 1.0,
    alpha_delete: float = 1.0,
    alpha_create: float = 1.0,
    beta_create: float = 0.1,
    device: torch.device = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute augmented OT coupling on GPU and extract permutation.
    
    Uses PyTorch backend for POT's Sinkhorn algorithm.
    
    Returns:
        coupling: [N_max, N_max] transport plan (numpy)
        perm: [N_max] permutation (src index i maps to tgt index perm[i])
    """
    if device is None:
        device = DEVICE
    
    N = src_state.shape[0]
    d = src_state.shape[1] - 1
    
    # Move to GPU
    src_emb = torch.tensor(src_state[:, :d], dtype=torch.float64, device=device)
    tgt_emb = torch.tensor(tgt_state[:, :d], dtype=torch.float64, device=device)
    src_pos_t = torch.tensor(src_positions, dtype=torch.float64, device=device)
    tgt_pos_t = torch.tensor(tgt_positions, dtype=torch.float64, device=device)
    src_act_t = torch.tensor(src_active, dtype=torch.bool, device=device)
    tgt_act_t = torch.tensor(tgt_active, dtype=torch.bool, device=device)
    
    # Compute 4-block cost matrix on GPU
    C = compute_augmented_ot_cost_torch(
        src_pos_t, tgt_pos_t,
        src_act_t, tgt_act_t,
        src_emb, tgt_emb,
        lambda_pos=lambda_pos,
        alpha_delete=alpha_delete,
        alpha_create=alpha_create,
        beta_create=beta_create,
    )
    
    # Normalize cost
    C = C / (C.max() + 1e-8)
    
    # Uniform marginals (on GPU)
    a = torch.ones(N, dtype=torch.float64, device=device) / N
    b = torch.ones(N, dtype=torch.float64, device=device) / N
    
    # Solve OT on GPU using sinkhorn_log (recommended for GPU)
    # POT auto-detects PyTorch tensors and uses GPU backend
    coupling = ot.bregman.sinkhorn_log(a, b, C, reg=reg, numItermax=200)
    
    # Extract permutation (argmax of each row)
    perm = torch.argmax(coupling, dim=1)
    
    # Move back to CPU/numpy
    coupling_np = coupling.float().cpu().numpy()
    perm_np = perm.cpu().numpy()
    
    return coupling_np, perm_np


def compute_augmented_ot_coupling(
    src_state: np.ndarray,      # [N_max, d+1]
    tgt_state: np.ndarray,      # [N_max, d+1]
    src_positions: np.ndarray,  # [N_max]
    tgt_positions: np.ndarray,  # [N_max]
    src_active: np.ndarray,     # [N_max] bool
    tgt_active: np.ndarray,     # [N_max] bool
    reg: float = 0.05,
    lambda_pos: float = 1.0,
    alpha_delete: float = 1.0,
    alpha_create: float = 1.0,
    beta_create: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute augmented OT coupling and extract permutation.
    
    Returns:
        coupling: [N_max, N_max] transport plan
        perm: [N_max] permutation (src index i maps to tgt index perm[i])
    """
    N = src_state.shape[0]
    d = src_state.shape[1] - 1
    
    # Extract embeddings without existence channel
    src_emb = src_state[:, :d]
    tgt_emb = tgt_state[:, :d]
    
    # Compute 4-block cost matrix
    C = compute_augmented_ot_cost(
        src_positions, tgt_positions,
        src_active, tgt_active,
        src_emb, tgt_emb,
        lambda_pos=lambda_pos,
        alpha_delete=alpha_delete,
        alpha_create=alpha_create,
        beta_create=beta_create,
    )
    
    # Normalize cost
    C = C / (C.max() + 1e-8)
    
    # Uniform marginals
    a = np.ones(N, dtype=np.float64) / N
    b = np.ones(N, dtype=np.float64) / N
    
    # Solve OT
    coupling = ot.sinkhorn(a, b, C, reg=reg, numItermax=200)
    
    # Extract permutation (argmax of each row)
    perm = np.argmax(coupling, axis=1)
    
    return coupling.astype(np.float32), perm


class AugmentedOTWriter:
    """Incrementally save augmented OT data in shards."""
    
    def __init__(self, output_dir: str, shard_size: int = 500, config: dict = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.config = config or {}
        
        self.current_shard = 0
        self.shard_data = {
            'ids': [],
            'src_states': [],      # [N_max, d+1]
            'tgt_states': [],      # [N_max, d+1]
            'src_positions': [],   # [N_max]
            'tgt_positions': [],   # [N_max]
            'src_active': [],      # [N_max] bool
            'tgt_active': [],      # [N_max] bool
            'permutations': [],    # [N_max] int
        }
        self.total_saved = 0
        
    def add(self, sample_id: str, src_state: np.ndarray, tgt_state: np.ndarray,
            src_pos: np.ndarray, tgt_pos: np.ndarray,
            src_active: np.ndarray, tgt_active: np.ndarray, perm: np.ndarray):
        """Add a sample to the current shard."""
        self.shard_data['ids'].append(sample_id)
        self.shard_data['src_states'].append(src_state)
        self.shard_data['tgt_states'].append(tgt_state)
        self.shard_data['src_positions'].append(src_pos)
        self.shard_data['tgt_positions'].append(tgt_pos)
        self.shard_data['src_active'].append(src_active)
        self.shard_data['tgt_active'].append(tgt_active)
        self.shard_data['permutations'].append(perm)
        
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
            src_states=np.stack(self.shard_data['src_states']),
            tgt_states=np.stack(self.shard_data['tgt_states']),
            src_positions=np.stack(self.shard_data['src_positions']),
            tgt_positions=np.stack(self.shard_data['tgt_positions']),
            src_active=np.stack(self.shard_data['src_active']),
            tgt_active=np.stack(self.shard_data['tgt_active']),
            permutations=np.stack(self.shard_data['permutations']),
        )
        
        self.total_saved += len(self.shard_data['ids'])
        print(f"  Saved shard {self.current_shard} ({len(self.shard_data['ids'])} samples, total: {self.total_saved})")
        
        # Clear memory
        self.shard_data = {k: [] for k in self.shard_data}
        self.current_shard += 1
        gc.collect()
    
    def finish(self):
        """Flush remaining data and save metadata."""
        self._flush_shard()
        
        metadata = {
            'num_shards': self.current_shard,
            'total_samples': self.total_saved,
            'shard_size': self.shard_size,
            'config': self.config,
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nSaved {self.total_saved} samples in {self.current_shard} shards")
        return metadata


def stream_paired_embeddings(nl_path: str, lean_path: str, max_len: int, max_samples: int = None):
    """Stream paired embeddings by ID, yielding one pair at a time."""
    
    # First pass: build index of Lean embeddings by ID
    print("Indexing Lean embeddings...")
    lean_pf = pq.ParquetFile(lean_path)
    lean_index = {}  # id -> (row_group, row_in_group)
    
    for rg_idx in tqdm(range(lean_pf.metadata.num_row_groups), desc="Indexing Lean"):
        df = lean_pf.read_row_group(rg_idx).to_pandas()
        for row_idx, row in df.iterrows():
            sample_id = row.get('id', str(len(lean_index)))
            lean_index[sample_id] = (rg_idx, row_idx, row)
    
    print(f"  Indexed {len(lean_index)} Lean embeddings")
    
    # Second pass: stream NL and yield pairs
    print("\nStreaming NL embeddings and yielding pairs...")
    nl_pf = pq.ParquetFile(nl_path)
    count = 0
    
    for rg_idx in tqdm(range(nl_pf.metadata.num_row_groups), desc="Processing"):
        if max_samples and count >= max_samples:
            break
        
        df = nl_pf.read_row_group(rg_idx).to_pandas()
        
        for _, row in df.iterrows():
            if max_samples and count >= max_samples:
                break
            
            sample_id = row.get('id', str(count))
            
            # Skip if no matching Lean
            if sample_id not in lean_index:
                continue
            
            # Extract NL embedding
            if 'hidden' in row:
                nl_hidden = row['hidden']
            elif 'embedding' in row:
                nl_hidden = row['embedding']
            else:
                nl_hidden = row.iloc[0]
            
            if isinstance(nl_hidden, np.ndarray):
                if nl_hidden.dtype == object:
                    nl_emb = np.stack([np.asarray(x, dtype=np.float32) for x in nl_hidden])
                else:
                    nl_emb = nl_hidden.astype(np.float32)
            else:
                nl_emb = np.asarray(nl_hidden, dtype=np.float32)
            
            if nl_emb.ndim == 1:
                nl_emb = nl_emb.reshape(1, -1)
            if nl_emb.shape[0] > max_len:
                nl_emb = nl_emb[:max_len]
            
            # Extract Lean embedding
            _, _, lean_row = lean_index[sample_id]
            
            if 'hidden' in lean_row:
                lean_hidden = lean_row['hidden']
            elif 'embedding' in lean_row:
                lean_hidden = lean_row['embedding']
            else:
                lean_hidden = lean_row.iloc[0]
            
            if isinstance(lean_hidden, np.ndarray):
                if lean_hidden.dtype == object:
                    lean_emb = np.stack([np.asarray(x, dtype=np.float32) for x in lean_hidden])
                else:
                    lean_emb = lean_hidden.astype(np.float32)
            else:
                lean_emb = np.asarray(lean_hidden, dtype=np.float32)
            
            if lean_emb.ndim == 1:
                lean_emb = lean_emb.reshape(1, -1)
            if lean_emb.shape[0] > max_len:
                lean_emb = lean_emb[:max_len]
            
            count += 1
            yield sample_id, nl_emb, lean_emb
    
    print(f"\nProcessed {count} pairs")


def main():
    parser = argparse.ArgumentParser(description="Compute Augmented OT for A-OT-CFM")
    parser.add_argument("--config", type=str, default="project_config.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=5000, help="Max training samples")
    parser.add_argument("--val-samples", type=int, default=500, help="Validation samples")
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--n-max", type=int, default=256, help="Fixed container size")
    parser.add_argument("--lambda-pos", type=float, default=1.0, help="Positional penalty")
    parser.add_argument("--alpha-delete", type=float, default=1.0, help="Deletion cost")
    parser.add_argument("--alpha-create", type=float, default=1.0, help="Creation cost")
    parser.add_argument("--beta-create", type=float, default=0.1, help="Creation norm penalty")
    parser.add_argument("--reg", type=float, default=0.1, help="Sinkhorn regularization")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (disable GPU)")
    args = parser.parse_args()

    # Set device
    global DEVICE
    if args.cpu:
        DEVICE = torch.device('cpu')
        print("Forcing CPU mode")
    else:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if DEVICE.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    neo_cfg = cfg.get("neural_ot", {})
    nl_path = neo_cfg.get("nl_embeddings")
    lean_path = neo_cfg.get("lean_embeddings")
    
    max_len = args.n_max
    total_samples = args.max_samples + args.val_samples
    
    base_output_dir = args.output_dir or os.path.join(
        neo_cfg.get("output_dir", "outputs/neural_ot"), "augmented_ot_shards"
    )
    train_dir = os.path.join(base_output_dir, "train")
    val_dir = os.path.join(base_output_dir, "val")

    print("="*80)
    print("Augmented OT-CFM Preprocessing")
    print("="*80)
    print(f"NL embeddings: {nl_path}")
    print(f"Lean embeddings: {lean_path}")
    print(f"Train samples: {args.max_samples}, Val samples: {args.val_samples}")
    print(f"Output: {base_output_dir}")
    print(f"N_max: {args.n_max}")
    print(f"λ_pos: {args.lambda_pos}, α_delete: {args.alpha_delete}")
    print(f"α_create: {args.alpha_create}, β_create: {args.beta_create}")
    print(f"Sinkhorn reg: {args.reg}")
    print("="*80 + "\n")

    ot_config = {
        'n_max': args.n_max,
        'lambda_pos': args.lambda_pos,
        'alpha_delete': args.alpha_delete,
        'alpha_create': args.alpha_create,
        'beta_create': args.beta_create,
        'reg': args.reg,
    }

    # Initialize writers for train and val
    train_writer = AugmentedOTWriter(output_dir=train_dir, shard_size=args.shard_size, config=ot_config)
    val_writer = AugmentedOTWriter(output_dir=val_dir, shard_size=args.shard_size, config=ot_config)

    # Collect pairs first (for proper train/val split)
    print("Collecting paired embeddings...")
    pairs = []
    d = None
    
    for sample_id, nl_emb, lean_emb in stream_paired_embeddings(nl_path, lean_path, max_len, total_samples):
        if d is None:
            d = nl_emb.shape[-1]
            print(f"Embedding dimension: {d}")
        pairs.append((sample_id, nl_emb, lean_emb))
        if len(pairs) >= total_samples:
            break
    
    print(f"Collected {len(pairs)} pairs")
    
    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(pairs))
    train_indices = indices[:args.max_samples]
    val_indices = indices[args.max_samples:args.max_samples + args.val_samples]
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")
    
    # Process train set
    print("\nProcessing TRAIN set...")
    print(f"Using GPU: {DEVICE}")
    for i in tqdm(train_indices, desc="Train OT"):
        sample_id, nl_emb, lean_emb = pairs[i]
        
        src_state, src_pos, src_active = create_augmented_state(nl_emb, args.n_max, d)
        tgt_state, tgt_pos, tgt_active = create_augmented_state(lean_emb, args.n_max, d)
        
        # Use GPU-accelerated OT
        coupling, perm = compute_augmented_ot_coupling_gpu(
            src_state, tgt_state, src_pos, tgt_pos, src_active, tgt_active,
            reg=args.reg, lambda_pos=args.lambda_pos,
            alpha_delete=args.alpha_delete, alpha_create=args.alpha_create,
            beta_create=args.beta_create,
            device=DEVICE,
        )
        
        train_writer.add(sample_id, src_state, tgt_state, src_pos, tgt_pos, src_active, tgt_active, perm)
    
    train_writer.finish()
    
    # Process val set
    print("\nProcessing VAL set...")
    for i in tqdm(val_indices, desc="Val OT"):
        sample_id, nl_emb, lean_emb = pairs[i]
        
        src_state, src_pos, src_active = create_augmented_state(nl_emb, args.n_max, d)
        tgt_state, tgt_pos, tgt_active = create_augmented_state(lean_emb, args.n_max, d)
        
        # Use GPU-accelerated OT
        coupling, perm = compute_augmented_ot_coupling_gpu(
            src_state, tgt_state, src_pos, tgt_pos, src_active, tgt_active,
            reg=args.reg, lambda_pos=args.lambda_pos,
            alpha_delete=args.alpha_delete, alpha_create=args.alpha_create,
            beta_create=args.beta_create,
            device=DEVICE,
        )
        
        val_writer.add(sample_id, src_state, tgt_state, src_pos, tgt_pos, src_active, tgt_active, perm)
    
    val_writer.finish()
    
    print("\n" + "="*80)
    print("Done!")
    print(f"Train: {train_dir}")
    print(f"Val: {val_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
