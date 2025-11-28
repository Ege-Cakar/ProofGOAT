import os
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class StreamingEmbeddingPairDataset(Dataset):
    """Memory-efficient dataset that streams embeddings from parquet files.
    
    Uses PyArrow to read data on-demand without loading entire files into memory.
    Builds an index of (row_group, row_within_group) for each sample.
    
    Args:
        nl_path: Path to NL embeddings parquet file
        lean_path: Path to Lean embeddings parquet file  
        max_len: Maximum sequence length (truncates longer sequences)
        cache_size: Number of row groups to cache in memory (default: 2)
    """
    
    def __init__(
        self,
        nl_path: str,
        lean_path: str,
        max_len: Optional[int] = None,
        cache_size: int = 2,
    ) -> None:
        self.nl_path = nl_path
        self.lean_path = lean_path
        self.max_len = max_len
        self.cache_size = cache_size
        
        # Open parquet files (doesn't load data)
        self.nl_pf = pq.ParquetFile(nl_path)
        self.lean_pf = pq.ParquetFile(lean_path)
        
        # Build index mapping global idx -> (row_group, local_idx)
        self.nl_index = self._build_index(self.nl_pf)
        self.lean_index = self._build_index(self.lean_pf)
        
        # Build ID mappings for alignment
        self.nl_id_to_idx = self._build_id_map(self.nl_pf)
        self.lean_id_to_idx = self._build_id_map(self.lean_pf)
        
        # Create aligned pairs (by ID if possible, otherwise by index)
        self.pairs = self._align_pairs()
        
        # LRU cache for row groups
        self._nl_cache: Dict[int, pd.DataFrame] = {}
        self._lean_cache: Dict[int, pd.DataFrame] = {}
        self._nl_cache_order: List[int] = []
        self._lean_cache_order: List[int] = []
        
        print(f"StreamingDataset: {len(self.pairs)} pairs, "
              f"NL: {self.nl_pf.metadata.num_rows} rows in {self.nl_pf.metadata.num_row_groups} groups, "
              f"Lean: {self.lean_pf.metadata.num_rows} rows in {self.lean_pf.metadata.num_row_groups} groups")
    
    def _build_index(self, pf: pq.ParquetFile) -> List[Tuple[int, int]]:
        """Build mapping from global index to (row_group, local_index)."""
        index = []
        for rg_idx in range(pf.metadata.num_row_groups):
            num_rows = pf.metadata.row_group(rg_idx).num_rows
            for local_idx in range(num_rows):
                index.append((rg_idx, local_idx))
        return index
    
    def _build_id_map(self, pf: pq.ParquetFile) -> Dict[str, int]:
        """Build mapping from ID to global index by reading only the 'id' column."""
        id_map = {}
        global_idx = 0
        for rg_idx in range(pf.metadata.num_row_groups):
            # Read only the 'id' column from this row group
            table = pf.read_row_group(rg_idx, columns=['id'])
            ids = table.column('id').to_pylist()
            for local_idx, id_val in enumerate(ids):
                if id_val is not None:
                    id_map[id_val] = global_idx
                global_idx += 1
        return id_map
    
    def _align_pairs(self) -> List[Tuple[int, int]]:
        """Create list of (nl_idx, lean_idx) pairs aligned by ID."""
        # Find common IDs
        common_ids = set(self.nl_id_to_idx.keys()) & set(self.lean_id_to_idx.keys())
        
        if common_ids:
            pairs = [(self.nl_id_to_idx[id_], self.lean_id_to_idx[id_]) 
                     for id_ in sorted(common_ids)]
            print(f"Aligned {len(pairs)} pairs by ID")
        else:
            # Fallback to index alignment
            n = min(len(self.nl_index), len(self.lean_index))
            pairs = [(i, i) for i in range(n)]
            print(f"No common IDs found, aligned {n} pairs by index")
        
        return pairs
    
    def _get_row_group(self, pf: pq.ParquetFile, rg_idx: int, 
                       cache: Dict, cache_order: List) -> pd.DataFrame:
        """Get a row group, using cache if available."""
        if rg_idx in cache:
            # Move to end of cache order (most recently used)
            cache_order.remove(rg_idx)
            cache_order.append(rg_idx)
            return cache[rg_idx]
        
        # Load row group
        df = pf.read_row_group(rg_idx).to_pandas()
        
        # Add to cache
        cache[rg_idx] = df
        cache_order.append(rg_idx)
        
        # Evict oldest if cache is full
        while len(cache_order) > self.cache_size:
            oldest = cache_order.pop(0)
            del cache[oldest]
        
        return df
    
    def _get_embedding(self, pf: pq.ParquetFile, global_idx: int,
                       index: List, cache: Dict, cache_order: List) -> np.ndarray:
        """Get embedding for a global index."""
        rg_idx, local_idx = index[global_idx]
        df = self._get_row_group(pf, rg_idx, cache, cache_order)
        row = df.iloc[local_idx]
        
        # Extract hidden data
        if "hidden" in df.columns:
            hidden_data = row["hidden"]
        elif "embedding" in df.columns:
            hidden_data = row["embedding"]
        else:
            hidden_data = row.iloc[0]
        
        # Convert to numpy array
        if isinstance(hidden_data, np.ndarray):
            if hidden_data.dtype == object:
                return np.stack([np.asarray(x, dtype=np.float32) for x in hidden_data])
            elif hidden_data.ndim >= 1:
                return hidden_data.astype(np.float32)
        
        return np.asarray(hidden_data, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        nl_idx, lean_idx = self.pairs[idx]
        
        h_nl = self._get_embedding(self.nl_pf, nl_idx, self.nl_index,
                                   self._nl_cache, self._nl_cache_order)
        h_lean = self._get_embedding(self.lean_pf, lean_idx, self.lean_index,
                                     self._lean_cache, self._lean_cache_order)
        
        # Ensure 2D shape [L, d]
        if h_nl.ndim == 1:
            h_nl = h_nl.reshape(1, -1)
        if h_lean.ndim == 1:
            h_lean = h_lean.reshape(1, -1)
        
        # Truncate to same length
        L = min(h_nl.shape[0], h_lean.shape[0])
        if self.max_len is not None:
            L = min(L, self.max_len)
        
        h_nl = h_nl[:L].copy()  # copy to make writable
        h_lean = h_lean[:L].copy()
        
        return torch.from_numpy(h_nl), torch.from_numpy(h_lean)


class EmbeddingPairDataset(Dataset):
    """Dataset yielding paired NL / Lean embeddings.

    Expects either:
      - two parquet files with records {'id', 'hidden': list[float32]}
      - two npz files with arrays named h0, h1... containing variable-length arrays

    Returns tuples: (h_nl, h_lean) where each is a torch.FloatTensor [L, d].
    Sequences are truncated to the same length `max_len` or to the min length of the pair.
    """

    def __init__(
        self,
        nl_path: str,
        lean_path: str,
        max_len: Optional[int] = None,
    ) -> None:
        self.nl_path = nl_path
        self.lean_path = lean_path
        self.max_len = max_len

        # load indices
        self.nl_data = self._load_index(nl_path)
        self.lean_data = self._load_index(lean_path)

        # Align by ids where possible. If both have ids, join on id, otherwise pair by index.
        nl_ids = [rec.get("id") for rec in self.nl_data]
        lean_ids = [rec.get("id") for rec in self.lean_data]

        if None not in nl_ids and None not in lean_ids:
            # build mapping for lean
            lean_map = {rid: rec for rec, rid in zip(self.lean_data, lean_ids)}
            paired = []
            for rec, rid in zip(self.nl_data, nl_ids):
                if rid in lean_map:
                    paired.append((rec, lean_map[rid]))
            self.pairs = paired
        else:
            # fallback: pair by index up to min length
            n = min(len(self.nl_data), len(self.lean_data))
            self.pairs = [(self.nl_data[i], self.lean_data[i]) for i in range(n)]

    def _load_index(self, path: str) -> List[Dict]:
        if path.endswith(".parquet") or path.endswith(".parq"):
            df = pd.read_parquet(path)
            records = []
            for i, row in df.iterrows():
                rec = {}
                # allow either 'hidden' or 'embedding' column
                if "hidden" in row:
                    hidden_data = row["hidden"]
                    # Handle case where hidden is a numpy array of objects (nested lists)
                    if isinstance(hidden_data, np.ndarray):
                        if hidden_data.dtype == object:
                            # Array of lists - stack them into a 2D array
                            rec["hidden"] = np.stack([np.asarray(x, dtype=np.float32) for x in hidden_data])
                        elif hidden_data.ndim == 2:
                            rec["hidden"] = hidden_data.astype(np.float32)
                        elif hidden_data.ndim == 1:
                            # Single vector
                            rec["hidden"] = hidden_data.astype(np.float32)
                        else:
                            rec["hidden"] = np.asarray(hidden_data, dtype=np.float32)
                    else:
                        rec["hidden"] = np.asarray(hidden_data, dtype=np.float32)
                elif "embedding" in row:
                    # fixed-size list column
                    emb_data = row["embedding"]
                    if isinstance(emb_data, np.ndarray):
                        if emb_data.dtype == object:
                            rec["hidden"] = np.stack([np.asarray(x, dtype=np.float32) for x in emb_data])
                        else:
                            rec["hidden"] = emb_data.astype(np.float32)
                    else:
                        rec["hidden"] = np.asarray(emb_data, dtype=np.float32)
                else:
                    # try to interpret first column
                    rec["hidden"] = np.asarray(row.iloc[0], dtype=np.float32)
                rec["id"] = row.get("id", None)
                records.append(rec)
            return records

        elif path.endswith(".npz"):
            arr = np.load(path, allow_pickle=True)
            records = []
            # arrays named h0, h1,... or keys list
            keys = sorted([k for k in arr.files if k.startswith("h")])
            if not keys:
                # fallback: take all
                keys = list(arr.files)
            for k in keys:
                records.append({"hidden": np.asarray(arr[k], dtype=np.float32), "id": None})
            return records

        else:
            raise ValueError(f"Unsupported embedding file: {path}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        rec_nl, rec_lean = self.pairs[idx]
        h_nl = np.asarray(rec_nl["hidden"], dtype=np.float32)
        h_lean = np.asarray(rec_lean["hidden"], dtype=np.float32)

        L = min(h_nl.shape[0], h_lean.shape[0])
        if self.max_len is not None:
            L = min(L, self.max_len)

        h_nl = h_nl[:L]
        h_lean = h_lean[:L]

        return torch.from_numpy(h_nl), torch.from_numpy(h_lean)


def collate_fn(batch):
    """Collate function stacks pairs into tensors [B, L, d]"""
    h_nl_list, h_lean_list = zip(*batch)
    # assume same L across batch (dataset truncates)
    L = min(x.shape[0] for x in h_nl_list)
    d = h_nl_list[0].shape[1]
    B = len(h_nl_list)

    h_nl = torch.stack([x[:L] for x in h_nl_list], dim=0)
    h_lean = torch.stack([x[:L] for x in h_lean_list], dim=0)
    return h_nl, h_lean


class OTAlignedDataset(Dataset):
    """Dataset that wraps embeddings with precomputed OT couplings.
    
    This dataset stores:
    - NL embeddings [L_i, d] for each sample i
    - Lean embeddings [L_i, d] for each sample i  
    - OT coupling [L_nl_i, L_lean_i] for each sample i
    
    The OT couplings define how NL tokens should map to Lean tokens.
    """
    
    def __init__(
        self,
        nl_embeddings: List[np.ndarray],
        lean_embeddings: List[np.ndarray],
        ot_couplings: List[np.ndarray],
        max_len: Optional[int] = None,
    ) -> None:
        """
        Args:
            nl_embeddings: List of NL embeddings, each [L_i, d]
            lean_embeddings: List of Lean embeddings, each [L_i, d]
            ot_couplings: List of OT couplings, each [L_nl_i, L_lean_i]
            max_len: Maximum sequence length (truncates longer sequences)
        """
        assert len(nl_embeddings) == len(lean_embeddings) == len(ot_couplings)
        
        self.nl_embeddings = nl_embeddings
        self.lean_embeddings = lean_embeddings
        self.ot_couplings = ot_couplings
        self.max_len = max_len
        
    def __len__(self) -> int:
        return len(self.nl_embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        h_nl = self.nl_embeddings[idx].copy()
        h_lean = self.lean_embeddings[idx].copy()
        coupling = self.ot_couplings[idx].copy()
        
        # Ensure 2D
        if h_nl.ndim == 1:
            h_nl = h_nl.reshape(1, -1)
        if h_lean.ndim == 1:
            h_lean = h_lean.reshape(1, -1)
        
        # Truncate if needed
        L_nl, L_lean = h_nl.shape[0], h_lean.shape[0]
        if self.max_len is not None:
            if L_nl > self.max_len:
                h_nl = h_nl[:self.max_len]
                coupling = coupling[:self.max_len, :]
                L_nl = self.max_len
            if L_lean > self.max_len:
                h_lean = h_lean[:self.max_len]
                coupling = coupling[:, :self.max_len]
                L_lean = self.max_len
        
        # Re-normalize coupling after truncation
        coupling = coupling / (coupling.sum() + 1e-8)
        
        return (
            torch.from_numpy(h_nl.astype(np.float32)),
            torch.from_numpy(h_lean.astype(np.float32)),
            torch.from_numpy(coupling.astype(np.float32))
        )


def collate_fn_with_coupling(batch):
    """Collate function for OTAlignedDataset that handles variable-length sequences.
    
    Returns:
        h_nl: [B, L_nl, d] - NL embeddings (padded to max length in batch)
        h_lean: [B, L_lean, d] - Lean embeddings (padded to max length in batch)
        coupling: [B, L_nl, L_lean] - OT couplings (padded)
    """
    h_nl_list, h_lean_list, coupling_list = zip(*batch)
    
    B = len(h_nl_list)
    d = h_nl_list[0].shape[1]
    
    # Find max lengths in this batch
    L_nl_max = max(x.shape[0] for x in h_nl_list)
    L_lean_max = max(x.shape[0] for x in h_lean_list)
    
    # Pad sequences
    h_nl_padded = torch.zeros(B, L_nl_max, d)
    h_lean_padded = torch.zeros(B, L_lean_max, d)
    coupling_padded = torch.zeros(B, L_nl_max, L_lean_max)
    
    for i in range(B):
        L_nl_i = h_nl_list[i].shape[0]
        L_lean_i = h_lean_list[i].shape[0]
        
        h_nl_padded[i, :L_nl_i, :] = h_nl_list[i]
        h_lean_padded[i, :L_lean_i, :] = h_lean_list[i]
        coupling_padded[i, :L_nl_i, :L_lean_i] = coupling_list[i]
    
    return h_nl_padded, h_lean_padded, coupling_padded

