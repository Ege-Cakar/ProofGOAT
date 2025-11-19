import os
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
            for _, row in df.iterrows():
                rec = {}

                # Case 1: there is an explicit "hidden" column (already [L, d] or [d])
                if "hidden" in df.columns:
                    rec["hidden"] = np.asarray(row["hidden"], dtype=np.float32)

                # Case 2: there is a single "embedding" column
                elif "embedding" in df.columns:
                    rec["hidden"] = np.asarray(row["embedding"], dtype=np.float32)

                else:
                    # Generic case: each column is a token embedding (list[float])
                    # e.g. columns 0..211, each is length d=2048
                    tokens = [np.asarray(v, dtype=np.float32) for v in row.values]
                    rec["hidden"] = np.stack(tokens, axis=0)  # shape [L, d]

                # Optional id column
                rec["id"] = row["id"] if "id" in df.columns else None
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
        h_nl = np.asarray(rec_nl["hidden"], dtype=np.float32).copy()
        h_lean = np.asarray(rec_lean["hidden"], dtype=np.float32).copy()

        L = min(h_nl.shape[0], h_lean.shape[0])
        if self.max_len is not None:
            L = min(L, self.max_len)

        h_nl = h_nl[:L]
        h_lean = h_lean[:L]

        return torch.from_numpy(h_nl), torch.from_numpy(h_lean)


def collate_fn(batch):
    """Collate function stacks pairs into tensors [B, L, d]"""
    h_nl_list, h_lean_list = zip(*batch)
    # Ensure each item is 2D (L, d). If an item is 1D, treat as (1, d).
    proc_nl = []
    proc_lean = []
    for x in h_nl_list:
        if x.dim() == 1:
            proc_nl.append(x.unsqueeze(0))
        else:
            proc_nl.append(x)
    for x in h_lean_list:
        if x.dim() == 1:
            proc_lean.append(x.unsqueeze(0))
        else:
            proc_lean.append(x)

    # compute common L and d
    L = min(x.shape[0] for x in proc_nl)
    L = min(L, min(x.shape[0] for x in proc_lean))
    d = proc_nl[0].shape[1]
    B = len(proc_nl)

    h_nl = torch.stack([x[:L] for x in proc_nl], dim=0)
    h_lean = torch.stack([x[:L] for x in proc_lean], dim=0)
    return h_nl, h_lean
