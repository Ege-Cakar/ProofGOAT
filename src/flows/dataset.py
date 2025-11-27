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

        # Ensure 2D [L, d]
        if h_nl.ndim == 1:
            h_nl = h_nl[None, :]
        if h_lean.ndim == 1:
            h_lean = h_lean[None, :]

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
