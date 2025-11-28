import os
from typing import Optional, List, Tuple, Dict, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
import pyarrow.parquet as pq

class EmbeddingPairDataset(IterableDataset):
    """Dataset yielding paired NL / Lean embeddings.
    Supports both in-memory loading (for small datasets) and streaming (for large datasets).
    """

    def __init__(
        self,
        nl_path: str,
        lean_path: str,
        max_len: Optional[int] = None,
        streaming: bool = False,
    ) -> None:
        self.nl_path = nl_path
        self.lean_path = lean_path
        self.max_len = max_len
        self.streaming = streaming

        if streaming:
            print(f"Initializing streaming dataset from {nl_path} and {lean_path}")
            # Verify files exist
            if not os.path.exists(nl_path):
                raise FileNotFoundError(f"NL file not found: {nl_path}")
            if not os.path.exists(lean_path):
                raise FileNotFoundError(f"Lean file not found: {lean_path}")
        else:
            # Load into memory (Pandas)
            self.df_nl = self._load_as_dataframe(nl_path)
            self.df_lean = self._load_as_dataframe(lean_path)
            
            print(f"Loaded NL data: {len(self.df_nl)} rows")
            print(f"Loaded Lean data: {len(self.df_lean)} rows")
            
            self.merged = self._align_data(self.df_nl, self.df_lean)
            print(f"Final paired dataset size: {len(self.merged)}")

    def _get_data_files(self, path: str) -> Union[str, List[str]]:
        if os.path.isdir(path):
            return sorted([
                os.path.join(path, f) for f in os.listdir(path) 
                if f.endswith(".parquet") or f.endswith(".parq")
            ])
        return path

    def _align_data(self, df_nl, df_lean):
        # Align by ids where possible
        if "id" in df_nl.columns and "id" in df_lean.columns:
            if df_nl["id"].is_unique and df_lean["id"].is_unique:
                print("Aligning by ID...")
                return pd.merge(df_nl, df_lean, on="id", how="inner", suffixes=("_nl", "_lean"))
            else:
                print("IDs not unique, falling back to index alignment.")
                return self._align_by_index(df_nl, df_lean)
        else:
            print("IDs not found, aligning by index...")
            return self._align_by_index(df_nl, df_lean)

    def _align_by_index(self, df_nl, df_lean):
        n = min(len(df_nl), len(df_lean))
        df_nl_sub = df_nl.iloc[:n].reset_index(drop=True)
        df_lean_sub = df_lean.iloc[:n].reset_index(drop=True)
        df_nl_sub = df_nl_sub.rename(columns=lambda x: x + "_nl" if x != "id" else "id_nl")
        df_lean_sub = df_lean_sub.rename(columns=lambda x: x + "_lean" if x != "id" else "id_lean")
        return pd.concat([df_nl_sub, df_lean_sub], axis=1)

    def _load_as_dataframe(self, path: str) -> pd.DataFrame:
        files = self._get_data_files(path)
        if isinstance(files, list):
            dfs = [self._load_single_file_as_df(f) for f in files]
            return pd.concat(dfs, ignore_index=True)
        return self._load_single_file_as_df(files)

    def _load_single_file_as_df(self, path: str) -> pd.DataFrame:
        if path.endswith(".parquet") or path.endswith(".parq"):
            df = pd.read_parquet(path)
            if "embedding" in df.columns and "hidden" not in df.columns:
                df["hidden"] = df["embedding"]
            elif "hidden" not in df.columns:
                cols = df.columns.tolist()
                if cols: df["hidden"] = df[cols[0]]
            if "id" not in df.columns:
                df["id"] = None
            return df[["id", "hidden"]]
        elif path.endswith(".npz"):
            arr = np.load(path, allow_pickle=True)
            keys = sorted([k for k in arr.files if k.startswith("h")]) or list(arr.files)
            return pd.DataFrame([{"hidden": arr[k], "id": None} for k in keys])
        else:
            raise ValueError(f"Unsupported embedding file: {path}")

    def __len__(self) -> int:
        if self.streaming:
            raise TypeError("Streaming dataset does not have a length")
        return len(self.merged)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if self.streaming:
            raise NotImplementedError("Use iteration for streaming dataset")
        
        row = self.merged.iloc[idx]
        return self._process_pair(row["hidden_nl"], row["hidden_lean"])

    def __iter__(self):
        if not self.streaming:
            # If not streaming, yield from merged dataframe
            for idx in range(len(self)):
                yield self[idx]
        else:
            # Streaming mode: use pyarrow to stream batches
            # Note: This assumes 1-to-1 correspondence between files if they are split,
            # or single files. For simplicity, we assume single files for now as per config.
            
            # If paths are directories, we'd need more complex logic. 
            # Assuming they are single parquet files for now (dsv2_...parquet)
            
            nl_pq = pq.ParquetFile(self.nl_path)
            lean_pq = pq.ParquetFile(self.lean_path)
            
            # Iterate row groups or batches
            # We use a small batch size for reading to keep memory low
            batch_size = 100 
            
            nl_iter = nl_pq.iter_batches(batch_size=batch_size)
            lean_iter = lean_pq.iter_batches(batch_size=batch_size)
            
            for nl_batch, lean_batch in zip(nl_iter, lean_iter):
                # Convert to pandas
                df_nl = nl_batch.to_pandas()
                df_lean = lean_batch.to_pandas()
                
                # Iterate rows in the batch
                for i in range(len(df_nl)):
                    if i >= len(df_lean): break
                    
                    row_nl = df_nl.iloc[i]
                    row_lean = df_lean.iloc[i]
                    
                    # Extract hidden
                    h_nl_raw = row_nl.get("hidden", row_nl.get("embedding"))
                    h_lean_raw = row_lean.get("hidden", row_lean.get("embedding"))
                    
                    # Fallback if column name is different (e.g. 0)
                    if h_nl_raw is None: h_nl_raw = row_nl.iloc[0]
                    if h_lean_raw is None: h_lean_raw = row_lean.iloc[0]
                    
                    yield self._process_pair(h_nl_raw, h_lean_raw)

    def _process_pair(self, h_nl_raw, h_lean_raw):
        h_nl = self._to_numpy(h_nl_raw)
        h_lean = self._to_numpy(h_lean_raw)

        # Ensure 2D [L, d]
        if h_nl.ndim == 1: h_nl = h_nl[None, :]
        if h_lean.ndim == 1: h_lean = h_lean[None, :]

        # Pad to max_len instead of truncating
        # If max_len is None, we can't really do this strategy effectively for batches 
        # unless we pick a dynamic max per batch, but here we assume global max_len is set.
        target_len = self.max_len if self.max_len is not None else 256
        
        def pad_and_augment(h, length):
            curr_len = h.shape[0]
            dim = h.shape[1]
            
            if curr_len > length:
                # Truncate if too long
                h = h[:length]
                curr_len = length
            
            # Create output tensor: [length, dim + 2]
            # dim + 2 because: +1 for existence, +1 for position
            out = np.zeros((length, dim + 2), dtype=np.float32)
            
            # Fill real data
            out[:curr_len, :dim] = h
            out[:curr_len, dim] = 1.0 # Existence = 1 for real
            
            # Fill void data (already 0s for embedding and existence)
            # Existence = 0 for void (already set by np.zeros)
            
            # Fill position: 0 to 1
            if length > 1:
                pos = np.linspace(0, 1, length, dtype=np.float32)
            else:
                pos = np.zeros((length,), dtype=np.float32)
                
            out[:, dim + 1] = pos
            
            return out

        h_nl_aug = pad_and_augment(h_nl, target_len)
        h_lean_aug = pad_and_augment(h_lean, target_len)

        return torch.from_numpy(h_nl_aug), torch.from_numpy(h_lean_aug)

    def _to_numpy(self, data) -> np.ndarray:
        if isinstance(data, np.ndarray):
            if data.dtype == object:
                 return np.stack([np.asarray(x, dtype=np.float32) for x in data])
            return data.astype(np.float32)
        elif isinstance(data, list):
             return np.array(data, dtype=np.float32)
        else:
             return np.asarray(data, dtype=np.float32)


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
