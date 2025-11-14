import json, os, pandas as pd, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from typing import List, Dict

def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_parquet(records: List[Dict], out_path: str):
    table = pa.Table.from_pandas(pd.DataFrame.from_records(records))
    pq.write_table(table, out_path)

def save_npz(obj: Dict, out_path: str):
    np.savez_compressed(out_path, **obj)

def verbose_print(*args):
    boundary = "=" * 100
    to_print = boundary + "\n" + " ".join(str(arg) for arg in args) + "\n" + boundary
    print(to_print, flush=True)

def load_embedding(path: str, id: int):
    """
    Load embedding from parquet file.
    
    Args:
        path: Path to the parquet file
        id: row ID of the embedding to load

    Note:
        The embedding is a numpy array with shape [num_tokens, hidden_dim]
        where num_tokens is the sequence length and hidden_dim is the model's hidden dimension
    Returns:
        Embedding as a list of floats
    """
    df = pd.read_parquet(path)
    embedding = np.vstack(df.iloc[id].values)
    return embedding.tolist()