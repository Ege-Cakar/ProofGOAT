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
