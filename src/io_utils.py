import json, os, pandas as pd, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from typing import List, Dict, Union

def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_parquet(records: Union[List[Dict], np.ndarray], out_path: str):
    """
    Save either:
      - list of dicts -> Parquet via pandas
      - 2D numpy array (N, H) -> Parquet with a single column 'embedding' as FixedSizeList<float32>[H]
    """
    if isinstance(records, np.ndarray):
        if records.ndim != 2:
            raise ValueError("Only 2D numpy arrays are supported for Parquet saving.")
        mat = records.astype(np.float32, copy=False)
        n, h = mat.shape
        values = pa.array(mat.reshape(n * h), type=pa.float32())
        fsl = pa.FixedSizeListArray.from_arrays(values, list_size=h)
        table = pa.Table.from_arrays([fsl], names=["embedding"])
        pq.write_table(table, out_path)
        return
    table = pa.Table.from_pandas(pd.DataFrame.from_records(records))
    pq.write_table(table, out_path)

def save_npz(obj: Dict, out_path: str):
    np.savez_compressed(out_path, **obj)
