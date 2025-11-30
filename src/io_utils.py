import json, os, pandas as pd, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from typing import List, Dict, Union, Optional
from pathlib import Path
import yaml

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
    # Use PyArrow to read only the specific row without loading entire file
    parquet_file = pq.ParquetFile(path)
    
    # Find which row group contains the target row
    row_group_idx = None
    rows_read = 0
    num_row_groups = parquet_file.metadata.num_row_groups
    for i in range(num_row_groups):
        rg_metadata = parquet_file.metadata.row_group(i)
        num_rows = rg_metadata.num_rows
        if id < rows_read + num_rows:
            row_group_idx = i
            break
        rows_read += num_rows
    
    if row_group_idx is None:
        raise IndexError(f"Row ID {id} is out of bounds")
    
    # Read only the row group containing the target row
    row_group = parquet_file.read_row_group(row_group_idx)
    local_row_id = id - rows_read
    
    # Convert to pandas for easier indexing (only one row group in memory)
    df = row_group.to_pandas()
    embedding_data = df.iloc[local_row_id]['hidden']
    
    # Convert to plain Python list first to handle all PyArrow/pandas types
    if hasattr(embedding_data, 'as_py'):
        # PyArrow type - convert to Python object
        embedding_list = embedding_data.as_py()
    elif hasattr(embedding_data, 'tolist'):
        # Pandas Series or numpy array - convert to list
        embedding_list = embedding_data.tolist()
    elif isinstance(embedding_data, np.ndarray):
        embedding_list = embedding_data.tolist()
    elif isinstance(embedding_data, list):
        embedding_list = embedding_data
    else:
        # Try to iterate and convert
        try:
            embedding_list = list(embedding_data)
        except (TypeError, ValueError):
            embedding_list = [embedding_data]
    
    # Now convert the Python list to numpy array
    # Handle nested lists (list of lists = 2D embedding matrix)
    embedding = None
    if isinstance(embedding_list, list) and len(embedding_list) > 0:
        # Check if it's nested (list of lists)
        if isinstance(embedding_list[0], list):
            # Nested list: [[...], [...], ...] - this is a 2D embedding
            # Each inner list represents one token's embedding
            embedding = np.array(embedding_list, dtype=np.float32)
        else:
            # Flat list: [...] - could be a flattened 2D embedding
            # First convert to numpy array
            embedding = np.array(embedding_list, dtype=np.float32)
            
            # Check if it might be a flattened 2D array by testing common dimensions
            if embedding.ndim == 1:
                total_size = embedding.shape[0]
                # Common hidden dimensions to try (prioritize 2048 as it's mentioned in the error)
                common_dims = [2048, 4096, 1024, 512, 768, 1280, 2560]
                reshaped = False
                for hidden_dim in common_dims:
                    if total_size % hidden_dim == 0:
                        num_tokens = total_size // hidden_dim
                        # Only reshape if num_tokens is reasonable (not too small, not too large)
                        if num_tokens > 1 and num_tokens < 10000:
                            embedding = embedding.reshape(num_tokens, hidden_dim)
                            reshaped = True
                            break
                if not reshaped:
                    # If no common dimension works, treat as 1D and reshape to 1xN
                    embedding = embedding.reshape(1, -1)
    else:
        # Empty or single value
        embedding = np.array(embedding_list, dtype=np.float32)
        if embedding.ndim == 0:
            embedding = embedding.reshape(1, 1)
        elif embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
    
    # Final check: ensure it's 2D (num_tokens, hidden_dim)
    if embedding.ndim == 1:
        # If still 1D, try one more time to reshape (shouldn't happen with above logic, but just in case)
        total_size = embedding.shape[0]
        common_dims = [2048, 4096, 1024, 512, 768, 1280, 2560]
        for hidden_dim in common_dims:
            if total_size % hidden_dim == 0:
                num_tokens = total_size // hidden_dim
                if num_tokens > 1 and num_tokens < 10000:
                    embedding = embedding.reshape(num_tokens, hidden_dim)
                    break
        else:
            # If no dimension works, reshape to 1xN
            embedding = embedding.reshape(1, -1)
    elif embedding.ndim == 0:
        embedding = embedding.reshape(1, 1)
    
    # Verify we have a proper 2D array before returning
    if embedding.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got {embedding.ndim}D with shape {embedding.shape}")

    return embedding.tolist()

def load_examples(examples_path: str, verbose: bool = False) -> Optional[str]:
    """
    Load examples from a YAML file.
    
    Args:
        examples_path: Path to the examples YAML file
        verbose: Whether to print verbose output
    
    Returns:
        Examples string if successfully loaded, None otherwise
    """
    try:
        # Resolve path - try as-is first, then relative to current working directory
        examples_path_obj = Path(examples_path)
        if not examples_path_obj.is_absolute():
            # Try relative to current working directory
            if not examples_path_obj.exists():
                # Try relative to HERMES directory (assuming we're in src/)
                cwd = Path.cwd()
                if cwd.name == "src":
                    examples_path_obj = cwd.parent / examples_path
                else:
                    examples_path_obj = cwd / examples_path
        else:
            examples_path_obj = Path(examples_path)
        
        with open(examples_path_obj, "r") as f:
            examples_data = yaml.safe_load(f)
            examples_in_prompt = examples_data.get("examples", "")
            if verbose:
                print(f"Loaded examples from {examples_path_obj} ({len(examples_in_prompt)} characters)")
            return examples_in_prompt
    except Exception as e:
        if verbose:
            print(f"Warning: Could not load examples from {examples_path}: {e}")
        return None
