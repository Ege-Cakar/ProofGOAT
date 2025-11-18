import argparse
import os
import sys
import yaml
import numpy as np
import torch
import random
import pyarrow as pa
import pyarrow.parquet as pq

# Resolve project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.io_utils import read_jsonl, ensure_dir, save_npz, save_parquet
from src.model_utils import load_model_and_tokenizer
from src.alignment import extract_hidden_states


def to_numpy(x):
    """
    Convert tensors or lists of tensors into a 2D numpy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    if isinstance(x, list):
        # list of tensors or lists
        arr = []
        for item in x:
            if isinstance(item, torch.Tensor):
                arr.append(item.detach().cpu().numpy())
            elif isinstance(item, np.ndarray):
                # Avoid copying existing numpy arrays; keep references
                arr.append(item)
            else:
                arr.append(np.array(item))
        # Return an object-dtype array to avoid a large contiguous allocation
        return np.array(arr, dtype=object)

    if isinstance(x, np.ndarray):
        return x

    # fallback
    return np.array(x)


def main(cfg):
    # ----------------------------
    # Load dataset
    # ----------------------------
    data_path = cfg["data"]["input_jsonl"]
    out_dir = cfg["data"]["out_dir"]
    ensure_dir(out_dir)

    pairs = read_jsonl(data_path)
    # Randomly select a subset of 5000 examples (or fewer if dataset smaller)
    rng = random.Random(42)
    k = min(1000, len(pairs))
    idxs = rng.sample(range(len(pairs)), k)
    subset = [pairs[i] for i in idxs]
    nl_texts = [ex["nl_proof"] for ex in subset]
    lean_texts = [ex["lean_proof"] for ex in subset]

    # ----------------------------
    # Load NL model
    # ----------------------------
    print("Loading NL model...")
    nl_model, nl_tokenizer = load_model_and_tokenizer(
        cfg["models"]["nl_model"],
        fp16=cfg["extract"]["fp16"],
    )

    # ----------------------------
    # Extract NL embeddings
    # ----------------------------
    print("Extracting NL embeddings...")
    embed_nl = extract_hidden_states(
        texts=nl_texts,
        tokenizer=nl_tokenizer,
        model=nl_model,
        layer=cfg["extract"]["nl_layer"],
        max_length=cfg["extract"]["max_length"],
        batch_size=cfg["extract"]["batch_size"],
    )
    embed_nl = to_numpy(embed_nl)

    # ----------------------------
    # Load Lean model
    # ----------------------------
    print("Loading Lean model...")
    lean_model, lean_tokenizer = load_model_and_tokenizer(
        cfg["models"]["lean_model"],
        fp16=cfg["extract"]["fp16"],
    )

    # ----------------------------
    # Extract Lean embeddings
    # ----------------------------
    print("Extracting Lean embeddings...")
    embed_lean = extract_hidden_states(
        texts=lean_texts,
        tokenizer=lean_tokenizer,
        model=lean_model,
        layer=cfg["extract"]["lean_layer"],
        max_length=cfg["extract"]["max_length"],
        batch_size=cfg["extract"]["batch_size"],
    )
    embed_lean = to_numpy(embed_lean)

    # ----------------------------
    # Save outputs
    # ----------------------------
    save_format = cfg["extract"]["save_format"]
    if save_format == "npz":
        # Save as named arrays to avoid large contiguous allocations
        nl_dict = {f"h{i}": arr for i, arr in enumerate(embed_nl)}
        lean_dict = {f"h{i}": arr for i, arr in enumerate(embed_lean)}
        save_npz(nl_dict, os.path.join(out_dir, "nl_embeddings.npz"))
        save_npz(lean_dict, os.path.join(out_dir, "lean_embeddings.npz"))
        print("Saved NPZ embeddings.")
    else:
        # Batched Parquet writes to limit RAM. Each row: id, hidden (list<list<float32>>)
        nl_path = os.path.join(out_dir, "nl_embeddings.parquet")
        lean_path = os.path.join(out_dir, "lean_embeddings.parquet")
        save_batch_size = int(cfg.get("extract", {}).get("save_batch_size", 1024))

        def write_parquet_batched(arr_list, id_list, out_path):
            writer = None
            n = len(arr_list)
            for start in range(0, n, save_batch_size):
                end = min(start + save_batch_size, n)
                ids = [str(id_list[i]) for i in range(start, end)]
                hidden = [arr.astype(np.float32).tolist() for arr in arr_list[start:end]]
                col_id = pa.array(ids, type=pa.string())
                col_hidden = pa.array(hidden, type=pa.list_(pa.list_(pa.float32())))
                tbl = pa.Table.from_arrays([col_id, col_hidden], names=["id", "hidden"])
                if writer is None:
                    writer = pq.ParquetWriter(out_path, schema=tbl.schema)
                writer.write_table(tbl)
            if writer is not None:
                writer.close()

        ids = [ex.get("id") for ex in subset]
        write_parquet_batched(embed_nl, ids, nl_path)
        write_parquet_batched(embed_lean, ids, lean_path)
        print("Saved Parquet embeddings (batched writes).")

    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="project_config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
