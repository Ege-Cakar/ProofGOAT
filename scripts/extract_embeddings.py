import argparse
import gc
import os
import sys
import yaml
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq

# Resolve project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.io_utils import ensure_dir, save_npz, save_parquet
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
            else:
                arr.append(np.array(item))
        return np.array(arr)

    if isinstance(x, np.ndarray):
        return x

    # fallback
    return np.array(x)

def iter_jsonl(path):
    import json as _json
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield _json.loads(s)


def chunked(iterable, size):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def main(cfg):
    # ----------------------------
    # Load dataset (streaming)
    # ----------------------------
    data_path = cfg["data"]["input_jsonl"]
    out_dir = cfg["data"]["out_dir"]
    ensure_dir(out_dir)

    # ----------------------------
    # Load NL model
    # ----------------------------
    print("Loading NL model...")
    nl_model, nl_tokenizer = load_model_and_tokenizer(
        cfg["models"]["nl_model"],
        fp16=cfg["extract"]["fp16"],
    )

    # ----------------------------
    # Extract NL embeddings (streaming shards)
    # ----------------------------
    print("Extracting NL embeddings...")
    nl_writer = None
    nl_out_path = os.path.join(out_dir, "nl_embeddings.parquet")
    for shard in chunked(iter_jsonl(data_path), size=256):
        nl_texts = [ex["nl_proof"] for ex in shard]
        nl_embeds = extract_hidden_states(
            texts=nl_texts,
            tokenizer=nl_tokenizer,
            model=nl_model,
            layer=cfg["extract"]["nl_layer"],
            max_length=cfg["extract"]["max_length"],
            batch_size=cfg["extract"]["batch_size"],
        )

        save_format = cfg["extract"]["save_format"]
        if save_format == "npz":
            obj = {f"h{i}": x for i, x in enumerate(nl_embeds)}
            save_npz(obj, os.path.join(out_dir, "nl_embeddings.npz"))
            # Overwrite single NPZ; if you want append semantics, prefer shards
        else:
            # Build Arrow array: list<list<float32>> to avoid pandas overhead
            hidden_list = [x.astype(np.float32).tolist() for x in nl_embeds]
            col = pa.array(hidden_list, type=pa.list_(pa.list_(pa.float32())))
            tbl = pa.Table.from_arrays([col], names=["hidden"])
            if nl_writer is None:
                nl_writer = pq.ParquetWriter(nl_out_path, schema=tbl.schema)
            nl_writer.write_table(tbl)

        del nl_texts, nl_embeds, shard
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if 'nl_writer' in locals() and nl_writer is not None:
        nl_writer.close()

    # ----------------------------
    # Load Lean model
    # ----------------------------
    print("Loading Lean model...")
    lean_model, lean_tokenizer = load_model_and_tokenizer(
        cfg["models"]["lean_model"],
        fp16=cfg["extract"]["fp16"],
    )

    # ----------------------------
    # Extract Lean embeddings (streaming shards)
    # ----------------------------
    print("Extracting Lean embeddings...")
    lean_writer = None
    lean_out_path = os.path.join(out_dir, "lean_embeddings.parquet")
    for shard in chunked(iter_jsonl(data_path), size=256):
        lean_texts = [ex["lean_proof"] for ex in shard]
        lean_embeds = extract_hidden_states(
            texts=lean_texts,
            tokenizer=lean_tokenizer,
            model=lean_model,
            layer=cfg["extract"]["lean_layer"],
            max_length=cfg["extract"]["max_length"],
            batch_size=cfg["extract"]["batch_size"],
        )

        save_format = cfg["extract"]["save_format"]
        if save_format == "npz":
            obj = {f"h{i}": x for i, x in enumerate(lean_embeds)}
            save_npz(obj, os.path.join(out_dir, "lean_embeddings.npz"))
        else:
            hidden_list = [x.astype(np.float32).tolist() for x in lean_embeds]
            col = pa.array(hidden_list, type=pa.list_(pa.list_(pa.float32())))
            tbl = pa.Table.from_arrays([col], names=["hidden"])
            if lean_writer is None:
                lean_writer = pq.ParquetWriter(lean_out_path, schema=tbl.schema)
            lean_writer.write_table(tbl)

        del lean_texts, lean_embeds, shard
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if 'lean_writer' in locals() and lean_writer is not None:
        lean_writer.close()

    print("Done. Saved outputs to:", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="project_config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
