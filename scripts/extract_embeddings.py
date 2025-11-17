import argparse
import gc
import os
import sys
import yaml
import numpy as np
import torch

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
    nl_shard_idx = 0
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
            save_npz(obj, os.path.join(out_dir, f"nl_embeddings-{nl_shard_idx:05d}.npz"))
        else:
            # Upcast to float32 for Parquet compatibility
            recs = [{"hidden": x.astype(np.float32).tolist()} for x in nl_embeds]
            save_parquet(recs, os.path.join(out_dir, f"nl_embeddings-{nl_shard_idx:05d}.parquet"))

        nl_shard_idx += 1
        del nl_texts, nl_embeds, shard
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    lean_shard_idx = 0
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
            save_npz(obj, os.path.join(out_dir, f"lean_embeddings-{lean_shard_idx:05d}.npz"))
        else:
            recs = [{"hidden": x.astype(np.float32).tolist()} for x in lean_embeds]
            save_parquet(recs, os.path.join(out_dir, f"lean_embeddings-{lean_shard_idx:05d}.parquet"))

        lean_shard_idx += 1
        del lean_texts, lean_embeds, shard
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Done. Saved shard files to:", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="project_config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
