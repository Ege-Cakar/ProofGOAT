import argparse
import os
import sys
import yaml
import numpy as np
import torch

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
            else:
                arr.append(np.array(item))
        return np.array(arr)

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
    nl_texts = [ex["nl_proof"] for ex in pairs]
    lean_texts = [ex["lean_proof"] for ex in pairs]

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
    # Handle dtype requirements (Parquet)
    # ----------------------------
    save_format = cfg["extract"]["save_format"]

    if save_format != "npz":
        # Parquet does NOT support float16
        if embed_nl.dtype == np.float16:
            embed_nl = embed_nl.astype(np.float32)
        if embed_lean.dtype == np.float16:
            embed_lean = embed_lean.astype(np.float32)

    # ----------------------------
    # Save outputs
    # ----------------------------
    if save_format == "npz":
        save_npz(embed_nl, os.path.join(out_dir, "nl_embeddings.npz"))
        save_npz(embed_lean, os.path.join(out_dir, "lean_embeddings.npz"))
        print("Saved NPZ embeddings.")
    else:
        save_parquet(embed_nl, os.path.join(out_dir, "nl_embeddings.parquet"))
        save_parquet(embed_lean, os.path.join(out_dir, "lean_embeddings.parquet"))
        print("Saved Parquet embeddings.")

    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="project_config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
