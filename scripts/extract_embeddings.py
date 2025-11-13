import argparse
import os
import sys
import yaml
import torch

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.io_utils import read_jsonl, ensure_dir, save_npz, save_parquet
from src.model_utils import load_model_and_tokenizer
from src.alignment import extract_hidden_states   # assuming this exists


def main(cfg):
    # ----------------------------
    # Load data
    # ----------------------------
    pairs = read_jsonl(cfg["data"]["input_jsonl"])
    nl_texts = [ex["nl_proof"] for ex in pairs]
    lean_texts = [ex["lean_proof"] for ex in pairs]

    # ----------------------------
    # Load NL model + tokenizer
    # ----------------------------
    print("Loading NL model...")
    nl_model, nl_tokenizer = load_model_and_tokenizer(
        cfg["models"]["nl_model"],
        cfg["extract"]["fp16"]
    )

    # ----------------------------
    # Extract NL embeddings
    # ----------------------------
    print("Extracting NL embeddings...")
    embed_nl = extract_hidden_states(
        nl_texts,
        nl_tokenizer,
        nl_model,
        layer=cfg["extract"]["nl_layer"],
        max_length=cfg["extract"]["max_length"],
        batch_size=cfg["extract"]["batch_size"],
    )

    # ----------------------------
    # Load Lean model + tokenizer
    # ----------------------------
    print("Loading Lean model...")
    lean_model, lean_tokenizer = load_model_and_tokenizer(
        cfg["models"]["lean_model"],
        cfg["extract"]["fp16"]
    )

    # ----------------------------
    # Extract Lean embeddings
    # ----------------------------
    print("Extracting Lean embeddings...")
    embed_lean = extract_hidden_states(
        lean_texts,
        lean_tokenizer,
        lean_model,
        layer=cfg["extract"]["lean_layer"],
        max_length=cfg["extract"]["max_length"],
        batch_size=cfg["extract"]["batch_size"],
    )

    # ----------------------------
    # Save outputs
    # ----------------------------
    out_dir = cfg["data"]["out_dir"]
    ensure_dir(out_dir)

    if cfg["extract"]["save_format"] == "npz":
        save_npz(embed_nl, os.path.join(out_dir, "nl_embeddings.npz"))
        save_npz(embed_lean, os.path.join(out_dir, "lean_embeddings.npz"))
        print("Saved embeddings in NPZ format.")
    else:
        save_parquet(embed_nl, os.path.join(out_dir, "nl_embeddings.parquet"))
        save_parquet(embed_lean, os.path.join(out_dir, "lean_embeddings.parquet"))
        print("Saved embeddings in Parquet format.")

    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="project_config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
