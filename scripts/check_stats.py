import os
import torch
import numpy as np
import pandas as pd
import argparse
import yaml

def load_embeddings(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        # Handle potential list/array columns
        if "hidden" in df.columns:
            data = df["hidden"].tolist()
        elif "embedding" in df.columns:
            data = df["embedding"].tolist()
        else:
            data = df.iloc[:, 0].tolist()
        
        # Convert to tensor
        return torch.tensor(np.array(data), dtype=torch.float32)
    elif path.endswith(".npz"):
        data = np.load(path)
        # Assume arrays are stored as 'h0', 'h1', etc or just take the first array found
        arrays = [data[k] for k in data.files if k.startswith("h")]
        if not arrays:
            arrays = [data[k] for k in data.files]
        return torch.tensor(np.concatenate(arrays, axis=0), dtype=torch.float32)
    else:
        raise ValueError(f"Unknown format: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="project_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    neo_cfg = cfg.get("neural_ot", {})
    nl_path = neo_cfg.get("nl_embeddings", os.path.join(cfg["data"]["out_dir"], "nl_embeddings.parquet"))
    lean_path = neo_cfg.get("lean_embeddings", os.path.join(cfg["data"]["out_dir"], "lean_embeddings.parquet"))

    print(f"Checking NL embeddings from: {nl_path}")
    if os.path.exists(nl_path):
        nl_emb = load_embeddings(nl_path)
        print(f"NL Shape: {nl_emb.shape}")
        print(f"NL Mean: {nl_emb.mean().item():.4f}")
        print(f"NL Std: {nl_emb.std().item():.4f}")
        print(f"NL Min: {nl_emb.min().item():.4f}")
        print(f"NL Max: {nl_emb.max().item():.4f}")
        print(f"NL Avg Norm: {torch.norm(nl_emb, dim=-1).mean().item():.4f}")
    else:
        print("NL embeddings file not found.")

    print("-" * 20)

    print(f"Checking Lean embeddings from: {lean_path}")
    if os.path.exists(lean_path):
        lean_emb = load_embeddings(lean_path)
        print(f"Lean Shape: {lean_emb.shape}")
        print(f"Lean Mean: {lean_emb.mean().item():.4f}")
        print(f"Lean Std: {lean_emb.std().item():.4f}")
        print(f"Lean Min: {lean_emb.min().item():.4f}")
        print(f"Lean Max: {lean_emb.max().item():.4f}")
        print(f"Lean Avg Norm: {torch.norm(lean_emb, dim=-1).mean().item():.4f}")
    else:
        print("Lean embeddings file not found.")

if __name__ == "__main__":
    main()
