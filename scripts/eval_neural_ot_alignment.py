"""Evaluate retrieval alignment before and after Neural OT transport.

Usage:
  python -m scripts.eval_neural_ot_alignment --config project_config.yaml --model outputs/neural_ot/neural_ot_best.pt
"""
import argparse
import os
import yaml

import torch
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(ROOT)

from src.flows.neural_ot import NeuralOTFlow
from src.alignment import retrieval_metrics


def load_embeddings(path: str):
    # accept parquet produced by extract_embeddings
    df = pd.read_parquet(path)
    arrs = [torch.tensor(x, dtype=torch.float32) for x in df["hidden"].tolist()]
    return arrs


def main(cfg, ckpt):
    neo_cfg = cfg.get("neural_ot", {})
    nl_path = neo_cfg.get("nl_embeddings", os.path.join(cfg["data"]["out_dir"], "nl_embeddings.parquet"))
    lean_path = neo_cfg.get("lean_embeddings", os.path.join(cfg["data"]["out_dir"], "lean_embeddings.parquet"))

    nl = load_embeddings(nl_path)
    lean = load_embeddings(lean_path)

    # build matrices by truncating to common length per example (use first N examples)
    N = min(len(nl), len(lean), 256)
    # pick first example length
    L = min(min(x.shape[0] for x in nl[:N]), min(x.shape[0] for x in lean[:N]))
    d = nl[0].shape[1]

    q = torch.stack([nl[i][:L] .mean(dim=0) for i in range(N)], dim=0)
    c = torch.stack([lean[i][:L].mean(dim=0) for i in range(N)], dim=0)

    # baseline
    baseline = retrieval_metrics(q, c)
    print("Baseline:", baseline)

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device)
    cfg_model = state.get("cfg", {})
    model = NeuralOTFlow(hidden_dim=cfg_model.get("hidden_dim", 4096), time_embed_dim=cfg_model.get("time_embed_dim", 128), num_layers=cfg_model.get("num_layers", 3), mlp_width=cfg_model.get("mlp_width", 2048))
    model.load_state_dict(state["model_state"], strict=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        q_t = model.transport_nl_to_lean(q.to(device)).cpu()

    ot_metrics = retrieval_metrics(q_t, c)
    print("After OT:", ot_metrics)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="project_config.yaml")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    if args.model is None:
        raise SystemExit("Please provide --model checkpoint path")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.model)
