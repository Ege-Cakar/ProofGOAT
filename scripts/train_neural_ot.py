"""Train a Neural OT velocity field on precomputed embeddings.

Usage:
  python -m scripts.train_neural_ot --config project_config.yaml
"""
import argparse
import os
import yaml
import math

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(ROOT)

from src.flows.dataset import EmbeddingPairDataset, collate_fn
from src.flows.neural_ot import NeuralOTFlow
from src.flows.losses import cycle_consistency_loss
from src.io_utils import ensure_dir


def main(cfg):
    nocuda = not torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    neo_cfg = cfg.get("neural_ot", {})
    nl_path = neo_cfg.get("nl_embeddings", os.path.join(cfg["data"]["out_dir"], "nl_embeddings.parquet"))
    lean_path = neo_cfg.get("lean_embeddings", os.path.join(cfg["data"]["out_dir"], "lean_embeddings.parquet"))
    out_dir = neo_cfg.get("output_dir", os.path.join(cfg["data"]["out_dir"], "neural_ot"))
    ensure_dir(out_dir)

    max_len = neo_cfg.get("max_len", 256)
    batch_size = neo_cfg.get("batch_size", 8)
    num_epochs = neo_cfg.get("num_epochs", 5)
    lr = neo_cfg.get("learning_rate", 1e-4)
    hidden_dim = neo_cfg.get("hidden_dim", 4096)
    time_embed_dim = neo_cfg.get("time_embed_dim", 128)
    num_layers = neo_cfg.get("num_layers", 3)
    mlp_width = neo_cfg.get("mlp_width", 2048)
    num_steps = neo_cfg.get("num_steps", 8)
    lambda_cycle = neo_cfg.get("lambda_cycle", 0.0)

    ds = EmbeddingPairDataset(nl_path, lean_path, max_len=max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # instantiate model
    model = NeuralOTFlow(hidden_dim=hidden_dim, time_embed_dim=time_embed_dim, num_layers=num_layers, mlp_width=mlp_width)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        count = 0
        for i, (h_nl, h_lean) in enumerate(dl):
            h_nl = h_nl.to(device)
            h_lean = h_lean.to(device)

            optimizer.zero_grad()
            l_fm = model.compute_flow_matching_loss(h_nl, h_lean)
            loss = l_fm
            if lambda_cycle > 0.0:
                l_cycle = cycle_consistency_loss(model, h_nl, num_steps=num_steps)
                loss = loss + lambda_cycle * l_cycle

            loss.backward()
            optimizer.step()

            running += float(loss.item())
            count += 1
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} iter {i+1} loss={running/count:.6f}")

        epoch_loss = running / max(1, count)
        print(f"Epoch {epoch+1} finished. avg loss: {epoch_loss:.6f}")

        # save checkpoint
        ckpt_path = os.path.join(out_dir, f"neural_ot_epoch{epoch+1}.pt")
        torch.save({"model_state": model.state_dict(), "cfg": neo_cfg}, ckpt_path)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(out_dir, "neural_ot_best.pt")
            torch.save({"model_state": model.state_dict(), "cfg": neo_cfg}, best_path)
            print(f"Saved best model to {best_path}")

    print("Training complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="project_config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
