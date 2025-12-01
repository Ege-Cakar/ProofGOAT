"""Sanity check for Neural OT components with random tensors."""
import torch
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(ROOT)

from src.flows.neural_ot import NeuralOTFlow
from src.flows.losses import flow_matching_loss


def main():
    B, L, d = 2, 16, 128
    h0 = torch.randn(B, L, d)
    h1 = torch.randn(B, L, d)

    model = NeuralOTFlow(hidden_dim=d, time_embed_dim=64, num_layers=2, mlp_width=256)

    loss = model.compute_flow_matching_loss(h0, h1)
    print("Flow matching loss:", loss.item())

    h_fwd = model.transport_nl_to_lean(h0, num_steps=4)
    print("Transported shape:", h_fwd.shape)


if __name__ == '__main__':
    main()
