from typing import List

import torch

from .neural_ot import NeuralOTFlow
from ..model_utils import load_model_and_tokenizer
from ..alignment import extract_hidden_states


def nl_texts_to_transported_embeddings(texts: List[str], cfg: dict, neural_ot: NeuralOTFlow, device: torch.device = None):
    """Encode NL texts using configured model, extract final-layer hidden states, and transport to Lean space.

    Returns list of numpy arrays (variable-length) transported to CPU.
    """
    if device is None:
        import torch as _torch
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    nl_model, nl_tokenizer = load_model_and_tokenizer(cfg["models"]["nl_model"], fp16=cfg.get("extract", {}).get("fp16", False))
    nl_model.to(device)
    nl_model.eval()

    h_list = extract_hidden_states(
        texts,
        tokenizer=nl_tokenizer,
        model=nl_model,
        layer=cfg.get("extract", {}).get("nl_layer", -1),
        max_length=cfg.get("extract", {}).get("max_length", 2048),
        batch_size=cfg.get("extract", {}).get("batch_size", 2),
    )

    # transport each
    transported = []
    for h in h_list:
        ht = torch.from_numpy(h.astype("float32")).unsqueeze(0).to(device)
        with torch.no_grad():
            h_t = neural_ot.transport_nl_to_lean(ht)
        transported.append(h_t.squeeze(0).cpu().numpy())
    return transported
