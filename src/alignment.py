import torch
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def extract_hidden_states(
    texts,
    tokenizer,
    model,
    layer=-1,
    max_length=2048,
    batch_size=2,
):
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Move tensors to the model's device when applicable.
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        enc = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in enc.items()}

        outputs = model(
            **enc,
            output_hidden_states=True,
            return_dict=True
        )

        hidden = outputs.hidden_states[layer]
        attn = enc.get("attention_mask")

        for j in range(hidden.shape[0]):
            if attn is not None:
                seq_len = int(attn[j].sum().item())
                all_embeddings.append(hidden[j, :seq_len].detach().cpu().numpy())
            else:
                all_embeddings.append(hidden[j].detach().cpu().numpy())

    return all_embeddings


@torch.no_grad()
def cosine_alignment_matrix(h_src: torch.Tensor, h_tgt: torch.Tensor, chunk: int = 4096):
    h_src = torch.nn.functional.normalize(h_src, dim=-1)
    h_tgt = torch.nn.functional.normalize(h_tgt, dim=-1)
    Ns, Nt = h_src.size(0), h_tgt.size(0)
    M = torch.empty(Ns, Nt, dtype=torch.float32)
    for i in range(0, Ns, chunk):
        s = h_src[i:i+chunk]
        M[i:i+chunk] = s @ h_tgt.T
    return M


def greedy_bipartite_alignment(M: torch.Tensor):
    return torch.argmax(M, dim=1).cpu().numpy()
