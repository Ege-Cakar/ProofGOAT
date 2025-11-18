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


def ot_transport_embeddings(h_nl: torch.Tensor, neural_ot_model, num_steps: int = 10) -> torch.Tensor:
    """Transport NL embeddings to Lean space using a NeuralOTFlow instance.

    h_nl: [B, L, d] or [L, d]
    Returns: same shape tensor transported.
    """
    was_batched = True
    if h_nl.dim() == 2:
        h_nl = h_nl.unsqueeze(0)
        was_batched = False
    device = next(neural_ot_model.parameters()).device
    h_nl = h_nl.to(device)
    with torch.no_grad():
        transported = neural_ot_model.transport_nl_to_lean(h_nl)
    transported = transported.cpu()
    if not was_batched:
        return transported.squeeze(0)
    return transported


def retrieval_metrics(query_embeddings: torch.Tensor, corpus_embeddings: torch.Tensor, ks=(1,5,10,20,50)):
    """Compute Recall@K and MRR for query vs corpus using cosine similarity.

    query_embeddings: [Nq, d]
    corpus_embeddings: [Nc, d]
    Returns dict with recall@k and mrr
    """
    q = torch.nn.functional.normalize(query_embeddings, dim=-1)
    c = torch.nn.functional.normalize(corpus_embeddings, dim=-1)
    sim = q @ c.T
    # for each query, rank corpus
    ranks = torch.argsort(sim, dim=1, descending=True)
    nq = sim.size(0)
    results = {}
    for k in ks:
        topk = ranks[:, :k]
        # assume ground-truth is diagonal (i.e., query i matches corpus i)
        hits = 0
        for i in range(nq):
            if (topk[i] == i).any():
                hits += 1
        results[f"recall@{k}"] = hits / nq

    # MRR
    rr = 0.0
    for i in range(nq):
        pos = (ranks[i] == i).nonzero(as_tuple=False)
        if pos.numel() == 0:
            continue
        rank = int(pos[0].item()) + 1
        rr += 1.0 / rank
    results["mrr"] = rr / nq
    return results

