import re
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any

def load_model_and_tokenizer(model_name, fp16):
    dtype = torch.float16 if fp16 else torch.float32

    # Force CPU, avoid device_map, avoid offloading
    try:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,   # Avoids offloading logic
            trust_remote_code=True,
        )
    except ValueError as e:
        msg = str(e)
        if "rope_scaling" in msg:
            # Extract factor if present; fall back to 1.0
            m = re.search(r"'factor':\s*([0-9]+(?:\.[0-9]+)?)", msg)
            factor = float(m.group(1)) if m else 1.0
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
                rope_scaling={"type": "linear", "factor": factor},
            )
        else:
            raise

    model.to("cpu")  # Explicitly place on CPU

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)

    return model, tok

@torch.no_grad()
def encode_texts(
    texts, tokenizer, model, layer: int = -1, max_length: int = 2048, batch_size: int = 2
):
    """
    Returns list of dicts with:
      - 'hidden': [seq_len, hidden_dim] (torch.float32 on CPU)
      - 'input_ids': [seq_len] (int)
      - 'offset_mapping': for fast token to char spans (optional; only fast tokenizers)
    """
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length, return_offsets_mapping=True
        )
        enc = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in enc.items()}
        outputs = model(**{k:v for k,v in enc.items() if k in ("input_ids","attention_mask")}, output_hidden_states=True)
        hs = outputs.hidden_states  # tuple: [layer0..layerN]
        take = hs[layer]  # [B, T, H]
        for bi in range(take.size(0)):
            seq_len = int(enc["attention_mask"][bi].sum().item())
            item = {
                "hidden": take[bi,:seq_len,:].float().cpu(),
                "input_ids": enc["input_ids"][bi,:seq_len].cpu(),
                "offset_mapping": enc["offset_mapping"][bi,:seq_len].cpu() if "offset_mapping" in enc else None
            }
            results.append(item)
    return results
