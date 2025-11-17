import re
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Any

def load_model_and_tokenizer(model_name, fp16):
    dtype = torch.float16 if fp16 else torch.float32

    # Load and sanitize config first to handle rope_scaling variants consistently
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    rs = getattr(cfg, "rope_scaling", None)
    if isinstance(rs, dict):
        # Keep only accepted keys; provide defaults if missing
        rs_type = rs.get("type", "linear")
        rs_factor = rs.get("factor", 1.0)
        setattr(cfg, "rope_scaling", {"type": rs_type, "factor": rs_factor})

    # Force CPU, avoid device_map, avoid offloading
    model = AutoModel.from_pretrained(
        model_name,
        config=cfg,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,   # Avoids offloading logic
        trust_remote_code=True,
    )

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
