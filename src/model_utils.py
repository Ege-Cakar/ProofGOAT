import json
import re
from typing import Any, Dict, Optional

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

def _infer_rope_override(model_name: str) -> Optional[Dict[str, Any]]:
    """Read raw config.json and, if YaRN-style rope_scaling is present,
    produce a sanitized override accepted by transformers (type+factor only).
    Returns None if no override is needed or config cannot be read.
    """
    try:
        # Prefer local cache first to avoid network; fall back to remote if available
        try:
            cfg_path = hf_hub_download(model_name, filename="config.json", local_files_only=True)
        except Exception:
            cfg_path = hf_hub_download(model_name, filename="config.json", local_files_only=False)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return None

    rs = cfg.get("rope_scaling")
    if isinstance(rs, dict):
        rs_type = str(rs.get("type", "linear")).lower()
        factor = float(rs.get("factor", 1.0))
        # If type is not one of accepted values, normalize to 'linear'
        if rs_type not in {"linear", "dynamic"}:
            rs_type = "linear"
        return {"type": rs_type, "factor": factor}
    return None


def load_model_and_tokenizer(model_name, fp16, causal: bool = False):
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if fp16 and use_cuda else torch.float32

    # Compute a rope_scaling override before any config validation occurs
    rope_override = _infer_rope_override(model_name)

    # Force CPU, avoid device_map, avoid offloading. Provide override if present
    model_class = AutoModelForCausalLM if causal else AutoModel
    rope_kwargs = {"rope_scaling": rope_override} if rope_override is not None else {}
    
    try:
        model = model_class.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            device_map="auto" if use_cuda else None,
            **rope_kwargs,
        )
    except ValueError as e:
        # If validation still fails for rope_scaling, disable it entirely as last resort
        if "rope_scaling" in str(e):
            model = model_class.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
                device_map="auto" if use_cuda else None,
                rope_scaling=None,
            )
        else:
            raise

    # If not using CUDA, keep on CPU; otherwise device_map handles placement
    if not use_cuda:
        model.to("cpu")

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
