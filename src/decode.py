import argparse
import os
from sqlite3 import Row
import sys
import yaml
import re
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import pprint
from pathlib import Path
from typing import Optional

# Set up paths: add project root and src directory to sys.path
# This allows imports from both src (local modules) and scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model_utils import load_model_and_tokenizer
from io_utils import read_jsonl, verbose_print, load_embedding, load_examples
from scripts.test_lean import test_lean_code

def extract_fl_proof_betweenTags(text, TAG):
    """
    Extracts content between XML-style tags like <TAG>...</TAG>
    """
    if text is not None:
        text = str(text)
    else:
        text = ""
    START_DELIMITER = "<{}>".format(TAG)
    END_DELIMITER = "</{}>".format(TAG)
    if (START_DELIMITER in text) and (END_DELIMITER in text):
        inner_str = (text.split(START_DELIMITER)[-1].split(END_DELIMITER)[0]).strip()
        return inner_str
    return ""

def extract_fl_proof(text: str) -> str:
    """
    Extracts the last content enclosed in ```lean4 ... ``` from the input text.

    Returns the last match as a string, including newlines. 
    Returns an empty string if no match is found.
    """
    pattern = r"```lean4\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()  # last match
    return ""

def wrap_prompt_in_query(informal_statement: str, informal_proof: str, has_lean_embeddings: bool = False, examples_in_prompt: Optional[str] = None):
    """
    Wraps the informal statement and proof in a prompt template for Kimina model.
    Returns messages in the format expected by apply_chat_template.
    
    Args:
        informal_statement: The informal statement of the theorem
        informal_proof: The informal proof in natural language
        has_lean_embeddings: Whether lean embeddings (soft virtual tokens) representing the solution are included
        examples_in_prompt: Optional string containing example problems and solutions to include in the prompt
    """
    if has_lean_embeddings:
        embedding_note = " Soft virtual tokens representing the target Lean 4 solution are provided as contextual embeddings - use these to guide your formalization."
    else:
        embedding_note = ""
    
    # Construct the user prompt following Kimina's expected format
    examples_section = ""
    if examples_in_prompt:
        examples_section = f"""Here are some examples of similar problems and their solutions:

{examples_in_prompt}

Now, solve the following problem:
"""
    
    user_prompt = f"""Translate the following proof in Lean 4.
{examples_section}
# Problem:
{informal_statement}

# Informal Proof:
{informal_proof}

Now autoformalize it in Lean 4 with a header.

<formal_proof>
```lean4
```
</formal_proof>{embedding_note}"""

    # Return messages in the format expected by apply_chat_template
    messages = [
        {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages

def decode_text(informal_statement: str, informal_proof: str, cfg: dict, lean_embedding: list = None, verbose: bool = False):
    """
    Given NL theorem-proof, decode to Lean theorem-proof.

    Args:
        informal_statement: The informal statement of the theorem
        informal_proof: The informal proof in natural language
        cfg: Configuration dictionary
        lean_embedding: The embedding of the Lean theorem-proof to be decoded as soft virtual tokens
                       Should be a numpy array with shape [num_tokens, hidden_dim]
        verbose: Whether to print verbose output
    """
    # Get local_dir from config, or None if not set
    local_dir = cfg["models"].get("local_dir")
    nl_model, nl_tokenizer = load_model_and_tokenizer(cfg["models"]["nl_model"], cfg["extract"]["fp16"], causal=True, local_dir=local_dir)


    # Get model embedding layer to check dimensions
    if hasattr(nl_model, 'get_input_embeddings'):
        embedding_layer = nl_model.get_input_embeddings()
    elif hasattr(nl_model, 'embed_tokens'):
        embedding_layer = nl_model.embed_tokens
    elif hasattr(nl_model, 'model') and hasattr(nl_model.model, 'embed_tokens'):
        embedding_layer = nl_model.model.embed_tokens
    else:
        raise AttributeError("Could not find embedding layer in model")

    # TEMPORARY: testing
    # pairs = read_jsonl(cfg["data"]["input_jsonl_with_text"])
    # lean_proof = pairs[1]["lean_proof"]
    # lean_tokenized = nl_tokenizer(lean_proof, return_tensors="pt").to(nl_model.device)
    # lean_input_ids = lean_tokenized["input_ids"]
    # lean_embeddings = embedding_layer(lean_input_ids)  # [batch_size, seq_len, hidden_dim]
    # lean_embedding = lean_embeddings[0].detach().cpu().numpy()  # [seq_len, hidden_dim]
    # print(f"Converted Lean proof to embeddings: shape {lean_embedding.shape}")

    # Check if lean embeddings are provided
    has_lean_embeddings = lean_embedding is not None

    # Load examples if needed
    examples_in_prompt = None
    if cfg["decode"].get("use_examples_in_prompt", False):
        examples_path = cfg["decode"].get("examples_path", "data/examples.yaml")
        examples_in_prompt = load_examples(examples_path, verbose=verbose)
    
    # Get messages in chat format
    messages = wrap_prompt_in_query(informal_statement.strip(), informal_proof.strip(), has_lean_embeddings=has_lean_embeddings, examples_in_prompt=examples_in_prompt)
    
    # Apply chat template to format the prompt according to Kimina's expected format
    try:
        formatted_text = nl_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        if verbose:
            verbose_print("Formatted prompt (from chat template): \n ", formatted_text)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not apply chat template: {e}. Falling back to plain text.")
        # Fallback: construct plain text from messages
        formatted_text = messages[0]["content"] + "\n\n" + messages[1]["content"]
    
    # Tokenize the formatted text
    tokenized_input = nl_tokenizer(formatted_text, return_tensors="pt").to(nl_model.device)
    if verbose:
        verbose_print("Tokenized input: \n ", tokenized_input)
    
    model_hidden_dim = embedding_layer.weight.shape[1]
    model_dtype = embedding_layer.weight.dtype
    model_device = embedding_layer.weight.device
    
    if verbose:
        verbose_print("Model embedding dimension:", model_hidden_dim)
        verbose_print("Model dtype:", model_dtype)
        verbose_print("Model device:", model_device)
    
    # Process lean_embedding as soft virtual tokens if provided
    if has_lean_embeddings:
        # Convert to numpy array if it's not already
        if isinstance(lean_embedding, torch.Tensor):
            lean_emb = lean_embedding.detach().cpu().numpy()
        elif isinstance(lean_embedding, np.ndarray):
            lean_emb = lean_embedding
        elif isinstance(lean_embedding, (list, tuple)):
            lean_emb = np.array(lean_embedding)
        else:
            raise TypeError(f"lean_embedding must be numpy array, torch tensor, or list, got {type(lean_embedding)}")
        
        # Check and reshape if needed
        if lean_emb.ndim == 1:
            # If 1D, assume it needs to be reshaped - this shouldn't happen normally
            raise ValueError(f"lean_embedding is 1D with shape {lean_emb.shape}, expected 2D [num_tokens, hidden_dim]")
        elif lean_emb.ndim == 2:
            num_lean_tokens, lean_hidden_dim = lean_emb.shape
            # Check dimension match
            if lean_hidden_dim != model_hidden_dim:
                raise ValueError(
                    f"Dimension mismatch: lean_embedding has hidden_dim={lean_hidden_dim}, "
                    f"but model expects {model_hidden_dim}. Please ensure embeddings are extracted "
                    f"from the same model architecture or have compatible dimensions."
                )
        else:
            raise ValueError(f"lean_embedding must be 2D [num_tokens, hidden_dim], got shape {lean_emb.shape}")
        
        if verbose:
            print("Lean embedding shape:", lean_emb.shape)
            print("Number of lean tokens:", num_lean_tokens)
        
        # Convert to torch tensor with correct dtype and device
        lean_emb_tensor = torch.from_numpy(lean_emb).to(dtype=model_dtype).to(model_device)
        
        # Get input embeddings from tokenized input
        input_ids = tokenized_input["input_ids"]
        input_embeddings = embedding_layer(input_ids)  # [batch_size, seq_len, hidden_dim]
        
        if verbose:
            print("Input embeddings shape:", input_embeddings.shape)
            print("Lean embedding tensor shape:", lean_emb_tensor.shape)
        
        # Prepend lean embeddings as soft virtual tokens
        # lean_emb_tensor: [num_lean_tokens, hidden_dim]
        # Need to add batch dimension: [1, num_lean_tokens, hidden_dim]
        lean_emb_batch = lean_emb_tensor.unsqueeze(0)  # [1, num_lean_tokens, hidden_dim]
        
        # Concatenate: [batch_size, num_lean_tokens + seq_len, hidden_dim]
        # Put lean embeddings FIRST (as context), then input embeddings
        combined_embeddings = torch.cat([input_embeddings, lean_emb_batch], dim=1)
        
        # Get total sequence length after concatenation
        batch_size, total_seq_len, hidden_dim = combined_embeddings.shape
        
        if verbose:
            print("Combined embeddings shape:", combined_embeddings.shape)
            print(f"Sequence breakdown: {num_lean_tokens} lean tokens + {input_embeddings.shape[1]} input tokens = {total_seq_len} total")
        
        # For Qwen-based models (like Kimina), we need to provide input_ids even when using inputs_embeds
        # Use padding token IDs for the lean embedding positions
        # Get padding token ID from tokenizer (fallback to 0 if not available)
        pad_token_id = nl_tokenizer.pad_token_id
        if pad_token_id is None:
            # Some tokenizers don't have explicit pad_token_id, try eos_token_id or unk_token_id
            pad_token_id = nl_tokenizer.eos_token_id if nl_tokenizer.eos_token_id is not None else (
                nl_tokenizer.unk_token_id if nl_tokenizer.unk_token_id is not None else 0
            )
            if verbose:
                print(f"Warning: No pad_token_id found, using {pad_token_id} (eos/unk/0)")
        
        # Create input_ids: actual input_ids first, then padding tokens for lean positions
        # Match the order of combined_embeddings: input_embeddings first, then lean_emb_batch
        lean_padding_ids = torch.full(
            (batch_size, num_lean_tokens),
            pad_token_id,
            dtype=input_ids.dtype,
            device=model_device
        )
        combined_input_ids = torch.cat([input_ids, lean_padding_ids], dim=1)
        
        if verbose:
            print(f"Using pad_token_id={pad_token_id} for lean embedding positions")
            print("Combined input_ids shape:", combined_input_ids.shape)
        
        # Adjust attention mask to include lean tokens
        attention_mask = tokenized_input.get("attention_mask", torch.ones_like(input_ids))
        # Create attention mask for lean tokens (all ones)
        # Match the order of embeddings: input tokens first, then lean tokens
        lean_attention = torch.ones((attention_mask.shape[0], num_lean_tokens), 
                                     dtype=attention_mask.dtype, device=attention_mask.device)
        combined_attention_mask = torch.cat([attention_mask, lean_attention], dim=1)
        
        if verbose:
            print("Original attention mask shape:", attention_mask.shape)
            print("Combined attention mask shape:", combined_attention_mask.shape)
        
        # Prepare inputs for generation with custom embeddings
        # For Qwen-based models, provide both inputs_embeds and input_ids
        # The inputs_embeds will be used, but input_ids is required for proper model behavior
        generation_kwargs = {
            "inputs_embeds": combined_embeddings,
            "input_ids": combined_input_ids,
            "attention_mask": combined_attention_mask,
            "max_new_tokens": cfg["decode"]["max_new_tokens"],
            "max_time": cfg["decode"]["max_time"],
            "temperature": cfg["decode"]["temperature"],
            "top_p": cfg["decode"]["top_p"],
        }
    else:
        # No lean embedding, use standard generation
        generation_kwargs = {
            **tokenized_input,
            "max_new_tokens": cfg["decode"]["max_new_tokens"],
            "max_time": cfg["decode"]["max_time"],
            "temperature": cfg["decode"]["temperature"],
            "top_p": cfg["decode"]["top_p"],
        }
    
    # Generate with soft virtual tokens
    outputs = nl_model.generate(**generation_kwargs)
    if verbose:
        print("Generated outputs shape:", outputs.shape)
    
    full_output = nl_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if verbose:
        print("Full output: \n ", full_output)
    
    # Extract the Lean code from the generated output
    lean_code = extract_fl_proof(full_output)
    if lean_code:
        return lean_code

    # If no Lean code found, return error message
    return "ERROR [No Lean code found in output]"


def decode_embedding(embedding, cfg: dict, verbose: bool = False):
    """
    Given a Lean embedding, exact or approximate, decode to Lean code.
    
    Args:
        embedding: numpy array, torch tensor, or list/tuple with shape (num_tokens, hidden_dim) 
                  - last-layer embedding just before being tokenized into output tokens
        cfg: Configuration dictionary containing model information
        verbose: Whether to print verbose output
        
    Returns:
        decoded_text: Decoded Lean code string
    """
    # Load model and tokenizer
    local_dir = cfg["models"].get("local_dir")
    model, tokenizer = load_model_and_tokenizer(
        cfg["models"]["nl_model"], 
        cfg["extract"]["fp16"], 
        causal=True, 
        local_dir=local_dir
    )
    
    # Get device
    device = next(model.parameters()).device
    
    # Get token embedding matrix
    if hasattr(model, 'get_input_embeddings'):
        embedding_layer = model.get_input_embeddings()
    elif hasattr(model, 'embed_tokens'):
        embedding_layer = model.embed_tokens
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embedding_layer = model.model.embed_tokens
    else:
        raise AttributeError("Could not find embedding layer in model")
    
    token_embeds = embedding_layer.weight  # (vocab_size, d_model)
    token_embeds = token_embeds.to(device)
    
    # Get the dtype from token embeddings to match it
    model_dtype = token_embeds.dtype
    
    # Convert embedding to torch tensor with matching dtype
    if isinstance(embedding, np.ndarray):
        embedding_tensor = torch.from_numpy(embedding).to(dtype=model_dtype)
    elif isinstance(embedding, torch.Tensor):
        embedding_tensor = embedding.to(dtype=model_dtype)
    elif isinstance(embedding, (list, tuple)):
        # Convert list/tuple to numpy array first, then to tensor
        embedding_array = np.array(embedding)
        embedding_tensor = torch.from_numpy(embedding_array).to(dtype=model_dtype)
    else:
        raise TypeError(f"embedding must be numpy array, torch tensor, or list/tuple, got {type(embedding)}")
    
    embedding_tensor = embedding_tensor.to(device)
    
    # Check dimensions
    if embedding_tensor.ndim == 1:
        # Single token embedding: reshape to (1, hidden_dim)
        embedding_tensor = embedding_tensor.unsqueeze(0)
    elif embedding_tensor.ndim != 2:
        raise ValueError(f"embedding must be 1D or 2D, got shape {embedding_tensor.shape}")
    
    num_tokens, hidden_dim = embedding_tensor.shape
    vocab_size = token_embeds.size(0)
    
    if hidden_dim != token_embeds.size(1):
        raise ValueError(
            f"Dimension mismatch: embedding has hidden_dim={hidden_dim}, "
            f"but model expects {token_embeds.size(1)}"
        )
    
    if verbose:
        print(f"Decoding embedding with shape: {embedding_tensor.shape}")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Hidden dimension: {hidden_dim}")
    
    model.eval()
    with torch.no_grad():
        # Normalize for cosine similarity
        x_norm = F.normalize(embedding_tensor, p=2, dim=-1)  # (num_tokens, hidden_dim)
        vocab_norm = F.normalize(token_embeds, p=2, dim=-1)  # (vocab_size, hidden_dim)
        
        # Compute cosine similarity: (num_tokens, vocab_size)
        similarity = x_norm @ vocab_norm.T
        
        # Get the most likely token for each position (greedy decoding)
        top_indices = torch.argmax(similarity, dim=-1)  # (num_tokens,)
        top_probs = torch.gather(
            torch.softmax(similarity, dim=-1),
            dim=-1,
            index=top_indices.unsqueeze(-1)
        ).squeeze(-1)  # (num_tokens,)
    
    # Convert token IDs to text
    token_ids = top_indices.cpu().tolist()
    
    if verbose:
        print(f"Decoded {num_tokens} tokens")
        print(f"Top token IDs: {token_ids[:10]}...")  # Show first 10
    
    # Decode tokens to text
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    
    if verbose:
        print(f"Decoded text (first 200 chars): {decoded_text[:200]}...")
    
    # Extract Lean code from the decoded text if it's wrapped in markdown or XML
    lean_code = extract_fl_proof(decoded_text)
    if lean_code:
        return lean_code
    
    # Return the decoded text as-is if no markdown formatting found
    return decoded_text

def main(cfg: dict):
    rows = range(1, 10)
    for row in rows:
        lean_embedding, id = load_embedding(f'{cfg["data"]["out_dir"]}/kimina17_all_lean_embeddings.parquet', row=row)
        pairs = read_jsonl(cfg["data"]["input_jsonl_with_text"])

        nl_text = pairs[id]["nl_text"]
        nl_proof = pairs[id]["nl_proof"]
        print(f"============== ID: {id} ==============")

        print(f"NL Text: {nl_text}")
        print(f"NL Proof: {nl_proof}")

        # shortcut_decoded_text = decode_embedding(lean_embedding, cfg, verbose=True)
        # print(f"Shortcut decoded text: {shortcut_decoded_text}")
        if cfg["decode"].get("soft_tokens", False):
            lean_embedding = None
        decoded_text = decode_text(nl_text, nl_proof, cfg, lean_embedding=lean_embedding, verbose=False)

        result = test_lean_code(decoded_text, verbose=False)
        pprint.pprint(f"Test result: {result}")
        if result["success"]:
            print("✅ decoded text:\n", decoded_text)
        sys.stdout.flush()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="project_config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Set up local_dir for models if not specified in config
    # Use relative path based on current working directory
    if "models" not in cfg:
        cfg["models"] = {}
    if "local_dir" not in cfg["models"] or cfg["models"]["local_dir"] is None:
        # Get current working directory and create models subdirectory path
        cwd = Path.cwd()
        models_dir = cwd / "cached_models"
        cfg["models"]["local_dir"] = str(models_dir.resolve())
        print(f"Setting models local_dir to: {cfg['models']['local_dir']}")

    main(cfg)