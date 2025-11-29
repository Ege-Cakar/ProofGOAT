from model_utils import load_model_and_tokenizer
from io_utils import read_jsonl, verbose_print, load_embedding
import argparse
import yaml
import re
import pandas as pd
import torch
import numpy as np
from pathlib import Path

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

def wrap_prompt_in_query(informal_statement: str, informal_proof: str, has_lean_embeddings: bool = False):
    """
    Wraps the informal statement and proof in a prompt template similar to ProofBridge.
    
    Args:
        informal_statement: The informal statement of the theorem
        informal_proof: The informal proof in natural language
        has_lean_embeddings: Whether lean embeddings (soft virtual tokens) representing the solution are included
    """
    if has_lean_embeddings:
        embedding_note = " Soft virtual tokens representing the target Lean 4 solution are provided as contextual embeddings - use these to guide your formalization."
    else:
        embedding_note = ""
    
    input_query_template = f'''
    You task is to take as input an informal proof in natural language and autoformalize it in Lean 4 with a header. 
    Think step-by-step and ensure that the output formal theorem is compilabile with Lean 4 (version 4.15.0).

    Here is the **actual** informal proof in natural language:
    <informal_statement>
    {informal_statement}
    </informal_statement>

    <informal_proof>
    {informal_proof}
    </informal_proof>

    Now autoformalize it in Lean 4.{embedding_note}
    '''

    return input_query_template

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

    # Check if lean embeddings are provided
    has_lean_embeddings = lean_embedding is not None
    
    input_text = wrap_prompt_in_query(informal_statement.strip(), informal_proof.strip(), has_lean_embeddings=has_lean_embeddings)
    if verbose:
        verbose_print("Wrapped input text: \n ", input_text)

    tokenized_input = nl_tokenizer(input_text, return_tensors="pt").to(nl_model.device)
    if verbose:
        verbose_print("Tokenized input: \n ", tokenized_input)

    # Get model embedding layer to check dimensions
    if hasattr(nl_model, 'get_input_embeddings'):
        embedding_layer = nl_model.get_input_embeddings()
    elif hasattr(nl_model, 'embed_tokens'):
        embedding_layer = nl_model.embed_tokens
    elif hasattr(nl_model, 'model') and hasattr(nl_model.model, 'embed_tokens'):
        embedding_layer = nl_model.model.embed_tokens
    else:
        raise AttributeError("Could not find embedding layer in model")
    
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
        combined_embeddings = torch.cat([lean_emb_batch, input_embeddings], dim=1)
        
        if verbose:
            print("Combined embeddings shape:", combined_embeddings.shape)
            print(f"Sequence breakdown: {num_lean_tokens} lean tokens + {input_embeddings.shape[1]} input tokens = {combined_embeddings.shape[1]} total")
        
        # Adjust attention mask to include lean tokens
        attention_mask = tokenized_input.get("attention_mask", torch.ones_like(input_ids))
        # Create attention mask for lean tokens (all ones)
        # Match the order: lean tokens first, then input tokens
        lean_attention = torch.ones((attention_mask.shape[0], num_lean_tokens), 
                                     dtype=attention_mask.dtype, device=attention_mask.device)
        combined_attention_mask = torch.cat([lean_attention, attention_mask], dim=1)
        
        # Store original input length (without lean tokens) for later extraction
        original_input_length = input_ids.shape[1]
        num_lean_tokens_for_extraction = num_lean_tokens  # Store for later use
        
        if verbose:
            print("Original attention mask shape:", attention_mask.shape)
            print("Combined attention mask shape:", combined_attention_mask.shape)
        
        # Prepare inputs for generation with custom embeddings
        # We need to use inputs_embeds instead of input_ids
        generation_kwargs = {
            "inputs_embeds": combined_embeddings,
            "attention_mask": combined_attention_mask,
            "max_new_tokens": cfg["decode"]["max_new_tokens"],
            "max_time": cfg["decode"]["max_time"],
            "do_sample": cfg["decode"]["do_sample"],
            "temperature": cfg["decode"]["temperature"],
            "top_k": cfg["decode"]["top_k"],
            "top_p": cfg["decode"]["top_p"]
        }
    else:
        # No lean embedding, use standard generation
        generation_kwargs = {
            **tokenized_input,
            "max_new_tokens": cfg["decode"]["max_new_tokens"],
            "max_time": cfg["decode"]["max_time"],
            "do_sample": cfg["decode"]["do_sample"],
            "temperature": cfg["decode"]["temperature"],
            "top_k": cfg["decode"]["top_k"],
            "top_p": cfg["decode"]["top_p"]
        }
        original_input_length = None  # Not needed when not using lean embeddings
        num_lean_tokens_for_extraction = 0  # No lean tokens to skip
    
    # Generate with soft virtual tokens
    outputs = nl_model.generate(**generation_kwargs)
    if verbose:
        print("Generated outputs shape:", outputs.shape)
    
    full_output = nl_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if verbose:
        print("Full output: \n ", full_output)
    
    # Extract the Lean code from the generated output (without the prompt)
    # First try to extract from <formal_proof> tags
    extracted_from_tags = extract_fl_proof_betweenTags(full_output, "formal_proof")
    if verbose:
        verbose_print("Extracted from tags: \n ", extracted_from_tags)
    if extracted_from_tags:
        # Then extract from ```lean4 ... ``` blocks within the tags
        lean_code = extract_fl_proof(extracted_from_tags)
        if lean_code:
            return lean_code
    # Fallback: try extracting directly from generated output
    lean_code = extract_fl_proof(full_output)
    if lean_code:
        return lean_code

    # If no Lean code found, return error message
    return "ERROR [No Lean code found in output]"



def main(cfg: dict):
    id = 0 # test with the first example
    pairs = read_jsonl(cfg["data"]["input_jsonl_with_text"])

    nl_text = pairs[id]["nl_text"]
    nl_proof = pairs[id]["nl_proof"]

    print(f"NL Text: {nl_text}")
    print(f"NL Proof: {nl_proof}")

    lean_embedding = load_embedding(f'{cfg["data"]["out_dir"]}/kimina17_all_lean_embeddings.parquet', id)
    # lean_embedding = None
    decoded_text = decode_text(nl_text, nl_proof, cfg, lean_embedding=lean_embedding, verbose=True)
    print(f"Decoded text: {decoded_text}")

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