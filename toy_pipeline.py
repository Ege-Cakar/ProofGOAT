import os
import json
import torch
import numpy as np
from google.generativeai import configure, GenerativeModel
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
import os

# Resolve the project root (one directory up from current script)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env from project root
load_dotenv(".env")

api_key = os.getenv("GEMINI_API_KEY")

# ---------------------------------------
# 0. User config
# ---------------------------------------
configure(api_key=api_key)


# HuggingFace model for embeddings (lightweight)
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Output folder
OUT_DIR = "toy_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------
# 1. Example NL pair (from ProofNet)
# ---------------------------------------
example = {
    "id": "Rudin|exercise_1_1a",
    "nl_statement": "If r is rational (r ≠ 0) and x is irrational, prove that r + x is irrational.",
    "nl_proof": "If r and r + x were both rational, then x = (r + x) − r would also be rational.",
    "formal_signature": "(x : ℝ) (y : ℚ) : irrational x → irrational (x + y)"
}

# ---------------------------------------
# 2. Compose Gemini prompt
# ---------------------------------------
prompt = f"""
You are a Lean 4 theorem-proving assistant.
Given the natural language statement and a sketch proof, produce a full Lean 4 theorem + proof that compiles.

Natural language statement:
"{example['nl_statement']}"

Sketch proof:
"{example['nl_proof']}"

Formal target statement type:
"{example['formal_signature']}"

Produce a complete Lean 4 proof using mathlib.
No informal explanations. Output only valid Lean code.
"""

# ---------------------------------------
# 3. Call Gemini
# ---------------------------------------
print("=== Generating Lean proof with Gemini ===")

model = GenerativeModel("gemini-2.5-flash")
response = model.generate_content(prompt)
lean_code = response.text

print("\n=== Lean Proof Generated ===")
print(lean_code)

# Save Lean code
lean_path = os.path.join(OUT_DIR, "example_proof.lean")
with open(lean_path, "w") as f:
    f.write(lean_code)

print(f"\nSaved Lean code to {lean_path}")

# ---------------------------------------
# 4. Load embedding model
# ---------------------------------------
print("\n=== Loading embedding model ===")
tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL)
model = AutoModel.from_pretrained(EMB_MODEL)

# ---------------------------------------
# 5. Embedding extraction function
# ---------------------------------------
def embed(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**tokens)
        # mean pooling
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return emb.cpu()

# ---------------------------------------
# 6. Extract embeddings
# ---------------------------------------
print("\n=== Extracting embeddings ===")
nl_text = example["nl_statement"] + "\n" + example["nl_proof"]

nl_emb = embed(nl_text)
lean_emb = embed(lean_code)

print("NL embedding shape:", nl_emb.shape)
print("Lean embedding shape:", lean_emb.shape)

# ---------------------------------------
# 7. Cosine similarity
# ---------------------------------------
cos_sim = torch.nn.functional.cosine_similarity(nl_emb.unsqueeze(0), lean_emb.unsqueeze(0)).item()
print("\n=== Alignment Result ===")
print("Cosine similarity between NL and Lean embeddings:", cos_sim)

# ---------------------------------------
# 8. Save results
# ---------------------------------------
np.save(os.path.join(OUT_DIR, "nl_emb.npy"), nl_emb.numpy())
np.save(os.path.join(OUT_DIR, "lean_emb.npy"), lean_emb.numpy())

print(f"\nEmbeddings saved in {OUT_DIR}/")
print("✅ Toy pipeline completed.")
print("")