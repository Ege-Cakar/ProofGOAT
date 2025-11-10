# HERMES: Optimal Transport for Bidirectional Proof Translation

HERMES is an experimental pipeline for aligning natural-language mathematical proofs with Lean formal proofs using continuous Optimal Transport (OT) flows. The goal is to investigate whether internal activations from one reasoning domain (NL or Lean) can improve performance in the other, and whether hidden representations contain more transferable reasoning information than raw text.

---

## Project Structure
```
HERMES/
├── scripts/
│ ├── create_pairs.py # Load ProofNet from HF Parquet and build pairs.jsonl
│ ├── prepare_dataset.py # Build metadata and preprocess text
│ ├── extract_embeddings.py # Extract LLM hidden states for NL and Lean proofs
│ └── utils/ # Helpers for alignment, I/O, batching
├── src/
│ ├── io_utils.py # read_jsonl, save_parquet
│ ├── model_utils.py # loading LLMs, tokenizer handling
│ ├── alignment.py # cosine similarity + OT alignment helpers
│ └── flows/ # neural ODE / flow-matching implementation
├── data/
│ ├── pairs.jsonl # combined NL + Lean statements (generated)
│ └── raw/ # downloaded Parquet files from HF
├── outputs/
│ ├── embeddings/ # stored NL / Lean hidden states
│ └── meta/ # metadata parquet files
└── project_config.yaml # model + extraction + dataset configuration
```
---

## Installation

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Dependencies:

- torch (>= 2.2)
- transformers
- accelerate
- pandas
- numpy
- pyarrow
- mlcroissant
- requests

---

## Data Preparation

### Step 1 — Build `pairs.jsonl`
```
python -m scripts.create_pairs
```
This:

- downloads `validation` and `test`
- extracts all rows into a unified JSONL record
- removes empty entries
- writes:

data/pairs.jsonl
---

## Embedding Extraction

Extract hidden states from NL and Lean models.
```
python -m scripts.extract_embeddings --config project_config.yaml
```
This:

- loads NL model: `deepseek-ai/DeepSeek-Prover-V2-7B`
- loads Lean model: `deepseek-ai/deepseek-coder-6.7b-base`
- tokenizes text
- extracts final-layer hidden representations
- saves compressed embedding tensors to:
outputs/embeddings/

---

## Optimal Transport Alignment

We learn a continuous transport map between domains.

Approach:

- Unbalanced dynamic OT for variable-length NL/Lean sequences
- Neural ODE parameterization
- Flow-matching or Benamou–Brenier loss
- LLM weights frozen; only the transport field v_θ is trained

---

## Evaluation

We evaluate:

- **Embedding alignment:** Recall@K, MRR
- **Proof generation correctness:** Lean4 compilation + semantic checks
- **Lean → NL translation:** BLEU / similarity
- **OT trajectory diagnostics:** latent flow visualization

---
