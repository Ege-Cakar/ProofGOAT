# Augmented Optimal Transport Conditional Flow Matching (A-OT-CFM)

## Project Overview

**HERMES** is a system for learning bidirectional mappings between Natural Language (NL) proof embeddings and Lean formal proof embeddings using Augmented Optimal Transport Flow Matching.

**Repository**: `Ege-Cakar/HERMES`  
**Branch**: `neural-ot`  
**Date**: November 28, 2025

---

## The Core Problem

We aim to learn a continuous mapping between two sequential datasets:
- **Source (A)**: Natural Language proof embeddings (sequence length $n$)
- **Target (B)**: Lean formal proof embeddings (sequence length $m$)

### Challenges
1. **Variable cardinality**: $n \neq m$ - sequences have different lengths
2. **Soft ordering**: Must respect temporal sequence while allowing local reordering
3. **Standard flow matching requires particle conservation** ($n \to n$)

### Solution
Reframe "mass splitting" (creation) and "deletion" as **state transitions within a fixed-capacity container**, using Optimal Transport to create training targets for Conditional Flow Matching.

---

## The Augmented State Space

### Fixed Container
All sequences embedded into fixed size $N_{max}$ (e.g., 256 or 512).

### State Vector
Each token is a vector $\tilde{x} \in \mathbb{R}^{d+1}$:

$$\tilde{x} = [\underbrace{\mathbf{h}}_{\text{Content}}, \quad \underbrace{e}_{\text{Existence}}]$$

- **Content** ($\mathbf{h}$): Semantic embedding (dimension $d = 2048$ from Kimina-Prover)
- **Existence** ($e$): Scalar where $1.0$ = Active, $0.0$ = Void

### Distributed Void Positioning ("The Ether")

**Critical innovation**: Voids are distributed uniformly across spatial domain $[0, 1]$, NOT packed at the end.

- **Active tokens**: positions $p_i \in \{ \frac{1}{n}, \frac{2}{n}, \dots, 1 \}$
- **Void tokens**: positions $p_k \in \{ \frac{1}{N-n}, \frac{2}{N-n}, \dots, 1 \}$

**Why?** Voids act as a "reserve bench" spanning the entire sequence. Creating a token at the beginning grabs a nearby void instead of dragging one from the end.

### Additive Positional Embeddings

Content augmented with sinusoidal positional embedding $P(p)$:

- **Source Active**: $[a_i + P(p_i), \quad 1]$
- **Source Void**: $[\mathbf{0} + P(p_k), \quad 0]$
- **Target Active**: $[b_j + P(p_j), \quad 1]$
- **Target Void**: $[\mathbf{0} + P(p_j), \quad 0]$

---

## Step 1: Alignment via Optimal Transport

For each data pair, solve discrete OT to determine which source tokens map to which target tokens.

### 4-Block Cost Matrix ($N_{max} \times N_{max}$)

| Block | Transition | Cost |
|-------|------------|------|
| 1 | Active → Active | $\|a_i - b_j\|^2 + \lambda(p_i - p_j)^2$ |
| 2 | Active → Void (deletion) | $\alpha_{\text{delete}} + \lambda(p_i - p_j)^2$ |
| 3 | Void → Active (creation) | $\alpha_{\text{create}} + \beta\|b_j\|^2 + \lambda(p_i - p_j)^2$ |
| 4 | Void → Void (background) | $\approx 0$ |

### Default Parameters
```python
lambda_pos = 1.0      # Positional penalty (soft ordering)
alpha_delete = 1.0    # Deletion cost
alpha_create = 1.0    # Creation cost  
beta_create = 0.1     # Creation norm penalty
ot_reg = 0.05         # Sinkhorn regularization (lower = sharper)
```

### Output
Sinkhorn algorithm produces transport plan $\pi^* \in \mathbb{R}^{N_{max} \times N_{max}}$.

---

## Step 2: Conditional Flow Matching

### Barycentric Projection

For each source token $i$, compute weighted target:

$$\tilde{x}_{tgt}[i] = \sum_j \frac{\pi^*[i,j]}{\sum_k \pi^*[i,k]} \cdot \tilde{x}_{target}[j]$$

This handles:
- **Translation**: Active→Active with soft matching
- **Creation**: Void→Active (existence $0 \to 1$, content injected)
- **Deletion**: Active→Void (existence $1 \to 0$)

### Probability Path (Linear)

$$\tilde{x}_t = (1-t)\tilde{x}_{src} + t \cdot \tilde{x}_{tgt}$$

### Target Velocity

$$u_t = \tilde{x}_{tgt} - \tilde{x}_{src}$$

### Loss Function

$$\mathcal{L}(\theta) = \| v_\theta(\tilde{X}_t, t) - u_t \|^2$$

---

## Model Architecture

### Transformer Velocity Field

```
Input: [B, N_max, d+1] (content + existence)
       + time conditioning

Architecture:
  - Input projection: d+1 → transformer_dim (divisible by num_heads)
  - Time embedding: sinusoidal + MLP → transformer_dim
  - Transformer Encoder: num_layers × (self-attention + FFN)
  - Output projection: transformer_dim → d+1

Output: [B, N_max, d+1] velocity
```

### Current Configuration (~313M params)
```python
hidden_dim = 2048        # Must match LLM embedding dim
time_embed_dim = 256
num_layers = 6
num_heads = 8
mlp_ratio = 4.0
dropout = 0.1
```

---

## Data Pipeline

### Source Data
- **NL Embeddings**: `outputs/kimina17_all_nl_embeddings.parquet` (69GB)
- **Lean Embeddings**: `outputs/kimina17_all_lean_embeddings.parquet` (18GB)
- **Total samples**: 44,553 pairs
- **Row groups**: 349 (128 samples each)

### Parquet Schema
```
id: string
hidden: list<list<double>>  # [seq_len, 2048]
```

### Streaming Pipeline
1. Load parquet row-group by row-group (~324MB per group)
2. Create augmented states (distributed voids, positional embeddings)
3. Compute OT coupling on GPU
4. Apply barycentric projection
5. Train on batch

---

## Key Files

### Training Scripts
| File | Description |
|------|-------------|
| `scripts/train_augmented_ot_online.py` | **Main script** - Online OT computation during training |
| `scripts/train_augmented_ot.py` | Precomputed shards version |
| `scripts/compute_augmented_ot.py` | Precompute OT shards |

### Core Functions (in `train_augmented_ot_online.py`)

```python
# Augmented state creation
create_augmented_state(embeddings, N_max, d)
  → (state, positions, is_active)

# 4-block cost matrix
compute_augmented_ot_cost_gpu(src_pos, tgt_pos, src_active, tgt_active, 
                              src_emb, tgt_emb, lambda_pos, alpha_delete,
                              alpha_create, beta_create)
  → C [N_max, N_max]

# OT coupling
compute_augmented_ot_coupling_gpu(src_state, tgt_state, ...)
  → (coupling, permutation)

# Model
AugmentedVelocityField(hidden_dim, time_embed_dim, num_layers, num_heads)
```

### Configuration
```yaml
# project_config.yaml
neural_ot:
  hidden_dim: 2048
  time_embed_dim: 64
  num_layers: 3
  mlp_width: 4096
  max_len: 512
  batch_size: 64
  num_epochs: 10
  learning_rate: 5e-4
  ot_reg: 0.05
  nl_embeddings: "outputs/kimina17_all_nl_embeddings.parquet"
  lean_embeddings: "outputs/kimina17_all_lean_embeddings.parquet"
  output_dir: "outputs/neural_ot"
```

---

## Training

### Command
```bash
cd /workspace/HERMES && source venv/bin/activate && \
python -m scripts.train_augmented_ot_online \
    --config project_config.yaml \
    --max-samples 5000 \
    --batch-size 32 \
    --num-epochs 10 \
    --n-max 256 \
    --ot-reg 0.05 \
    --no-wandb
```

### Training Run (Nov 28, 2025)
- **Run directory**: `outputs/neural_ot/augmented_online_run_20251128_064541/`
- **Completed**: 8 epochs before storage exhaustion
- **Best checkpoint**: `best_model.pt` (epoch 7, step 632)
- **Checkpoint size**: ~3.6GB each

### Observed Training
- Loss decreased from ~4.6 → ~1.9 in first epoch
- Continued improving through epoch 7

---

## Inference (Generation)

To generate Lean proof from NL input:

1. **Preprocess**:
   - Pad NL embeddings with voids to size $N_{max}$
   - Apply positional embeddings
   - Set existence channels (1 for active, 0 for void)

2. **Integrate ODE**:
   - Feed initial state $\tilde{X}_0$ to ODE solver (Euler/RK4)
   - Call $v_\theta$ repeatedly from $t=0$ to $t=1$

3. **Postprocess**:
   - Filter: Keep tokens where existence > 0.5
   - Extract: Remaining vectors are predicted Lean sequence

---

## Known Issues & Notes

### Storage
- Each checkpoint is ~3.6GB
- 10 checkpoints = 36GB
- **TODO**: Add checkpoint cleanup (keep only last N)

### Metrics
- Metrics saved at end via `finish()` - lost if training interrupted
- **TODO**: Save metrics incrementally

### Resume Training
- **TODO**: Add `--resume` flag
- Checkpoints contain full state (model, optimizer, scheduler)

### OT Computation
- Computing OT for each sample is expensive (~1-2s per sample)
- Batched Sinkhorn on GPU helps but still bottleneck
- Alternative: Precompute OT shards offline

---

## Dependencies

```
torch>=2.2
transformers
pyarrow
pandas
numpy
tqdm
pot>=0.9.0  # Python Optimal Transport
pyyaml
matplotlib  # for plotting
```

---

## References

- Flow Matching: Lipman et al. (2022)
- Optimal Transport for ML: Peyré & Cuturi (2019)
- OT-CFM: Tong et al. (2023)
- Kimina-Prover: AI-MO (2024)
