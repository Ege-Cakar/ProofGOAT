# Random Embedding Generation for Testing

This script generates synthetic paired embeddings (NL/Lean) with structured distributions to test the Neural OT pipeline without running expensive LLM extraction jobs.

## Quick Start

### 1. Separate Gaussians (Recommended for Testing)

Generate NL and Lean embeddings from two different Gaussian distributions with distinct means and covariances:

```bash
uv run python -m scripts.generate_random_embeddings \
    --distribution-type separate \
    --num-samples 64 \
    --hidden-dim 2048 \
    --output-dir outputs/random_embeddings \
    --mean-shift 2.0 \
    --nl-cov-scale 1.0 \
    --lean-cov-scale 1.5
```

**Properties:**
- NL ~ N(μ_NL, Σ_NL) with structured sinusoidal mean
- Lean ~ N(μ_Lean, Σ_Lean) with structured cosinusoidal mean
- Non-trivial optimal transport map to learn
- Mean shift controls separation between distributions

### 2. Affine Transformation

Generate pairs where Lean = A * NL + b + noise (learnable linear map):

```bash
uv run python -m scripts.generate_random_embeddings \
    --distribution-type affine \
    --num-samples 64 \
    --hidden-dim 2048 \
    --output-dir outputs/random_embeddings \
    --affine-scale 0.9 \
    --rotation-angle 0.5 \
    --noise-scale 0.1
```

**Properties:**
- True transformation is affine (rotation + scaling + translation)
- Noise scale controls difficulty
- Ground truth transport map is known

### 3. Coupled (Simple Baseline)

Generate pairs where Lean = NL + noise (minimal structure):

```bash
uv run python -m scripts.generate_random_embeddings \
    --distribution-type coupled \
    --num-samples 64 \
    --hidden-dim 2048 \
    --output-dir outputs/random_embeddings \
    --noise-scale 0.05
```

**Properties:**
- Simplest case: just adding noise
- Good for sanity checking
- Limited learning required

## Parameters

### Common Parameters
- `--num-samples`: Number of paired examples (default: 64)
- `--min-len`: Minimum sequence length (default: 32)
- `--max-len`: Maximum sequence length (default: 128)
- `--hidden-dim`: Embedding dimension (default: 2048)
- `--seed`: Random seed for reproducibility (default: 0)

### Separate Gaussian Parameters
- `--mean-shift`: Shift between NL and Lean means (default: 2.0)
- `--nl-cov-scale`: Covariance scale for NL (default: 1.0)
- `--lean-cov-scale`: Covariance scale for Lean (default: 1.5)

### Affine Transform Parameters
- `--affine-scale`: Scaling factor for transformation matrix (default: 0.8)
- `--rotation-angle`: Rotation angle in radians (default: 0.3)
- `--noise-scale`: Additive noise scale (default: 0.05)

## Output

Creates two files:
- `nl_embeddings_random.parquet`: Natural language embeddings
- `lean_embeddings_random.parquet`: Lean proof embeddings

Format: Each row has `id` (string) and `hidden` (list of token vectors)

## Using Generated Embeddings for Training

Update `project_config.yaml`:

```yaml
neural_ot:
  nl_embeddings: "outputs/random_embeddings/nl_embeddings_random.parquet"
  lean_embeddings: "outputs/random_embeddings/lean_embeddings_random.parquet"
  hidden_dim: 2048  # Must match --hidden-dim
```

Then train:

```bash
uv run python -m scripts.train_neural_ot --config project_config.yaml --no-wandb
```

## Difficulty Levels

**Easy** (should converge quickly):
```bash
--distribution-type coupled --noise-scale 0.05
```

**Medium** (non-trivial but learnable):
```bash
--distribution-type separate --mean-shift 2.0
```

**Hard** (complex transformation):
```bash
--distribution-type affine --affine-scale 0.9 --rotation-angle 0.5 --noise-scale 0.2
```

## Verifying Generated Data

Check statistics:
```python
import pandas as pd
import numpy as np

df_nl = pd.read_parquet("outputs/random_embeddings/nl_embeddings_random.parquet")
df_lean = pd.read_parquet("outputs/random_embeddings/lean_embeddings_random.parquet")

# Get first sample
nl_sample = np.vstack([np.array(x) for x in df_nl['hidden'].iloc[0]])
lean_sample = np.vstack([np.array(x) for x in df_lean['hidden'].iloc[0]])

print(f"NL shape: {nl_sample.shape}")
print(f"NL mean: {nl_sample.mean():.4f}, std: {nl_sample.std():.4f}")
print(f"Lean mean: {lean_sample.mean():.4f}, std: {lean_sample.std():.4f}")
print(f"Distance: {np.linalg.norm(nl_sample - lean_sample):.4f}")
```

## Tips

1. **Match dimensions**: Make sure `--hidden-dim` matches your actual embeddings or config
2. **Start simple**: Test with `coupled` distribution first, then try `separate`
3. **Vary difficulty**: Adjust `mean-shift` or `noise-scale` to control learning difficulty
4. **Reproducibility**: Use `--seed` for consistent results across runs
5. **Size**: Start with `--num-samples 64` for quick tests, increase for longer training

## Examples

**Quick sanity check** (easy, fast):
```bash
uv run python -m scripts.generate_random_embeddings \
    --distribution-type coupled \
    --num-samples 32 \
    --min-len 16 \
    --max-len 64 \
    --hidden-dim 2048
```

**Standard test** (realistic difficulty):
```bash
uv run python -m scripts.generate_random_embeddings \
    --distribution-type separate \
    --num-samples 128 \
    --hidden-dim 2048 \
    --mean-shift 2.5 \
    --lean-cov-scale 2.0
```

**Challenging test** (requires more training):
```bash
uv run python -m scripts.generate_random_embeddings \
    --distribution-type affine \
    --num-samples 256 \
    --hidden-dim 2048 \
    --affine-scale 1.2 \
    --rotation-angle 1.0 \
    --noise-scale 0.3
```
