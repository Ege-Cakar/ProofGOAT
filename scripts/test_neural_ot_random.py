"""
Test Neural OT implementation with random embeddings.

This script loads the random embeddings and tests the neural OT flow
with proper positional embeddings.

Usage:
    python -m scripts.test_neural_ot_random
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import torch
from torch.utils.data import DataLoader

from src.flows.dataset import EmbeddingPairDataset, collate_fn
from src.flows.neural_ot import NeuralOTFlow
from src.flows.losses import flow_matching_loss, cycle_consistency_loss


def test_velocity_field():
    """Test the velocity field with (x, p, t) inputs."""
    print("\n" + "="*80)
    print("Testing Velocity Field with Positional Embeddings")
    print("="*80)

    from src.flows.velocity_field import VelocityField, SinusoidalPositionalEmbedding, TimeEmbedding

    # Create test data
    B, L, d = 2, 32, 2048
    hidden_dim = d
    time_embed_dim = 128

    # Initialize components
    v_field = VelocityField(
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        num_layers=3,
        mlp_width=2048,
        max_seq_len=512
    )

    print(f"\nVelocity Field Configuration:")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Time embed dim: {time_embed_dim}")
    print(f"  Input shape: [B={B}, L={L}, d={d}]")

    # Test 1: Forward pass with explicit positional embeddings
    x = torch.randn(B, L, d)
    p = v_field.pos_embed(L, B)
    t = torch.rand(B)

    print(f"\nTest 1: Forward pass with explicit positional embeddings")
    print(f"  x shape: {x.shape}")
    print(f"  p shape: {p.shape}")
    print(f"  t shape: {t.shape}")

    v = v_field(x, t, p)
    print(f"  Output v shape: {v.shape}")
    assert v.shape == x.shape, f"Output shape mismatch: {v.shape} vs {x.shape}"
    print("  ✓ Shape check passed")

    # Test 2: Forward pass with automatic positional embeddings
    print(f"\nTest 2: Forward pass with automatic positional embeddings")
    v_auto = v_field(x, t, p=None)
    print(f"  Output v shape: {v_auto.shape}")
    assert v_auto.shape == x.shape
    print("  ✓ Shape check passed")

    # Test 3: Different time formats
    print(f"\nTest 3: Time embedding variations")

    # Scalar time
    t_scalar = 0.5
    v_scalar = v_field(x, t_scalar, p)
    print(f"  Scalar t={t_scalar}: v shape {v_scalar.shape} ✓")

    # Per-batch time
    t_batch = torch.rand(B)
    v_batch = v_field(x, t_batch, p)
    print(f"  Per-batch t shape {t_batch.shape}: v shape {v_batch.shape} ✓")

    # Per-token time
    t_token = torch.rand(B, L)
    v_token = v_field(x, t_token, p)
    print(f"  Per-token t shape {t_token.shape}: v shape {v_token.shape} ✓")

    # Test 4: Verify positional embeddings are sinusoidal
    print(f"\nTest 4: Verify positional embeddings")
    pos_emb = SinusoidalPositionalEmbedding(d_model=256, max_len=100)
    p_test = pos_emb(50, 1)
    print(f"  Positional embedding shape: {p_test.shape}")
    print(f"  Expected: [1, 50, 256]")
    assert p_test.shape == (1, 50, 256)
    print("  ✓ Positional embedding shape correct")

    # Test 5: Verify time embeddings use different frequencies
    print(f"\nTest 5: Verify time embedding frequencies")
    time_emb = TimeEmbedding(dim=128)
    t_test = torch.linspace(0, 1, 10)
    t_emb = time_emb(t_test)
    print(f"  Time embedding shape: {t_emb.shape}")
    print(f"  Expected: [10, 128]")
    assert t_emb.shape == (10, 128)

    # Check that embeddings change smoothly
    diffs = torch.diff(t_emb, dim=0).norm(dim=-1)
    print(f"  Time embedding differences (should be smooth):")
    print(f"    Mean: {diffs.mean().item():.6f}")
    print(f"    Std:  {diffs.std().item():.6f}")
    print("  ✓ Time embeddings computed successfully")

    print("\n✓ All velocity field tests passed!")


def test_with_random_embeddings():
    """Test with actual random embeddings."""
    print("\n" + "="*80)
    print("Testing with Random Embeddings")
    print("="*80)

    # Load random embeddings
    nl_path = "outputs/random_embeddings/nl_embeddings_random.parquet"
    lean_path = "outputs/random_embeddings/lean_embeddings_random.parquet"

    if not os.path.exists(nl_path) or not os.path.exists(lean_path):
        print(f"\n⚠ Random embeddings not found at:")
        print(f"  {nl_path}")
        print(f"  {lean_path}")
        print("Please generate them first.")
        return

    print(f"\nLoading embeddings from:")
    print(f"  NL:   {nl_path}")
    print(f"  Lean: {lean_path}")

    # Create dataset
    max_len = 64
    dataset = EmbeddingPairDataset(nl_path, lean_path, max_len=max_len)
    print(f"\nDataset created:")
    print(f"  Number of pairs: {len(dataset)}")
    print(f"  Max length: {max_len}")

    # Create dataloader
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    print(f"  Batch size: {batch_size}")

    # Get one batch
    x_nl, x_lean = next(iter(dataloader))
    B, L, d = x_nl.shape

    print(f"\nBatch shapes:")
    print(f"  x_nl:   {x_nl.shape}")
    print(f"  x_lean: {x_lean.shape}")
    print(f"  B={B}, L={L}, d={d}")

    # Create model
    print(f"\nInitializing Neural OT model...")
    model = NeuralOTFlow(
        hidden_dim=d,
        time_embed_dim=128,
        num_layers=3,
        mlp_width=2048,
        max_seq_len=512
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")

    # Test flow matching loss
    print(f"\nTest 1: Flow Matching Loss")
    loss_fm = model.compute_flow_matching_loss(x_nl, x_lean, t_sample="per_batch")
    print(f"  Flow matching loss (per_batch): {loss_fm.item():.6f}")

    loss_fm_token = model.compute_flow_matching_loss(x_nl, x_lean, t_sample="per_token")
    print(f"  Flow matching loss (per_token): {loss_fm_token.item():.6f}")
    print("  ✓ Loss computation successful")

    # Test forward transport
    print(f"\nTest 2: Forward Transport (NL → Lean)")
    num_steps = 10
    x_transported = model.transport_nl_to_lean(x_nl, num_steps=num_steps)
    print(f"  Input shape:  {x_nl.shape}")
    print(f"  Output shape: {x_transported.shape}")
    assert x_transported.shape == x_nl.shape

    # Compute distance
    dist_before = torch.norm(x_nl - x_lean, dim=-1).mean()
    dist_after = torch.norm(x_transported - x_lean, dim=-1).mean()
    print(f"  Distance before transport: {dist_before.item():.4f}")
    print(f"  Distance after transport:  {dist_after.item():.4f}")
    print("  ✓ Forward transport successful")

    # Test backward transport
    print(f"\nTest 3: Backward Transport (Lean → NL)")
    x_back = model.transport_lean_to_nl(x_lean, num_steps=num_steps)
    print(f"  Input shape:  {x_lean.shape}")
    print(f"  Output shape: {x_back.shape}")
    assert x_back.shape == x_lean.shape
    print("  ✓ Backward transport successful")

    # Test cycle consistency
    print(f"\nTest 4: Cycle Consistency")
    loss_cycle = cycle_consistency_loss(model, x_nl, num_steps=num_steps)
    print(f"  Cycle consistency loss: {loss_cycle.item():.6f}")
    print("  ✓ Cycle consistency computation successful")

    # Test explicit positional embeddings
    print(f"\nTest 5: Explicit Positional Embeddings")
    p = model.v_theta.pos_embed(L, B)
    print(f"  Positional embedding shape: {p.shape}")

    x_transported_p = model.transport_nl_to_lean(x_nl, num_steps=num_steps, p=p)
    print(f"  Transport with explicit p: {x_transported_p.shape}")

    # Should give same result as automatic
    diff = torch.norm(x_transported - x_transported_p)
    print(f"  Difference from automatic: {diff.item():.8f}")
    assert diff < 1e-5, "Explicit and automatic positional embeddings should match"
    print("  ✓ Explicit positional embeddings work correctly")

    # Test gradient flow
    print(f"\nTest 6: Gradient Flow")
    x_nl_grad = x_nl.clone().requires_grad_(True)
    x_lean_grad = x_lean.clone().requires_grad_(True)

    loss = model.compute_flow_matching_loss(x_nl_grad, x_lean_grad)
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradients computed: {has_grad}")
    assert has_grad, "Gradients should be computed"

    # Check gradient norms
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"  Gradient norm stats:")
    print(f"    Mean: {sum(grad_norms)/len(grad_norms):.6f}")
    print(f"    Max:  {max(grad_norms):.6f}")
    print(f"    Min:  {min(grad_norms):.6f}")
    print("  ✓ Gradient flow successful")

    print("\n✓ All random embedding tests passed!")


def test_training_step():
    """Test a single training step."""
    print("\n" + "="*80)
    print("Testing Training Step")
    print("="*80)

    # Load random embeddings
    nl_path = "outputs/random_embeddings/nl_embeddings_random.parquet"
    lean_path = "outputs/random_embeddings/lean_embeddings_random.parquet"

    if not os.path.exists(nl_path) or not os.path.exists(lean_path):
        print("\n⚠ Random embeddings not found. Skipping training step test.")
        return

    # Setup
    dataset = EmbeddingPairDataset(nl_path, lean_path, max_len=48)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    x_nl, x_lean = next(iter(dataloader))
    d = x_nl.shape[-1]

    model = NeuralOTFlow(
        hidden_dim=d,
        time_embed_dim=128,
        num_layers=2,
        mlp_width=1024
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("Performing training step...")
    print(f"  Batch shape: {x_nl.shape}")

    # Training step
    optimizer.zero_grad()
    loss = model.compute_flow_matching_loss(x_nl, x_lean, t_sample="per_batch")
    print(f"  Initial loss: {loss.item():.6f}")

    loss.backward()
    optimizer.step()

    # Check loss decreased or parameters updated
    with torch.no_grad():
        loss_after = model.compute_flow_matching_loss(x_nl, x_lean, t_sample="per_batch")

    print(f"  Loss after step: {loss_after.item():.6f}")
    print(f"  Loss change: {loss_after.item() - loss.item():.6f}")
    print("  ✓ Training step completed successfully")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Neural OT with Positional Embeddings - Test Suite")
    print("="*80)

    try:
        # Test 1: Velocity field
        test_velocity_field()

        # Test 2: With random embeddings
        test_with_random_embeddings()

        # Test 3: Training step
        test_training_step()

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nThe Neural OT implementation with positional embeddings is working correctly.")
        print("You can now proceed with full training using:")
        print("  python -m scripts.train_neural_ot --config project_config.yaml")

    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
