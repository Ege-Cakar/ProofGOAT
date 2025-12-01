"""Generate paired random embeddings for NL/Lean domains with structured distributions.

This creates two distinct Gaussian distributions with different means and covariances,
providing a non-trivial optimal transport map to learn. The NL embeddings are sampled
from one distribution, and Lean embeddings are sampled from another, with optional
structured relationships between them.

Example usage:
    uv run python -m scripts.generate_random_embeddings \
        --num-samples 64 --min-len 32 --max-len 128 \
        --hidden-dim 2048 --output-dir outputs/random_embeddings \
        --distribution-type separate
"""
import argparse
import logging
from pathlib import Path
import numpy as np

from src.io_utils import ensure_dir, save_parquet


def build_random_pair_coupled(rng: np.random.Generator, length: int, hidden_dim: int, noise_scale: float):
    """Generate coupled pair: NL from one distribution, Lean is NL + noise."""
    base = rng.standard_normal((length, hidden_dim), dtype=np.float32)
    lean = base + noise_scale * rng.standard_normal((length, hidden_dim), dtype=np.float32)
    return base, lean


def build_random_pair_separate(
    rng: np.random.Generator,
    length: int,
    hidden_dim: int,
    nl_mean: np.ndarray,
    nl_cov_scale: float,
    lean_mean: np.ndarray,
    lean_cov_scale: float,
    shared_base: bool = False
):
    """Generate pairs from two different Gaussian distributions.

    Args:
        rng: Random number generator
        length: Sequence length
        hidden_dim: Embedding dimension
        nl_mean: Mean vector for NL distribution [hidden_dim]
        nl_cov_scale: Scale factor for NL covariance
        lean_mean: Mean vector for Lean distribution [hidden_dim]
        lean_cov_scale: Scale factor for Lean covariance
        shared_base: If True, use same base noise (creates token correspondence)

    Returns:
        nl_arr, lean_arr: Token embeddings from two distributions
    """
    if shared_base:
        # Use same base noise for both - creates token-level correspondence
        # This means token i in NL has a meaningful relationship to token i in Lean
        base_noise = rng.standard_normal((length, hidden_dim), dtype=np.float32)
        nl_arr = base_noise * nl_cov_scale + nl_mean[np.newaxis, :]
        lean_arr = base_noise * lean_cov_scale + lean_mean[np.newaxis, :]
    else:
        # Independent sampling - no token correspondence (will cause training issues!)
        nl_base = rng.standard_normal((length, hidden_dim), dtype=np.float32)
        nl_arr = nl_base * nl_cov_scale + nl_mean[np.newaxis, :]

        lean_base = rng.standard_normal((length, hidden_dim), dtype=np.float32)
        lean_arr = lean_base * lean_cov_scale + lean_mean[np.newaxis, :]

    return nl_arr, lean_arr


def build_random_pair_affine(
    rng: np.random.Generator,
    length: int,
    hidden_dim: int,
    transform_matrix: np.ndarray,
    translation: np.ndarray,
    noise_scale: float
):
    """Generate pairs with affine transformation: Lean = A * NL + b + noise.

    Args:
        rng: Random number generator
        length: Sequence length
        hidden_dim: Embedding dimension
        transform_matrix: Linear transformation matrix [hidden_dim, hidden_dim]
        translation: Translation vector [hidden_dim]
        noise_scale: Scale of additive noise

    Returns:
        nl_arr, lean_arr: Token embeddings with affine relationship
    """
    # Sample NL from standard normal
    nl_arr = rng.standard_normal((length, hidden_dim), dtype=np.float32)

    # Apply affine transformation: Lean = (A @ NL^T)^T + b + noise
    lean_arr = (nl_arr @ transform_matrix.T) + translation[np.newaxis, :]

    # Add noise
    if noise_scale > 0:
        lean_arr += noise_scale * rng.standard_normal((length, hidden_dim), dtype=np.float32)

    return nl_arr, lean_arr


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # Setup distribution parameters based on type
    if args.distribution_type == "coupled":
        logging.info("Using COUPLED distribution (NL + noise)")
    elif args.distribution_type == "separate":
        mode = "with shared base (token correspondence)" if args.shared_base else "independent (no correspondence)"
        logging.info(f"Using SEPARATE distributions (two Gaussians) {mode}")
        # Create different mean vectors
        nl_mean = np.zeros(args.hidden_dim, dtype=np.float32)
        lean_mean = np.ones(args.hidden_dim, dtype=np.float32) * args.mean_shift

        # Add some structure to the means (not just uniform shift)
        # Use sinusoidal pattern for NL
        for i in range(0, args.hidden_dim, 100):
            nl_mean[i:i+100] += np.sin(np.linspace(0, 2*np.pi, min(100, args.hidden_dim-i)))

        # Use different pattern for Lean
        for i in range(0, args.hidden_dim, 100):
            lean_mean[i:i+100] += np.cos(np.linspace(0, 2*np.pi, min(100, args.hidden_dim-i)))

        logging.info("NL mean norm: %.4f, Lean mean norm: %.4f",
                    np.linalg.norm(nl_mean), np.linalg.norm(lean_mean))
        logging.info("NL cov scale: %.4f, Lean cov scale: %.4f",
                    args.nl_cov_scale, args.lean_cov_scale)
    elif args.distribution_type == "affine":
        logging.info("Using AFFINE transformation (Lean = A*NL + b + noise)")
        # Create a structured transformation matrix (not just random)
        # Use rotation-like structure + scaling
        transform_matrix = np.eye(args.hidden_dim, dtype=np.float32)

        # Add rotation in 2D subspaces
        for i in range(0, args.hidden_dim-1, 2):
            angle = args.rotation_angle * (i / args.hidden_dim)
            c, s = np.cos(angle), np.sin(angle)
            transform_matrix[i:i+2, i:i+2] = np.array([[c, -s], [s, c]], dtype=np.float32)

        # Add scaling
        transform_matrix *= args.affine_scale

        # Translation vector
        translation = rng.normal(0, args.mean_shift, size=args.hidden_dim).astype(np.float32)

        logging.info("Transform matrix norm: %.4f", np.linalg.norm(transform_matrix))
        logging.info("Translation norm: %.4f", np.linalg.norm(translation))

    logging.info(
        "Generating %d samples | len range [%d, %d] | hidden_dim=%d",
        args.num_samples,
        args.min_len,
        args.max_len,
        args.hidden_dim,
    )

    nl_records = []
    lean_records = []
    for idx in range(args.num_samples):
        length = int(rng.integers(args.min_len, args.max_len + 1))

        # Generate pair based on distribution type
        if args.distribution_type == "coupled":
            nl_arr, lean_arr = build_random_pair_coupled(rng, length, args.hidden_dim, args.noise_scale)
        elif args.distribution_type == "separate":
            nl_arr, lean_arr = build_random_pair_separate(
                rng, length, args.hidden_dim,
                nl_mean, args.nl_cov_scale,
                lean_mean, args.lean_cov_scale,
                shared_base=args.shared_base
            )
        elif args.distribution_type == "affine":
            nl_arr, lean_arr = build_random_pair_affine(
                rng, length, args.hidden_dim,
                transform_matrix, translation, args.noise_scale
            )
        else:
            raise ValueError(f"Unknown distribution type: {args.distribution_type}")

        nl_records.append({"id": f"sample_{idx}", "hidden": nl_arr.tolist()})
        lean_records.append({"id": f"sample_{idx}", "hidden": lean_arr.tolist()})
        if (idx + 1) % max(1, args.num_samples // 10) == 0 or idx == args.num_samples - 1:
            logging.info("Generated %d/%d samples (latest length=%d)", idx + 1, args.num_samples, length)

    nl_path = out_dir / args.nl_name
    lean_path = out_dir / args.lean_name
    save_parquet(nl_records, str(nl_path))
    save_parquet(lean_records, str(lean_path))
    logging.info("Wrote NL embeddings to %s", nl_path)
    logging.info("Wrote Lean embeddings to %s", lean_path)

    # Log statistics
    nl_sample = np.array(nl_records[0]["hidden"])
    lean_sample = np.array(lean_records[0]["hidden"])
    logging.info("Sample statistics (first example):")
    logging.info("  NL:   mean=%.4f, std=%.4f", nl_sample.mean(), nl_sample.std())
    logging.info("  Lean: mean=%.4f, std=%.4f", lean_sample.mean(), lean_sample.std())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate paired embeddings with learnable OT map")
    ap.add_argument("--output-dir", default="outputs/random_embeddings")
    ap.add_argument("--nl-name", default="nl_embeddings_random.parquet")
    ap.add_argument("--lean-name", default="lean_embeddings_random.parquet")
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--min-len", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)

    # Distribution type
    ap.add_argument(
        "--distribution-type",
        choices=["coupled", "separate", "affine"],
        default="separate",
        help="Type of distribution: 'coupled' (NL+noise), 'separate' (two Gaussians), 'affine' (linear transform)"
    )

    # Parameters for 'coupled' distribution
    ap.add_argument("--noise-scale", type=float, default=0.05,
                   help="Noise scale for coupled and affine distributions")

    # Parameters for 'separate' distribution
    ap.add_argument("--mean-shift", type=float, default=2.0,
                   help="Mean shift between NL and Lean distributions (separate mode)")
    ap.add_argument("--nl-cov-scale", type=float, default=1.0,
                   help="Covariance scale for NL distribution (separate mode)")
    ap.add_argument("--lean-cov-scale", type=float, default=1.5,
                   help="Covariance scale for Lean distribution (separate mode)")
    ap.add_argument("--shared-base", action="store_true",
                   help="Use shared base noise for token correspondence (separate mode) - RECOMMENDED")

    # Parameters for 'affine' distribution
    ap.add_argument("--affine-scale", type=float, default=0.8,
                   help="Scaling factor for affine transformation matrix")
    ap.add_argument("--rotation-angle", type=float, default=0.3,
                   help="Rotation angle for affine transformation (in radians)")

    args = ap.parse_args()
    if args.min_len > args.max_len:
        raise ValueError("min_len must be <= max_len")
    main(args)
