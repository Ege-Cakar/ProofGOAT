"""Generate paired random embeddings for NL/Lean domains.

This is useful for sanity-checking the Neural OT pipeline without running
expensive extraction jobs. Example usage:

    uv run python -m scripts.generate_random_embeddings \
        --num-samples 64 --min-len 32 --max-len 128 \
        --hidden-dim 2048 --output-dir outputs/random_embeddings
"""
import argparse
import logging
from pathlib import Path
import numpy as np

from src.io_utils import ensure_dir, save_parquet


def build_random_pair(rng: np.random.Generator, length: int, hidden_dim: int, noise_scale: float):
    base = rng.standard_normal((length, hidden_dim), dtype=np.float32)
    lean = base + noise_scale * rng.standard_normal((length, hidden_dim), dtype=np.float32)
    return base, lean


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
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
        nl_arr, lean_arr = build_random_pair(rng, length, args.hidden_dim, args.noise_scale)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/random_embeddings")
    ap.add_argument("--nl-name", default="nl_embeddings_random.parquet")
    ap.add_argument("--lean-name", default="lean_embeddings_random.parquet")
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--min-len", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=2048)
    ap.add_argument("--noise-scale", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.min_len > args.max_len:
        raise ValueError("min_len must be <= max_len")
    main(args)
