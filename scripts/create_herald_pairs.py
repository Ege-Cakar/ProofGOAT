import os
import json
import random
from typing import List, Dict


OUT_PATH = "data/herald_pairs.jsonl"


def is_valid(ex: Dict) -> bool:
    return isinstance(ex.get("informal_proof"), str) and isinstance(ex.get("formal_proof"), str)


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "Missing dependency: datasets. Install with `pip install datasets` or add it to requirements.txt"
        )

    print("=====================================")
    print(" Loading FrenzyMath/Herald_proofs and creating 80/20 split")
    print("=====================================")

    # Load all available splits and combine
    dsd = load_dataset("FrenzyMath/Herald_proofs")

    records: List[Dict] = []
    total = 0
    for split_name in dsd.keys():
        ds = dsd[split_name]
        total += len(ds)
        for ex in ds:
            if not is_valid(ex):
                continue
            records.append({
                # Prefer existing id if present; otherwise create one later
                "_id": ex.get("id"),
                "nl_proof": ex["informal_proof"],
                "lean_proof": ex["formal_proof"],
            })

    if not records:
        raise SystemExit("No valid records with formal_proof/informal_proof found.")

    print(f"Loaded {len(records)} valid examples (out of {total}).")

    # Deterministic shuffle and 80/20 split
    random.seed(42)
    random.shuffle(records)
    k = int(0.8 * len(records))

    for i, ex in enumerate(records):
        raw_id = ex.pop("_id") if ex.get("_id") else f"herald_{i:07d}"
        ex["id"] = str(raw_id)
        ex["split"] = "train" if i < k else "test"

    # Write JSONL
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in records:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("=====================================")
    print(f"✅ Saved {len(records)} examples to {OUT_PATH}")
    print(f"Split: train={k}, test={len(records)-k}")
    print("=====================================")


if __name__ == "__main__":
    main()
