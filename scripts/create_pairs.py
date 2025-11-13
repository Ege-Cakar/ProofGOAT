import os
import json
import requests
import pandas as pd

RAW_DIR = "data/raw"
OUT_PATH = "data/pairs.jsonl"

# These URLs are correct for the ProofNet Parquet data.
PARQUET_URLS = {
    "validation": "https://huggingface.co/api/datasets/hoskinson-center/proofnet/parquet/plain_text/validation/0.parquet",
    "test":       "https://huggingface.co/api/datasets/hoskinson-center/proofnet/parquet/plain_text/test/0.parquet",
}


def download_parquet(url, local_path):
    """Download a parquet file from HuggingFace's dataset API."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print(f"Downloading {url}")
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download {url}: HTTP {r.status_code}")

    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return local_path


def load_parquet(local_path):
    df = pd.read_parquet(local_path)
    print("Columns:", list(df.columns))
    return df


def is_valid(row):
    # Drop rows with missing fields
    if not isinstance(row["nl_statement"], str):
        return False
    if not isinstance(row["formal_statement"], str):
        return False
    return True


def main():
    print("=====================================")
    print(" Creating NL → Lean pairs from ProofNet")
    print("=====================================")

    pairs = []
    dropped = 0

    for split, url in PARQUET_URLS.items():
        local_file = f"{RAW_DIR}/{split}.parquet"

        if not os.path.exists(local_file):
            local_file = download_parquet(url, local_file)

        df = load_parquet(local_file)

        print(f"{split}: {len(df)} rows")

        for _, row in df.iterrows():
            if is_valid(row):
                pairs.append({
                    "id": row["id"],
                    "nl_text": row["nl_statement"],
                    "nl_proof": row["nl_proof"],
                    "lean_text": row["formal_statement"],
                    "src_header": row["src_header"],
                    "split": split
                })
            else:
                dropped += 1

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(OUT_PATH, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print("=====================================")
    print(f"✅ Saved {len(pairs)} examples to {OUT_PATH}")
    print(f"🚫 Dropped {dropped} empty/missing examples")
    print("=====================================")


if __name__ == "__main__":
    main()
