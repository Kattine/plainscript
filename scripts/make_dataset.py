"""Download medical simplification pairs and save them under data/raw.

Sources:
- lighteval/med_paragraph_simplification (Cochrane, CC-BY)
- pszemraj/scientific_lay_summarisation-plos-norm (optional)
Portions of this file were developed with assistance from Claude (Anthropic). https://claude.ai
"""

import argparse
import json
import os

from huggingface_hub import hf_hub_download


RAW_DIR = os.path.join("data", "raw")

COCHRANE_REPO = "lighteval/med_paragraph_simplification"
COCHRANE_FILES = [
    "data/train-00000-of-00001.parquet",
    "data/validation-00000-of-00001.parquet",
    "data/test-00000-of-00001.parquet",
]

PLOS_REPO = "pszemraj/scientific_lay_summarisation-plos-norm"
PLOS_FILES = ["train.parquet"]

# Candidate names for source (technical) and target (plain-language) text.
INPUT_COL_CANDIDATES = ["source", "abstract", "article", "input", "complex", "query"]
OUTPUT_COL_CANDIDATES = ["target", "plain_summary", "summary", "output", "simple", "answer"]


def _pick_columns(columns: list[str]) -> tuple[str, str]:
    """Pick likely source/target text columns from a parquet schema."""
    cols_lower = {c.lower(): c for c in columns}

    input_col = next(
        (cols_lower[c] for c in INPUT_COL_CANDIDATES if c in cols_lower), None
    )
    output_col = next(
        (cols_lower[c] for c in OUTPUT_COL_CANDIDATES if c in cols_lower), None
    )

    if input_col is None or output_col is None:
        raise RuntimeError(
            f"Could not identify text columns. Actual columns: {columns}. "
            f"Update INPUT_COL_CANDIDATES / OUTPUT_COL_CANDIDATES to match."
        )

    return input_col, output_col


def _load_parquet_pairs(
    repo_id: str,
    filenames: list[str],
    max_samples: int,
    truncate_input: int | None = None,
) -> list[dict]:
    """Load pairs from one or more parquet files in a HF dataset repo."""
    import pandas as pd

    pairs = []
    input_col = output_col = None

    for filename in filenames:
        print(f"Downloading {repo_id}/{filename} ...")
        local_path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset"
        )
        df = pd.read_parquet(local_path)

        if input_col is None:
            input_col, output_col = _pick_columns(list(df.columns))
            print(f"  Using columns: input='{input_col}', output='{output_col}'")

        for _, row in df.iterrows():
            source = str(row.get(input_col, "")).strip()
            target = str(row.get(output_col, "")).strip()

            if truncate_input:
                source = source[:truncate_input]

            if source and target and len(source) > 50 and len(target) > 50:
                pairs.append({"abstract": source, "plain_summary": target})
            if len(pairs) >= max_samples:
                break
        if len(pairs) >= max_samples:
            break

    return pairs


def fetch_cochrane_pairs(max_samples: int = 4500) -> list[dict]:
    """Download Cochrane pairs and write them to cochrane_pairs.json."""
    pairs = _load_parquet_pairs(COCHRANE_REPO, COCHRANE_FILES, max_samples)

    os.makedirs(RAW_DIR, exist_ok=True)
    output_path = os.path.join(RAW_DIR, "cochrane_pairs.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(pairs)} Cochrane pairs to {output_path}")
    if pairs:
        print("\n--- Sample pair (verify this looks right) ---")
        print(f"TECHNICAL: {pairs[0]['abstract'][:200]}...")
        print(f"PLAIN:     {pairs[0]['plain_summary'][:200]}...")
        print("---------------------------------------------")

    return pairs


def fetch_plos_pairs(max_samples: int = 2000) -> list[dict]:
    """Download optional PLOS pairs and write them to plos_pairs.json."""
    # PLOS can contain full articles, so trim to a short abstract-like chunk.
    pairs = _load_parquet_pairs(
        PLOS_REPO, PLOS_FILES, max_samples, truncate_input=1500
    )

    os.makedirs(RAW_DIR, exist_ok=True)
    output_path = os.path.join(RAW_DIR, "plos_pairs.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(pairs)} PLOS pairs to {output_path}")
    return pairs


def fetch_all_data(include_plos: bool = False):
    """Run the data download steps (Cochrane + optional PLOS)."""
    os.makedirs(RAW_DIR, exist_ok=True)

    print("=== Fetching Cochrane pairs ===")
    cochrane = fetch_cochrane_pairs()

    plos = []
    if include_plos:
        print("\n=== Fetching PLOS pairs ===")
        plos = fetch_plos_pairs()

    total = len(cochrane) + len(plos)
    print(f"\nTotal: Cochrane={len(cochrane)}, PLOS={len(plos)}, Combined={total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch medical text simplification data."
    )
    parser.add_argument(
        "--include-plos",
        action="store_true",
        help="Also download PLOS data. Default: Cochrane only.",
    )
    args = parser.parse_args()
    fetch_all_data(include_plos=args.include_plos)
