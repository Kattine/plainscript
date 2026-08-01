import json
import os
import random


RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

SYSTEM_PROMPT = (
    "You are a medical text simplifier. Rewrite the following medical text "
    "into plain language that a patient with no medical background can "
    "understand. Preserve all key findings and conclusions. Do not add "
    "information not present in the original text."
)


def load_raw_pairs() -> list[dict]:
    all_pairs = []

    for filename in ["cochrane_pairs.json", "plos_pairs.json"]:
        filepath = os.path.join(RAW_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                pairs = json.load(f)
            print(f"Loaded {len(pairs)} pairs from {filename}")
            all_pairs.extend(pairs)
        else:
            print(f"Warning: {filepath} not found, skipping.")

    return all_pairs


def format_instruction(abstract: str, plain_summary: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": abstract},
            {"role": "assistant", "content": plain_summary},
        ]
    }


def build_training_data(val_ratio: float = 0.1, seed: int = 42):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    pairs = load_raw_pairs()
    if not pairs:
        print("Error: No data found. Run make_dataset.py first.")
        return

    formatted = [
        format_instruction(p["abstract"], p["plain_summary"])
        for p in pairs
    ]

    random.seed(seed)
    random.shuffle(formatted)

    val_size = int(len(formatted) * val_ratio)
    val_data = formatted[:val_size]
    train_data = formatted[val_size:]

    train_path = os.path.join(PROCESSED_DIR, "train.jsonl")
    val_path = os.path.join(PROCESSED_DIR, "val.jsonl")

    for data, path in [(train_data, train_path), (val_data, val_path)]:
        with open(path, "w", encoding="utf-8") as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved {len(train_data)} training examples to {train_path}")
    print(f"Saved {len(val_data)} validation examples to {val_path}")


if __name__ == "__main__":
    build_training_data()
