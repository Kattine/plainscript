"""Quick sanity checks for processed train/val JSONL files."""

import json
import os


PROCESSED_DIR = os.path.join("data", "processed")
EXPECTED_ROLES = ["system", "user", "assistant"]


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file and return parsed records."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_num} of {path}: {exc.msg}",
                    exc.doc,
                    exc.pos,
                )
    return records


def validate_record(record: dict) -> list[str]:
    """Check one record and return any structure problems."""
    problems = []

    if "messages" not in record:
        problems.append("missing 'messages' key")
        return problems

    roles = [m.get("role") for m in record["messages"]]
    if roles != EXPECTED_ROLES:
        problems.append(f"unexpected role order: {roles}")

    for msg in record["messages"]:
        if not msg.get("content", "").strip():
            problems.append(f"empty content for role '{msg.get('role')}'")

    return problems


def summarize_lengths(records: list[dict]) -> None:
    """Print rough length stats for user and assistant messages."""
    user_lens = []
    assistant_lens = []

    for record in records:
        for msg in record.get("messages", []):
            word_count = len(msg.get("content", "").split())
            if msg.get("role") == "user":
                user_lens.append(word_count)
            elif msg.get("role") == "assistant":
                assistant_lens.append(word_count)

    def stats(values: list[int]) -> str:
        if not values:
            return "no data"
        values_sorted = sorted(values)
        n = len(values_sorted)
        median = values_sorted[n // 2]
        return (
            f"min={min(values)}, median={median}, "
            f"max={max(values)}, mean={sum(values) // n}"
        )

    print(f"  User (technical) word counts:     {stats(user_lens)}")
    print(f"  Assistant (plain) word counts:    {stats(assistant_lens)}")


def validate_file(path: str) -> None:
    """Validate one JSONL file and print a short report."""
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        return

    records = load_jsonl(path)
    print(f"\n{path}")
    print(f"  Records: {len(records)}")

    all_problems = []
    for i, record in enumerate(records):
        problems = validate_record(record)
        if problems:
            all_problems.append((i, problems))

    if all_problems:
        print(f"  PROBLEMS found in {len(all_problems)} records:")
        for idx, problems in all_problems[:5]:
            print(f"    record {idx}: {', '.join(problems)}")
        if len(all_problems) > 5:
            print(f"    ... and {len(all_problems) - 5} more")
    else:
        print("  Structure: all records valid")

    summarize_lengths(records)


def main():
    """Run validation on train and validation splits."""
    print("=== Validating processed data ===")
    validate_file(os.path.join(PROCESSED_DIR, "train.jsonl"))
    validate_file(os.path.join(PROCESSED_DIR, "val.jsonl"))

    # Print one full sample for a quick manual check.
    train_path = os.path.join(PROCESSED_DIR, "train.jsonl")
    if os.path.exists(train_path):
        records = load_jsonl(train_path)
        if records:
            print("\n=== First training example (full) ===")
            print(json.dumps(records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
