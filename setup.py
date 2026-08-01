"""Run the whole pipeline: fetch data, build features, and train."""

from scripts.make_dataset import fetch_all_data
from scripts.build_features import build_training_data
from scripts.model import train_model


def main():
    """Execute fetch -> preprocess -> train."""
    print("=== Step 1: Fetching medical abstract / lay summary pairs ===")
    fetch_all_data()

    print("\n=== Step 2: Building instruction-tuning dataset ===")
    build_training_data()

    print("\n=== Step 3: Fine-tuning model with QLoRA ===")
    train_model()

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
