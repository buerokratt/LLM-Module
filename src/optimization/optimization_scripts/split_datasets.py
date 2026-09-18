"""
Data splitting script for DSPy optimization datasets.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
import sys
from src.loki_logger import LokiLogger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Initialize Loki logger
logger = LokiLogger(service_name="split-datasets")


def load_dataset(filepath: Path) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data: List[Dict[str, Any]], filepath: Path) -> None:
    """Save dataset to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(data)} examples to {filepath}")


def split_dataset(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.2,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset following DSPy's recommendation: 20% train, 80% validation.

    Why this unusual split?
    - Prompt optimizers can overfit easily to small training sets
    - Need large validation set for stable evaluation
    - DSPy optimizers work better with more validation data

    Args:
        data: Full dataset
        train_ratio: Ratio for training set (default 0.2 for DSPy)
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data)
    """
    if shuffle:
        random.seed(seed)
        data = data.copy()
        random.shuffle(data)

    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    return train_data, val_data


def split_guardrails_dataset(
    input_path: Path, output_dir: Path, train_ratio: float = 0.2
) -> None:
    """
    Split guardrails dataset and ensure balanced safe/unsafe distribution.

    This is critical for security - we need balanced representation in both
    train and validation sets to properly evaluate safety performance.
    """
    logger.info(f" Splitting guardrails dataset from {input_path}")

    data = load_dataset(input_path)

    # Separate safe and unsafe examples for stratified split
    unsafe_examples = [ex for ex in data if ex["should_block"]]
    safe_examples = [ex for ex in data if not ex["should_block"]]

    logger.info(
        f"Total: {len(data)} | Unsafe: {len(unsafe_examples)} | Safe: {len(safe_examples)}"
    )

    # Split each category independently to maintain balance
    unsafe_train, unsafe_val = split_dataset(unsafe_examples, train_ratio)
    safe_train, safe_val = split_dataset(safe_examples, train_ratio)

    # Combine and shuffle
    train_data = unsafe_train + safe_train
    val_data = unsafe_val + safe_val

    random.seed(42)
    random.shuffle(train_data)
    random.shuffle(val_data)

    # Save splits
    save_dataset(train_data, output_dir / "train" / "guardrails_train.json")
    save_dataset(val_data, output_dir / "val" / "guardrails_val.json")

    logger.info("Guardrails split complete:")
    logger.info(
        f"   Train: {len(train_data)} examples "
        f"(Unsafe: {sum(1 for ex in train_data if ex['should_block'])}, "
        f"Safe: {sum(1 for ex in train_data if not ex['should_block'])})"
    )
    logger.info(
        f"   Val: {len(val_data)} examples "
        f"(Unsafe: {sum(1 for ex in val_data if ex['should_block'])}, "
        f"Safe: {sum(1 for ex in val_data if not ex['should_block'])})"
    )


def split_refiner_dataset(
    input_path: Path, output_dir: Path, train_ratio: float = 0.2
) -> None:
    """Split prompt refiner dataset."""
    logger.info(f"Splitting refiner dataset from {input_path}")

    data = load_dataset(input_path)
    train_data, val_data = split_dataset(data, train_ratio)

    save_dataset(train_data, output_dir / "train" / "refiner_train.json")
    save_dataset(val_data, output_dir / "val" / "refiner_val.json")

    logger.info(
        f"Refiner split complete: Train={len(train_data)} | Val={len(val_data)}"
    )


def split_generator_dataset(
    input_path: Path, output_dir: Path, train_ratio: float = 0.2
) -> None:
    """
    Split response generator dataset and ensure in-scope/out-of-scope balance.

    Critical for properly training the model to detect when it should/shouldn't
    answer based on available context.
    """
    logger.info(f"Splitting generator dataset from {input_path}")

    data = load_dataset(input_path)

    # Separate in-scope and out-of-scope for stratified split
    in_scope = [ex for ex in data if ex["should_be_in_scope"]]
    out_of_scope = [ex for ex in data if not ex["should_be_in_scope"]]

    logger.info(
        f"Total: {len(data)} | In-scope: {len(in_scope)} | Out-of-scope: {len(out_of_scope)}"
    )

    # Split each category
    in_scope_train, in_scope_val = split_dataset(in_scope, train_ratio)
    out_scope_train, out_scope_val = split_dataset(out_of_scope, train_ratio)

    # Combine and shuffle
    train_data = in_scope_train + out_scope_train
    val_data = in_scope_val + out_scope_val

    random.seed(42)
    random.shuffle(train_data)
    random.shuffle(val_data)

    # Save splits
    save_dataset(train_data, output_dir / "train" / "generator_train.json")
    save_dataset(val_data, output_dir / "val" / "generator_val.json")

    logger.info("Generator split complete:")
    logger.info(
        f"   Train: {len(train_data)} examples "
        f"(In-scope: {sum(1 for ex in train_data if ex['should_be_in_scope'])}, "
        f"Out-of-scope: {sum(1 for ex in train_data if not ex['should_be_in_scope'])})"
    )
    logger.info(
        f"   Val: {len(val_data)} examples "
        f"(In-scope: {sum(1 for ex in val_data if ex['should_be_in_scope'])}, "
        f"Out-of-scope: {sum(1 for ex in val_data if not ex['should_be_in_scope'])})"
    )


def main() -> None:
    """Main execution function."""
    logger.info("Starting DSPy dataset splitting process")

    # Define paths relative to script location
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent / "optimization_data"

    try:
        # Split guardrails dataset
        split_guardrails_dataset(
            input_path=base_dir / "guardrails" / "guardrails_dataset.json",
            output_dir=base_dir / "guardrails",
        )

        # Split refiner dataset
        split_refiner_dataset(
            input_path=base_dir / "refiner" / "refiner_dataset.json",
            output_dir=base_dir / "refiner",
        )

        # Split generator dataset
        split_generator_dataset(
            input_path=base_dir / "generator" / "generator_dataset.json",
            output_dir=base_dir / "generator",
        )

        logger.info("All datasets split successfully!")
        logger.info("Check the train/ and val/ subdirectories for split files")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please ensure all dataset JSON files are created first")
        raise
    except Exception as e:
        logger.error(f"Error during dataset splitting: {e}")
        raise


if __name__ == "__main__":
    main()
