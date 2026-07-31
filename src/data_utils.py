"""Utilities for preparing glycan text datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.notebook_utils import resolve_random_seed


def load_nonempty_sequences(raw_file_path: str | Path) -> list[str]:
    """Load the non-empty glycan sequences from a plain-text dataset file."""

    raw_file_path = Path(raw_file_path)
    with raw_file_path.open("r", encoding="utf-8") as file:
        all_glycan_sequences = file.readlines()

    return [sequence.strip() for sequence in all_glycan_sequences if sequence.strip()]


def split_sequences(
    sequences: list[str],
    held_out_fraction: float = 0.20,
    random_seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split sequences into train, validation, and test subsets.

    The raw file is expected to contain one glycan sequence per line. Blank
    lines should already have been removed before calling this helper.
    ``held_out_fraction`` controls the combined validation-plus-test size.
    The held-out set is then split evenly into validation and test subsets.
    """

    train_sequences, held_out_sequences = train_test_split(
        sequences,
        test_size=held_out_fraction,
        random_state=random_seed,
    )

    val_sequences, test_sequences = train_test_split(
        held_out_sequences,
        test_size=0.50,
        random_state=random_seed,
    )

    return train_sequences, val_sequences, test_sequences


def build_split_summary(
    train_sequences: list[str],
    val_sequences: list[str],
    test_sequences: list[str],
) -> pd.DataFrame:
    """Create a compact split summary table for notebook display and saving."""

    return pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "num_sequences": [
                len(train_sequences),
                len(val_sequences),
                len(test_sequences),
            ],
        }
    )


def save_split_files(
    output_dir: str | Path,
    train_sequences: list[str],
    val_sequences: list[str],
    test_sequences: list[str],
) -> dict[str, Path]:
    """Write the split text files and return their paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train_path": output_dir / "train.txt",
        "val_path": output_dir / "val.txt",
        "test_path": output_dir / "test.txt",
    }

    for split_name, sequences in (
        ("train_path", train_sequences),
        ("val_path", val_sequences),
        ("test_path", test_sequences),
    ):
        with split_paths[split_name].open("w", encoding="utf-8") as output_file:
            for sequence in sequences:
                output_file.write(sequence + "\n")

    return split_paths


def run_data_split_pipeline(
    raw_file_path: str | Path,
    output_dir: str | Path,
    held_out_fraction: float = 0.20,
    random_seed: int | None = 42,
) -> dict[str, object]:
    """Run the full split workflow used by notebook 01.

    This helper loads the raw sequences, resolves the active seed, performs the
    train/validation/test split, writes the split files, and returns the main
    notebook-friendly outputs in one dictionary.
    """

    raw_file_path = Path(raw_file_path)
    output_dir = Path(output_dir)

    print(f"Reading raw data from: {raw_file_path}")
    sequences = load_nonempty_sequences(raw_file_path)
    print(f"Total sequences loaded: {len(sequences)}")

    resolved_seed = resolve_random_seed(random_seed)
    print(f"Random seed: {resolved_seed}")

    train_sequences, val_sequences, test_sequences = split_sequences(
        sequences=sequences,
        held_out_fraction=held_out_fraction,
        random_seed=resolved_seed,
    )

    print("Split successful")
    print(f" - Train: {len(train_sequences)} sequences")
    print(f" - Validation: {len(val_sequences)} sequences")
    print(f" - Test: {len(test_sequences)} sequences")

    split_paths = save_split_files(
        output_dir=output_dir,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
    )
    summary_df = build_split_summary(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
    )

    summary_path = output_dir / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved dataset splits to: {output_dir}")
    print(f"Saved split summary to: {summary_path}")

    return {
        "random_seed": resolved_seed,
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "test_sequences": test_sequences,
        "split_summary_df": summary_df,
        "split_summary_path": summary_path,
        "split_paths": split_paths,
    }


def split_and_save_data(
    raw_file_path: str | Path,
    output_dir: str | Path,
    held_out_fraction: float = 0.20,
    random_seed: int | None = 42,
) -> dict[str, object]:
    """Backward-compatible wrapper for the notebook-01 split pipeline."""

    return run_data_split_pipeline(
        raw_file_path=raw_file_path,
        output_dir=output_dir,
        held_out_fraction=held_out_fraction,
        random_seed=random_seed,
    )
