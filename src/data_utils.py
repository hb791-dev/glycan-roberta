"""Utilities for preparing glycan text datasets."""

from __future__ import annotations

import os

from sklearn.model_selection import train_test_split


def split_and_save_data(raw_file_path: str, output_dir: str) -> None:
    """Split a raw glycan text file into train, validation, and test sets.

    The raw file is expected to contain one glycan sequence per line. Blank
    lines are removed before splitting. The split uses a fixed random seed so
    repeated runs produce the same partitions from the same input file.
    """
    print(f"Reading raw data from: {raw_file_path}")

    with open(raw_file_path, "r", encoding="utf-8") as file:
        all_glycan_sequences = file.readlines()

    cleaned_sequences = [sequence.strip() for sequence in all_glycan_sequences if sequence.strip()]

    print(f"Total sequences loaded: {len(cleaned_sequences)}")

    train_sequences, held_out_sequences = train_test_split(
        cleaned_sequences,
        test_size=0.20,
        random_state=42,
    )

    val_sequences, test_sequences = train_test_split(
        held_out_sequences,
        test_size=0.50,
        random_state=42,
    )

    print("Split successful")
    print(f" - Train: {len(train_sequences)} sequences")
    print(f" - Validation: {len(val_sequences)} sequences")
    print(f" - Test: {len(test_sequences)} sequences")

    os.makedirs(output_dir, exist_ok=True)

    split_paths = {
        "train": os.path.join(output_dir, "train.txt"),
        "val": os.path.join(output_dir, "val.txt"),
        "test": os.path.join(output_dir, "test.txt"),
    }

    for split_name, sequences in (
        ("train", train_sequences),
        ("val", val_sequences),
        ("test", test_sequences),
    ):
        with open(split_paths[split_name], "w", encoding="utf-8") as output_file:
            for sequence in sequences:
                output_file.write(sequence + "\n")

    print(f"Saved dataset splits to: {output_dir}")
