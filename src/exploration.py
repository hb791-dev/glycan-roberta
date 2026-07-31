"""Reusable helpers for the raw-data exploration notebook.

The notebook should explain *why* the dataset is being inspected, while this
module handles the repeatable mechanics such as loading text, building summary
tables, plotting the length distribution, and saving small reference outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_glycan_sequences(raw_data_path: str | Path) -> list[str]:
    """Load non-empty glycan sequences from a plain-text file."""

    raw_data_path = Path(raw_data_path)
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_data_path}")

    with raw_data_path.open("r", encoding="utf-8") as file:
        sequences = [line.strip() for line in file if line.strip()]

    if not sequences:
        raise ValueError(
            f"No non-empty glycan sequences were found in {raw_data_path}."
        )

    return sequences


def build_sequence_length_summary(
    sequences: list[str],
    preview_count: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build the main tables used in notebook 00.

    Returns
    -------
    dataset_summary_df:
        One-row-per-metric summary of dataset size and character lengths.
    preview_df:
        A small table that shows example sequences and their character lengths.
    sequence_lengths:
        A NumPy array of character lengths used for plotting.
    """

    if not sequences:
        raise ValueError("At least one sequence is required to build a summary.")

    sequence_lengths = np.array([len(sequence) for sequence in sequences], dtype=int)

    dataset_summary_df = pd.DataFrame(
        {
            "metric": [
                "num_sequences",
                "min_char_length",
                "mean_char_length",
                "median_char_length",
                "max_char_length",
                "p95_char_length",
                "p99_char_length",
            ],
            "value": [
                len(sequences),
                int(sequence_lengths.min()),
                float(sequence_lengths.mean()),
                float(np.median(sequence_lengths)),
                int(sequence_lengths.max()),
                float(np.percentile(sequence_lengths, 95)),
                float(np.percentile(sequence_lengths, 99)),
            ],
        }
    )

    preview_count = max(1, min(preview_count, len(sequences)))
    preview_sequences = sequences[:preview_count]
    preview_df = pd.DataFrame(
        {
            "example_index": list(range(preview_count)),
            "sequence": preview_sequences,
            "char_length": [len(sequence) for sequence in preview_sequences],
        }
    )

    return dataset_summary_df, preview_df, sequence_lengths


def plot_sequence_length_distribution(
    sequence_lengths: np.ndarray,
    output_path: str | Path,
    title: str = "Distribution of Raw Glycan Sequence Lengths",
    bins: int = 40,
) -> Path:
    """Create and save the sequence-length histogram used in notebook 00."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The histogram provides a quick visual check for long-tail behavior that
    # could affect later padding and truncation decisions.
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=bins, color="#4C78A8", edgecolor="white")
    plt.title(title)
    plt.xlabel("Sequence length (characters)")
    plt.ylabel("Number of sequences")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return output_path


def save_exploration_outputs(
    output_dir: str | Path,
    dataset_summary_df: pd.DataFrame,
    preview_df: pd.DataFrame,
) -> dict[str, Path]:
    """Save the summary tables created by notebook 00."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "dataset_summary.csv"
    preview_path = output_dir / "example_sequences.csv"

    dataset_summary_df.to_csv(summary_path, index=False)
    preview_df.to_csv(preview_path, index=False)

    return {
        "dataset_summary_path": summary_path,
        "example_sequences_path": preview_path,
    }
