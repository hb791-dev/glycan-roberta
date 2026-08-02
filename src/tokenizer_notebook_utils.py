"""Notebook-facing helpers shared by the tokenizer-generation notebooks.

These helpers keep the tokenizer notebooks focused on the distinct tokenizer
strategy being built, while centralizing repeated workflow tasks such as path
construction, summary saving, and simple inspection-table generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_tokenizer_paths(
    project_root: str | Path,
    tokenizer_family: str,
    setting_label: str,
    train_split_filename: str = "train.txt",
) -> dict[str, Path]:
    """Build the standard input and output paths for one tokenizer notebook."""

    project_root = Path(project_root)
    tokenizer_output_dir = project_root / "tokenizers" / tokenizer_family / setting_label

    return {
        "train_data_path": project_root / "data" / "splits" / train_split_filename,
        "tokenizer_output_dir": tokenizer_output_dir,
        "config_summary_path": tokenizer_output_dir / "tokenizer_config_summary.json",
        "inspection_preview_path": tokenizer_output_dir / "inspection_preview.csv",
        "vocab_path": tokenizer_output_dir / "vocab.json",
    }


def build_tokenizer_output_paths(
    tokenizer_output_dir: str | Path,
    include_inspection_preview: bool = True,
    include_merges_file: bool = False,
) -> dict[str, Path]:
    """Return the output paths a tokenizer notebook may save.

    This dictionary is mainly used with the shared overwrite validator so the
    notebook can consistently check whether outputs already exist.
    """

    tokenizer_output_dir = Path(tokenizer_output_dir)
    output_paths = {
        "tokenizer_json_path": tokenizer_output_dir / "tokenizer.json",
        "tokenizer_config_path": tokenizer_output_dir / "tokenizer_config.json",
        "special_tokens_map_path": tokenizer_output_dir / "special_tokens_map.json",
        "config_summary_path": tokenizer_output_dir / "tokenizer_config_summary.json",
        "vocab_path": tokenizer_output_dir / "vocab.json",
    }

    if include_merges_file:
        output_paths["merges_path"] = tokenizer_output_dir / "merges.txt"

    if include_inspection_preview:
        output_paths["inspection_preview_path"] = tokenizer_output_dir / "inspection_preview.csv"

    return output_paths


def save_tokenizer_summary(
    summary_payload: dict[str, object],
    summary_path: str | Path,
) -> Path:
    """Write the tokenizer summary JSON for one tokenizer run."""

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return summary_path


def build_tokenizer_inspection_preview(
    tokenizer,
    sequences: list[str],
    num_samples: int = 3,
    max_display_tokens: int = 30,
) -> pd.DataFrame:
    """Create the lightweight inspection table shown in tokenizer notebooks."""

    sample_sequences = sequences[:num_samples]
    inspection_rows = []

    for sample_index, sequence in enumerate(sample_sequences, start=1):
        token_ids = tokenizer.encode(sequence, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        inspection_rows.append(
            {
                "sample_index": sample_index,
                "sequence": sequence,
                "num_tokens": len(tokens),
                "tokens": " | ".join(tokens[:max_display_tokens]),
            }
        )

    return pd.DataFrame(inspection_rows)


def save_inspection_preview(
    inspection_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save the lightweight inspection preview CSV for one tokenizer run."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inspection_df.to_csv(output_path, index=False)
    return output_path


def load_training_sequences_for_tokenizer(train_data_path: str | Path) -> list[str]:
    """Load the training split used by tokenizer-generation notebooks."""

    train_data_path = Path(train_data_path)
    with train_data_path.open("r", encoding="utf-8") as file:
        sequences = [line.strip() for line in file if line.strip()]

    if not sequences:
        raise ValueError(f"No training sequences found in {train_data_path}")

    return sequences
