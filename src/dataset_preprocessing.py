"""Reusable helpers for notebook 03 dataset preprocessing.

This module keeps the notebook focused on explaining preprocessing decisions
while shared code handles tokenized-dataset path building, preview-table
creation, tensor export, and summary-file writing.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast

from src.data_utils import load_nonempty_sequences
from src.notebook_utils import require_existing_path, resolve_random_seed


def build_preprocessing_paths(
    project_root: str | Path,
    tokenizer_family: str,
    setting_label: str,
    train_split_filename: str = "train.txt",
    val_split_filename: str = "val.txt",
    test_split_filename: str = "test.txt",
) -> dict[str, Path]:
    """Build the standard input and output paths for notebook 03."""

    project_root = Path(project_root)
    splits_dir = project_root / "data" / "splits"
    tokenizer_dir = project_root / "tokenizers" / tokenizer_family / setting_label
    output_dataset_dir = (
        project_root / "tokenized_datasets" / tokenizer_family / setting_label
    )

    return {
        "splits_dir": splits_dir,
        "tokenizer_dir": tokenizer_dir,
        "output_dataset_dir": output_dataset_dir,
        "train_path": splits_dir / train_split_filename,
        "val_path": splits_dir / val_split_filename,
        "test_path": splits_dir / test_split_filename,
        "train_dataset_path": output_dataset_dir / "train_dataset.pt",
        "val_dataset_path": output_dataset_dir / "val_dataset.pt",
        "test_dataset_path": output_dataset_dir / "test_dataset.pt",
        "tokenization_preview_path": output_dataset_dir / "tokenization_preview.csv",
        "preprocessing_summary_path": output_dataset_dir / "preprocessing_summary.json",
    }


def load_fast_tokenizer(tokenizer_dir: str | Path) -> PreTrainedTokenizerFast:
    """Load a saved Hugging Face fast tokenizer from disk."""

    tokenizer_dir = require_existing_path(tokenizer_dir, "Tokenizer directory")
    return PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))


def load_split_sequences(split_path: str | Path) -> list[str]:
    """Load one split file while discarding blank lines."""

    split_path = require_existing_path(split_path, "Split file")
    sequences = load_nonempty_sequences(split_path)
    if not sequences:
        raise ValueError(f"No non-empty sequences were found in {split_path}.")
    return sequences


def build_tokenization_preview(
    tokenizer: PreTrainedTokenizerFast,
    sequences: list[str],
    sample_size: int = 3,
    random_seed: int = 42,
    max_display_tokens: int = 40,
) -> tuple[pd.DataFrame, Counter]:
    """Create a lightweight tokenization preview table and token counter."""

    if not sequences:
        raise ValueError("At least one sequence is required for the preview.")

    sample_size = max(1, min(sample_size, len(sequences)))
    preview_sequences = (
        random.Random(random_seed).sample(sequences, sample_size)
        if len(sequences) > sample_size
        else list(sequences)
    )

    all_tokens: list[str] = []
    preview_rows: list[dict[str, object]] = []

    for sample_index, sequence in enumerate(preview_sequences, start=1):
        token_ids = tokenizer.encode(sequence, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        all_tokens.extend(tokens)
        preview_rows.append(
            {
                "sample_index": sample_index,
                "sequence": sequence,
                "num_tokens": len(tokens),
                "tokens": " | ".join(tokens[:max_display_tokens]),
            }
        )

    return pd.DataFrame(preview_rows), Counter(all_tokens)


def prepare_tokenizer_preview(
    tokenizer_dir: str | Path,
    train_split_path: str | Path,
    sample_size: int = 3,
    random_seed: int | None = 42,
    max_display_tokens: int = 40,
) -> dict[str, object]:
    """Load the tokenizer and training split, then build the preview outputs."""

    train_sequences = load_split_sequences(train_split_path)
    tokenizer = load_fast_tokenizer(tokenizer_dir)
    resolved_preview_seed = resolve_random_seed(random_seed)
    preview_df, preview_token_counts = build_tokenization_preview(
        tokenizer=tokenizer,
        sequences=train_sequences,
        sample_size=sample_size,
        random_seed=resolved_preview_seed,
        max_display_tokens=max_display_tokens,
    )

    return {
        "train_sequences": train_sequences,
        "tokenizer": tokenizer,
        "preview_df": preview_df,
        "preview_token_counts": preview_token_counts,
        "preview_random_seed": resolved_preview_seed,
    }


def summarize_token_lengths(
    tokenizer: PreTrainedTokenizerFast,
    sequences: list[str],
    rounding_multiple: int = 8,
) -> tuple[dict[str, float | int], pd.DataFrame, np.ndarray]:
    """Summarize token lengths and propose a padded sequence length."""

    if not sequences:
        raise ValueError("At least one sequence is required for length analysis.")
    if rounding_multiple <= 0:
        raise ValueError("rounding_multiple must be a positive integer.")

    token_lengths = np.array(
        [len(tokenizer.encode(sequence, add_special_tokens=False)) for sequence in sequences],
        dtype=int,
    )

    # Add BOS and EOS because the exported tensors include both special tokens.
    total_lengths = token_lengths + 2

    p95_length = float(np.percentile(total_lengths, 95))
    p99_length = float(np.percentile(total_lengths, 99))
    selected_max_length = int(
        np.ceil(p99_length / rounding_multiple) * rounding_multiple
    )

    length_summary: dict[str, float | int] = {
        "max_observed_length": int(total_lengths.max()),
        "p95_length": p95_length,
        "p99_length": p99_length,
        "selected_max_length": selected_max_length,
    }

    length_summary_df = pd.DataFrame(
        {
            "metric": list(length_summary.keys()),
            "value": list(length_summary.values()),
        }
    )

    return length_summary, length_summary_df, total_lengths


def _require_special_token_id(
    token_id: int | None,
    token_name: str,
) -> int:
    """Raise a clear error when a required tokenizer special token is missing."""

    if token_id is None:
        raise ValueError(
            f"The tokenizer is missing a required {token_name} token id. "
            "Confirm that the tokenizer was saved with the standard special tokens."
        )
    return int(token_id)


def build_padded_dataset(
    tokenizer: PreTrainedTokenizerFast,
    sequences: list[str],
    max_seq_len: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Convert sequences into padded input-id and attention-mask tensors."""

    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer.")

    pad_id = _require_special_token_id(tokenizer.pad_token_id, "pad")
    bos_id = _require_special_token_id(tokenizer.bos_token_id, "BOS")
    eos_id = _require_special_token_id(tokenizer.eos_token_id, "EOS")
    unk_id = tokenizer.unk_token_id

    input_ids_matrix: list[list[int]] = []
    attention_mask_matrix: list[list[int]] = []
    truncated_count = 0
    total_unk_tokens = 0
    total_active_tokens = 0
    sequences_with_unk = 0

    for sequence in sequences:
        token_ids = tokenizer.encode(sequence, add_special_tokens=False)
        sequence_ids = [bos_id] + token_ids + [eos_id]

        if len(sequence_ids) > max_seq_len:
            sequence_ids = sequence_ids[:max_seq_len]
            attention_mask = [1] * max_seq_len
            truncated_count += 1
        else:
            pad_length = max_seq_len - len(sequence_ids)
            attention_mask = [1] * len(sequence_ids) + [0] * pad_length
            sequence_ids = sequence_ids + [pad_id] * pad_length

        unk_count = sequence_ids.count(unk_id) if unk_id is not None else 0
        total_unk_tokens += unk_count
        total_active_tokens += sum(attention_mask)
        if unk_count:
            sequences_with_unk += 1

        input_ids_matrix.append(sequence_ids)
        attention_mask_matrix.append(attention_mask)

    dataset = {
        "input_ids": torch.tensor(input_ids_matrix, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_matrix, dtype=torch.long),
    }

    summary = {
        "num_sequences": len(sequences),
        "num_truncated": truncated_count,
        "truncation_rate": truncated_count / len(sequences) if sequences else 0.0,
        "num_sequences_with_unk": sequences_with_unk,
        "sequence_unk_rate": (
            sequences_with_unk / len(sequences) if sequences else 0.0
        ),
        "total_unk_tokens": total_unk_tokens,
        "unk_token_rate": (
            total_unk_tokens / total_active_tokens if total_active_tokens else 0.0
        ),
        "tensor_shape": list(dataset["input_ids"].shape),
    }

    return dataset, summary


def tokenize_split_datasets(
    tokenizer: PreTrainedTokenizerFast,
    preprocessing_paths: dict[str, str | Path],
    max_seq_len: int,
) -> dict[str, object]:
    """Tokenize the train, validation, and test splits for notebook 03."""

    datasets: dict[str, dict[str, torch.Tensor]] = {}
    split_summaries: dict[str, dict[str, object]] = {}

    for split_name in ("train", "val", "test"):
        split_path = preprocessing_paths[f"{split_name}_path"]
        sequences = load_split_sequences(split_path)
        dataset, summary = build_padded_dataset(
            tokenizer=tokenizer,
            sequences=sequences,
            max_seq_len=max_seq_len,
        )
        datasets[split_name] = dataset
        split_summaries[split_name] = summary

    split_summary_df = pd.DataFrame(
        [
            {"split": split_name, **split_summaries[split_name]}
            for split_name in ("train", "val", "test")
        ]
    )

    return {
        "datasets": datasets,
        "split_summaries": split_summaries,
        "split_summary_df": split_summary_df,
    }


def build_preprocessing_summary_payload(
    tokenizer_family: str,
    setting_label: str,
    tokenizer_dir: str | Path,
    selected_max_length: int,
    length_summary: dict[str, float | int],
    split_summaries: dict[str, dict[str, object]],
    saved_files: list[str],
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the summary JSON payload saved with the exported datasets."""

    payload: dict[str, object] = {
        "tokenizer_family": tokenizer_family,
        "setting_label": setting_label,
        "tokenizer_dir": str(tokenizer_dir),
        "selected_max_length": int(selected_max_length),
        "length_summary": length_summary,
        "train_summary": split_summaries["train"],
        "val_summary": split_summaries["val"],
        "test_summary": split_summaries["test"],
        "saved_files": saved_files,
    }

    if extra_fields:
        payload.update(extra_fields)

    return payload


def save_preprocessing_outputs(
    datasets: dict[str, dict[str, torch.Tensor]],
    preview_df: pd.DataFrame,
    summary_payload: dict[str, object],
    output_paths: dict[str, str | Path],
) -> dict[str, Path]:
    """Save notebook 03 tensor, preview, and summary artifacts."""

    train_dataset_path = Path(output_paths["train_dataset_path"])
    val_dataset_path = Path(output_paths["val_dataset_path"])
    test_dataset_path = Path(output_paths["test_dataset_path"])
    tokenization_preview_path = Path(output_paths["tokenization_preview_path"])
    preprocessing_summary_path = Path(output_paths["preprocessing_summary_path"])

    output_dir = train_dataset_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(datasets["train"], train_dataset_path)
    torch.save(datasets["val"], val_dataset_path)
    torch.save(datasets["test"], test_dataset_path)
    preview_df.to_csv(tokenization_preview_path, index=False)
    preprocessing_summary_path.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    return {
        "train_dataset_path": train_dataset_path,
        "val_dataset_path": val_dataset_path,
        "test_dataset_path": test_dataset_path,
        "tokenization_preview_path": tokenization_preview_path,
        "preprocessing_summary_path": preprocessing_summary_path,
    }


def save_preprocessing_run(
    tokenizer_family: str,
    setting_label: str,
    tokenizer_dir: str | Path,
    selected_max_length: int,
    length_summary: dict[str, float | int],
    split_summaries: dict[str, dict[str, object]],
    datasets: dict[str, dict[str, torch.Tensor]],
    preview_df: pd.DataFrame,
    output_paths: dict[str, str | Path],
    extra_summary_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the summary payload and save all notebook 03 outputs."""

    saved_files = [
        "train_dataset.pt",
        "val_dataset.pt",
        "test_dataset.pt",
        "tokenization_preview.csv",
        "preprocessing_summary.json",
    ]
    summary_payload = build_preprocessing_summary_payload(
        tokenizer_family=tokenizer_family,
        setting_label=setting_label,
        tokenizer_dir=tokenizer_dir,
        selected_max_length=selected_max_length,
        length_summary=length_summary,
        split_summaries=split_summaries,
        saved_files=saved_files,
        extra_fields=extra_summary_fields,
    )
    saved_paths = save_preprocessing_outputs(
        datasets=datasets,
        preview_df=preview_df,
        summary_payload=summary_payload,
        output_paths=output_paths,
    )

    return {
        "summary_payload": summary_payload,
        "saved_paths": saved_paths,
    }
