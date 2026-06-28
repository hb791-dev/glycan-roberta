"""Helpers for maintaining the run index for GlycanProject2."""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd


RUN_INDEX_COLUMNS = [
    "experiment_name",
    "tokenizer_family",
    "setting_label",
    "run_mode",
    "parent_experiment_name",
    "mlm_probability",
    "num_hidden_layers",
    "attention_heads",
    "hidden_size",
    "intermediate_size",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "epochs",
    "early_stopping_patience",
    "tokenizer_dir",
    "tokenized_dataset_dir",
    "checkpoint_dir",
    "results_dir",
    "validation_summary_path",
    "test_metrics_path",
    "qualitative_probe_path",
    "notebook_used",
    "git_commit",
    "run_status",
    "notes",
]

DEFAULT_KEY_FIELDS = ("experiment_name", "tokenizer_family")


def _normalize_value(value):
    """Convert missing values to empty strings for CSV storage."""
    if value is None:
        return ""
    return value


def _ensure_all_columns(index_df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing expected columns without dropping existing data."""
    for column in RUN_INDEX_COLUMNS:
        if column not in index_df.columns:
            index_df[column] = ""

    return index_df[RUN_INDEX_COLUMNS]


def ensure_run_index(index_path: str) -> pd.DataFrame:
    """Create an empty run index if one does not already exist."""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    if not os.path.exists(index_path) or os.path.getsize(index_path) == 0:
        index_df = pd.DataFrame(columns=RUN_INDEX_COLUMNS)
        index_df.to_csv(index_path, index=False)
        return index_df

    index_df = pd.read_csv(index_path, dtype=str).fillna("")
    index_df = _ensure_all_columns(index_df)
    index_df.to_csv(index_path, index=False)
    return index_df


def load_run_index(index_path: str) -> pd.DataFrame:
    """Load the run index and ensure the expected schema is present."""
    index_df = ensure_run_index(index_path)
    return _ensure_all_columns(index_df)


def upsert_run_record(
    index_path: str,
    record: dict,
    key_fields: Iterable[str] = DEFAULT_KEY_FIELDS,
) -> pd.DataFrame:
    """Insert or update one run record in the run index.

    A row is matched using the provided key fields. If a matching row exists, it
    is updated in place. Otherwise, a new row is appended.
    """
    key_fields = tuple(key_fields)

    for field in key_fields:
        if field not in record or record[field] in (None, ""):
            raise ValueError(f"Missing required key field: {field}")

    index_df = load_run_index(index_path)
    normalized_record = {column: _normalize_value(record.get(column, "")) for column in RUN_INDEX_COLUMNS}

    if index_df.empty:
        updated_df = pd.DataFrame([normalized_record], columns=RUN_INDEX_COLUMNS)
    else:
        match_mask = pd.Series(True, index=index_df.index)
        for field in key_fields:
            match_mask &= index_df[field].astype(str) == str(normalized_record[field])

        if match_mask.any():
            match_index = index_df.index[match_mask][0]
            for column, value in normalized_record.items():
                index_df.at[match_index, column] = value
            updated_df = index_df
        else:
            new_row_df = pd.DataFrame([normalized_record], columns=RUN_INDEX_COLUMNS)
            updated_df = pd.concat([index_df, new_row_df], ignore_index=True)

    updated_df = _ensure_all_columns(updated_df)
    updated_df.to_csv(index_path, index=False)
    return updated_df
