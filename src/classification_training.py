"""Helpers for multi-label glycan classification fine-tuning.

This module is the notebook-10 counterpart to ``src/classification_prep.py``.
Its job is to keep the repetitive training setup out of the notebook so the
notebook can stay focused on:

- choosing one pretrained checkpoint
- choosing clear Drive output folders
- training one sequence classifier at a time
- reviewing validation behavior before touching the test set

The project is still comparing multiple tokenizer families, so these helpers
also make the output-folder naming explicit. That way classifier runs derived
from different pretrained checkpoints do not get mixed together in Drive.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

if TYPE_CHECKING:
    from collections.abc import Mapping


ACCESSION_COLUMN = "glycan_id"
SEQUENCE_COLUMN = "sequence"
SPLIT_COLUMN = "split"
LABEL_JSON_COLUMN = "labels_json"
LABEL_LIST_COLUMN = "labels"
LABEL_VECTOR_COLUMN = "label_vector"


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------

def _require_columns(dataframe: "pd.DataFrame", required_columns: list[str], frame_name: str) -> None:
    """Raise a readable error when a dataframe is missing expected columns."""
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{frame_name} is missing required columns: {missing_columns}")


def _load_json_label_list(value) -> list[str]:
    """Return one parsed label list from a saved ``labels_json`` cell."""
    if isinstance(value, list):
        return [str(label).strip() for label in value if str(label).strip()]

    if value is None:
        return []

    text_value = str(value).strip()
    if text_value == "":
        return []

    parsed_value = json.loads(text_value)
    if not isinstance(parsed_value, list):
        raise ValueError(f"Expected a JSON list of labels, got: {parsed_value!r}")

    return [str(label).strip() for label in parsed_value if str(label).strip()]


def _serialize_json(value: object) -> str:
    """Return a deterministic ASCII JSON string for CSV export."""
    return json.dumps(value, ensure_ascii=True, sort_keys=isinstance(value, dict))


# ---------------------------------------------------------------------------
# Loading prepared notebook-09 artifacts
# ---------------------------------------------------------------------------

def parse_label_json_column(classification_df: "pd.DataFrame") -> "pd.DataFrame":
    """Convert the saved ``labels_json`` strings back into Python label lists.

    Notebook 09 saves label lists as JSON strings so the CSV files are easy to
    move around. Notebook 10 needs the real label lists back so they can be
    converted into multi-hot targets.
    """
    _require_columns(classification_df, [LABEL_JSON_COLUMN], "classification_df")

    parsed_df = classification_df.copy()
    parsed_df[LABEL_LIST_COLUMN] = parsed_df[LABEL_JSON_COLUMN].map(_load_json_label_list)
    return parsed_df


def load_classification_tables(
    train_csv_path: str | Path,
    val_csv_path: str | Path,
    test_csv_path: str | Path,
    label_vocabulary_path: str | Path,
) -> dict[str, "pd.DataFrame"]:
    """Load the prepared classification split tables and label vocabulary."""
    train_df = parse_label_json_column(pd.read_csv(train_csv_path))
    val_df = parse_label_json_column(pd.read_csv(val_csv_path))
    test_df = parse_label_json_column(pd.read_csv(test_csv_path))
    label_vocabulary_df = pd.read_csv(label_vocabulary_path)

    for frame_name, dataframe in (
        ("train_df", train_df),
        ("val_df", val_df),
        ("test_df", test_df),
    ):
        _require_columns(
            dataframe,
            [ACCESSION_COLUMN, SEQUENCE_COLUMN, LABEL_JSON_COLUMN, LABEL_LIST_COLUMN],
            frame_name,
        )

    _require_columns(label_vocabulary_df, ["label_id", "label_name"], "label_vocabulary_df")

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "label_vocabulary_df": label_vocabulary_df,
    }


# ---------------------------------------------------------------------------
# Label encoding helpers
# ---------------------------------------------------------------------------

def build_label_name_to_id(label_vocabulary_df: "pd.DataFrame") -> dict[str, int]:
    """Create a stable mapping from subtype label name to integer label ID."""
    _require_columns(label_vocabulary_df, ["label_id", "label_name"], "label_vocabulary_df")

    label_name_to_id: dict[str, int] = {}
    for row in label_vocabulary_df.itertuples(index=False):
        label_name = str(row.label_name).strip()
        label_id = int(row.label_id)
        if label_name in label_name_to_id:
            raise ValueError(f"Duplicate label name in label vocabulary: {label_name!r}")
        label_name_to_id[label_name] = label_id

    return label_name_to_id


def validate_label_coverage(
    classification_df: "pd.DataFrame",
    label_name_to_id: "Mapping[str, int]",
) -> None:
    """Check that every observed label is present in the saved vocabulary."""
    _require_columns(classification_df, [LABEL_LIST_COLUMN], "classification_df")

    observed_labels = {
        str(label_name)
        for label_values in classification_df[LABEL_LIST_COLUMN].tolist()
        for label_name in label_values
    }
    missing_labels = sorted(observed_labels - set(label_name_to_id))
    if missing_labels:
        raise ValueError(
            "Found labels in the prepared classification table that are missing "
            f"from label_vocabulary.csv: {missing_labels}"
        )


def encode_multilabel_targets(
    classification_df: "pd.DataFrame",
    label_name_to_id: "Mapping[str, int]",
) -> "pd.DataFrame":
    """Convert each glycan's label list into a multi-hot float target vector."""
    _require_columns(classification_df, [LABEL_LIST_COLUMN], "classification_df")
    validate_label_coverage(classification_df, label_name_to_id)

    num_labels = len(label_name_to_id)
    encoded_df = classification_df.copy()
    encoded_targets: list[list[float]] = []

    for label_values in encoded_df[LABEL_LIST_COLUMN].tolist():
        target_vector = np.zeros(num_labels, dtype=np.float32)
        for label_name in label_values:
            target_vector[int(label_name_to_id[str(label_name)])] = 1.0
        encoded_targets.append(target_vector.tolist())

    encoded_df[LABEL_VECTOR_COLUMN] = encoded_targets
    return encoded_df


# ---------------------------------------------------------------------------
# Tokenization and dataset wrappers
# ---------------------------------------------------------------------------

def load_classification_tokenizer(model_dir: str | Path):
    """Load the tokenizer that matches the saved pretrained checkpoint."""
    return AutoTokenizer.from_pretrained(str(model_dir))


def tokenize_classification_dataframe(
    classification_df: "pd.DataFrame",
    tokenizer,
    max_length: int,
) -> dict[str, object]:
    """Tokenize glycan sequences and keep multi-label targets aligned.

    The classification notebook works at the sequence level, so every row needs:
    - tokenized input IDs
    - tokenized attention masks
    - one multi-hot float label vector
    """
    _require_columns(
        classification_df,
        [ACCESSION_COLUMN, SEQUENCE_COLUMN, LABEL_VECTOR_COLUMN],
        "classification_df",
    )

    encoded_batch = tokenizer(
        classification_df[SEQUENCE_COLUMN].tolist(),
        padding="max_length",
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )

    label_tensor = torch.tensor(
        classification_df[LABEL_VECTOR_COLUMN].tolist(),
        dtype=torch.float32,
    )

    return {
        "input_ids": encoded_batch["input_ids"],
        "attention_mask": encoded_batch["attention_mask"],
        "labels": label_tensor,
        # Keep a clean metadata copy so later notebook cells can line predictions
        # back up with accessions and sequences without rebuilding that table.
        "metadata_df": classification_df.reset_index(drop=True).copy(),
    }


class GlycanClassificationDataset(Dataset):
    """Dataset wrapper for tokenized multi-label glycan classification rows."""

    def __init__(self, tokenized_bundle: dict[str, object]):
        self.input_ids = tokenized_bundle["input_ids"]
        self.attention_mask = tokenized_bundle["attention_mask"]
        self.labels = tokenized_bundle["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def build_classification_datasets(
    train_df: "pd.DataFrame",
    val_df: "pd.DataFrame",
    test_df: "pd.DataFrame",
    tokenizer,
    label_name_to_id: "Mapping[str, int]",
    max_length: int,
) -> dict[str, object]:
    """Build tokenized train/validation/test datasets for classification."""
    encoded_train_df = encode_multilabel_targets(train_df, label_name_to_id)
    encoded_val_df = encode_multilabel_targets(val_df, label_name_to_id)
    encoded_test_df = encode_multilabel_targets(test_df, label_name_to_id)

    train_bundle = tokenize_classification_dataframe(encoded_train_df, tokenizer, max_length=max_length)
    val_bundle = tokenize_classification_dataframe(encoded_val_df, tokenizer, max_length=max_length)
    test_bundle = tokenize_classification_dataframe(encoded_test_df, tokenizer, max_length=max_length)

    return {
        "train_df": encoded_train_df,
        "val_df": encoded_val_df,
        "test_df": encoded_test_df,
        "train_bundle": train_bundle,
        "val_bundle": val_bundle,
        "test_bundle": test_bundle,
        "train_dataset": GlycanClassificationDataset(train_bundle),
        "val_dataset": GlycanClassificationDataset(val_bundle),
        "test_dataset": GlycanClassificationDataset(test_bundle),
    }


# ---------------------------------------------------------------------------
# Model and output-path helpers
# ---------------------------------------------------------------------------

def derive_classification_run_names(model_dir: str | Path) -> dict[str, str]:
    """Derive tokenizer and experiment names from one saved pretrained model path.

    The project is still comparing tokenizer families, so saved classifier
    outputs should be nested under the tokenizer family and experiment that
    produced the pretrained checkpoint being fine-tuned.
    """
    model_path = Path(model_dir)
    if len(model_path.parts) < 3:
        raise ValueError(f"Model path is too short to derive run names: {model_path}")

    return {
        "model_dir_name": model_path.name,
        "experiment_name": model_path.parent.name,
        "tokenizer_family": model_path.parent.parent.name,
    }


def build_classification_output_paths(
    project_root: str | Path,
    tokenizer_family: str,
    experiment_name: str,
    classifier_run_label: str = "default_run",
) -> dict[str, str]:
    """Build clear Drive output paths for one classifier fine-tuning run."""
    project_root = Path(project_root)
    results_dir = (
        project_root
        / "results"
        / "classification_finetuning"
        / tokenizer_family
        / experiment_name
        / classifier_run_label
    )
    checkpoint_dir = (
        project_root
        / "checkpoints"
        / "classification"
        / tokenizer_family
        / experiment_name
        / classifier_run_label
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return {
        "results_dir": str(results_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "best_model_dir": str(checkpoint_dir / "best_model"),
        "training_config_path": str(results_dir / "training_config.json"),
        "trainer_state_copy_path": str(results_dir / "trainer_state.json"),
        "loss_history_path": str(results_dir / "loss_history.csv"),
        "loss_curve_path": str(results_dir / "loss_curves.png"),
        "validation_metrics_path": str(results_dir / "validation_metrics.csv"),
        "validation_threshold_scan_path": str(results_dir / "validation_threshold_scan.csv"),
        "validation_prediction_table_path": str(results_dir / "validation_prediction_table.csv"),
        "best_threshold_path": str(results_dir / "best_threshold.json"),
        "label_vocabulary_snapshot_path": str(results_dir / "label_vocabulary_snapshot.csv"),
    }


def save_json(payload: dict[str, object], output_path: str | Path) -> None:
    """Write one small JSON file to disk with pretty indentation."""
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_sequence_classification_model(
    pretrained_model_dir: str | Path,
    num_labels: int,
    initialization_mode: str = "mlm_checkpoint",
    device: str | None = None,
):
    """Load a multi-label sequence classifier with a configurable initialization.

    ``mlm_checkpoint`` reuses the encoder weights from the saved MLM checkpoint
    and initializes a fresh classification head with the requested number of
    output labels.

    ``random_init`` keeps the same architecture and tokenizer vocabulary but
    starts the entire classifier from random weights using the saved config as
    a template.
    """
    initialization_mode = str(initialization_mode).strip().lower()

    if initialization_mode == "mlm_checkpoint":
        model = AutoModelForSequenceClassification.from_pretrained(
            str(pretrained_model_dir),
            num_labels=int(num_labels),
            problem_type="multi_label_classification",
        )
    elif initialization_mode == "random_init":
        config = AutoConfig.from_pretrained(str(pretrained_model_dir))
        config.num_labels = int(num_labels)
        config.problem_type = "multi_label_classification"
        model = AutoModelForSequenceClassification.from_config(config)
    else:
        raise ValueError(
            "initialization_mode must be either 'mlm_checkpoint' or 'random_init'."
        )

    if device is not None:
        model = model.to(torch.device(device))

    return model


# ---------------------------------------------------------------------------
# Multi-label prediction helpers
# ---------------------------------------------------------------------------

def sigmoid_predictions_from_logits(logits) -> np.ndarray:
    """Convert raw classifier logits into per-label probabilities."""
    logits_array = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-logits_array))


def binarize_multilabel_predictions(probabilities, threshold: float) -> np.ndarray:
    """Turn per-label probabilities into binary predictions using one threshold."""
    probability_array = np.asarray(probabilities, dtype=float)
    return (probability_array >= float(threshold)).astype(int)


def summarize_prediction_density(binary_predictions) -> dict[str, float]:
    """Report how many labels the model predicts per glycan on average."""
    prediction_array = np.asarray(binary_predictions, dtype=int)
    labels_per_example = prediction_array.sum(axis=1)

    return {
        "mean_predicted_labels_per_glycan": float(labels_per_example.mean()),
        "median_predicted_labels_per_glycan": float(np.median(labels_per_example)),
        "max_predicted_labels_for_one_glycan": int(labels_per_example.max()) if len(labels_per_example) else 0,
    }


def _safe_average_precision_summary(true_labels, predicted_probabilities) -> dict[str, float]:
    """Compute macro and weighted average precision robustly.

    Some labels can be very sparse. Rather than relying on one global sklearn
    call that can be noisy when a label has no positives, this helper computes
    one label at a time and then aggregates only across labels with support.
    """
    from sklearn.metrics import average_precision_score

    true_array = np.asarray(true_labels, dtype=int)
    probability_array = np.asarray(predicted_probabilities, dtype=float)
    label_support = true_array.sum(axis=0)

    per_label_ap: list[float] = []
    per_label_support: list[int] = []

    for label_index in range(true_array.shape[1]):
        support = int(label_support[label_index])
        if support == 0:
            continue

        average_precision = average_precision_score(
            true_array[:, label_index],
            probability_array[:, label_index],
        )
        per_label_ap.append(float(average_precision))
        per_label_support.append(support)

    if not per_label_ap:
        return {
            "macro_average_precision": float("nan"),
            "weighted_average_precision": float("nan"),
        }

    return {
        "macro_average_precision": float(np.mean(per_label_ap)),
        "weighted_average_precision": float(np.average(per_label_ap, weights=per_label_support)),
    }


def compute_multilabel_metrics(
    true_labels,
    predicted_labels,
    predicted_probabilities,
) -> dict[str, float]:
    """Compute compact multi-label metrics for validation or test review."""
    true_array = np.asarray(true_labels, dtype=int)
    predicted_array = np.asarray(predicted_labels, dtype=int)
    probability_array = np.asarray(predicted_probabilities, dtype=float)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_array,
        predicted_array,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_array,
        predicted_array,
        average="weighted",
        zero_division=0,
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        true_array,
        predicted_array,
        average="micro",
        zero_division=0,
    )

    density_summary = summarize_prediction_density(predicted_array)
    ap_summary = _safe_average_precision_summary(true_array, probability_array)

    return {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "exact_match_accuracy": float(accuracy_score(true_array, predicted_array)),
        "mean_true_labels_per_glycan": float(true_array.sum(axis=1).mean()),
        **density_summary,
        **ap_summary,
    }


def build_hf_compute_metrics(threshold: float = 0.5):
    """Return a Hugging Face Trainer-compatible ``compute_metrics`` callback."""

    def _compute_metrics(eval_prediction) -> dict[str, float]:
        predictions = eval_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        probabilities = sigmoid_predictions_from_logits(predictions)
        binary_predictions = binarize_multilabel_predictions(probabilities, threshold=threshold)
        return compute_multilabel_metrics(
            true_labels=eval_prediction.label_ids,
            predicted_labels=binary_predictions,
            predicted_probabilities=probabilities,
        )

    return _compute_metrics


def scan_global_thresholds(
    probabilities,
    true_labels,
    thresholds: Sequence[float],
) -> "pd.DataFrame":
    """Compare a small set of global thresholds on one validation output table."""
    probability_array = np.asarray(probabilities, dtype=float)
    true_array = np.asarray(true_labels, dtype=int)

    threshold_rows = []
    for threshold in thresholds:
        binary_predictions = binarize_multilabel_predictions(probability_array, threshold=threshold)
        metric_row = compute_multilabel_metrics(
            true_labels=true_array,
            predicted_labels=binary_predictions,
            predicted_probabilities=probability_array,
        )
        metric_row["threshold"] = float(threshold)
        threshold_rows.append(metric_row)

    threshold_df = pd.DataFrame(threshold_rows)
    if not threshold_df.empty:
        threshold_df = threshold_df.sort_values(
            ["macro_f1", "weighted_f1", "threshold"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    return threshold_df


def save_threshold_scan(threshold_results_df: "pd.DataFrame", output_path: str | Path) -> None:
    """Save the validation threshold comparison table for later review."""
    threshold_results_df.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Prediction-table helpers
# ---------------------------------------------------------------------------

def _label_id_to_name_lookup(label_vocabulary_df: "pd.DataFrame") -> dict[int, str]:
    """Return the inverse label lookup for readable saved prediction tables."""
    _require_columns(label_vocabulary_df, ["label_id", "label_name"], "label_vocabulary_df")
    return {
        int(row.label_id): str(row.label_name)
        for row in label_vocabulary_df.itertuples(index=False)
    }


def build_classification_prediction_table(
    source_df: "pd.DataFrame",
    probabilities,
    predicted_labels,
    label_vocabulary_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Create a readable table of true labels, predicted labels, and probabilities.

    The full 41-label probability vector is useful for later deeper analysis,
    but it is awkward to read in a notebook table. This export keeps a smaller
    human-facing summary by storing:

    - the true label set
    - the predicted positive label set
    - the probabilities attached to the predicted positive labels
    - a top-5 label-probability preview for quick inspection
    """
    _require_columns(
        source_df,
        [ACCESSION_COLUMN, SEQUENCE_COLUMN, LABEL_LIST_COLUMN],
        "source_df",
    )

    probability_array = np.asarray(probabilities, dtype=float)
    predicted_array = np.asarray(predicted_labels, dtype=int)
    label_id_to_name = _label_id_to_name_lookup(label_vocabulary_df)

    prediction_rows: list[dict[str, object]] = []

    for row_index, row in enumerate(source_df.itertuples(index=False)):
        row_probabilities = probability_array[row_index]
        row_predictions = predicted_array[row_index]
        predicted_label_names = [
            label_id_to_name[label_index]
            for label_index, is_predicted in enumerate(row_predictions.tolist())
            if int(is_predicted) == 1
        ]
        predicted_probability_map = {
            label_id_to_name[label_index]: float(row_probabilities[label_index])
            for label_index, is_predicted in enumerate(row_predictions.tolist())
            if int(is_predicted) == 1
        }

        top_label_indices = np.argsort(row_probabilities)[::-1][:5]
        top_label_probability_map = {
            label_id_to_name[int(label_index)]: float(row_probabilities[int(label_index)])
            for label_index in top_label_indices
        }

        prediction_rows.append(
            {
                ACCESSION_COLUMN: getattr(row, ACCESSION_COLUMN),
                SEQUENCE_COLUMN: getattr(row, SEQUENCE_COLUMN),
                SPLIT_COLUMN: getattr(row, SPLIT_COLUMN) if hasattr(row, SPLIT_COLUMN) else "",
                "num_true_labels": len(getattr(row, LABEL_LIST_COLUMN)),
                "true_labels_json": _serialize_json(getattr(row, LABEL_LIST_COLUMN)),
                "num_predicted_labels": len(predicted_label_names),
                "predicted_labels_json": _serialize_json(predicted_label_names),
                "predicted_label_probabilities_json": _serialize_json(predicted_probability_map),
                "top5_label_probabilities_json": _serialize_json(top_label_probability_map),
            }
        )

    return pd.DataFrame(prediction_rows)
