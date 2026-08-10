"""Helpers for multi-label glycan classification fine-tuning.

This module supports notebook 10 by moving reusable run validation,
tokenization, output-path handling, and validation-review logic out of the
notebook body. The notebook can stay focused on:

- choosing one classifier run mode and one starting checkpoint
- loading the prepared notebook-09 classification tables
- training one multi-label classifier at a time
- reviewing validation behavior before touching the test set
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
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from src.notebook_utils import (
    require_existing_path,
    resolve_random_seed,
    validate_output_paths,
    validate_tokenizer_family,
)
from src.training_diagnostics import (
    load_trainer_history,
    merge_loss_history,
    recommend_continuation,
    split_train_eval_history,
    summarize_best_epoch,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


NOTEBOOK_PATH = "notebooks/10_classification_finetuning.ipynb"
VALID_CLASSIFIER_RUN_MODES = {
    "fresh_mlm_checkpoint",
    "fresh_random_init",
    "resume_checkpoint",
    "continue_best_model",
}

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


def _save_json(payload: dict[str, object], output_path: str | Path) -> Path:
    """Write one formatted JSON file and return its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _normalize_classifier_run_mode(run_mode: str) -> str:
    """Return one validated classifier run mode."""
    normalized_run_mode = str(run_mode).strip().lower()
    if normalized_run_mode not in VALID_CLASSIFIER_RUN_MODES:
        supported_modes = ", ".join(sorted(VALID_CLASSIFIER_RUN_MODES))
        raise ValueError(f"RUN_MODE must be one of: {supported_modes}")
    return normalized_run_mode


def _resolve_classifier_schedule(
    run_mode: str,
    initial_num_train_epochs: int,
    continuation_num_train_epochs: int,
    base_learning_rate: float,
    continuation_learning_rate: float,
) -> dict[str, int | float]:
    """Convert run-mode settings into effective epochs and learning rate."""
    if run_mode in {"fresh_mlm_checkpoint", "fresh_random_init"}:
        return {
            "num_train_epochs": int(initial_num_train_epochs),
            "learning_rate": float(base_learning_rate),
        }

    if run_mode == "resume_checkpoint":
        return {
            "num_train_epochs": int(initial_num_train_epochs) + int(continuation_num_train_epochs),
            "learning_rate": float(base_learning_rate),
        }

    if run_mode == "continue_best_model":
        return {
            "num_train_epochs": int(continuation_num_train_epochs),
            "learning_rate": float(continuation_learning_rate),
        }

    raise ValueError(f"Unsupported RUN_MODE: {run_mode}")


def _require_nonempty_run_stem(classifier_run_stem: str) -> str:
    """Return a non-empty classifier run stem."""
    cleaned_value = str(classifier_run_stem).strip()
    if not cleaned_value:
        raise ValueError("CLASSIFIER_RUN_STEM must not be empty.")
    return cleaned_value


def _validate_classifier_resume_source(run_mode: str, resume_source_dir: str | Path) -> Path:
    """Confirm that one classifier continuation source has the expected folder type."""
    resume_source_path = require_existing_path(resume_source_dir, "Resume source directory")

    if run_mode == "resume_checkpoint" and "checkpoint-" not in resume_source_path.name:
        raise ValueError("resume_checkpoint mode must point to a checkpoint-* directory.")

    if run_mode == "continue_best_model" and resume_source_path.name != "best_model":
        raise ValueError("continue_best_model mode must point to a best_model directory.")

    return resume_source_path


def _extract_classifier_run_coordinates(
    project_root: str | Path,
    resume_source_path: str | Path,
) -> dict[str, str]:
    """Derive tokenizer family, pretraining experiment, and classifier label from one saved classifier path."""
    project_root = Path(project_root).resolve()
    resume_source_path = Path(resume_source_path).resolve()
    checkpoint_root = (project_root / "checkpoints" / "classification").resolve()

    try:
        relative_parts = resume_source_path.relative_to(checkpoint_root).parts
    except ValueError as exc:
        raise ValueError(
            "RESUME_SOURCE_DIR must live under the classifier checkpoint root:\n"
            f"{checkpoint_root}"
        ) from exc

    if len(relative_parts) < 4:
        raise ValueError(
            "RESUME_SOURCE_DIR must follow the standard classifier folder layout:\n"
            "checkpoints/classification/<tokenizer_family>/<experiment_name>/"
            "<classifier_run_label>/<checkpoint-or-best_model>"
        )

    tokenizer_family = validate_tokenizer_family(relative_parts[0])
    experiment_name = str(relative_parts[1]).strip()
    classifier_run_label = str(relative_parts[2]).strip()

    if not experiment_name or not classifier_run_label:
        raise ValueError("Could not derive experiment_name or classifier_run_label from RESUME_SOURCE_DIR.")

    return {
        "tokenizer_family": tokenizer_family,
        "experiment_name": experiment_name,
        "classifier_run_label": classifier_run_label,
    }


def _build_classifier_run_label(run_mode: str, classifier_run_stem: str) -> str:
    """Build one readable classifier run label from the selected run mode."""
    if run_mode == "fresh_mlm_checkpoint":
        return f"{classifier_run_stem}_mlm"
    if run_mode == "fresh_random_init":
        return f"{classifier_run_stem}_randominit"
    if run_mode == "continue_best_model":
        return f"{classifier_run_stem}_cont"
    raise ValueError(f"Unsupported run mode for new classifier label construction: {run_mode}")


def _resolve_resume_tokenizer_source(
    resume_source_path: str | Path,
    output_paths: dict[str, Path],
) -> Path:
    """Choose the tokenizer source directory for a resumed classifier run.

    A saved checkpoint directory should contain model weights, but the most
    reliable tokenizer source is either the original model_source_dir recorded
    in ``training_config.json`` or, if that is missing, the checkpoint folder
    itself.
    """
    resume_source_path = Path(resume_source_path)
    training_config_path = output_paths["training_config_path"]

    if training_config_path.exists():
        try:
            training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            training_config = {}
        original_model_source_dir = training_config.get("model_source_dir", "")
        if str(original_model_source_dir).strip():
            return require_existing_path(original_model_source_dir, "Original model source directory")

    return resume_source_path


# ---------------------------------------------------------------------------
# Notebook-10 path helpers
# ---------------------------------------------------------------------------

def build_classification_prep_input_paths(
    project_root: str | Path,
    classification_prep_dirname: str = "classification_prep",
) -> dict[str, Path]:
    """Build the prepared notebook-09 input paths used by notebook 10."""
    project_root = Path(project_root)
    prep_dir = project_root / "results" / str(classification_prep_dirname).strip()
    return {
        "classification_prep_dir": prep_dir,
        "train_classification_path": prep_dir / "train_classification.csv",
        "val_classification_path": prep_dir / "val_classification.csv",
        "test_classification_path": prep_dir / "test_classification.csv",
        "label_vocabulary_path": prep_dir / "label_vocabulary.csv",
    }


def build_classification_output_paths(
    project_root: str | Path,
    tokenizer_family: str,
    experiment_name: str,
    classifier_run_label: str,
) -> dict[str, Path]:
    """Build the standard checkpoint and results paths for one classifier run."""
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
        "results_dir": results_dir,
        "checkpoint_dir": checkpoint_dir,
        "best_model_dir": checkpoint_dir / "best_model",
        "training_config_path": results_dir / "training_config.json",
        "trainer_state_copy_path": results_dir / "trainer_state.json",
        "loss_history_path": results_dir / "loss_history.csv",
        "loss_curve_path": results_dir / "loss_curves.png",
        "validation_metrics_path": results_dir / "validation_metrics.csv",
        "validation_threshold_scan_path": results_dir / "validation_threshold_scan.csv",
        "validation_prediction_table_path": results_dir / "validation_prediction_table.csv",
        "best_threshold_path": results_dir / "best_threshold.json",
        "label_vocabulary_snapshot_path": results_dir / "label_vocabulary_snapshot.csv",
        "validation_review_summary_path": results_dir / "validation_review_summary.json",
    }


def build_classification_output_validation_paths(output_paths: dict[str, Path]) -> dict[str, Path]:
    """Return the notebook-10 files treated as overwrite-checked outputs."""
    return {
        "training_config_path": output_paths["training_config_path"],
        "trainer_state_copy_path": output_paths["trainer_state_copy_path"],
        "loss_history_path": output_paths["loss_history_path"],
        "loss_curve_path": output_paths["loss_curve_path"],
        "validation_metrics_path": output_paths["validation_metrics_path"],
        "validation_threshold_scan_path": output_paths["validation_threshold_scan_path"],
        "validation_prediction_table_path": output_paths["validation_prediction_table_path"],
        "best_threshold_path": output_paths["best_threshold_path"],
        "label_vocabulary_snapshot_path": output_paths["label_vocabulary_snapshot_path"],
        "validation_review_summary_path": output_paths["validation_review_summary_path"],
    }


def prepare_classification_run(
    project_root: str | Path,
    classification_prep_dirname: str,
    run_mode: str,
    pretrained_model_dir: str | Path | None,
    classifier_run_stem: str,
    resume_source_dir: str | Path | None,
    overwrite_existing_outputs: bool,
    initial_num_train_epochs: int,
    continuation_num_train_epochs: int,
    base_learning_rate: float,
    continuation_learning_rate: float,
    random_seed: int | None,
    notebook_used: str = NOTEBOOK_PATH,
) -> dict[str, object]:
    """Validate notebook-10 settings, resolve paths, and build a reusable run context."""
    project_root = require_existing_path(project_root, "Project root")
    normalized_run_mode = _normalize_classifier_run_mode(run_mode)
    classifier_run_stem = _require_nonempty_run_stem(classifier_run_stem)
    prep_input_paths = build_classification_prep_input_paths(
        project_root=project_root,
        classification_prep_dirname=classification_prep_dirname,
    )

    for description, path in (
        ("Prepared train classification table", prep_input_paths["train_classification_path"]),
        ("Prepared validation classification table", prep_input_paths["val_classification_path"]),
        ("Prepared test classification table", prep_input_paths["test_classification_path"]),
        ("Prepared label vocabulary", prep_input_paths["label_vocabulary_path"]),
    ):
        require_existing_path(path, description)

    schedule = _resolve_classifier_schedule(
        run_mode=normalized_run_mode,
        initial_num_train_epochs=initial_num_train_epochs,
        continuation_num_train_epochs=continuation_num_train_epochs,
        base_learning_rate=base_learning_rate,
        continuation_learning_rate=continuation_learning_rate,
    )

    resume_from_checkpoint = None
    model_source_dir = None
    tokenizer_source_dir = None
    model_load_mode = None
    source_summary_path = None

    if normalized_run_mode in {"fresh_mlm_checkpoint", "fresh_random_init"}:
        model_source_dir = require_existing_path(pretrained_model_dir, "PRETRAINED_MODEL_DIR")
        run_names = derive_classification_run_names(model_source_dir)
        tokenizer_family = validate_tokenizer_family(run_names["tokenizer_family"])
        experiment_name = str(run_names["experiment_name"]).strip()
        classifier_run_label = _build_classifier_run_label(normalized_run_mode, classifier_run_stem)
        model_load_mode = (
            "mlm_checkpoint"
            if normalized_run_mode == "fresh_mlm_checkpoint"
            else "random_init"
        )
        tokenizer_source_dir = model_source_dir
        source_summary_path = model_source_dir
    else:
        normalized_resume_source = _validate_classifier_resume_source(
            run_mode=normalized_run_mode,
            resume_source_dir=resume_source_dir,
        )
        coordinates = _extract_classifier_run_coordinates(
            project_root=project_root,
            resume_source_path=normalized_resume_source,
        )
        tokenizer_family = coordinates["tokenizer_family"]
        experiment_name = coordinates["experiment_name"]

        if normalized_run_mode == "resume_checkpoint":
            classifier_run_label = coordinates["classifier_run_label"]
            resume_from_checkpoint = normalized_resume_source
            model_source_dir = normalized_resume_source
        else:
            classifier_run_label = _build_classifier_run_label(normalized_run_mode, classifier_run_stem)
            model_source_dir = normalized_resume_source
            tokenizer_source_dir = normalized_resume_source

        model_load_mode = "classifier_checkpoint"
        source_summary_path = normalized_resume_source

    output_paths = build_classification_output_paths(
        project_root=project_root,
        tokenizer_family=tokenizer_family,
        experiment_name=experiment_name,
        classifier_run_label=classifier_run_label,
    )

    if normalized_run_mode != "resume_checkpoint":
        validate_output_paths(
            build_classification_output_validation_paths(output_paths),
            overwrite_existing_outputs=overwrite_existing_outputs,
        )

    if normalized_run_mode == "resume_checkpoint":
        tokenizer_source_dir = _resolve_resume_tokenizer_source(
            resume_source_path=normalized_resume_source,
            output_paths=output_paths,
        )

    resolved_random_seed = resolve_random_seed(random_seed)
    training_config = {
        "project_root": str(project_root),
        "classification_prep_dirname": str(classification_prep_dirname).strip(),
        "run_mode": normalized_run_mode,
        "pretrained_model_dir": str(pretrained_model_dir) if pretrained_model_dir else None,
        "resume_source_dir": str(resume_source_dir) if resume_source_dir else None,
        "model_source_dir": str(model_source_dir) if model_source_dir else None,
        "tokenizer_source_dir": str(tokenizer_source_dir) if tokenizer_source_dir else None,
        "model_load_mode": model_load_mode,
        "classifier_run_stem": classifier_run_stem,
        "classifier_run_label": classifier_run_label,
        "overwrite_existing_outputs": bool(overwrite_existing_outputs),
        "tokenizer_family": tokenizer_family,
        "pretrain_experiment_name": experiment_name,
        "initial_num_train_epochs": int(initial_num_train_epochs),
        "continuation_num_train_epochs": int(continuation_num_train_epochs),
        "num_train_epochs": int(schedule["num_train_epochs"]),
        "base_learning_rate": float(base_learning_rate),
        "continuation_learning_rate": float(continuation_learning_rate),
        "learning_rate": float(schedule["learning_rate"]),
        "seed": int(resolved_random_seed),
        "results_dir": str(output_paths["results_dir"]),
        "checkpoint_dir": str(output_paths["checkpoint_dir"]),
        "best_model_dir": str(output_paths["best_model_dir"]),
        "notebook_used": notebook_used,
    }

    return {
        "settings": training_config,
        "prep_input_paths": prep_input_paths,
        "output_paths": output_paths,
        "resume_from_checkpoint": resume_from_checkpoint,
        "model_source_dir": model_source_dir,
        "tokenizer_source_dir": tokenizer_source_dir,
        "model_load_mode": model_load_mode,
        "source_summary_path": source_summary_path,
    }


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
    """Load the tokenizer that matches one saved classifier or MLM checkpoint."""
    return AutoTokenizer.from_pretrained(str(model_dir))


def tokenize_classification_dataframe(
    classification_df: "pd.DataFrame",
    tokenizer,
    max_length: int,
) -> dict[str, object]:
    """Tokenize glycan sequences and keep multi-label targets aligned."""
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
# Model and training-argument helpers
# ---------------------------------------------------------------------------

def derive_classification_run_names(model_dir: str | Path) -> dict[str, str]:
    """Derive tokenizer and experiment names from one saved pretrained model path.

    Prefer sibling experiment metadata when it is available because some older
    continuation runs were saved with extra nesting that makes plain parent-path
    inference brittle. Fall back to the historical path-based convention only
    when metadata cannot be found.
    """
    model_path = Path(model_dir)

    for candidate_dir in model_path.parents:
        metadata_path = candidate_dir / "experiment_metadata.json"
        if not metadata_path.exists():
            continue

        try:
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        tokenizer_family = str(
            metadata_payload.get("live_hyperparameters", {}).get("tokenizer_family", "")
        ).strip()
        experiment_name = str(metadata_payload.get("experiment_name", "")).strip()
        if tokenizer_family and experiment_name:
            return {
                "model_dir_name": model_path.name,
                "experiment_name": experiment_name,
                "tokenizer_family": tokenizer_family,
            }

    if len(model_path.parts) < 3:
        raise ValueError(f"Model path is too short to derive run names: {model_path}")

    return {
        "model_dir_name": model_path.name,
        "experiment_name": model_path.parent.name,
        "tokenizer_family": model_path.parent.parent.name,
    }


def load_sequence_classification_model(
    pretrained_model_dir: str | Path,
    num_labels: int,
    initialization_mode: str = "mlm_checkpoint",
    device: str | None = None,
):
    """Load a multi-label sequence classifier from one requested starting mode."""
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
    elif initialization_mode == "classifier_checkpoint":
        model = AutoModelForSequenceClassification.from_pretrained(str(pretrained_model_dir))
    else:
        raise ValueError(
            "initialization_mode must be one of: "
            "'mlm_checkpoint', 'random_init', or 'classifier_checkpoint'."
        )

    if device is not None:
        model = model.to(torch.device(device))

    return model


def build_classification_training_arguments(
    checkpoint_dir: str | Path,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    num_train_epochs: int,
    weight_decay: float,
    save_total_limit: int,
    random_seed: int,
) -> dict[str, object]:
    """Create the Hugging Face training arguments used by notebook 10."""
    fp16_enabled = bool(torch.cuda.is_available())

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        learning_rate=float(learning_rate),
        per_device_train_batch_size=int(train_batch_size),
        per_device_eval_batch_size=int(eval_batch_size),
        num_train_epochs=int(num_train_epochs),
        weight_decay=float(weight_decay),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=int(save_total_limit),
        seed=int(random_seed),
        data_seed=int(random_seed),
        report_to="none",
        disable_tqdm=True,
        fp16=fp16_enabled,
    )

    return {
        "training_args": training_args,
        "fp16_enabled": fp16_enabled,
    }


def save_json(payload: dict[str, object], output_path: str | Path) -> Path:
    """Write one small JSON file to disk with pretty indentation."""
    return _save_json(payload, output_path)


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
    """Compute macro and weighted average precision robustly."""
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
# Validation-review helpers
# ---------------------------------------------------------------------------

def build_classification_validation_review(
    trainer_state_path: str | Path,
    tokenizer_family: str,
    pretrain_experiment_name: str,
    classifier_run_label: str,
    run_mode: str,
    total_epochs: int,
) -> dict[str, object]:
    """Build the notebook-10 validation review bundle from saved trainer history."""
    log_history_df = load_trainer_history(str(trainer_state_path))
    train_rows, eval_rows = split_train_eval_history(log_history_df)

    if train_rows.empty or eval_rows.empty:
        raise ValueError("Trainer history does not contain both training and validation loss rows.")

    loss_history_df = merge_loss_history(train_rows, eval_rows)
    best_epoch_summary = summarize_best_epoch(eval_rows)
    continuation_recommendation = recommend_continuation(
        eval_rows,
        total_epochs=int(total_epochs),
    )

    validation_review_summary = {
        "tokenizer_family": str(tokenizer_family),
        "pretrain_experiment_name": str(pretrain_experiment_name),
        "classifier_run_label": str(classifier_run_label),
        "run_mode": str(run_mode),
        "best_epoch_summary": best_epoch_summary,
        "continuation_recommendation": continuation_recommendation,
    }

    summary_df = pd.DataFrame(
        {
            "metric": [
                "best_epoch",
                "best_val_loss",
                "last_epoch",
                "last_val_loss",
                "continuation_recommendation",
            ],
            "value": [
                best_epoch_summary["best_epoch"],
                best_epoch_summary["best_val_loss"],
                best_epoch_summary["last_epoch"],
                best_epoch_summary["last_val_loss"],
                continuation_recommendation,
            ],
        }
    )

    return {
        "log_history_df": log_history_df,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "loss_history_df": loss_history_df,
        "best_epoch_summary": best_epoch_summary,
        "continuation_recommendation": continuation_recommendation,
        "validation_review_summary": validation_review_summary,
        "summary_df": summary_df,
    }


def save_classification_validation_review(
    validation_review_bundle: dict[str, object],
    output_paths: dict[str, Path],
) -> dict[str, Path]:
    """Save the classifier validation-review artifacts for notebook 10."""
    validation_review_bundle["loss_history_df"].to_csv(output_paths["loss_history_path"], index=False)
    _save_json(
        validation_review_bundle["validation_review_summary"],
        output_paths["validation_review_summary_path"],
    )

    return {
        "loss_history_path": output_paths["loss_history_path"],
        "validation_review_summary_path": output_paths["validation_review_summary_path"],
    }


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
    """Create a readable table of true labels, predicted labels, and probabilities."""
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
