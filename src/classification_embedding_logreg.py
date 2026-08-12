"""Helpers for notebook 14 binary logistic-regression probes on embeddings.

This module keeps the notebook-facing workflow for one specific comparison:
can a simple linear classifier separate ``N-glycan`` rows from other labeled
glycans when it only sees one saved embedding space at a time?

The goal is not to replace notebook 10 classification fine-tuning. Instead,
this is a lighter-weight linear-separability probe that makes it easier to
compare:

- tokenizer families
- architecture choices
- pretrained versus fine-tuned model states
- embedding pooling rules
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import gc
from html import escape
import json
from pathlib import Path
import re
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch

from src.classification_embedding_umap import (
    SUPPORTED_MODEL_VARIANTS,
    annotate_classification_umap_metadata,
    filter_classification_dataframe_by_split,
    load_combined_classification_splits,
    resolve_embedding_model_dir,
)
from src.classification_training import ACCESSION_COLUMN, LABEL_LIST_COLUMN, SEQUENCE_COLUMN, SPLIT_COLUMN
from src.notebook_utils import require_existing_path, stringify_path_values, validate_output_paths, write_json
from src.similarity_core import embed_sequences, load_similarity_artifacts, normalize_pooling_strategy


TARGET_COLUMN = "is_n_glycan_binary"
TARGET_LABEL_COLUMN = "binary_target_label"
DEFAULT_POSITIVE_LABEL = "N-glycan"
DEFAULT_NEGATIVE_LABEL = "Not N-glycan (including unlabeled)"
REGISTRY_REQUIRED_COLUMNS = {
    "experiment_name",
    "tokenizer_family",
    "run_mode",
    "num_hidden_layers",
    "attention_heads",
    "hidden_size",
    "run_status",
}
PLOT_UNIT_INTERVAL_METRICS = {"roc_auc", "average_precision", "f1", "balanced_accuracy", "accuracy"}
REPORT_PATH_PREFIXES = (
    "/content/drive/MyDrive/",
    "/drive/MyDrive/",
    "file:///content/drive/MyDrive/",
    "file:///drive/MyDrive/",
)

MODEL_VARIANT_LABELS = {
    "pretrained_mlm": "Pretrained MLM",
    "classification_mlm_init": "Classifier, MLM init",
    "classification_random_init": "Classifier, random init",
}


def load_run_registry(registry_csv_path: str | Path) -> pd.DataFrame:
    """Load the cleaned run registry used to select pretrained experiments."""
    registry_csv_path = require_existing_path(registry_csv_path, "Run registry CSV")
    registry_df = pd.read_csv(registry_csv_path)
    missing_columns = sorted(REGISTRY_REQUIRED_COLUMNS.difference(registry_df.columns))
    if missing_columns:
        raise ValueError(f"Run registry is missing required columns: {missing_columns}")
    return registry_df


def resolve_run_registry_path(
    *,
    drive_root: str | Path,
    repo_dir: str | Path | None = None,
    cleaned_registry_filename: str = "registry_cleaned_run_index.csv",
    fallback_registry_filename: str = "run_index.csv",
) -> Path:
    """Resolve the run-registry CSV using the project's Drive-first layout.

    Notebook 14 should prefer the cleaned audit snapshot when it exists in the
    Drive-backed project registry folder. If that file is not available, the
    notebook falls back to the maintained project run index.
    """
    candidate_paths: list[Path] = [
        Path(drive_root) / "registry" / cleaned_registry_filename,
        Path(drive_root) / cleaned_registry_filename,
        Path(drive_root) / "registry" / fallback_registry_filename,
    ]
    if repo_dir is not None:
        candidate_paths.extend(
            [
                Path(repo_dir) / cleaned_registry_filename,
                Path(repo_dir) / "registry" / fallback_registry_filename,
            ]
        )

    seen_paths: set[Path] = set()
    checked_paths: list[Path] = []
    for candidate_path in candidate_paths:
        resolved_candidate = Path(candidate_path)
        if resolved_candidate in seen_paths:
            continue
        seen_paths.add(resolved_candidate)
        checked_paths.append(resolved_candidate)
        if resolved_candidate.exists():
            return resolved_candidate

    checked_path_text = "\n".join(f"- {path}" for path in checked_paths)
    raise FileNotFoundError(
        "Run registry CSV not found. Checked:\n"
        f"{checked_path_text}"
    )


def build_architecture_label(
    num_hidden_layers: int | str,
    hidden_size: int | str,
    attention_heads: int | str,
) -> str:
    """Return a compact architecture label used in tables and plots."""
    return f"L{int(num_hidden_layers)} H{int(hidden_size)} A{int(attention_heads)}"


def _normalize_optional_string_list(values: Sequence[str] | None) -> list[str]:
    cleaned_values = [str(value).strip() for value in (values or []) if str(value).strip()]
    return cleaned_values


def _filter_registry_runs(
    registry_df: pd.DataFrame,
    *,
    tokenizer_families: Sequence[str] | None,
    experiment_names: Sequence[str] | None,
    only_fresh_runs: bool,
    only_tested_runs: bool,
) -> pd.DataFrame:
    """Return the registry subset requested by the notebook settings."""
    selected_df = registry_df.copy()

    normalized_tokenizer_families = _normalize_optional_string_list(tokenizer_families)
    if normalized_tokenizer_families:
        selected_df = selected_df.loc[selected_df["tokenizer_family"].isin(normalized_tokenizer_families)].copy()

    normalized_experiment_names = _normalize_optional_string_list(experiment_names)
    if normalized_experiment_names:
        selected_df = selected_df.loc[selected_df["experiment_name"].isin(normalized_experiment_names)].copy()

    if only_fresh_runs:
        selected_df = selected_df.loc[selected_df["run_mode"].map(str).eq("fresh")].copy()

    if only_tested_runs:
        selected_df = selected_df.loc[selected_df["run_status"].fillna("").map(str).eq("tested")].copy()

    return selected_df


def _build_one_registry_run_spec(
    row,
    *,
    model_variant: str,
    classifier_mlm_run_label: str,
    classifier_random_run_label: str,
) -> dict[str, Any]:
    """Convert one registry row plus one model state into a run-spec record."""
    architecture_label = build_architecture_label(
        row.num_hidden_layers,
        row.hidden_size,
        row.attention_heads,
    )
    variant_label = MODEL_VARIANT_LABELS[model_variant]
    return {
        "tokenizer_family": str(row.tokenizer_family),
        "experiment_name": str(row.experiment_name),
        "model_variant": str(model_variant),
        "architecture_label": architecture_label,
        "display_label": f"{row.tokenizer_family} | {architecture_label} | {variant_label}",
        "registry_run_mode": str(row.run_mode),
        "registry_run_status": str(row.run_status),
        "setting_label": str(getattr(row, "setting_label", "")),
        "classifier_mlm_run_label": str(classifier_mlm_run_label).strip(),
        "classifier_random_run_label": str(classifier_random_run_label).strip(),
        "num_hidden_layers": int(row.num_hidden_layers),
        "hidden_size": int(row.hidden_size),
        "attention_heads": int(row.attention_heads),
    }


def build_registry_run_specs(
    registry_df: pd.DataFrame,
    *,
    tokenizer_families: Sequence[str] | None = None,
    experiment_names: Sequence[str] | None = None,
    model_variants: Sequence[str] = SUPPORTED_MODEL_VARIANTS,
    classifier_mlm_run_label: str = "cls_lr2e-5_ep10_bs16_mlm",
    classifier_random_run_label: str = "cls_lr2e-5_ep10_bs16_randominit",
    only_fresh_runs: bool = True,
    only_tested_runs: bool = True,
) -> list[dict[str, Any]]:
    """Expand one registry slice into embedding-comparison run specs.

    Each selected pretrained experiment is expanded into one or more model
    states. The resulting records are designed to be notebook-friendly and can
    be passed directly into ``run_embedding_logreg_suite``.
    """
    selected_df = _filter_registry_runs(
        registry_df,
        tokenizer_families=tokenizer_families,
        experiment_names=experiment_names,
        only_fresh_runs=only_fresh_runs,
        only_tested_runs=only_tested_runs,
    )

    if selected_df.empty:
        return []

    invalid_variants = [variant for variant in model_variants if variant not in SUPPORTED_MODEL_VARIANTS]
    if invalid_variants:
        raise ValueError(
            f"Unsupported model_variants: {invalid_variants}. "
            f"Choose from {SUPPORTED_MODEL_VARIANTS}."
        )

    sort_columns = [
        "tokenizer_family",
        "num_hidden_layers",
        "hidden_size",
        "attention_heads",
        "experiment_name",
    ]
    selected_df = selected_df.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    run_specs: list[dict[str, Any]] = []
    for row in selected_df.itertuples(index=False):
        for model_variant in model_variants:
            run_specs.append(
                _build_one_registry_run_spec(
                    row,
                    model_variant=model_variant,
                    classifier_mlm_run_label=classifier_mlm_run_label,
                    classifier_random_run_label=classifier_random_run_label,
                )
            )

    return run_specs


def build_embedding_logreg_output_paths(
    project_root: str | Path,
    tokenizer_family: str,
    experiment_name: str,
    comparison_run_label: str,
) -> dict[str, Path]:
    """Return the standard notebook-14 output paths for one comparison run."""
    project_root = Path(project_root)
    results_dir = (
        project_root
        / "results"
        / "classification_embedding_logreg"
        / str(tokenizer_family).strip()
        / str(experiment_name).strip()
        / str(comparison_run_label).strip()
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    return {
        "results_dir": results_dir,
        "per_run_dir": results_dir / "per_run",
        "run_config_path": results_dir / "run_config.json",
        "run_manifest_path": results_dir / "run_manifest.csv",
        "skipped_runs_path": results_dir / "skipped_runs.csv",
        "target_summary_path": results_dir / "target_summary.csv",
        "split_metrics_path": results_dir / "split_metrics.csv",
        "train_summary_path": results_dir / "train_summary.csv",
        "val_summary_path": results_dir / "val_summary.csv",
        "test_summary_path": results_dir / "test_summary.csv",
        "train_plot_path": results_dir / "train_metric_grid.png",
        "val_plot_path": results_dir / "val_metric_grid.png",
        "test_plot_path": results_dir / "test_metric_grid.png",
        "html_report_path": results_dir / "n_glycan_logistic_regression_report.html",
    }


def prepare_n_glycan_probe_dataframe(
    *,
    train_csv_path: str | Path,
    val_csv_path: str | Path,
    test_csv_path: str | Path,
    label_vocabulary_path: str | Path,
    splits_to_include: Sequence[str] | None = None,
    exclude_unlabeled_rows: bool = False,
    positive_label: str = DEFAULT_POSITIVE_LABEL,
    negative_label: str = DEFAULT_NEGATIVE_LABEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load notebook-09 outputs and derive a binary N-glycan probe target.

    When ``exclude_unlabeled_rows`` is ``False``, rows with no surviving
    subtype labels are kept and treated as part of the non-``N-glycan`` class.
    """
    classification_df = load_combined_classification_splits(
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        test_csv_path=test_csv_path,
    )
    classification_df = filter_classification_dataframe_by_split(
        classification_df=classification_df,
        splits_to_include=splits_to_include,
    )
    label_vocabulary_df = pd.read_csv(require_existing_path(label_vocabulary_path, "Label vocabulary CSV"))

    annotated_df = annotate_classification_umap_metadata(
        classification_df=classification_df,
        label_vocabulary_df=label_vocabulary_df,
    )
    annotated_df["is_labeled_row"] = annotated_df[LABEL_LIST_COLUMN].map(
        lambda values: bool(values) if isinstance(values, list) else False
    )
    if exclude_unlabeled_rows:
        annotated_df = annotated_df.loc[annotated_df["is_labeled_row"]].copy()

    annotated_df[TARGET_COLUMN] = annotated_df["main_glycan_class"].eq("N-glycan").astype(int)
    annotated_df[TARGET_LABEL_COLUMN] = annotated_df[TARGET_COLUMN].map(
        lambda value: positive_label if int(value) == 1 else negative_label
    )
    return annotated_df.reset_index(drop=True), label_vocabulary_df


def summarize_binary_target(annotated_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the binary target counts by split and label."""
    summary_df = (
        annotated_df.groupby([SPLIT_COLUMN, TARGET_LABEL_COLUMN], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values([SPLIT_COLUMN, "count", TARGET_LABEL_COLUMN], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return summary_df


def summarize_main_glycan_class_by_split(annotated_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the broader glycan-class counts for quick notebook review."""
    summary_df = (
        annotated_df.groupby([SPLIT_COLUMN, "main_glycan_class"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values([SPLIT_COLUMN, "count", "main_glycan_class"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return summary_df


def build_embedding_logreg_run_config(
    *,
    drive_root: str | Path,
    classification_prep_dir: str | Path,
    checkpoints_dir: str | Path,
    pooling_strategy: str,
    splits_to_include: Sequence[str],
    train_splits: Sequence[str],
    evaluation_splits: Sequence[str],
    exclude_unlabeled_rows: bool,
    probability_threshold: float,
    logreg_c: float,
    class_weight: str | dict[int, float] | None,
    max_iter: int,
    random_state: int,
    batch_size: int,
    max_length: int | None,
    comparison_run_label: str,
    output_paths: Mapping[str, Path],
    run_specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the saved config payload for one notebook-14 comparison run."""
    return {
        "drive_root": str(drive_root),
        "classification_prep_dir": str(classification_prep_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "pooling_strategy": str(pooling_strategy),
        "splits_to_include": [str(split_name) for split_name in splits_to_include],
        "train_splits": [str(split_name) for split_name in train_splits],
        "evaluation_splits": [str(split_name) for split_name in evaluation_splits],
        "exclude_unlabeled_rows": bool(exclude_unlabeled_rows),
        "probability_threshold": float(probability_threshold),
        "logreg_c": float(logreg_c),
        "class_weight": class_weight,
        "max_iter": int(max_iter),
        "random_state": int(random_state),
        "batch_size": int(batch_size),
        "max_length": max_length,
        "comparison_run_label": str(comparison_run_label),
        "output_paths": stringify_path_values(dict(output_paths)),
        "run_specs": [stringify_path_values(dict(spec)) for spec in run_specs],
    }


def resolve_run_model_dir_from_spec(
    checkpoints_dir: str | Path,
    run_spec: Mapping[str, Any],
    model_subdir: str = "best_model",
) -> Path:
    """Resolve one embedding checkpoint directory from a notebook run spec."""
    return resolve_embedding_model_dir(
        checkpoints_dir=checkpoints_dir,
        tokenizer_family=str(run_spec["tokenizer_family"]),
        experiment_name=str(run_spec["experiment_name"]),
        model_variant=str(run_spec["model_variant"]),
        classifier_mlm_run_label=run_spec.get("classifier_mlm_run_label"),
        classifier_random_run_label=run_spec.get("classifier_random_run_label"),
        model_subdir=model_subdir,
    )


def _slugify_text(value: str) -> str:
    """Convert a human-readable run label part into a safe folder-name piece."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "run"


def build_run_slug(run_spec: Mapping[str, Any]) -> str:
    """Build one stable folder name for one compared run."""
    return "__".join(
        [
            _slugify_text(run_spec["tokenizer_family"]),
            _slugify_text(run_spec["experiment_name"]),
            _slugify_text(run_spec["model_variant"]),
        ]
    )


def build_run_manifest(
    run_specs: Sequence[Mapping[str, Any]],
    checkpoints_dir: str | Path,
    *,
    model_subdir: str = "best_model",
) -> pd.DataFrame:
    """Return one manifest row per requested run spec."""
    manifest_rows: list[dict[str, Any]] = []
    for run_spec in run_specs:
        model_dir = resolve_run_model_dir_from_spec(
            checkpoints_dir=checkpoints_dir,
            run_spec=run_spec,
            model_subdir=model_subdir,
        )
        manifest_rows.append(
            {
                **dict(run_spec),
                "run_slug": build_run_slug(run_spec),
                "model_dir": str(model_dir),
                "model_dir_exists": bool(model_dir.exists()),
            }
        )
    return pd.DataFrame(manifest_rows)


def _build_row_embedding_matrix(
    annotated_df: pd.DataFrame,
    *,
    model_dir: str | Path,
    pooling_strategy: str,
    batch_size: int,
    max_length: int | None,
    device: str | None = None,
) -> np.ndarray:
    """Embed the unique sequences once, then map them back to dataframe rows."""
    model_dir = require_existing_path(model_dir, "Embedding model directory")
    normalized_pooling_strategy = normalize_pooling_strategy(pooling_strategy)

    sequence_series = annotated_df[SEQUENCE_COLUMN].fillna("").map(str).map(str.strip)
    if sequence_series.eq("").any():
        raise ValueError("Annotated dataframe contains blank sequences.")

    # Many glycans appear more than once across train/val/test tables. We embed
    # each unique sequence once, then map those vectors back onto the full row
    # table so repeated rows share the exact same frozen embedding.
    unique_sequences = pd.Index(sequence_series.drop_duplicates().tolist())
    tokenizer, model, runtime_device = load_similarity_artifacts(str(model_dir), device=device)
    try:
        embedding_tensor = embed_sequences(
            sequences=unique_sequences.tolist(),
            tokenizer=tokenizer,
            model=model,
            device=runtime_device,
            max_length=max_length,
            batch_size=batch_size,
            pooling_strategy=normalized_pooling_strategy,
        )
        # Normalize once here so the downstream probe compares model states from
        # the same cosine-style embedding scale.
        normalized_tensor = torch.nn.functional.normalize(embedding_tensor, p=2, dim=1)
        normalized_embeddings = normalized_tensor.cpu().numpy()
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sequence_to_index = {sequence: index for index, sequence in enumerate(unique_sequences.tolist())}
    row_indices = sequence_series.map(sequence_to_index).to_numpy(dtype=int)
    return normalized_embeddings[row_indices]


def fit_binary_logistic_probe(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    *,
    regularization_c: float = 1.0,
    class_weight: str | dict[int, float] | None = "balanced",
    max_iter: int = 2000,
    random_state: int = 42,
) -> Pipeline:
    """Fit one standardized logistic-regression probe."""
    probe_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=float(regularization_c),
                    class_weight=class_weight,
                    max_iter=int(max_iter),
                    random_state=int(random_state),
                    solver="liblinear",
                ),
            ),
        ]
    )
    probe_model.fit(train_embeddings, train_targets)
    return probe_model


def _safe_metric(metric_fn, *args, **kwargs) -> float:
    """Return one metric value, using NaN when a split has only one class."""
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return float("nan")


def evaluate_binary_logistic_probe(
    probe_model: Pipeline,
    split_df: pd.DataFrame,
    split_embeddings: np.ndarray,
    *,
    run_spec: Mapping[str, Any],
    split_name: str,
    probability_threshold: float = 0.5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate one trained probe on one split."""
    target_values = split_df[TARGET_COLUMN].to_numpy(dtype=int)
    predicted_probabilities = probe_model.predict_proba(split_embeddings)[:, 1]
    predicted_labels = (predicted_probabilities >= float(probability_threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        target_values,
        predicted_labels,
        labels=[0, 1],
    ).ravel()

    metric_row = {
        **dict(run_spec),
        "split": str(split_name),
        "row_count": int(len(split_df)),
        "positive_count": int(target_values.sum()),
        "negative_count": int(len(target_values) - target_values.sum()),
        "positive_rate": float(target_values.mean()) if len(target_values) else float("nan"),
        "threshold": float(probability_threshold),
        "accuracy": float(accuracy_score(target_values, predicted_labels)),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, target_values, predicted_labels),
        "precision": _safe_metric(precision_score, target_values, predicted_labels, zero_division=0),
        "recall": _safe_metric(recall_score, target_values, predicted_labels, zero_division=0),
        "f1": _safe_metric(f1_score, target_values, predicted_labels, zero_division=0),
        "roc_auc": _safe_metric(roc_auc_score, target_values, predicted_probabilities),
        "average_precision": _safe_metric(average_precision_score, target_values, predicted_probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    # Keep the saved prediction table compact and notebook-friendly by carrying
    # forward only the columns that help interpret class balance and errors.
    prediction_columns = [
        ACCESSION_COLUMN,
        SEQUENCE_COLUMN,
        SPLIT_COLUMN,
        "primary_subtype_label",
        "n_o_category",
        "main_glycan_class",
        TARGET_COLUMN,
        TARGET_LABEL_COLUMN,
    ]
    available_prediction_columns = [column for column in prediction_columns if column in split_df.columns]
    prediction_df = split_df.loc[:, available_prediction_columns].copy()
    prediction_df["predicted_probability_n_glycan"] = predicted_probabilities
    prediction_df["predicted_label"] = predicted_labels
    prediction_df["correct_prediction"] = (predicted_labels == target_values).astype(int)
    prediction_df["display_label"] = run_spec.get("display_label", "")
    prediction_df["model_variant"] = run_spec.get("model_variant", "")
    prediction_df["architecture_label"] = run_spec.get("architecture_label", "")
    prediction_df["tokenizer_family"] = run_spec.get("tokenizer_family", "")
    prediction_df["experiment_name"] = run_spec.get("experiment_name", "")
    prediction_df["split"] = str(split_name)

    return metric_row, prediction_df


def _save_per_run_outputs(
    per_run_dir: str | Path,
    run_spec: Mapping[str, Any],
    metrics_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> dict[str, Path]:
    """Save one run's metrics and prediction tables."""
    run_dir = Path(per_run_dir) / build_run_slug(run_spec)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "split_metrics.csv"
    predictions_path = run_dir / "prediction_table.csv"
    config_path = run_dir / "run_spec.json"

    metrics_df.to_csv(metrics_path, index=False)
    prediction_df.to_csv(predictions_path, index=False)
    write_json(config_path, stringify_path_values(dict(run_spec)))

    return {
        "run_dir": run_dir,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "config_path": config_path,
    }


def _plot_metric_grid(
    metrics_df: pd.DataFrame,
    *,
    split_name: str,
    metric_names: Sequence[str],
    output_path: str | Path,
) -> Path:
    """Save one bar-plot grid for the requested metrics on one split."""
    plot_df = metrics_df.loc[metrics_df["split"] == str(split_name)].copy()
    if plot_df.empty:
        raise ValueError(f"No metric rows were found for split {split_name!r}.")

    ordered_variant_labels = list(MODEL_VARIANT_LABELS.values())
    variant_color_lookup = {
        ordered_variant_labels[0]: "#3b6fb6",
        ordered_variant_labels[1]: "#2f9e44",
        ordered_variant_labels[2]: "#c26d1a",
    }
    plot_df["variant_label"] = plot_df["model_variant"].map(MODEL_VARIANT_LABELS).fillna(plot_df["model_variant"])
    plot_df["plot_color"] = plot_df["variant_label"].map(variant_color_lookup).fillna("#666666")
    plot_df = plot_df.sort_values(
        ["tokenizer_family", "num_hidden_layers", "hidden_size", "attention_heads", "model_variant"],
        kind="stable",
    ).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure_height = max(2.6 * len(metric_names), 7.0)
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(16, figure_height))
    if len(metric_names) == 1:
        axes = [axes]

    y_positions = np.arange(len(plot_df))
    for axis, metric_name in zip(axes, metric_names):
        metric_values = plot_df[metric_name].to_numpy(dtype=float)
        axis.barh(y_positions, metric_values, color=plot_df["plot_color"].tolist())
        axis.set_yticks(y_positions)
        axis.set_yticklabels(plot_df["display_label"])
        axis.set_title(f"{split_name} {metric_name}")
        axis.set_xlabel(metric_name)
        axis.grid(axis="x", alpha=0.20)
        if metric_name in PLOT_UNIT_INTERVAL_METRICS:
            axis.set_xlim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_dataframe_html(frame_df: pd.DataFrame) -> str:
    """Render one dataframe as a compact HTML table for saved reports."""
    if frame_df.empty:
        return "<p class='subtle'>No rows were available for this section.</p>"
    return frame_df.to_html(index=False, classes="report-table", border=0)


def _render_plot_card(
    plot_path: str | Path | None,
    *,
    report_dir: str | Path,
    title: str,
) -> str:
    """Render one locally saved metric plot as a linked image card."""
    if not plot_path:
        return (
            "<div class='plot-card'>"
            f"<h3>{escape(title)}</h3>"
            "<p class='subtle'>No plot was created for this split.</p>"
            "</div>"
        )

    plot_path = Path(plot_path)
    if not plot_path.exists():
        return (
            "<div class='plot-card'>"
            f"<h3>{escape(title)}</h3>"
            f"<p class='subtle'>Missing image: {escape(str(plot_path))}</p>"
            "</div>"
        )

    relative_plot_path = plot_path.relative_to(Path(report_dir))
    image_src = escape(relative_plot_path.as_posix(), quote=True)
    return (
        "<div class='plot-card'>"
        f"<h3>{escape(title)}</h3>"
        f"<a href='{image_src}' target='_blank' rel='noopener'>"
        f"<img src='{image_src}' alt='{escape(title, quote=True)}'>"
        "</a>"
        "<p class='subtle'>Click the image to open the full-size saved PNG.</p>"
        "</div>"
    )


def render_embedding_logreg_html_report(
    *,
    output_dir: str | Path,
    report_title: str,
    manifest_df: pd.DataFrame,
    target_summary_df: pd.DataFrame,
    class_summary_df: pd.DataFrame,
    train_summary_df: pd.DataFrame,
    val_summary_df: pd.DataFrame,
    test_summary_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    plot_paths: Mapping[str, Path],
) -> Path:
    """Render a saved HTML report for one notebook-14 comparison run."""
    output_dir = Path(output_dir)
    html_path = output_dir / "n_glycan_logistic_regression_report.html"

    summary_cards = [
        ("Requested runs", len(manifest_df)),
        ("Existing model dirs", int(manifest_df["model_dir_exists"].sum()) if "model_dir_exists" in manifest_df else 0),
        ("Completed runs", len(test_summary_df)),
        ("Skipped runs", len(skipped_df)),
    ]
    summary_cards_html = "".join(
        (
            "<div class='card'>"
            f"<h3>{escape(label)}</h3>"
            f"<p class='metric'>{value}</p>"
            "</div>"
        )
        for label, value in summary_cards
    )
    plot_cards_html = "".join(
        [
            _render_plot_card(plot_paths.get("train"), report_dir=output_dir, title="Train metric grid"),
            _render_plot_card(plot_paths.get("val"), report_dir=output_dir, title="Validation metric grid"),
            _render_plot_card(plot_paths.get("test"), report_dir=output_dir, title="Test metric grid"),
        ]
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(report_title)}</title>
  <style>
    :root {{
      --ink: #1f2933;
      --muted: #52606d;
      --paper: #f6f2e8;
      --card: #fffaf1;
      --line: #d8cfbf;
      --accent: #9f4c24;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: linear-gradient(180deg, #fbf7ef 0%, var(--paper) 100%);
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.5;
    }}
    header {{
      padding: 34px 42px 20px;
      border-bottom: 1px solid var(--line);
    }}
    h1, h2, h3 {{ line-height: 1.15; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    h2 {{ margin: 30px 0 12px; font-size: 24px; }}
    h3 {{ margin: 0 0 8px; font-size: 18px; }}
    p {{ margin: 0 0 12px; }}
    .subtle {{ color: var(--muted); }}
    .container {{ padding: 24px 42px 52px; }}
    .summary-grid, .plot-grid {{
      display: grid;
      gap: 16px;
    }}
    .summary-grid {{
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}
    .plot-grid {{
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }}
    .card, .plot-card {{
      background: rgba(255, 250, 241, 0.96);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 28px rgba(74, 57, 35, 0.08);
    }}
    .card {{
      padding: 16px;
    }}
    .plot-card {{
      padding: 14px;
    }}
    .metric {{
      margin: 0;
      color: var(--accent);
      font-size: 28px;
      font-weight: bold;
    }}
    .plot-card a {{
      display: block;
      color: inherit;
      text-decoration: none;
    }}
    .plot-card img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: rgba(255, 250, 241, 0.96);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 28px rgba(74, 57, 35, 0.08);
      padding: 12px;
    }}
    table.report-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      background: white;
    }}
    table.report-table th,
    table.report-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid #ece6d9;
      text-align: left;
      vertical-align: top;
    }}
    table.report-table th {{
      background: #f5efe1;
      position: sticky;
      top: 0;
    }}
    code {{
      background: #f2ebdc;
      padding: 2px 5px;
      border-radius: 6px;
      font-size: 0.95em;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(report_title)}</h1>
    <p class="subtle">
      This report compares one tokenizer plus one pretrained experiment across the saved model states
      <code>pretrained_mlm</code>, <code>classification_mlm_init</code>, and <code>classification_random_init</code>.
    </p>
  </header>
  <main class="container">
    <section>
      <h2>Run overview</h2>
      <div class="summary-grid">{summary_cards_html}</div>
    </section>
    <section>
      <h2>Requested runs</h2>
      <div class="table-wrap">{_render_dataframe_html(manifest_df)}</div>
    </section>
    <section>
      <h2>Binary target counts</h2>
      <p class="subtle">Unlabeled rows remain in the negative class when the notebook keeps them.</p>
      <div class="table-wrap">{_render_dataframe_html(target_summary_df)}</div>
    </section>
    <section>
      <h2>Broad glycan-class counts</h2>
      <div class="table-wrap">{_render_dataframe_html(class_summary_df)}</div>
    </section>
    <section>
      <h2>Saved metric plots</h2>
      <div class="plot-grid">{plot_cards_html}</div>
    </section>
    <section>
      <h2>Train summary</h2>
      <div class="table-wrap">{_render_dataframe_html(train_summary_df)}</div>
    </section>
    <section>
      <h2>Validation summary</h2>
      <div class="table-wrap">{_render_dataframe_html(val_summary_df)}</div>
    </section>
    <section>
      <h2>Test summary</h2>
      <div class="table-wrap">{_render_dataframe_html(test_summary_df)}</div>
    </section>
    <section>
      <h2>Skipped runs</h2>
      <div class="table-wrap">{_render_dataframe_html(skipped_df)}</div>
    </section>
  </main>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


def _iter_local_html_references(html_text: str) -> list[str]:
    """Return local image and link references found in one HTML document."""
    references = re.findall(r"""(?:src|href)=["']([^"']+)["']""", str(html_text), flags=re.IGNORECASE)
    local_references: list[str] = []
    for reference in references:
        normalized_reference = str(reference).strip()
        if not normalized_reference:
            continue
        if normalized_reference.startswith(("#", "data:", "mailto:", "javascript:")):
            continue
        if "://" in normalized_reference:
            continue
        local_references.append(normalized_reference)
    return local_references


def _resolve_local_export_reference(reference_path: str, source_dir: str | Path) -> Path | None:
    """Resolve one local HTML reference against the source report directory."""
    try:
        source_dir = Path(source_dir).resolve()
        candidate_path = (source_dir / str(reference_path)).resolve()
    except OSError:
        return None
    return candidate_path


def _scan_text_for_sensitive_strings(
    text: str,
    *,
    source_name: str,
    extra_blocked_strings: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Return rows for text snippets that still look tied to a local environment."""
    blocked_strings = list(REPORT_PATH_PREFIXES)
    blocked_strings.extend(str(value) for value in (extra_blocked_strings or []) if str(value).strip())

    scan_rows: list[dict[str, str]] = []
    lower_text = str(text)
    for blocked_string in blocked_strings:
        if blocked_string and blocked_string in lower_text:
            scan_rows.append(
                {
                    "source_name": str(source_name),
                    "match_type": "blocked_string",
                    "matched_text": str(blocked_string),
                }
            )
    return scan_rows


def export_public_embedding_logreg_html(
    *,
    probe_results: Mapping[str, Any] | None = None,
    plot_paths: Mapping[str, Any] | None = None,
    export_dir: str | Path | None = None,
    repo_public_subdir: str | None = None,
    repo_owner: str | None = None,
    repo_name: str | None = None,
    repo_ref: str = "main",
    extra_blocked_strings: Sequence[str] | None = None,
) -> dict[str, object]:
    """Copy one notebook-14 HTML report and its local assets into a clean folder."""
    source_plot_paths = plot_paths if plot_paths is not None else (probe_results or {})
    source_html_path = source_plot_paths.get("html_report_path")
    if source_html_path is None:
        raise ValueError("Provide either probe_results or plot_paths when exporting public HTML.")
    if export_dir is None:
        raise ValueError("export_dir is required for export_public_embedding_logreg_html().")

    source_html_path = Path(source_html_path).resolve()
    output_path = source_html_path.parent
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    pending_relative_paths: list[Path] = [Path(source_html_path.name)]
    copied_file_rows: list[dict[str, str]] = []
    dependency_issue_rows: list[dict[str, str]] = []
    scan_rows: list[dict[str, str]] = []
    seen_relative_paths: set[str] = set()

    while pending_relative_paths:
        relative_path = Path(pending_relative_paths.pop(0))
        relative_key = relative_path.as_posix()
        if relative_key in seen_relative_paths:
            continue
        seen_relative_paths.add(relative_key)

        source_path = (output_path / relative_path).resolve()
        if not source_path.exists():
            dependency_issue_rows.append(
                {
                    "reference_path": relative_key,
                    "issue_type": "missing_source_file",
                    "source_html": "",
                }
            )
            continue

        target_path = export_path / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied_file_rows.append(
            {
                "relative_path": relative_key,
                "source_path": str(source_path),
                "export_path": str(target_path),
            }
        )

        if source_path.suffix.lower() not in {".html", ".htm", ".css", ".js"}:
            continue

        text = source_path.read_text(encoding="utf-8")
        scan_rows.extend(
            _scan_text_for_sensitive_strings(
                text,
                source_name=relative_key,
                extra_blocked_strings=extra_blocked_strings,
            )
        )

        if source_path.suffix.lower() not in {".html", ".htm"}:
            continue

        for referenced_path in _iter_local_html_references(text):
            resolved_reference = _resolve_local_export_reference(referenced_path, source_path.parent)
            if resolved_reference is None:
                dependency_issue_rows.append(
                    {
                        "reference_path": referenced_path,
                        "issue_type": "unsupported_or_invalid_reference",
                        "source_html": relative_key,
                    }
                )
                continue
            try:
                pending_relative_paths.append(resolved_reference.relative_to(output_path))
            except ValueError:
                dependency_issue_rows.append(
                    {
                        "reference_path": referenced_path,
                        "issue_type": "reference_outside_output_dir",
                        "source_html": relative_key,
                    }
                )

    copied_files_df = pd.DataFrame(copied_file_rows)
    dependency_issues_df = pd.DataFrame(dependency_issue_rows)
    scan_results_df = pd.DataFrame(scan_rows)

    repo_index_path = ""
    githack_url = ""
    if repo_public_subdir:
        repo_index_path = f"{repo_public_subdir.rstrip('/')}/n_glycan_logistic_regression_report.html"
        if repo_owner and repo_name:
            githack_url = (
                f"https://raw.githack.com/{repo_owner}/{repo_name}/{repo_ref}/"
                f"{repo_index_path}"
            )

    return {
        "public_export_dir": str(export_path),
        "repo_public_subdir": repo_public_subdir or "",
        "repo_index_path": repo_index_path,
        "githack_url": githack_url,
        "copied_files_df": copied_files_df,
        "dependency_issues_df": dependency_issues_df,
        "scan_results_df": scan_results_df,
        "has_dependency_issues": not dependency_issues_df.empty,
        "has_sensitive_matches": not scan_results_df.empty,
    }


def build_metric_split_summary(metrics_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Return one split-only summary table sorted for quick notebook review."""
    summary_df = metrics_df.loc[metrics_df["split"] == str(split_name)].copy()
    if summary_df.empty:
        return summary_df

    summary_df = summary_df.sort_values(
        ["roc_auc", "average_precision", "f1", "balanced_accuracy"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    return summary_df


def run_embedding_logreg_suite(
    *,
    annotated_df: pd.DataFrame,
    run_specs: Sequence[Mapping[str, Any]],
    checkpoints_dir: str | Path,
    output_paths: Mapping[str, Path],
    pooling_strategy: str = "mean",
    batch_size: int = 32,
    max_length: int | None = None,
    train_splits: Sequence[str] = ("train",),
    evaluation_splits: Sequence[str] = ("train", "val", "test"),
    probability_threshold: float = 0.5,
    regularization_c: float = 1.0,
    class_weight: str | dict[int, float] | None = "balanced",
    max_iter: int = 2000,
    random_state: int = 42,
    model_subdir: str = "best_model",
    overwrite_existing_outputs: bool = True,
    fail_on_missing_model_dirs: bool = False,
    metric_plot_names: Sequence[str] = ("roc_auc", "average_precision", "f1", "balanced_accuracy"),
    report_title: str = "N-glycan logistic-regression comparison",
    device: str | None = None,
) -> dict[str, Any]:
    """Run the notebook-14 embedding comparison end to end."""
    normalized_pooling_strategy = normalize_pooling_strategy(pooling_strategy)
    output_paths = {key: Path(value) for key, value in output_paths.items()}
    output_paths["per_run_dir"].mkdir(parents=True, exist_ok=True)

    validate_output_paths(
        {
            "run_manifest_path": output_paths["run_manifest_path"],
            "skipped_runs_path": output_paths["skipped_runs_path"],
            "split_metrics_path": output_paths["split_metrics_path"],
            "train_summary_path": output_paths["train_summary_path"],
            "val_summary_path": output_paths["val_summary_path"],
            "test_summary_path": output_paths["test_summary_path"],
            "train_plot_path": output_paths["train_plot_path"],
            "val_plot_path": output_paths["val_plot_path"],
            "test_plot_path": output_paths["test_plot_path"],
            "html_report_path": output_paths["html_report_path"],
        },
        overwrite_existing_outputs=overwrite_existing_outputs,
    )

    target_summary_df = summarize_binary_target(annotated_df)
    target_summary_df.to_csv(output_paths["target_summary_path"], index=False)
    class_summary_df = summarize_main_glycan_class_by_split(annotated_df)

    train_splits = tuple(str(split_name) for split_name in train_splits)
    evaluation_splits = tuple(str(split_name) for split_name in evaluation_splits)
    manifest_df = build_run_manifest(
        run_specs=run_specs,
        checkpoints_dir=checkpoints_dir,
        model_subdir=model_subdir,
    )
    manifest_df.to_csv(output_paths["run_manifest_path"], index=False)

    all_metric_rows: list[dict[str, Any]] = []
    all_prediction_frames: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, Any]] = []

    for run_spec in run_specs:
        model_dir = resolve_run_model_dir_from_spec(
            checkpoints_dir=checkpoints_dir,
            run_spec=run_spec,
            model_subdir=model_subdir,
        )
        if not model_dir.exists():
            skipped_row = {
                **dict(run_spec),
                "model_dir": str(model_dir),
                "reason": "model_dir_missing",
            }
            skipped_rows.append(skipped_row)
            if fail_on_missing_model_dirs:
                raise FileNotFoundError(f"Model directory not found for run spec: {skipped_row}")
            continue

        # Reuse one row-aligned embedding matrix for every split in this run so
        # the train/val/test probe comparison reflects one consistent model state.
        row_embeddings = _build_row_embedding_matrix(
            annotated_df=annotated_df,
            model_dir=model_dir,
            pooling_strategy=normalized_pooling_strategy,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        run_df = annotated_df.reset_index(drop=True).copy()
        train_mask = run_df[SPLIT_COLUMN].isin(train_splits)
        if not train_mask.any():
            raise ValueError(f"No train rows were found for run spec: {run_spec}")

        train_targets = run_df.loc[train_mask, TARGET_COLUMN].to_numpy(dtype=int)
        if len(np.unique(train_targets)) < 2:
            raise ValueError(
                "The train split for the N-glycan probe only contains one class. "
                "Adjust the filtering settings before rerunning."
            )

        # Fit one simple linear probe on the train split only. The same fitted
        # probe is then carried unchanged into validation and test.
        probe_model = fit_binary_logistic_probe(
            train_embeddings=row_embeddings[train_mask.to_numpy()],
            train_targets=train_targets,
            regularization_c=regularization_c,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )

        run_metric_rows: list[dict[str, Any]] = []
        run_prediction_frames: list[pd.DataFrame] = []
        for split_name in evaluation_splits:
            split_mask = run_df[SPLIT_COLUMN].map(str).eq(str(split_name))
            if not split_mask.any():
                continue

            metric_row, prediction_df = evaluate_binary_logistic_probe(
                probe_model=probe_model,
                split_df=run_df.loc[split_mask].reset_index(drop=True),
                split_embeddings=row_embeddings[split_mask.to_numpy()],
                run_spec=run_spec,
                split_name=str(split_name),
                probability_threshold=probability_threshold,
            )
            run_metric_rows.append(metric_row)
            run_prediction_frames.append(prediction_df)

        run_metrics_df = pd.DataFrame(run_metric_rows)
        run_prediction_df = pd.concat(run_prediction_frames, ignore_index=True)
        per_run_paths = _save_per_run_outputs(
            per_run_dir=output_paths["per_run_dir"],
            run_spec=run_spec,
            metrics_df=run_metrics_df,
            prediction_df=run_prediction_df,
        )

        for metric_row in run_metric_rows:
            metric_row["model_dir"] = str(model_dir)
            metric_row["per_run_dir"] = str(per_run_paths["run_dir"])
        all_metric_rows.extend(run_metric_rows)
        all_prediction_frames.append(run_prediction_df)

    skipped_df = pd.DataFrame(skipped_rows)
    skipped_df.to_csv(output_paths["skipped_runs_path"], index=False)

    if not all_metric_rows:
        raise RuntimeError(
            "No embedding probe runs completed successfully. Check the run specs and model directories."
        )

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_df.to_csv(output_paths["split_metrics_path"], index=False)

    # Save one sorted summary per split so the notebook can display train, val,
    # and test rankings directly without repeating sorting logic inline.
    split_summaries = {}
    for split_name, summary_path in (
        ("train", output_paths["train_summary_path"]),
        ("val", output_paths["val_summary_path"]),
        ("test", output_paths["test_summary_path"]),
    ):
        summary_df = build_metric_split_summary(metrics_df, split_name)
        summary_df.to_csv(summary_path, index=False)
        split_summaries[split_name] = summary_df

    created_plot_paths = {}
    for split_name, plot_path in (
        ("train", output_paths["train_plot_path"]),
        ("val", output_paths["val_plot_path"]),
        ("test", output_paths["test_plot_path"]),
    ):
        if split_summaries[split_name].empty:
            continue
        created_plot_paths[split_name] = _plot_metric_grid(
            metrics_df=metrics_df,
            split_name=split_name,
            metric_names=metric_plot_names,
            output_path=plot_path,
        )

    html_report_path = render_embedding_logreg_html_report(
        output_dir=output_paths["results_dir"],
        report_title=report_title,
        manifest_df=manifest_df,
        target_summary_df=target_summary_df,
        class_summary_df=class_summary_df,
        train_summary_df=split_summaries["train"],
        val_summary_df=split_summaries["val"],
        test_summary_df=split_summaries["test"],
        skipped_df=skipped_df,
        plot_paths=created_plot_paths,
    )

    return {
        "target_summary_df": target_summary_df,
        "class_summary_df": class_summary_df,
        "manifest_df": manifest_df,
        "skipped_df": skipped_df,
        "metrics_df": metrics_df,
        "prediction_df": pd.concat(all_prediction_frames, ignore_index=True),
        "train_summary_df": split_summaries["train"],
        "val_summary_df": split_summaries["val"],
        "test_summary_df": split_summaries["test"],
        "plot_paths": created_plot_paths,
        "html_report_path": html_report_path,
        "output_paths": dict(output_paths),
    }


def save_embedding_logreg_run_config(
    output_path: str | Path,
    config_payload: Mapping[str, Any],
) -> Path:
    """Write one pretty JSON config payload for notebook-14 runs."""
    return write_json(output_path, stringify_path_values(dict(config_payload)))
