"""Helpers for notebook 14 logistic-regression probes on embeddings.

This module keeps the notebook-facing workflow for lightweight linear probes on
saved embedding spaces. The default use is still the original binary
``N-glycan`` versus other-glycan separation task, but the helpers also support
a harder N-glycan subclass comparison that stays inside the ``N-glycan``
subset.

The goal is not to replace notebook 10 classification fine-tuning. Instead,
this is a lighter-weight linear-separability probe that makes it easier to
compare:

- tokenizer families
- architecture choices
- pretrained versus fine-tuned model states
- embedding pooling rules
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import gc
from html import escape
from itertools import combinations
import json
from pathlib import Path
import re
import shutil
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch

from src.classification_embedding_umap import (
    SUPPORTED_MODEL_VARIANTS,
    annotate_classification_umap_metadata,
    build_umap_dataframe,
    compute_umap_projection,
    filter_classification_dataframe_by_split,
    load_combined_classification_splits,
    resolve_embedding_model_dir,
    transform_umap_projection,
)
from src.classification_training import ACCESSION_COLUMN, LABEL_LIST_COLUMN, SEQUENCE_COLUMN, SPLIT_COLUMN
from src.notebook_utils import require_existing_path, stringify_path_values, validate_output_paths, write_json
from src.n_glycan_structural_classification import (
    STRUCTURAL_EXCLUDE_TRUE_CONTRADICTIONS,
    STRUCTURAL_INCLUDE_TRUE_CONTRADICTIONS,
    SUPPORTED_STRUCTURAL_CONTRADICTION_POLICIES,
    build_structural_classification_output_paths,
)
from src.similarity_core import (
    embed_sequences,
    load_similarity_artifacts,
    normalize_embedding_layer_index,
    normalize_pooling_strategy,
)


TARGET_COLUMN = "probe_target_code"
TARGET_LABEL_COLUMN = "probe_target_label"
TARGET_KIND_COLUMN = "probe_target_kind"
PREDICTED_PROBABILITY_COLUMN = "predicted_probability_target_class"
PREDICTED_PROBABILITIES_JSON_COLUMN = "predicted_probabilities_json"
DEFAULT_POSITIVE_LABEL = "N-glycan"
DEFAULT_NEGATIVE_LABEL = "Not N-glycan (including unlabeled)"
DEFAULT_BINARY_TARGET_NAME = "N-glycan vs other"
DEFAULT_SUBCLASS_TARGET_NAME = "N-glycan subclass probe"
CURRENT_PROJECT_LABEL_SOURCE = "current_project_labels"
STRUCTURAL_RULE_LABEL_SOURCE = "structural_rule_labels"
SUPPORTED_PROBE_LABEL_SOURCES = (
    CURRENT_PROJECT_LABEL_SOURCE,
    STRUCTURAL_RULE_LABEL_SOURCE,
)
DEFAULT_STRUCTURAL_CONTRADICTION_POLICY = STRUCTURAL_EXCLUDE_TRUE_CONTRADICTIONS
SUPPORTED_PROBE_TARGET_MODES = (
    "n_glycan_binary",
    "n_glycan_subclass_multiclass",
)
SUBCLASS_EXCLUSION_REASON_COLUMN = "probe_exclusion_reason"
N_GLYCAN_SUBCLASS_COLUMN = "n_glycan_subclass_label"
N_GLYCAN_SUBCLASS_MATCHES_COLUMN = "n_glycan_subclass_matches"
N_GLYCAN_SUBCLASS_MATCH_COUNT_COLUMN = "n_glycan_subclass_match_count"
DEFAULT_N_GLYCAN_SUBCLASS_CATEGORIES = (
    "High mannose",
    "Complex",
    "Hybrid",
)
N_GLYCAN_SUBCLASS_KEYWORDS = {
    "High mannose": ("high mannose", "high-mannose"),
    "Complex": ("complex n", "complex-n", "bisected"),
    "Hybrid": ("hybrid",),
    "Paucimannose": ("paucimannose",),
}
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
MODEL_VARIANT_ORDER = (
    "pretrained_mlm",
    "classification_mlm_init",
    "classification_random_init",
)
VARIANT_COLOR_LOOKUP = {
    "pretrained_mlm": "#3b6fb6",
    "classification_mlm_init": "#2f9e44",
    "classification_random_init": "#c26d1a",
}

MODEL_VARIANT_LABELS = {
    "pretrained_mlm": "Pretrained MLM",
    "classification_mlm_init": "Classifier, MLM init",
    "classification_random_init": "Classifier, random init",
}
STANDARD_PROBE_SPLITS = ("train", "val", "test")
DEFAULT_HIGHLIGHT_POINT_COLORS = ("#c0392b", "#1f78b4", "#2f9e44", "#8e44ad")


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
    """Resolve the run registry from the Drive-backed project layout first."""
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

    checked_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for candidate_path in candidate_paths:
        normalized_path = Path(candidate_path)
        if normalized_path in seen_paths:
            continue
        seen_paths.add(normalized_path)
        checked_paths.append(normalized_path)
        if normalized_path.exists():
            return normalized_path

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


def _normalize_run_label_candidates(values: str | Sequence[str] | None) -> list[str]:
    """Normalize one run-label setting into a de-duplicated candidate list."""
    if values is None:
        return []
    if isinstance(values, str):
        normalized_values = [values]
    else:
        normalized_values = list(values)

    cleaned_values: list[str] = []
    for value in normalized_values:
        cleaned_value = str(value).strip()
        if not cleaned_value or cleaned_value in cleaned_values:
            continue
        cleaned_values.append(cleaned_value)
    return cleaned_values


def _build_classification_variant_model_dir(
    *,
    checkpoints_dir: str | Path,
    tokenizer_family: str,
    experiment_name: str,
    model_variant: str,
    classifier_run_label: str,
    model_subdir: str,
) -> Path:
    """Build one classification-model directory candidate for label resolution."""
    return resolve_embedding_model_dir(
        checkpoints_dir=checkpoints_dir,
        tokenizer_family=tokenizer_family,
        experiment_name=experiment_name,
        model_variant=model_variant,
        classifier_mlm_run_label=classifier_run_label,
        classifier_random_run_label=classifier_run_label,
        model_subdir=model_subdir,
    )


def _resolve_one_classifier_run_label(
    *,
    checkpoints_dir: str | Path,
    tokenizer_family: str,
    experiment_name: str,
    model_variant: str,
    run_label_candidates: str | Sequence[str] | None,
    model_subdir: str = "best_model",
) -> dict[str, Any]:
    """Resolve one classifier run label by checking a list of saved-folder candidates."""
    candidate_labels = _normalize_run_label_candidates(run_label_candidates)
    if not candidate_labels:
        raise ValueError(f"At least one run-label candidate is required for {model_variant}.")

    candidate_rows: list[dict[str, Any]] = []
    for run_label in candidate_labels:
        model_dir = _build_classification_variant_model_dir(
            checkpoints_dir=checkpoints_dir,
            tokenizer_family=tokenizer_family,
            experiment_name=experiment_name,
            model_variant=model_variant,
            classifier_run_label=run_label,
            model_subdir=model_subdir,
        )
        candidate_rows.append(
            {
                "model_variant": model_variant,
                "candidate_run_label": run_label,
                "candidate_model_dir": str(model_dir),
                "candidate_exists": bool(model_dir.exists()),
            }
        )

    selected_row = next((row for row in candidate_rows if row["candidate_exists"]), candidate_rows[0])
    selection_status = "matched_existing_candidate" if selected_row["candidate_exists"] else "no_candidate_exists"
    return {
        "model_variant": model_variant,
        "selected_run_label": str(selected_row["candidate_run_label"]),
        "selected_model_dir": str(selected_row["candidate_model_dir"]),
        "selected_model_dir_exists": bool(selected_row["candidate_exists"]),
        "selection_status": selection_status,
        "candidate_rows": candidate_rows,
    }


def resolve_classifier_run_label_choices(
    *,
    checkpoints_dir: str | Path,
    tokenizer_family: str,
    experiment_name: str,
    classifier_mlm_run_label_candidates: str | Sequence[str] | None,
    classifier_random_run_label_candidates: str | Sequence[str] | None,
    model_subdir: str = "best_model",
) -> dict[str, dict[str, Any]]:
    """Resolve both classifier-state run labels using Drive folder existence."""
    return {
        "classification_mlm_init": _resolve_one_classifier_run_label(
            checkpoints_dir=checkpoints_dir,
            tokenizer_family=tokenizer_family,
            experiment_name=experiment_name,
            model_variant="classification_mlm_init",
            run_label_candidates=classifier_mlm_run_label_candidates,
            model_subdir=model_subdir,
        ),
        "classification_random_init": _resolve_one_classifier_run_label(
            checkpoints_dir=checkpoints_dir,
            tokenizer_family=tokenizer_family,
            experiment_name=experiment_name,
            model_variant="classification_random_init",
            run_label_candidates=classifier_random_run_label_candidates,
            model_subdir=model_subdir,
        ),
    }


def build_classifier_run_label_resolution_table(
    resolution_map: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Flatten the classifier run-label resolution map for notebook display."""
    table_rows: list[dict[str, Any]] = []
    for model_variant in MODEL_VARIANT_ORDER:
        if model_variant not in resolution_map:
            continue
        resolution = resolution_map[model_variant]
        table_rows.append(
            {
                "model_variant": model_variant,
                "variant_label": MODEL_VARIANT_LABELS.get(model_variant, model_variant),
                "selected_run_label": resolution.get("selected_run_label", ""),
                "selected_model_dir_exists": bool(resolution.get("selected_model_dir_exists", False)),
                "selection_status": resolution.get("selection_status", ""),
                "candidate_run_labels": " | ".join(
                    str(row.get("candidate_run_label", ""))
                    for row in resolution.get("candidate_rows", [])
                ),
            }
        )
    return pd.DataFrame(table_rows)


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


def resolve_single_registry_run(
    registry_df: pd.DataFrame,
    *,
    tokenizer_family: str,
    experiment_name: str,
    only_fresh_runs: bool = True,
    only_tested_runs: bool = True,
):
    """Return one registry row for the selected tokenizer family and experiment."""
    selected_df = _filter_registry_runs(
        registry_df,
        tokenizer_families=[tokenizer_family],
        experiment_names=[experiment_name],
        only_fresh_runs=only_fresh_runs,
        only_tested_runs=only_tested_runs,
    )
    if selected_df.empty:
        raise ValueError(
            "No registry rows matched the requested tokenizer family and experiment. "
            "Check TOKENIZER_FAMILY, EXPERIMENT_NAME, or the registry filters."
        )
    if len(selected_df) != 1:
        raise ValueError(
            "Snapshot progression expects exactly one registry row for the selected "
            "tokenizer family and experiment."
        )
    return selected_df.iloc[0]


def build_snapshot_run_specs(
    registry_df: pd.DataFrame,
    *,
    checkpoints_dir: str | Path,
    tokenizer_family: str,
    experiment_name: str,
    snapshot_model_variant: str = "pretrained_mlm",
    snapshot_model_subdirs: Sequence[str] = ("best_model",),
    classifier_mlm_run_label: str = "cls_lr2e-5_ep10_bs16_mlm",
    classifier_random_run_label: str = "cls_lr2e-5_ep10_bs16_randominit",
    only_fresh_runs: bool = True,
    only_tested_runs: bool = True,
) -> list[dict[str, Any]]:
    """Build ordered run specs for one model lineage across saved snapshots."""
    normalized_variant = str(snapshot_model_variant).strip()
    if normalized_variant not in SUPPORTED_MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported snapshot_model_variant {snapshot_model_variant!r}. "
            f"Choose from {SUPPORTED_MODEL_VARIANTS}."
        )

    normalized_snapshot_subdirs = [
        str(snapshot_name).strip()
        for snapshot_name in snapshot_model_subdirs
        if str(snapshot_name).strip()
    ]
    if not normalized_snapshot_subdirs:
        raise ValueError("At least one snapshot model subdirectory is required.")

    row = resolve_single_registry_run(
        registry_df,
        tokenizer_family=tokenizer_family,
        experiment_name=experiment_name,
        only_fresh_runs=only_fresh_runs,
        only_tested_runs=only_tested_runs,
    )
    base_run_spec = _build_one_registry_run_spec(
        row,
        model_variant=normalized_variant,
        classifier_mlm_run_label=classifier_mlm_run_label,
        classifier_random_run_label=classifier_random_run_label,
    )

    snapshot_run_specs: list[dict[str, Any]] = []
    for snapshot_order, snapshot_model_subdir in enumerate(normalized_snapshot_subdirs):
        model_dir = resolve_embedding_model_dir(
            checkpoints_dir=checkpoints_dir,
            tokenizer_family=tokenizer_family,
            experiment_name=experiment_name,
            model_variant=normalized_variant,
            classifier_mlm_run_label=classifier_mlm_run_label,
            classifier_random_run_label=classifier_random_run_label,
            model_subdir=snapshot_model_subdir,
        )
        snapshot_label = str(snapshot_model_subdir)
        snapshot_run_specs.append(
            {
                **base_run_spec,
                "model_dir": str(model_dir),
                "comparison_axis_name": "snapshot",
                "comparison_label": snapshot_label,
                "snapshot_label": snapshot_label,
                "snapshot_model_subdir": snapshot_model_subdir,
                "snapshot_order": int(snapshot_order),
                "display_label": (
                    f"{MODEL_VARIANT_LABELS.get(normalized_variant, normalized_variant)}"
                    f" | {snapshot_label}"
                ),
                "comparison_group_label": (
                    f"{tokenizer_family} | {base_run_spec['architecture_label']} | "
                    f"{MODEL_VARIANT_LABELS.get(normalized_variant, normalized_variant)}"
                ),
            }
        )

    return snapshot_run_specs


def _format_embedding_layer_label(embedding_layer_index: int) -> str:
    """Return a notebook-friendly label for one requested embedding layer."""
    normalized_index = normalize_embedding_layer_index(embedding_layer_index)
    if normalized_index == 0:
        return "Layer 0 (input embeddings)"
    if normalized_index == -1:
        return "Layer -1 (final layer)"
    if normalized_index == -2:
        return "Layer -2 (penultimate layer)"
    return f"Layer {normalized_index}"


def build_layer_run_specs(
    registry_df: pd.DataFrame,
    *,
    checkpoints_dir: str | Path,
    tokenizer_family: str,
    experiment_name: str,
    model_variant: str = "pretrained_mlm",
    model_subdir: str = "best_model",
    embedding_layer_indices: Sequence[int] = (0, -1),
    classifier_mlm_run_label: str = "cls_lr2e-5_ep10_bs16_mlm",
    classifier_random_run_label: str = "cls_lr2e-5_ep10_bs16_randominit",
    only_fresh_runs: bool = True,
    only_tested_runs: bool = True,
) -> list[dict[str, Any]]:
    """Build ordered run specs for one fixed model directory across embedding layers."""
    normalized_variant = str(model_variant).strip()
    if normalized_variant not in SUPPORTED_MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported model_variant {model_variant!r}. "
            f"Choose from {SUPPORTED_MODEL_VARIANTS}."
        )

    normalized_layer_indices: list[int] = []
    for one_index in embedding_layer_indices:
        normalized_index = normalize_embedding_layer_index(one_index)
        if normalized_index in normalized_layer_indices:
            continue
        normalized_layer_indices.append(normalized_index)
    if not normalized_layer_indices:
        raise ValueError("At least one embedding layer index is required.")

    row = resolve_single_registry_run(
        registry_df,
        tokenizer_family=tokenizer_family,
        experiment_name=experiment_name,
        only_fresh_runs=only_fresh_runs,
        only_tested_runs=only_tested_runs,
    )
    base_run_spec = _build_one_registry_run_spec(
        row,
        model_variant=normalized_variant,
        classifier_mlm_run_label=classifier_mlm_run_label,
        classifier_random_run_label=classifier_random_run_label,
    )
    model_dir = resolve_embedding_model_dir(
        checkpoints_dir=checkpoints_dir,
        tokenizer_family=tokenizer_family,
        experiment_name=experiment_name,
        model_variant=normalized_variant,
        classifier_mlm_run_label=classifier_mlm_run_label,
        classifier_random_run_label=classifier_random_run_label,
        model_subdir=model_subdir,
    )

    layer_run_specs: list[dict[str, Any]] = []
    for layer_order, layer_index in enumerate(normalized_layer_indices):
        layer_label = _format_embedding_layer_label(layer_index)
        layer_run_specs.append(
            {
                **base_run_spec,
                "model_dir": str(model_dir),
                "model_subdir": str(model_subdir),
                "comparison_axis_name": "layer",
                "comparison_label": layer_label,
                "layer_label": layer_label,
                "embedding_layer_index": int(layer_index),
                "snapshot_label": layer_label,
                "snapshot_model_subdir": str(model_subdir),
                "snapshot_order": int(layer_order),
                "display_label": (
                    f"{MODEL_VARIANT_LABELS.get(normalized_variant, normalized_variant)}"
                    f" | {layer_label}"
                ),
                "comparison_group_label": (
                    f"{tokenizer_family} | {base_run_spec['architecture_label']} | "
                    f"{MODEL_VARIANT_LABELS.get(normalized_variant, normalized_variant)} | {model_subdir}"
                ),
            }
        )

    return layer_run_specs


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
        "class_summary_path": results_dir / "class_summary.csv",
        "edge_case_summary_path": results_dir / "probe_edge_case_summary.csv",
        "edge_case_detail_path": results_dir / "probe_edge_case_details.csv",
        "split_metrics_path": results_dir / "split_metrics.csv",
        "train_summary_path": results_dir / "train_summary.csv",
        "val_summary_path": results_dir / "val_summary.csv",
        "test_summary_path": results_dir / "test_summary.csv",
        "train_plot_path": results_dir / "train_metric_grid.png",
        "val_plot_path": results_dir / "val_metric_grid.png",
        "test_plot_path": results_dir / "test_metric_grid.png",
        "val_roc_plot_path": results_dir / "val_roc_curve.png",
        "test_roc_plot_path": results_dir / "test_roc_curve.png",
        "val_pr_plot_path": results_dir / "val_precision_recall_curve.png",
        "test_pr_plot_path": results_dir / "test_precision_recall_curve.png",
        "val_confusion_plot_path": results_dir / "val_confusion_matrix_grid.png",
        "test_confusion_plot_path": results_dir / "test_confusion_matrix_grid.png",
        "val_probability_plot_path": results_dir / "val_probability_histogram_grid.png",
        "test_probability_plot_path": results_dir / "test_probability_histogram_grid.png",
        "test_snapshot_progression_plot_path": results_dir / "test_snapshot_metric_progression.png",
        "snapshot_umap_coordinates_path": results_dir / "snapshot_umap_coordinates.csv",
        "snapshot_umap_plot_path": results_dir / "snapshot_umap_highlight_grid.png",
        "highlight_accession_table_path": results_dir / "highlight_accession_positions.csv",
        "highlight_similarity_table_path": results_dir / "highlight_accession_pairwise_similarity.csv",
        "highlight_neighbor_table_path": results_dir / "highlight_accession_neighbors.csv",
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
    """Backward-compatible wrapper for the original binary N-glycan probe."""
    probe_bundle = prepare_logreg_probe_dataframe(
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        test_csv_path=test_csv_path,
        label_vocabulary_path=label_vocabulary_path,
        splits_to_include=splits_to_include,
        probe_target_mode="n_glycan_binary",
        exclude_unlabeled_rows=exclude_unlabeled_rows,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    return probe_bundle["annotated_probe_df"], probe_bundle["label_vocabulary_df"]


def _normalize_probe_label_source(probe_label_source: str) -> str:
    """Return one validated notebook-14 probe-label source name."""

    normalized_source = str(probe_label_source).strip()
    if normalized_source not in SUPPORTED_PROBE_LABEL_SOURCES:
        raise ValueError(
            f"Unsupported probe_label_source {probe_label_source!r}. "
            f"Choose from {SUPPORTED_PROBE_LABEL_SOURCES}."
        )
    return normalized_source


def _normalize_structural_contradiction_policy(structural_contradiction_policy: str) -> str:
    """Return one validated structural contradiction-handling policy."""

    normalized_policy = str(structural_contradiction_policy).strip()
    if normalized_policy not in SUPPORTED_STRUCTURAL_CONTRADICTION_POLICIES:
        raise ValueError(
            f"Unsupported structural_contradiction_policy {structural_contradiction_policy!r}. "
            f"Choose from {SUPPORTED_STRUCTURAL_CONTRADICTION_POLICIES}."
        )
    return normalized_policy


def _normalize_probe_target_mode(probe_target_mode: str) -> str:
    """Return one validated notebook-14 probe-target mode."""
    normalized_mode = str(probe_target_mode).strip()
    if normalized_mode not in SUPPORTED_PROBE_TARGET_MODES:
        raise ValueError(
            f"Unsupported probe_target_mode {probe_target_mode!r}. "
            f"Choose from {SUPPORTED_PROBE_TARGET_MODES}."
        )
    return normalized_mode


def _normalize_subclass_categories(subclass_categories: Sequence[str] | None) -> tuple[str, ...]:
    """Return ordered, de-duplicated N-glycan subclass labels."""
    normalized_categories: list[str] = []
    for category_name in subclass_categories or DEFAULT_N_GLYCAN_SUBCLASS_CATEGORIES:
        cleaned_name = str(category_name).strip()
        if cleaned_name and cleaned_name not in normalized_categories:
            normalized_categories.append(cleaned_name)
    if not normalized_categories:
        raise ValueError("At least one N-glycan subclass category is required.")
    return tuple(normalized_categories)


def _normalize_subtype_label_text(label_name: str) -> str:
    """Return one normalized subtype-label string for keyword matching."""
    normalized_label = str(label_name).strip().lower().replace("_", " ")
    normalized_label = normalized_label.replace("-", " ")
    normalized_label = re.sub(r"\s+", " ", normalized_label)
    return normalized_label


def _infer_n_glycan_subclass_matches(label_values: Sequence[str]) -> list[str]:
    """Return each broad N-glycan subclass keyword group matched by the labels."""
    normalized_labels = [
        _normalize_subtype_label_text(label_name)
        for label_name in label_values
        if str(label_name).strip()
    ]
    matched_categories: list[str] = []
    for category_name, keywords in N_GLYCAN_SUBCLASS_KEYWORDS.items():
        if any(any(keyword in label_name for keyword in keywords) for label_name in normalized_labels):
            matched_categories.append(category_name)
    return matched_categories


def _prepare_base_probe_dataframe(
    *,
    train_csv_path: str | Path,
    val_csv_path: str | Path,
    test_csv_path: str | Path,
    label_vocabulary_path: str | Path,
    splits_to_include: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load notebook-09 outputs and add shared notebook-14 annotation columns."""
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
    annotated_df["has_multiple_labels"] = annotated_df["num_labels"].gt(1)
    annotated_df[N_GLYCAN_SUBCLASS_MATCHES_COLUMN] = annotated_df[LABEL_LIST_COLUMN].map(_infer_n_glycan_subclass_matches)
    annotated_df[N_GLYCAN_SUBCLASS_MATCH_COUNT_COLUMN] = annotated_df[N_GLYCAN_SUBCLASS_MATCHES_COLUMN].map(len)
    annotated_df[N_GLYCAN_SUBCLASS_COLUMN] = annotated_df[N_GLYCAN_SUBCLASS_MATCHES_COLUMN].map(
        lambda matches: matches[0] if len(matches) == 1 else ""
    )
    return annotated_df.reset_index(drop=True), label_vocabulary_df


def _build_probe_edge_case_summary(
    *,
    full_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    probe_target_mode: str,
    subclass_categories: Sequence[str],
) -> pd.DataFrame:
    """Summarize notebook-14 edge cases so probe results stay interpretable."""
    summary_rows: list[dict[str, object]] = []
    supported_categories = set(subclass_categories)
    ordered_splits = ["all", *STANDARD_PROBE_SPLITS]

    for split_name in ordered_splits:
        if split_name == "all":
            split_full_df = full_df.copy()
            split_probe_df = probe_df.copy()
            split_label = "all"
        else:
            split_full_df = full_df.loc[full_df[SPLIT_COLUMN].map(str).eq(split_name)].copy()
            split_probe_df = probe_df.loc[probe_df[SPLIT_COLUMN].map(str).eq(split_name)].copy()
            split_label = split_name

        metric_rows = [
            ("rows_available_after_split_filter", len(split_full_df)),
            ("rows_kept_for_probe", len(split_probe_df)),
            ("unlabeled_rows", int((~split_full_df["is_labeled_row"]).sum())),
            ("rows_with_multiple_subtype_labels", int(split_full_df["has_multiple_labels"].sum())),
            ("mixed_n_o_rows", int(split_full_df["n_o_category"].map(str).eq("Mixed N/O").sum())),
            ("n_glycan_rows", int(split_full_df["main_glycan_class"].map(str).eq("N-glycan").sum())),
        ]

        if probe_target_mode == "n_glycan_subclass_multiclass":
            n_glycan_mask = split_full_df["main_glycan_class"].map(str).eq("N-glycan")
            ambiguous_mask = n_glycan_mask & split_full_df[N_GLYCAN_SUBCLASS_MATCH_COUNT_COLUMN].gt(1)
            unmapped_mask = n_glycan_mask & split_full_df[N_GLYCAN_SUBCLASS_MATCH_COUNT_COLUMN].eq(0)
            unsupported_mask = (
                n_glycan_mask
                & split_full_df[N_GLYCAN_SUBCLASS_MATCH_COUNT_COLUMN].eq(1)
                & ~split_full_df[N_GLYCAN_SUBCLASS_COLUMN].isin(supported_categories)
            )
            supported_single_mask = (
                n_glycan_mask
                & split_full_df[N_GLYCAN_SUBCLASS_MATCH_COUNT_COLUMN].eq(1)
                & split_full_df[N_GLYCAN_SUBCLASS_COLUMN].isin(supported_categories)
            )
            metric_rows.extend(
                [
                    ("n_glycan_rows_with_supported_single_subclass", int(supported_single_mask.sum())),
                    ("n_glycan_rows_with_multiple_supported_subclasses", int(ambiguous_mask.sum())),
                    ("n_glycan_rows_with_no_supported_subclass_match", int(unmapped_mask.sum())),
                    ("n_glycan_rows_with_rare_or_excluded_subclass", int(unsupported_mask.sum())),
                ]
            )

        for metric_name, metric_value in metric_rows:
            summary_rows.append(
                {
                    "split": split_label,
                    "edge_case_metric": str(metric_name),
                    "count": int(metric_value),
                }
            )

    return pd.DataFrame(summary_rows)


def _build_probe_edge_case_detail_table(
    *,
    full_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    probe_target_mode: str,
    subclass_categories: Sequence[str],
) -> pd.DataFrame:
    """Return one detailed table for rows that deserve manual inspection."""
    detail_df = full_df.copy()
    detail_df["is_kept_for_probe"] = False
    if ACCESSION_COLUMN in probe_df.columns and SEQUENCE_COLUMN in probe_df.columns and not probe_df.empty:
        kept_keys = set(
            zip(
                probe_df[ACCESSION_COLUMN].fillna("").map(str),
                probe_df[SEQUENCE_COLUMN].fillna("").map(str),
                probe_df[SPLIT_COLUMN].fillna("").map(str),
            )
        )
        detail_df["is_kept_for_probe"] = [
            (str(accession), str(sequence), str(split_name)) in kept_keys
            for accession, sequence, split_name in zip(
                detail_df[ACCESSION_COLUMN].fillna("").map(str),
                detail_df[SEQUENCE_COLUMN].fillna("").map(str),
                detail_df[SPLIT_COLUMN].fillna("").map(str),
            )
        ]

    detail_df[SUBCLASS_EXCLUSION_REASON_COLUMN] = ""
    supported_categories = set(subclass_categories)

    if probe_target_mode == "n_glycan_binary":
        detail_df.loc[~detail_df["is_labeled_row"], SUBCLASS_EXCLUSION_REASON_COLUMN] = "unlabeled_row_kept_as_negative"
        detail_df.loc[
            detail_df["n_o_category"].map(str).eq("Mixed N/O"),
            SUBCLASS_EXCLUSION_REASON_COLUMN,
        ] = "mixed_n_o_labels"
    else:
        non_n_mask = ~detail_df["main_glycan_class"].map(str).eq("N-glycan")
        ambiguous_mask = detail_df["main_glycan_class"].map(str).eq("N-glycan") & detail_df[N_GLYCAN_SUBCLASS_MATCH_COUNT_COLUMN].gt(1)
        unmapped_mask = detail_df["main_glycan_class"].map(str).eq("N-glycan") & detail_df[N_GLYCAN_SUBCLASS_MATCH_COUNT_COLUMN].eq(0)
        unsupported_mask = (
            detail_df["main_glycan_class"].map(str).eq("N-glycan")
            & detail_df[N_GLYCAN_SUBCLASS_MATCH_COUNT_COLUMN].eq(1)
            & ~detail_df[N_GLYCAN_SUBCLASS_COLUMN].isin(supported_categories)
        )
        detail_df.loc[non_n_mask, SUBCLASS_EXCLUSION_REASON_COLUMN] = "not_n_glycan_row"
        detail_df.loc[unmapped_mask, SUBCLASS_EXCLUSION_REASON_COLUMN] = "no_supported_n_glycan_subclass_match"
        detail_df.loc[ambiguous_mask, SUBCLASS_EXCLUSION_REASON_COLUMN] = "multiple_supported_n_glycan_subclasses"
        detail_df.loc[unsupported_mask, SUBCLASS_EXCLUSION_REASON_COLUMN] = "rare_or_excluded_n_glycan_subclass"

    detail_df = detail_df.loc[
        detail_df["has_multiple_labels"]
        | detail_df["n_o_category"].map(str).eq("Mixed N/O")
        | detail_df[SUBCLASS_EXCLUSION_REASON_COLUMN].ne("")
    ].copy()

    if detail_df.empty:
        return detail_df

    detail_df[N_GLYCAN_SUBCLASS_MATCHES_COLUMN] = detail_df[N_GLYCAN_SUBCLASS_MATCHES_COLUMN].map(
        lambda values: " | ".join(values)
    )
    return _select_existing_columns(
        detail_df,
        [
            SPLIT_COLUMN,
            ACCESSION_COLUMN,
            SEQUENCE_COLUMN,
            "main_glycan_class",
            "n_o_category",
            "num_labels",
            "has_multiple_labels",
            "label_signature",
            N_GLYCAN_SUBCLASS_COLUMN,
            N_GLYCAN_SUBCLASS_MATCHES_COLUMN,
            N_GLYCAN_SUBCLASS_MATCH_COUNT_COLUMN,
            "is_kept_for_probe",
            SUBCLASS_EXCLUSION_REASON_COLUMN,
        ],
    )


def prepare_logreg_probe_dataframe(
    *,
    train_csv_path: str | Path,
    val_csv_path: str | Path,
    test_csv_path: str | Path,
    label_vocabulary_path: str | Path,
    splits_to_include: Sequence[str] | None = None,
    probe_target_mode: str = "n_glycan_binary",
    exclude_unlabeled_rows: bool = False,
    positive_label: str = DEFAULT_POSITIVE_LABEL,
    negative_label: str = DEFAULT_NEGATIVE_LABEL,
    subclass_categories: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load notebook-09 outputs and derive one notebook-14 probe target table."""
    normalized_mode = _normalize_probe_target_mode(probe_target_mode)
    normalized_subclass_categories = _normalize_subclass_categories(subclass_categories)
    annotated_df, label_vocabulary_df = _prepare_base_probe_dataframe(
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        test_csv_path=test_csv_path,
        label_vocabulary_path=label_vocabulary_path,
        splits_to_include=splits_to_include,
    )
    full_df = annotated_df.copy()

    if normalized_mode == "n_glycan_binary":
        probe_df = annotated_df.copy()
        if exclude_unlabeled_rows:
            probe_df = probe_df.loc[probe_df["is_labeled_row"]].copy()
        probe_df[TARGET_COLUMN] = probe_df["main_glycan_class"].eq("N-glycan").astype(int)
        probe_df[TARGET_LABEL_COLUMN] = probe_df[TARGET_COLUMN].map(
            lambda value: positive_label if int(value) == 1 else negative_label
        )
        probe_df[TARGET_KIND_COLUMN] = "binary"
        probe_target_name = DEFAULT_BINARY_TARGET_NAME
    else:
        probe_df = annotated_df.loc[
            annotated_df["main_glycan_class"].map(str).eq("N-glycan")
        ].copy()
        probe_df = probe_df.loc[
            probe_df[N_GLYCAN_SUBCLASS_MATCH_COUNT_COLUMN].eq(1)
            & probe_df[N_GLYCAN_SUBCLASS_COLUMN].isin(normalized_subclass_categories)
        ].copy()
        subclass_code_lookup = {
            subclass_name: subclass_index
            for subclass_index, subclass_name in enumerate(normalized_subclass_categories)
        }
        probe_df[TARGET_LABEL_COLUMN] = probe_df[N_GLYCAN_SUBCLASS_COLUMN].map(str)
        probe_df[TARGET_COLUMN] = probe_df[TARGET_LABEL_COLUMN].map(subclass_code_lookup).astype(int)
        probe_df[TARGET_KIND_COLUMN] = "multiclass"
        probe_target_name = DEFAULT_SUBCLASS_TARGET_NAME

    probe_df["probe_target_mode"] = normalized_mode
    probe_df["probe_target_name"] = probe_target_name
    probe_df = probe_df.reset_index(drop=True)

    target_summary_df = summarize_probe_target(probe_df)
    class_summary_df = summarize_main_glycan_class_by_split(probe_df)
    edge_case_summary_df = _build_probe_edge_case_summary(
        full_df=full_df,
        probe_df=probe_df,
        probe_target_mode=normalized_mode,
        subclass_categories=normalized_subclass_categories,
    )
    edge_case_detail_df = _build_probe_edge_case_detail_table(
        full_df=full_df,
        probe_df=probe_df,
        probe_target_mode=normalized_mode,
        subclass_categories=normalized_subclass_categories,
    )

    return {
        "annotated_probe_df": probe_df,
        "full_annotated_df": full_df,
        "label_vocabulary_df": label_vocabulary_df,
        "target_summary_df": target_summary_df,
        "class_summary_df": class_summary_df,
        "edge_case_summary_df": edge_case_summary_df,
        "edge_case_detail_df": edge_case_detail_df,
        "probe_target_mode": normalized_mode,
        "probe_target_name": probe_target_name,
        "subclass_categories": normalized_subclass_categories,
    }


def load_saved_probe_dataframe_bundle(
    *,
    probe_rows_path: str | Path,
    edge_case_summary_path: str | Path | None = None,
    edge_case_detail_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load one previously prepared probe dataset and its edge-case tables."""

    probe_df = pd.read_csv(require_existing_path(probe_rows_path, "Prepared probe rows CSV")).copy()
    required_columns = {
        ACCESSION_COLUMN,
        SEQUENCE_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        TARGET_LABEL_COLUMN,
        TARGET_KIND_COLUMN,
    }
    missing_columns = sorted(required_columns - set(probe_df.columns))
    if missing_columns:
        raise ValueError(
            "Prepared probe rows CSV is missing required columns: "
            f"{missing_columns}"
        )

    probe_df[TARGET_COLUMN] = probe_df[TARGET_COLUMN].astype(int)
    probe_df[TARGET_LABEL_COLUMN] = probe_df[TARGET_LABEL_COLUMN].fillna("").map(str)
    probe_df[TARGET_KIND_COLUMN] = probe_df[TARGET_KIND_COLUMN].fillna("").map(str)

    edge_case_summary_df = (
        pd.read_csv(require_existing_path(edge_case_summary_path, "Probe edge-case summary CSV")).copy()
        if edge_case_summary_path is not None
        else pd.DataFrame()
    )
    edge_case_detail_df = (
        pd.read_csv(require_existing_path(edge_case_detail_path, "Probe edge-case detail CSV")).copy()
        if edge_case_detail_path is not None
        else pd.DataFrame()
    )

    probe_target_name = ""
    if "probe_target_name" in probe_df.columns:
        probe_target_name_values = [
            str(value).strip()
            for value in probe_df["probe_target_name"].dropna().tolist()
            if str(value).strip()
        ]
        if probe_target_name_values:
            probe_target_name = probe_target_name_values[0]
    if not probe_target_name:
        target_kind = str(probe_df[TARGET_KIND_COLUMN].mode(dropna=True).iat[0]) if not probe_df.empty else ""
        probe_target_name = (
            DEFAULT_SUBCLASS_TARGET_NAME
            if target_kind == "multiclass"
            else DEFAULT_BINARY_TARGET_NAME
        )

    subclass_categories: tuple[str, ...] = tuple()
    if "n_glycan_subclass_label" in probe_df.columns:
        subclass_categories = tuple(
            label_name
            for label_name in probe_df["n_glycan_subclass_label"].dropna().map(str).tolist()
            if label_name
        )
        subclass_categories = tuple(dict.fromkeys(subclass_categories))

    return {
        "annotated_probe_df": probe_df.reset_index(drop=True),
        "full_annotated_df": probe_df.reset_index(drop=True),
        "label_vocabulary_df": pd.DataFrame(),
        "target_summary_df": summarize_probe_target(probe_df),
        "class_summary_df": summarize_main_glycan_class_by_split(probe_df),
        "edge_case_summary_df": edge_case_summary_df,
        "edge_case_detail_df": edge_case_detail_df,
        "probe_target_mode": (
            "n_glycan_subclass_multiclass"
            if probe_df[TARGET_KIND_COLUMN].eq("multiclass").any()
            else "n_glycan_binary"
        ),
        "probe_target_name": probe_target_name,
        "subclass_categories": subclass_categories,
    }


def prepare_probe_dataframe_for_notebook14(
    *,
    probe_label_source: str,
    train_csv_path: str | Path,
    val_csv_path: str | Path,
    test_csv_path: str | Path,
    label_vocabulary_path: str | Path,
    splits_to_include: Sequence[str] | None = None,
    probe_target_mode: str = "n_glycan_binary",
    exclude_unlabeled_rows: bool = False,
    positive_label: str = DEFAULT_POSITIVE_LABEL,
    negative_label: str = DEFAULT_NEGATIVE_LABEL,
    subclass_categories: Sequence[str] | None = None,
    structural_results_dir: str | Path | None = None,
    structural_run_label: str | None = None,
    project_root: str | Path | None = None,
    structural_contradiction_policy: str = DEFAULT_STRUCTURAL_CONTRADICTION_POLICY,
) -> dict[str, Any]:
    """Prepare one notebook-14 probe bundle from current labels or structural exports."""

    normalized_source = _normalize_probe_label_source(probe_label_source)
    normalized_mode = _normalize_probe_target_mode(probe_target_mode)
    normalized_contradiction_policy = _normalize_structural_contradiction_policy(
        structural_contradiction_policy
    )

    if normalized_source == CURRENT_PROJECT_LABEL_SOURCE:
        probe_bundle = prepare_logreg_probe_dataframe(
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path,
            test_csv_path=test_csv_path,
            label_vocabulary_path=label_vocabulary_path,
            splits_to_include=splits_to_include,
            probe_target_mode=normalized_mode,
            exclude_unlabeled_rows=exclude_unlabeled_rows,
            positive_label=positive_label,
            negative_label=negative_label,
            subclass_categories=subclass_categories,
        )
        probe_bundle["probe_label_source"] = normalized_source
        probe_bundle["structural_contradiction_policy"] = None
        return probe_bundle

    if structural_results_dir is None:
        if project_root is None or not str(project_root).strip():
            raise ValueError(
                "project_root is required when structural_results_dir is not provided."
            )
        if structural_run_label is None or not str(structural_run_label).strip():
            raise ValueError(
                "structural_run_label is required when structural_results_dir is not provided."
            )
        structural_output_paths = build_structural_classification_output_paths(
            project_root,
            run_label=str(structural_run_label),
        )
    else:
        structural_results_dir = Path(structural_results_dir)
        structural_output_paths = {
            "binary_probe_rows_path": structural_results_dir / "structural_binary_probe_rows.csv",
            "binary_probe_edge_case_summary_path": structural_results_dir / "structural_binary_probe_edge_case_summary.csv",
            "binary_probe_edge_case_detail_path": structural_results_dir / "structural_binary_probe_edge_case_details.csv",
            "binary_probe_without_contradictions_rows_path": structural_results_dir / "structural_binary_probe_rows_excluding_true_contradictions.csv",
            "binary_probe_without_contradictions_edge_case_summary_path": structural_results_dir / "structural_binary_probe_edge_case_summary_excluding_true_contradictions.csv",
            "binary_probe_without_contradictions_edge_case_detail_path": structural_results_dir / "structural_binary_probe_edge_case_details_excluding_true_contradictions.csv",
            "subclass_probe_rows_path": structural_results_dir / "structural_subclass_probe_rows.csv",
            "subclass_probe_edge_case_summary_path": structural_results_dir / "structural_subclass_probe_edge_case_summary.csv",
            "subclass_probe_edge_case_detail_path": structural_results_dir / "structural_subclass_probe_edge_case_details.csv",
            "subclass_probe_without_contradictions_rows_path": structural_results_dir / "structural_subclass_probe_rows_excluding_true_contradictions.csv",
            "subclass_probe_without_contradictions_edge_case_summary_path": structural_results_dir / "structural_subclass_probe_edge_case_summary_excluding_true_contradictions.csv",
            "subclass_probe_without_contradictions_edge_case_detail_path": structural_results_dir / "structural_subclass_probe_edge_case_details_excluding_true_contradictions.csv",
        }

    use_contradiction_filtered_structural_rows = (
        normalized_contradiction_policy == STRUCTURAL_EXCLUDE_TRUE_CONTRADICTIONS
    )
    if normalized_mode == "n_glycan_binary":
        probe_bundle = load_saved_probe_dataframe_bundle(
            probe_rows_path=(
                structural_output_paths["binary_probe_without_contradictions_rows_path"]
                if use_contradiction_filtered_structural_rows
                else structural_output_paths["binary_probe_rows_path"]
            ),
            edge_case_summary_path=(
                structural_output_paths["binary_probe_without_contradictions_edge_case_summary_path"]
                if use_contradiction_filtered_structural_rows
                else structural_output_paths["binary_probe_edge_case_summary_path"]
            ),
            edge_case_detail_path=(
                structural_output_paths["binary_probe_without_contradictions_edge_case_detail_path"]
                if use_contradiction_filtered_structural_rows
                else structural_output_paths["binary_probe_edge_case_detail_path"]
            ),
        )
    else:
        probe_bundle = load_saved_probe_dataframe_bundle(
            probe_rows_path=(
                structural_output_paths["subclass_probe_without_contradictions_rows_path"]
                if use_contradiction_filtered_structural_rows
                else structural_output_paths["subclass_probe_rows_path"]
            ),
            edge_case_summary_path=(
                structural_output_paths["subclass_probe_without_contradictions_edge_case_summary_path"]
                if use_contradiction_filtered_structural_rows
                else structural_output_paths["subclass_probe_edge_case_summary_path"]
            ),
            edge_case_detail_path=(
                structural_output_paths["subclass_probe_without_contradictions_edge_case_detail_path"]
                if use_contradiction_filtered_structural_rows
                else structural_output_paths["subclass_probe_edge_case_detail_path"]
            ),
        )

    probe_bundle["probe_target_mode"] = normalized_mode
    probe_bundle["probe_label_source"] = normalized_source
    probe_bundle["structural_contradiction_policy"] = normalized_contradiction_policy
    return probe_bundle


def summarize_probe_target(annotated_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the active notebook-14 target counts by split and label."""
    if annotated_df.empty:
        return pd.DataFrame(columns=[SPLIT_COLUMN, TARGET_LABEL_COLUMN, "count"])
    summary_df = (
        annotated_df.groupby([SPLIT_COLUMN, TARGET_LABEL_COLUMN], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values([SPLIT_COLUMN, "count", TARGET_LABEL_COLUMN], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return summary_df


def summarize_binary_target(annotated_df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for the original binary target summary."""
    return summarize_probe_target(annotated_df)


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
    embedding_layer_index: int | None,
    embedding_layer_indices: Sequence[int] | None = None,
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
        "embedding_layer_index": (
            int(embedding_layer_index) if embedding_layer_index is not None else None
        ),
        "embedding_layer_indices": (
            [int(layer_index) for layer_index in embedding_layer_indices]
            if embedding_layer_indices is not None
            else None
        ),
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
    explicit_model_dir = str(run_spec.get("model_dir", "")).strip()
    if explicit_model_dir:
        return Path(explicit_model_dir)
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
    slug_parts = [
        _slugify_text(run_spec["tokenizer_family"]),
        _slugify_text(run_spec["experiment_name"]),
        _slugify_text(run_spec["model_variant"]),
    ]
    comparison_label = (
        str(run_spec.get("comparison_label", "")).strip()
        or str(run_spec.get("layer_label", "")).strip()
        or str(run_spec.get("snapshot_label", "")).strip()
    )
    if comparison_label:
        slug_parts.append(_slugify_text(comparison_label))
    return "__".join(slug_parts)


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


def _build_per_run_output_paths(
    per_run_dir: str | Path,
    run_spec: Mapping[str, Any],
) -> dict[str, Path]:
    """Return the standard saved files for one compared model state."""
    run_dir = Path(per_run_dir) / build_run_slug(run_spec)
    return {
        "run_dir": run_dir,
        "metrics_path": run_dir / "split_metrics.csv",
        "predictions_path": run_dir / "prediction_table.csv",
        "config_path": run_dir / "run_spec.json",
    }


def _build_row_embedding_matrix(
    annotated_df: pd.DataFrame,
    *,
    model_dir: str | Path,
    pooling_strategy: str,
    embedding_layer_index: int,
    batch_size: int,
    max_length: int | None,
    device: str | None = None,
) -> np.ndarray:
    """Embed the unique sequences once, then map them back to dataframe rows."""
    model_dir = require_existing_path(model_dir, "Embedding model directory")
    normalized_pooling_strategy = normalize_pooling_strategy(pooling_strategy)
    normalized_embedding_layer_index = normalize_embedding_layer_index(embedding_layer_index)

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
            embedding_layer_index=normalized_embedding_layer_index,
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


def fit_logistic_probe(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    *,
    regularization_c: float = 1.0,
    class_weight: str | dict[int, float] | None = "balanced",
    max_iter: int = 2000,
    random_state: int = 42,
) -> Pipeline:
    """Fit one standardized logistic-regression probe."""
    num_target_classes = len(np.unique(train_targets))
    solver_name = "liblinear" if num_target_classes == 2 else "lbfgs"
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
                    solver=solver_name,
                    multi_class="auto",
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


def evaluate_logistic_probe(
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
    predicted_labels = probe_model.predict(split_embeddings).astype(int)
    predicted_probability_matrix = probe_model.predict_proba(split_embeddings)
    model_classes = probe_model.named_steps["logreg"].classes_
    num_target_classes = len(model_classes)

    metric_row = {
        **dict(run_spec),
        "split": str(split_name),
        "row_count": int(len(split_df)),
        "threshold": float(probability_threshold),
        "accuracy": float(accuracy_score(target_values, predicted_labels)),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, target_values, predicted_labels),
        "target_kind": "binary" if num_target_classes == 2 else "multiclass",
        "target_class_count": int(num_target_classes),
    }

    target_label_lookup = (
        split_df[[TARGET_COLUMN, TARGET_LABEL_COLUMN]]
        .drop_duplicates()
        .sort_values(TARGET_COLUMN, kind="stable")
    )
    code_to_label_lookup = {
        int(row[TARGET_COLUMN]): str(row[TARGET_LABEL_COLUMN])
        for _, row in target_label_lookup.iterrows()
    }
    ordered_target_labels = [code_to_label_lookup.get(int(class_code), str(class_code)) for class_code in model_classes]

    if num_target_classes == 2:
        positive_class_index = int(np.where(model_classes == 1)[0][0]) if 1 in model_classes else 1
        predicted_probabilities = predicted_probability_matrix[:, positive_class_index]
        predicted_binary_labels = (predicted_probabilities >= float(probability_threshold)).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            target_values,
            predicted_binary_labels,
            labels=[0, 1],
        ).ravel()
        metric_row.update(
            {
                "positive_count": int(target_values.sum()),
                "negative_count": int(len(target_values) - target_values.sum()),
                "positive_rate": float(target_values.mean()) if len(target_values) else float("nan"),
                "precision": _safe_metric(precision_score, target_values, predicted_binary_labels, zero_division=0),
                "recall": _safe_metric(recall_score, target_values, predicted_binary_labels, zero_division=0),
                "f1": _safe_metric(f1_score, target_values, predicted_binary_labels, zero_division=0),
                "macro_precision": _safe_metric(precision_score, target_values, predicted_binary_labels, zero_division=0),
                "macro_recall": _safe_metric(recall_score, target_values, predicted_binary_labels, zero_division=0),
                "macro_f1": _safe_metric(f1_score, target_values, predicted_binary_labels, zero_division=0),
                "weighted_f1": _safe_metric(f1_score, target_values, predicted_binary_labels, average="weighted", zero_division=0),
                "roc_auc": _safe_metric(roc_auc_score, target_values, predicted_probabilities),
                "average_precision": _safe_metric(average_precision_score, target_values, predicted_probabilities),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "confusion_matrix_json": json.dumps([[int(tn), int(fp)], [int(fn), int(tp)]]),
                "target_label_order_json": json.dumps([code_to_label_lookup.get(0, "0"), code_to_label_lookup.get(1, "1")]),
            }
        )
    else:
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            target_values,
            predicted_labels,
            average="macro",
            zero_division=0,
        )
        weighted_f1 = f1_score(target_values, predicted_labels, average="weighted", zero_division=0)
        multiclass_confusion = confusion_matrix(target_values, predicted_labels, labels=model_classes)
        metric_row.update(
            {
                "positive_count": float("nan"),
                "negative_count": float("nan"),
                "positive_rate": float("nan"),
                "precision": float(macro_precision),
                "recall": float(macro_recall),
                "f1": float(macro_f1),
                "macro_precision": float(macro_precision),
                "macro_recall": float(macro_recall),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                "roc_auc": float("nan"),
                "average_precision": float("nan"),
                "confusion_matrix_json": json.dumps(multiclass_confusion.astype(int).tolist()),
                "target_label_order_json": json.dumps(ordered_target_labels),
            }
        )

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
    prediction_df["predicted_label"] = predicted_labels
    prediction_df["predicted_label_name"] = [
        code_to_label_lookup.get(int(label_code), str(label_code))
        for label_code in predicted_labels
    ]
    prediction_df["correct_prediction"] = (predicted_labels == target_values).astype(int)
    if num_target_classes == 2:
        prediction_df[PREDICTED_PROBABILITY_COLUMN] = predicted_probabilities
    else:
        prediction_df[PREDICTED_PROBABILITIES_JSON_COLUMN] = [
            json.dumps(
                {
                    code_to_label_lookup.get(int(class_code), str(class_code)): float(class_probability)
                    for class_code, class_probability in zip(model_classes, row_probabilities)
                },
                sort_keys=True,
            )
            for row_probabilities in predicted_probability_matrix
        ]
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
    per_run_paths = _build_per_run_output_paths(per_run_dir, run_spec)
    run_dir = per_run_paths["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(per_run_paths["metrics_path"], index=False)
    prediction_df.to_csv(per_run_paths["predictions_path"], index=False)
    write_json(per_run_paths["config_path"], stringify_path_values(dict(run_spec)))

    return per_run_paths


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

    plot_df["variant_order"] = plot_df["model_variant"].map(
        lambda value: MODEL_VARIANT_ORDER.index(value) if value in MODEL_VARIANT_ORDER else len(MODEL_VARIANT_ORDER)
    )
    plot_df["variant_label"] = plot_df["model_variant"].map(MODEL_VARIANT_LABELS).fillna(plot_df["model_variant"])
    plot_df["plot_color"] = plot_df["model_variant"].map(VARIANT_COLOR_LOOKUP).fillna("#666666")
    plot_df = plot_df.sort_values(
        ["variant_order", "tokenizer_family", "num_hidden_layers", "hidden_size", "attention_heads", "model_variant"],
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


def _iter_split_prediction_groups(
    prediction_df: pd.DataFrame,
    split_name: str,
) -> list[tuple[str, str, pd.DataFrame]]:
    """Return split-filtered prediction groups ordered by model state."""
    split_df = prediction_df.loc[prediction_df["split"] == str(split_name)].copy()
    if split_df.empty:
        return []

    grouped_frames: list[tuple[str, str, pd.DataFrame]] = []
    for model_variant in MODEL_VARIANT_ORDER:
        model_df = split_df.loc[split_df["model_variant"] == model_variant].copy()
        if model_df.empty:
            continue
        display_label = str(model_df["display_label"].iloc[0])
        grouped_frames.append((model_variant, display_label, model_df.reset_index(drop=True)))
    return grouped_frames


def _prediction_group_is_binary(model_df: pd.DataFrame) -> bool:
    """Return True when one prediction group carries binary-probe probabilities."""
    return PREDICTED_PROBABILITY_COLUMN in model_df.columns and model_df[TARGET_COLUMN].nunique() >= 2


def _plot_roc_curve_comparison(
    prediction_df: pd.DataFrame,
    *,
    split_name: str,
    output_path: str | Path,
) -> Path | None:
    """Save one split-level ROC comparison across available model states."""
    grouped_frames = _iter_split_prediction_groups(prediction_df, split_name)
    valid_groups = [
        (model_variant, display_label, model_df)
        for model_variant, display_label, model_df in grouped_frames
        if _prediction_group_is_binary(model_df)
    ]
    if not valid_groups:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 6))

    for model_variant, display_label, model_df in valid_groups:
        target_values = model_df[TARGET_COLUMN].to_numpy(dtype=int)
        predicted_probabilities = model_df[PREDICTED_PROBABILITY_COLUMN].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(target_values, predicted_probabilities)
        auc_value = _safe_metric(roc_auc_score, target_values, predicted_probabilities)
        axis.plot(
            fpr,
            tpr,
            linewidth=2.2,
            color=VARIANT_COLOR_LOOKUP.get(model_variant, "#666666"),
            label=f"{MODEL_VARIANT_LABELS.get(model_variant, display_label)} (AUC {auc_value:.3f})",
        )

    axis.plot([0, 1], [0, 1], linestyle="--", color="#8a8f98", linewidth=1.2, label="Chance")
    axis.set_title(f"{split_name.title()} ROC comparison")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.20)
    axis.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_precision_recall_comparison(
    prediction_df: pd.DataFrame,
    *,
    split_name: str,
    output_path: str | Path,
) -> Path | None:
    """Save one split-level precision-recall comparison across model states."""
    grouped_frames = _iter_split_prediction_groups(prediction_df, split_name)
    valid_groups = [
        (model_variant, display_label, model_df)
        for model_variant, display_label, model_df in grouped_frames
        if _prediction_group_is_binary(model_df)
    ]
    if not valid_groups:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 6))

    for model_variant, display_label, model_df in valid_groups:
        target_values = model_df[TARGET_COLUMN].to_numpy(dtype=int)
        predicted_probabilities = model_df[PREDICTED_PROBABILITY_COLUMN].to_numpy(dtype=float)
        precision_values, recall_values, _ = precision_recall_curve(target_values, predicted_probabilities)
        ap_value = _safe_metric(average_precision_score, target_values, predicted_probabilities)
        axis.plot(
            recall_values,
            precision_values,
            linewidth=2.2,
            color=VARIANT_COLOR_LOOKUP.get(model_variant, "#666666"),
            label=f"{MODEL_VARIANT_LABELS.get(model_variant, display_label)} (AP {ap_value:.3f})",
        )

    baseline_rate = float(valid_groups[0][2][TARGET_COLUMN].mean())
    axis.axhline(baseline_rate, linestyle="--", color="#8a8f98", linewidth=1.2, label="Positive-rate baseline")
    axis.set_title(f"{split_name.title()} precision-recall comparison")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.20)
    axis.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_confusion_matrix_grid(
    metrics_df: pd.DataFrame,
    *,
    split_name: str,
    output_path: str | Path,
) -> Path | None:
    """Save one confusion-matrix grid that compares available model states."""
    split_metrics_df = metrics_df.loc[metrics_df["split"] == str(split_name)].copy()
    if split_metrics_df.empty:
        return None

    split_metrics_df["variant_order"] = split_metrics_df["model_variant"].map(
        lambda value: MODEL_VARIANT_ORDER.index(value) if value in MODEL_VARIANT_ORDER else len(MODEL_VARIANT_ORDER)
    )
    split_metrics_df = split_metrics_df.sort_values(["variant_order", "display_label"], kind="stable").reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_panels = len(split_metrics_df)
    num_columns = min(2, num_panels)
    num_rows = int(np.ceil(num_panels / num_columns))
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(4.6 * num_columns, 4.8 * num_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for axis, row in zip(axes_flat, split_metrics_df.itertuples(index=False)):
        matrix = np.array(json.loads(str(row.confusion_matrix_json)), dtype=float)
        axis.imshow(matrix, cmap="Blues")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f"{int(matrix[row_index, column_index])}",
                    ha="center",
                    va="center",
                    color="#0f172a",
                    fontsize=11,
                    fontweight="bold",
                )
        target_labels = json.loads(str(getattr(row, "target_label_order_json", "[]")))
        if not target_labels:
            target_labels = [str(index_value) for index_value in range(matrix.shape[0])]
        axis.set_xticks(np.arange(len(target_labels)))
        axis.set_xticklabels(target_labels, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(target_labels)))
        axis.set_yticklabels(target_labels)
        metric_label = "F1" if str(getattr(row, "target_kind", "binary")) == "binary" else "Macro F1"
        f1_value = float(getattr(row, "f1", float("nan")))
        axis.set_title(
            f"{MODEL_VARIANT_LABELS.get(row.model_variant, row.model_variant)}\n"
            f"Acc {float(row.accuracy):.3f} | {metric_label} {f1_value:.3f}"
        )
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")

    for axis in axes_flat[num_panels:]:
        axis.axis("off")

    fig.suptitle(f"{split_name.title()} confusion matrices", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_probability_histogram_grid(
    prediction_df: pd.DataFrame,
    *,
    split_name: str,
    output_path: str | Path,
    probability_threshold: float,
) -> Path | None:
    """Save one grid of probability histograms split by true class."""
    grouped_frames = _iter_split_prediction_groups(prediction_df, split_name)
    grouped_frames = [
        (model_variant, display_label, model_df)
        for model_variant, display_label, model_df in grouped_frames
        if _prediction_group_is_binary(model_df)
    ]
    if not grouped_frames:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_panels = len(grouped_frames)
    num_columns = min(2, num_panels)
    num_rows = int(np.ceil(num_panels / num_columns))
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(4.8 * num_columns, 4.6 * num_rows),
        squeeze=False,
        sharey=True,
    )
    axes_flat = axes.ravel()

    bins = np.linspace(0.0, 1.0, 21)
    for axis, (model_variant, _, model_df) in zip(axes_flat, grouped_frames):
        negative_scores = model_df.loc[model_df[TARGET_COLUMN].eq(0), PREDICTED_PROBABILITY_COLUMN].to_numpy(dtype=float)
        positive_scores = model_df.loc[model_df[TARGET_COLUMN].eq(1), PREDICTED_PROBABILITY_COLUMN].to_numpy(dtype=float)
        axis.hist(
            negative_scores,
            bins=bins,
            alpha=0.70,
            color="#8da0b3",
            label="True 0",
        )
        axis.hist(
            positive_scores,
            bins=bins,
            alpha=0.70,
            color=VARIANT_COLOR_LOOKUP.get(model_variant, "#666666"),
            label="True 1",
        )
        axis.axvline(float(probability_threshold), linestyle="--", color="#111827", linewidth=1.2)
        axis.set_title(MODEL_VARIANT_LABELS.get(model_variant, model_variant))
        axis.set_xlabel("Predicted probability of the positive class")
        axis.set_ylabel("Row count")
        axis.set_xlim(0.0, 1.0)
        axis.grid(alpha=0.16)
        axis.legend(loc="upper center", frameon=False)

    for axis in axes_flat[num_panels:]:
        axis.axis("off")

    fig.suptitle(f"{split_name.title()} predicted-probability distributions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _normalize_highlight_accessions(highlight_accessions: Sequence[str] | None) -> list[str]:
    """Return highlight accessions in first-seen order without blanks."""
    normalized_accessions: list[str] = []
    for accession in highlight_accessions or ():
        cleaned_accession = str(accession).strip()
        if not cleaned_accession or cleaned_accession in normalized_accessions:
            continue
        normalized_accessions.append(cleaned_accession)
    return normalized_accessions


def _select_highlight_rows(
    annotated_df: pd.DataFrame,
    highlight_accessions: Sequence[str] | None,
) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    """Return one representative row index for each highlighted accession."""
    normalized_accessions = _normalize_highlight_accessions(highlight_accessions)
    if not normalized_accessions:
        return pd.DataFrame(), {}, []

    accession_series = annotated_df[ACCESSION_COLUMN].fillna("").map(str).map(str.strip)
    selected_rows: list[pd.Series] = []
    highlight_row_lookup: dict[str, int] = {}
    missing_accessions: list[str] = []

    for accession in normalized_accessions:
        matched_indices = accession_series.index[accession_series.eq(accession)].tolist()
        if not matched_indices:
            missing_accessions.append(accession)
            continue
        chosen_index = int(matched_indices[0])
        highlight_row_lookup[accession] = chosen_index
        selected_row = annotated_df.loc[chosen_index].copy()
        selected_row["highlight_occurrence_count"] = int(len(matched_indices))
        selected_rows.append(selected_row)

    highlight_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    return highlight_df, highlight_row_lookup, missing_accessions


def _plot_snapshot_metric_progression(
    metrics_df: pd.DataFrame,
    *,
    split_name: str,
    output_path: str | Path,
    metric_names: Sequence[str] = ("roc_auc", "average_precision", "f1", "balanced_accuracy"),
) -> Path | None:
    """Plot held-out metric trends across ordered snapshot labels."""
    plot_df = metrics_df.loc[metrics_df["split"] == str(split_name)].copy()
    if plot_df.empty or "snapshot_order" not in plot_df.columns:
        return None

    axis_name = "layer" if "layer_label" in plot_df.columns and plot_df["layer_label"].fillna("").map(str).str.strip().any() else "snapshot"
    axis_label_column = "layer_label" if axis_name == "layer" and "layer_label" in plot_df.columns else "snapshot_label"
    axis_title = "Embedding layer" if axis_name == "layer" else "Snapshot"
    axis_title_plural = "embedding layers" if axis_name == "layer" else "snapshots"
    plot_df["snapshot_order"] = plot_df["snapshot_order"].astype(int)
    plot_df[axis_label_column] = plot_df[axis_label_column].fillna("").map(str)
    plot_df = plot_df.sort_values(["comparison_group_label", "snapshot_order", "display_label"], kind="stable")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    group_column = "comparison_group_label" if "comparison_group_label" in plot_df.columns else "model_variant"
    group_values = plot_df[group_column].fillna("").map(str).unique().tolist()

    fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, max(7, 2.6 * len(metric_names))), sharex=True)
    if len(metric_names) == 1:
        axes = [axes]

    color_map = plt.get_cmap("tab10")
    group_colors = {group_name: color_map(index % color_map.N) for index, group_name in enumerate(group_values)}
    for axis, metric_name in zip(axes, metric_names):
        for group_name in group_values:
            group_df = plot_df.loc[plot_df[group_column].eq(group_name)].sort_values("snapshot_order", kind="stable")
            axis.plot(
                group_df["snapshot_order"],
                group_df[metric_name].to_numpy(dtype=float),
                marker="o",
                linewidth=2.2,
                color=group_colors[group_name],
                label=group_name,
            )
        axis.set_ylabel(metric_name)
        axis.set_title(f"{split_name.title()} {metric_name} across {axis_title_plural}")
        axis.grid(alpha=0.18)
        if metric_name in PLOT_UNIT_INTERVAL_METRICS:
            axis.set_ylim(0.0, 1.0)

    tick_df = plot_df.sort_values("snapshot_order", kind="stable").drop_duplicates("snapshot_order")
    axes[-1].set_xticks(tick_df["snapshot_order"].to_list())
    axes[-1].set_xticklabels(tick_df[axis_label_column].to_list(), rotation=25, ha="right")
    axes[-1].set_xlabel(axis_title)
    if len(group_values) > 1:
        axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _build_snapshot_umap_long_dataframe(
    annotated_df: pd.DataFrame,
    completed_run_specs: Sequence[Mapping[str, Any]],
    row_embeddings_by_run_slug: Mapping[str, np.ndarray],
    *,
    umap_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    umap_random_state: int,
) -> pd.DataFrame:
    """Build one long dataframe with anchor-based UMAP coordinates for each snapshot.

    The first ordered snapshot becomes the anchor space. Later snapshots are
    projected into that fitted space with ``transform`` so the panels are easier
    to compare directly. If two snapshots produce numerically identical row
    embeddings, the later snapshot reuses the earlier snapshot's coordinates
    instead of getting a second, potentially misleading UMAP layout.
    """
    if not completed_run_specs:
        return pd.DataFrame()

    ordered_specs = sorted(
        completed_run_specs,
        key=lambda run_spec: (
            int(run_spec.get("snapshot_order", 0)),
            str(run_spec.get("display_label", "")),
        ),
    )
    anchor_run_spec = ordered_specs[0]
    anchor_run_slug = build_run_slug(anchor_run_spec)
    anchor_embeddings = row_embeddings_by_run_slug[anchor_run_slug]
    anchor_coordinates, reducer = compute_umap_projection(
        anchor_embeddings,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=umap_random_state,
        return_reducer=True,
    )

    snapshot_coordinates_by_slug: dict[str, np.ndarray] = {anchor_run_slug: anchor_coordinates}
    duplicate_snapshot_lookup: dict[str, str] = {}
    seen_run_slugs: list[str] = [anchor_run_slug]

    for run_spec in ordered_specs[1:]:
        run_slug = build_run_slug(run_spec)
        run_embeddings = row_embeddings_by_run_slug[run_slug]
        duplicate_source_slug = next(
            (
                seen_run_slug
                for seen_run_slug in seen_run_slugs
                if np.allclose(run_embeddings, row_embeddings_by_run_slug[seen_run_slug], atol=1e-8, rtol=1e-6)
            ),
            "",
        )
        if duplicate_source_slug:
            snapshot_coordinates_by_slug[run_slug] = snapshot_coordinates_by_slug[duplicate_source_slug].copy()
            duplicate_snapshot_lookup[run_slug] = duplicate_source_slug
        else:
            snapshot_coordinates_by_slug[run_slug] = transform_umap_projection(run_embeddings, reducer)
        seen_run_slugs.append(run_slug)

    coordinate_frames: list[pd.DataFrame] = []
    for run_spec in ordered_specs:
        run_slug = build_run_slug(run_spec)
        snapshot_coordinates = snapshot_coordinates_by_slug[run_slug]
        snapshot_df = build_umap_dataframe(annotated_df, snapshot_coordinates)
        snapshot_df["run_slug"] = run_slug
        snapshot_df["display_label"] = str(run_spec.get("display_label", ""))
        snapshot_df["model_variant"] = str(run_spec.get("model_variant", ""))
        snapshot_df["snapshot_label"] = str(run_spec.get("snapshot_label", ""))
        snapshot_df["comparison_label"] = str(run_spec.get("comparison_label", run_spec.get("snapshot_label", "")))
        snapshot_df["layer_label"] = str(run_spec.get("layer_label", ""))
        snapshot_df["comparison_axis_name"] = str(run_spec.get("comparison_axis_name", "snapshot"))
        snapshot_df["embedding_layer_index"] = run_spec.get("embedding_layer_index")
        snapshot_df["snapshot_order"] = int(run_spec.get("snapshot_order", 0))
        duplicate_source_slug = duplicate_snapshot_lookup.get(run_slug, "")
        duplicate_source_label = ""
        if duplicate_source_slug:
            duplicate_source_spec = next(
                (
                    one_run_spec
                    for one_run_spec in ordered_specs
                    if build_run_slug(one_run_spec) == duplicate_source_slug
                ),
                None,
            )
            if duplicate_source_spec is not None:
                duplicate_source_label = str(duplicate_source_spec.get("snapshot_label", ""))
        snapshot_df["is_duplicate_snapshot"] = bool(duplicate_source_slug)
        snapshot_df["duplicate_snapshot_of"] = duplicate_source_label
        coordinate_frames.append(snapshot_df)

    return pd.concat(coordinate_frames, ignore_index=True)


def _plot_snapshot_umap_grid(
    snapshot_umap_df: pd.DataFrame,
    *,
    output_path: str | Path,
    color_column: str,
    highlight_accessions: Sequence[str] | None = None,
    title: str = "Shared UMAP across snapshots",
) -> Path | None:
    """Render one grid of shared-UMAP panels with highlighted accessions."""
    if snapshot_umap_df.empty:
        return None
    if color_column not in snapshot_umap_df.columns:
        raise ValueError(f"UMAP color column {color_column!r} is missing from the snapshot dataframe.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_accessions = _normalize_highlight_accessions(highlight_accessions)
    ordered_snapshots = (
        snapshot_umap_df[["snapshot_order", "snapshot_label"]]
        .drop_duplicates()
        .sort_values("snapshot_order", kind="stable")
    )
    num_panels = len(ordered_snapshots)
    num_columns = min(2, num_panels)
    num_rows = int(np.ceil(num_panels / max(1, num_columns)))
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(7.2 * max(1, num_columns), 6.2 * max(1, num_rows)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.ravel()

    category_values = snapshot_umap_df[color_column].fillna("missing").map(str)
    ordered_categories = category_values.value_counts().index.tolist()
    category_color_map = plt.get_cmap("tab10")
    category_colors = {
        category_name: category_color_map(index % category_color_map.N)
        for index, category_name in enumerate(ordered_categories)
    }
    highlight_colors = {
        accession: DEFAULT_HIGHLIGHT_POINT_COLORS[index % len(DEFAULT_HIGHLIGHT_POINT_COLORS)]
        for index, accession in enumerate(ordered_accessions)
    }

    for axis, row in zip(axes_flat, ordered_snapshots.itertuples(index=False)):
        panel_df = snapshot_umap_df.loc[snapshot_umap_df["snapshot_order"].eq(int(row.snapshot_order))].copy()
        panel_df[color_column] = panel_df[color_column].fillna("missing").map(str)
        for category_name in ordered_categories:
            category_df = panel_df.loc[panel_df[color_column].eq(category_name)]
            if category_df.empty:
                continue
            axis.scatter(
                category_df["umap_1"],
                category_df["umap_2"],
                s=14,
                alpha=0.58,
                color=category_colors[category_name],
                edgecolors="none",
                label=category_name,
            )

        highlight_df = panel_df.loc[panel_df[ACCESSION_COLUMN].fillna("").map(str).isin(ordered_accessions)].copy()
        for highlight_row in highlight_df.itertuples(index=False):
            accession = str(getattr(highlight_row, ACCESSION_COLUMN))
            axis.scatter(
                [float(highlight_row.umap_1)],
                [float(highlight_row.umap_2)],
                s=95,
                color=highlight_colors.get(accession, "#111827"),
                edgecolors="white",
                linewidths=1.0,
                zorder=4,
            )
            axis.text(
                float(highlight_row.umap_1) + 0.15,
                float(highlight_row.umap_2) + 0.15,
                accession,
                fontsize=9,
                color=highlight_colors.get(accession, "#111827"),
                weight="bold",
                zorder=5,
            )

        duplicate_source_label = ""
        if "duplicate_snapshot_of" in panel_df.columns and not panel_df.empty:
            duplicate_source_values = [
                str(value).strip()
                for value in panel_df["duplicate_snapshot_of"].dropna().map(str).tolist()
                if str(value).strip()
            ]
            if duplicate_source_values:
                duplicate_source_label = duplicate_source_values[0]
        panel_title = str(row.snapshot_label)
        if duplicate_source_label:
            panel_title = f"{panel_title}\n(reuses {duplicate_source_label} coordinates)"
        axis.set_title(panel_title)
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.grid(alpha=0.16)

    for axis in axes_flat[num_panels:]:
        axis.axis("off")

    if ordered_categories:
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor=category_colors[name], label=name)
            for name in ordered_categories
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=min(4, len(handles)), frameon=False)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _build_highlight_accession_positions_table(
    snapshot_umap_df: pd.DataFrame,
    highlight_accessions: Sequence[str] | None,
) -> pd.DataFrame:
    """Return one long table of shared-UMAP coordinates for highlighted glycans."""
    ordered_accessions = _normalize_highlight_accessions(highlight_accessions)
    if snapshot_umap_df.empty or not ordered_accessions:
        return pd.DataFrame()

    highlight_df = snapshot_umap_df.loc[
        snapshot_umap_df[ACCESSION_COLUMN].fillna("").map(str).isin(ordered_accessions)
    ].copy()
    if highlight_df.empty:
        return highlight_df

    highlight_df["highlight_rank"] = highlight_df[ACCESSION_COLUMN].map(
        lambda accession: ordered_accessions.index(str(accession))
    )
    highlight_df = highlight_df.sort_values(
        ["highlight_rank", "snapshot_order", ACCESSION_COLUMN],
        kind="stable",
    ).reset_index(drop=True)
    return _select_existing_columns(
        highlight_df,
        [
            ACCESSION_COLUMN,
            "sequence",
            "split",
            "primary_subtype_label",
            "main_glycan_class",
            "comparison_label",
            "layer_label",
            "embedding_layer_index",
            "snapshot_label",
            "snapshot_order",
            "umap_1",
            "umap_2",
        ],
    )


def _build_highlight_similarity_table(
    annotated_df: pd.DataFrame,
    completed_run_specs: Sequence[Mapping[str, Any]],
    row_embeddings_by_run_slug: Mapping[str, np.ndarray],
    highlight_row_lookup: Mapping[str, int],
) -> pd.DataFrame:
    """Return pairwise cosine similarity rows for the highlighted glycans."""
    if not highlight_row_lookup or not completed_run_specs:
        return pd.DataFrame()

    ordered_accessions = list(highlight_row_lookup)
    similarity_rows: list[dict[str, Any]] = []
    for run_spec in completed_run_specs:
        run_slug = build_run_slug(run_spec)
        row_embeddings = row_embeddings_by_run_slug[run_slug]
        for accession_a, accession_b in combinations(ordered_accessions, 2):
            index_a = int(highlight_row_lookup[accession_a])
            index_b = int(highlight_row_lookup[accession_b])
            cosine_similarity = float(np.dot(row_embeddings[index_a], row_embeddings[index_b]))
            similarity_rows.append(
                {
                    "comparison_label": str(run_spec.get("comparison_label", run_spec.get("snapshot_label", ""))),
                    "layer_label": str(run_spec.get("layer_label", "")),
                    "embedding_layer_index": run_spec.get("embedding_layer_index"),
                    "snapshot_label": str(run_spec.get("snapshot_label", "")),
                    "snapshot_order": int(run_spec.get("snapshot_order", 0)),
                    "display_label": str(run_spec.get("display_label", "")),
                    "accession_a": accession_a,
                    "accession_b": accession_b,
                    "sequence_a": str(annotated_df.iloc[index_a][SEQUENCE_COLUMN]),
                    "sequence_b": str(annotated_df.iloc[index_b][SEQUENCE_COLUMN]),
                    "cosine_similarity": cosine_similarity,
                }
            )
    return pd.DataFrame(similarity_rows).sort_values(
        ["snapshot_order", "accession_a", "accession_b"],
        kind="stable",
    ).reset_index(drop=True)


def _build_highlight_neighbor_table(
    annotated_df: pd.DataFrame,
    completed_run_specs: Sequence[Mapping[str, Any]],
    row_embeddings_by_run_slug: Mapping[str, np.ndarray],
    highlight_row_lookup: Mapping[str, int],
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the nearest saved neighbors for each highlighted glycan."""
    if not highlight_row_lookup or not completed_run_specs:
        return pd.DataFrame()

    neighbor_rows: list[dict[str, Any]] = []
    accession_series = annotated_df[ACCESSION_COLUMN].fillna("").map(str)
    for run_spec in completed_run_specs:
        run_slug = build_run_slug(run_spec)
        row_embeddings = row_embeddings_by_run_slug[run_slug]
        for accession, row_index in highlight_row_lookup.items():
            query_embedding = row_embeddings[int(row_index)]
            similarity_scores = row_embeddings @ query_embedding
            ranking_indices = np.argsort(-similarity_scores)
            kept_count = 0
            for neighbor_index in ranking_indices:
                neighbor_accession = str(accession_series.iloc[int(neighbor_index)])
                if neighbor_accession == accession:
                    continue
                neighbor_row = annotated_df.iloc[int(neighbor_index)]
                neighbor_rows.append(
                    {
                        "comparison_label": str(run_spec.get("comparison_label", run_spec.get("snapshot_label", ""))),
                        "layer_label": str(run_spec.get("layer_label", "")),
                        "embedding_layer_index": run_spec.get("embedding_layer_index"),
                        "snapshot_label": str(run_spec.get("snapshot_label", "")),
                        "snapshot_order": int(run_spec.get("snapshot_order", 0)),
                        "query_accession": accession,
                        "neighbor_rank": int(kept_count + 1),
                        "neighbor_accession": neighbor_accession,
                        "neighbor_sequence": str(neighbor_row[SEQUENCE_COLUMN]),
                        "neighbor_split": str(neighbor_row[SPLIT_COLUMN]),
                        "neighbor_main_glycan_class": str(neighbor_row.get("main_glycan_class", "")),
                        "cosine_similarity": float(similarity_scores[int(neighbor_index)]),
                    }
                )
                kept_count += 1
                if kept_count >= int(top_n):
                    break
    return pd.DataFrame(neighbor_rows).sort_values(
        ["snapshot_order", "query_accession", "neighbor_rank"],
        kind="stable",
    ).reset_index(drop=True)


def build_snapshot_progression_artifacts(
    *,
    annotated_df: pd.DataFrame,
    run_specs: Sequence[Mapping[str, Any]],
    output_paths: Mapping[str, Path],
    row_embeddings_by_run_slug: Mapping[str, np.ndarray],
    split_name: str = "test",
    metrics_df: pd.DataFrame | None = None,
    highlight_accessions: Sequence[str] | None = None,
    highlight_neighbor_count: int = 10,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.10,
    umap_metric: str = "cosine",
    umap_random_state: int = 42,
    umap_color_column: str = "main_glycan_class",
) -> dict[str, Any]:
    """Build the snapshot-progression plots and tables for notebook 14."""
    normalized_highlight_accessions = _normalize_highlight_accessions(highlight_accessions)
    completed_run_specs = [
        run_spec
        for run_spec in run_specs
        if build_run_slug(run_spec) in row_embeddings_by_run_slug
    ]
    comparison_axis_name = str(completed_run_specs[0].get("comparison_axis_name", "snapshot")) if completed_run_specs else "snapshot"
    comparison_title = "Embedding layer" if comparison_axis_name == "layer" else "Saved snapshot"
    comparison_title_plural = "embedding layers" if comparison_axis_name == "layer" else "saved snapshots"
    if not completed_run_specs:
        return {
            "snapshot_progression_plot_path": None,
            "snapshot_umap_plot_path": None,
            "snapshot_umap_df": pd.DataFrame(),
            "highlight_positions_df": pd.DataFrame(),
            "highlight_similarity_df": pd.DataFrame(),
            "highlight_neighbors_df": pd.DataFrame(),
            "requested_highlight_accessions": normalized_highlight_accessions,
            "found_highlight_accessions": [],
            "missing_highlight_accessions": normalized_highlight_accessions,
        }

    snapshot_progression_plot_path = None
    if metrics_df is not None:
        snapshot_progression_plot_path = _plot_snapshot_metric_progression(
            metrics_df,
            split_name=split_name,
            output_path=output_paths["test_snapshot_progression_plot_path"],
        )

    _, highlight_row_lookup, missing_highlight_accessions = _select_highlight_rows(
        annotated_df,
        normalized_highlight_accessions,
    )
    found_highlight_accessions = list(highlight_row_lookup)

    snapshot_umap_df = _build_snapshot_umap_long_dataframe(
        annotated_df,
        completed_run_specs,
        row_embeddings_by_run_slug,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        umap_random_state=umap_random_state,
    )
    snapshot_umap_df.to_csv(output_paths["snapshot_umap_coordinates_path"], index=False)
    snapshot_umap_plot_path = _plot_snapshot_umap_grid(
        snapshot_umap_df,
        output_path=output_paths["snapshot_umap_plot_path"],
        color_column=umap_color_column,
        highlight_accessions=found_highlight_accessions,
        title=f"Shared UMAP across compared {comparison_title_plural}",
    )

    highlight_positions_df = _build_highlight_accession_positions_table(
        snapshot_umap_df,
        found_highlight_accessions,
    )
    highlight_similarity_df = _build_highlight_similarity_table(
        annotated_df,
        completed_run_specs,
        row_embeddings_by_run_slug,
        highlight_row_lookup,
    )
    highlight_neighbors_df = _build_highlight_neighbor_table(
        annotated_df,
        completed_run_specs,
        row_embeddings_by_run_slug,
        highlight_row_lookup,
        top_n=highlight_neighbor_count,
    )
    highlight_positions_df.to_csv(output_paths["highlight_accession_table_path"], index=False)
    highlight_similarity_df.to_csv(output_paths["highlight_similarity_table_path"], index=False)
    highlight_neighbors_df.to_csv(output_paths["highlight_neighbor_table_path"], index=False)

    return {
        "snapshot_progression_plot_path": snapshot_progression_plot_path,
        "snapshot_umap_plot_path": snapshot_umap_plot_path,
        "snapshot_umap_df": snapshot_umap_df,
        "highlight_positions_df": highlight_positions_df,
        "highlight_similarity_df": highlight_similarity_df,
        "highlight_neighbors_df": highlight_neighbors_df,
        "requested_highlight_accessions": normalized_highlight_accessions,
        "found_highlight_accessions": found_highlight_accessions,
        "missing_highlight_accessions": missing_highlight_accessions,
        "comparison_axis_name": comparison_axis_name,
        "comparison_title": comparison_title,
        "comparison_title_plural": comparison_title_plural,
    }


def _render_dataframe_html(frame_df: pd.DataFrame) -> str:
    """Render one dataframe as a compact HTML table for saved reports."""
    if frame_df.empty:
        return "<p class='subtle'>No rows were available for this section.</p>"
    return frame_df.to_html(index=False, classes="report-table", border=0)


def _select_existing_columns(frame_df: pd.DataFrame, candidate_columns: Sequence[str]) -> pd.DataFrame:
    """Return one dataframe restricted to the requested columns when present."""
    selected_columns = [column for column in candidate_columns if column in frame_df.columns]
    if not selected_columns:
        return frame_df.copy()
    return frame_df.loc[:, selected_columns].copy()


def _build_public_manifest_table(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Return one public-safe manifest table without local filesystem paths."""
    return _select_existing_columns(
        manifest_df,
        [
            "display_label",
            "model_variant",
            "comparison_label",
            "layer_label",
            "embedding_layer_index",
            "snapshot_label",
            "model_dir_exists",
            "registry_run_status",
        ],
    )


def _build_public_metric_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return one public-safe split summary without local filesystem paths."""
    return _select_existing_columns(
        summary_df,
        [
            "display_label",
            "model_variant",
            "target_kind",
            "target_class_count",
            "comparison_label",
            "layer_label",
            "embedding_layer_index",
            "snapshot_label",
            "row_count",
            "positive_count",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
            "f1",
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
            "accuracy",
        ],
    )


def _build_public_skipped_table(skipped_df: pd.DataFrame) -> pd.DataFrame:
    """Return one public-safe skipped-run table without local filesystem paths."""
    return _select_existing_columns(
        skipped_df,
        [
            "display_label",
            "model_variant",
            "reason",
        ],
    )


def _build_public_copied_files_table(copied_files_df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact export-audit table without local absolute paths."""
    public_df = _select_existing_columns(
        copied_files_df,
        [
            "relative_path",
            "file_type",
        ],
    )
    if "relative_path" in public_df.columns:
        public_df = public_df.sort_values("relative_path", kind="stable").reset_index(drop=True)
    return public_df


def _build_image_data_uri(image_path: str | Path) -> str:
    """Return one base64 data URI for a saved local plot image."""
    image_path = require_existing_path(image_path, "Plot image")
    suffix = image_path.suffix.lower()
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
    }.get(suffix, "application/octet-stream")
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_image}"


def _render_plot_card(
    plot_path: str | Path | None,
    *,
    title: str,
) -> str:
    """Render one locally saved metric plot as an embedded image card."""
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

    image_src = escape(_build_image_data_uri(plot_path), quote=True)
    return (
        "<div class='plot-card'>"
        f"<h3>{escape(title)}</h3>"
        f"<img src='{image_src}' alt='{escape(title, quote=True)}'>"
        f"<p class='subtle'>Saved PNG: {escape(plot_path.name)}</p>"
        "</div>"
    )


def render_embedding_logreg_html_report(
    *,
    output_dir: str | Path,
    report_title: str,
    manifest_df: pd.DataFrame,
    target_summary_df: pd.DataFrame,
    class_summary_df: pd.DataFrame,
    edge_case_summary_df: pd.DataFrame | None = None,
    edge_case_detail_df: pd.DataFrame | None = None,
    train_summary_df: pd.DataFrame,
    val_summary_df: pd.DataFrame,
    test_summary_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    plot_paths: Mapping[str, Path],
    diagnostic_plot_paths: Mapping[str, Path | None],
    snapshot_analysis: Mapping[str, Any] | None = None,
) -> Path:
    """Render a saved HTML report for one notebook-14 comparison run."""
    output_dir = Path(output_dir)
    html_path = output_dir / "n_glycan_logistic_regression_report.html"
    combined_summary_df = pd.concat(
        [train_summary_df, val_summary_df, test_summary_df],
        ignore_index=True,
    )
    completed_run_count = (
        int(combined_summary_df["display_label"].nunique())
        if "display_label" in combined_summary_df.columns
        else 0
    )

    summary_cards = [
        ("Requested runs", len(manifest_df)),
        ("Existing model dirs", int(manifest_df["model_dir_exists"].sum()) if "model_dir_exists" in manifest_df else 0),
        ("Completed runs", completed_run_count),
        ("Skipped runs", len(skipped_df)),
    ]
    public_manifest_df = _build_public_manifest_table(manifest_df)
    public_test_summary_df = _build_public_metric_summary_table(test_summary_df)
    public_skipped_df = _build_public_skipped_table(skipped_df)
    public_edge_case_summary_df = _select_existing_columns(
        pd.DataFrame(edge_case_summary_df if edge_case_summary_df is not None else pd.DataFrame()),
        ["split", "edge_case_metric", "count"],
    )
    public_edge_case_detail_df = _select_existing_columns(
        pd.DataFrame(edge_case_detail_df if edge_case_detail_df is not None else pd.DataFrame()),
        [
            SPLIT_COLUMN,
            ACCESSION_COLUMN,
            "main_glycan_class",
            "n_o_category",
            "num_labels",
            "has_multiple_labels",
            "label_signature",
            N_GLYCAN_SUBCLASS_COLUMN,
            N_GLYCAN_SUBCLASS_MATCHES_COLUMN,
            "is_kept_for_probe",
            SUBCLASS_EXCLUSION_REASON_COLUMN,
        ],
    )
    snapshot_analysis = dict(snapshot_analysis or {})
    comparison_axis_name = str(snapshot_analysis.get("comparison_axis_name", "")).strip()
    if not comparison_axis_name:
        comparison_axis_name = (
            "layer"
            if "layer_label" in manifest_df.columns and manifest_df["layer_label"].fillna("").map(str).str.strip().any()
            else "snapshot"
        )
    comparison_title = "Embedding layer" if comparison_axis_name == "layer" else "Snapshot"
    comparison_title_plural = "embedding layers" if comparison_axis_name == "layer" else "snapshots"
    summary_cards_html = "".join(
        (
            "<div class='card'>"
            f"<h3>{escape(label)}</h3>"
            f"<p class='metric'>{value}</p>"
            "</div>"
        )
        for label, value in summary_cards
    )
    metric_grid_cards_html = ""
    if plot_paths.get("test"):
        metric_grid_cards_html = _render_plot_card(
            plot_paths.get("test"),
            title="Held-out test metric grid",
        )
    skipped_callout_html = ""
    if not public_skipped_df.empty:
        skipped_callout_html = (
            "<section class='callout warn'>"
            f"<h2>{escape(comparison_title)} warning</h2>"
            "<p>At least one requested model state did not resolve to a saved model folder. "
            "That state is excluded from the comparison plots below.</p>"
            f"<div class='table-wrap'>{_render_dataframe_html(public_skipped_df)}</div>"
            "</section>"
        )

    test_diagnostic_section_html = ""
    test_plot_keys = (
        "test_roc",
        "test_pr",
        "test_confusion",
        "test_probability",
    )
    if any(diagnostic_plot_paths.get(plot_key) for plot_key in test_plot_keys):
        test_cards = "".join(
            [
                _render_plot_card(
                    diagnostic_plot_paths.get("test_roc"),
                    title="Test ROC curve",
                ),
                _render_plot_card(
                    diagnostic_plot_paths.get("test_pr"),
                    title="Test precision-recall curve",
                ),
                _render_plot_card(
                    diagnostic_plot_paths.get("test_confusion"),
                    title="Test confusion matrices",
                ),
                _render_plot_card(
                    diagnostic_plot_paths.get("test_probability"),
                    title="Test probability distributions",
                ),
            ]
        )
        test_diagnostic_section_html = (
            "<section>"
            "<h2>Held-out test diagnostics</h2>"
            "<p class='subtle'>These plots show the final held-out test behavior for each compared embedding source.</p>"
            f"<div class='plot-grid diagnostics-grid'>{test_cards}</div>"
            "</section>"
        )

    test_summary_section_html = ""
    if not public_test_summary_df.empty:
        test_summary_section_html = (
            "<section>"
            "<h2>Held-out test summary</h2>"
            "<div class='table-wrap'>"
            f"{_render_dataframe_html(public_test_summary_df)}"
            "</div>"
            "</section>"
        )

    edge_case_section_html = ""
    if not public_edge_case_summary_df.empty or not public_edge_case_detail_df.empty:
        edge_case_section_html = (
            "<section>"
            "<h2>Probe edge cases</h2>"
            "<p class='subtle'>These tables flag rows that are unlabeled, carry multiple subtype labels, "
            "mix broad glycan families, or fall outside the kept N-glycan subclass buckets.</p>"
            f"<div class='table-wrap'>{_render_dataframe_html(public_edge_case_summary_df)}</div>"
            "<h3>Edge-case detail rows</h3>"
            f"<div class='table-wrap'>{_render_dataframe_html(public_edge_case_detail_df)}</div>"
            "</section>"
        )

    snapshot_section_html = ""
    snapshot_progression_plot_path = snapshot_analysis.get("snapshot_progression_plot_path")
    snapshot_umap_plot_path = snapshot_analysis.get("snapshot_umap_plot_path")
    highlight_positions_df = snapshot_analysis.get("highlight_positions_df", pd.DataFrame())
    highlight_similarity_df = snapshot_analysis.get("highlight_similarity_df", pd.DataFrame())
    highlight_neighbors_df = snapshot_analysis.get("highlight_neighbors_df", pd.DataFrame())
    requested_highlight_accessions = snapshot_analysis.get("requested_highlight_accessions", [])
    found_highlight_accessions = snapshot_analysis.get("found_highlight_accessions", [])
    missing_highlight_accessions = snapshot_analysis.get("missing_highlight_accessions", [])
    if snapshot_progression_plot_path or snapshot_umap_plot_path or not highlight_positions_df.empty:
        snapshot_cards = "".join(
            [
                _render_plot_card(
                    snapshot_progression_plot_path,
                    title=f"Held-out test {comparison_axis_name} progression",
                ),
                _render_plot_card(
                    snapshot_umap_plot_path,
                    title=f"Shared UMAP across compared {comparison_title_plural}",
                ),
            ]
        )
        highlight_positions_public_df = _select_existing_columns(
            highlight_positions_df,
            [
                ACCESSION_COLUMN,
                "sequence",
                "split",
                "main_glycan_class",
                "comparison_label",
                "layer_label",
                "embedding_layer_index",
                "snapshot_label",
                "umap_1",
                "umap_2",
            ],
        )
        highlight_similarity_public_df = _select_existing_columns(
            highlight_similarity_df,
            [
                "comparison_label",
                "layer_label",
                "embedding_layer_index",
                "snapshot_label",
                "accession_a",
                "accession_b",
                "cosine_similarity",
            ],
        )
        highlight_neighbors_public_df = _select_existing_columns(
            highlight_neighbors_df,
            [
                "comparison_label",
                "layer_label",
                "embedding_layer_index",
                "snapshot_label",
                "query_accession",
                "neighbor_rank",
                "neighbor_accession",
                "neighbor_main_glycan_class",
                "cosine_similarity",
            ],
        )
        highlight_status_html = ""
        if requested_highlight_accessions:
            highlight_status_lines = [
                f"Requested highlights: {escape(', '.join(map(str, requested_highlight_accessions)))}.",
                f"Found in active probe rows: {escape(', '.join(map(str, found_highlight_accessions)) if found_highlight_accessions else 'none')}.",
            ]
            if missing_highlight_accessions:
                highlight_status_lines.append(
                    "Skipped because they were not present in the active probe rows: "
                    f"{escape(', '.join(map(str, missing_highlight_accessions)))}."
                )
            highlight_status_html = (
                f"<p class='subtle'>{' '.join(highlight_status_lines)}</p>"
            )
        snapshot_section_html = (
            "<section>"
            f"<h2>{escape(comparison_title)} comparison</h2>"
            f"<p class='subtle'>These sections track how one model state changes across compared {escape(comparison_title_plural)} using the held-out test probe and a shared UMAP projection.</p>"
            f"{highlight_status_html}"
            f"<div class='plot-grid diagnostics-grid'>{snapshot_cards}</div>"
            "<h3>Highlighted accession positions</h3>"
            f"<div class='table-wrap'>{_render_dataframe_html(highlight_positions_public_df)}</div>"
            "<h3>Highlighted accession pairwise similarity</h3>"
            f"<div class='table-wrap'>{_render_dataframe_html(highlight_similarity_public_df)}</div>"
            "<h3>Highlighted accession nearest neighbors</h3>"
            f"<div class='table-wrap'>{_render_dataframe_html(highlight_neighbors_public_df)}</div>"
            "</section>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(report_title)}</title>
  <style>
    :root {{
      --ink: #16202a;
      --muted: #5b6674;
      --paper: #eef3f7;
      --card: #ffffff;
      --line: #d6dee7;
      --accent: #1f4f82;
      --accent-soft: #eaf2fb;
      --warn: #9a4f17;
      --warn-soft: #fff3e6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #ffffff 0%, rgba(255, 255, 255, 0) 32%),
        linear-gradient(180deg, #f7fafc 0%, var(--paper) 100%);
      font-family: "Aptos", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      line-height: 1.58;
    }}
    header {{
      padding: 34px 42px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(10px);
    }}
    h1, h2, h3 {{ line-height: 1.12; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    h2 {{ margin: 30px 0 12px; font-size: 24px; }}
    h3 {{ margin: 0 0 8px; font-size: 18px; }}
    p {{ margin: 0 0 12px; }}
    .subtle {{ color: var(--muted); }}
    .container {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 26px 40px 56px;
    }}
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
    .diagnostics-grid {{
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    }}
    .card, .plot-card {{
      background: rgba(255, 255, 255, 0.97);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 14px 36px rgba(27, 39, 51, 0.08);
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
    .callout {{
      margin-top: 24px;
      padding: 18px;
      border-radius: 20px;
      border: 1px solid var(--line);
      background: var(--accent-soft);
      box-shadow: 0 14px 36px rgba(27, 39, 51, 0.06);
    }}
    .callout.warn {{
      background: var(--warn-soft);
      border-color: #f0c89c;
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
      background: rgba(255, 255, 255, 0.97);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 14px 36px rgba(27, 39, 51, 0.08);
      padding: 12px;
    }}
    table.report-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: white;
    }}
    table.report-table th,
    table.report-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid #e6edf5;
      text-align: left;
      vertical-align: top;
    }}
    table.report-table th {{
      background: #f2f6fa;
      position: sticky;
      top: 0;
    }}
    code {{
      background: #edf3f9;
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
		      This report fits each logistic-regression probe on the training split and
		      compares the final held-out <code>test</code> performance across the
		      requested saved embedding sources. When snapshot progression is enabled,
		      the report also shows how one lineage changes across checkpoints.
		    </p>
	  </header>
  <main class="container">
    <section>
      <h2>Run overview</h2>
      <div class="summary-grid">{summary_cards_html}</div>
    </section>
    <section>
      <h2>Requested model states</h2>
      <div class="table-wrap">{_render_dataframe_html(public_manifest_df)}</div>
    </section>
    {skipped_callout_html}
    <section>
      <h2>Target balance</h2>
      <p class="subtle">Unlabeled rows remain in the negative class when the notebook keeps them.</p>
      <div class="table-wrap">{_render_dataframe_html(target_summary_df)}</div>
    </section>
    <section>
      <h2>Broad glycan-class counts</h2>
      <div class="table-wrap">{_render_dataframe_html(class_summary_df)}</div>
    </section>
		    <section>
		      <h2>Held-out test ranking</h2>
		      <p class="subtle">This summarizes the held-out test ranking across the compared model states.</p>
		      <div class="plot-grid">{metric_grid_cards_html}</div>
		    </section>
		    {test_diagnostic_section_html}
		    {test_summary_section_html}
            {edge_case_section_html}
            {snapshot_section_html}
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
                "file_type": source_path.suffix.lower().lstrip(".") or "no_suffix",
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

    copied_files_df = _build_public_copied_files_table(pd.DataFrame(copied_file_rows))
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

    if "snapshot_order" in summary_df.columns:
        summary_df = summary_df.sort_values(
            ["snapshot_order", "display_label"],
            kind="stable",
        ).reset_index(drop=True)
    else:
        preferred_sort_columns = [
            "roc_auc",
            "average_precision",
            "f1",
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
            "accuracy",
        ]
        available_sort_columns = [
            column_name
            for column_name in preferred_sort_columns
            if column_name in summary_df.columns and not summary_df[column_name].isna().all()
        ]
        if not available_sort_columns:
            available_sort_columns = ["display_label"]
        summary_df = summary_df.sort_values(
            available_sort_columns,
            ascending=[False] * len(available_sort_columns),
            kind="stable",
        ).reset_index(drop=True)
    return summary_df


def validate_probe_split_configuration(
    *,
    train_splits: Sequence[str],
    evaluation_splits: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate notebook-14 split settings against the standard project layout.

    Notebook 14 is intentionally built around the project's fixed
    ``train``/``val``/``test`` split names so the saved report layout stays
    predictable and consistent with the rest of the workflow.
    """
    normalized_train_splits = tuple(str(split_name).strip() for split_name in train_splits if str(split_name).strip())
    normalized_evaluation_splits = tuple(
        str(split_name).strip() for split_name in evaluation_splits if str(split_name).strip()
    )
    if not normalized_train_splits:
        raise ValueError("At least one train split is required for notebook 14.")
    if not normalized_evaluation_splits:
        raise ValueError("At least one evaluation split is required for notebook 14.")

    invalid_train_splits = sorted(set(normalized_train_splits).difference(STANDARD_PROBE_SPLITS))
    invalid_evaluation_splits = sorted(set(normalized_evaluation_splits).difference(STANDARD_PROBE_SPLITS))
    if invalid_train_splits or invalid_evaluation_splits:
        raise ValueError(
            "Notebook 14 expects the standard project split names only. "
            f"Supported splits: {STANDARD_PROBE_SPLITS}. "
            f"Invalid train_splits: {invalid_train_splits or 'none'}. "
            f"Invalid evaluation_splits: {invalid_evaluation_splits or 'none'}."
        )

    return normalized_train_splits, normalized_evaluation_splits


def run_embedding_logreg_suite(
    *,
    annotated_df: pd.DataFrame,
    run_specs: Sequence[Mapping[str, Any]],
    checkpoints_dir: str | Path,
    output_paths: Mapping[str, Path],
    target_summary_df: pd.DataFrame | None = None,
    edge_case_summary_df: pd.DataFrame | None = None,
    edge_case_detail_df: pd.DataFrame | None = None,
    pooling_strategy: str = "mean",
    embedding_layer_index: int | None = None,
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
    collect_row_embeddings: bool = False,
    build_snapshot_progression: bool = False,
    highlight_accessions: Sequence[str] | None = None,
    highlight_neighbor_count: int = 10,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.10,
    umap_metric: str = "cosine",
    umap_random_state: int = 42,
    umap_color_column: str = "main_glycan_class",
) -> dict[str, Any]:
    """Run the notebook-14 embedding comparison end to end."""
    normalized_pooling_strategy = normalize_pooling_strategy(pooling_strategy)
    normalized_embedding_layer_index = normalize_embedding_layer_index(embedding_layer_index)
    train_splits, evaluation_splits = validate_probe_split_configuration(
        train_splits=train_splits,
        evaluation_splits=evaluation_splits,
    )
    output_paths = {key: Path(value) for key, value in output_paths.items()}
    output_paths["per_run_dir"].mkdir(parents=True, exist_ok=True)
    manifest_df = build_run_manifest(
        run_specs=run_specs,
        checkpoints_dir=checkpoints_dir,
        model_subdir=model_subdir,
    )

    planned_output_paths = {
        "run_manifest_path": output_paths["run_manifest_path"],
        "skipped_runs_path": output_paths["skipped_runs_path"],
        "target_summary_path": output_paths["target_summary_path"],
        "class_summary_path": output_paths["class_summary_path"],
        "edge_case_summary_path": output_paths["edge_case_summary_path"],
        "edge_case_detail_path": output_paths["edge_case_detail_path"],
        "split_metrics_path": output_paths["split_metrics_path"],
        "train_summary_path": output_paths["train_summary_path"],
        "val_summary_path": output_paths["val_summary_path"],
        "test_summary_path": output_paths["test_summary_path"],
        "train_plot_path": output_paths["train_plot_path"],
        "val_plot_path": output_paths["val_plot_path"],
        "test_plot_path": output_paths["test_plot_path"],
        "val_roc_plot_path": output_paths["val_roc_plot_path"],
        "test_roc_plot_path": output_paths["test_roc_plot_path"],
        "val_pr_plot_path": output_paths["val_pr_plot_path"],
        "test_pr_plot_path": output_paths["test_pr_plot_path"],
        "val_confusion_plot_path": output_paths["val_confusion_plot_path"],
        "test_confusion_plot_path": output_paths["test_confusion_plot_path"],
        "val_probability_plot_path": output_paths["val_probability_plot_path"],
        "test_probability_plot_path": output_paths["test_probability_plot_path"],
        "test_snapshot_progression_plot_path": output_paths["test_snapshot_progression_plot_path"],
        "snapshot_umap_coordinates_path": output_paths["snapshot_umap_coordinates_path"],
        "snapshot_umap_plot_path": output_paths["snapshot_umap_plot_path"],
        "highlight_accession_table_path": output_paths["highlight_accession_table_path"],
        "highlight_similarity_table_path": output_paths["highlight_similarity_table_path"],
        "highlight_neighbor_table_path": output_paths["highlight_neighbor_table_path"],
        "html_report_path": output_paths["html_report_path"],
    }
    for run_spec in run_specs:
        per_run_paths = _build_per_run_output_paths(output_paths["per_run_dir"], run_spec)
        planned_output_paths.update(
            {
                f"{build_run_slug(run_spec)}__metrics": per_run_paths["metrics_path"],
                f"{build_run_slug(run_spec)}__predictions": per_run_paths["predictions_path"],
                f"{build_run_slug(run_spec)}__config": per_run_paths["config_path"],
            }
        )

    validate_output_paths(
        planned_output_paths,
        overwrite_existing_outputs=overwrite_existing_outputs,
    )

    if target_summary_df is None:
        target_summary_df = summarize_probe_target(annotated_df)
    target_summary_df.to_csv(output_paths["target_summary_path"], index=False)
    class_summary_df = summarize_main_glycan_class_by_split(annotated_df)
    class_summary_df.to_csv(output_paths["class_summary_path"], index=False)
    edge_case_summary_df = pd.DataFrame(edge_case_summary_df if edge_case_summary_df is not None else pd.DataFrame())
    edge_case_detail_df = pd.DataFrame(edge_case_detail_df if edge_case_detail_df is not None else pd.DataFrame())
    edge_case_summary_df.to_csv(output_paths["edge_case_summary_path"], index=False)
    edge_case_detail_df.to_csv(output_paths["edge_case_detail_path"], index=False)
    manifest_df.to_csv(output_paths["run_manifest_path"], index=False)

    all_metric_rows: list[dict[str, Any]] = []
    all_prediction_frames: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, Any]] = []
    row_embeddings_by_run_slug: dict[str, np.ndarray] = {}

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
        run_embedding_layer_index = normalize_embedding_layer_index(
            run_spec.get("embedding_layer_index", normalized_embedding_layer_index)
        )
        row_embeddings = _build_row_embedding_matrix(
            annotated_df=annotated_df,
            model_dir=model_dir,
            pooling_strategy=normalized_pooling_strategy,
            embedding_layer_index=run_embedding_layer_index,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        run_slug = build_run_slug(run_spec)
        if collect_row_embeddings or build_snapshot_progression:
            row_embeddings_by_run_slug[run_slug] = row_embeddings
        run_df = annotated_df.reset_index(drop=True).copy()
        train_mask = run_df[SPLIT_COLUMN].isin(train_splits)
        if not train_mask.any():
            raise ValueError(f"No train rows were found for run spec: {run_spec}")

        train_targets = run_df.loc[train_mask, TARGET_COLUMN].to_numpy(dtype=int)
        num_train_classes = len(np.unique(train_targets))
        if num_train_classes < 2:
            raise ValueError(
                "The train split for the notebook-14 probe only contains one class. "
                "Adjust the filtering settings before rerunning."
            )

        # Fit one simple linear probe on the train split only. The same fitted
        # probe is then carried unchanged into validation and test.
        probe_model = fit_logistic_probe(
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

            metric_row, prediction_df = evaluate_logistic_probe(
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
    combined_prediction_df = pd.concat(all_prediction_frames, ignore_index=True)

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

    diagnostic_plot_paths = {
        "val_roc": _plot_roc_curve_comparison(
            combined_prediction_df,
            split_name="val",
            output_path=output_paths["val_roc_plot_path"],
        ),
        "test_roc": _plot_roc_curve_comparison(
            combined_prediction_df,
            split_name="test",
            output_path=output_paths["test_roc_plot_path"],
        ),
        "val_pr": _plot_precision_recall_comparison(
            combined_prediction_df,
            split_name="val",
            output_path=output_paths["val_pr_plot_path"],
        ),
        "test_pr": _plot_precision_recall_comparison(
            combined_prediction_df,
            split_name="test",
            output_path=output_paths["test_pr_plot_path"],
        ),
        "val_confusion": _plot_confusion_matrix_grid(
            metrics_df,
            split_name="val",
            output_path=output_paths["val_confusion_plot_path"],
        ),
        "test_confusion": _plot_confusion_matrix_grid(
            metrics_df,
            split_name="test",
            output_path=output_paths["test_confusion_plot_path"],
        ),
        "val_probability": _plot_probability_histogram_grid(
            combined_prediction_df,
            split_name="val",
            output_path=output_paths["val_probability_plot_path"],
            probability_threshold=probability_threshold,
        ),
        "test_probability": _plot_probability_histogram_grid(
            combined_prediction_df,
            split_name="test",
            output_path=output_paths["test_probability_plot_path"],
            probability_threshold=probability_threshold,
        ),
    }

    snapshot_analysis = {}
    if build_snapshot_progression:
        snapshot_analysis = build_snapshot_progression_artifacts(
            annotated_df=annotated_df,
            run_specs=run_specs,
            output_paths=output_paths,
            row_embeddings_by_run_slug=row_embeddings_by_run_slug,
            split_name="test",
            metrics_df=metrics_df,
            highlight_accessions=highlight_accessions,
            highlight_neighbor_count=highlight_neighbor_count,
            umap_neighbors=umap_neighbors,
            umap_min_dist=umap_min_dist,
            umap_metric=umap_metric,
            umap_random_state=umap_random_state,
            umap_color_column=umap_color_column,
        )

    html_report_path = render_embedding_logreg_html_report(
        output_dir=output_paths["results_dir"],
        report_title=report_title,
        manifest_df=manifest_df,
        target_summary_df=target_summary_df,
        class_summary_df=class_summary_df,
        edge_case_summary_df=edge_case_summary_df,
        edge_case_detail_df=edge_case_detail_df,
        train_summary_df=split_summaries["train"],
        val_summary_df=split_summaries["val"],
        test_summary_df=split_summaries["test"],
        skipped_df=skipped_df,
        plot_paths=created_plot_paths,
        diagnostic_plot_paths=diagnostic_plot_paths,
        snapshot_analysis=snapshot_analysis,
    )

    return {
        "target_summary_df": target_summary_df,
        "class_summary_df": class_summary_df,
        "edge_case_summary_df": edge_case_summary_df,
        "edge_case_detail_df": edge_case_detail_df,
        "manifest_df": manifest_df,
        "skipped_df": skipped_df,
        "metrics_df": metrics_df,
        "prediction_df": combined_prediction_df,
        "train_summary_df": split_summaries["train"],
        "val_summary_df": split_summaries["val"],
        "test_summary_df": split_summaries["test"],
        "plot_paths": created_plot_paths,
        "diagnostic_plot_paths": diagnostic_plot_paths,
        "snapshot_analysis": snapshot_analysis,
        "html_report_path": html_report_path,
        "output_paths": dict(output_paths),
        "row_embeddings_by_run_slug": row_embeddings_by_run_slug,
    }


def save_embedding_logreg_run_config(
    output_path: str | Path,
    config_payload: Mapping[str, Any],
) -> Path:
    """Write one pretty JSON config payload for notebook-14 runs."""
    return write_json(output_path, stringify_path_values(dict(config_payload)))
