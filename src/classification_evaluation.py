"""Helpers for final multi-label glycan classification evaluation.

This module is the notebook-11 counterpart to the classification prep and
training helpers. The main job here is to keep the final test-set evaluation
clean and separate from model selection:

- load one saved classifier checkpoint
- load the saved validation-selected threshold
- run one locked test-set prediction pass
- save final metrics, per-label tables, ROC curves, and PR curves

The plotting defaults are intentionally conservative. All labels are evaluated
and saved in tables, but only the top supported labels are plotted at first so
the figures stay readable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, precision_recall_fscore_support, roc_curve
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.classification_training import (
    ACCESSION_COLUMN,
    LABEL_LIST_COLUMN,
    SEQUENCE_COLUMN,
    SPLIT_COLUMN,
    GlycanClassificationDataset,
    _serialize_json,
    binarize_multilabel_predictions,
    build_classification_prediction_table,
    build_label_name_to_id,
    compute_multilabel_metrics,
    encode_multilabel_targets,
    parse_label_json_column,
    sigmoid_predictions_from_logits,
    tokenize_classification_dataframe,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def load_best_threshold(best_threshold_path: str | Path) -> dict[str, object]:
    """Load the saved validation-selected threshold from notebook 10."""
    return json.loads(Path(best_threshold_path).read_text(encoding="utf-8"))


def load_classifier_artifacts(
    model_dir: str | Path,
    label_vocabulary_path: str | Path,
    device: str | None = None,
):
    """Load the saved classifier, tokenizer, and label vocabulary snapshot."""
    label_vocabulary_df = pd.read_csv(label_vocabulary_path)
    label_name_to_id = build_label_name_to_id(label_vocabulary_df)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    runtime_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(runtime_device)
    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "runtime_device": runtime_device,
        "label_vocabulary_df": label_vocabulary_df,
        "label_name_to_id": label_name_to_id,
    }


def load_test_classification_table(test_csv_path: str | Path) -> "pd.DataFrame":
    """Load the prepared test classification table from notebook 09."""
    test_df = parse_label_json_column(pd.read_csv(test_csv_path))
    test_df = test_df.copy()
    test_df[SPLIT_COLUMN] = "test"
    return test_df


def build_test_classification_dataset(
    test_df: "pd.DataFrame",
    tokenizer,
    label_name_to_id: "Mapping[str, int]",
    max_length: int,
) -> dict[str, object]:
    """Encode labels and tokenize the test dataframe for final evaluation."""
    encoded_test_df = encode_multilabel_targets(test_df, label_name_to_id)
    test_bundle = tokenize_classification_dataframe(
        encoded_test_df,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return {
        "test_df": encoded_test_df,
        "test_bundle": test_bundle,
        "test_dataset": GlycanClassificationDataset(test_bundle),
    }


def run_classifier_predictions(
    model,
    evaluation_dataset,
    runtime_device,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """Run one forward-pass sweep over the evaluation dataset."""
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=batch_size)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in evaluation_loader:
            input_ids = batch["input_ids"].to(runtime_device)
            attention_mask = batch["attention_mask"].to(runtime_device)
            labels = batch["labels"].to(runtime_device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    true_labels = torch.cat(all_labels, dim=0).numpy()
    probabilities = sigmoid_predictions_from_logits(logits)

    return {
        "logits": logits,
        "true_labels": true_labels,
        "probabilities": probabilities,
    }


def _build_monotonic_pr_curve(binary_true, scores):
    """Return an interpolated monotonic precision-recall envelope and AP."""
    from sklearn.metrics import average_precision_score

    positive_count = int(binary_true.sum())

    if positive_count == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0

    if positive_count == len(binary_true):
        return np.array([0.0, 1.0]), np.array([1.0, 1.0]), 1.0

    precision, recall, _ = precision_recall_curve(binary_true, scores)
    average_precision = average_precision_score(binary_true, scores)

    recall = recall[::-1]
    precision = precision[::-1]

    unique_recall = np.unique(recall)
    max_precision = np.array([precision[recall == value].max() for value in unique_recall])
    monotonic_precision = np.maximum.accumulate(max_precision[::-1])[::-1]

    return unique_recall, monotonic_precision, float(average_precision)


def _build_roc_curve(binary_true, scores):
    """Return an ROC curve and AUC with stable edge-case handling."""
    positive_count = int(binary_true.sum())

    if positive_count == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0

    if positive_count == len(binary_true):
        return np.array([0.0, 1.0]), np.array([1.0, 1.0]), 1.0

    fpr, tpr, _ = roc_curve(binary_true, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, float(roc_auc)


def compute_per_label_metrics(
    true_labels,
    predicted_labels,
    label_vocabulary_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Compute one-row-per-label precision, recall, F1, and support."""
    true_array = np.asarray(true_labels, dtype=int)
    predicted_array = np.asarray(predicted_labels, dtype=int)

    per_label_precision, per_label_recall, per_label_f1, per_label_support = (
        precision_recall_fscore_support(
            true_array,
            predicted_array,
            average=None,
            zero_division=0,
        )
    )

    per_label_df = label_vocabulary_df[["label_id", "label_name"]].copy()
    per_label_df["precision"] = per_label_precision
    per_label_df["recall"] = per_label_recall
    per_label_df["f1"] = per_label_f1
    per_label_df["support"] = per_label_support

    return per_label_df.sort_values(["support", "label_name"], ascending=[False, True]).reset_index(drop=True)


def build_multilabel_roc_summary(
    true_labels,
    predicted_probabilities,
    label_vocabulary_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Compute one-vs-rest ROC-AUC for every label."""
    true_array = np.asarray(true_labels, dtype=int)
    probability_array = np.asarray(predicted_probabilities, dtype=float)

    rows = []
    for row in label_vocabulary_df.itertuples(index=False):
        label_id = int(row.label_id)
        binary_true = true_array[:, label_id]
        label_scores = probability_array[:, label_id]
        _, _, roc_auc = _build_roc_curve(binary_true, label_scores)
        rows.append(
            {
                "label_id": label_id,
                "label_name": str(row.label_name),
                "support": int(binary_true.sum()),
                "auc": float(roc_auc),
            }
        )

    return pd.DataFrame(rows).sort_values(["support", "label_name"], ascending=[False, True]).reset_index(drop=True)


def build_multilabel_pr_summary(
    true_labels,
    predicted_probabilities,
    label_vocabulary_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Compute one-vs-rest monotonic PR summary rows for every label."""
    true_array = np.asarray(true_labels, dtype=int)
    probability_array = np.asarray(predicted_probabilities, dtype=float)

    rows = []
    for row in label_vocabulary_df.itertuples(index=False):
        label_id = int(row.label_id)
        binary_true = true_array[:, label_id]
        label_scores = probability_array[:, label_id]
        _, _, average_precision = _build_monotonic_pr_curve(binary_true, label_scores)
        rows.append(
            {
                "label_id": label_id,
                "label_name": str(row.label_name),
                "support": int(binary_true.sum()),
                "average_precision": float(average_precision),
            }
        )

    return pd.DataFrame(rows).sort_values(["support", "label_name"], ascending=[False, True]).reset_index(drop=True)


def build_curve_aggregate_summary(
    roc_summary_df: "pd.DataFrame",
    pr_summary_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Build macro and weighted aggregate summaries for ROC-AUC and PR-AUC."""
    roc_weights = roc_summary_df["support"].to_numpy(dtype=float)
    pr_weights = pr_summary_df["support"].to_numpy(dtype=float)

    rows = [
        {
            "metric_family": "roc_auc",
            "aggregation": "macro",
            "value": float(roc_summary_df["auc"].mean()),
        },
        {
            "metric_family": "roc_auc",
            "aggregation": "weighted",
            "value": float(np.average(roc_summary_df["auc"], weights=roc_weights)),
        },
        {
            "metric_family": "average_precision",
            "aggregation": "macro",
            "value": float(pr_summary_df["average_precision"].mean()),
        },
        {
            "metric_family": "average_precision",
            "aggregation": "weighted",
            "value": float(np.average(pr_summary_df["average_precision"], weights=pr_weights)),
        },
    ]

    return pd.DataFrame(rows)


def select_top_supported_labels(
    label_summary_df: "pd.DataFrame",
    top_k: int = 10,
) -> list[str]:
    """Return the top-k label names by support for first-pass plotting."""
    return label_summary_df.sort_values(["support", "label_name"], ascending=[False, True])["label_name"].head(top_k).tolist()


def plot_selected_roc_curves(
    true_labels,
    predicted_probabilities,
    label_vocabulary_df: "pd.DataFrame",
    selected_label_names: list[str],
    save_path: str | Path | None = None,
) -> "pd.DataFrame":
    """Plot one-vs-rest ROC curves for a selected set of labels."""
    true_array = np.asarray(true_labels, dtype=int)
    probability_array = np.asarray(predicted_probabilities, dtype=float)
    selected_df = label_vocabulary_df.loc[label_vocabulary_df["label_name"].isin(selected_label_names)].copy()

    plt.figure(figsize=(8, 6))
    rows = []

    for row in selected_df.itertuples(index=False):
        label_id = int(row.label_id)
        binary_true = true_array[:, label_id]
        label_scores = probability_array[:, label_id]
        fpr, tpr, roc_auc = _build_roc_curve(binary_true, label_scores)
        rows.append(
            {
                "label_id": label_id,
                "label_name": str(row.label_name),
                "support": int(binary_true.sum()),
                "auc": float(roc_auc),
            }
        )
        # Keep the legend readable. The numeric AUC values are still saved in
        # the companion CSV summary, so the plot legend only needs label names.
        plt.plot(fpr, tpr, label=str(row.label_name))

    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves for top supported subtype labels")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return pd.DataFrame(rows).sort_values(["support", "label_name"], ascending=[False, True]).reset_index(drop=True)


def plot_selected_monotonic_pr_curves(
    true_labels,
    predicted_probabilities,
    label_vocabulary_df: "pd.DataFrame",
    selected_label_names: list[str],
    save_path: str | Path | None = None,
) -> "pd.DataFrame":
    """Plot monotonic one-vs-rest PR curves for a selected set of labels."""
    true_array = np.asarray(true_labels, dtype=int)
    probability_array = np.asarray(predicted_probabilities, dtype=float)
    selected_df = label_vocabulary_df.loc[label_vocabulary_df["label_name"].isin(selected_label_names)].copy()

    plt.figure(figsize=(8, 6))
    rows = []

    for row in selected_df.itertuples(index=False):
        label_id = int(row.label_id)
        binary_true = true_array[:, label_id]
        label_scores = probability_array[:, label_id]
        recall, precision, average_precision = _build_monotonic_pr_curve(binary_true, label_scores)
        rows.append(
            {
                "label_id": label_id,
                "label_name": str(row.label_name),
                "support": int(binary_true.sum()),
                "average_precision": float(average_precision),
            }
        )
        # As with ROC, keep the plot legend simple and leave the numeric
        # average-precision values in the saved CSV summary table.
        plt.plot(recall, precision, label=str(row.label_name))

    plt.xlabel("Recall")
    plt.ylabel("Interpolated precision")
    plt.title("Monotonic PR curves for top supported subtype labels")
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return pd.DataFrame(rows).sort_values(["support", "label_name"], ascending=[False, True]).reset_index(drop=True)


def build_classification_evaluation_output_paths(
    project_root: str | Path,
    tokenizer_family: str,
    experiment_name: str,
    classifier_run_label: str,
) -> dict[str, str]:
    """Build the Drive output paths for one final classification evaluation."""
    project_root = Path(project_root)
    results_dir = (
        project_root
        / "results"
        / "classification_evaluation"
        / tokenizer_family
        / experiment_name
        / classifier_run_label
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    return {
        "results_dir": str(results_dir),
        "evaluation_config_path": str(results_dir / "evaluation_config.json"),
        "test_metrics_csv_path": str(results_dir / "test_metrics.csv"),
        "test_metrics_json_path": str(results_dir / "test_metrics.json"),
        "per_label_metrics_path": str(results_dir / "per_label_metrics.csv"),
        "roc_summary_path": str(results_dir / "roc_auc_per_label.csv"),
        "pr_summary_path": str(results_dir / "average_precision_per_label.csv"),
        "curve_aggregate_summary_path": str(results_dir / "curve_aggregate_summary.csv"),
        "top10_roc_summary_path": str(results_dir / "top10_supported_roc_summary.csv"),
        "top10_pr_summary_path": str(results_dir / "top10_supported_pr_summary.csv"),
        "top10_roc_plot_path": str(results_dir / "top10_supported_roc_curves.png"),
        "top10_pr_plot_path": str(results_dir / "top10_supported_pr_curves.png"),
        "test_prediction_table_path": str(results_dir / "test_prediction_table.csv"),
    }


def save_classification_evaluation_outputs(
    test_metrics: dict[str, object],
    per_label_metrics_df: "pd.DataFrame",
    roc_summary_df: "pd.DataFrame",
    pr_summary_df: "pd.DataFrame",
    curve_aggregate_summary_df: "pd.DataFrame",
    top10_roc_summary_df: "pd.DataFrame",
    top10_pr_summary_df: "pd.DataFrame",
    test_prediction_table_df: "pd.DataFrame",
    output_paths: dict[str, str],
) -> None:
    """Write final evaluation tables and summaries to disk."""
    pd.DataFrame([test_metrics]).to_csv(output_paths["test_metrics_csv_path"], index=False)
    Path(output_paths["test_metrics_json_path"]).write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )
    per_label_metrics_df.to_csv(output_paths["per_label_metrics_path"], index=False)
    roc_summary_df.to_csv(output_paths["roc_summary_path"], index=False)
    pr_summary_df.to_csv(output_paths["pr_summary_path"], index=False)
    curve_aggregate_summary_df.to_csv(output_paths["curve_aggregate_summary_path"], index=False)
    top10_roc_summary_df.to_csv(output_paths["top10_roc_summary_path"], index=False)
    top10_pr_summary_df.to_csv(output_paths["top10_pr_summary_path"], index=False)
    test_prediction_table_df.to_csv(output_paths["test_prediction_table_path"], index=False)
