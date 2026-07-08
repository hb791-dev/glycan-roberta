"""Helpers for looking at rare-token behavior after notebook 6 runs.

This module keeps the post-processing pieces out of the notebook so the
notebook can stay focused on setup, interpretation, and quick displays.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_rarity_inputs(
    test_summary_path: str | Path,
    per_class_metrics_path: str | Path,
    top1_roc_per_class_path: str | Path,
    top1_pr_per_class_path: str | Path,
):
    """Load the saved notebook-6 artifacts needed for rarity analysis."""
    test_summary_path = Path(test_summary_path)
    per_class_metrics_path = Path(per_class_metrics_path)
    top1_roc_per_class_path = Path(top1_roc_per_class_path)
    top1_pr_per_class_path = Path(top1_pr_per_class_path)

    test_summary = json.loads(test_summary_path.read_text())
    per_class_metrics = pd.read_csv(per_class_metrics_path)
    top1_roc_per_class = pd.read_csv(top1_roc_per_class_path)
    top1_pr_per_class = pd.read_csv(top1_pr_per_class_path)

    return {
        "test_summary": test_summary,
        "per_class_metrics": per_class_metrics,
        "top1_roc_per_class": top1_roc_per_class,
        "top1_pr_per_class": top1_pr_per_class,
    }


def merge_rarity_tables(per_class_metrics, top1_roc_per_class, top1_pr_per_class):
    """Merge the saved class-level tables into one notebook-friendly frame."""
    merged = per_class_metrics.merge(
        top1_roc_per_class[
            [
                "token_id",
                "token",
                "support",
                "correct_count",
                "incorrect_count",
                "top1_accuracy",
                "auc",
            ]
        ],
        on=["token_id", "token", "support"],
        how="left",
    )
    merged = merged.merge(
        top1_pr_per_class[["token_id", "token", "support", "average_precision"]],
        on=["token_id", "token", "support"],
        how="left",
    )
    return merged.sort_values(["support", "token"], ascending=[False, True]).reset_index(drop=True)


def assign_support_bins(merged_df, support_bins):
    """Attach readable support-bin labels to the merged class table."""
    df = merged_df.copy()
    ordered_labels = []

    def _label_for_support(support):
        support = int(support)
        for lower, upper in support_bins:
            if upper is None and support >= lower:
                return f"{lower}+"
            if lower <= support <= upper:
                return f"{lower}-{upper}"
        return "outside_bins"

    for lower, upper in support_bins:
        if upper is None:
            ordered_labels.append(f"{lower}+")
        else:
            ordered_labels.append(f"{lower}-{upper}")

    df["support_bin"] = df["support"].map(_label_for_support)
    df["support_bin"] = pd.Categorical(
        df["support_bin"],
        categories=ordered_labels,
        ordered=True,
    )
    return df


def add_rarity_flags(merged_df, rare_support_max=24):
    """Mark whether each token falls into the headline rare-token cutoff."""
    df = merged_df.copy()
    df["is_rare"] = df["support"] <= int(rare_support_max)
    return df


def build_rarity_bin_summary(merged_df):
    """Summarize support, F1, AP, and AUC within each support bin."""
    grouped = (
        merged_df.groupby("support_bin", dropna=False, sort=False, observed=False)
        .agg(
            num_token_classes=("token", "count"),
            total_support=("support", "sum"),
            mean_support=("support", "mean"),
            median_support=("support", "median"),
            mean_f1=("f1", "mean"),
            median_f1=("f1", "median"),
            mean_top1_accuracy=("top1_accuracy", "mean"),
            mean_average_precision=("average_precision", "mean"),
            mean_auc=("auc", "mean"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values("support_bin").reset_index(drop=True)
    total_classes = grouped["num_token_classes"].sum()
    if total_classes > 0:
        grouped["share_of_token_classes"] = grouped["num_token_classes"] / total_classes
    else:
        grouped["share_of_token_classes"] = 0.0
    return grouped


def build_rare_token_table(merged_df, rare_support_max=24):
    """Return the rare-token rows sorted from smallest support upward."""
    rare_df = merged_df.loc[merged_df["support"] <= int(rare_support_max)].copy()
    if {"incorrect_count", "support"}.issubset(rare_df.columns):
        rare_df["top1_error_rate"] = np.where(
            rare_df["support"] > 0,
            rare_df["incorrect_count"] / rare_df["support"],
            np.nan,
        )

    display_columns = [
        "token_id",
        "token",
        "support",
        "support_bin",
        "precision",
        "recall",
        "f1",
        "top1_accuracy",
        "top1_error_rate",
        "average_precision",
        "auc",
        "correct_count",
        "incorrect_count",
    ]
    available_columns = [column_name for column_name in display_columns if column_name in rare_df.columns]
    rare_df = rare_df[available_columns]
    return rare_df.sort_values(["support", "f1", "token"], ascending=[True, True, True]).reset_index(drop=True)


def build_problem_rare_token_table(merged_df, rare_support_max=24):
    """Return rare tokens with real errors, ranked by worst F1 first."""
    rare_df = build_rare_token_table(merged_df, rare_support_max=rare_support_max).copy()

    if "incorrect_count" in rare_df.columns:
        rare_df = rare_df.loc[rare_df["incorrect_count"] > 0].copy()

    sort_columns = [column_name for column_name in ["f1", "incorrect_count", "top1_accuracy", "support", "token"] if column_name in rare_df.columns]
    ascending = []
    for column_name in sort_columns:
        if column_name in {"f1", "top1_accuracy", "support", "token"}:
            ascending.append(True)
        else:
            ascending.append(False)

    if sort_columns:
        rare_df = rare_df.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)

    spotlight_columns = [
        "token",
        "support",
        "f1",
        "top1_accuracy",
        "correct_count",
        "incorrect_count",
    ]
    available_columns = [column_name for column_name in spotlight_columns if column_name in rare_df.columns]
    return rare_df[available_columns]


def _safe_corr(x_values, y_values):
    """Return a simple Pearson correlation or NaN when it cannot be computed."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if len(x_values) < 2:
        return float("nan")
    if np.allclose(x_values, x_values[0]) or np.allclose(y_values, y_values[0]):
        return float("nan")
    return float(np.corrcoef(x_values, y_values)[0, 1])


def compute_rarity_summary(merged_df, test_summary, rare_support_max=24):
    """Build one compact summary dictionary for the notebook and saved JSON."""
    rare_df = merged_df.loc[merged_df["support"] <= int(rare_support_max)]
    common_df = merged_df.loc[merged_df["support"] >= 100]
    support_values = merged_df["support"].astype(float).to_numpy()

    summary = {
        "num_token_classes": int(len(merged_df)),
        "num_classes_support_lt10": int((merged_df["support"] < 10).sum()),
        "num_classes_support_lt25": int((merged_df["support"] < 25).sum()),
        "median_support": float(merged_df["support"].median()),
        "min_support": int(merged_df["support"].min()),
        "max_support": int(merged_df["support"].max()),
        "mean_f1_support_lt25": float(rare_df["f1"].mean()) if not rare_df.empty else float("nan"),
        "mean_f1_support_ge100": float(common_df["f1"].mean()) if not common_df.empty else float("nan"),
        "corr_support_f1": _safe_corr(support_values, merged_df["f1"]),
        "corr_support_average_precision": _safe_corr(support_values, merged_df["average_precision"]),
        "corr_support_auc": _safe_corr(support_values, merged_df["auc"]),
        "macro_precision": float(test_summary["macro_precision"]),
        "macro_recall": float(test_summary["macro_recall"]),
        "macro_f1": float(test_summary["macro_f1"]),
        "weighted_precision": float(test_summary["weighted_precision"]),
        "weighted_recall": float(test_summary["weighted_recall"]),
        "weighted_f1": float(test_summary["weighted_f1"]),
        "macro_weighted_f1_gap": float(test_summary["weighted_f1"] - test_summary["macro_f1"]),
    }
    return summary


def plot_support_distribution(merged_df, output_path):
    """Plot how many token classes land in each support bin."""
    output_path = Path(output_path)
    counts = (
        merged_df.groupby("support_bin", sort=False, observed=False)["token"]
        .count()
        .reset_index(name="num_token_classes")
    )
    counts = counts.sort_values("support_bin").reset_index(drop=True)
    x_labels = counts["support_bin"].astype(str)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x_labels, counts["num_token_classes"], color="#4C72B0")
    plt.xlabel("Support bin")
    plt.ylabel("Number of token classes")
    plt.title("How many token classes fall in each support bin?")
    plt.grid(axis="y", alpha=0.2)

    for bar, count in zip(bars, counts["num_token_classes"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_support_metric_scatter(merged_df, metric_col, output_path, rare_support_max=None):
    """Plot one metric against raw support."""
    output_path = Path(output_path)
    metric_labels = {
        "f1": "F1",
        "average_precision": "Average precision",
        "auc": "ROC AUC",
    }

    plt.figure(figsize=(8, 5))
    if rare_support_max is not None:
        rare_mask = merged_df["support"] <= int(rare_support_max)
        plt.scatter(
            merged_df.loc[~rare_mask, "support"],
            merged_df.loc[~rare_mask, metric_col],
            alpha=0.7,
            color="#4C72B0",
            label=f"Support > {int(rare_support_max)}",
        )
        plt.scatter(
            merged_df.loc[rare_mask, "support"],
            merged_df.loc[rare_mask, metric_col],
            alpha=0.85,
            color="#DD8452",
            label=f"Support <= {int(rare_support_max)}",
        )
        plt.axvline(
            int(rare_support_max),
            color="#6C6C6C",
            linestyle="--",
            linewidth=1.2,
        )
        plt.legend(frameon=False)
    else:
        plt.scatter(merged_df["support"], merged_df[metric_col], alpha=0.7, color="#DD8452")

    plt.xlabel("Support in masked test set")
    plt.ylabel(metric_labels.get(metric_col, metric_col))
    plt.title(f"{metric_labels.get(metric_col, metric_col)} vs support")
    plt.grid(alpha=0.2)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_metric_by_support_bin(bin_summary_df, metric_col, output_path):
    """Plot one averaged metric across the readable support bins."""
    output_path = Path(output_path)
    metric_labels = {
        "mean_f1": "Mean F1",
        "mean_average_precision": "Mean average precision",
        "mean_auc": "Mean ROC AUC",
    }
    plot_df = bin_summary_df.sort_values("support_bin").reset_index(drop=True)
    x_labels = plot_df["support_bin"].astype(str)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x_labels, plot_df[metric_col], color="#55A868")
    plt.xlabel("Support bin")
    plt.ylabel(metric_labels.get(metric_col, metric_col))
    plt.title(f"{metric_labels.get(metric_col, metric_col)} by support bin")
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, plot_df[metric_col]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def save_rarity_outputs(
    output_dir,
    merged_df,
    bin_summary_df,
    rare_token_df,
    rarity_summary,
    rarity_config,
):
    """Write the main rarity outputs and return the saved file paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_metrics_path = output_dir / "merged_rarity_metrics.csv"
    rarity_bin_summary_path = output_dir / "rarity_bin_summary.csv"
    rare_token_table_path = output_dir / "rare_token_table.csv"
    rarity_summary_path = output_dir / "rarity_summary.json"
    rarity_config_path = output_dir / "rarity_config.json"

    merged_df.to_csv(merged_metrics_path, index=False)
    bin_summary_df.to_csv(rarity_bin_summary_path, index=False)
    rare_token_df.to_csv(rare_token_table_path, index=False)
    rarity_summary_path.write_text(json.dumps(rarity_summary, indent=2))
    rarity_config_path.write_text(json.dumps(rarity_config, indent=2))

    return {
        "merged_metrics_path": merged_metrics_path,
        "rarity_bin_summary_path": rarity_bin_summary_path,
        "rare_token_table_path": rare_token_table_path,
        "rarity_summary_path": rarity_summary_path,
        "rarity_config_path": rarity_config_path,
    }
