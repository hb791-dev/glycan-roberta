"""Helpers for classification-focused glycan embedding UMAP exploration.

This module keeps notebook-facing logic for:

- loading the prepared classification tables from notebook 09
- deriving broader label views from the existing subtype labels
- deriving broad branching categories from sequence structure
- resolving pretrained vs classification-fine-tuned checkpoints
- computing and saving UMAP projections plus category-colored plots

The goal is to let the notebook stay small and parameter-driven while the
repeatable mechanics live in one reusable helper module.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.classification_training import (
    LABEL_JSON_COLUMN,
    LABEL_LIST_COLUMN,
    SEQUENCE_COLUMN,
    SPLIT_COLUMN,
    parse_label_json_column,
)

if TYPE_CHECKING:
    import torch


SUPPORTED_MODEL_VARIANTS = (
    "pretrained_mlm",
    "classification_mlm_init",
    "classification_random_init",
)

SUPPORTED_COLOR_COLUMNS = (
    "primary_subtype_label",
    "n_o_category",
    "main_glycan_class",
    "broad_branching",
)

_N_GLYCAN_KEYWORDS = (
    "n-linked",
    "n-glycan",
    "n glycan",
    "high mannose",
    "paucimannose",
    "hybrid",
    "complex n",
    "bisected",
)

_O_GLYCAN_KEYWORDS = (
    "o-linked",
    "o-glycan",
    "o glycan",
    "mucin",
    "o-mannose",
    "o-fucose",
    "o-glucose",
    "o-glcnac",
    "o-galactose",
    "o-galnac",
    "ogalnac",
)

_OTHER_MAIN_CLASS_KEYWORDS = (
    "glycosphingolipid",
    "glycolipid",
    "ganglioside",
    "globoside",
    "gpi",
    "glycosaminoglycan",
    "hepar",
    "chondroitin",
    "dermatan",
    "keratan",
    "hyaluron",
)


def _require_columns(dataframe: "pd.DataFrame", required_columns: Sequence[str], frame_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{frame_name} is missing required columns: {missing_columns}")


def load_combined_classification_splits(
    train_csv_path: str | Path,
    val_csv_path: str | Path,
    test_csv_path: str | Path,
) -> "pd.DataFrame":
    """Load and combine the prepared classification split tables from notebook 09."""
    split_frames: list[pd.DataFrame] = []

    for split_name, split_path in (
        ("train", train_csv_path),
        ("val", val_csv_path),
        ("test", test_csv_path),
    ):
        split_df = parse_label_json_column(pd.read_csv(split_path)).copy()
        split_df[SPLIT_COLUMN] = split_name
        _require_columns(
            split_df,
            [SEQUENCE_COLUMN, LABEL_JSON_COLUMN, LABEL_LIST_COLUMN, SPLIT_COLUMN],
            f"{split_name}_df",
        )
        split_frames.append(split_df)

    combined_df = pd.concat(split_frames, ignore_index=True)
    combined_df[SEQUENCE_COLUMN] = combined_df[SEQUENCE_COLUMN].fillna("").map(str).map(str.strip)
    if combined_df[SEQUENCE_COLUMN].eq("").any():
        raise ValueError("Combined classification dataframe contains blank sequences.")
    return combined_df


def filter_classification_dataframe_by_split(
    classification_df: "pd.DataFrame",
    splits_to_include: Sequence[str] | None = None,
) -> "pd.DataFrame":
    """Keep only the requested train/val/test subsets for exploration."""
    _require_columns(classification_df, [SPLIT_COLUMN], "classification_df")

    normalized_splits = [str(split_name).strip().lower() for split_name in (splits_to_include or [])]
    if not normalized_splits:
        return classification_df.copy().reset_index(drop=True)

    filtered_df = classification_df.loc[
        classification_df[SPLIT_COLUMN].map(lambda value: str(value).strip().lower()).isin(normalized_splits)
    ].copy()
    return filtered_df.reset_index(drop=True)


def build_label_support_lookup(label_vocabulary_df: "pd.DataFrame") -> dict[str, int]:
    """Build a support lookup so multi-label glycans can choose one display label."""
    _require_columns(label_vocabulary_df, ["label_name", "support_total"], "label_vocabulary_df")
    return {
        str(row.label_name): int(row.support_total)
        for row in label_vocabulary_df.itertuples(index=False)
    }


def _normalize_label_text(label_name: str) -> str:
    return str(label_name).strip().lower().replace("_", " ")


def choose_primary_subtype_label(
    label_values: Sequence[str],
    label_support_lookup: Mapping[str, int],
) -> str:
    """Choose one deterministic subtype label for single-color UMAP rendering.

    The underlying task is multi-label, but a scatter plot needs one category
    per point. This helper picks the most-supported label among the glycan's
    assigned subtype labels, breaking ties alphabetically for reproducibility.
    """
    normalized_labels = [str(label).strip() for label in label_values if str(label).strip()]
    if not normalized_labels:
        return "unlabeled"

    ordered_labels = sorted(
        normalized_labels,
        key=lambda label_name: (-int(label_support_lookup.get(label_name, 0)), label_name),
    )
    return ordered_labels[0]


def infer_n_o_category(label_values: Sequence[str]) -> str:
    """Collapse subtype labels into an N-vs-O style category."""
    normalized_labels = [_normalize_label_text(label_name) for label_name in label_values]
    has_n = any(any(keyword in label_name for keyword in _N_GLYCAN_KEYWORDS) for label_name in normalized_labels)
    has_o = any(any(keyword in label_name for keyword in _O_GLYCAN_KEYWORDS) for label_name in normalized_labels)

    if has_n and has_o:
        return "Mixed N/O"
    if has_n:
        return "N-glycan"
    if has_o:
        return "O-glycan"
    return "Neither/Other"


def infer_main_glycan_class(label_values: Sequence[str]) -> str:
    """Collapse subtype labels into three broad classes for overview plots."""
    n_o_category = infer_n_o_category(label_values)
    if n_o_category == "N-glycan":
        return "N-glycan"
    if n_o_category == "O-glycan":
        return "O-glycan"

    normalized_labels = [_normalize_label_text(label_name) for label_name in label_values]
    has_other_main_class = any(
        any(keyword in label_name for keyword in _OTHER_MAIN_CLASS_KEYWORDS)
        for label_name in normalized_labels
    )
    if has_other_main_class:
        return "Other glycan"

    return "Other glycan"


def count_branch_openings(sequence: str) -> int:
    """Count broad branch openings in a compact glycan sequence."""
    sequence = str(sequence)
    return sequence.count("(") + sequence.count("[")


def estimate_max_branch_depth(sequence: str) -> int:
    """Estimate maximum branch nesting depth from bracket-like delimiters."""
    sequence = str(sequence)
    current_depth = 0
    max_depth = 0

    for character in sequence:
        if character in "([":
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif character in ")]":
            current_depth = max(0, current_depth - 1)

    return max_depth


def infer_broad_branching(sequence: str) -> str:
    """Map a sequence to a coarse branching bucket."""
    branch_count = count_branch_openings(sequence)
    branch_depth = estimate_max_branch_depth(sequence)

    if branch_count == 0:
        return "Unbranched"
    if branch_count == 1 and branch_depth <= 1:
        return "Single branch"
    if branch_count == 2 and branch_depth <= 1:
        return "Two branches"
    return "Highly branched"


def annotate_classification_umap_metadata(
    classification_df: "pd.DataFrame",
    label_vocabulary_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Add broad-view label columns used for classification UMAP coloring."""
    _require_columns(classification_df, [SEQUENCE_COLUMN, LABEL_LIST_COLUMN, SPLIT_COLUMN], "classification_df")
    label_support_lookup = build_label_support_lookup(label_vocabulary_df)

    annotated_df = classification_df.copy()
    annotated_df["num_labels"] = annotated_df[LABEL_LIST_COLUMN].map(
        lambda value: len(value) if isinstance(value, list) else 0
    )
    annotated_df["primary_subtype_label"] = annotated_df[LABEL_LIST_COLUMN].map(
        lambda labels: choose_primary_subtype_label(labels, label_support_lookup)
    )
    annotated_df["n_o_category"] = annotated_df[LABEL_LIST_COLUMN].map(infer_n_o_category)
    annotated_df["main_glycan_class"] = annotated_df[LABEL_LIST_COLUMN].map(infer_main_glycan_class)
    annotated_df["branch_open_count"] = annotated_df[SEQUENCE_COLUMN].map(count_branch_openings)
    annotated_df["max_branch_depth"] = annotated_df[SEQUENCE_COLUMN].map(estimate_max_branch_depth)
    annotated_df["broad_branching"] = annotated_df[SEQUENCE_COLUMN].map(infer_broad_branching)
    annotated_df["label_signature"] = annotated_df[LABEL_LIST_COLUMN].map(
        lambda labels: " | ".join(sorted(str(label).strip() for label in labels if str(label).strip()))
        if isinstance(labels, list)
        else ""
    )
    return annotated_df


def summarize_umap_categories(
    annotated_df: "pd.DataFrame",
    category_columns: Sequence[str] = SUPPORTED_COLOR_COLUMNS,
) -> "pd.DataFrame":
    """Summarize category counts for notebook display and saved outputs."""
    summary_rows: list[dict[str, object]] = []

    for category_column in category_columns:
        if category_column not in annotated_df.columns:
            raise ValueError(f"Missing category column for summary: {category_column!r}")

        counts = Counter(annotated_df[category_column].fillna("missing").map(str).tolist())
        for category_name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            summary_rows.append(
                {
                    "category_column": category_column,
                    "category_name": str(category_name),
                    "count": int(count),
                }
            )

    return pd.DataFrame(summary_rows)


def resolve_embedding_model_dir(
    checkpoints_dir: str | Path,
    tokenizer_family: str,
    experiment_name: str,
    model_variant: str,
    classifier_mlm_run_label: str | None = None,
    classifier_random_run_label: str | None = None,
    model_subdir: str = "best_model",
) -> Path:
    """Resolve one embedding checkpoint directory from notebook-friendly settings."""
    checkpoints_dir = Path(checkpoints_dir)
    normalized_variant = str(model_variant).strip()

    if normalized_variant not in SUPPORTED_MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported model_variant {model_variant!r}. Expected one of {SUPPORTED_MODEL_VARIANTS}."
        )

    if normalized_variant == "pretrained_mlm":
        return checkpoints_dir / str(tokenizer_family) / str(experiment_name) / str(model_subdir)

    if normalized_variant == "classification_mlm_init":
        classifier_run_label = str(classifier_mlm_run_label or "").strip()
        if not classifier_run_label:
            raise ValueError("classifier_mlm_run_label is required for classification_mlm_init.")
        return (
            checkpoints_dir
            / "classification"
            / str(tokenizer_family)
            / str(experiment_name)
            / classifier_run_label
            / str(model_subdir)
        )

    classifier_run_label = str(classifier_random_run_label or "").strip()
    if not classifier_run_label:
        raise ValueError("classifier_random_run_label is required for classification_random_init.")
    return (
        checkpoints_dir
        / "classification"
        / str(tokenizer_family)
        / str(experiment_name)
        / classifier_run_label
        / str(model_subdir)
    )


def build_classification_umap_output_paths(
    project_root: str | Path,
    tokenizer_family: str,
    experiment_name: str,
    model_variant: str,
    pooling_strategy: str,
    output_run_label: str = "default_run",
) -> dict[str, str]:
    """Build results paths for one classification UMAP exploration run."""
    project_root = Path(project_root)
    results_dir = (
        project_root
        / "results"
        / "classification_embedding_umap"
        / str(tokenizer_family)
        / str(experiment_name)
        / str(model_variant)
        / f"{str(pooling_strategy)}_pool"
        / str(output_run_label)
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    return {
        "results_dir": str(results_dir),
        "run_config_path": str(results_dir / "run_config.json"),
        "annotated_glycans_path": str(results_dir / "annotated_glycans.csv"),
        "category_summary_path": str(results_dir / "category_summary.csv"),
        "umap_coordinates_path": str(results_dir / "umap_coordinates.csv"),
        "primary_subtype_plot_path": str(results_dir / "umap_primary_subtype.png"),
        "n_o_plot_path": str(results_dir / "umap_n_vs_o.png"),
        "main_class_plot_path": str(results_dir / "umap_main_glycan_class.png"),
        "branching_plot_path": str(results_dir / "umap_broad_branching.png"),
    }


def _to_embedding_array(embeddings) -> np.ndarray:
    """Return embeddings as one NumPy float array."""
    if hasattr(embeddings, "detach"):
        return embeddings.detach().cpu().numpy()
    return np.asarray(embeddings, dtype=float)


def compute_umap_projection(
    embeddings,
    n_neighbors: int = 15,
    min_dist: float = 0.10,
    metric: str = "cosine",
    random_state: int = 42,
    return_reducer: bool = False,
):
    """Project embeddings to two dimensions with UMAP.

    When ``return_reducer`` is ``True``, this helper also returns the fitted
    reducer so later embedding sets can be transformed into the same UMAP
    coordinate space.
    """
    try:
        import umap
    except ImportError as error:
        raise ImportError(
            "UMAP is not installed. Install the 'umap-learn' package before running this notebook."
        ) from error

    embedding_array = _to_embedding_array(embeddings)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=int(random_state),
    )
    coordinates = reducer.fit_transform(embedding_array)
    if return_reducer:
        return coordinates, reducer
    return coordinates


def transform_umap_projection(embeddings, reducer) -> np.ndarray:
    """Project embeddings into one already-fitted UMAP space."""
    embedding_array = _to_embedding_array(embeddings)
    return reducer.transform(embedding_array)


def build_umap_dataframe(
    annotated_df: "pd.DataFrame",
    umap_coordinates,
) -> "pd.DataFrame":
    """Attach UMAP coordinates to the annotated classification dataframe."""
    coordinate_array = np.asarray(umap_coordinates, dtype=float)
    if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 2:
        raise ValueError(
            f"Expected UMAP coordinates with shape (n_rows, 2), got {coordinate_array.shape}."
        )

    if len(annotated_df) != len(coordinate_array):
        raise ValueError(
            "Annotated dataframe length does not match number of UMAP coordinates: "
            f"{len(annotated_df)} vs {len(coordinate_array)}."
        )

    umap_df = annotated_df.reset_index(drop=True).copy()
    umap_df["umap_1"] = coordinate_array[:, 0]
    umap_df["umap_2"] = coordinate_array[:, 1]
    return umap_df


def _prepare_plot_categories(
    umap_df: "pd.DataFrame",
    category_column: str,
    top_k_categories: int | None = None,
    other_label: str = "Other",
) -> "pd.DataFrame":
    plot_df = umap_df.copy()
    plot_df[category_column] = plot_df[category_column].fillna("missing").map(str)

    if top_k_categories is None or top_k_categories <= 0:
        return plot_df

    top_categories = (
        plot_df[category_column]
        .value_counts()
        .head(int(top_k_categories))
        .index
        .tolist()
    )
    plot_df[category_column] = plot_df[category_column].map(
        lambda value: value if value in top_categories else str(other_label)
    )
    return plot_df


def plot_umap_by_category(
    umap_df: "pd.DataFrame",
    category_column: str,
    output_path: str | Path,
    title: str,
    top_k_categories: int | None = None,
    other_label: str = "Other",
    point_size: int | float = 18,
    alpha: float = 0.82,
    figure_size: tuple[int | float, int | float] = (10, 8),
) -> Path:
    """Render and save one category-colored UMAP plot."""
    _require_columns(umap_df, ["umap_1", "umap_2", category_column], "umap_df")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = _prepare_plot_categories(
        umap_df=umap_df,
        category_column=category_column,
        top_k_categories=top_k_categories,
        other_label=other_label,
    )
    ordered_categories = (
        plot_df[category_column]
        .value_counts()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    color_map = plt.get_cmap("tab20")
    category_colors = {
        category_name: color_map(index % color_map.N)
        for index, category_name in enumerate(ordered_categories)
    }

    plt.figure(figsize=figure_size)
    for category_name in ordered_categories:
        category_df = plot_df.loc[plot_df[category_column] == category_name]
        plt.scatter(
            category_df["umap_1"],
            category_df["umap_2"],
            s=point_size,
            alpha=alpha,
            label=category_name,
            color=category_colors[category_name],
            edgecolors="none",
        )

    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(alpha=0.18)
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title=category_column,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def save_json(payload: dict[str, object], output_path: str | Path) -> None:
    """Write one small JSON file to disk with pretty indentation."""
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_classification_umap_outputs(
    umap_df: "pd.DataFrame",
    category_summary_df: "pd.DataFrame",
    output_paths: Mapping[str, str],
) -> dict[str, str]:
    """Save the core tables produced by the classification UMAP workflow."""
    umap_df.to_csv(output_paths["annotated_glycans_path"], index=False)
    umap_df.to_csv(output_paths["umap_coordinates_path"], index=False)
    category_summary_df.to_csv(output_paths["category_summary_path"], index=False)
    return dict(output_paths)
