"""Helpers for comparing similarity-scaleup outputs across model runs.

Notebook 12 uses these helpers after notebook 8 has already produced the
similarity CSVs. Nothing here loads a transformer model, so this analysis can
run on CPU-only Colab sessions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_COLUMNS = ["mean", "median", "std_dev", "min", "q05", "q25", "q75", "q95", "max"]


def _read_csv_required(path: str | Path) -> "pd.DataFrame":
    """Read one required CSV and fail with the exact missing path if needed."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required comparison input: {csv_path}")
    return pd.read_csv(csv_path)


def _parse_labels_json(label_value) -> list[str]:
    """Parse labels stored as JSON text; blank or missing labels become empty."""
    if pd.isna(label_value):
        return []
    if isinstance(label_value, list):
        return [str(value) for value in label_value]

    text_value = str(label_value).strip()
    if not text_value:
        return []

    try:
        parsed_value = json.loads(text_value)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed_value, list):
        return [str(value) for value in parsed_value]
    return []


def build_glycan_label_lookup(label_table_path: str | Path) -> dict[str, dict[str, object]]:
    """Build accession and sequence lookup dictionaries for subtype labels.

    Notebook 8's corpus uses internal `test_row_...` IDs, so the useful join key
    is usually the glycan sequence. Query glycans may have real accessions, so we
    keep accession and sequence lookup maps.
    """
    label_df = _read_csv_required(label_table_path)
    required_columns = {"glycan_id", "sequence", "labels_json"}
    missing_columns = sorted(required_columns - set(label_df.columns))
    if missing_columns:
        raise ValueError(f"Label table is missing required columns: {missing_columns}")

    records_by_accession: dict[str, dict[str, object]] = {}
    records_by_sequence: dict[str, dict[str, object]] = {}

    for row in label_df.itertuples(index=False):
        accession = str(getattr(row, "glycan_id")).strip()
        sequence = str(getattr(row, "sequence")).strip()
        labels = _parse_labels_json(getattr(row, "labels_json"))
        label_set = frozenset(labels)
        record = {
            "glycan_id": accession,
            "sequence": sequence,
            "labels": labels,
            "label_set": label_set,
        }
        if accession:
            records_by_accession[accession] = record
        if sequence and sequence not in records_by_sequence:
            records_by_sequence[sequence] = record

    return {
        "by_accession": records_by_accession,
        "by_sequence": records_by_sequence,
    }


def load_similarity_model_outputs(run_specs: list[dict[str, str | Path]]) -> dict[str, "pd.DataFrame"]:
    """Load the core notebook-8 CSVs for several model runs."""
    all_vs_all_rows = []
    specific_rows = []
    threshold_rows = []
    ranked_rows = []
    top_neighbor_rows = []

    for spec in run_specs:
        model_id = str(spec["model_id"])
        model_label = str(spec.get("model_label", model_id))
        run_dir = Path(spec["run_dir"])

        metadata = {
            "model_id": model_id,
            "model_label": model_label,
            "run_dir": str(run_dir),
        }

        all_vs_all_df = _read_csv_required(run_dir / "all_vs_all_summary.csv")
        all_vs_all_rows.append(all_vs_all_df.assign(**metadata))

        specific_df = _read_csv_required(run_dir / "specific_vs_all_distribution_summary.csv")
        specific_rows.append(specific_df.assign(**metadata))

        threshold_df = _read_csv_required(run_dir / "specific_vs_all_threshold_summary.csv")
        threshold_rows.append(threshold_df.assign(**metadata))

        ranked_df = _read_csv_required(run_dir / "specific_vs_all_ranked.csv")
        ranked_rows.append(ranked_df.assign(**metadata))

        top_neighbors_df = _read_csv_required(run_dir / "all_vs_all_top_neighbors.csv")
        top_neighbor_rows.append(top_neighbors_df.assign(**metadata))

    return {
        "all_vs_all_summary": pd.concat(all_vs_all_rows, ignore_index=True),
        "specific_vs_all_summary": pd.concat(specific_rows, ignore_index=True),
        "threshold_summary": pd.concat(threshold_rows, ignore_index=True),
        "specific_vs_all_ranked": pd.concat(ranked_rows, ignore_index=True),
        "all_vs_all_top_neighbors": pd.concat(top_neighbor_rows, ignore_index=True),
    }


def save_comparison_tables(
    comparison_tables: dict[str, "pd.DataFrame"],
    output_dir: str | Path,
) -> dict[str, str]:
    """Save comparison tables and return a name-to-path manifest."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, str] = {}
    for table_name, table_df in comparison_tables.items():
        file_path = output_path / f"{table_name}.csv"
        table_df.to_csv(file_path, index=False)
        saved_paths[table_name] = str(file_path)
    return saved_paths


def plot_all_vs_all_summary(
    all_vs_all_summary_df: "pd.DataFrame",
    output_path: str | Path,
) -> None:
    """Plot mean/median all-vs-all similarity for each model run."""
    plot_df = all_vs_all_summary_df.sort_values("model_id")
    x_positions = range(len(plot_df))

    plt.figure(figsize=(8, 5))
    plt.plot(x_positions, plot_df["mean"], marker="o", label="Mean")
    plt.plot(x_positions, plot_df["median"], marker="o", label="Median")
    plt.xticks(x_positions, plot_df["model_label"], rotation=20, ha="right")
    plt.ylabel("Cosine similarity")
    plt.title("All-vs-all similarity summary by model")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_query_similarity_summary(
    specific_summary_df: "pd.DataFrame",
    output_path: str | Path,
) -> None:
    """Plot query-specific median similarity across model runs."""
    plot_df = specific_summary_df.pivot(
        index="query_accession",
        columns="model_label",
        values="median",
    )

    ax = plot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_xlabel("Query glycan")
    ax.set_ylabel("Median cosine similarity vs test set")
    ax.set_title("Specific-vs-all median similarity by query")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_threshold_cloud_sizes(
    threshold_summary_df: "pd.DataFrame",
    output_path: str | Path,
) -> dict[str, str]:
    """Plot threshold-cloud sizes for each query and model."""
    plot_df = threshold_summary_df.copy()
    plot_df["threshold_label"] = plot_df["threshold"].map(lambda value: f">= {value:.2f}")
    saved_paths: dict[str, str] = {}

    for threshold_label, threshold_df in plot_df.groupby("threshold_label", sort=False):
        pivot_df = threshold_df.pivot(
            index="query_accession",
            columns="model_label",
            values="cloud_size",
        )
        ax = pivot_df.plot(kind="bar", figsize=(9, 5))
        ax.set_xlabel("Query glycan")
        ax.set_ylabel("Number of neighbors in cloud")
        ax.set_title(f"Similarity cloud size by model ({threshold_label})")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()

        threshold_suffix = threshold_label.replace(">= ", "ge_").replace(".", "p")
        threshold_path = Path(output_path).with_name(
            f"{Path(output_path).stem}_{threshold_suffix}{Path(output_path).suffix}"
        )
        plt.savefig(threshold_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        saved_paths[threshold_label] = str(threshold_path)

    return saved_paths


def build_top_neighbor_overlap(
    ranked_df: "pd.DataFrame",
    top_k: int = 25,
) -> "pd.DataFrame":
    """Compare how much the top query neighbors overlap between model pairs."""
    top_df = ranked_df.loc[ranked_df["rank"] <= int(top_k)].copy()
    rows = []

    for query_accession, query_df in top_df.groupby("query_accession", sort=False):
        model_ids = sorted(query_df["model_id"].unique())
        for left_index, left_model in enumerate(model_ids):
            left_neighbors = set(
                query_df.loc[query_df["model_id"] == left_model, "corpus_accession"]
            )
            for right_model in model_ids[left_index + 1 :]:
                right_neighbors = set(
                    query_df.loc[query_df["model_id"] == right_model, "corpus_accession"]
                )
                intersection_size = len(left_neighbors & right_neighbors)
                union_size = len(left_neighbors | right_neighbors)
                rows.append(
                    {
                        "query_accession": query_accession,
                        "model_a": left_model,
                        "model_b": right_model,
                        "top_k": int(top_k),
                        "overlap_count": intersection_size,
                        "jaccard_overlap": intersection_size / union_size if union_size else 0.0,
                    }
                )

    return pd.DataFrame(rows)


def build_threshold_cloud_overlap(
    ranked_df: "pd.DataFrame",
    thresholds: list[float],
) -> "pd.DataFrame":
    """Compare overlap between threshold-defined query clouds."""
    rows = []

    for query_accession, query_df in ranked_df.groupby("query_accession", sort=False):
        model_ids = sorted(query_df["model_id"].unique())
        for threshold in thresholds:
            threshold_sets = {
                model_id: set(
                    query_df.loc[
                        (query_df["model_id"] == model_id)
                        & (query_df["cosine_similarity"] >= float(threshold)),
                        "corpus_accession",
                    ]
                )
                for model_id in model_ids
            }
            for left_index, left_model in enumerate(model_ids):
                left_neighbors = threshold_sets[left_model]
                for right_model in model_ids[left_index + 1 :]:
                    right_neighbors = threshold_sets[right_model]
                    intersection_size = len(left_neighbors & right_neighbors)
                    union_size = len(left_neighbors | right_neighbors)
                    rows.append(
                        {
                            "query_accession": query_accession,
                            "threshold": float(threshold),
                            "model_a": left_model,
                            "model_b": right_model,
                            "cloud_size_a": len(left_neighbors),
                            "cloud_size_b": len(right_neighbors),
                            "overlap_count": intersection_size,
                            "jaccard_overlap": intersection_size / union_size if union_size else 0.0,
                        }
                    )

    return pd.DataFrame(rows)


def build_cloud_label_overlap_summary(
    ranked_df: "pd.DataFrame",
    label_lookup: dict[str, dict[str, dict[str, object]]],
    thresholds: list[float],
) -> "pd.DataFrame":
    """Measure whether query clouds share labels with the query glycan.

    Missing labels are counted separately as unknown instead of being treated as
    negative examples.
    """
    by_accession = label_lookup["by_accession"]
    by_sequence = label_lookup["by_sequence"]
    rows = []

    for (model_id, model_label, query_accession), query_df in ranked_df.groupby(
        ["model_id", "model_label", "query_accession"],
        sort=False,
    ):
        query_sequence = str(query_df["query_sequence"].iloc[0])
        query_record = by_accession.get(str(query_accession)) or by_sequence.get(query_sequence)
        query_labels = set(query_record["label_set"]) if query_record else set()

        for threshold in thresholds:
            cloud_df = query_df.loc[query_df["cosine_similarity"] >= float(threshold)].copy()
            exact_label_set_matches = 0
            any_label_overlap = 0
            no_label_overlap = 0
            neighbors_without_labels = 0

            for row in cloud_df.itertuples(index=False):
                corpus_sequence = str(getattr(row, "corpus_sequence"))
                corpus_record = by_sequence.get(corpus_sequence)
                corpus_labels = set(corpus_record["label_set"]) if corpus_record else set()

                if not corpus_labels:
                    neighbors_without_labels += 1
                elif corpus_labels == query_labels:
                    exact_label_set_matches += 1
                    any_label_overlap += 1
                elif query_labels & corpus_labels:
                    any_label_overlap += 1
                else:
                    no_label_overlap += 1

            labeled_neighbors = len(cloud_df) - neighbors_without_labels
            rows.append(
                {
                    "model_id": model_id,
                    "model_label": model_label,
                    "query_accession": query_accession,
                    "threshold": float(threshold),
                    "query_labels_json": json.dumps(sorted(query_labels)),
                    "cloud_size": int(len(cloud_df)),
                    "labeled_neighbors": int(labeled_neighbors),
                    "neighbors_without_labels": int(neighbors_without_labels),
                    "exact_label_set_matches": int(exact_label_set_matches),
                    "any_label_overlap": int(any_label_overlap),
                    "no_label_overlap": int(no_label_overlap),
                    "exact_label_set_match_rate": (
                        exact_label_set_matches / labeled_neighbors if labeled_neighbors else float("nan")
                    ),
                    "any_label_overlap_rate": (
                        any_label_overlap / labeled_neighbors if labeled_neighbors else float("nan")
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_similarity_model_comparison(
    run_specs: list[dict[str, str | Path]],
    label_table_path: str | Path | None,
    output_dir: str | Path,
    cloud_thresholds: list[float],
    top_k_neighbors: int = 25,
) -> dict[str, object]:
    """Run the complete no-GPU model-comparison workflow."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loaded_tables = load_similarity_model_outputs(run_specs)
    comparison_tables = {
        "all_vs_all_model_comparison": loaded_tables["all_vs_all_summary"],
        "specific_vs_all_model_comparison": loaded_tables["specific_vs_all_summary"],
        "threshold_cloud_size_model_comparison": loaded_tables["threshold_summary"],
        "top_neighbor_overlap_model_comparison": build_top_neighbor_overlap(
            loaded_tables["specific_vs_all_ranked"],
            top_k=top_k_neighbors,
        ),
        "threshold_cloud_overlap_model_comparison": build_threshold_cloud_overlap(
            loaded_tables["specific_vs_all_ranked"],
            thresholds=cloud_thresholds,
        ),
    }

    if label_table_path is not None:
        label_lookup = build_glycan_label_lookup(label_table_path)
        comparison_tables["cloud_label_overlap_model_comparison"] = build_cloud_label_overlap_summary(
            loaded_tables["specific_vs_all_ranked"],
            label_lookup=label_lookup,
            thresholds=cloud_thresholds,
        )

    saved_table_paths = save_comparison_tables(comparison_tables, output_path)

    plot_paths = {
        "all_vs_all_plot": str(output_path / "all_vs_all_model_comparison.png"),
        "specific_vs_all_plot": str(output_path / "specific_vs_all_query_medians.png"),
    }
    plot_all_vs_all_summary(
        comparison_tables["all_vs_all_model_comparison"],
        plot_paths["all_vs_all_plot"],
    )
    plot_query_similarity_summary(
        comparison_tables["specific_vs_all_model_comparison"],
        plot_paths["specific_vs_all_plot"],
    )
    threshold_plot_paths = plot_threshold_cloud_sizes(
        comparison_tables["threshold_cloud_size_model_comparison"],
        output_path / "threshold_cloud_size_model_comparison.png",
    )
    plot_paths["threshold_cloud_size_plots"] = threshold_plot_paths

    manifest = {
        "output_dir": str(output_path),
        "table_paths": saved_table_paths,
        "plot_paths": plot_paths,
        "run_specs": [
            {**spec, "run_dir": str(spec["run_dir"])}
            for spec in run_specs
        ],
        "cloud_thresholds": cloud_thresholds,
        "top_k_neighbors": int(top_k_neighbors),
    }
    manifest_path = output_path / "similarity_model_comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "tables": comparison_tables,
        "table_paths": saved_table_paths,
        "plot_paths": plot_paths,
        "manifest_path": str(manifest_path),
    }
