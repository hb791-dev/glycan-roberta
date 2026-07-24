"""Helpers for comparing similarity-scaleup outputs across model runs.

Notebook 12 uses these helpers after notebook 8 has already produced the
similarity CSVs. Nothing here loads a transformer model, so this analysis can
run on CPU-only Colab sessions.
"""

from __future__ import annotations

import json
import shutil
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_COLUMNS = ["mean", "median", "std_dev", "min", "q05", "q25", "q75", "q95", "max"]


HTML_STYLE = """
:root {
  --ink: #1d2433;
  --muted: #667085;
  --paper: #f7f3ea;
  --card: #fffaf0;
  --line: #dfd5c4;
  --accent: #a84f2a;
  --good: #1f7a4d;
  --warn: #a15c00;
  --bad: #a12b2b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: radial-gradient(circle at top left, #fff6dd 0, var(--paper) 38%, #ede4d2 100%);
  font-family: Georgia, "Times New Roman", serif;
  line-height: 1.45;
}
header {
  padding: 34px 42px 22px;
  border-bottom: 1px solid var(--line);
}
h1, h2, h3 { margin: 0 0 10px; line-height: 1.1; }
h1 { font-size: 34px; }
h2 { font-size: 24px; margin-top: 30px; }
h3 { font-size: 18px; }
.subtle { color: var(--muted); }
.container { padding: 24px 42px 48px; }
.summary-grid, .plot-grid, .model-grid {
  display: grid;
  gap: 16px;
}
.summary-grid {
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}
.plot-grid {
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
}
.model-grid {
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
}
.card, .plot-card, .query-section {
  background: rgba(255, 250, 240, 0.92);
  border: 1px solid var(--line);
  border-radius: 18px;
  box-shadow: 0 10px 28px rgba(91, 67, 37, 0.08);
}
.card { padding: 16px; }
.plot-card { padding: 14px; }
.plot-card img {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 12px;
  background: white;
}
.query-section {
  padding: 20px;
  margin: 22px 0;
}
.query-head {
  display: grid;
  grid-template-columns: minmax(180px, 260px) 1fr;
  gap: 18px;
  align-items: start;
  margin-bottom: 18px;
}
.cartoon-box {
  min-height: 120px;
  display: grid;
  place-items: center;
  background: white;
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 10px;
}
.cartoon-box img {
  max-width: 100%;
  max-height: 170px;
}
.missing-cartoon {
  color: var(--muted);
  font-size: 13px;
  text-align: center;
}
.sequence {
  max-height: 75px;
  overflow: auto;
  padding: 8px;
  background: #fff;
  border: 1px dashed var(--line);
  border-radius: 10px;
  font-family: "Menlo", "Consolas", monospace;
  font-size: 12px;
}
.badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  border: 1px solid var(--line);
  padding: 3px 8px;
  font-size: 12px;
  background: #fff;
}
.badge.good { color: var(--good); border-color: rgba(31, 122, 77, 0.35); }
.badge.warn { color: var(--warn); border-color: rgba(161, 92, 0, 0.35); }
.badge.bad { color: var(--bad); border-color: rgba(161, 43, 43, 0.35); }
.neighbor-card {
  margin: 10px 0;
  padding: 12px;
  border-radius: 14px;
  border: 1px solid var(--line);
  background: #fffdf7;
}
.neighbor-title {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-weight: bold;
}
.mini-cartoon {
  margin: 8px 0;
  background: white;
  border: 1px solid #eee2d0;
  border-radius: 12px;
  padding: 8px;
  min-height: 95px;
  display: grid;
  place-items: center;
}
.mini-cartoon img {
  max-width: 100%;
  max-height: 110px;
}
table {
  width: 100%;
  border-collapse: collapse;
  background: white;
  border-radius: 14px;
  overflow: hidden;
}
th, td {
  border-bottom: 1px solid var(--line);
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
  font-size: 13px;
}
th { background: #efe3cf; }
@media (max-width: 760px) {
  header, .container { padding-left: 18px; padding-right: 18px; }
  .query-head { grid-template-columns: 1fr; }
}
"""


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


def _safe_stem(value: str) -> str:
    """Create a stable filename-friendly stem from a model or accession label."""
    safe_chars = []
    for character in str(value):
        if character.isalnum() or character in {"-", "_"}:
            safe_chars.append(character)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_") or "item"


def _relative_path(path: str | Path, start_dir: str | Path) -> str:
    """Return a browser-friendly relative path for files used by the HTML report."""
    return Path(path).resolve().relative_to(Path(start_dir).resolve()).as_posix()


def _resolve_cartoon_source_path(local_image_path: str | Path, run_dir: str | Path) -> Path | None:
    """Find a cached cartoon image even when the manifest came from Colab paths.

    The manifests store `/content/drive/...` paths in Colab. On a local Mac those
    exact paths do not exist, but the same SVG filenames usually exist inside the
    run's `cartoons` folder. This fallback makes local smoke tests work too.
    """
    if pd.isna(local_image_path):
        return None

    candidate_path = Path(str(local_image_path))
    if candidate_path.exists():
        return candidate_path

    fallback_path = Path(run_dir) / "cartoons" / candidate_path.name
    if fallback_path.exists():
        return fallback_path

    return None


def _copy_cartoon_for_report(
    source_path: Path | None,
    asset_dir: Path,
    image_key: str,
) -> str | None:
    """Copy one cached cartoon into the report assets folder and return its path."""
    if source_path is None:
        return None

    asset_dir.mkdir(parents=True, exist_ok=True)
    output_path = asset_dir / f"{_safe_stem(image_key)}{source_path.suffix.lower()}"
    if not output_path.exists():
        shutil.copy2(source_path, output_path)
    return str(output_path)


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

    for model_order, spec in enumerate(run_specs):
        model_id = str(spec["model_id"])
        model_label = str(spec.get("model_label", model_id))
        run_dir = Path(spec["run_dir"])

        metadata = {
            "model_id": model_id,
            "model_label": model_label,
            "model_order": model_order,
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


def build_cartoon_lookup_from_run_manifests(
    run_specs: list[dict[str, str | Path]],
) -> dict[str, dict[str, str]]:
    """Build a sequence/accession lookup for cartoons cached by notebook 8.

    The comparison report only uses cartoons that notebook 8 already cached. This
    keeps notebook 12 lightweight and makes it safe to rerun on CPU-only Colab.
    """
    lookup: dict[str, dict[str, str]] = {
        "by_sequence": {},
        "by_accession": {},
    }
    for spec in run_specs:
        run_dir = Path(spec["run_dir"])
        manifest_path = run_dir / "scaleup_cartoon_manifest.csv"
        if not manifest_path.exists():
            continue

        manifest_df = pd.read_csv(manifest_path)
        for row in manifest_df.itertuples(index=False):
            sequence = str(getattr(row, "sequence", "")).strip()
            accession = str(getattr(row, "accession", "")).strip()
            local_image_path = getattr(row, "local_image_path", "")
            source_path = _resolve_cartoon_source_path(local_image_path, run_dir)
            if source_path is None:
                continue

            record = {
                "sequence": sequence,
                "accession": "" if accession.lower() == "nan" else accession,
                "image_path": str(source_path),
            }
            if sequence and sequence not in lookup["by_sequence"]:
                lookup["by_sequence"][sequence] = record
            if record["accession"] and record["accession"] not in lookup["by_accession"]:
                lookup["by_accession"][record["accession"]] = record

    return lookup


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
    plot_df = all_vs_all_summary_df.sort_values("model_order")
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
    model_order = (
        specific_summary_df[["model_label", "model_order"]]
        .drop_duplicates()
        .sort_values("model_order")["model_label"]
        .tolist()
    )
    plot_df = specific_summary_df.pivot(
        index="query_accession",
        columns="model_label",
        values="median",
    )
    plot_df = plot_df[model_order]

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
    model_order = (
        plot_df[["model_label", "model_order"]]
        .drop_duplicates()
        .sort_values("model_order")["model_label"]
        .tolist()
    )
    saved_paths: dict[str, str] = {}

    for threshold_label, threshold_df in plot_df.groupby("threshold_label", sort=False):
        pivot_df = threshold_df.pivot(
            index="query_accession",
            columns="model_label",
            values="cloud_size",
        )
        pivot_df = pivot_df[model_order]
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


def build_threshold_cloud_size_summary(
    ranked_df: "pd.DataFrame",
    thresholds: list[float],
) -> "pd.DataFrame":
    """Build cloud-size summaries directly from ranked query similarities.

    Notebook 8 may only save a threshold summary for one cutoff. Recomputing the
    size here keeps notebook 12's cloud-size plots aligned with the thresholds
    used for cloud overlap and label-overlap analysis.
    """
    rows = []

    group_columns = ["model_id", "model_label", "model_order", "run_dir", "query_accession"]
    for group_values, query_df in ranked_df.groupby(group_columns, sort=False):
        model_id, model_label, model_order, run_dir, query_accession = group_values
        query_sequence = str(query_df["query_sequence"].iloc[0])

        for threshold in thresholds:
            cloud_df = query_df.loc[query_df["cosine_similarity"] >= float(threshold)]
            rows.append(
                {
                    "query_accession": query_accession,
                    "query_sequence": query_sequence,
                    "threshold": float(threshold),
                    "cloud_size": int(len(cloud_df)),
                    "max_similarity": (
                        float(cloud_df["cosine_similarity"].max()) if len(cloud_df) else float("nan")
                    ),
                    "min_similarity": (
                        float(cloud_df["cosine_similarity"].min()) if len(cloud_df) else float("nan")
                    ),
                    "model_id": model_id,
                    "model_label": model_label,
                    "model_order": int(model_order),
                    "run_dir": run_dir,
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

    for (model_id, model_label, model_order, query_accession), query_df in ranked_df.groupby(
        ["model_id", "model_label", "model_order", "query_accession"],
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
                    "model_order": int(model_order),
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


def _lookup_labels(
    label_lookup: dict[str, dict[str, dict[str, object]]] | None,
    accession: str,
    sequence: str,
) -> list[str]:
    """Find subtype labels by accession first, then by sequence."""
    if label_lookup is None:
        return []

    by_accession = label_lookup["by_accession"]
    by_sequence = label_lookup["by_sequence"]
    record = by_accession.get(str(accession)) or by_sequence.get(str(sequence))
    if not record:
        return []
    return list(record["labels"])


def _lookup_cartoon(
    cartoon_lookup: dict[str, dict[str, str]],
    accession: str,
    sequence: str,
) -> dict[str, str] | None:
    """Find a cached cartoon by accession or sequence."""
    return (
        cartoon_lookup["by_accession"].get(str(accession))
        or cartoon_lookup["by_sequence"].get(str(sequence))
    )


def _label_relation(query_labels: list[str], neighbor_labels: list[str]) -> str:
    """Describe how a neighbor's labels relate to the query labels."""
    query_set = set(query_labels)
    neighbor_set = set(neighbor_labels)
    if not neighbor_set:
        return "no_prepared_label"
    if neighbor_set == query_set:
        return "exact_label_set"
    if query_set & neighbor_set:
        return "shared_label"
    return "no_label_overlap"


def build_html_neighbor_gallery_table(
    ranked_df: "pd.DataFrame",
    label_lookup: dict[str, dict[str, dict[str, object]]] | None,
    cartoon_lookup: dict[str, dict[str, str]],
    cloud_threshold: float,
    top_n_neighbors: int,
) -> "pd.DataFrame":
    """Build the small neighbor table used by the HTML gallery report."""
    cloud_membership: dict[tuple[str, str], set[str]] = {}
    threshold_df = ranked_df.loc[ranked_df["cosine_similarity"] >= float(cloud_threshold)].copy()

    for row in threshold_df.itertuples(index=False):
        key = (str(getattr(row, "query_accession")), str(getattr(row, "corpus_sequence")))
        cloud_membership.setdefault(key, set()).add(str(getattr(row, "model_label")))

    rows = []
    top_df = ranked_df.loc[ranked_df["rank"] <= int(top_n_neighbors)].copy()
    for row in top_df.itertuples(index=False):
        query_accession = str(getattr(row, "query_accession"))
        query_sequence = str(getattr(row, "query_sequence"))
        corpus_accession = str(getattr(row, "corpus_accession"))
        corpus_sequence = str(getattr(row, "corpus_sequence"))

        query_labels = _lookup_labels(label_lookup, query_accession, query_sequence)
        neighbor_labels = _lookup_labels(label_lookup, corpus_accession, corpus_sequence)
        neighbor_cartoon = _lookup_cartoon(cartoon_lookup, corpus_accession, corpus_sequence)
        query_cartoon = _lookup_cartoon(cartoon_lookup, query_accession, query_sequence)
        in_models = sorted(cloud_membership.get((query_accession, corpus_sequence), set()))

        rows.append(
            {
                "query_accession": query_accession,
                "query_sequence": query_sequence,
                "query_labels_json": json.dumps(sorted(query_labels)),
                "query_cartoon_path": query_cartoon["image_path"] if query_cartoon else "",
                "model_id": str(getattr(row, "model_id")),
                "model_label": str(getattr(row, "model_label")),
                "model_order": int(getattr(row, "model_order")),
                "rank": int(getattr(row, "rank")),
                "cosine_similarity": float(getattr(row, "cosine_similarity")),
                "corpus_accession": corpus_accession,
                "display_accession": (
                    neighbor_cartoon["accession"]
                    if neighbor_cartoon and neighbor_cartoon.get("accession")
                    else corpus_accession
                ),
                "corpus_sequence": corpus_sequence,
                "neighbor_labels_json": json.dumps(sorted(neighbor_labels)),
                "label_relation": _label_relation(query_labels, neighbor_labels),
                "neighbor_cartoon_path": neighbor_cartoon["image_path"] if neighbor_cartoon else "",
                "in_models_at_threshold_json": json.dumps(in_models),
            }
        )

    return pd.DataFrame(rows)


def _render_labels(labels_json: str) -> str:
    """Render a compact list of labels as HTML badges."""
    labels = _parse_labels_json(labels_json)
    if not labels:
        return '<span class="badge warn">no prepared label</span>'
    return "".join(f'<span class="badge">{escape(label)}</span>' for label in labels)


def _render_relation_badge(relation: str) -> str:
    """Render the neighbor/query label relationship."""
    badge_class = {
        "exact_label_set": "good",
        "shared_label": "good",
        "no_prepared_label": "warn",
        "no_label_overlap": "bad",
    }.get(relation, "")
    label = {
        "exact_label_set": "exact label set",
        "shared_label": "shares label",
        "no_prepared_label": "no prepared label",
        "no_label_overlap": "no label overlap",
    }.get(relation, relation)
    return f'<span class="badge {badge_class}">{escape(label)}</span>'


def _render_cartoon_image(image_path: str, output_dir: Path, mini: bool = False) -> str:
    """Render a cached cartoon image if available."""
    css_class = "mini-cartoon" if mini else "cartoon-box"
    if not image_path:
        return f'<div class="{css_class}"><div class="missing-cartoon">cartoon not cached</div></div>'

    copied_path = _copy_cartoon_for_report(
        Path(image_path) if Path(image_path).exists() else None,
        output_dir / "html_assets" / "cartoons",
        Path(image_path).stem,
    )
    if copied_path is None:
        return f'<div class="{css_class}"><div class="missing-cartoon">cartoon file missing</div></div>'

    relative_image_path = _relative_path(copied_path, output_dir)
    return (
        f'<div class="{css_class}">'
        f'<img src="{escape(relative_image_path)}" alt="glycan cartoon">'
        "</div>"
    )


def _render_plot_image(plot_path: str, output_dir: Path, title: str) -> str:
    """Render one saved PNG plot inside the HTML report."""
    if not plot_path or not Path(plot_path).exists():
        return ""
    relative_plot_path = _relative_path(plot_path, output_dir)
    return (
        '<div class="plot-card">'
        f"<h3>{escape(title)}</h3>"
        f'<img src="{escape(relative_plot_path)}" alt="{escape(title)}">'
        "</div>"
    )


def render_similarity_comparison_html_report(
    output_dir: str | Path,
    report_title: str,
    comparison_tables: dict[str, "pd.DataFrame"],
    plot_paths: dict[str, object],
    gallery_table_df: "pd.DataFrame",
    cloud_threshold: float,
    top_n_neighbors: int,
) -> str:
    """Render a professor-friendly HTML report for model similarity comparison."""
    output_path = Path(output_dir)
    html_path = output_path / "similarity_model_comparison_report.html"

    all_vs_all_df = comparison_tables["all_vs_all_model_comparison"].sort_values("model_order")
    label_overlap_df = comparison_tables.get("cloud_label_overlap_model_comparison", pd.DataFrame())
    top_overlap_df = comparison_tables["top_neighbor_overlap_model_comparison"]

    all_vs_all_rows = []
    for row in all_vs_all_df.itertuples(index=False):
        all_vs_all_rows.append(
            "<tr>"
            f"<td>{escape(str(row.model_label))}</td>"
            f"<td>{float(row.mean):.3f}</td>"
            f"<td>{float(row.median):.3f}</td>"
            f"<td>{float(row.q95):.3f}</td>"
            "</tr>"
        )

    label_summary_html = ""
    if not label_overlap_df.empty:
        threshold_label_df = label_overlap_df.loc[
            label_overlap_df["threshold"].round(6) == round(float(cloud_threshold), 6)
        ]
        summary_df = (
            threshold_label_df.groupby(["model_order", "model_label"], as_index=False)[
                ["exact_label_set_match_rate", "any_label_overlap_rate", "cloud_size"]
            ]
            .mean()
            .sort_values("model_order")
        )
        rows = []
        for row in summary_df.itertuples(index=False):
            rows.append(
                "<tr>"
                f"<td>{escape(str(row.model_label))}</td>"
                f"<td>{float(row.exact_label_set_match_rate):.3f}</td>"
                f"<td>{float(row.any_label_overlap_rate):.3f}</td>"
                f"<td>{float(row.cloud_size):.1f}</td>"
                "</tr>"
            )
        label_summary_html = (
            "<h2>Cloud label consistency</h2>"
            f"<p class=\"subtle\">Averaged across query glycans at threshold >= {cloud_threshold:.2f}.</p>"
            "<table><thead><tr><th>Model</th><th>Exact label-set match</th>"
            "<th>Any label overlap</th><th>Mean cloud size</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    pair_rows = []
    pair_summary_df = (
        top_overlap_df.groupby(["model_a", "model_b"], as_index=False)[["overlap_count", "jaccard_overlap"]]
        .mean()
        .sort_values(["model_a", "model_b"])
    )
    for row in pair_summary_df.itertuples(index=False):
        pair_rows.append(
            "<tr>"
            f"<td>{escape(str(row.model_a))}</td>"
            f"<td>{escape(str(row.model_b))}</td>"
            f"<td>{float(row.overlap_count):.1f}</td>"
            f"<td>{float(row.jaccard_overlap):.3f}</td>"
            "</tr>"
        )

    threshold_plot_paths = plot_paths.get("threshold_cloud_size_plots", {})
    threshold_plot_key = f">= {cloud_threshold:.2f}"
    plot_html = "".join(
        [
            _render_plot_image(str(plot_paths.get("all_vs_all_plot", "")), output_path, "Whole-space similarity"),
            _render_plot_image(str(plot_paths.get("specific_vs_all_plot", "")), output_path, "Query median similarity"),
            _render_plot_image(
                str(threshold_plot_paths.get(threshold_plot_key, "")),
                output_path,
                f"Cloud sizes at >= {cloud_threshold:.2f}",
            ),
        ]
    )

    query_sections = []
    for query_accession, query_df in gallery_table_df.groupby("query_accession", sort=False):
        first_row = query_df.iloc[0]
        query_labels_html = _render_labels(str(first_row["query_labels_json"]))
        query_cartoon_html = _render_cartoon_image(
            str(first_row["query_cartoon_path"]),
            output_path,
            mini=False,
        )

        model_columns = []
        for model_label, model_df in query_df.sort_values(["model_order", "rank"]).groupby(
            "model_label",
            sort=False,
        ):
            neighbor_cards = []
            for row in model_df.itertuples(index=False):
                in_models = _parse_labels_json(getattr(row, "in_models_at_threshold_json"))
                in_model_badges = "".join(
                    f'<span class="badge">{escape(model)}</span>' for model in in_models
                )
                if not in_model_badges:
                    in_model_badges = '<span class="badge warn">none</span>'
                neighbor_cards.append(
                    '<div class="neighbor-card">'
                    '<div class="neighbor-title">'
                    f'<span>#{int(row.rank)} {escape(str(row.display_accession))}</span>'
                    f'<span>{float(row.cosine_similarity):.3f}</span>'
                    "</div>"
                    f'{_render_cartoon_image(str(row.neighbor_cartoon_path), output_path, mini=True)}'
                    '<div class="badge-row">'
                    f'{_render_relation_badge(str(row.label_relation))}'
                    f"{_render_labels(str(row.neighbor_labels_json))}"
                    "</div>"
                    '<div class="badge-row">'
                    '<span class="badge warn">in threshold cloud:</span>'
                    f"{in_model_badges}"
                    "</div>"
                    f'<div class="sequence">{escape(str(row.corpus_sequence))}</div>'
                    "</div>"
                )

            model_columns.append(
                '<div class="card">'
                f"<h3>{escape(str(model_label))}</h3>"
                f"<p class=\"subtle\">Top {top_n_neighbors} neighbors from this model.</p>"
                f"{''.join(neighbor_cards)}"
                "</div>"
            )

        query_sections.append(
            '<section class="query-section">'
            f"<h2>Query {escape(str(query_accession))}</h2>"
            '<div class="query-head">'
            f"{query_cartoon_html}"
            "<div>"
            '<div class="badge-row">'
            f"{query_labels_html}"
            "</div>"
            f'<div class="sequence">{escape(str(first_row["query_sequence"]))}</div>'
            "</div>"
            "</div>"
            '<div class="model-grid">'
            f"{''.join(model_columns)}"
            "</div>"
            "</section>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(report_title)}</title>
  <style>{HTML_STYLE}</style>
</head>
<body>
  <header>
    <h1>{escape(report_title)}</h1>
    <p class="subtle">Reusable similarity comparison report. It reads notebook 8 outputs and does not load a model.</p>
  </header>
  <main class="container">
    <div class="summary-grid">
      <div class="card">
        <h3>What this report checks</h3>
        <p>Do different model states retrieve the same glycan neighborhoods, and do those neighborhoods look semantically meaningful by subtype labels and cartoons?</p>
      </div>
      <div class="card">
        <h3>Gallery settings</h3>
        <p>Cloud threshold: <strong>>= {cloud_threshold:.2f}</strong><br>Neighbors shown per query/model: <strong>{top_n_neighbors}</strong></p>
      </div>
    </div>

    <h2>Saved comparison plots</h2>
    <div class="plot-grid">{plot_html}</div>

    <h2>Whole-space summary</h2>
    <table>
      <thead><tr><th>Model</th><th>Mean</th><th>Median</th><th>Q95</th></tr></thead>
      <tbody>{''.join(all_vs_all_rows)}</tbody>
    </table>

    {label_summary_html}

    <h2>Top-neighbor overlap</h2>
    <p class="subtle">Lower overlap means the models are choosing different nearest-neighbor sets.</p>
    <table>
      <thead><tr><th>Model A</th><th>Model B</th><th>Mean overlap count</th><th>Mean Jaccard</th></tr></thead>
      <tbody>{''.join(pair_rows)}</tbody>
    </table>

    <h2>Query galleries</h2>
    <p class="subtle">These sections are the visual sanity check: same query, model-specific nearest neighbors, cartoons, labels, and threshold-cloud membership.</p>
    {''.join(query_sections)}
  </main>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return str(html_path)


def build_similarity_model_comparison(
    run_specs: list[dict[str, str | Path]],
    label_table_path: str | Path | None,
    output_dir: str | Path,
    cloud_thresholds: list[float],
    top_k_neighbors: int = 25,
    html_cloud_threshold: float = 0.90,
    html_top_n_neighbors: int = 8,
    report_title: str = "Similarity Model Comparison",
) -> dict[str, object]:
    """Run the complete no-GPU model-comparison workflow."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loaded_tables = load_similarity_model_outputs(run_specs)
    label_lookup = None
    if label_table_path is not None:
        label_lookup = build_glycan_label_lookup(label_table_path)

    cartoon_lookup = build_cartoon_lookup_from_run_manifests(run_specs)

    comparison_tables = {
        "all_vs_all_model_comparison": loaded_tables["all_vs_all_summary"],
        "specific_vs_all_model_comparison": loaded_tables["specific_vs_all_summary"],
        "threshold_cloud_size_model_comparison": build_threshold_cloud_size_summary(
            loaded_tables["specific_vs_all_ranked"],
            thresholds=cloud_thresholds,
        ),
        "top_neighbor_overlap_model_comparison": build_top_neighbor_overlap(
            loaded_tables["specific_vs_all_ranked"],
            top_k=top_k_neighbors,
        ),
        "threshold_cloud_overlap_model_comparison": build_threshold_cloud_overlap(
            loaded_tables["specific_vs_all_ranked"],
            thresholds=cloud_thresholds,
        ),
        "html_neighbor_gallery_table": build_html_neighbor_gallery_table(
            loaded_tables["specific_vs_all_ranked"],
            label_lookup=label_lookup,
            cartoon_lookup=cartoon_lookup,
            cloud_threshold=html_cloud_threshold,
            top_n_neighbors=html_top_n_neighbors,
        ),
    }

    if label_lookup is not None:
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

    html_report_path = render_similarity_comparison_html_report(
        output_dir=output_path,
        report_title=report_title,
        comparison_tables=comparison_tables,
        plot_paths=plot_paths,
        gallery_table_df=comparison_tables["html_neighbor_gallery_table"],
        cloud_threshold=html_cloud_threshold,
        top_n_neighbors=html_top_n_neighbors,
    )
    plot_paths["html_report_path"] = html_report_path

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
        "html_cloud_threshold": float(html_cloud_threshold),
        "html_top_n_neighbors": int(html_top_n_neighbors),
        "report_title": report_title,
    }
    manifest_path = output_path / "similarity_model_comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "tables": comparison_tables,
        "table_paths": saved_table_paths,
        "plot_paths": plot_paths,
        "manifest_path": str(manifest_path),
    }
