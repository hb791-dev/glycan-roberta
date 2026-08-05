"""Helpers for matched CLS/mean/max similarity-pooling comparisons.

Notebook 13 uses these helpers after notebook 8 has already produced the
similarity CSVs for one exact checkpoint under three pooling strategies.
Nothing here loads transformer weights, so the workflow can run on CPU-only
Colab sessions.
"""

from __future__ import annotations

from collections.abc import Sequence
import base64
import itertools
import json
import mimetypes
import shutil
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.similarity_core import normalize_pooling_strategy
from src.similarity_scaleup import (
    _iter_local_html_references,
    _resolve_local_export_reference,
    _scan_text_for_sensitive_strings,
)


POOLING_ORDER = ("cls", "mean", "max")
MATCH_KEYS = [
    "query_accession",
    "corpus_accession",
    "query_sequence",
    "corpus_sequence",
]
SUMMARY_COLUMNS = ["mean", "median", "std_dev", "min", "q05", "q25", "q75", "q95", "max"]

REPORT_PATH_PREFIXES = (
    "/content/drive/MyDrive/",
    "/drive/MyDrive/",
    "file:///content/drive/MyDrive/",
    "file:///drive/MyDrive/",
)


HTML_STYLE = """
:root {
  --ink: #1f2937;
  --muted: #5b6472;
  --paper: #f3f5f8;
  --card: #ffffff;
  --line: #d7dde6;
  --line-strong: #bcc6d4;
  --accent: #0f4c81;
  --accent-soft: #e8f1fb;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, #ffffff 0%, rgba(255, 255, 255, 0) 34%),
    linear-gradient(180deg, #f7f9fc 0%, #eef2f7 100%);
  font-family: "Aptos", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  line-height: 1.55;
}
header {
  padding: 32px 40px 24px;
  border-bottom: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.84);
  backdrop-filter: blur(10px);
}
.container {
  max-width: 1680px;
  margin: 0 auto;
  padding: 24px 40px 56px;
}
.card {
  margin: 18px 0;
  padding: 22px 24px;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
}
h1, h2, h3 { margin: 0 0 12px; line-height: 1.1; }
h1 {
  font-size: 34px;
  font-weight: 700;
  letter-spacing: -0.03em;
}
h2 {
  font-size: 22px;
  margin-top: 2px;
  font-weight: 650;
  letter-spacing: -0.02em;
}
h3 { font-size: 18px; }
p, li { margin: 0 0 10px; }
ul { margin: 0; padding-left: 20px; }
.subtle { color: var(--muted); }
.plot-wrap img {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 12px;
  background: white;
  border: 1px solid var(--line);
}
.table-scroll {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: white;
}
table.report-table {
  width: 100%;
  border-collapse: collapse;
  background: white;
  min-width: 980px;
}
table.report-table th,
table.report-table td {
  border-bottom: 1px solid var(--line);
  padding: 10px 12px;
  text-align: left;
  vertical-align: top;
  font-size: 13px;
  white-space: normal;
  word-break: break-word;
}
table.report-table thead th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: #f7f9fc;
  border-bottom: 1px solid var(--line-strong);
  font-weight: 650;
}
table.report-table tbody tr:nth-child(even) {
  background: #fbfcfe;
}
table.report-table tbody tr:hover {
  background: var(--accent-soft);
}
table.report-table td:nth-child(3),
table.report-table td:nth-child(4) {
  min-width: 300px;
}
.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 12px 0 0;
}
.badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  border: 1px solid var(--line);
  padding: 5px 10px;
  background: var(--accent-soft);
  font-size: 12px;
  color: var(--accent);
  font-weight: 600;
}
.mono {
  display: inline-block;
  padding: 8px 10px;
  border-radius: 10px;
  background: #f7f9fc;
  border: 1px solid var(--line);
  font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
  font-size: 12px;
}
.callout {
  margin-top: 14px;
  padding: 14px 16px;
  border-radius: 12px;
  border: 1px solid var(--line);
  background: #f9fbfd;
}
.callout h3 {
  margin-bottom: 8px;
}
.callout ul {
  margin: 0;
  padding-left: 18px;
}
.callout li {
  margin-bottom: 6px;
}
.term-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px 18px;
  margin-top: 12px;
}
.term-card {
  padding: 12px 14px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #fbfcfe;
}
.term-card strong {
  display: block;
  margin-bottom: 4px;
  color: var(--accent);
}
.gallery-grid {
  display: grid;
  gap: 18px;
}
.query-gallery {
  border: 1px solid var(--line);
  border-radius: 16px;
  background: #fbfcfe;
  overflow: hidden;
}
.query-gallery-header {
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr);
  gap: 18px;
  padding: 18px;
  border-bottom: 1px solid var(--line);
  background: white;
}
.query-gallery-copy h3 {
  margin-bottom: 6px;
}
.query-gallery-copy p {
  color: var(--muted);
}
.cartoon-box,
.mini-cartoon {
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: white;
  overflow: hidden;
}
.cartoon-box {
  min-height: 160px;
  padding: 10px;
}
.mini-cartoon {
  width: 76px;
  min-width: 76px;
  height: 76px;
  padding: 6px;
}
.cartoon-box img,
.mini-cartoon img {
  display: block;
  max-width: 100%;
  max-height: 100%;
}
.missing-cartoon {
  color: var(--muted);
  font-size: 12px;
  text-align: center;
}
.query-neighbor-table {
  padding: 0 18px 18px;
}
.neighbor-cartoon-cell {
  width: 94px;
}
.numeric {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
@media (max-width: 760px) {
  header, .container { padding-left: 18px; padding-right: 18px; }
  .card { padding: 18px; }
  h1 { font-size: 28px; }
  h2 { font-size: 20px; }
  .query-gallery-header { grid-template-columns: 1fr; }
}
"""


def _read_csv_required(path: str | Path) -> "pd.DataFrame":
    """Read one required CSV and fail with the exact missing path if needed."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required pooling-comparison input: {csv_path}")
    return pd.read_csv(csv_path)


def _safe_stem(value: str) -> str:
    """Return one filesystem-safe stem."""
    safe_chars = []
    for character in str(value):
        if character.isalnum() or character in {"-", "_"}:
            safe_chars.append(character)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_") or "item"


def _stringify_pathlike_values(record: dict[str, object]) -> dict[str, object]:
    """Convert Path-like values before saving notebook-facing JSON."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in record.items()
    }


def build_notebook8_output_run_label(
    base_output_run_label: str,
    tokenizer_family: str,
    model_output_id: str,
    pooling_strategy: str,
) -> str:
    """Return the notebook-8 output folder name for one pooling strategy."""
    normalized_pooling = normalize_pooling_strategy(pooling_strategy)
    pooling_label = f"{normalized_pooling}_pool"
    output_label = f"{tokenizer_family}__{model_output_id}__{pooling_label}"
    return f"{base_output_run_label}__{output_label}"


def build_pooling_run_specs(
    scaleup_results_root: str | Path,
    tokenizer_family: str,
    experiment_name: str,
    checkpoint_source: str,
    model_output_id: str,
    base_output_run_label: str,
    classifier_run_label: str | None = None,
    pooling_strategies: Sequence[str] = POOLING_ORDER,
) -> list[dict[str, str]]:
    """Build the matched notebook-8 run directories for one exact checkpoint."""
    results_root = Path(scaleup_results_root)
    checkpoint_source = str(checkpoint_source).strip().lower()
    if checkpoint_source not in {"pretraining", "classification"}:
        raise ValueError("checkpoint_source must be either 'pretraining' or 'classification'.")

    if checkpoint_source == "classification":
        if not str(classifier_run_label or "").strip():
            raise ValueError("classifier_run_label is required for classification pooling comparisons.")
        parent_dir = (
            results_root
            / "classification"
            / tokenizer_family
            / experiment_name
            / str(classifier_run_label)
        )
    else:
        parent_dir = results_root / tokenizer_family / experiment_name

    run_specs = []
    for pooling_strategy in pooling_strategies:
        normalized_pooling = normalize_pooling_strategy(pooling_strategy)
        run_specs.append(
            {
                "pooling_strategy": normalized_pooling,
                "model_id": str(model_output_id),
                "model_label": str(model_output_id),
                "run_dir": str(
                    parent_dir
                    / build_notebook8_output_run_label(
                        base_output_run_label=base_output_run_label,
                        tokenizer_family=tokenizer_family,
                        model_output_id=model_output_id,
                        pooling_strategy=normalized_pooling,
                    )
                ),
            }
        )
    return run_specs


def _sort_like_notebook8(frame_df: "pd.DataFrame", columns: Sequence[str]) -> "pd.DataFrame":
    """Return one stable sort by key columns with reset row order."""
    return frame_df.sort_values(list(columns), kind="stable").reset_index(drop=True)


def _assert_matching_frames(
    reference_df: "pd.DataFrame",
    candidate_df: "pd.DataFrame",
    columns: Sequence[str],
    description: str,
) -> None:
    """Raise a clear error if two notebook-8 outputs do not match exactly."""
    left_df = _sort_like_notebook8(reference_df.loc[:, list(columns)].copy(), columns)
    right_df = _sort_like_notebook8(candidate_df.loc[:, list(columns)].copy(), columns)
    try:
        pd.testing.assert_frame_equal(left_df, right_df, check_dtype=False, check_like=False)
    except AssertionError as error:
        raise ValueError(f"Matched pooling requirement failed for {description}.") from error


def load_matched_pooling_outputs(
    run_specs: Sequence[dict[str, str | Path]],
    expected_pooling_strategies: Sequence[str] = POOLING_ORDER,
) -> dict[str, object]:
    """Load notebook-8 outputs for matched CLS/mean/max comparisons."""
    expected_poolings = [normalize_pooling_strategy(value) for value in expected_pooling_strategies]
    if sorted(expected_poolings) != sorted(set(expected_poolings)):
        raise ValueError("expected_pooling_strategies must not contain duplicates.")

    artifacts_by_pooling: dict[str, dict[str, object]] = {}
    for spec in run_specs:
        pooling_strategy = normalize_pooling_strategy(spec.get("pooling_strategy"))  # type: ignore[arg-type]
        run_dir = Path(spec["run_dir"])
        config_path = run_dir / "similarity_scaleup_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing required notebook-8 config: {config_path}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_pooling = normalize_pooling_strategy(config.get("pooling_strategy"))
        if config_pooling != pooling_strategy:
            raise ValueError(
                f"Pooling mismatch for {run_dir}: spec requested {pooling_strategy!r} "
                f"but saved config says {config_pooling!r}."
            )

        artifacts_by_pooling[pooling_strategy] = {
            "spec": dict(spec),
            "run_dir": run_dir,
            "config": config,
            "selected_glycans": _read_csv_required(run_dir / "selected_glycans.csv"),
            "test_corpus": _read_csv_required(run_dir / "test_corpus_sequences.csv"),
            "specific_vs_all_ranked": _read_csv_required(run_dir / "specific_vs_all_ranked.csv"),
            "specific_vs_all_summary": _read_csv_required(run_dir / "specific_vs_all_distribution_summary.csv"),
            "all_vs_all_summary": _read_csv_required(run_dir / "all_vs_all_summary.csv"),
        }

    missing_poolings = [pooling for pooling in expected_poolings if pooling not in artifacts_by_pooling]
    extra_poolings = [pooling for pooling in artifacts_by_pooling if pooling not in expected_poolings]
    if missing_poolings or extra_poolings:
        raise ValueError(
            "Matched pooling comparisons require exactly these pooling strategies: "
            f"{expected_poolings}. Missing={missing_poolings}, extra={extra_poolings}."
        )

    reference_pooling = expected_poolings[0]
    reference_artifacts = artifacts_by_pooling[reference_pooling]
    reference_config = reference_artifacts["config"]
    reference_queries_df = reference_artifacts["selected_glycans"]
    reference_corpus_df = reference_artifacts["test_corpus"]
    reference_ranked_df = reference_artifacts["specific_vs_all_ranked"]

    shared_config_keys = [
        "model_dir",
        "max_length",
        "batch_size",
        "accession_col",
        "sequence_col",
        "selected_accessions",
    ]

    run_summary_rows = []
    for pooling_strategy in expected_poolings:
        artifacts = artifacts_by_pooling[pooling_strategy]
        config = artifacts["config"]

        for config_key in shared_config_keys:
            if config.get(config_key) != reference_config.get(config_key):
                raise ValueError(
                    "Matched pooling requirement failed because notebook-8 settings "
                    f"for {config_key!r} differ between {reference_pooling!r} and {pooling_strategy!r}."
                )

        _assert_matching_frames(
            reference_queries_df,
            artifacts["selected_glycans"],
            columns=["accession", "sequence"],
            description="selected_glycans.csv",
        )
        _assert_matching_frames(
            reference_corpus_df,
            artifacts["test_corpus"],
            columns=["accession", "sequence"],
            description="test_corpus_sequences.csv",
        )
        _assert_matching_frames(
            reference_ranked_df,
            artifacts["specific_vs_all_ranked"],
            columns=MATCH_KEYS,
            description="specific_vs_all_ranked.csv pair identities",
        )

        all_vs_all_summary_df = artifacts["all_vs_all_summary"]
        if len(all_vs_all_summary_df) != 1:
            raise ValueError(
                f"Expected one-row all_vs_all_summary.csv for {artifacts['run_dir']}, "
                f"found {len(all_vs_all_summary_df)} rows."
            )

        ranked_df = artifacts["specific_vs_all_ranked"]
        non_self_pair_count = int((~ranked_df["is_self_match"]).sum()) if "is_self_match" in ranked_df.columns else len(ranked_df)
        run_summary_rows.append(
            {
                "pooling_strategy": pooling_strategy,
                "run_dir": str(artifacts["run_dir"]),
                "model_dir": str(config.get("model_dir", "")),
                "output_name": str(config.get("output_name", "")),
                "query_count": int(len(artifacts["selected_glycans"])),
                "corpus_count": int(len(artifacts["test_corpus"])),
                "matched_pair_count": int(len(ranked_df)),
                "non_self_pair_count": non_self_pair_count,
                "all_vs_all_mean": float(all_vs_all_summary_df.iloc[0]["mean"]),
                "all_vs_all_median": float(all_vs_all_summary_df.iloc[0]["median"]),
            }
        )

    return {
        "pooling_order": expected_poolings,
        "artifacts_by_pooling": artifacts_by_pooling,
        "matched_run_summary": pd.DataFrame(run_summary_rows),
        "shared_model_dir": str(reference_config.get("model_dir", "")),
        "selected_glycans_df": reference_queries_df.copy(),
        "test_corpus_df": reference_corpus_df.copy(),
    }


def build_matched_pooling_score_table(
    matched_outputs: dict[str, object],
) -> "pd.DataFrame":
    """Merge notebook-8 score rows so each glycan pair has cls/mean/max scores side by side."""
    pooling_order = matched_outputs["pooling_order"]
    artifacts_by_pooling = matched_outputs["artifacts_by_pooling"]

    merged_df: pd.DataFrame | None = None
    for pooling_strategy in pooling_order:
        ranked_df = artifacts_by_pooling[pooling_strategy]["specific_vs_all_ranked"].copy()
        rename_map = {
            "cosine_similarity": f"{pooling_strategy}_similarity",
            "rank": f"{pooling_strategy}_rank",
        }
        keep_columns = MATCH_KEYS + list(rename_map) + [
            column_name
            for column_name in ["is_self_match"]
            if column_name in ranked_df.columns
        ]
        pooling_df = ranked_df.loc[:, keep_columns].rename(columns=rename_map)

        if merged_df is None:
            merged_df = pooling_df
            continue

        join_columns = MATCH_KEYS + (["is_self_match"] if "is_self_match" in pooling_df.columns else [])
        merged_df = merged_df.merge(pooling_df, on=join_columns, how="inner", validate="one_to_one")

    if merged_df is None:
        raise ValueError("No pooling outputs were available to merge.")

    query_lookup_df = matched_outputs["selected_glycans_df"].copy()
    if {"accession", "label"}.issubset(query_lookup_df.columns):
        merged_df = merged_df.merge(
            query_lookup_df.loc[:, ["accession", "label"]].rename(
                columns={"accession": "query_accession", "label": "query_label"}
            ),
            on="query_accession",
            how="left",
        )

    return _sort_like_notebook8(merged_df, MATCH_KEYS)


def _filter_score_rows(
    merged_df: "pd.DataFrame",
    include_self_matches: bool,
) -> "pd.DataFrame":
    """Return the rows that should contribute to the pooling metrics."""
    filtered_df = merged_df.copy()
    if not include_self_matches and "is_self_match" in filtered_df.columns:
        filtered_df = filtered_df.loc[~filtered_df["is_self_match"]].copy()
    return filtered_df.reset_index(drop=True)


def _summarize_similarity_values(values: "pd.Series") -> dict[str, float]:
    """Return notebook-friendly summary stats for one score column."""
    cleaned_values = pd.Series(values, dtype=float).dropna()
    if cleaned_values.empty:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std_dev": float("nan"),
            "min": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "max": float("nan"),
        }

    return {
        "count": int(cleaned_values.size),
        "mean": float(cleaned_values.mean()),
        "median": float(cleaned_values.median()),
        "std_dev": float(cleaned_values.std(ddof=0)),
        "min": float(cleaned_values.min()),
        "q05": float(cleaned_values.quantile(0.05)),
        "q25": float(cleaned_values.quantile(0.25)),
        "q75": float(cleaned_values.quantile(0.75)),
        "q95": float(cleaned_values.quantile(0.95)),
        "max": float(cleaned_values.max()),
    }


def build_pooling_similarity_summary(
    merged_df: "pd.DataFrame",
    include_self_matches: bool = False,
) -> "pd.DataFrame":
    """Summarize the matched pairwise scores for each pooling strategy."""
    filtered_df = _filter_score_rows(merged_df, include_self_matches=include_self_matches)
    rows = []
    for pooling_strategy in POOLING_ORDER:
        score_column = f"{pooling_strategy}_similarity"
        row = {
            "pooling_strategy": pooling_strategy,
            "include_self_matches": bool(include_self_matches),
        }
        row.update(_summarize_similarity_values(filtered_df[score_column]))
        rows.append(row)
    return pd.DataFrame(rows)


def build_pooling_similarity_summary_by_query(
    merged_df: "pd.DataFrame",
    include_self_matches: bool = False,
) -> "pd.DataFrame":
    """Summarize matched pairwise scores per query glycan and pooling strategy."""
    filtered_df = _filter_score_rows(merged_df, include_self_matches=include_self_matches)
    rows = []
    group_columns = ["query_accession"]
    if "query_label" in filtered_df.columns:
        group_columns.append("query_label")

    for group_values, query_df in filtered_df.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        base_record = dict(zip(group_columns, group_values, strict=False))
        for pooling_strategy in POOLING_ORDER:
            row = dict(base_record)
            row["pooling_strategy"] = pooling_strategy
            row["include_self_matches"] = bool(include_self_matches)
            row.update(_summarize_similarity_values(query_df[f"{pooling_strategy}_similarity"]))
            rows.append(row)

    return pd.DataFrame(rows)


def build_pooling_pairwise_correlations(
    merged_df: "pd.DataFrame",
    include_self_matches: bool = False,
) -> "pd.DataFrame":
    """Compute Pearson and Spearman correlations between pooling strategies."""
    filtered_df = _filter_score_rows(merged_df, include_self_matches=include_self_matches)
    rows = []
    for left_pooling, right_pooling in itertools.combinations(POOLING_ORDER, 2):
        left_column = f"{left_pooling}_similarity"
        right_column = f"{right_pooling}_similarity"
        pair_df = filtered_df.loc[:, [left_column, right_column]].dropna()
        rows.append(
            {
                "pooling_a": left_pooling,
                "pooling_b": right_pooling,
                "include_self_matches": bool(include_self_matches),
                "pair_count": int(len(pair_df)),
                "pearson_correlation": float(pair_df[left_column].corr(pair_df[right_column], method="pearson")),
                "spearman_correlation": float(pair_df[left_column].corr(pair_df[right_column], method="spearman")),
            }
        )
    return pd.DataFrame(rows)


def build_pooling_top_k_overlap(
    merged_df: "pd.DataFrame",
    top_k: int = 25,
) -> "pd.DataFrame":
    """Compare matched top-k neighbor sets for each query glycan."""
    filtered_df = _filter_score_rows(merged_df, include_self_matches=False)
    rows = []
    for query_accession, query_df in filtered_df.groupby("query_accession", sort=False):
        for left_pooling, right_pooling in itertools.combinations(POOLING_ORDER, 2):
            left_neighbors = set(
                query_df.loc[query_df[f"{left_pooling}_rank"] <= int(top_k), "corpus_accession"]
            )
            right_neighbors = set(
                query_df.loc[query_df[f"{right_pooling}_rank"] <= int(top_k), "corpus_accession"]
            )
            union_size = len(left_neighbors | right_neighbors)
            overlap_accessions = sorted(left_neighbors & right_neighbors)
            rows.append(
                {
                    "query_accession": query_accession,
                    "pooling_a": left_pooling,
                    "pooling_b": right_pooling,
                    "top_k": int(top_k),
                    "pooling_a_neighbor_count": len(left_neighbors),
                    "pooling_b_neighbor_count": len(right_neighbors),
                    "overlap_count": len(overlap_accessions),
                    "jaccard_overlap": len(overlap_accessions) / union_size if union_size else 0.0,
                    "overlap_accessions_json": json.dumps(overlap_accessions),
                }
            )
    return pd.DataFrame(rows)


def build_pooling_top_k_overlap_summary(
    top_k_overlap_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Average the per-query top-k overlap metrics for each pooling pair."""
    if top_k_overlap_df.empty:
        return pd.DataFrame()
    summary_df = (
        top_k_overlap_df.groupby(["pooling_a", "pooling_b", "top_k"], as_index=False)[
            ["overlap_count", "jaccard_overlap"]
        ]
        .mean()
        .sort_values(["pooling_a", "pooling_b"], kind="stable")
        .reset_index(drop=True)
    )
    return summary_df


def build_pooling_three_way_top_k_overlap_summary(
    merged_df: "pd.DataFrame",
    top_k: int = 25,
) -> "pd.DataFrame":
    """Build Venn-style three-way top-k overlap counts for each query glycan."""
    filtered_df = _filter_score_rows(merged_df, include_self_matches=False)
    rows = []
    for query_accession, query_df in filtered_df.groupby("query_accession", sort=False):
        cls_neighbors = set(query_df.loc[query_df["cls_rank"] <= int(top_k), "corpus_accession"])
        mean_neighbors = set(query_df.loc[query_df["mean_rank"] <= int(top_k), "corpus_accession"])
        max_neighbors = set(query_df.loc[query_df["max_rank"] <= int(top_k), "corpus_accession"])

        all_three = cls_neighbors & mean_neighbors & max_neighbors
        cls_mean_only = (cls_neighbors & mean_neighbors) - max_neighbors
        cls_max_only = (cls_neighbors & max_neighbors) - mean_neighbors
        mean_max_only = (mean_neighbors & max_neighbors) - cls_neighbors
        cls_only = cls_neighbors - mean_neighbors - max_neighbors
        mean_only = mean_neighbors - cls_neighbors - max_neighbors
        max_only = max_neighbors - cls_neighbors - mean_neighbors

        rows.append(
            {
                "query_accession": query_accession,
                "top_k": int(top_k),
                "cls_only_count": len(cls_only),
                "mean_only_count": len(mean_only),
                "max_only_count": len(max_only),
                "cls_mean_only_count": len(cls_mean_only),
                "cls_max_only_count": len(cls_max_only),
                "mean_max_only_count": len(mean_max_only),
                "all_three_count": len(all_three),
                "cls_only_accessions_json": json.dumps(sorted(cls_only)),
                "mean_only_accessions_json": json.dumps(sorted(mean_only)),
                "max_only_accessions_json": json.dumps(sorted(max_only)),
                "cls_mean_only_accessions_json": json.dumps(sorted(cls_mean_only)),
                "cls_max_only_accessions_json": json.dumps(sorted(cls_max_only)),
                "mean_max_only_accessions_json": json.dumps(sorted(mean_max_only)),
                "all_three_accessions_json": json.dumps(sorted(all_three)),
            }
        )

    return pd.DataFrame(rows)


def build_pooling_query_inspection_table(
    merged_df: "pd.DataFrame",
    query_accessions: Sequence[str] | None = None,
    top_n: int = 15,
) -> "pd.DataFrame":
    """Build a compact per-query neighbor-inspection table for notebook follow-up."""
    filtered_df = _filter_score_rows(merged_df, include_self_matches=False).copy()
    filtered_df["mean_similarity_across_poolings"] = filtered_df[
        ["cls_similarity", "mean_similarity", "max_similarity"]
    ].mean(axis=1)
    filtered_df["rank_spread"] = filtered_df[
        ["cls_rank", "mean_rank", "max_rank"]
    ].max(axis=1) - filtered_df[["cls_rank", "mean_rank", "max_rank"]].min(axis=1)

    if query_accessions:
        filtered_df = filtered_df.loc[
            filtered_df["query_accession"].isin([str(value) for value in query_accessions])
        ].copy()

    rows = []
    for query_accession, query_df in filtered_df.groupby("query_accession", sort=False):
        review_df = query_df.sort_values(
            ["mean_similarity_across_poolings", "rank_spread", "corpus_accession"],
            ascending=[False, True, True],
            kind="stable",
        ).head(int(top_n))
        rows.append(review_df)

    if not rows:
        return pd.DataFrame(columns=list(filtered_df.columns))
    return pd.concat(rows, ignore_index=True)


def plot_pooling_metric_matrix(
    merged_df: "pd.DataFrame",
    output_path: str | Path,
    include_self_matches: bool = False,
    sample_size: int | None = 20_000,
    random_state: int = 7,
    title: str = "Matched pooling comparison",
) -> str:
    """Plot the requested 3x3 histogram/scatter matrix for CLS/mean/max."""
    filtered_df = _filter_score_rows(merged_df, include_self_matches=include_self_matches)
    plot_df = filtered_df.copy()
    if sample_size is not None and len(plot_df) > int(sample_size):
        plot_df = plot_df.sample(n=int(sample_size), random_state=int(random_state))

    score_columns = {pooling: f"{pooling}_similarity" for pooling in POOLING_ORDER}
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    fig.suptitle(title, fontsize=16)

    for row_index, row_pooling in enumerate(POOLING_ORDER):
        for column_index, column_pooling in enumerate(POOLING_ORDER):
            ax = axes[row_index, column_index]
            row_column = score_columns[row_pooling]
            column_column = score_columns[column_pooling]

            if row_pooling == column_pooling:
                ax.hist(plot_df[row_column].dropna(), bins=40, color="#c97734", alpha=0.88, edgecolor="white")
                ax.set_title(f"{row_pooling.upper()} distribution")
                ax.set_xlabel("Cosine similarity")
                ax.set_ylabel("Count")
                ax.grid(alpha=0.2)
                continue

            ax.scatter(
                plot_df[column_column],
                plot_df[row_column],
                s=7,
                alpha=0.18,
                color="#2f627b",
                linewidths=0,
            )
            diagonal_min = min(
                float(plot_df[column_column].min()),
                float(plot_df[row_column].min()),
            )
            diagonal_max = max(
                float(plot_df[column_column].max()),
                float(plot_df[row_column].max()),
            )
            ax.plot(
                [diagonal_min, diagonal_max],
                [diagonal_min, diagonal_max],
                linestyle="--",
                linewidth=1.2,
                color="#8b4a25",
            )
            ax.set_xlabel(f"{column_pooling.upper()} similarity")
            ax.set_ylabel(f"{row_pooling.upper()} similarity")
            ax.set_title(f"{row_pooling.upper()} vs {column_pooling.upper()}")
            ax.grid(alpha=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return str(output_path)


def save_pooling_comparison_tables(
    comparison_tables: dict[str, "pd.DataFrame"],
    output_dir: str | Path,
) -> dict[str, str]:
    """Save pooling-comparison tables and return a name-to-path manifest."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, str] = {}
    for table_name, table_df in comparison_tables.items():
        file_path = output_path / f"{table_name}.csv"
        table_df.to_csv(file_path, index=False)
        saved_paths[table_name] = str(file_path)
    return saved_paths


def _file_to_data_uri(path: str | Path) -> str | None:
    """Convert an image file to a data URI so the HTML can stay portable."""
    file_path = Path(path)
    if not file_path.exists():
        return None

    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    encoded_bytes = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_bytes}"


def _render_dataframe_html(frame_df: "pd.DataFrame") -> str:
    """Render one dataframe with compact number formatting."""
    if frame_df.empty:
        return "<p class='subtle'>No rows were generated for this section.</p>"
    return frame_df.to_html(index=False, classes="report-table", border=0, escape=False)


def _safe_stem(value: str) -> str:
    """Create a stable filename-friendly stem for copied report assets."""
    safe_chars = []
    for character in str(value):
        if character.isalnum() or character in {"-", "_"}:
            safe_chars.append(character)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_") or "item"


def _relative_path(path: str | Path, start_dir: str | Path) -> str:
    """Return a browser-friendly relative path inside the HTML report folder."""
    return Path(path).resolve().relative_to(Path(start_dir).resolve()).as_posix()


def _resolve_cartoon_source_path(local_image_path: str | Path, run_dir: str | Path) -> Path | None:
    """Find a cached cartoon image even when a manifest carries a Colab path."""
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
    """Copy one cached cartoon into the HTML assets folder and return its path."""
    if source_path is None:
        return None

    asset_dir.mkdir(parents=True, exist_ok=True)
    output_path = asset_dir / f"{_safe_stem(image_key)}{source_path.suffix.lower()}"
    if not output_path.exists():
        shutil.copy2(source_path, output_path)
    return str(output_path)


def build_cartoon_lookup_from_run_manifests(
    run_specs: Sequence[dict[str, str | Path]],
) -> dict[str, dict[str, dict[str, str]]]:
    """Build a sequence/accession lookup for notebook-8 cartoons already cached on disk."""
    lookup: dict[str, dict[str, dict[str, str]]] = {
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


def _lookup_cartoon(
    cartoon_lookup: dict[str, dict[str, dict[str, str]]],
    accession: str,
    sequence: str,
) -> dict[str, str] | None:
    """Find a cached cartoon by accession first, then by sequence."""
    return (
        cartoon_lookup["by_accession"].get(str(accession))
        or cartoon_lookup["by_sequence"].get(str(sequence))
    )


def _render_cartoon_image(
    image_path: str,
    output_dir: Path,
    image_key: str,
    mini: bool = False,
    embed_images: bool = True,
) -> str:
    """Render a cached cartoon image if available."""
    css_class = "mini-cartoon" if mini else "cartoon-box"
    if not image_path:
        return f'<div class="{css_class}"><div class="missing-cartoon">cartoon not cached</div></div>'

    source_path = Path(image_path) if Path(image_path).exists() else None
    if source_path is None:
        return f'<div class="{css_class}"><div class="missing-cartoon">cartoon file missing</div></div>'

    if embed_images:
        image_source = _file_to_data_uri(source_path)
    else:
        copied_path = _copy_cartoon_for_report(
            source_path,
            output_dir / "html_assets" / "cartoons",
            image_key=image_key,
        )
        image_source = _relative_path(copied_path, output_dir) if copied_path else None

    if image_source is None:
        return f'<div class="{css_class}"><div class="missing-cartoon">cartoon file missing</div></div>'

    return (
        f'<div class="{css_class}">'
        f'<img src="{escape(image_source)}" alt="glycan cartoon">'
        "</div>"
    )


def _make_report_path_relative(path_value: str | Path | object) -> str:
    """Convert local absolute paths into project-relative report labels."""
    text = str(path_value or "").strip()
    if not text:
        return ""

    normalized_text = text.replace("\\", "/")
    for prefix in REPORT_PATH_PREFIXES:
        if normalized_text.startswith(prefix):
            relative_text = normalized_text[len(prefix):].lstrip("/")
            relative_parts = relative_text.split("/", 1)
            return relative_parts[1] if len(relative_parts) == 2 else relative_parts[0]

    if normalized_text.startswith("/Users/"):
        relative_parts = normalized_text.split("/")[4:]
        if relative_parts:
            return "/".join(relative_parts)

    return normalized_text


def _prepare_matched_run_summary_for_report(frame_df: "pd.DataFrame") -> "pd.DataFrame":
    """Keep only the report-level run summary columns for the HTML view."""
    report_df = frame_df.copy()
    keep_columns = [
        "pooling_strategy",
        "query_count",
        "corpus_count",
        "matched_pair_count",
        "non_self_pair_count",
        "all_vs_all_mean",
        "all_vs_all_median",
    ]
    available_columns = [column_name for column_name in keep_columns if column_name in report_df.columns]
    return report_df.loc[:, available_columns].copy()


def _prepare_three_way_overlap_for_report(frame_df: "pd.DataFrame") -> "pd.DataFrame":
    """Keep only the high-level overlap counts in the HTML report."""
    keep_columns = [
        "query_accession",
        "top_k",
        "cls_only_count",
        "mean_only_count",
        "max_only_count",
        "cls_mean_only_count",
        "cls_max_only_count",
        "mean_max_only_count",
        "all_three_count",
    ]
    available_columns = [column_name for column_name in keep_columns if column_name in frame_df.columns]
    return frame_df.loc[:, available_columns].copy()


def _prepare_query_inspection_for_report(frame_df: "pd.DataFrame") -> "pd.DataFrame":
    """Remove bulky sequence columns from the HTML view while keeping the scored comparison columns."""
    report_df = frame_df.copy()
    if "is_self_match" in report_df.columns and report_df["is_self_match"].nunique(dropna=False) == 1:
        if bool(report_df["is_self_match"].iloc[0]) is False:
            report_df = report_df.drop(columns=["is_self_match"])

    keep_columns = [
        "query_accession",
        "corpus_accession",
        "cls_similarity",
        "cls_rank",
        "mean_similarity",
        "mean_rank",
        "max_similarity",
        "max_rank",
        "mean_similarity_across_poolings",
        "rank_spread",
    ]
    available_columns = [column_name for column_name in keep_columns if column_name in report_df.columns]
    return report_df.loc[:, available_columns].copy()


def _build_query_gallery_table(
    frame_df: "pd.DataFrame",
    cartoon_lookup: dict[str, dict[str, dict[str, str]]],
) -> "pd.DataFrame":
    """Attach cached cartoon paths to the selected notebook-13 inspection rows."""
    rows = []
    for row in frame_df.itertuples(index=False):
        query_accession = str(getattr(row, "query_accession"))
        query_sequence = str(getattr(row, "query_sequence"))
        corpus_accession = str(getattr(row, "corpus_accession"))
        corpus_sequence = str(getattr(row, "corpus_sequence"))

        query_cartoon = _lookup_cartoon(cartoon_lookup, query_accession, query_sequence)
        corpus_cartoon = _lookup_cartoon(cartoon_lookup, corpus_accession, corpus_sequence)
        rows.append(
            {
                "query_accession": query_accession,
                "query_sequence": query_sequence,
                "query_cartoon_path": query_cartoon["image_path"] if query_cartoon else "",
                "corpus_accession": corpus_accession,
                "display_accession": (
                    corpus_cartoon["accession"]
                    if corpus_cartoon and corpus_cartoon.get("accession")
                    else corpus_accession
                ),
                "corpus_cartoon_path": corpus_cartoon["image_path"] if corpus_cartoon else "",
                "cls_similarity": float(getattr(row, "cls_similarity")),
                "cls_rank": int(getattr(row, "cls_rank")),
                "mean_similarity": float(getattr(row, "mean_similarity")),
                "mean_rank": int(getattr(row, "mean_rank")),
                "max_similarity": float(getattr(row, "max_similarity")),
                "max_rank": int(getattr(row, "max_rank")),
                "mean_similarity_across_poolings": float(getattr(row, "mean_similarity_across_poolings")),
                "rank_spread": int(getattr(row, "rank_spread")),
            }
        )
    return pd.DataFrame(rows)


def _render_query_gallery_sections(
    gallery_df: "pd.DataFrame",
    output_dir: Path,
    embed_images: bool = True,
) -> str:
    """Render one cartoon gallery section per query glycan."""
    if gallery_df.empty:
        return "<p class='subtle'>No query gallery rows were generated.</p>"

    sections = []
    for query_accession, query_df in gallery_df.groupby("query_accession", sort=False):
        first_row = query_df.iloc[0]
        query_cartoon_html = _render_cartoon_image(
            str(first_row["query_cartoon_path"]),
            output_dir=output_dir,
            image_key=f"{query_accession}_query",
            mini=False,
            embed_images=embed_images,
        )

        body_rows = []
        for row in query_df.itertuples(index=False):
            neighbor_cartoon_html = _render_cartoon_image(
                str(row.corpus_cartoon_path),
                output_dir=output_dir,
                image_key=f"{query_accession}_{row.corpus_accession}",
                mini=True,
                embed_images=embed_images,
            )
            body_rows.append(
                "<tr>"
                f'<td class="neighbor-cartoon-cell">{neighbor_cartoon_html}</td>'
                f"<td>{escape(str(row.display_accession))}</td>"
                f'<td class="numeric">{int(row.cls_rank)}</td>'
                f'<td class="numeric">{float(row.cls_similarity):.3f}</td>'
                f'<td class="numeric">{int(row.mean_rank)}</td>'
                f'<td class="numeric">{float(row.mean_similarity):.3f}</td>'
                f'<td class="numeric">{int(row.max_rank)}</td>'
                f'<td class="numeric">{float(row.max_similarity):.3f}</td>'
                f'<td class="numeric">{float(row.mean_similarity_across_poolings):.3f}</td>'
                f'<td class="numeric">{int(row.rank_spread)}</td>'
                "</tr>"
            )

        sections.append(
            '<section class="query-gallery">'
            '<div class="query-gallery-header">'
            f"{query_cartoon_html}"
            '<div class="query-gallery-copy">'
            f"<h3>Query {escape(str(query_accession))}</h3>"
            "<p>These rows show the same selected neighbors across the three pooling rules. The cartoons make it easier to see when rank changes coincide with visible structural differences.</p>"
            "</div>"
            "</div>"
            '<div class="query-neighbor-table">'
            '<div class="table-scroll">'
            '<table class="report-table">'
            "<thead><tr>"
            "<th>Cartoon</th><th>Neighbor</th><th>CLS rank</th><th>CLS sim</th>"
            "<th>Mean rank</th><th>Mean sim</th><th>Max rank</th><th>Max sim</th>"
            "<th>Mean sim across poolings</th><th>Rank spread</th>"
            "</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            "</table>"
            "</div>"
            "</div>"
            "</section>"
        )

    return "".join(sections)


def render_pooling_metric_comparison_html_report(
    output_dir: str | Path,
    report_title: str,
    comparison_tables: dict[str, "pd.DataFrame"],
    plot_paths: dict[str, str],
    shared_model_dir: str,
    inspect_query_accessions: Sequence[str] | None,
    top_k_neighbors: int,
    run_specs: Sequence[dict[str, str | Path]] | None = None,
    embed_images: bool = True,
) -> str:
    """Render one presentation-friendly HTML report for notebook 13."""
    output_path = Path(output_dir)
    html_path = output_path / "pooling_metric_comparison_report.html"
    plot_source = plot_paths["pooling_matrix_plot"]
    plot_reference = _file_to_data_uri(plot_source) if embed_images else Path(plot_source).name

    query_note = ", ".join(str(value) for value in (inspect_query_accessions or [])) or "first configured queries"
    summary_df = comparison_tables["pooling_similarity_summary"]
    overlap_summary_df = comparison_tables["pooling_top_k_overlap_summary"]
    matched_run_summary_df = _prepare_matched_run_summary_for_report(
        comparison_tables["matched_pooling_run_summary"]
    )
    three_way_overlap_df = _prepare_three_way_overlap_for_report(
        comparison_tables["pooling_top_k_three_way_overlap"]
    )
    inspection_df = _prepare_query_inspection_for_report(
        comparison_tables["pooling_query_inspection"]
    )
    gallery_html = "<p class='subtle'>No cached cartoons were available for the selected notebook-8 runs.</p>"
    if run_specs is not None:
        cartoon_lookup = build_cartoon_lookup_from_run_manifests(run_specs)
        gallery_df = _build_query_gallery_table(
            comparison_tables["pooling_query_inspection"],
            cartoon_lookup=cartoon_lookup,
        )
        gallery_html = _render_query_gallery_sections(
            gallery_df=gallery_df,
            output_dir=output_path,
            embed_images=embed_images,
        )
    html = f"""<!doctype html>
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
    <p class="subtle">Matched notebook-8 pooling comparison for one exact checkpoint.</p>
    <div class="badge-row">
      <span class="badge">Checkpoint held fixed</span>
      <span class="badge">Poolings compared: CLS, mean, max</span>
      <span class="badge">Top-k overlap: {int(top_k_neighbors)}</span>
    </div>
  </header>
  <main class="container">
    <section class="card">
      <h2>What This Report Checks</h2>
      <p>This report compares notebook-8 outputs only after holding the model checkpoint, query set, and corpus fixed. The only intended change is the pooling rule.</p>
      <div class="callout">
        <h3>How To Read The Main Metrics</h3>
        <div class="term-grid">
          <div class="term-card">
            <strong>Similarity</strong>
            <span>Cosine similarity for one query-corpus pair under one pooling rule. Larger values indicate stronger embedding agreement within that pooling setup.</span>
          </div>
          <div class="term-card">
            <strong>Rank</strong>
            <span>The position of one corpus glycan after sorting neighbors for a single query. Rank 1 is the closest non-self match.</span>
          </div>
          <div class="term-card">
            <strong>Overlap count</strong>
            <span>The number of corpus glycans that appear in both top-k neighbor lists being compared.</span>
          </div>
          <div class="term-card">
            <strong>Jaccard overlap</strong>
            <span>The shared-neighbor count divided by the size of the combined neighbor set. A value near 1 means the two lists are very similar.</span>
          </div>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Matched Run Summary</h2>
      <p class="subtle">Each row should represent the same experiment under a different pooling rule. This table is the alignment check before interpreting any downstream difference.</p>
      <div class="callout">
        <h3>Column Guide</h3>
        <ul>
          <li><strong>pooling_strategy</strong>: which embedding pooling rule produced the row.</li>
          <li><strong>query_count</strong> and <strong>corpus_count</strong>: the number of selected query glycans and corpus glycans included in the comparison.</li>
          <li><strong>matched_pair_count</strong>: the total number of query-corpus pairs after aligning the three notebook-8 outputs.</li>
          <li><strong>non_self_pair_count</strong>: the number of aligned pairs after excluding self matches when applicable.</li>
          <li><strong>all_vs_all_mean</strong> and <strong>all_vs_all_median</strong>: whole-corpus similarity summaries from the original notebook-8 outputs for that pooling rule.</li>
        </ul>
      </div>
      <div class="table-scroll">{_render_dataframe_html(matched_run_summary_df)}</div>
    </section>

    <section class="card">
      <h2>Global Score Summary</h2>
      <p class="subtle">These rows summarize matched query-corpus pairs after excluding self matches.</p>
      <div class="callout">
        <h3>How To Interpret These Columns</h3>
        <ul>
          <li><strong>count</strong>: the number of aligned query-corpus pairs contributing to the row.</li>
          <li><strong>mean</strong> and <strong>median</strong>: the center of the similarity distribution for that pooling rule.</li>
          <li><strong>std_dev</strong>: how spread out the similarities are.</li>
          <li><strong>q05</strong>, <strong>q25</strong>, <strong>q75</strong>, and <strong>q95</strong>: percentile cutoffs that show the lower tail, middle spread, and upper tail.</li>
          <li><strong>Important</strong>: these values are best compared within this report, not treated as universal thresholds across different pooling methods or checkpoints.</li>
        </ul>
      </div>
      <div class="table-scroll">{_render_dataframe_html(summary_df)}</div>
    </section>

    <section class="card">
      <h2>Pairwise Correlations</h2>
      <p class="subtle">This table asks whether two pooling rules score the same pairs similarly and whether they preserve the same ordering of pairs.</p>
      <div class="callout">
        <h3>Column Guide</h3>
        <ul>
          <li><strong>pearson_correlation</strong>: agreement in the raw score values. High Pearson means the two poolings move up and down together, even if one uses a different score scale.</li>
          <li><strong>spearman_correlation</strong>: agreement in rank ordering. High Spearman means the two poolings tend to sort pairs similarly.</li>
          <li><strong>pair_count</strong>: the number of aligned pairs used in the correlation calculation.</li>
        </ul>
      </div>
      <div class="table-scroll">{_render_dataframe_html(comparison_tables["pooling_pairwise_correlations"])}</div>
    </section>

    <section class="card plot-wrap">
      <h2>3x3 Pooling Matrix</h2>
      <p class="subtle">Diagonal panels show the score distribution for one pooling rule. Off-diagonal panels compare the same query-corpus pairs under two pooling rules. A tight diagonal pattern means the two poolings behave similarly; a wider cloud means they reorder pairs more substantially.</p>
      <img src="{escape(str(plot_reference or ''))}" alt="Pooling comparison matrix">
    </section>

    <section class="card">
      <h2>Top-k Neighbor Overlap</h2>
      <p class="subtle">Higher overlap means the pooling rules preserve more of the same local neighborhoods for each query glycan.</p>
      <div class="callout">
        <h3>Column Guide</h3>
        <ul>
          <li><strong>top_k</strong>: the neighborhood size used for the comparison.</li>
          <li><strong>overlap_count</strong>: the average number of shared neighbors across the per-query top-k lists for that pooling pair.</li>
          <li><strong>jaccard_overlap</strong>: the average shared-neighbor fraction after dividing by the size of the combined neighbor set.</li>
          <li>As a rough rule, high correlation with low overlap means the scores move together globally but still swap enough neighbors to matter locally.</li>
        </ul>
      </div>
      <div class="table-scroll">{_render_dataframe_html(overlap_summary_df)}</div>
    </section>

    <section class="card">
      <h2>Three-Way Query Overlap</h2>
      <p class="subtle">Each row breaks one query glycan's top-k neighborhood into Venn-diagram-style regions across `cls`, `mean`, and `max` pooling.</p>
      <div class="callout">
        <h3>Column Guide</h3>
        <ul>
          <li><strong>cls_only_count</strong>, <strong>mean_only_count</strong>, and <strong>max_only_count</strong>: neighbors unique to one pooling rule.</li>
          <li><strong>cls_mean_only_count</strong>, <strong>cls_max_only_count</strong>, and <strong>mean_max_only_count</strong>: neighbors shared by exactly two poolings.</li>
          <li><strong>all_three_count</strong>: neighbors shared by all three pooling rules.</li>
          <li>The HTML report keeps only the count columns. The matching accession lists are still preserved in the exported CSV files.</li>
        </ul>
      </div>
      <div class="table-scroll">{_render_dataframe_html(three_way_overlap_df)}</div>
    </section>

    <section class="card">
      <h2>Query Cartoon Galleries</h2>
      <p class="subtle">These panels add the visual comparison view that the score tables cannot provide on their own. Each query section shows the selected neighbors with cartoons and the three pooling-specific ranks side by side.</p>
      <div class="gallery-grid">{gallery_html}</div>
    </section>

    <section class="card">
      <h2>Inspection Scores</h2>
      <p class="subtle">This compact table keeps the numerical comparison view for the configured query glycans. The default notebook focus is: {escape(query_note)}.</p>
      <div class="callout">
        <h3>Column Guide</h3>
        <ul>
          <li><strong>cls_similarity</strong>, <strong>mean_similarity</strong>, and <strong>max_similarity</strong>: the score assigned to the same query-corpus pair under each pooling rule.</li>
          <li><strong>cls_rank</strong>, <strong>mean_rank</strong>, and <strong>max_rank</strong>: where that corpus glycan lands in each pooling-specific neighbor list.</li>
          <li><strong>mean_similarity_across_poolings</strong>: the simple average of the three similarity values, used here to surface strong candidates that stay good overall.</li>
          <li><strong>rank_spread</strong>: the difference between the best and worst rank across pooling rules. Large values flag pooling-sensitive neighbors.</li>
          <li>The HTML version hides the long sequence strings to keep the table readable. The full sequences remain available in the saved CSV exports.</li>
        </ul>
      </div>
      <div class="table-scroll">{_render_dataframe_html(inspection_df)}</div>
    </section>
  </main>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return str(html_path)


def build_pooling_metric_comparison(
    run_specs: Sequence[dict[str, str | Path]],
    output_dir: str | Path,
    report_title: str = "Pooling Metric Comparison",
    top_k_neighbors: int = 25,
    inspect_query_accessions: Sequence[str] | None = None,
    inspect_top_n: int = 15,
    scatter_sample_size: int | None = 20_000,
    embed_html_images: bool = True,
) -> dict[str, object]:
    """Run the complete notebook-13 matched pooling comparison workflow."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    matched_outputs = load_matched_pooling_outputs(run_specs)
    merged_scores_df = build_matched_pooling_score_table(matched_outputs)

    comparison_tables = {
        "matched_pooling_run_summary": matched_outputs["matched_run_summary"],
        "pooling_score_pairs_merged": merged_scores_df,
        "pooling_similarity_summary": build_pooling_similarity_summary(merged_scores_df, include_self_matches=False),
        "pooling_similarity_summary_by_query": build_pooling_similarity_summary_by_query(
            merged_scores_df,
            include_self_matches=False,
        ),
        "pooling_pairwise_correlations": build_pooling_pairwise_correlations(
            merged_scores_df,
            include_self_matches=False,
        ),
    }
    comparison_tables["pooling_top_k_overlap_by_query"] = build_pooling_top_k_overlap(
        merged_scores_df,
        top_k=top_k_neighbors,
    )
    comparison_tables["pooling_top_k_overlap_summary"] = build_pooling_top_k_overlap_summary(
        comparison_tables["pooling_top_k_overlap_by_query"]
    )
    comparison_tables["pooling_top_k_three_way_overlap"] = build_pooling_three_way_top_k_overlap_summary(
        merged_scores_df,
        top_k=top_k_neighbors,
    )
    comparison_tables["pooling_query_inspection"] = build_pooling_query_inspection_table(
        merged_scores_df,
        query_accessions=inspect_query_accessions,
        top_n=inspect_top_n,
    )

    saved_table_paths = save_pooling_comparison_tables(comparison_tables, output_path)
    plot_paths = {
        "pooling_matrix_plot": plot_pooling_metric_matrix(
            merged_scores_df,
            output_path / "pooling_metric_matrix.png",
            include_self_matches=False,
            sample_size=scatter_sample_size,
            title=report_title,
        )
    }
    html_report_path = render_pooling_metric_comparison_html_report(
        output_dir=output_path,
        report_title=report_title,
        comparison_tables=comparison_tables,
        plot_paths=plot_paths,
        shared_model_dir=str(matched_outputs["shared_model_dir"]),
        inspect_query_accessions=inspect_query_accessions,
        top_k_neighbors=top_k_neighbors,
        run_specs=run_specs,
        embed_images=embed_html_images,
    )
    plot_paths["html_report_path"] = html_report_path

    manifest = {
        "output_dir": str(output_path),
        "shared_model_dir": str(matched_outputs["shared_model_dir"]),
        "table_paths": saved_table_paths,
        "plot_paths": plot_paths,
        "report_title": report_title,
        "top_k_neighbors": int(top_k_neighbors),
        "inspect_query_accessions": [str(value) for value in (inspect_query_accessions or [])],
        "inspect_top_n": int(inspect_top_n),
        "scatter_sample_size": None if scatter_sample_size is None else int(scatter_sample_size),
        "embed_html_images": bool(embed_html_images),
        "run_specs": [
            _stringify_pathlike_values(dict(spec))
            for spec in run_specs
        ],
    }
    manifest_path = output_path / "pooling_metric_comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "tables": comparison_tables,
        "table_paths": saved_table_paths,
        "plot_paths": plot_paths,
        "manifest_path": str(manifest_path),
        "matched_outputs": matched_outputs,
    }


def export_public_pooling_metric_comparison_html(
    comparison_outputs: dict | None = None,
    plot_paths: dict | None = None,
    export_dir=None,
    repo_public_subdir: str | None = None,
    repo_owner: str | None = None,
    repo_name: str | None = None,
    repo_ref: str = "main",
    extra_blocked_strings: Sequence[str] | None = None,
) -> dict[str, object]:
    """Copy one notebook-13 HTML report and its local dependencies into a clean folder."""
    source_plot_paths = plot_paths if plot_paths is not None else (comparison_outputs or {}).get("plot_paths")
    if source_plot_paths is None:
        raise ValueError("Provide either comparison_outputs or plot_paths when exporting public HTML.")
    if export_dir is None:
        raise ValueError("export_dir is required for export_public_pooling_metric_comparison_html().")

    source_html_path = Path(source_plot_paths["html_report_path"]).resolve()
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
                        "issue_type": "unsupported_or_external_reference",
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
        repo_index_path = f"{repo_public_subdir.rstrip('/')}/pooling_metric_comparison_report.html"
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
