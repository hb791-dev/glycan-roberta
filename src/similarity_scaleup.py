"""Test-set scale-up similarity workflow helpers.

This module contains the broader held-out-corpus analysis path: dataframe
normalization, query-vs-corpus comparison, threshold clouds, scale-up HTML
reports, and the end-to-end save/run orchestration for notebook 8.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.glycan_cartoons import (
    build_cartoon_manifest,
    cache_cartoon_images,
    cartoon_lookup_from_manifest,
    format_glycan_sequence_block,
)
from src.similarity_core import _image_path_to_data_uri, embed_sequences

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Dataframe cleanup and validation
# ---------------------------------------------------------------------------

def _clean_similarity_dataframe(sequence_df, accession_col: str, sequence_col: str) -> "pd.DataFrame":
    """Return a copy with predictable string columns and reset row order.

    The scale-up notebook needs to move between pandas, embedding tensors, CSVs,
    and HTML reports without losing row alignment. Normalizing the key columns
    early keeps the rest of the helpers simpler and easier to reason about.
    """
    import pandas as pd

    cleaned_df = pd.DataFrame(sequence_df).copy()
    cleaned_df[accession_col] = cleaned_df[accession_col].fillna("").map(str).map(str.strip)
    cleaned_df[sequence_col] = cleaned_df[sequence_col].fillna("").map(str).map(str.strip)
    return cleaned_df.reset_index(drop=True)


def validate_scaleup_similarity_inputs(
    model_dir,
    corpus_df,
    query_df,
    accession_col: str = "accession",
    sequence_col: str = "sequence",
    output_dir=None,
) -> None:
    """Validate the inputs needed for the test-set scale-up similarity workflow."""
    import pandas as pd

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    required_model_files = ["config.json"]
    missing_files = [filename for filename in required_model_files if not (model_path / filename).exists()]
    if missing_files:
        raise FileNotFoundError(f"Model directory is missing required files: {missing_files}")

    if corpus_df is None or len(pd.DataFrame(corpus_df)) == 0:
        raise ValueError("Add at least one corpus glycan before running the scale-up analysis.")

    if query_df is None or len(pd.DataFrame(query_df)) == 0:
        raise ValueError("Add at least one selected glycan before running the scale-up analysis.")

    for frame_name, frame in (("corpus_df", corpus_df), ("query_df", query_df)):
        frame_columns = set(pd.DataFrame(frame).columns)
        missing_columns = [column for column in (accession_col, sequence_col) if column not in frame_columns]
        if missing_columns:
            raise ValueError(f"{frame_name} is missing required columns: {missing_columns}")

        cleaned_frame = _clean_similarity_dataframe(frame, accession_col=accession_col, sequence_col=sequence_col)
        if cleaned_frame[sequence_col].eq("").any():
            blank_rows = cleaned_frame.index[cleaned_frame[sequence_col].eq("")].tolist()
            raise ValueError(f"{frame_name} contains blank sequences at rows: {blank_rows[:10]}")

    cleaned_corpus_df = _clean_similarity_dataframe(corpus_df, accession_col=accession_col, sequence_col=sequence_col)
    cleaned_query_df = _clean_similarity_dataframe(query_df, accession_col=accession_col, sequence_col=sequence_col)

    if output_dir is not None:
        # Make the output directory early so path problems are caught before the
        # more expensive embedding work starts.
        Path(output_dir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Embedding lookup helpers
# ---------------------------------------------------------------------------

def build_embedding_lookup_for_dataframe(
    sequence_df,
    tokenizer,
    model,
    accession_col: str = "accession",
    sequence_col: str = "sequence",
    device: str | torch.device | None = None,
    max_length: int | None = None,
    batch_size: int = 32,
) -> dict:
    """Embed one dataframe of glycans while reusing duplicate sequence embeddings.

    The corpus dataframe may carry accessions or other metadata that differ even
    when the underlying glycan sequence is identical. This helper embeds each
    unique sequence once, then maps the resulting vectors back onto the dataframe
    row order used by the notebook and saved reports.
    """
    cleaned_df = _clean_similarity_dataframe(sequence_df, accession_col=accession_col, sequence_col=sequence_col)
    unique_sequences: list[str] = []
    sequence_to_index: dict[str, int] = {}

    for sequence in cleaned_df[sequence_col].tolist():
        if sequence not in sequence_to_index:
            sequence_to_index[sequence] = len(unique_sequences)
            unique_sequences.append(sequence)

    unique_embeddings = embed_sequences(
        unique_sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    normalized_unique_embeddings = torch.nn.functional.normalize(unique_embeddings, p=2, dim=1)

    row_embedding_indices = [sequence_to_index[sequence] for sequence in cleaned_df[sequence_col].tolist()]
    row_index_tensor = torch.tensor(row_embedding_indices, dtype=torch.long)
    row_embeddings = unique_embeddings[row_index_tensor]
    normalized_row_embeddings = normalized_unique_embeddings[row_index_tensor]

    # Keep the mapping indices on the dataframe so downstream notebook code can
    # debug row-to-embedding alignment without re-deriving it by hand.
    cleaned_df = cleaned_df.copy()
    cleaned_df["_embedding_index"] = row_embedding_indices

    return {
        "sequence_df": cleaned_df,
        "unique_sequences": unique_sequences,
        "sequence_to_index": sequence_to_index,
        "unique_embeddings": unique_embeddings,
        "normalized_unique_embeddings": normalized_unique_embeddings,
        "row_embeddings": row_embeddings,
        "normalized_embeddings": normalized_row_embeddings,
    }


# ---------------------------------------------------------------------------
# Distribution summaries and all-vs-all analysis
# ---------------------------------------------------------------------------

def _summarize_similarity_values(similarity_values, extra_fields: dict | None = None) -> dict:
    """Summarize one collection of cosine similarities with notebook-friendly stats."""
    import numpy as np

    values = np.asarray(list(similarity_values), dtype=float)
    summary = dict(extra_fields or {})
    if values.size == 0:
        summary.update(
            {
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
        )
        return summary

    summary.update(
        {
            "count": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std_dev": float(values.std(ddof=0)),
            "min": float(values.min()),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
        }
    )
    return summary


def build_similarity_distribution_summary(
    similarity_rows_df,
    group_columns: Sequence[str],
    score_col: str = "cosine_similarity",
) -> "pd.DataFrame":
    """Aggregate similarity values into one summary row per requested group."""
    import pandas as pd

    if score_col not in similarity_rows_df.columns:
        raise ValueError(f"Expected score column {score_col!r} in similarity_rows_df.")

    summary_rows = []
    grouped = similarity_rows_df.groupby(list(group_columns), sort=False, dropna=False)
    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        extra_fields = dict(zip(group_columns, group_key, strict=False))
        summary_rows.append(
            _summarize_similarity_values(
                group_df[score_col].tolist(),
                extra_fields=extra_fields,
            )
        )

    return pd.DataFrame(summary_rows)


def _build_matrix_labels(sequence_df, accession_col: str) -> list[str]:
    """Build readable, unique labels for saved similarity matrices."""
    labels: list[str] = []
    label_counts: dict[str, int] = {}

    for row_number, accession in enumerate(sequence_df[accession_col].tolist(), start=1):
        base_label = accession if str(accession).strip() else f"row_{row_number:04d}"
        label_counts[base_label] = label_counts.get(base_label, 0) + 1
        duplicate_count = label_counts[base_label]
        labels.append(base_label if duplicate_count == 1 else f"{base_label}__{duplicate_count}")

    return labels


def build_all_vs_all_artifacts(
    corpus_df,
    normalized_embeddings,
    accession_col: str = "accession",
    sequence_col: str = "sequence",
    top_k: int = 10,
) -> dict:
    """Build matrix, summary, and top-neighbor views for the full test-set corpus.

    This is the background distribution for the entire held-out test set. It
    tells us what "typical" similarities look like before we zoom in on any one
    selected glycan.
    """
    import pandas as pd

    cleaned_corpus_df = _clean_similarity_dataframe(corpus_df, accession_col=accession_col, sequence_col=sequence_col)
    similarity_tensor = normalized_embeddings @ normalized_embeddings.T
    matrix_labels = _build_matrix_labels(cleaned_corpus_df, accession_col=accession_col)
    similarity_matrix_df = pd.DataFrame(
        similarity_tensor.numpy(),
        index=matrix_labels,
        columns=matrix_labels,
    )

    corpus_size = len(cleaned_corpus_df)
    upper_triangle_indices = torch.triu_indices(corpus_size, corpus_size, offset=1)
    unique_pair_scores = similarity_tensor[upper_triangle_indices[0], upper_triangle_indices[1]].numpy()
    off_diagonal_summary_df = pd.DataFrame(
        [
            _summarize_similarity_values(
                unique_pair_scores,
                extra_fields={
                    "scope": "all_vs_all_unique_pairs",
                    "num_glycans": int(corpus_size),
                },
            )
        ]
    )

    top_neighbor_rows = []
    neighbor_count = min(int(top_k), max(corpus_size - 1, 0))
    if neighbor_count > 0:
        for row_index in range(corpus_size):
            score_row = similarity_tensor[row_index].clone()
            # Force the diagonal out of the ranking so "nearest neighbor" means
            # something useful rather than just returning the glycan itself.
            score_row[row_index] = float("-inf")
            top_indices = torch.topk(score_row, k=neighbor_count).indices.tolist()
            source_row = cleaned_corpus_df.iloc[row_index]

            for rank, neighbor_index in enumerate(top_indices, start=1):
                neighbor_row = cleaned_corpus_df.iloc[neighbor_index]
                top_neighbor_rows.append(
                    {
                        "source_accession": source_row[accession_col],
                        "source_sequence": source_row[sequence_col],
                        "neighbor_accession": neighbor_row[accession_col],
                        "neighbor_sequence": neighbor_row[sequence_col],
                        "rank": rank,
                        "cosine_similarity": float(similarity_tensor[row_index, neighbor_index].item()),
                    }
                )

    top_neighbors_df = pd.DataFrame(
        top_neighbor_rows,
        columns=[
            "source_accession",
            "source_sequence",
            "neighbor_accession",
            "neighbor_sequence",
            "rank",
            "cosine_similarity",
        ],
    )
    return {
        "similarity_tensor": similarity_tensor,
        "similarity_matrix_df": similarity_matrix_df,
        "unique_pair_scores": unique_pair_scores,
        "off_diagonal_summary_df": off_diagonal_summary_df,
        "top_neighbors_df": top_neighbors_df,
    }


# ---------------------------------------------------------------------------
# Specific-vs-all ranking and threshold-cloud helpers
# ---------------------------------------------------------------------------

def compare_queries_to_corpus(
    query_df,
    query_normalized_embeddings,
    corpus_df,
    corpus_normalized_embeddings,
    accession_col: str = "accession",
    sequence_col: str = "sequence",
) -> "pd.DataFrame":
    """Compare one or more selected glycans against an entire test-set corpus.

    The returned dataframe is the master ranked list. The threshold-cloud view is
    just a filtered slice of this ranked list at one or more similarity cutoffs.
    """
    import pandas as pd

    cleaned_query_df = _clean_similarity_dataframe(query_df, accession_col=accession_col, sequence_col=sequence_col)
    cleaned_corpus_df = _clean_similarity_dataframe(corpus_df, accession_col=accession_col, sequence_col=sequence_col)
    comparison_frames = []

    for query_index, query_row in cleaned_query_df.iterrows():
        similarity_scores = corpus_normalized_embeddings @ query_normalized_embeddings[query_index]
        comparison_df = cleaned_corpus_df.rename(
            columns={
                accession_col: "corpus_accession",
                sequence_col: "corpus_sequence",
            }
        ).copy()
        comparison_df.insert(0, "query_sequence", query_row[sequence_col])
        comparison_df.insert(0, "query_accession", query_row[accession_col])
        comparison_df["cosine_similarity"] = similarity_scores.numpy()

        # In the scale-up workflow the held-out corpus may only have plain text
        # split rows rather than trusted accession labels. Sequence identity is
        # therefore the most stable self-match rule: anything with the exact
        # same sequence as the query is treated as the self row for ranking and
        # summary purposes.
        comparison_df["is_self_match"] = comparison_df["corpus_sequence"] == query_row[sequence_col]
        comparison_df = comparison_df.sort_values(
            ["cosine_similarity", "corpus_accession", "corpus_sequence"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        comparison_df["rank"] = None

        non_self_indices = comparison_df.index[~comparison_df["is_self_match"]].tolist()
        for rank, row_position in enumerate(non_self_indices, start=1):
            comparison_df.at[row_position, "rank"] = rank

        comparison_frames.append(comparison_df)

    return pd.concat(comparison_frames, ignore_index=True)


def build_threshold_cloud_table(
    query_results_df,
    thresholds: Sequence[float],
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Return threshold-cloud membership rows plus one summary row per threshold.

    A "similarity cloud" is simply: take the specific-vs-all ranked results for a
    query, remove the self-match, and keep every glycan whose score is at least
    the chosen threshold.
    """
    import pandas as pd

    threshold_membership_rows = []
    threshold_summary_rows = []
    # Deduplicate and sort once so the notebook can pass thresholds in any order.
    sorted_thresholds = sorted({float(threshold) for threshold in thresholds}, reverse=True)

    for (query_accession, query_sequence), query_group_df in query_results_df.groupby(
        ["query_accession", "query_sequence"],
        sort=False,
    ):
        non_self_df = query_group_df.loc[~query_group_df["is_self_match"]].copy()
        non_self_df = non_self_df.sort_values(
            ["cosine_similarity", "corpus_accession", "corpus_sequence"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)

        for threshold in sorted_thresholds:
            # Each threshold defines one cloud: all non-self neighbors scoring at
            # or above that cutoff.
            cloud_df = non_self_df.loc[non_self_df["cosine_similarity"] >= threshold].copy().reset_index(drop=True)
            cloud_size = int(len(cloud_df))
            threshold_summary_rows.append(
                {
                    "query_accession": query_accession,
                    "query_sequence": query_sequence,
                    "threshold": float(threshold),
                    "cloud_size": cloud_size,
                    "max_similarity": float(cloud_df["cosine_similarity"].max()) if cloud_size else float("nan"),
                    "min_similarity": float(cloud_df["cosine_similarity"].min()) if cloud_size else float("nan"),
                }
            )

            if cloud_size == 0:
                continue

            cloud_df = cloud_df.copy()
            cloud_df["threshold"] = float(threshold)
            cloud_df["cloud_rank"] = range(1, cloud_size + 1)
            cloud_df["cloud_size"] = cloud_size
            threshold_membership_rows.extend(cloud_df.to_dict("records"))

    threshold_membership_df = pd.DataFrame(
        threshold_membership_rows,
        columns=[
            "query_accession",
            "query_sequence",
            "corpus_accession",
            "corpus_sequence",
            "cosine_similarity",
            "is_self_match",
            "rank",
            "threshold",
            "cloud_rank",
            "cloud_size",
        ],
    )
    threshold_summary_df = pd.DataFrame(
        threshold_summary_rows,
        columns=[
            "query_accession",
            "query_sequence",
            "threshold",
            "cloud_size",
            "max_similarity",
            "min_similarity",
        ],
    )

    if not threshold_membership_df.empty:
        threshold_membership_df = threshold_membership_df.sort_values(
            ["query_accession", "threshold", "cloud_rank"],
            ascending=[True, False, True],
            kind="stable",
        ).reset_index(drop=True)

    if not threshold_summary_df.empty:
        threshold_summary_df = threshold_summary_df.sort_values(
            ["query_accession", "threshold"],
            ascending=[True, False],
            kind="stable",
        ).reset_index(drop=True)

    return threshold_membership_df, threshold_summary_df


# ---------------------------------------------------------------------------
# Plotting and small HTML helpers
# ---------------------------------------------------------------------------

def plot_similarity_distribution_histogram(
    similarity_values,
    output_path,
    title: str,
    bins: int = 40,
) -> Path:
    """Save a histogram for one similarity distribution and return the file path."""
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    values = np.asarray(list(similarity_values), dtype=float)
    if values.size == 0:
        raise ValueError("Need at least one similarity value to plot a distribution histogram.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        values,
        bins=int(bins),
        range=(-1.0, 1.0),
        color="#4c78a8",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axvline(values.mean(), color="#d62728", linestyle="--", linewidth=1.5, label="mean")
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _collect_scaleup_html_sequences(
    query_df,
    query_results_df,
    threshold_cloud_df,
    neighbor_limit: int,
    cloud_limit: int,
) -> list[str]:
    """Collect the subset of sequences that should get cartoons in the HTML reports.

    We only fetch cartoons for glycans that will actually appear on the HTML
    pages. That keeps the report generation step lighter and the cached asset
    folder easier to inspect.
    """
    sequences: list[str] = []
    seen_sequences: set[str] = set()

    def _maybe_add(sequence: str) -> None:
        if sequence not in seen_sequences:
            seen_sequences.add(sequence)
            sequences.append(sequence)

    for sequence in query_df["sequence"].tolist():
        _maybe_add(str(sequence))

    for _, query_group_df in query_results_df.groupby(["query_accession", "query_sequence"], sort=False):
        neighbor_slice = (
            query_group_df.loc[~query_group_df["is_self_match"]]
            .sort_values(["rank"], kind="stable")
            .head(int(neighbor_limit))
        )
        for sequence in neighbor_slice["corpus_sequence"].tolist():
            _maybe_add(str(sequence))

    if not threshold_cloud_df.empty:
        for _, cloud_group_df in threshold_cloud_df.groupby(
            ["query_accession", "query_sequence", "threshold"],
            sort=False,
        ):
            limited_cloud_df = cloud_group_df.sort_values(["cloud_rank"], kind="stable").head(int(cloud_limit))
            for sequence in limited_cloud_df["corpus_sequence"].tolist():
                _maybe_add(str(sequence))

    return sequences


def _collect_scaleup_html_sequence_records(
    query_df,
    query_results_df,
    threshold_cloud_df,
    neighbor_limit: int,
    cloud_limit: int,
) -> list[dict[str, str]]:
    """Collect HTML-visible glycans together with any accession we already know.

    The report only needs cartoons for glycans that will actually appear in the
    HTML. Keeping accession information attached here lets the cartoon helper use
    direct accession image URLs when they are available instead of depending on a
    separate lookup service.
    """
    sequence_records: list[dict[str, str]] = []
    seen_sequences: set[str] = set()

    def _maybe_add(sequence: str, accession: str) -> None:
        normalized_sequence = str(sequence).strip()
        if not normalized_sequence or normalized_sequence in seen_sequences:
            return
        seen_sequences.add(normalized_sequence)
        sequence_records.append(
            {
                "sequence": normalized_sequence,
                "accession": str(accession).strip(),
            }
        )

    for query_row in query_df.itertuples(index=False):
        _maybe_add(str(query_row.sequence), str(query_row.accession))

    for _, query_group_df in query_results_df.groupby(["query_accession", "query_sequence"], sort=False):
        neighbor_slice = (
            query_group_df.loc[~query_group_df["is_self_match"]]
            .sort_values(["rank"], kind="stable")
            .head(int(neighbor_limit))
        )
        for row in neighbor_slice.itertuples(index=False):
            _maybe_add(str(row.corpus_sequence), str(row.corpus_accession))

    if not threshold_cloud_df.empty:
        for _, cloud_group_df in threshold_cloud_df.groupby(
            ["query_accession", "query_sequence", "threshold"],
            sort=False,
        ):
            limited_cloud_df = cloud_group_df.sort_values(["cloud_rank"], kind="stable").head(int(cloud_limit))
            for row in limited_cloud_df.itertuples(index=False):
                _maybe_add(str(row.corpus_sequence), str(row.corpus_accession))

    return sequence_records


def _format_summary_table_rows(summary_row: dict, ordered_fields: Sequence[tuple[str, str]]) -> str:
    """Render a compact two-column HTML summary table from one summary row."""
    row_html_parts = []
    for field_name, label in ordered_fields:
        value = summary_row.get(field_name, "")
        if isinstance(value, float):
            display_value = f"{value:.3f}" if value == value else "NA"
        else:
            display_value = str(value)
        row_html_parts.append(
            "<tr>"
            f"<th>{escape(label)}</th>"
            f"<td>{escape(display_value)}</td>"
            "</tr>"
        )

    return "<table class='summary-table'><tbody>" + "".join(row_html_parts) + "</tbody></table>"


# ---------------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------------

def attach_pca_to_scaleup_index_html(
    index_html_path,
    image_filename: str | None = None,
    focus_accession: str | None = None,
    threshold: float = 0.90,
    pca_panels: Sequence[dict] | None = None,
) -> Path:
    """Insert one or more PCA sections into the saved top-level HTML report.

    On reruns, this helper replaces the older PCA section instead of stacking
    duplicates. When multiple focus accessions are supplied, each gets its own
    PCA card tied to the same active threshold.
    """
    index_html_path = Path(index_html_path)
    html = index_html_path.read_text(encoding="utf-8")

    start_marker = "<!-- PCA_SECTION_START -->"
    end_marker = "<!-- PCA_SECTION_END -->"
    if start_marker in html and end_marker in html:
        start_index = html.index(start_marker)
        end_index = html.index(end_marker) + len(end_marker)
        html = html[:start_index] + html[end_index:]

    if pca_panels is None:
        if image_filename is None or focus_accession is None:
            raise ValueError("Provide either pca_panels or the single-image PCA arguments.")
        pca_panels = [
            {
                "image_filename": image_filename,
                "focus_accession": focus_accession,
                "threshold": float(threshold),
            }
        ]

    panel_html_parts = []
    for panel in pca_panels:
        panel_threshold = float(panel.get("threshold", threshold))
        panel_focus_accession = str(panel["focus_accession"])
        panel_image_filename = str(panel["image_filename"])
        panel_html_parts.append(
            "  <div class='analysis-card'>"
            "<h2>PCA Embedding View</h2>"
            f"<p>This PCA uses the active threshold <strong>{panel_threshold:.2f}</strong> "
            f"and highlights the cloud for <strong>{escape(panel_focus_accession)}</strong>. "
            "It is a simple embedding-space view to support the ranked neighbors and "
            "threshold-cloud results, not to replace them.</p>"
            f"<img src='{escape(panel_image_filename, quote=True)}' "
            f"alt='PCA embedding view for {escape(panel_focus_accession)} at threshold {panel_threshold:.2f}'>"
            "</div>"
        )

    pca_html = "\n  <!-- PCA_SECTION_START -->\n" + "\n".join(panel_html_parts) + "\n  <!-- PCA_SECTION_END -->\n"
    html = html.replace("</body>", pca_html + "</body>")
    index_html_path.write_text(html, encoding="utf-8")
    return index_html_path


def attach_pca_to_specific_html(
    query_html_path,
    image_path,
    focus_accession: str,
    threshold: float,
) -> Path:
    """Insert one embedded PCA section into one accession-specific HTML report.

    The PCA belongs with the accession whose cloud is being highlighted. This
    helper embeds the saved image as a data URI so the HTML stays portable even
    if the surrounding files are moved or opened directly from disk.
    """
    query_html_path = Path(query_html_path)
    html = query_html_path.read_text(encoding="utf-8")

    start_marker = "<!-- PCA_SECTION_START -->"
    end_marker = "<!-- PCA_SECTION_END -->"
    if start_marker in html and end_marker in html:
        start_index = html.index(start_marker)
        end_index = html.index(end_marker) + len(end_marker)
        html = html[:start_index] + html[end_index:]

    image_data_uri = _image_path_to_data_uri(image_path)
    pca_html = (
        "\n  <!-- PCA_SECTION_START -->\n"
        "  <div class='analysis-card'>"
        "<h2>PCA Embedding View</h2>"
        f"<p class='section-note'>This PCA uses the active threshold <strong>{threshold:.2f}</strong> "
        f"and highlights the cloud for <strong>{escape(str(focus_accession))}</strong>. "
        "It is a simple embedding-space view to support the ranked neighbors and "
        "threshold-cloud results, not to replace them.</p>"
        f"<img src='{escape(image_data_uri, quote=True)}' "
        f"alt='PCA embedding view for {escape(str(focus_accession))} at threshold {threshold:.2f}'>"
        "</div>\n  <!-- PCA_SECTION_END -->\n"
    )
    html = html.replace("</body>", pca_html + "</body>")
    query_html_path.write_text(html, encoding="utf-8")
    return query_html_path


def build_focus_cloud_membership(
    threshold_cloud_df,
    accession: str,
    threshold: float,
) -> tuple["pd.DataFrame", set[str]]:
    """Return one query's threshold-cloud rows plus the accession membership set."""
    focus_cloud_df = threshold_cloud_df.loc[
        (threshold_cloud_df["query_accession"] == accession)
        & (threshold_cloud_df["threshold"] == float(threshold))
    ].copy()
    focus_cloud_accessions = set(focus_cloud_df["corpus_accession"].tolist())
    return focus_cloud_df, focus_cloud_accessions


def _normalize_focus_accessions(
    focus_accessions,
    selected_accessions: Sequence[str],
) -> list[str]:
    """Return a validated list of focus accessions for PCA rendering.

    The PCA helper supports three cases:
    - `None`: default to the first selected accession
    - one accession as a string
    - multiple accessions as a list/tuple
    """
    selected_accession_list = [str(accession) for accession in selected_accessions]
    if not selected_accession_list:
        raise ValueError("Need at least one selected glycan to choose PCA focus accessions.")

    if focus_accessions is None:
        requested_focus_accessions = [selected_accession_list[0]]
    elif isinstance(focus_accessions, str):
        requested_focus_accessions = [focus_accessions]
    else:
        requested_focus_accessions = [str(accession) for accession in focus_accessions]

    normalized_focus_accessions: list[str] = []
    seen_accessions: set[str] = set()
    for accession in requested_focus_accessions:
        if accession not in seen_accessions:
            normalized_focus_accessions.append(accession)
            seen_accessions.add(accession)

    invalid_accessions = [
        accession for accession in normalized_focus_accessions
        if accession not in selected_accession_list
    ]
    if invalid_accessions:
        raise ValueError("PCA focus accession(s) must be included in the selected glycan run panel.")

    return normalized_focus_accessions


def save_scaleup_pca_outputs(
    results_bundle: dict,
    query_metadata_df,
    output_dir,
    focus_accessions: str | Sequence[str] | None = None,
    threshold: float = 0.90,
    background_sample_size: int | None = 2000,
    random_state: int = 7,
    background_point_size: int | float = 10,
    cloud_point_size: int | float = 28,
    query_point_size: int | float = 110,
    image_filename: str = "pca_embedding_view.png",
    coordinates_filename: str = "pca_coordinates.csv",
    selected_filename: str = "pca_selected_glycans.csv",
    include_in_html: bool = True,
) -> dict:
    """Build, save, and optionally attach scale-up PCA view(s) to HTML.

    This helper treats PCA as a supporting report artifact built from the
    already-computed similarity embeddings. It saves one image, one full
    coordinate table, one selected-glycan coordinate table, and one or more
    focus-specific PCA images. It can also patch the accession-specific HTML
    reports so those PCA views travel with the rest of the results.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    query_df = _clean_similarity_dataframe(results_bundle["query_df"], accession_col="accession", sequence_col="sequence")
    corpus_df = _clean_similarity_dataframe(results_bundle["corpus_df"], accession_col="accession", sequence_col="sequence")
    selected_accessions = query_df["accession"].tolist()
    if not selected_accessions:
        raise ValueError("Need at least one selected glycan to build PCA outputs.")
    normalized_focus_accessions = _normalize_focus_accessions(focus_accessions, selected_accessions)

    metadata_df = pd.DataFrame(query_metadata_df).copy()
    if "accession" not in metadata_df.columns:
        raise ValueError("query_metadata_df must include an 'accession' column.")
    label_lookup = (
        metadata_df.assign(accession=metadata_df["accession"].map(str))
        .set_index("accession")
        .get("label")
    )
    label_lookup = {} if label_lookup is None else label_lookup.to_dict()

    query_plot_df = query_df[["accession", "sequence"]].copy()
    query_plot_df["label"] = query_plot_df["accession"].map(
        lambda accession: str(label_lookup.get(str(accession), accession))
    )
    corpus_plot_df = corpus_df[["accession", "sequence"]].copy()

    corpus_embeddings = results_bundle["corpus_embedding_bundle"]["normalized_embeddings"].numpy()
    query_embeddings = results_bundle["query_embedding_bundle"]["normalized_embeddings"].numpy()
    all_embeddings = np.vstack([corpus_embeddings, query_embeddings])

    pca = PCA(n_components=2)
    all_coordinates = pca.fit_transform(all_embeddings)

    corpus_coordinates = all_coordinates[: len(corpus_plot_df)]
    query_coordinates = all_coordinates[len(corpus_plot_df) :]
    corpus_plot_df["pc1"] = corpus_coordinates[:, 0]
    corpus_plot_df["pc2"] = corpus_coordinates[:, 1]
    query_plot_df["pc1"] = query_coordinates[:, 0]
    query_plot_df["pc2"] = query_coordinates[:, 1]

    explained_variance = pca.explained_variance_ratio_ * 100

    html_dir = Path(results_bundle["saved_paths"]["html_dir"])
    pca_coordinates_path = output_path / coordinates_filename
    pca_selected_path = output_path / selected_filename
    query_pca_export_df = query_plot_df[["accession", "sequence", "label", "pc1", "pc2"]].copy()
    query_pca_export_df["point_group"] = "selected_glycan"
    corpus_pca_export_df = corpus_plot_df[["accession", "sequence", "pc1", "pc2"]].copy()
    corpus_pca_export_df["point_group"] = "test_background"
    pca_coordinates_df = pd.concat(
        [corpus_pca_export_df, query_pca_export_df],
        ignore_index=True,
        sort=False,
    )
    pca_coordinates_df.to_csv(pca_coordinates_path, index=False)
    selected_coordinates_df = query_plot_df[["accession", "label", "pc1", "pc2"]].copy()
    selected_coordinates_df.to_csv(pca_selected_path, index=False)

    pca_image_paths: dict[str, Path] = {}
    focus_cloud_dfs: dict[str, "pd.DataFrame"] = {}
    background_counts: dict[str, int] = {}
    focus_cloud_sizes: dict[str, int] = {}
    pca_html_paths: dict[str, Path] = {}

    for focus_accession in normalized_focus_accessions:
        focus_cloud_df, focus_cloud_accessions = build_focus_cloud_membership(
            results_bundle["threshold_cloud_df"],
            accession=focus_accession,
            threshold=threshold,
        )
        focus_cloud_dfs[focus_accession] = focus_cloud_df
        focus_cloud_sizes[focus_accession] = int(len(focus_cloud_df))

        focus_plot_df = corpus_plot_df.copy()
        focus_plot_df["in_focus_cloud"] = focus_plot_df["accession"].isin(focus_cloud_accessions)

        background_plot_df = focus_plot_df.loc[~focus_plot_df["in_focus_cloud"]].copy()
        if background_sample_size is not None and len(background_plot_df) > int(background_sample_size):
            background_plot_df = background_plot_df.sample(
                n=int(background_sample_size),
                random_state=int(random_state),
            )
        background_counts[focus_accession] = int(len(background_plot_df))
        focus_cloud_plot_df = focus_plot_df.loc[focus_plot_df["in_focus_cloud"]].copy()

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            background_plot_df["pc1"],
            background_plot_df["pc2"],
            s=background_point_size,
            c="#c7c7c7",
            alpha=0.45,
            edgecolors="none",
            label="Test set background",
        )
        if not focus_cloud_plot_df.empty:
            ax.scatter(
                focus_cloud_plot_df["pc1"],
                focus_cloud_plot_df["pc2"],
                s=cloud_point_size,
                c="#2a9d8f",
                alpha=0.85,
                edgecolors="white",
                linewidths=0.4,
                label=f"{focus_accession} active cloud",
            )
        ax.scatter(
            query_plot_df["pc1"],
            query_plot_df["pc2"],
            s=query_point_size,
            c="#d1495b",
            marker="X",
            edgecolors="black",
            linewidths=0.7,
            label="Selected glycans",
        )
        for row in query_plot_df.itertuples(index=False):
            ax.annotate(
                row.accession,
                (row.pc1, row.pc2),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title("PCA of test-set embeddings with selected glycans overlaid")
        ax.set_xlabel(f"PC1 ({explained_variance[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({explained_variance[1]:.1f}% variance)")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        fig.tight_layout()

        if len(normalized_focus_accessions) == 1:
            pca_image_path = html_dir / image_filename
        else:
            image_stem = Path(image_filename).stem
            image_suffix = Path(image_filename).suffix or ".png"
            pca_image_path = html_dir / f"{image_stem}_{focus_accession}{image_suffix}"
        fig.savefig(pca_image_path, dpi=200)
        plt.close(fig)

        pca_image_paths[focus_accession] = pca_image_path
    if include_in_html:
        query_html_paths = results_bundle["saved_paths"].get("query_html_paths", {})
        for focus_accession, pca_image_path in pca_image_paths.items():
            query_html_path = query_html_paths.get(focus_accession)
            if query_html_path is None:
                continue
            pca_html_paths[focus_accession] = attach_pca_to_specific_html(
                query_html_path=query_html_path,
                image_path=pca_image_path,
                focus_accession=focus_accession,
                threshold=float(threshold),
            )

    primary_focus_accession = normalized_focus_accessions[0]
    primary_pca_image_path = pca_image_paths[primary_focus_accession]
    return {
        "pca_image_path": primary_pca_image_path,
        "pca_image_paths": pca_image_paths,
        "pca_html_paths": pca_html_paths,
        "pca_coordinates_path": pca_coordinates_path,
        "pca_selected_path": pca_selected_path,
        "selected_coordinates_df": selected_coordinates_df,
        "pca_coordinates_df": pca_coordinates_df,
        "focus_cloud_df": focus_cloud_dfs[primary_focus_accession],
        "focus_cloud_dfs": focus_cloud_dfs,
        "focus_accession": primary_focus_accession,
        "focus_accessions": normalized_focus_accessions,
        "threshold": float(threshold),
        "explained_variance": explained_variance,
        "background_count": background_counts[primary_focus_accession],
        "background_counts": background_counts,
        "focus_cloud_size": focus_cloud_sizes[primary_focus_accession],
        "focus_cloud_sizes": focus_cloud_sizes,
    }


# ---------------------------------------------------------------------------
# HTML report builders
# ---------------------------------------------------------------------------

def render_specific_vs_all_html(
    query_row,
    query_results_df,
    distribution_summary_row: dict,
    threshold_summary_df,
    threshold_cloud_df,
    cartoon_lookup: dict[str, dict[str, str]],
    output_path,
    output_name: str,
    histogram_image_path: str | None = None,
    neighbor_limit: int = 50,
    cloud_display_limit: int = 100,
) -> Path:
    """Render one standalone HTML page for a selected glycan's specific-vs-all results."""
    query_accession = str(query_row["accession"])
    query_sequence = str(query_row["sequence"])
    query_cartoon = cartoon_lookup.get(query_sequence)

    non_self_df = (
        query_results_df.loc[~query_results_df["is_self_match"]]
        .sort_values(["rank"], kind="stable")
        .reset_index(drop=True)
    )
    # Keep the nearest-neighbor table separate from the cloud tables below.
    # It gives one simple ranked view even if the user later changes thresholds.
    top_neighbors_df = non_self_df.head(int(neighbor_limit))

    top_neighbor_rows = []
    for row in top_neighbors_df.itertuples(index=False):
        neighbor_cartoon = cartoon_lookup.get(row.corpus_sequence)
        top_neighbor_rows.append(
            "<tr>"
            f"<td>{int(row.rank)}</td>"
            f"<td>{escape(str(row.corpus_accession))}</td>"
            f"<td>{row.cosine_similarity:.3f}</td>"
            f"<td>{format_glycan_sequence_block(str(row.corpus_sequence), neighbor_cartoon)}</td>"
            "</tr>"
        )

    threshold_summary_rows = []
    for row in threshold_summary_df.itertuples(index=False):
        threshold_summary_rows.append(
            "<tr>"
            f"<td>{row.threshold:.2f}</td>"
            f"<td>{int(row.cloud_size)}</td>"
            f"<td>{'NA' if row.max_similarity != row.max_similarity else f'{row.max_similarity:.3f}'}</td>"
            f"<td>{'NA' if row.min_similarity != row.min_similarity else f'{row.min_similarity:.3f}'}</td>"
            "</tr>"
        )

    cloud_section_parts = []
    for row in threshold_summary_df.itertuples(index=False):
        threshold_value = float(row.threshold)
        threshold_cloud_rows = threshold_cloud_df.loc[
            threshold_cloud_df["threshold"] == threshold_value
        ].sort_values(["cloud_rank"], kind="stable")
        limited_cloud_rows = threshold_cloud_rows.head(int(cloud_display_limit))

        if int(row.cloud_size) == 0:
            cloud_section_parts.append(
                "<div class='cloud-panel'>"
                f"<h2>Similarity Cloud: score >= {threshold_value:.2f}</h2>"
                "<p class='section-note'>No test-set glycans cleared this threshold once the self-match was removed.</p>"
                "</div>"
            )
            continue

        cloud_row_html_parts = []
        for cloud_row in limited_cloud_rows.itertuples(index=False):
            cloud_cartoon = cartoon_lookup.get(cloud_row.corpus_sequence)
            cloud_row_html_parts.append(
                "<tr>"
                f"<td>{int(cloud_row.cloud_rank)}</td>"
                f"<td>{escape(str(cloud_row.corpus_accession))}</td>"
                f"<td>{cloud_row.cosine_similarity:.3f}</td>"
                f"<td>{format_glycan_sequence_block(str(cloud_row.corpus_sequence), cloud_cartoon)}</td>"
                "</tr>"
            )

        truncation_note = ""
        if len(threshold_cloud_rows) > len(limited_cloud_rows):
            truncation_note = (
                f"<p class='section-note'>Showing the first {len(limited_cloud_rows)} cloud members here. "
                "The CSV output keeps the full threshold cloud.</p>"
            )

        cloud_section_parts.append(
            "<div class='cloud-panel'>"
            f"<h2>Similarity Cloud: score >= {threshold_value:.2f}</h2>"
            f"<p class='section-note'>Cloud size: {int(row.cloud_size)} glycans.</p>"
            f"{truncation_note}"
            "<table>"
            "<thead><tr>"
            "<th>Cloud Rank</th>"
            "<th>Accession</th>"
            "<th>Cosine Similarity</th>"
            "<th>Glycan</th>"
            "</tr></thead>"
            f"<tbody>{''.join(cloud_row_html_parts)}</tbody>"
            "</table>"
            "</div>"
        )

    histogram_html = ""
    if histogram_image_path:
        histogram_html = (
            "<div class='analysis-card'>"
            "<h2>Specific-vs-All Similarity Distribution</h2>"
            f"<img src='{escape(histogram_image_path, quote=True)}' alt='Similarity histogram for {escape(query_accession)}'>"
            "</div>"
        )

    summary_html = _format_summary_table_rows(
        distribution_summary_row,
        ordered_fields=(
            ("count", "Compared test glycans"),
            ("mean", "Mean similarity"),
            ("median", "Median similarity"),
            ("std_dev", "Standard deviation"),
            ("min", "Minimum"),
            ("q05", "5th percentile"),
            ("q25", "25th percentile"),
            ("q75", "75th percentile"),
            ("q95", "95th percentile"),
            ("max", "Maximum"),
        ),
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(output_name)} - {escape(query_accession)}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px auto;
      max-width: 1220px;
      color: #222;
      line-height: 1.45;
      padding: 0 18px 36px;
    }}
    h1, h2 {{
      color: #111;
    }}
    .query-panel,
    .analysis-card,
    .cloud-panel {{
      background: #fafafa;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 16px 18px;
      margin-bottom: 24px;
    }}
    .analysis-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      margin-bottom: 24px;
    }}
    .analysis-card img {{
      width: 100%;
      height: auto;
      border: 1px solid #eee;
      background: white;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f0f0f0;
    }}
    .summary-table th {{
      width: 220px;
      background: #f7f7f7;
    }}
    .section-note {{
      margin: 0 0 10px;
      color: #555;
    }}
    .sequence-block {{
      display: flex;
      align-items: flex-start;
      gap: 16px;
      min-width: 360px;
    }}
    .cartoon img {{
      max-width: 180px;
      max-height: 70px;
      border: 1px solid #eee;
      background: white;
      padding: 4px;
    }}
    .cartoon-caption {{
      margin-top: 6px;
      font-size: 12px;
      color: #666;
    }}
    .cartoon-missing {{
      color: #777;
      font-size: 13px;
      border: 1px dashed #bbb;
      padding: 8px 10px;
      background: #fafafa;
    }}
    .sequence-text {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-size: 13px;
      font-family: Menlo, Monaco, Consolas, monospace;
    }}
  </style>
</head>
<body>
  <h1>{escape(output_name)} - {escape(query_accession)}</h1>
  <div class="query-panel">
    <h2>Selected Glycan</h2>
    {format_glycan_sequence_block(query_sequence, query_cartoon)}
  </div>
  <div class="analysis-grid">
    {histogram_html}
    <div class="analysis-card">
      <h2>Distribution Summary</h2>
      <p class="section-note">This is the shape of the full specific-vs-all score distribution after removing the self-match.</p>
      {summary_html}
    </div>
  </div>
  <div class="analysis-card">
    <h2>Threshold Cloud Summary</h2>
    <p class="section-note">This is the quick count table for the threshold-based similarity clouds.</p>
    <table>
      <thead><tr><th>Threshold</th><th>Cloud Size</th><th>Max Similarity</th><th>Min Similarity</th></tr></thead>
      <tbody>{''.join(threshold_summary_rows)}</tbody>
    </table>
  </div>
  <div class="analysis-card">
    <h2>Top Similar Glycans</h2>
    <p class="section-note">This is the ranked nearest-neighbor view, separate from the threshold clouds below.</p>
    <table>
      <thead><tr><th>Rank</th><th>Accession</th><th>Cosine Similarity</th><th>Glycan</th></tr></thead>
      <tbody>{''.join(top_neighbor_rows)}</tbody>
    </table>
  </div>
  {''.join(cloud_section_parts)}
</body>
</html>
"""

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def render_scaleup_index_html(
    query_df,
    query_distribution_summary_df,
    threshold_summary_df,
    query_html_paths: dict[str, Path],
    output_path,
    output_name: str,
    all_vs_all_summary_row: dict,
    cartoon_lookup: dict[str, dict[str, str]],
    all_vs_all_histogram_path: str | None = None,
) -> Path:
    """Render one top-level HTML index for the scale-up similarity analysis."""
    query_rows_html = []

    for query_row in query_df.itertuples(index=False):
        query_accession = str(query_row.accession)
        query_sequence = str(query_row.sequence)
        summary_row = (
            query_distribution_summary_df.loc[
                query_distribution_summary_df["query_accession"] == query_accession
            ]
            .iloc[0]
            .to_dict()
        )
        cloud_090 = threshold_summary_df.loc[
            (threshold_summary_df["query_accession"] == query_accession)
            & (threshold_summary_df["threshold"] == 0.90)
        ]
        cloud_090_size = int(cloud_090["cloud_size"].iloc[0]) if not cloud_090.empty else 0
        query_cartoon = cartoon_lookup.get(query_sequence)
        report_path = query_html_paths[query_accession]
        query_rows_html.append(
            "<tr>"
            f"<td>{escape(query_accession)}</td>"
            f"<td>{format_glycan_sequence_block(query_sequence, query_cartoon)}</td>"
            f"<td>{summary_row['max']:.3f}</td>"
            f"<td>{summary_row['median']:.3f}</td>"
            f"<td>{cloud_090_size}</td>"
            f"<td><a href='{escape(report_path.name, quote=True)}'>{escape(report_path.name)}</a></td>"
            "</tr>"
        )

    all_vs_all_histogram_html = ""
    if all_vs_all_histogram_path:
        all_vs_all_histogram_html = (
            "<div class='analysis-card'>"
            "<h2>All-vs-All Background Distribution</h2>"
            f"<img src='{escape(all_vs_all_histogram_path, quote=True)}' alt='All-vs-all similarity histogram'>"
            "</div>"
        )

    all_vs_all_summary_html = _format_summary_table_rows(
        all_vs_all_summary_row,
        ordered_fields=(
            ("num_glycans", "Test glycans"),
            ("count", "Unique glycan pairs"),
            ("mean", "Mean similarity"),
            ("median", "Median similarity"),
            ("std_dev", "Standard deviation"),
            ("min", "Minimum"),
            ("q05", "5th percentile"),
            ("q25", "25th percentile"),
            ("q75", "75th percentile"),
            ("q95", "95th percentile"),
            ("max", "Maximum"),
        ),
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(output_name)} - Similarity Scale-Up Index</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px auto;
      max-width: 1220px;
      color: #222;
      line-height: 1.45;
      padding: 0 18px 36px;
    }}
    .analysis-card {{
      background: #fafafa;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 16px 18px;
      margin-bottom: 24px;
    }}
    .analysis-card img {{
      width: 100%;
      height: auto;
      border: 1px solid #eee;
      background: white;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f0f0f0;
    }}
    .summary-table th {{
      width: 220px;
      background: #f7f7f7;
    }}
    .sequence-block {{
      display: flex;
      align-items: flex-start;
      gap: 16px;
      min-width: 360px;
    }}
    .cartoon img {{
      max-width: 180px;
      max-height: 70px;
      border: 1px solid #eee;
      background: white;
      padding: 4px;
    }}
    .cartoon-caption {{
      margin-top: 6px;
      font-size: 12px;
      color: #666;
    }}
    .cartoon-missing {{
      color: #777;
      font-size: 13px;
      border: 1px dashed #bbb;
      padding: 8px 10px;
      background: #fafafa;
    }}
    .sequence-text {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-size: 13px;
      font-family: Menlo, Monaco, Consolas, monospace;
    }}
  </style>
</head>
<body>
  <h1>{escape(output_name)} - Similarity Scale-Up Index</h1>
  <p>This report is the test-set scale-up companion to the smaller manual variant notebook.</p>
  {all_vs_all_histogram_html}
  <div class="analysis-card">
    <h2>All-vs-All Summary</h2>
    {all_vs_all_summary_html}
  </div>
  <div class="analysis-card">
    <h2>Selected Glycan Reports</h2>
    <table>
      <thead>
        <tr>
          <th>Accession</th>
          <th>Selected Glycan</th>
          <th>Best Non-Self Similarity</th>
          <th>Median Specific-vs-All Similarity</th>
          <th>Cloud Size At 0.90</th>
          <th>Report</th>
        </tr>
      </thead>
      <tbody>
        {''.join(query_rows_html)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Save helpers and public orchestration
# ---------------------------------------------------------------------------

def save_scaleup_similarity_outputs(
    corpus_df,
    query_df,
    all_vs_all_matrix_df,
    all_vs_all_unique_pair_scores,
    all_vs_all_summary_df,
    all_vs_all_top_neighbors_df,
    specific_vs_all_results_df,
    specific_vs_all_summary_df,
    threshold_cloud_df,
    threshold_summary_df,
    output_dir,
    output_name: str,
    config_payload: dict,
    developer_email: str | None = None,
    cartoon_image_format: str = "svg",
    lookup_timeout: int = 60,
    html_neighbor_limit: int = 50,
    html_cloud_limit: int = 100,
) -> dict:
    """Write the scale-up similarity outputs, including portable HTML reports, to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    corpus_path = output_path / "test_corpus_sequences.csv"
    query_path = output_path / "selected_glycans.csv"
    all_vs_all_matrix_path = output_path / "all_vs_all_similarity_matrix.csv"
    all_vs_all_summary_path = output_path / "all_vs_all_summary.csv"
    all_vs_all_top_neighbors_path = output_path / "all_vs_all_top_neighbors.csv"
    specific_results_path = output_path / "specific_vs_all_ranked.csv"
    specific_summary_path = output_path / "specific_vs_all_distribution_summary.csv"
    threshold_cloud_path = output_path / "specific_vs_all_threshold_clouds.csv"
    threshold_summary_path = output_path / "specific_vs_all_threshold_summary.csv"
    cartoon_manifest_path = output_path / "scaleup_cartoon_manifest.csv"
    config_path = output_path / "similarity_scaleup_config.json"
    html_dir = output_path / "html"
    histogram_dir = output_path / "histograms"
    cartoon_dir = output_path / "cartoons"
    html_dir.mkdir(parents=True, exist_ok=True)
    histogram_dir.mkdir(parents=True, exist_ok=True)
    cartoon_dir.mkdir(parents=True, exist_ok=True)

    corpus_df.to_csv(corpus_path, index=False)
    query_df.to_csv(query_path, index=False)
    all_vs_all_matrix_df.to_csv(all_vs_all_matrix_path)
    all_vs_all_summary_df.to_csv(all_vs_all_summary_path, index=False)
    all_vs_all_top_neighbors_df.to_csv(all_vs_all_top_neighbors_path, index=False)
    specific_vs_all_results_df.to_csv(specific_results_path, index=False)
    specific_vs_all_summary_df.to_csv(specific_summary_path, index=False)
    threshold_cloud_df.to_csv(threshold_cloud_path, index=False)
    threshold_summary_df.to_csv(threshold_summary_path, index=False)

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config_payload, file, indent=2)

    all_vs_all_histogram_path = plot_similarity_distribution_histogram(
        all_vs_all_unique_pair_scores,
        histogram_dir / "all_vs_all_similarity_histogram.png",
        f"{output_name} all-vs-all similarity distribution",
    )
    all_vs_all_histogram_data_uri = _image_path_to_data_uri(all_vs_all_histogram_path)

    query_histogram_paths: dict[str, Path] = {}
    query_html_paths: dict[str, Path] = {}

    html_sequences = _collect_scaleup_html_sequences(
        query_df=query_df,
        query_results_df=specific_vs_all_results_df,
        threshold_cloud_df=threshold_cloud_df,
        neighbor_limit=html_neighbor_limit,
        cloud_limit=html_cloud_limit,
    )
    html_sequence_records = _collect_scaleup_html_sequence_records(
        query_df=query_df,
        query_results_df=specific_vs_all_results_df,
        threshold_cloud_df=threshold_cloud_df,
        neighbor_limit=html_neighbor_limit,
        cloud_limit=html_cloud_limit,
    )
    accession_by_sequence = {
        str(record["sequence"]): str(record["accession"]).strip()
        for record in html_sequence_records
        if str(record["sequence"]).strip()
    }
    existing_cartoon_manifest_df = None
    if cartoon_manifest_path.exists():
        import pandas as pd

        existing_cartoon_manifest_df = pd.read_csv(cartoon_manifest_path)
    # Build cartoons only for sequences that will actually be shown in HTML.
    cartoon_manifest_df = build_cartoon_manifest(
        sequences=html_sequences,
        developer_email=developer_email,
        accession_by_sequence=accession_by_sequence,
        image_format=cartoon_image_format,
        display="compact",
        lookup_timeout=lookup_timeout,
        existing_manifest_df=existing_cartoon_manifest_df,
    )
    cartoon_manifest_df = cache_cartoon_images(
        cartoon_manifest_df=cartoon_manifest_df,
        asset_dir=cartoon_dir,
        image_format=cartoon_image_format,
        download_timeout=lookup_timeout,
    )
    cartoon_manifest_df.to_csv(cartoon_manifest_path, index=False)
    cartoon_lookup = cartoon_lookup_from_manifest(cartoon_manifest_df)

    for query_row in query_df.itertuples(index=False):
        query_accession = str(query_row.accession)
        query_results_subset = specific_vs_all_results_df.loc[
            specific_vs_all_results_df["query_accession"] == query_accession
        ].copy()
        query_summary_row = (
            specific_vs_all_summary_df.loc[specific_vs_all_summary_df["query_accession"] == query_accession]
            .iloc[0]
            .to_dict()
        )
        query_threshold_summary_df = threshold_summary_df.loc[
            threshold_summary_df["query_accession"] == query_accession
        ].copy()
        query_threshold_cloud_df = threshold_cloud_df.loc[
            threshold_cloud_df["query_accession"] == query_accession
        ].copy()

        non_self_scores = query_results_subset.loc[~query_results_subset["is_self_match"], "cosine_similarity"].tolist()
        query_histogram_paths[query_accession] = plot_similarity_distribution_histogram(
            non_self_scores,
            histogram_dir / f"{query_accession}_specific_vs_all_histogram.png",
            f"{output_name} {query_accession} specific-vs-all similarity distribution",
        )
        query_histogram_data_uri = _image_path_to_data_uri(query_histogram_paths[query_accession])
        # Each selected glycan gets its own standalone HTML page so the report can
        # be opened directly in a browser and shared as static files.
        query_html_paths[query_accession] = render_specific_vs_all_html(
            query_row={"accession": query_row.accession, "sequence": query_row.sequence},
            query_results_df=query_results_subset,
            distribution_summary_row=query_summary_row,
            threshold_summary_df=query_threshold_summary_df,
            threshold_cloud_df=query_threshold_cloud_df,
            cartoon_lookup=cartoon_lookup,
            output_path=html_dir / f"{query_accession}_specific_vs_all.html",
            output_name=output_name,
            histogram_image_path=query_histogram_data_uri,
            neighbor_limit=html_neighbor_limit,
            cloud_display_limit=html_cloud_limit,
        )

    index_html_path = render_scaleup_index_html(
        query_df=query_df,
        query_distribution_summary_df=specific_vs_all_summary_df,
        threshold_summary_df=threshold_summary_df,
        query_html_paths=query_html_paths,
        output_path=html_dir / "index.html",
        output_name=output_name,
        all_vs_all_summary_row=all_vs_all_summary_df.iloc[0].to_dict(),
        cartoon_lookup=cartoon_lookup,
        all_vs_all_histogram_path=all_vs_all_histogram_data_uri,
    )

    return {
        "corpus_path": corpus_path,
        "query_path": query_path,
        "all_vs_all_matrix_path": all_vs_all_matrix_path,
        "all_vs_all_summary_path": all_vs_all_summary_path,
        "all_vs_all_top_neighbors_path": all_vs_all_top_neighbors_path,
        "specific_results_path": specific_results_path,
        "specific_summary_path": specific_summary_path,
        "threshold_cloud_path": threshold_cloud_path,
        "threshold_summary_path": threshold_summary_path,
        "cartoon_manifest_path": cartoon_manifest_path,
        "config_path": config_path,
        "html_dir": html_dir,
        "histogram_dir": histogram_dir,
        "cartoon_dir": cartoon_dir,
        "all_vs_all_histogram_path": all_vs_all_histogram_path,
        "query_histogram_paths": query_histogram_paths,
        "query_html_paths": query_html_paths,
        "index_html_path": index_html_path,
    }


def run_scaleup_similarity_analysis(
    tokenizer,
    model,
    corpus_df,
    query_df,
    output_dir,
    output_name: str,
    developer_email: str | None = None,
    accession_col: str = "accession",
    sequence_col: str = "sequence",
    thresholds: Sequence[float] = (0.95, 0.90, 0.85, 0.80),
    cartoon_image_format: str = "svg",
    lookup_timeout: int = 60,
    device: str | torch.device | None = None,
    max_length: int | None = None,
    batch_size: int = 32,
    all_vs_all_top_k: int = 10,
    html_neighbor_limit: int = 50,
    html_cloud_limit: int = 100,
    model_dir=None,
) -> dict:
    """Run the full test-set similarity scale-up workflow and save the outputs."""
    cleaned_corpus_df = _clean_similarity_dataframe(corpus_df, accession_col=accession_col, sequence_col=sequence_col)
    cleaned_query_df = _clean_similarity_dataframe(query_df, accession_col=accession_col, sequence_col=sequence_col)

    # First embed the full held-out test set. Those embeddings power both the
    # all-vs-all background distribution and the specific-vs-all query analysis.
    corpus_embedding_bundle = build_embedding_lookup_for_dataframe(
        cleaned_corpus_df,
        tokenizer=tokenizer,
        model=model,
        accession_col=accession_col,
        sequence_col=sequence_col,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    # The selected glycans do not need to be members of the held-out test split.
    # Embed them separately so specific-vs-all can compare an external query panel
    # against the full test corpus.
    query_embedding_bundle = build_embedding_lookup_for_dataframe(
        cleaned_query_df,
        tokenizer=tokenizer,
        model=model,
        accession_col=accession_col,
        sequence_col=sequence_col,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    query_normalized_embeddings = query_embedding_bundle["normalized_embeddings"]

    all_vs_all_artifacts = build_all_vs_all_artifacts(
        corpus_df=cleaned_corpus_df,
        normalized_embeddings=corpus_embedding_bundle["normalized_embeddings"],
        accession_col=accession_col,
        sequence_col=sequence_col,
        top_k=all_vs_all_top_k,
    )

    specific_vs_all_results_df = compare_queries_to_corpus(
        query_df=cleaned_query_df,
        query_normalized_embeddings=query_normalized_embeddings,
        corpus_df=cleaned_corpus_df,
        corpus_normalized_embeddings=corpus_embedding_bundle["normalized_embeddings"],
        accession_col=accession_col,
        sequence_col=sequence_col,
    )
    specific_vs_all_summary_df = build_similarity_distribution_summary(
        specific_vs_all_results_df.loc[~specific_vs_all_results_df["is_self_match"]].copy(),
        group_columns=("query_accession", "query_sequence"),
    )
    threshold_cloud_df, threshold_summary_df = build_threshold_cloud_table(
        specific_vs_all_results_df,
        thresholds=thresholds,
    )

    # The HTML reports want simple column names so the exported CSVs and notebook
    # displays read naturally instead of carrying "corpus_" / "query_" prefixes
    # everywhere except where they clarify directionality.
    notebook_query_df = cleaned_query_df.rename(
        columns={
            accession_col: "accession",
            sequence_col: "sequence",
        }
    ).copy()

    config_payload = {
        "analysis_type": "similarity_scaleup",
        "model_dir": str(model_dir) if model_dir is not None else "",
        "output_dir": str(output_dir),
        "accession_col": accession_col,
        "sequence_col": sequence_col,
        "thresholds": [float(threshold) for threshold in thresholds],
        "cartoon_image_format": cartoon_image_format,
        "lookup_timeout": int(lookup_timeout),
        "max_length": max_length,
        "batch_size": int(batch_size),
        "all_vs_all_top_k": int(all_vs_all_top_k),
        "html_neighbor_limit": int(html_neighbor_limit),
        "html_cloud_limit": int(html_cloud_limit),
        "selected_accessions": notebook_query_df["accession"].tolist(),
    }
    saved_paths = save_scaleup_similarity_outputs(
        corpus_df=corpus_embedding_bundle["sequence_df"].rename(
            columns={accession_col: "accession", sequence_col: "sequence"}
        ),
        query_df=notebook_query_df,
        all_vs_all_matrix_df=all_vs_all_artifacts["similarity_matrix_df"],
        all_vs_all_unique_pair_scores=all_vs_all_artifacts["unique_pair_scores"],
        all_vs_all_summary_df=all_vs_all_artifacts["off_diagonal_summary_df"],
        all_vs_all_top_neighbors_df=all_vs_all_artifacts["top_neighbors_df"],
        specific_vs_all_results_df=specific_vs_all_results_df,
        specific_vs_all_summary_df=specific_vs_all_summary_df,
        threshold_cloud_df=threshold_cloud_df,
        threshold_summary_df=threshold_summary_df,
        output_dir=output_dir,
        output_name=output_name,
        config_payload=config_payload,
        developer_email=developer_email,
        cartoon_image_format=cartoon_image_format,
        lookup_timeout=lookup_timeout,
        html_neighbor_limit=html_neighbor_limit,
        html_cloud_limit=html_cloud_limit,
    )

    return {
        "corpus_df": corpus_embedding_bundle["sequence_df"],
        "query_df": notebook_query_df,
        "corpus_embedding_bundle": corpus_embedding_bundle,
        "query_embedding_bundle": query_embedding_bundle,
        "all_vs_all_artifacts": all_vs_all_artifacts,
        "specific_vs_all_results_df": specific_vs_all_results_df,
        "specific_vs_all_summary_df": specific_vs_all_summary_df,
        "threshold_cloud_df": threshold_cloud_df,
        "threshold_summary_df": threshold_summary_df,
        "saved_paths": saved_paths,
        "config_payload": config_payload,
    }
