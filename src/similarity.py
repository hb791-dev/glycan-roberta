"""Utilities for embedding glycans and analyzing saved MLM checkpoints.

This module keeps the reusable mechanics out of notebooks so analysis notebooks
can focus on configuration, interpretation, and display.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from html import escape
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.glycan_cartoons import (
    build_cartoon_manifest,
    cache_cartoon_images,
    cartoon_lookup_from_manifest,
    format_glycan_sequence_block,
)

if TYPE_CHECKING:
    import pandas as pd


VARIANT_SET_ORDER = ("linkage", "monosaccharide", "branch_terminal")


def resolve_device(device: str | None = None) -> torch.device:
    """Return the requested runtime device, defaulting to CUDA when available."""
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_similarity_artifacts(model_dir: str, device: str | None = None):
    """Load a saved tokenizer and MLM checkpoint for embedding comparisons."""
    runtime_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_dir).to(runtime_device)
    model.eval()
    return tokenizer, model, runtime_device


def _get_encoder(model):
    """Return the base encoder beneath a masked-language-model head."""
    if hasattr(model, "base_model"):
        return model.base_model

    raise AttributeError("Model does not expose a base encoder for embedding extraction.")


def _effective_max_length(tokenizer, max_length: int | None) -> int | None:
    """Avoid passing unusably large tokenizer sentinel lengths into truncation."""
    if max_length is not None:
        return max_length

    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    if tokenizer_max_length is None or tokenizer_max_length > 100_000:
        return None

    return int(tokenizer_max_length)


def tokenize_sequence(sequence: str, tokenizer) -> list[str]:
    """Return the tokenized view of one glycan sequence."""
    return tokenizer.tokenize(sequence)


def collect_preview_sequences(
    sequence_pairs: Sequence[dict],
    matrix_sequences: Sequence[str],
) -> list[str]:
    """Return each unique sequence once in first-seen order.

    The notebook compares some sequences directly as named pairs and may also
    include a wider panel for the similarity matrix. This helper builds a single
    preview list so the tokenization table covers every sequence that appears in
    either view without duplicating rows.
    """
    preview_sequences: list[str] = []
    seen_sequences: set[str] = set()

    for pair in sequence_pairs:
        for key in ("seq1", "seq2"):
            sequence = pair[key]
            if sequence not in seen_sequences:
                preview_sequences.append(sequence)
                seen_sequences.add(sequence)

    for sequence in matrix_sequences:
        if sequence not in seen_sequences:
            preview_sequences.append(sequence)
            seen_sequences.add(sequence)

    return preview_sequences


def build_variant_preview_sequences(variant_records: Sequence[dict]) -> list[str]:
    """Return each anchor/variant sequence once in first-seen order."""
    preview_sequences: list[str] = []
    seen_sequences: set[str] = set()

    for record in variant_records:
        for key in ("anchor_sequence", "variant_sequence"):
            sequence = str(record[key])
            if sequence not in seen_sequences:
                preview_sequences.append(sequence)
                seen_sequences.add(sequence)

    return preview_sequences


def build_tokenization_preview(sequences: Sequence[str], tokenizer) -> "pd.DataFrame":
    """Return token text previews for a list of glycan sequences."""
    import pandas as pd

    preview_rows = []
    for sequence in sequences:
        tokens = tokenize_sequence(sequence, tokenizer)
        preview_rows.append(
            {
                "sequence": sequence,
                "tokens": " | ".join(tokens),
                "token_count": len(tokens),
            }
        )

    return pd.DataFrame(preview_rows)


@torch.no_grad()
def embed_sequences(
    sequences: Sequence[str],
    tokenizer,
    model,
    device: str | torch.device | None = None,
    max_length: int | None = None,
    batch_size: int = 32,
) -> torch.Tensor:
    """Mean-pool final hidden states over non-special, non-padding tokens."""
    if not sequences:
        raise ValueError("At least one sequence is required for embedding.")

    runtime_device = resolve_device(str(device)) if device is not None else next(model.parameters()).device
    encoder = _get_encoder(model)
    use_max_length = _effective_max_length(tokenizer, max_length)
    embedding_batches = []

    for start_index in range(0, len(sequences), batch_size):
        batch_sequences = list(sequences[start_index : start_index + batch_size])
        # Tokenize one mini-batch at a time so bigger review sets do not blow up memory.
        encoded_batch = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=use_max_length,
        )
        encoded_batch = {name: tensor.to(runtime_device) for name, tensor in encoded_batch.items()}

        hidden_states = encoder(**encoded_batch).last_hidden_state
        # Start from the attention mask, then strip out special tokens so the pooled
        # embedding reflects the glycan content tokens rather than CLS/SEP/PAD noise.
        content_mask = encoded_batch["attention_mask"].bool()
        for token_id in tokenizer.all_special_ids:
            content_mask &= encoded_batch["input_ids"] != token_id

        content_mask = content_mask.unsqueeze(-1).type_as(hidden_states)
        pooled_hidden = (hidden_states * content_mask).sum(dim=1)
        token_counts = content_mask.sum(dim=1).clamp(min=1e-9)
        embedding_batches.append((pooled_hidden / token_counts).cpu())

    return torch.cat(embedding_batches, dim=0)


def compare_sequence_pair(
    seq1: str,
    seq2: str,
    tokenizer,
    model,
    device: str | torch.device | None = None,
    max_length: int | None = None,
) -> dict:
    """Embed two sequences and return cosine similarity plus token previews."""
    embeddings = embed_sequences(
        [seq1, seq2],
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=2,
    )
    cosine_similarity = torch.nn.functional.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()

    return {
        "seq1": seq1,
        "seq2": seq2,
        "cosine_similarity": cosine_similarity,
        "seq1_tokens": tokenize_sequence(seq1, tokenizer),
        "seq2_tokens": tokenize_sequence(seq2, tokenizer),
    }


def compare_sequence_pairs(
    sequence_pairs: Sequence[dict],
    tokenizer,
    model,
    device: str | torch.device | None = None,
    max_length: int | None = None,
) -> "pd.DataFrame":
    """Return one comparison row per named pair of glycan sequences."""
    import pandas as pd

    comparison_rows = []

    for pair in sequence_pairs:
        result = compare_sequence_pair(
            pair["seq1"],
            pair["seq2"],
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=max_length,
        )
        row = {
            "pair_name": pair["pair_name"],
            "seq1": result["seq1"],
            "seq2": result["seq2"],
            "cosine_similarity": result["cosine_similarity"],
            "seq1_tokens": " | ".join(result["seq1_tokens"]),
            "seq2_tokens": " | ".join(result["seq2_tokens"]),
        }
        if "group_name" in pair:
            row["group_name"] = pair["group_name"]
        comparison_rows.append(row)

    return pd.DataFrame(comparison_rows)


def similarity_matrix(
    sequences: Sequence[str],
    tokenizer,
    model,
    device: str | torch.device | None = None,
    max_length: int | None = None,
    batch_size: int = 32,
) -> torch.Tensor:
    """Return the full pairwise cosine-similarity matrix for a sequence list."""
    embeddings = embed_sequences(
        sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    # Normalize first so the matrix multiply is exactly cosine similarity.
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return normalized_embeddings @ normalized_embeddings.T


def similarity_matrix_dataframe(
    sequences: Sequence[str],
    tokenizer,
    model,
    device: str | torch.device | None = None,
    max_length: int | None = None,
    batch_size: int = 32,
) -> "pd.DataFrame":
    """Return the pairwise cosine-similarity matrix as a labeled dataframe."""
    import pandas as pd

    similarity_tensor = similarity_matrix(
        sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    return pd.DataFrame(similarity_tensor.numpy(), index=sequences, columns=sequences)


def _normalize_variant_set_name(variant_set: str) -> str:
    return str(variant_set).strip().lower().replace("-", "_").replace("/", "_")


def _variant_set_sort_key(variant_set: str) -> tuple[int, str]:
    normalized = _normalize_variant_set_name(variant_set)
    if normalized in VARIANT_SET_ORDER:
        return VARIANT_SET_ORDER.index(normalized), normalized
    return len(VARIANT_SET_ORDER), normalized


def _sort_variant_results(df: "pd.DataFrame") -> "pd.DataFrame":
    sortable = df.copy()
    sortable["_variant_set_order"] = sortable["variant_set"].map(lambda value: _variant_set_sort_key(value)[0])
    sortable["_variant_set_name"] = sortable["variant_set"].map(lambda value: _variant_set_sort_key(value)[1])
    # Sort by anchor first, then keep the linkage / monosaccharide / branch-terminal
    # sections in a stable human-readable order for notebook tables and HTML output.
    sortable = sortable.sort_values(
        [
            "anchor_id",
            "rank_within_anchor",
            "_variant_set_order",
            "_variant_set_name",
            "variant_id",
        ],
        kind="stable",
    )
    return sortable.drop(columns=["_variant_set_order", "_variant_set_name"]).reset_index(drop=True)


def validate_similarity_inputs(
    model_dir,
    sequence_pairs: Sequence[dict],
    matrix_sequences: Sequence[str],
    output_dir=None,
) -> None:
    """Validate the minimum inputs required for one similarity-analysis run."""
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    required_model_files = ["config.json"]
    missing_files = [filename for filename in required_model_files if not (model_path / filename).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Model directory is missing required files: {missing_files}"
        )

    if not sequence_pairs:
        raise ValueError("Add at least one sequence pair before running the analysis.")

    if not matrix_sequences:
        raise ValueError("Add at least one matrix sequence before running the analysis.")

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)


def validate_variant_similarity_inputs(
    model_dir,
    variant_records: Sequence[dict],
    output_dir=None,
) -> None:
    """Validate the minimum inputs required for anchor-to-variant analysis."""
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    required_model_files = ["config.json"]
    missing_files = [filename for filename in required_model_files if not (model_path / filename).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Model directory is missing required files: {missing_files}"
        )

    if not variant_records:
        raise ValueError("Add at least one variant record before running the analysis.")

    required_record_fields = (
        "anchor_id",
        "anchor_sequence",
        "variant_set",
        "variant_id",
        "edit_type",
        "edit_description",
        "variant_sequence",
    )
    for record_number, record in enumerate(variant_records, start=1):
        missing_fields = [field for field in required_record_fields if field not in record]
        if missing_fields:
            raise ValueError(
                f"Variant record {record_number} is missing required fields: {missing_fields}"
            )
        for field in required_record_fields:
            if str(record[field]).strip() == "":
                raise ValueError(f"Variant record {record_number} has a blank value for {field!r}.")

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)


def compare_anchor_variants(
    variant_records: Sequence[dict],
    tokenizer,
    model,
    device: str | torch.device | None = None,
    max_length: int | None = None,
    batch_size: int = 32,
) -> "pd.DataFrame":
    """Compare each anchor glycan against its configured variants."""
    import pandas as pd

    # Embed each unique sequence once, then reuse those vectors for every anchor-variant
    # comparison instead of re-embedding duplicates.
    preview_sequences = build_variant_preview_sequences(variant_records)
    embeddings = embed_sequences(
        preview_sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    sequence_to_index = {sequence: index for index, sequence in enumerate(preview_sequences)}

    comparison_rows = []
    for record in variant_records:
        anchor_sequence = str(record["anchor_sequence"])
        variant_sequence = str(record["variant_sequence"])
        anchor_index = sequence_to_index[anchor_sequence]
        variant_index = sequence_to_index[variant_sequence]
        cosine_similarity = torch.nn.functional.cosine_similarity(
            normalized_embeddings[anchor_index : anchor_index + 1],
            normalized_embeddings[variant_index : variant_index + 1],
        ).item()

        comparison_rows.append(
            {
                **record,
                "anchor_sequence": anchor_sequence,
                "variant_sequence": variant_sequence,
                "cosine_similarity": cosine_similarity,
            }
        )

    variant_results_df = pd.DataFrame(comparison_rows)
    variant_results_df["_variant_set_order"] = variant_results_df["variant_set"].map(
        lambda value: _variant_set_sort_key(value)[0]
    )
    variant_results_df["_variant_set_name"] = variant_results_df["variant_set"].map(
        lambda value: _variant_set_sort_key(value)[1]
    )
    # Force one explicit 1-9 ordering per anchor by sorting on similarity first and
    # then using stable tie-breakers when similarities are extremely close or equal.
    variant_results_df = variant_results_df.sort_values(
        [
            "anchor_id",
            "cosine_similarity",
            "_variant_set_order",
            "_variant_set_name",
            "variant_id",
        ],
        ascending=[True, False, True, True, True],
        kind="stable",
    )
    variant_results_df["rank_within_anchor"] = variant_results_df.groupby("anchor_id").cumcount() + 1
    variant_results_df = variant_results_df.drop(columns=["_variant_set_order", "_variant_set_name"])
    return _sort_variant_results(variant_results_df)


def build_anchor_similarity_matrices(
    variant_records: Sequence[dict],
    tokenizer,
    model,
    device: str | torch.device | None = None,
    max_length: int | None = None,
    batch_size: int = 32,
) -> dict[str, "pd.DataFrame"]:
    """Return one local similarity matrix per anchor group."""
    anchor_to_sequences: dict[str, list[str]] = {}
    for record in variant_records:
        anchor_id = str(record["anchor_id"])
        anchor_sequences = anchor_to_sequences.setdefault(anchor_id, [])
        # Preserve first-seen order so the saved matrix lines up with the anchor and
        # variant order used elsewhere in the report.
        for sequence in (str(record["anchor_sequence"]), str(record["variant_sequence"])):
            if sequence not in anchor_sequences:
                anchor_sequences.append(sequence)

    return {
        anchor_id: similarity_matrix_dataframe(
            sequences=sequences,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
        )
        for anchor_id, sequences in anchor_to_sequences.items()
    }


def plot_similarity_heatmap(similarity_df, output_path, title: str) -> None:
    """Display the similarity heatmap inline and save it to disk."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(similarity_df.values, cmap="viridis", vmin=-1.0, vmax=1.0)
    fig.colorbar(image, ax=ax, label="Cosine similarity")
    ax.set_xticks(range(len(similarity_df.columns)))
    ax.set_xticklabels(similarity_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(similarity_df.index)))
    ax.set_yticklabels(similarity_df.index)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.show()
    plt.close(fig)


def plot_variant_similarity_histogram(variant_rows_df, output_path, title: str) -> Path:
    """Save one histogram of cosine similarities for a variant collection."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    similarity_values = variant_rows_df["cosine_similarity"].tolist()
    bin_count = min(10, max(5, len(similarity_values)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        similarity_values,
        bins=bin_count,
        range=(-1.0, 1.0),
        color="#4c78a8",
        edgecolor="white",
        linewidth=1.0,
    )
    # The dashed line makes it easier to eyeball whether one anchor's variants are
    # clustering high or low overall.
    ax.axvline(sum(similarity_values) / len(similarity_values), color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Variant count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def build_variant_summary_tables(variant_results_df) -> "pd.DataFrame":
    """Return one summary table with per-set and all-variant anchor summaries."""
    import pandas as pd

    # One compact table for "how spread out are the scores?" within each anchor/set.
    set_summary_df = (
        variant_results_df.groupby(["anchor_id", "variant_set"], sort=False)["cosine_similarity"]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
        .rename(columns={"std": "std_dev"})
    )
    set_summary_df["std_dev"] = set_summary_df["std_dev"].fillna(0.0)

    # Add one anchor-level rollup so each anchor also has a single summary across all 9 variants.
    overall_summary_df = (
        variant_results_df.groupby("anchor_id", sort=False)["cosine_similarity"]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
        .rename(columns={"std": "std_dev"})
    )
    overall_summary_df["std_dev"] = overall_summary_df["std_dev"].fillna(0.0)
    overall_summary_df.insert(1, "variant_set", "all_variants")

    distribution_summary_df = pd.concat([overall_summary_df, set_summary_df], ignore_index=True)
    distribution_summary_df["_variant_set_order"] = distribution_summary_df["variant_set"].map(
        lambda value: -1 if str(value) == "all_variants" else _variant_set_sort_key(str(value))[0]
    )
    distribution_summary_df = distribution_summary_df.sort_values(
        ["anchor_id", "_variant_set_order", "variant_set"],
        kind="stable",
    ).drop(columns=["_variant_set_order"]).reset_index(drop=True)
    return distribution_summary_df


def _variant_set_heading(variant_set: str) -> str:
    return str(variant_set).replace("_", " ").replace("-", " ").title()


def _humanize_label(label: str) -> str:
    return str(label).replace("_", " ").replace("-", " ").title()


def _image_path_to_data_uri(image_path) -> str:
    """Return one local image file as a base64 data URI for standalone HTML."""
    image_path = Path(image_path)
    suffix = image_path.suffix.lower()
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
    }.get(suffix, "application/octet-stream")
    encoded_bytes = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_bytes}"


def render_anchor_similarity_html(
    anchor_id: str,
    anchor_rows_df,
    cartoon_lookup: dict[str, dict[str, str]],
    output_path,
    output_name: str,
    histogram_image_path: str | None = None,
) -> Path:
    """Render one anchor-focused HTML page."""
    anchor_sequence = str(anchor_rows_df["anchor_sequence"].iloc[0])
    anchor_cartoon = cartoon_lookup.get(anchor_sequence)
    analysis_media_html = ""
    if histogram_image_path:
        image_sections = []
        if histogram_image_path:
            image_sections.append(
                "<div class='analysis-card'>"
                "<h3>Similarity Distribution Across All 9 Variants</h3>"
                f"<img src='{escape(histogram_image_path, quote=True)}' alt='Histogram for {escape(anchor_id)}'>"
                "</div>"
            )
        analysis_media_html = (
            "<div class='analysis-panel'>"
            + "".join(image_sections)
            + "</div>"
        )

    ranked_row_html_parts = []
    sorted_rows = anchor_rows_df.sort_values(
        ["rank_within_anchor", "variant_id"],
        kind="stable",
    )
    for row in sorted_rows.itertuples(index=False):
        variant_cartoon = cartoon_lookup.get(row.variant_sequence)
        ranked_row_html_parts.append(
            "<tr>"
            f"<td><span class='rank-chip'>{int(row.rank_within_anchor)}</span></td>"
            f"<td>{escape(str(row.variant_id))}</td>"
            f"<td>{escape(_humanize_label(str(row.variant_set)))}</td>"
            f"<td>{escape(_humanize_label(str(row.edit_type)))}</td>"
            f"<td>{escape(str(row.edit_description))}</td>"
            f"<td>{row.cosine_similarity:.3f}</td>"
            f"<td>{format_glycan_sequence_block(str(row.variant_sequence), variant_cartoon)}</td>"
            "</tr>"
        )
    ranked_table_html = (
        "<div class='ranking-panel'>"
        "<h2>All 9 Variants In Rank Order</h2>"
        "<p class='section-note'>This table is sorted by overall rank across the full anchor set, not by edit family.</p>"
        "<table>"
        "<thead><tr>"
        "<th>Overall Rank</th>"
        "<th>Variant ID</th>"
        "<th>Edit Family</th>"
        "<th>Edit Type</th>"
        "<th>Edit Description</th>"
        "<th>Cosine Similarity</th>"
        "<th>Variant</th>"
        "</tr></thead>"
        f"<tbody>{''.join(ranked_row_html_parts)}</tbody>"
        "</table>"
        "</div>"
    )

    section_html_parts = []
    grouped = anchor_rows_df.groupby("variant_set", sort=False)
    for variant_set, set_df in grouped:
        row_html_parts = []
        set_sorted_rows = set_df.sort_values(
            ["rank_within_anchor", "variant_id"],
            kind="stable",
        )
        for row in set_sorted_rows.itertuples(index=False):
            variant_cartoon = cartoon_lookup.get(row.variant_sequence)
            row_html_parts.append(
                "<tr>"
                f"<td>{escape(str(row.variant_id))}</td>"
                f"<td><span class='rank-chip'>{int(row.rank_within_anchor)}</span></td>"
                f"<td>{escape(_humanize_label(str(row.edit_type)))}</td>"
                f"<td>{escape(str(row.edit_description))}</td>"
                f"<td>{row.cosine_similarity:.3f}</td>"
                f"<td>{format_glycan_sequence_block(str(row.variant_sequence), variant_cartoon)}</td>"
                "</tr>"
            )

        section_html_parts.append(
            "<div class='set-panel'>"
            f"<h2>{escape(_variant_set_heading(variant_set))}</h2>"
            "<p class='section-note'>The overall-rank column still refers to the full 1-9 anchor ordering.</p>"
            "<table>"
            "<thead><tr>"
            "<th>Variant ID</th>"
            "<th>Overall Rank</th>"
            "<th>Edit Type</th>"
            "<th>Edit Description</th>"
            "<th>Cosine Similarity</th>"
            "<th>Variant</th>"
            "</tr></thead>"
            f"<tbody>{''.join(row_html_parts)}</tbody>"
            "</table>"
            "</div>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(output_name)} - {escape(anchor_id)}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px auto;
      max-width: 1180px;
      color: #222;
      line-height: 1.4;
      padding: 0 18px 36px;
    }}
    h1, h2 {{
      color: #111;
    }}
    .anchor-panel {{
      background: #f7f7f7;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 18px;
      margin-bottom: 28px;
    }}
    .analysis-panel {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      margin-bottom: 28px;
    }}
    .analysis-card {{
      background: #fafafa;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 16px;
    }}
    .ranking-panel {{
      margin-bottom: 28px;
    }}
    .set-panel {{
      margin-bottom: 28px;
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
      margin-bottom: 28px;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 10px 12px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #f0f0f0;
    }}
    .section-note {{
      margin: 0 0 12px;
      color: #555;
    }}
    .rank-chip {{
      display: inline-block;
      min-width: 28px;
      text-align: center;
      font-weight: 700;
      border-radius: 999px;
      padding: 4px 8px;
      background: #1f2937;
      color: white;
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
  <h1>{escape(output_name)} - Anchor {escape(anchor_id)}</h1>
  <div class="anchor-panel">
    <h2>Anchor Sequence</h2>
    {format_glycan_sequence_block(anchor_sequence, anchor_cartoon)}
    <p><strong>Variants in this group:</strong> {len(anchor_rows_df)}</p>
  </div>
  {analysis_media_html}
  {ranked_table_html}
  {''.join(section_html_parts)}
</body>
</html>
"""

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def render_variant_index_html(
    variant_results_df,
    cartoon_lookup: dict[str, dict[str, str]],
    anchor_html_paths: dict[str, Path],
    output_path,
    output_name: str,
    overall_histogram_path: str | None = None,
) -> Path:
    """Render one top-level HTML index across all anchors."""
    summary_rows = []
    for anchor_id, anchor_df in variant_results_df.groupby("anchor_id", sort=False):
        anchor_sequence = str(anchor_df["anchor_sequence"].iloc[0])
        anchor_cartoon = cartoon_lookup.get(anchor_sequence)
        report_path = anchor_html_paths[anchor_id]
        summary_rows.append(
            "<tr>"
            f"<td>{escape(anchor_id)}</td>"
            f"<td>{format_glycan_sequence_block(anchor_sequence, anchor_cartoon)}</td>"
            f"<td>{len(anchor_df)}</td>"
            f"<td>{anchor_df['cosine_similarity'].max():.3f}</td>"
            f"<td>{anchor_df['cosine_similarity'].min():.3f}</td>"
            f"<td><a href='{escape(report_path.name, quote=True)}'>{escape(report_path.name)}</a></td>"
            "</tr>"
        )

    overall_histogram_html = ""
    if overall_histogram_path:
        overall_histogram_html = (
            "<div class='analysis-card'>"
            "<h2>All Variant Similarities</h2>"
            f"<img src='{escape(overall_histogram_path, quote=True)}' alt='Overall similarity histogram'>"
            "</div>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(output_name)} - Variant Similarity Index</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px auto;
      max-width: 1180px;
      color: #222;
      line-height: 1.4;
      padding: 0 18px 36px;
    }}
    .analysis-card {{
      background: #fafafa;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 16px;
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
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #f0f0f0;
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
  <h1>{escape(output_name)} - Variant Similarity Index</h1>
  <p>This index links to one HTML page per anchor glycan.</p>
  {overall_histogram_html}
  <table>
    <thead>
      <tr>
        <th>Anchor ID</th>
        <th>Anchor</th>
        <th>Variant Count</th>
        <th>Max Similarity</th>
        <th>Min Similarity</th>
        <th>Anchor Report</th>
      </tr>
    </thead>
    <tbody>
      {''.join(summary_rows)}
    </tbody>
  </table>
</body>
</html>
"""

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def save_similarity_outputs(
    pair_results_df,
    tokenization_preview_df,
    similarity_df,
    output_dir,
    output_name: str,
    config_payload: dict,
) -> dict:
    """Write pairwise similarity outputs to disk and return the saved file paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pair_results_path = output_path / "similarity_pairs.csv"
    tokenization_preview_path = output_path / "tokenization_preview.csv"
    similarity_matrix_path = output_path / "similarity_matrix.csv"
    heatmap_path = output_path / "similarity_heatmap.png"
    config_path = output_path / "similarity_config.json"

    pair_results_df.to_csv(pair_results_path, index=False)
    tokenization_preview_df.to_csv(tokenization_preview_path, index=False)
    similarity_df.to_csv(similarity_matrix_path)
    plot_similarity_heatmap(similarity_df, heatmap_path, f"{output_name} similarity heatmap")

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config_payload, file, indent=2)

    return {
        "pair_results_path": pair_results_path,
        "tokenization_preview_path": tokenization_preview_path,
        "similarity_matrix_path": similarity_matrix_path,
        "heatmap_path": heatmap_path,
        "config_path": config_path,
    }


def save_variant_similarity_outputs(
    variant_results_df,
    tokenization_preview_df,
    cartoon_manifest_df,
    anchor_matrices: dict[str, "pd.DataFrame"],
    distribution_summary_df,
    output_dir,
    output_name: str,
    config_payload: dict,
) -> dict:
    """Write anchor-to-variant outputs, including HTML reports, to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    variant_results_path = output_path / "variant_similarity_results.csv"
    tokenization_preview_path = output_path / "variant_tokenization_preview.csv"
    cartoon_manifest_path = output_path / "variant_cartoon_manifest.csv"
    distribution_summary_path = output_path / "variant_distribution_summary.csv"
    config_path = output_path / "variant_similarity_config.json"
    html_dir = output_path / "html"
    matrix_dir = output_path / "anchor_matrices"
    histogram_dir = output_path / "histograms"
    cartoon_dir = output_path / "cartoons"
    html_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    histogram_dir.mkdir(parents=True, exist_ok=True)
    cartoon_dir.mkdir(parents=True, exist_ok=True)

    # Save remote cartoons locally before the HTML is rendered. This is especially
    # important for on-demand Glymage task URLs, which may expire after the notebook
    # run even though the analysis results themselves are perfectly valid.
    cartoon_manifest_df = cache_cartoon_images(
        cartoon_manifest_df=cartoon_manifest_df,
        asset_dir=cartoon_dir,
        image_format=str(config_payload.get("cartoon_image_format", "svg")),
        download_timeout=int(config_payload.get("lookup_timeout", 60)),
    )

    variant_results_df.to_csv(variant_results_path, index=False)
    tokenization_preview_df.to_csv(tokenization_preview_path, index=False)
    cartoon_manifest_df.to_csv(cartoon_manifest_path, index=False)
    distribution_summary_df.to_csv(distribution_summary_path, index=False)

    anchor_matrix_paths: dict[str, Path] = {}
    for anchor_id, matrix_df in anchor_matrices.items():
        matrix_path = matrix_dir / f"{anchor_id}_similarity_matrix.csv"
        matrix_df.to_csv(matrix_path)
        anchor_matrix_paths[anchor_id] = matrix_path

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config_payload, file, indent=2)

    cartoon_lookup = cartoon_lookup_from_manifest(cartoon_manifest_df)
    # Save one overall distribution first so the index page has a quick "big picture" view.
    overall_histogram_path = plot_variant_similarity_histogram(
        variant_results_df,
        histogram_dir / "all_variants_similarity_histogram.png",
        f"{output_name} all-variant similarity distribution",
    )
    overall_histogram_data_uri = _image_path_to_data_uri(overall_histogram_path)

    anchor_html_paths: dict[str, Path] = {}
    anchor_histogram_paths: dict[str, Path] = {}
    for anchor_id, anchor_df in variant_results_df.groupby("anchor_id", sort=False):
        # Each anchor gets one histogram across all 9 variants in that anchor group.
        anchor_histogram_paths[anchor_id] = plot_variant_similarity_histogram(
            anchor_df,
            histogram_dir / f"{anchor_id}_similarity_histogram.png",
            f"{output_name} {anchor_id} all-variant similarity distribution",
        )
        anchor_histogram_data_uri = _image_path_to_data_uri(anchor_histogram_paths[anchor_id])
        anchor_html_paths[anchor_id] = render_anchor_similarity_html(
            anchor_id=str(anchor_id),
            anchor_rows_df=anchor_df,
            cartoon_lookup=cartoon_lookup,
            output_path=html_dir / f"{anchor_id}_variant_similarity.html",
            output_name=output_name,
            histogram_image_path=anchor_histogram_data_uri,
        )

    index_html_path = render_variant_index_html(
        variant_results_df=variant_results_df,
        cartoon_lookup=cartoon_lookup,
        anchor_html_paths=anchor_html_paths,
        output_path=html_dir / "index.html",
        output_name=output_name,
        overall_histogram_path=overall_histogram_data_uri,
    )

    return {
        "variant_results_path": variant_results_path,
        "tokenization_preview_path": tokenization_preview_path,
        "cartoon_manifest_path": cartoon_manifest_path,
        "distribution_summary_path": distribution_summary_path,
        "config_path": config_path,
        "html_dir": html_dir,
        "histogram_dir": histogram_dir,
        "cartoon_dir": cartoon_dir,
        "overall_histogram_path": overall_histogram_path,
        "index_html_path": index_html_path,
        "anchor_html_paths": anchor_html_paths,
        "anchor_histogram_paths": anchor_histogram_paths,
        "anchor_matrix_paths": anchor_matrix_paths,
    }


def run_similarity_analysis(
    tokenizer,
    model,
    sequence_pairs: Sequence[dict],
    matrix_sequences: Sequence[str],
    output_dir,
    output_name: str,
    device: str | torch.device | None = None,
    max_length: int | None = None,
    batch_size: int = 32,
    model_dir=None,
) -> dict:
    """Run one end-to-end curated pairwise similarity analysis."""
    preview_sequences = collect_preview_sequences(sequence_pairs, matrix_sequences)

    pair_results_df = compare_sequence_pairs(
        sequence_pairs,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
    )
    tokenization_preview_df = build_tokenization_preview(preview_sequences, tokenizer)
    similarity_df = similarity_matrix_dataframe(
        matrix_sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    config_payload = {
        "analysis_type": "curated_pairs",
        "model_dir": str(model_dir) if model_dir is not None else "",
        "output_dir": str(output_dir),
        "sequence_pairs": list(sequence_pairs),
        "matrix_sequences": list(matrix_sequences),
        "max_length": max_length,
        "batch_size": batch_size,
    }
    saved_paths = save_similarity_outputs(
        pair_results_df=pair_results_df,
        tokenization_preview_df=tokenization_preview_df,
        similarity_df=similarity_df,
        output_dir=output_dir,
        output_name=output_name,
        config_payload=config_payload,
    )

    return {
        "pair_results_df": pair_results_df,
        "tokenization_preview_df": tokenization_preview_df,
        "similarity_df": similarity_df,
        "preview_sequences": preview_sequences,
        "saved_paths": saved_paths,
        "config_payload": config_payload,
    }


def run_variant_similarity_analysis(
    tokenizer,
    model,
    variant_records: Sequence[dict],
    output_dir,
    output_name: str,
    developer_email: str | None = None,
    cartoon_image_format: str = "svg",
    cartoon_display: str = "compact",
    lookup_timeout: int = 60,
    device: str | torch.device | None = None,
    max_length: int | None = None,
    batch_size: int = 32,
    model_dir=None,
) -> dict:
    """Run anchor-to-variant similarity analysis and save HTML reports."""
    preview_sequences = build_variant_preview_sequences(variant_records)
    variant_results_df = compare_anchor_variants(
        variant_records=variant_records,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    tokenization_preview_df = build_tokenization_preview(preview_sequences, tokenizer)
    cartoon_manifest_df = build_cartoon_manifest(
        sequences=preview_sequences,
        developer_email=developer_email,
        image_format=cartoon_image_format,
        display=cartoon_display,
        lookup_timeout=lookup_timeout,
    )
    # This summary table combines per-set stats with one "all 9 variants" row per anchor.
    distribution_summary_df = build_variant_summary_tables(variant_results_df)
    anchor_matrices = build_anchor_similarity_matrices(
        variant_records=variant_records,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    config_payload = {
        "analysis_type": "anchor_variant_sets",
        "model_dir": str(model_dir) if model_dir is not None else "",
        "output_dir": str(output_dir),
        "variant_records": list(variant_records),
        "developer_email": developer_email or "",
        "cartoon_image_format": cartoon_image_format,
        "cartoon_display": cartoon_display,
        "lookup_timeout": lookup_timeout,
        "max_length": max_length,
        "batch_size": batch_size,
    }
    saved_paths = save_variant_similarity_outputs(
        variant_results_df=variant_results_df,
        tokenization_preview_df=tokenization_preview_df,
        cartoon_manifest_df=cartoon_manifest_df,
        anchor_matrices=anchor_matrices,
        distribution_summary_df=distribution_summary_df,
        output_dir=output_dir,
        output_name=output_name,
        config_payload=config_payload,
    )

    return {
        "variant_results_df": variant_results_df,
        "tokenization_preview_df": tokenization_preview_df,
        "cartoon_manifest_df": cartoon_manifest_df,
        "distribution_summary_df": distribution_summary_df,
        "anchor_matrices": anchor_matrices,
        "preview_sequences": preview_sequences,
        "saved_paths": saved_paths,
        "config_payload": config_payload,
    }


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
        Path(output_dir).mkdir(parents=True, exist_ok=True)


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
    """Build matrix, summary, and top-neighbor views for the full test-set corpus."""
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


def compare_queries_to_corpus(
    query_df,
    query_normalized_embeddings,
    corpus_df,
    corpus_normalized_embeddings,
    accession_col: str = "accession",
    sequence_col: str = "sequence",
) -> "pd.DataFrame":
    """Compare one or more selected glycans against an entire test-set corpus."""
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
    """Return threshold-cloud membership rows plus one summary row per threshold."""
    import pandas as pd

    threshold_membership_rows = []
    threshold_summary_rows = []
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
    """Collect the subset of sequences that should get cartoons in the HTML reports."""
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
    cartoon_manifest_df = build_cartoon_manifest(
        sequences=html_sequences,
        developer_email=developer_email,
        image_format=cartoon_image_format,
        display="compact",
        lookup_timeout=lookup_timeout,
    )
    cartoon_manifest_df = cache_cartoon_images(
        cartoon_manifest_df=cartoon_manifest_df,
        asset_dir=cartoon_dir,
        image_format=cartoon_image_format,
        download_timeout=lookup_timeout,
    )
    cartoon_manifest_path = output_path / "scaleup_cartoon_manifest.csv"
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
