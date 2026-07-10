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
