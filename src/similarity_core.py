"""Core similarity helpers for glycan embedding and curated pair analysis.

This module holds the lower-level pieces that other similarity workflows build on:
loading checkpoints, embedding sequences, tokenization previews, matrix helpers,
and the original curated pair analysis.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Runtime and model-loading helpers
# ---------------------------------------------------------------------------

def resolve_device(device: str | None = None) -> torch.device:
    """Return the requested runtime device, defaulting to CUDA when available."""
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_similarity_artifacts(model_dir: str, device: str | None = None):
    """Load a saved tokenizer and encoder-compatible checkpoint.

    Similarity work only needs the shared encoder that produces token-level
    hidden states. Loading with ``AutoModel`` keeps this path compatible with
    both saved MLM checkpoints and saved classification checkpoints.
    """
    runtime_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(runtime_device)
    model.eval()
    return tokenizer, model, runtime_device


def _get_encoder(model):
    """Return the encoder that produces token embeddings.

    Some checkpoints are reloaded as task-specific wrappers and some are loaded
    directly as the base encoder. In both cases, similarity work only needs the
    object that exposes ``last_hidden_state`` for token-level embeddings.
    """
    if hasattr(model, "base_model") and model.base_model is not None:
        return model.base_model

    if hasattr(model, "encoder") or hasattr(model, "embeddings"):
        return model

    raise AttributeError("Model does not expose a base encoder for embedding extraction.")


def _effective_max_length(tokenizer, max_length: int | None) -> int | None:
    """Return a safe max length for tokenizer truncation.

    Some Hugging Face tokenizers use very large sentinel values to mean
    "effectively unbounded". Passing those values into truncation is not useful,
    so this helper converts them back to `None`.
    """
    if max_length is not None:
        return max_length

    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    if tokenizer_max_length is None or tokenizer_max_length > 100_000:
        return None

    return int(tokenizer_max_length)


# ---------------------------------------------------------------------------
# Small sequence and HTML helpers
# ---------------------------------------------------------------------------

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
    """Return each anchor/variant sequence once in first-seen order.

    The variant workflow reuses the same anchor sequence across several edited
    versions. This helper deduplicates those repeats so downstream tokenization
    previews and embedding calls stay compact.
    """
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
    """Return a simple dataframe showing how the tokenizer splits each sequence."""
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


def _build_content_mask(encoded_batch, tokenizer) -> torch.Tensor:
    """Return a boolean mask that keeps only real glycan content tokens.

    The tokenizer adds padding and model-specific special tokens such as CLS/SEP.
    Those tokens are useful for the model internals, but they should not count
    toward the pooled embedding that we use for similarity.
    """
    content_mask = encoded_batch["attention_mask"].bool()
    for token_id in tokenizer.all_special_ids:
        content_mask &= encoded_batch["input_ids"] != token_id
    return content_mask


def normalize_pooling_strategy(pooling_strategy: str | None = None) -> str:
    """Return one validated pooling-strategy label."""
    normalized = str(pooling_strategy or "mean").strip().lower()
    if normalized not in {"cls", "mean", "max"}:
        raise ValueError(
            f"Unsupported pooling_strategy {pooling_strategy!r}. Expected 'cls', 'mean', or 'max'."
        )
    return normalized


def _cls_pool_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    """Return the hidden state at the first token position for each sequence."""
    return hidden_states[:, 0, :]


def _mean_pool_hidden_states(hidden_states: torch.Tensor, content_mask: torch.Tensor) -> torch.Tensor:
    """Average token vectors across the kept token positions for one batch."""
    expanded_mask = content_mask.unsqueeze(-1).type_as(hidden_states)
    pooled_hidden = (hidden_states * expanded_mask).sum(dim=1)
    token_counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
    return pooled_hidden / token_counts


def _max_pool_hidden_states(hidden_states: torch.Tensor, content_mask: torch.Tensor) -> torch.Tensor:
    """Take the per-dimension maximum across kept token positions for one batch."""
    expanded_mask = content_mask.unsqueeze(-1)
    masked_hidden_states = hidden_states.masked_fill(~expanded_mask, float("-inf"))
    pooled_hidden = masked_hidden_states.max(dim=1).values
    # If a row somehow loses every token after masking, fall back to zeros instead
    # of keeping -inf values in the saved embeddings.
    empty_rows = ~content_mask.any(dim=1)
    pooled_hidden[empty_rows] = 0.0
    return pooled_hidden


# ---------------------------------------------------------------------------
# Embedding and similarity helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_sequences(
    sequences: Sequence[str],
    tokenizer,
    model,
    device: str | torch.device | None = None,
    max_length: int | None = None,
    batch_size: int = 32,
    pooling_strategy: str = "mean",
) -> torch.Tensor:
    """Embed glycan sequences with configurable pooling over real content tokens.

    The function works in mini-batches so larger similarity jobs can run without
    trying to place the entire dataset on the GPU or CPU at once.
    """
    if not sequences:
        raise ValueError("At least one sequence is required for embedding.")

    runtime_device = (
        resolve_device(str(device))
        if device is not None
        else next(model.parameters()).device
    )
    encoder = _get_encoder(model)
    use_max_length = _effective_max_length(tokenizer, max_length)
    normalized_pooling_strategy = normalize_pooling_strategy(pooling_strategy)
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
        if normalized_pooling_strategy == "cls":
            pooled_batch = _cls_pool_hidden_states(hidden_states)
        else:
            content_mask = _build_content_mask(encoded_batch, tokenizer)
            if normalized_pooling_strategy == "mean":
                pooled_batch = _mean_pool_hidden_states(hidden_states, content_mask)
            else:
                pooled_batch = _max_pool_hidden_states(hidden_states, content_mask)
        embedding_batches.append(pooled_batch.cpu())

    return torch.cat(embedding_batches, dim=0)


def compare_sequence_pair(
    seq1: str,
    seq2: str,
    tokenizer,
    model,
    device: str | torch.device | None = None,
    max_length: int | None = None,
    pooling_strategy: str = "mean",
) -> dict:
    """Embed two sequences and return cosine similarity plus token previews."""
    embeddings = embed_sequences(
        [seq1, seq2],
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=2,
        pooling_strategy=pooling_strategy,
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
    pooling_strategy: str = "mean",
) -> "pd.DataFrame":
    """Return one comparison row per named pair of glycan sequences.

    This is the compact table used in the original "curated examples" style of
    analysis where each row has a human-readable pair label.
    """
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
            pooling_strategy=pooling_strategy,
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
    pooling_strategy: str = "mean",
) -> torch.Tensor:
    """Return the full pairwise cosine-similarity matrix for a sequence list."""
    embeddings = embed_sequences(
        sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        pooling_strategy=pooling_strategy,
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
    pooling_strategy: str = "mean",
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
        pooling_strategy=pooling_strategy,
    )
    return pd.DataFrame(similarity_tensor.numpy(), index=sequences, columns=sequences)


# ---------------------------------------------------------------------------
# Curated pair analysis: validation, plots, save helpers
# ---------------------------------------------------------------------------

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
        # Create the folder early so notebook runs fail fast if the path is invalid.
        Path(output_dir).mkdir(parents=True, exist_ok=True)


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


def save_similarity_outputs(
    pair_results_df,
    tokenization_preview_df,
    similarity_df,
    output_dir,
    output_name: str,
    config_payload: dict,
) -> dict:
    """Write the curated-pair outputs to disk and return the saved file paths."""
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
    pooling_strategy: str = "mean",
    model_dir=None,
) -> dict:
    """Run the original curated-pair similarity workflow end to end.

    This path is the smallest, most notebook-friendly workflow: compare a few
    named pairs, preview how they tokenize, then build a small similarity matrix
    for a hand-picked panel of sequences.
    """
    preview_sequences = collect_preview_sequences(sequence_pairs, matrix_sequences)

    pair_results_df = compare_sequence_pairs(
        sequence_pairs,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        pooling_strategy=pooling_strategy,
    )
    tokenization_preview_df = build_tokenization_preview(preview_sequences, tokenizer)
    similarity_df = similarity_matrix_dataframe(
        matrix_sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        pooling_strategy=pooling_strategy,
    )

    normalized_pooling_strategy = normalize_pooling_strategy(pooling_strategy)
    config_payload = {
        "analysis_type": "curated_pairs",
        "model_dir": str(model_dir) if model_dir is not None else "",
        "output_dir": str(output_dir),
        "sequence_pairs": list(sequence_pairs),
        "matrix_sequences": list(matrix_sequences),
        "max_length": max_length,
        "batch_size": batch_size,
        "pooling_strategy": normalized_pooling_strategy,
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
