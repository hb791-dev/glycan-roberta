"""Utilities for embedding glycans and analyzing saved MLM checkpoints.

This module keeps the reusable mechanics out of notebooks so analysis notebooks
can focus on configuration, interpretation, and display.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

if TYPE_CHECKING:
    import pandas as pd


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
        encoded_batch = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=use_max_length,
        )
        encoded_batch = {name: tensor.to(runtime_device) for name, tensor in encoded_batch.items()}

        hidden_states = encoder(**encoded_batch).last_hidden_state
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
        comparison_rows.append(
            {
                "pair_name": pair["pair_name"],
                "seq1": result["seq1"],
                "seq2": result["seq2"],
                "cosine_similarity": result["cosine_similarity"],
                "seq1_tokens": " | ".join(result["seq1_tokens"]),
                "seq2_tokens": " | ".join(result["seq2_tokens"]),
            }
        )

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


def validate_similarity_inputs(
    model_dir,
    sequence_pairs: Sequence[dict],
    matrix_sequences: Sequence[str],
    output_dir=None,
) -> None:
    """Validate the minimum inputs required for one similarity-analysis run.

    The notebook should fail early with clear messages when a checkpoint path is
    wrong or the requested sequence lists are empty. Doing this in src keeps the
    checks consistent if the same workflow is reused elsewhere.
    """
    from pathlib import Path

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


def plot_similarity_heatmap(similarity_df, output_path, title: str) -> None:
    """Display the similarity heatmap inline and save it to disk.

    Keeping the plotting logic here avoids repeating formatting code in the
    notebook and makes future plot changes apply everywhere consistently.
    """
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
    """Write similarity outputs to disk and return the saved file paths.

    The returned dictionary lets notebooks print or reuse output paths without
    reconstructing them manually.
    """
    from pathlib import Path
    import json

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
    model_dir=None,
) -> dict:
    """Run one end-to-end similarity analysis and return notebook-ready results.

    This function is intentionally high level: it computes the pairwise results,
    tokenization preview, full similarity matrix, and output files in one call.
    The notebook can then focus on showing the returned tables and paths.
    """
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
