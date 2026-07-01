"""Utilities for embedding glycans and comparing saved MLM checkpoints."""

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
