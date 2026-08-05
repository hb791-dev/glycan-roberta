"""General-purpose helpers for project notebooks.

This module is meant for notebook-facing utilities that are reusable across
multiple notebooks but are not specifically part of runtime bootstrapping.
Examples include overwrite checks, shared path validation, and seed handling.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


SUPPORTED_TOKENIZER_FAMILIES = (
    "byte_bpe",
    "glyberta",
    "manual",
    "hybrid_char_bpe",
    "linkage_block",
    "donor_bound",
    "semi_atomic",
)


def require_existing_path(path: str | Path, description: str) -> Path:
    """Raise a descriptive error when an expected file or folder is missing."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def validate_output_paths(
    output_paths: dict[str, Path],
    overwrite_existing_outputs: bool,
) -> None:
    """Check whether planned output files already exist.

    Many notebooks in this project save results to predictable file names. This
    helper centralizes the overwrite policy so each notebook can stay focused
    on analysis steps instead of repeating the same file-existence logic.
    """

    existing_outputs = [
        Path(path) for path in output_paths.values() if Path(path).exists()
    ]

    if existing_outputs and not overwrite_existing_outputs:
        existing_display = "\n".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            "Existing output files were found, and overwriting is disabled.\n"
            f"{existing_display}\n\n"
            "Set OVERWRITE_EXISTING_OUTPUTS = True to allow replacement."
        )


def resolve_random_seed(random_seed: int | None) -> int:
    """Return a concrete random seed for the current run.

    When the caller does not provide a seed, this helper chooses one and
    returns it so the notebook can print and reuse the exact value.
    """

    if random_seed is None:
        return random.randrange(2**32)
    return int(random_seed)


def validate_tokenizer_family(
    tokenizer_family: str,
    *,
    supported_families: tuple[str, ...] = SUPPORTED_TOKENIZER_FAMILIES,
) -> str:
    """Return a normalized tokenizer family or raise a clear error."""

    normalized_family = str(tokenizer_family).strip()
    if normalized_family not in supported_families:
        valid_display = ", ".join(supported_families)
        raise ValueError(
            f"Unsupported tokenizer family: {normalized_family}. "
            f"Choose from: {valid_display}"
        )
    return normalized_family


def stringify_path_values(record: dict[str, object]) -> dict[str, object]:
    """Convert any Path values to strings before JSON serialization."""

    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in record.items()
    }


def write_json(path: str | Path, payload: dict[str, object]) -> Path:
    """Write one JSON file with consistent formatting across notebooks."""

    path = Path(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
