"""General-purpose helpers for project notebooks.

This module is meant for notebook-facing utilities that are reusable across
multiple notebooks but are not specifically part of runtime bootstrapping.
Examples include overwrite checks, shared path validation, and seed handling.
"""

from __future__ import annotations

import random
from pathlib import Path


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
