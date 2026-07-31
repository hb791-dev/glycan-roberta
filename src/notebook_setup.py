"""Shared setup helpers for project notebooks.

These helpers are designed for Colab-based notebooks that read data from
Google Drive and import project code from a synced GitHub repository.
The goal is to keep each notebook focused on the analysis instead of
repeating environment and path setup logic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.notebook_utils import require_existing_path


DEFAULT_CONFIG_PATH = Path("/content/drive/MyDrive/glycan_roberta_config.json")
REQUIRED_CONFIG_KEYS = ("project_root", "github_owner", "repo_name", "github_ref")


@dataclass(frozen=True)
class NotebookContext:
    """Bundle the main paths and settings a notebook needs.

    A small context object keeps the notebook code readable because the
    notebook can refer to ``ctx.project_root`` or ``ctx.results_dir``
    instead of rebuilding the same paths in every file.
    """

    notebook_name: str
    project_root: Path
    repo_dir: Path
    data_dir: Path
    raw_data_dir: Path
    splits_dir: Path
    results_dir: Path
    tokenizers_dir: Path
    tokenized_datasets_dir: Path
    checkpoints_dir: Path
    registry_dir: Path
    public_reports_dir: Path
    config: dict[str, Any]


def mount_google_drive_if_needed(mount_point: str = "/content/drive") -> Path:
    """Mount Google Drive when running inside Colab.

    The import is kept inside the function so the module can still be
    imported in a non-Colab environment for smoke tests or documentation work.
    """

    mount_path = Path(mount_point)
    if mount_path.exists():
        return mount_path

    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive mounting is only available inside a Colab runtime."
        ) from exc

    print("Mounting Google Drive...")
    drive.mount(mount_point)
    return mount_path


def load_project_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the shared notebook configuration from JSON.

    Storing project-root and repository settings in one file avoids the need
    to edit the same values at the top of every notebook.
    """

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            "Notebook config file not found. Expected to find it at "
            f"{config_path}."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        missing_display = ", ".join(missing_keys)
        raise KeyError(
            f"Notebook config is missing required keys: {missing_display}"
        )

    if "repo_dir" not in config:
        config["repo_dir"] = f"/content/{config['repo_name']}"

    return config


def sync_repo(
    owner: str,
    repo_name: str,
    repo_dir: str | Path,
    github_ref: str,
) -> Path:
    """Clone the repository if needed, otherwise fast-forward pull the branch.

    ``check=True`` is used so a failed repository sync stops the notebook
    immediately instead of allowing it to continue with stale helper code.
    """

    repo_dir = Path(repo_dir)
    repo_url = f"https://github.com/{owner}/{repo_name}.git"

    if not repo_dir.exists():
        print(f"Cloning repository from {repo_url} ...")
        subprocess.run(
            ["git", "clone", "--quiet", repo_url, str(repo_dir)],
            check=True,
        )
    else:
        print(f"Repository already present at {repo_dir}.")

    print(f"Updating repository to the latest {github_ref} changes...")
    subprocess.run(
        ["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", github_ref],
        check=True,
    )

    return repo_dir


def add_repo_to_sys_path(repo_dir: str | Path) -> Path:
    """Ensure the repository root is available for ``src`` imports."""

    repo_dir = Path(repo_dir).resolve()
    repo_dir_str = str(repo_dir)
    if repo_dir_str not in sys.path:
        sys.path.insert(0, repo_dir_str)
    return repo_dir


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_notebook_context(
    notebook_name: str,
    config: dict[str, Any],
    repo_dir: str | Path,
) -> NotebookContext:
    """Build the shared path context used by project notebooks."""

    project_root = require_existing_path(config["project_root"], "Project root")
    repo_dir = require_existing_path(repo_dir, "Repository directory")

    data_dir = project_root / "data"
    raw_data_dir = data_dir / "raw"
    splits_dir = data_dir / "splits"
    results_dir = ensure_directory(project_root / "results")
    tokenizers_dir = ensure_directory(project_root / "tokenizers")
    tokenized_datasets_dir = ensure_directory(project_root / "tokenized_datasets")
    checkpoints_dir = ensure_directory(project_root / "checkpoints")
    registry_dir = ensure_directory(project_root / "registry")
    public_reports_dir = ensure_directory(project_root / "public_reports")

    return NotebookContext(
        notebook_name=notebook_name,
        project_root=project_root,
        repo_dir=Path(repo_dir),
        data_dir=data_dir,
        raw_data_dir=raw_data_dir,
        splits_dir=splits_dir,
        results_dir=results_dir,
        tokenizers_dir=tokenizers_dir,
        tokenized_datasets_dir=tokenized_datasets_dir,
        checkpoints_dir=checkpoints_dir,
        registry_dir=registry_dir,
        public_reports_dir=public_reports_dir,
        config=config,
    )


def bootstrap_notebook(
    notebook_name: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    require_drive: bool = True,
    require_repo_sync: bool = True,
) -> NotebookContext:
    """Prepare the runtime and return a standard notebook context.

    Parameters
    ----------
    notebook_name:
        A short label used only for readable status messages.
    config_path:
        Location of the shared JSON config file that stores the project root
        and repository settings.
    require_drive:
        When ``True``, the helper mounts Google Drive before reading the config.
    require_repo_sync:
        When ``True``, the helper pulls the latest repository version after the
        config is loaded.
    """

    print(f"Preparing notebook runtime for: {notebook_name}")

    if require_drive:
        mount_google_drive_if_needed()

    config = load_project_config(config_path)
    repo_dir = Path(config["repo_dir"])

    if require_repo_sync:
        repo_dir = sync_repo(
            owner=config["github_owner"],
            repo_name=config["repo_name"],
            repo_dir=repo_dir,
            github_ref=config["github_ref"],
        )
    else:
        repo_dir = require_existing_path(repo_dir, "Repository directory")

    add_repo_to_sys_path(repo_dir)
    context = build_notebook_context(
        notebook_name=notebook_name,
        config=config,
        repo_dir=repo_dir,
    )

    print("Notebook runtime ready.")
    print(f" - Repository directory: {context.repo_dir}")
    print(f" - Project root: {context.project_root}")

    return context
