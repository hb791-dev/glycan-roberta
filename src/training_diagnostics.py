"""Helpers for reviewing trainer history after MLM pretraining.

This module supports notebook 05 by keeping repetitive validation-review
workflow code out of the notebook body. The notebook can stay focused on:

- choosing one finished pretraining run
- inspecting training and validation loss history
- saving a compact validation review back to Drive
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.notebook_utils import require_existing_path, validate_output_paths
from src.run_index import upsert_run_record


NOTEBOOK_PATH = "notebooks/05_validation_diagnostics.ipynb"
SUPPORTED_TOKENIZER_FAMILIES = (
    "byte_bpe",
    "glyberta",
    "manual",
    "hybrid_char_bpe",
    "linkage_block",
    "donor_bound",
    "semi_atomic",
)


def _load_json(input_path: str | Path) -> dict[str, object]:
    """Read one small JSON file from disk."""

    input_path = Path(input_path)
    return json.loads(input_path.read_text(encoding="utf-8"))


def _save_json(payload: dict[str, object], output_path: str | Path) -> Path:
    """Write one formatted JSON file and return its path."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def build_validation_paths(
    project_root: str | Path,
    tokenizer_family: str,
    experiment_name: str,
) -> dict[str, Path]:
    """Build the standard Drive paths used by notebook 05."""

    project_root = Path(project_root)
    checkpoint_dir = project_root / "checkpoints" / tokenizer_family / experiment_name
    validation_results_dir = (
        project_root / "results" / "validation" / tokenizer_family / experiment_name
    )

    return {
        "project_root": project_root,
        "checkpoint_dir": checkpoint_dir,
        "trainer_state_path": checkpoint_dir / "trainer_state.json",
        "experiment_metadata_path": checkpoint_dir / "experiment_metadata.json",
        "run_index_path": project_root / "registry" / "run_index.csv",
        "validation_results_dir": validation_results_dir,
        "loss_plot_path": validation_results_dir / "loss_curves.png",
        "loss_history_path": validation_results_dir / "loss_history.csv",
        "validation_summary_path": validation_results_dir / "validation_summary.json",
    }


def build_validation_output_paths(
    validation_paths: dict[str, Path],
) -> dict[str, Path]:
    """Return the notebook-05 files that are treated as overwrite-checked outputs."""

    return {
        "loss_plot_path": validation_paths["loss_plot_path"],
        "loss_history_path": validation_paths["loss_history_path"],
        "validation_summary_path": validation_paths["validation_summary_path"],
    }


def prepare_validation_run(
    project_root: str | Path,
    tokenizer_family: str,
    experiment_name: str,
    overwrite_existing_outputs: bool = False,
    notebook_used: str = NOTEBOOK_PATH,
) -> dict[str, object]:
    """Validate notebook-05 settings and load the saved experiment metadata."""

    tokenizer_family = str(tokenizer_family).strip()
    experiment_name = str(experiment_name).strip()

    if tokenizer_family not in SUPPORTED_TOKENIZER_FAMILIES:
        supported_families = ", ".join(SUPPORTED_TOKENIZER_FAMILIES)
        raise ValueError(
            f"Unsupported tokenizer family: {tokenizer_family}. "
            f"Choose from: {supported_families}"
        )

    if not experiment_name:
        raise ValueError("EXPERIMENT_NAME must not be empty.")

    paths = build_validation_paths(
        project_root=project_root,
        tokenizer_family=tokenizer_family,
        experiment_name=experiment_name,
    )

    require_existing_path(paths["checkpoint_dir"], "Checkpoint directory")
    require_existing_path(paths["trainer_state_path"], "Trainer state file")
    require_existing_path(paths["experiment_metadata_path"], "Experiment metadata file")
    require_existing_path(paths["run_index_path"], "Run index file")
    paths["validation_results_dir"].mkdir(parents=True, exist_ok=True)
    validate_output_paths(
        build_validation_output_paths(paths),
        overwrite_existing_outputs=overwrite_existing_outputs,
    )

    experiment_metadata = _load_json(paths["experiment_metadata_path"])
    metadata_experiment_name = str(
        experiment_metadata.get("experiment_name", "")
    ).strip()
    metadata_tokenizer_family = str(
        experiment_metadata.get("live_hyperparameters", {}).get("tokenizer_family", "")
    ).strip()

    if metadata_experiment_name and metadata_experiment_name != experiment_name:
        raise ValueError(
            "EXPERIMENT_NAME does not match the saved experiment metadata: "
            f"{experiment_name} vs {metadata_experiment_name}"
        )

    if metadata_tokenizer_family and metadata_tokenizer_family != tokenizer_family:
        raise ValueError(
            "TOKENIZER_FAMILY does not match the saved experiment metadata: "
            f"{tokenizer_family} vs {metadata_tokenizer_family}"
        )

    return {
        "experiment_name": experiment_name,
        "tokenizer_family": tokenizer_family,
        "notebook_used": notebook_used,
        "paths": paths,
        "experiment_metadata": experiment_metadata,
    }


def load_trainer_history(trainer_state_path: str) -> pd.DataFrame:
    """Load the Hugging Face trainer log history into a dataframe."""
    with open(trainer_state_path, "r", encoding="utf-8") as file:
        trainer_state = json.load(file)

    return pd.DataFrame(trainer_state["log_history"])


def split_train_eval_history(log_df: pd.DataFrame):
    """Split trainer history into training-loss and validation-loss tables."""
    train_rows = log_df[log_df["loss"].notna() & log_df["epoch"].notna()].copy()
    eval_rows = log_df[log_df["eval_loss"].notna() & log_df["epoch"].notna()].copy()

    train_rows = (
        train_rows[["epoch", "loss"]]
        .rename(columns={"loss": "train_loss"})
        .sort_values("epoch")
    )
    eval_rows = (
        eval_rows[["epoch", "eval_loss"]]
        .rename(columns={"eval_loss": "val_loss"})
        .sort_values("epoch")
    )

    return train_rows, eval_rows


def plot_loss_curves(train_rows: pd.DataFrame, eval_rows: pd.DataFrame) -> None:
    """Plot training and validation loss against epoch."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_rows["epoch"], train_rows["train_loss"], label="Training loss")
    plt.plot(eval_rows["epoch"], eval_rows["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def save_loss_curve_plot(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    output_path: str,
) -> None:
    """Save the training and validation loss plot to disk."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_rows["epoch"], train_rows["train_loss"], label="Training loss")
    plt.plot(eval_rows["epoch"], eval_rows["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def summarize_best_epoch(eval_rows: pd.DataFrame) -> dict:
    """Return the best and final validation checkpoints from one run."""
    best_idx = eval_rows["val_loss"].idxmin()
    best_row = eval_rows.loc[best_idx]

    return {
        "best_epoch": float(best_row["epoch"]),
        "best_val_loss": float(best_row["val_loss"]),
        "last_epoch": float(eval_rows["epoch"].max()),
        "last_val_loss": float(eval_rows.iloc[-1]["val_loss"]),
    }


def recommend_continuation(
    eval_rows: pd.DataFrame,
    total_epochs: int,
    tail_fraction: float = 0.15,
    recent_window: int = 5,
) -> str:
    """Return a simple continuation recommendation from validation history.

    This helper is intentionally conservative. It only recommends continuation
    when the best validation epoch is late in training and the most recent
    validation window has not clearly worsened.
    """
    best_idx = eval_rows["val_loss"].idxmin()
    best_epoch = float(eval_rows.loc[best_idx, "epoch"])

    tail_start = total_epochs * (1 - tail_fraction)
    best_in_tail = best_epoch >= tail_start
    recent_losses = eval_rows.tail(recent_window)["val_loss"].to_numpy()

    if best_in_tail and len(recent_losses) >= 2 and recent_losses[-1] <= recent_losses[0]:
        return "Continuation is worth considering."

    return "Continuation is probably not necessary."


def merge_loss_history(train_rows: pd.DataFrame, eval_rows: pd.DataFrame) -> pd.DataFrame:
    """Combine training and validation loss tables on epoch for export."""
    return pd.merge(train_rows, eval_rows, on="epoch", how="outer").sort_values("epoch")


def build_validation_review(
    run_context: dict[str, object],
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
) -> dict[str, object]:
    """Build the main notebook-05 validation summary objects.

    This helper packages the core review outputs so the notebook can display the
    summary table while shared code keeps the repeated JSON and CSV structure
    consistent across runs.
    """

    experiment_metadata = run_context["experiment_metadata"]
    paths = run_context["paths"]
    live_hyperparameters = experiment_metadata.get("live_hyperparameters", {})
    total_epochs = int(float(live_hyperparameters["epochs"]))

    loss_history_export = merge_loss_history(train_rows, eval_rows)
    best_epoch_summary = summarize_best_epoch(eval_rows)
    continuation_recommendation = recommend_continuation(
        eval_rows,
        total_epochs=total_epochs,
    )

    validation_summary = {
        "experiment_name": run_context["experiment_name"],
        "tokenizer_family": live_hyperparameters.get(
            "tokenizer_family",
            run_context["tokenizer_family"],
        ),
        "setting_label": live_hyperparameters.get("setting_label", ""),
        "run_mode": live_hyperparameters.get("run_mode", ""),
        "checkpoint_dir": str(paths["checkpoint_dir"]),
        "trainer_state_path": str(paths["trainer_state_path"]),
        "loss_plot_path": str(paths["loss_plot_path"]),
        "loss_history_path": str(paths["loss_history_path"]),
        "best_epoch_summary": best_epoch_summary,
        "continuation_recommendation": continuation_recommendation,
    }

    summary_df = pd.DataFrame(
        {
            "metric": [
                "best_epoch",
                "best_val_loss",
                "last_epoch",
                "last_val_loss",
                "continuation_recommendation",
            ],
            "value": [
                best_epoch_summary["best_epoch"],
                best_epoch_summary["best_val_loss"],
                best_epoch_summary["last_epoch"],
                best_epoch_summary["last_val_loss"],
                continuation_recommendation,
            ],
        }
    )

    return {
        "loss_history_export": loss_history_export,
        "best_epoch_summary": best_epoch_summary,
        "continuation_recommendation": continuation_recommendation,
        "validation_summary": validation_summary,
        "summary_df": summary_df,
    }


def _build_validation_run_record(
    run_context: dict[str, object],
    validation_summary: dict[str, object],
    notes: str,
) -> dict[str, object]:
    """Build the run-index update payload for one validation review."""

    experiment_metadata = run_context["experiment_metadata"]
    live_hyperparameters = experiment_metadata.get("live_hyperparameters", {})
    vault_routing = experiment_metadata.get("vault_routing", {})
    paths = run_context["paths"]

    return {
        "experiment_name": run_context["experiment_name"],
        "tokenizer_family": validation_summary["tokenizer_family"],
        "setting_label": validation_summary["setting_label"],
        "run_mode": validation_summary["run_mode"],
        "parent_experiment_name": live_hyperparameters.get("parent_experiment_name", ""),
        "mlm_probability": live_hyperparameters.get("mlm_probability", ""),
        "num_hidden_layers": live_hyperparameters.get("num_hidden_layers", ""),
        "attention_heads": live_hyperparameters.get("attention_heads", ""),
        "hidden_size": live_hyperparameters.get("hidden_size", ""),
        "intermediate_size": live_hyperparameters.get("intermediate_size", ""),
        "batch_size": live_hyperparameters.get("batch_size", ""),
        "learning_rate": live_hyperparameters.get("learning_rate", ""),
        "weight_decay": live_hyperparameters.get("weight_decay", ""),
        "epochs": live_hyperparameters.get("epochs", ""),
        "early_stopping_patience": live_hyperparameters.get(
            "early_stopping_patience", ""
        ),
        "tokenizer_dir": vault_routing.get("tokenizer_dir", ""),
        "tokenized_dataset_dir": vault_routing.get("tokenized_dataset_dir", ""),
        "checkpoint_dir": vault_routing.get(
            "checkpoint_dir",
            str(paths["checkpoint_dir"]),
        ),
        "results_dir": str(paths["validation_results_dir"]),
        "validation_summary_path": str(paths["validation_summary_path"]),
        "notebook_used": run_context["notebook_used"],
        "git_commit": experiment_metadata.get("git_commit", ""),
        "run_status": experiment_metadata.get("run_status", ""),
        "notes": notes,
    }


def save_validation_review(
    run_context: dict[str, object],
    validation_summary: dict[str, object],
    loss_history_export: pd.DataFrame,
    notes: str,
) -> dict[str, Path]:
    """Save notebook-05 outputs and register the validation review in the index."""

    paths = run_context["paths"]
    loss_history_export.to_csv(paths["loss_history_path"], index=False)
    _save_json(validation_summary, paths["validation_summary_path"])

    upsert_run_record(
        str(paths["run_index_path"]),
        _build_validation_run_record(
            run_context=run_context,
            validation_summary=validation_summary,
            notes=notes,
        ),
    )

    return {
        "loss_history_path": paths["loss_history_path"],
        "validation_summary_path": paths["validation_summary_path"],
    }
