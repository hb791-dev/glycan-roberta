"""Helpers for reviewing trainer history after MLM pretraining."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd


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
