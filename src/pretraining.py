"""Shared helpers for notebook 04 RoBERTa masked-language-model pretraining.

The pretraining notebook needs to stay beginner-friendly because it is a
workflow notebook, not a software module. These helpers keep the repeated
runtime mechanics out of the notebook body while preserving the existing
Drive folder layout, experiment naming, and metadata files that later
notebooks already depend on.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    RobertaConfig,
    RobertaForMaskedLM,
    TrainerCallback,
    TrainingArguments,
)

from src.dataset_preprocessing import load_fast_tokenizer
from src.notebook_utils import require_existing_path, resolve_random_seed
from src.run_index import upsert_run_record


NOTEBOOK_PATH = "notebooks/04_roberta_pretraining.ipynb"
VALID_RUN_MODES = {"fresh", "resume_checkpoint", "continue_best_model"}
VALID_INTERVAL_STRATEGIES = {"epoch", "steps"}


class GlycanMLMDataset(Dataset):
    """Wrap saved tensor dictionaries so Hugging Face Trainer can read them."""

    def __init__(self, dataset_dict: dict[str, torch.Tensor]):
        self.input_ids = dataset_dict["input_ids"]
        self.attention_mask = dataset_dict["attention_mask"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def _to_json_safe(value):
    """Convert notebook metadata values into JSON-safe objects."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    return value


def save_json(payload: dict[str, object], output_path: str | Path) -> Path:
    """Write one formatted JSON file and return its path."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_to_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    return output_path


def build_pretraining_paths(
    project_root: str | Path,
    tokenizer_family: str,
    setting_label: str,
) -> dict[str, Path]:
    """Build the standard Drive paths used by notebook 04."""

    project_root = Path(project_root)
    checkpoint_root = project_root / "checkpoints" / tokenizer_family
    tokenizer_dir = project_root / "tokenizers" / tokenizer_family / setting_label
    tokenized_dataset_dir = (
        project_root / "tokenized_datasets" / tokenizer_family / setting_label
    )

    return {
        "project_root": project_root,
        "checkpoint_root": checkpoint_root,
        "tokenizer_dir": tokenizer_dir,
        "tokenized_dataset_dir": tokenized_dataset_dir,
        "run_index_path": project_root / "registry" / "run_index.csv",
    }


def _normalize_run_mode(run_mode: str) -> str:
    """Return one validated run mode string."""

    normalized_run_mode = str(run_mode).strip().lower()
    if normalized_run_mode not in VALID_RUN_MODES:
        supported_modes = ", ".join(sorted(VALID_RUN_MODES))
        raise ValueError(f"RUN_MODE must be one of: {supported_modes}")
    return normalized_run_mode


def _sanitize_experiment_name(value: str, field_name: str) -> str:
    """Reject paths when the notebook expects only a plain experiment name."""

    text_value = str(value).strip()
    cleaned_value = Path(text_value).name
    if not cleaned_value:
        raise ValueError(f"{field_name} must not be empty.")
    if cleaned_value != text_value:
        raise ValueError(
            f"{field_name} must contain an experiment name only, not a path: {value}"
        )
    return cleaned_value


def _derive_parent_experiment_name(resume_source_dir: str | Path) -> str:
    """Infer the experiment name from a checkpoint or best-model folder."""

    resume_source_dir = Path(resume_source_dir)
    parent_dir = resume_source_dir.parent
    if not parent_dir.name:
        raise ValueError(
            "Could not derive the parent experiment name from RESUME_SOURCE_DIR."
        )
    return parent_dir.name


def _resolve_training_schedule(
    run_mode: str,
    resume_source_dir: str | Path | None,
    initial_epochs: int,
    continuation_epochs: int,
    base_learning_rate: float,
    continuation_learning_rate: float,
) -> dict[str, int | float]:
    """Convert run-mode settings into the effective epochs and learning rate."""

    if run_mode == "fresh":
        return {
            "epochs": int(initial_epochs),
            "learning_rate": float(base_learning_rate),
        }

    if run_mode == "resume_checkpoint":
        if not resume_source_dir:
            raise ValueError("resume_checkpoint mode requires RESUME_SOURCE_DIR.")
        return {
            "epochs": int(initial_epochs) + int(continuation_epochs),
            "learning_rate": float(base_learning_rate),
        }

    if run_mode == "continue_best_model":
        if not resume_source_dir:
            raise ValueError("continue_best_model mode requires RESUME_SOURCE_DIR.")
        return {
            "epochs": int(continuation_epochs),
            "learning_rate": float(continuation_learning_rate),
        }

    raise ValueError(f"Unsupported run mode: {run_mode}")


def _normalize_interval_strategy(value: str, field_name: str) -> str:
    """Validate one Hugging Face interval strategy value."""

    normalized_value = str(value).strip().lower()
    if normalized_value not in VALID_INTERVAL_STRATEGIES:
        supported_values = ", ".join(sorted(VALID_INTERVAL_STRATEGIES))
        raise ValueError(f"{field_name} must be one of: {supported_values}")
    return normalized_value


def _normalize_positive_int(value: int | None, field_name: str) -> int | None:
    """Validate one optional positive integer notebook setting."""

    if value is None:
        return None

    normalized_value = int(value)
    if normalized_value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return normalized_value


def _resolve_checkpoint_schedule(
    checkpoint_save_strategy: str,
    checkpoint_save_steps: int | None,
    evaluation_strategy: str | None,
    evaluation_steps: int | None,
    early_checkpoint_save_end_step: int | None,
    early_checkpoint_save_interval: int | None,
    load_best_model_at_end: bool,
) -> dict[str, object]:
    """Resolve one validated checkpoint and evaluation schedule."""

    resolved_save_strategy = _normalize_interval_strategy(
        checkpoint_save_strategy,
        "CHECKPOINT_SAVE_STRATEGY",
    )
    resolved_evaluation_strategy = (
        resolved_save_strategy
        if evaluation_strategy is None
        else _normalize_interval_strategy(
            evaluation_strategy,
            "EVALUATION_STRATEGY",
        )
    )

    resolved_save_steps = None
    if resolved_save_strategy == "steps":
        resolved_save_steps = _normalize_positive_int(
            checkpoint_save_steps,
            "CHECKPOINT_SAVE_STEPS",
        )
        if resolved_save_steps is None:
            raise ValueError(
                "CHECKPOINT_SAVE_STEPS is required when CHECKPOINT_SAVE_STRATEGY='steps'."
            )

    resolved_evaluation_steps = None
    if resolved_evaluation_strategy == "steps":
        if evaluation_steps is None and resolved_save_strategy == "steps":
            resolved_evaluation_steps = resolved_save_steps
        else:
            resolved_evaluation_steps = _normalize_positive_int(
                evaluation_steps,
                "EVALUATION_STEPS",
            )

        if resolved_evaluation_steps is None:
            raise ValueError(
                "EVALUATION_STEPS is required when EVALUATION_STRATEGY='steps'."
            )

    resolved_early_checkpoint_save_end_step = _normalize_positive_int(
        early_checkpoint_save_end_step,
        "EARLY_CHECKPOINT_SAVE_END_STEP",
    )
    resolved_early_checkpoint_save_interval = _normalize_positive_int(
        early_checkpoint_save_interval,
        "EARLY_CHECKPOINT_SAVE_INTERVAL",
    )

    if resolved_early_checkpoint_save_end_step is not None:
        if resolved_save_strategy != "steps":
            raise ValueError(
                "EARLY_CHECKPOINT_SAVE_END_STEP requires "
                "CHECKPOINT_SAVE_STRATEGY='steps'."
            )
        if resolved_early_checkpoint_save_interval is None:
            raise ValueError(
                "EARLY_CHECKPOINT_SAVE_INTERVAL is required when "
                "EARLY_CHECKPOINT_SAVE_END_STEP is set."
            )
        if (
            resolved_early_checkpoint_save_end_step
            >= resolved_save_steps
        ):
            raise ValueError(
                "EARLY_CHECKPOINT_SAVE_END_STEP must be smaller than "
                "CHECKPOINT_SAVE_STEPS so the early-step burst stays distinct "
                "from the regular save cadence."
            )
    elif resolved_early_checkpoint_save_interval is not None:
        raise ValueError(
            "EARLY_CHECKPOINT_SAVE_INTERVAL requires "
            "EARLY_CHECKPOINT_SAVE_END_STEP to be set."
        )

    if load_best_model_at_end and resolved_save_strategy != resolved_evaluation_strategy:
        raise ValueError(
            "load_best_model_at_end=True requires CHECKPOINT_SAVE_STRATEGY and "
            "EVALUATION_STRATEGY to match."
        )

    if (
        load_best_model_at_end
        and resolved_save_strategy == "steps"
        and resolved_save_steps % resolved_evaluation_steps != 0
    ):
        raise ValueError(
            "When step-based saving is used with load_best_model_at_end=True, "
            "CHECKPOINT_SAVE_STEPS must be a multiple of EVALUATION_STEPS."
        )

    return {
        "checkpoint_save_strategy": resolved_save_strategy,
        "checkpoint_save_steps": resolved_save_steps,
        "evaluation_strategy": resolved_evaluation_strategy,
        "evaluation_steps": resolved_evaluation_steps,
        "early_checkpoint_save_end_step": resolved_early_checkpoint_save_end_step,
        "early_checkpoint_save_interval": resolved_early_checkpoint_save_interval,
        "load_best_model_at_end": bool(load_best_model_at_end),
    }


def _validate_resume_source(run_mode: str, resume_source_dir: str | Path) -> Path:
    """Confirm that a continuation run points at the expected saved folder type."""

    resume_source_path = require_existing_path(
        resume_source_dir,
        "Resume source directory",
    )

    if run_mode == "resume_checkpoint" and "checkpoint-" not in resume_source_path.name:
        raise ValueError(
            "resume_checkpoint mode must point to a checkpoint-* directory."
        )

    if run_mode == "continue_best_model" and resume_source_path.name != "best_model":
        raise ValueError(
            "continue_best_model mode must point to a best_model directory."
        )

    return resume_source_path


def _validate_resume_architecture(
    resume_source_dir: str | Path,
    num_hidden_layers: int,
    attention_heads: int,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    """Check that a continuation run still matches the requested architecture."""

    config_path = Path(resume_source_dir) / "config.json"
    if not config_path.exists():
        return

    resume_config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_values = {
        "num_hidden_layers": int(num_hidden_layers),
        "num_attention_heads": int(attention_heads),
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size),
    }

    for key, expected_value in expected_values.items():
        observed_value = resume_config.get(key)
        if observed_value != expected_value:
            raise ValueError(
                f"Resume model mismatch for {key}: expected {expected_value}, "
                f"found {observed_value}"
            )


def _validate_resume_location_and_metadata(
    resume_source_dir: str | Path,
    checkpoint_root: str | Path,
    tokenizer_dir: str | Path,
    tokenized_dataset_dir: str | Path,
    tokenizer_family: str,
    setting_label: str,
) -> None:
    """Reject nested or mismatched continuation sources before training starts.

    Continuation runs should point to one of the standard notebook-04 outputs:

    - ``.../checkpoints/<tokenizer_family>/<experiment_name>/checkpoint-*``
    - ``.../checkpoints/<tokenizer_family>/<experiment_name>/best_model``

    This check prevents the notebook from quietly accepting a deeply nested path
    or a saved run that belongs to a different tokenizer family or dataset
    setting than the one selected in the user settings cell.
    """

    resume_source_dir = Path(resume_source_dir)
    checkpoint_root = Path(checkpoint_root).resolve()
    parent_experiment_dir = resume_source_dir.parent.resolve()

    if parent_experiment_dir.parent != checkpoint_root:
        raise ValueError(
            "RESUME_SOURCE_DIR must point to a checkpoint or best-model folder "
            "that lives directly inside the selected tokenizer family's "
            f"checkpoint root:\nExpected parent root: {checkpoint_root}\n"
            f"Observed parent root: {parent_experiment_dir.parent}"
        )

    metadata_path = parent_experiment_dir / "experiment_metadata.json"
    if not metadata_path.exists():
        return

    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_family = str(
        metadata_payload.get("live_hyperparameters", {}).get("tokenizer_family", "")
    ).strip()
    metadata_setting = str(
        metadata_payload.get("live_hyperparameters", {}).get("setting_label", "")
    ).strip()
    metadata_tokenizer_dir = str(
        metadata_payload.get("vault_routing", {}).get("tokenizer_dir", "")
    ).strip()
    metadata_dataset_dir = str(
        metadata_payload.get("vault_routing", {}).get("tokenized_dataset_dir", "")
    ).strip()

    if metadata_family and metadata_family != str(tokenizer_family):
        raise ValueError(
            "RESUME_SOURCE_DIR belongs to a different tokenizer family than the "
            f"one selected in this notebook: {metadata_family} vs {tokenizer_family}"
        )

    if metadata_setting and metadata_setting != str(setting_label):
        raise ValueError(
            "RESUME_SOURCE_DIR belongs to a different tokenizer setting than the "
            f"one selected in this notebook: {metadata_setting} vs {setting_label}"
        )

    if metadata_tokenizer_dir and Path(metadata_tokenizer_dir) != Path(tokenizer_dir):
        raise ValueError(
            "RESUME_SOURCE_DIR was trained with a different tokenizer directory "
            "than the one selected in this notebook:\n"
            f"Metadata tokenizer dir: {metadata_tokenizer_dir}\n"
            f"Selected tokenizer dir: {tokenizer_dir}"
        )

    if metadata_dataset_dir and Path(metadata_dataset_dir) != Path(tokenized_dataset_dir):
        raise ValueError(
            "RESUME_SOURCE_DIR was trained with a different tokenized dataset "
            "directory than the one selected in this notebook:\n"
            f"Metadata dataset dir: {metadata_dataset_dir}\n"
            f"Selected dataset dir: {tokenized_dataset_dir}"
        )


def _get_git_commit(repo_dir: str | Path) -> str:
    """Return the exact repository commit used for this notebook run."""

    repo_dir = require_existing_path(repo_dir, "Repository directory")
    return (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
        )
        .decode("utf-8")
        .strip()
    )


def _format_learning_rate_tag(value: float) -> str:
    """Convert one learning-rate value into the folder-name format in use."""

    return str(value).replace(".", "")


def _build_base_experiment_name(
    run_mode: str,
    parent_experiment_name: str | None,
    mlm_probability: float,
    num_hidden_layers: int,
    hidden_size: int,
    attention_heads: int,
    learning_rate: float,
    epochs: int,
    setting_label: str,
) -> str:
    """Build the base experiment name before versioning is applied."""

    architecture_tag = f"L{num_hidden_layers}_H{hidden_size}_A{attention_heads}"
    learning_rate_tag = _format_learning_rate_tag(learning_rate)

    if run_mode == "fresh":
        return (
            f"mlm{int(float(mlm_probability) * 100)}_{architecture_tag}_"
            f"lr{learning_rate_tag}_ep{epochs}_set{setting_label}"
        )

    if run_mode == "resume_checkpoint":
        return f"{parent_experiment_name}_resume_toep{epochs}"

    return f"{parent_experiment_name}_cont_lr{learning_rate_tag}_ep{epochs}"


def _resolve_unique_directory(base_dir: str | Path) -> Path:
    """Return one unused experiment directory, adding a version suffix if needed."""

    base_dir = Path(base_dir)
    if not base_dir.exists():
        return base_dir

    version = 2
    while True:
        candidate_dir = base_dir.parent / f"{base_dir.name}_v{version}"
        if not candidate_dir.exists():
            return candidate_dir
        version += 1


def _build_pretraining_run_record(
    run_context: dict[str, object],
    status: str,
) -> dict[str, object]:
    """Build the row written to the shared run index."""

    settings = run_context["settings"]
    paths = run_context["paths"]

    return {
        "experiment_name": run_context["experiment_name"],
        "tokenizer_family": settings["tokenizer_family"],
        "setting_label": settings["setting_label"],
        "run_mode": settings["run_mode"],
        "parent_experiment_name": settings["parent_experiment_name"],
        "mlm_probability": settings["mlm_probability"],
        "num_hidden_layers": settings["num_hidden_layers"],
        "attention_heads": settings["attention_heads"],
        "hidden_size": settings["hidden_size"],
        "intermediate_size": settings["intermediate_size"],
        "batch_size": settings["batch_size"],
        "learning_rate": settings["learning_rate"],
        "weight_decay": settings["weight_decay"],
        "epochs": settings["epochs"],
        "early_stopping_patience": settings["early_stopping_patience"],
        "checkpoint_save_strategy": settings["checkpoint_save_strategy"],
        "checkpoint_save_steps": settings["checkpoint_save_steps"],
        "evaluation_strategy": settings["evaluation_strategy"],
        "evaluation_steps": settings["evaluation_steps"],
        "early_checkpoint_save_end_step": settings["early_checkpoint_save_end_step"],
        "early_checkpoint_save_interval": settings["early_checkpoint_save_interval"],
        "save_total_limit": settings["save_total_limit"],
        "tokenizer_dir": str(paths["tokenizer_dir"]),
        "tokenized_dataset_dir": str(paths["tokenized_dataset_dir"]),
        "checkpoint_dir": str(paths["checkpoint_dir"]),
        "results_dir": str(paths["checkpoint_dir"]),
        "notebook_used": run_context["notebook_used"],
        "git_commit": run_context["git_commit"],
        "run_status": status,
        "notes": "",
    }


def write_pretraining_run_state(
    run_context: dict[str, object],
    metadata_payload: dict[str, object],
    status: str,
    extra_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Update metadata JSON and the run index for one run state."""

    metadata_payload["run_status"] = status
    if extra_metadata:
        metadata_payload.update(_to_json_safe(extra_metadata))

    save_json(metadata_payload, run_context["paths"]["experiment_metadata_path"])
    upsert_run_record(
        str(run_context["paths"]["run_index_path"]),
        _build_pretraining_run_record(run_context, status=status),
    )
    return metadata_payload


def prepare_pretraining_run(
    project_root: str | Path,
    repo_dir: str | Path,
    tokenizer_family: str,
    setting_label: str,
    run_mode: str,
    parent_experiment_name: str | None,
    resume_source_dir: str | Path | None,
    mlm_probability: float,
    num_hidden_layers: int,
    attention_heads: int,
    hidden_size: int,
    intermediate_size: int,
    max_position_embeddings: int,
    batch_size: int,
    weight_decay: float,
    checkpoint_save_strategy: str,
    checkpoint_save_steps: int | None,
    evaluation_strategy: str | None,
    evaluation_steps: int | None,
    early_checkpoint_save_end_step: int | None,
    early_checkpoint_save_interval: int | None,
    save_total_limit: int,
    early_stopping_patience: int,
    logging_steps: int,
    random_seed: int | None,
    initial_epochs: int,
    continuation_epochs: int,
    base_learning_rate: float,
    continuation_learning_rate: float,
    notebook_used: str = NOTEBOOK_PATH,
) -> dict[str, object]:
    """Validate notebook settings, build output paths, and register the run."""

    project_root = require_existing_path(project_root, "Project root")
    paths = build_pretraining_paths(
        project_root=project_root,
        tokenizer_family=tokenizer_family,
        setting_label=setting_label,
    )
    require_existing_path(paths["tokenizer_dir"], "Tokenizer directory")
    require_existing_path(paths["tokenized_dataset_dir"], "Tokenized dataset directory")

    normalized_run_mode = _normalize_run_mode(run_mode)
    training_schedule = _resolve_training_schedule(
        run_mode=normalized_run_mode,
        resume_source_dir=resume_source_dir,
        initial_epochs=initial_epochs,
        continuation_epochs=continuation_epochs,
        base_learning_rate=base_learning_rate,
        continuation_learning_rate=continuation_learning_rate,
    )
    checkpoint_schedule = _resolve_checkpoint_schedule(
        checkpoint_save_strategy=checkpoint_save_strategy,
        checkpoint_save_steps=checkpoint_save_steps,
        evaluation_strategy=evaluation_strategy,
        evaluation_steps=evaluation_steps,
        early_checkpoint_save_end_step=early_checkpoint_save_end_step,
        early_checkpoint_save_interval=early_checkpoint_save_interval,
        load_best_model_at_end=True,
    )

    normalized_resume_source = None
    resolved_parent_experiment_name = parent_experiment_name
    if normalized_run_mode != "fresh":
        normalized_resume_source = _validate_resume_source(
            run_mode=normalized_run_mode,
            resume_source_dir=resume_source_dir,
        )
        derived_parent_name = _derive_parent_experiment_name(normalized_resume_source)

        if parent_experiment_name:
            sanitized_parent_name = _sanitize_experiment_name(
                parent_experiment_name,
                "PARENT_EXPERIMENT_NAME",
            )
            if sanitized_parent_name != derived_parent_name:
                raise ValueError(
                    "PARENT_EXPERIMENT_NAME does not match the experiment implied "
                    f"by RESUME_SOURCE_DIR: {sanitized_parent_name} vs "
                    f"{derived_parent_name}"
                )
            resolved_parent_experiment_name = sanitized_parent_name
        else:
            resolved_parent_experiment_name = derived_parent_name

        _validate_resume_architecture(
            resume_source_dir=normalized_resume_source,
            num_hidden_layers=num_hidden_layers,
            attention_heads=attention_heads,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        _validate_resume_location_and_metadata(
            resume_source_dir=normalized_resume_source,
            checkpoint_root=paths["checkpoint_root"],
            tokenizer_dir=paths["tokenizer_dir"],
            tokenized_dataset_dir=paths["tokenized_dataset_dir"],
            tokenizer_family=tokenizer_family,
            setting_label=setting_label,
        )

    resolved_random_seed = resolve_random_seed(random_seed)
    git_commit = _get_git_commit(repo_dir)
    base_experiment_name = _build_base_experiment_name(
        run_mode=normalized_run_mode,
        parent_experiment_name=resolved_parent_experiment_name,
        mlm_probability=mlm_probability,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        learning_rate=float(training_schedule["learning_rate"]),
        epochs=int(training_schedule["epochs"]),
        setting_label=setting_label,
    )

    checkpoint_dir = _resolve_unique_directory(paths["checkpoint_root"] / base_experiment_name)
    best_model_dir = checkpoint_dir / "best_model"
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    log_dir = checkpoint_dir / "logs"
    experiment_metadata_path = checkpoint_dir / "experiment_metadata.json"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    paths["run_index_path"].parent.mkdir(parents=True, exist_ok=True)

    full_paths = {
        **paths,
        "checkpoint_dir": checkpoint_dir,
        "best_model_dir": best_model_dir,
        "trainer_state_path": trainer_state_path,
        "log_dir": log_dir,
        "experiment_metadata_path": experiment_metadata_path,
    }

    settings = {
        "run_mode": normalized_run_mode,
        "parent_experiment_name": resolved_parent_experiment_name,
        "resume_source_dir": (
            str(normalized_resume_source) if normalized_resume_source else None
        ),
        "tokenizer_family": tokenizer_family,
        "setting_label": setting_label,
        "mlm_probability": float(mlm_probability),
        "num_hidden_layers": int(num_hidden_layers),
        "attention_heads": int(attention_heads),
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size),
        "max_position_embeddings": int(max_position_embeddings),
        "batch_size": int(batch_size),
        "learning_rate": float(training_schedule["learning_rate"]),
        "weight_decay": float(weight_decay),
        "epochs": int(training_schedule["epochs"]),
        "early_stopping_patience": int(early_stopping_patience),
        "checkpoint_save_strategy": checkpoint_schedule["checkpoint_save_strategy"],
        "checkpoint_save_steps": checkpoint_schedule["checkpoint_save_steps"],
        "evaluation_strategy": checkpoint_schedule["evaluation_strategy"],
        "evaluation_steps": checkpoint_schedule["evaluation_steps"],
        "early_checkpoint_save_end_step": checkpoint_schedule[
            "early_checkpoint_save_end_step"
        ],
        "early_checkpoint_save_interval": checkpoint_schedule[
            "early_checkpoint_save_interval"
        ],
        "save_total_limit": int(save_total_limit),
        "load_best_model_at_end": checkpoint_schedule["load_best_model_at_end"],
        "logging_steps": int(logging_steps),
        "random_seed": int(resolved_random_seed),
        "initial_epochs": int(initial_epochs),
        "continuation_epochs": int(continuation_epochs),
        "base_learning_rate": float(base_learning_rate),
        "continuation_learning_rate": float(continuation_learning_rate),
    }

    metadata_payload = {
        "experiment_name": checkpoint_dir.name,
        "notebook_used": notebook_used,
        "git_commit": git_commit,
        "vault_routing": {
            "tokenizer_dir": str(full_paths["tokenizer_dir"]),
            "tokenized_dataset_dir": str(full_paths["tokenized_dataset_dir"]),
            "checkpoint_dir": str(full_paths["checkpoint_dir"]),
            "best_model_dir": str(full_paths["best_model_dir"]),
            "run_index_path": str(full_paths["run_index_path"]),
        },
        "live_hyperparameters": settings.copy(),
        "run_status": "configured",
    }

    run_context = {
        "experiment_name": checkpoint_dir.name,
        "notebook_used": notebook_used,
        "git_commit": git_commit,
        "resolved_random_seed": int(resolved_random_seed),
        "paths": full_paths,
        "settings": settings,
        "metadata_payload": metadata_payload,
    }

    write_pretraining_run_state(
        run_context=run_context,
        metadata_payload=metadata_payload,
        status="configured",
    )
    return run_context


def load_pretraining_tokenizer(tokenizer_dir: str | Path) -> dict[str, object]:
    """Load the saved tokenizer and report the key token IDs used for MLM."""

    tokenizer = load_fast_tokenizer(tokenizer_dir)
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer is missing a pad token id.")
    if tokenizer.mask_token_id is None:
        raise ValueError("The tokenizer is missing a mask token id.")

    return {
        "tokenizer": tokenizer,
        "vocab_size": int(len(tokenizer)),
        "pad_token_id": int(tokenizer.pad_token_id),
        "mask_token_id": int(tokenizer.mask_token_id),
    }


def _load_saved_dataset(dataset_path: str | Path) -> dict[str, torch.Tensor]:
    """Load one tokenized dataset tensor bundle from notebook 03."""

    dataset_path = require_existing_path(dataset_path, "Tokenized dataset file")
    dataset_dict = torch.load(dataset_path)
    required_keys = {"input_ids", "attention_mask"}
    missing_keys = required_keys - set(dataset_dict)
    if missing_keys:
        raise ValueError(
            f"Dataset file is missing required keys {sorted(missing_keys)}: {dataset_path}"
        )
    return dataset_dict


def load_pretraining_datasets(
    tokenized_dataset_dir: str | Path,
    max_position_embeddings: int,
) -> dict[str, object]:
    """Load the train and validation tensors used for notebook 04 training."""

    tokenized_dataset_dir = require_existing_path(
        tokenized_dataset_dir,
        "Tokenized dataset directory",
    )
    train_path = tokenized_dataset_dir / "train_dataset.pt"
    val_path = tokenized_dataset_dir / "val_dataset.pt"
    summary_path = tokenized_dataset_dir / "preprocessing_summary.json"

    raw_train = _load_saved_dataset(train_path)
    raw_val = _load_saved_dataset(val_path)
    train_dataset = GlycanMLMDataset(raw_train)
    val_dataset = GlycanMLMDataset(raw_val)

    if train_dataset.input_ids.ndim != 2:
        raise ValueError(
            "The saved training tensor should be two-dimensional: "
            f"{tuple(train_dataset.input_ids.shape)}"
        )

    sequence_width = int(train_dataset.input_ids.shape[1])
    if sequence_width > int(max_position_embeddings):
        raise ValueError(
            "MAX_POSITION_EMBEDDINGS is smaller than the tokenized sequence width: "
            f"{max_position_embeddings} < {sequence_width}"
        )

    preprocessing_summary = {}
    if summary_path.exists():
        preprocessing_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_dataset_path": train_path,
        "val_dataset_path": val_path,
        "preprocessing_summary_path": summary_path,
        "preprocessing_summary": preprocessing_summary,
        "train_size": int(len(train_dataset)),
        "val_size": int(len(val_dataset)),
        "sequence_width": sequence_width,
    }


def initialize_roberta_mlm_model(
    run_mode: str,
    resume_source_dir: str | Path | None,
    vocab_size: int,
    max_position_embeddings: int,
    num_hidden_layers: int,
    attention_heads: int,
    hidden_size: int,
    intermediate_size: int,
    pad_token_id: int,
) -> dict[str, object]:
    """Build the MLM model for a fresh run or load a saved continuation model."""

    model_config = RobertaConfig(
        vocab_size=int(vocab_size),
        max_position_embeddings=int(max_position_embeddings),
        num_hidden_layers=int(num_hidden_layers),
        num_attention_heads=int(attention_heads),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        pad_token_id=int(pad_token_id),
        type_vocab_size=1,
    )

    if run_mode in {"fresh", "resume_checkpoint"}:
        model = RobertaForMaskedLM(model_config)
    elif run_mode == "continue_best_model":
        model = RobertaForMaskedLM.from_pretrained(str(resume_source_dir))
        model_config = model.config
    else:
        raise ValueError(f"Unsupported RUN_MODE: {run_mode}")

    return {
        "config": model_config,
        "model": model,
        "total_trainable_parameters": count_trainable_parameters(model),
    }


def count_trainable_parameters(model) -> int:
    """Return the total number of trainable model parameters."""

    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )


def build_training_components(
    tokenizer,
    checkpoint_dir: str | Path,
    mlm_probability: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    weight_decay: float,
    checkpoint_save_strategy: str,
    checkpoint_save_steps: int | None,
    evaluation_strategy: str,
    evaluation_steps: int | None,
    early_checkpoint_save_end_step: int | None,
    early_checkpoint_save_interval: int | None,
    save_total_limit: int,
    early_stopping_patience: int,
    logging_steps: int,
    random_seed: int,
) -> dict[str, object]:
    """Create the MLM collator and Hugging Face training arguments."""

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=float(mlm_probability),
    )
    fp16_enabled = bool(torch.cuda.is_available())

    training_argument_kwargs = {
        "output_dir": str(checkpoint_dir),
        "eval_strategy": str(evaluation_strategy),
        "save_strategy": str(checkpoint_save_strategy),
        "learning_rate": float(learning_rate),
        "per_device_train_batch_size": int(batch_size),
        "per_device_eval_batch_size": int(batch_size),
        "num_train_epochs": int(epochs),
        "weight_decay": float(weight_decay),
        "save_total_limit": int(save_total_limit),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": int(logging_steps),
        "disable_tqdm": True,
        "report_to": "none",
        "seed": int(random_seed),
        "data_seed": int(random_seed),
        "fp16": fp16_enabled,
    }
    if checkpoint_save_strategy == "steps":
        training_argument_kwargs["save_steps"] = int(checkpoint_save_steps)
    if evaluation_strategy == "steps":
        training_argument_kwargs["eval_steps"] = int(evaluation_steps)

    training_args = TrainingArguments(**training_argument_kwargs)

    return {
        "data_collator": data_collator,
        "training_args": training_args,
        "fp16_enabled": fp16_enabled,
        "trainer_callbacks": build_pretraining_trainer_callbacks(
            early_stopping_patience=early_stopping_patience,
            early_checkpoint_save_end_step=early_checkpoint_save_end_step,
            early_checkpoint_save_interval=early_checkpoint_save_interval,
        ),
    }


class EarlyStepCheckpointCallback(TrainerCallback):
    """Save dense early checkpoints before switching to the regular cadence."""

    def __init__(
        self,
        end_step: int | None,
        step_interval: int | None,
    ) -> None:
        self.end_step = int(end_step) if end_step is not None else None
        self.step_interval = int(step_interval) if step_interval is not None else None

    def on_step_end(self, args, state, control, **kwargs):
        if self.end_step is None or self.step_interval is None:
            return control

        current_step = int(state.global_step)
        if (
            current_step > 0
            and current_step <= self.end_step
            and current_step % self.step_interval == 0
        ):
            control.should_save = True
        return control


def build_pretraining_trainer_callbacks(
    early_stopping_patience: int,
    early_checkpoint_save_end_step: int | None,
    early_checkpoint_save_interval: int | None,
) -> list[TrainerCallback]:
    """Build the trainer callbacks used by notebook 04."""

    callbacks: list[TrainerCallback] = [
        EarlyStoppingCallback(early_stopping_patience=int(early_stopping_patience)),
    ]
    if early_checkpoint_save_end_step is not None:
        callbacks.append(
            EarlyStepCheckpointCallback(
                end_step=early_checkpoint_save_end_step,
                step_interval=early_checkpoint_save_interval,
            )
        )
    return callbacks


def patch_transformers_tqdm_for_plain_text() -> None:
    """Force plain-text progress bars so saved notebooks stay GitHub-friendly."""

    from tqdm.std import tqdm as plain_tqdm

    import tqdm.auto as tqdm_auto

    tqdm_auto.tqdm = plain_tqdm

    try:
        import tqdm.notebook as tqdm_notebook
    except Exception:
        tqdm_notebook = None

    if tqdm_notebook is not None:
        tqdm_notebook.tqdm = plain_tqdm

    try:
        import transformers.modeling_utils as modeling_utils
    except Exception:
        modeling_utils = None

    if modeling_utils is not None:
        modeling_utils.tqdm = plain_tqdm

    try:
        import transformers.trainer as trainer_module
    except Exception:
        trainer_module = None

    if trainer_module is not None:
        trainer_module.tqdm = plain_tqdm
