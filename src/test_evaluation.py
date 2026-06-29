"""Helpers for held-out masked-token evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, PreTrainedTokenizerFast, pipeline


class GlycanDataset(Dataset):
    """Dataset wrapper for saved tokenized glycan tensors."""

    def __init__(self, dataset_dict):
        self.input_ids = dataset_dict["input_ids"]
        self.attention_mask = dataset_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


class MaskedEvalDataset(Dataset):
    """Dataset wrapper for deterministic masked evaluation examples."""

    def __init__(self, masked_dataset_dict):
        self.input_ids = masked_dataset_dict["input_ids"]
        self.attention_mask = masked_dataset_dict["attention_mask"]
        self.labels = masked_dataset_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def load_test_artifacts(model_dir: str, test_dataset_path: str, device):
    """Load the saved tokenizer, best model, and tokenized test dataset."""
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_dir).to(device)
    model.eval()

    raw_test = torch.load(test_dataset_path)
    test_dataset = GlycanDataset(raw_test)

    return tokenizer, model, test_dataset


def build_masked_test_dataset(
    test_dataset: GlycanDataset,
    tokenizer,
    mlm_probability: float = 0.15,
    seed: int = 42,
):
    """Build a deterministic masked version of the tokenized test set.

    The output mirrors Hugging Face MLM inputs:
    - ``input_ids`` contains explicit mask tokens at sampled positions
    - ``labels`` contains the original token ID only at masked positions
    - all unmasked positions are set to ``-100`` in ``labels``
    """
    input_ids = test_dataset.input_ids.clone()
    attention_mask = test_dataset.attention_mask.clone()
    labels = input_ids.clone()

    special_token_ids = {
        token_id
        for token_id in [
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.unk_token_id,
            tokenizer.mask_token_id,
        ]
        if token_id is not None
    }

    generator = torch.Generator().manual_seed(seed)
    non_special_mask = attention_mask.bool()
    for token_id in special_token_ids:
        non_special_mask &= input_ids != token_id

    random_matrix = torch.rand(input_ids.shape, generator=generator)
    masked_positions = (random_matrix < mlm_probability) & non_special_mask

    # Sequences with eligible tokens should contribute at least one masked
    # position so sequence-level metrics are defined consistently.
    for row_index in range(masked_positions.shape[0]):
        if masked_positions[row_index].sum() == 0:
            candidate_positions = torch.where(non_special_mask[row_index])[0]
            if len(candidate_positions) > 0:
                selected_position = candidate_positions[
                    torch.randint(len(candidate_positions), (1,), generator=generator)
                ]
                masked_positions[row_index, selected_position] = True

    labels[~masked_positions] = -100

    masked_input_ids = input_ids.clone()
    masked_input_ids[masked_positions] = tokenizer.mask_token_id

    return {
        "input_ids": masked_input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_masking_summary(masked_dataset_dict, mlm_probability: float, mask_seed: int) -> pd.DataFrame:
    """Summarize the deterministic masking used for held-out test evaluation."""
    labels = masked_dataset_dict["labels"]
    masked_positions = int((labels != -100).sum().item())
    test_sequences = int(labels.shape[0])

    return pd.DataFrame(
        {
            "metric": ["test_sequences", "masked_positions", "mlm_probability", "mask_seed"],
            "value": [test_sequences, masked_positions, mlm_probability, mask_seed],
        }
    )


def run_mlm_test_predictions(model, masked_dataset_dict, batch_size: int, device):
    """Run MLM predictions on the masked test set.

    Returns token-level ground truth and predictions at masked positions, along
    with sequence-level flags indicating whether all masked tokens in a sequence
    were recovered within top-1 or top-3 predictions.
    """
    eval_dataset = MaskedEvalDataset(masked_dataset_dict)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    all_true = []
    all_pred = []
    all_probs = []
    sequence_top1_flags = []
    sequence_top3_flags = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            top3_preds = torch.topk(probs, k=3, dim=-1).indices

            for row_index in range(input_ids.shape[0]):
                masked_positions = labels[row_index] != -100

                if masked_positions.sum() == 0:
                    continue

                true_tokens = labels[row_index][masked_positions]
                pred_tokens = preds[row_index][masked_positions]
                prob_tokens = probs[row_index][masked_positions]
                top3_tokens = top3_preds[row_index][masked_positions]

                all_true.append(true_tokens.cpu())
                all_pred.append(pred_tokens.cpu())
                all_probs.append(prob_tokens.cpu())

                token_top1_correct = pred_tokens == true_tokens
                token_top3_correct = (top3_tokens == true_tokens.unsqueeze(1)).any(dim=1)

                sequence_top1_flags.append(bool(token_top1_correct.all().item()))
                sequence_top3_flags.append(bool(token_top3_correct.all().item()))

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    y_probs = torch.cat(all_probs).numpy()

    return y_true, y_pred, y_probs, sequence_top1_flags, sequence_top3_flags


def top_k_accuracy(y_true, y_probs, k: int) -> float:
    """Compute token-level top-k accuracy from a probability matrix."""
    top_k = np.argpartition(y_probs, -k, axis=1)[:, -k:]
    correct = np.any(top_k == y_true[:, None], axis=1)
    return float(correct.mean())


def macro_specificity(y_true, y_pred) -> float:
    """Compute macro-averaged one-vs-rest specificity."""
    classes = np.unique(y_true)
    specificities = []

    for class_id in classes:
        true_binary = y_true == class_id
        pred_binary = y_pred == class_id

        tn = np.sum((~true_binary) & (~pred_binary))
        fp = np.sum((~true_binary) & pred_binary)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)

    return float(np.mean(specificities))


def compute_topk_metrics(y_true, y_probs, y_pred, sequence_top1_flags, sequence_top3_flags) -> dict:
    """Compute token-level and sequence-level top-k metrics."""
    return {
        "token_top1_accuracy": float((y_true == y_pred).mean()),
        "token_top3_accuracy": top_k_accuracy(y_true, y_probs, k=3),
        "sequence_top1_accuracy": float(np.mean(sequence_top1_flags)),
        "sequence_top3_accuracy": float(np.mean(sequence_top3_flags)),
    }


def compute_summary_classification_metrics(y_true, y_pred) -> dict:
    """Compute macro and weighted masked-token classification metrics."""
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "macro_sensitivity": float(macro_recall),
        "macro_specificity": macro_specificity(y_true, y_pred),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }


def compute_per_class_metrics(y_true, y_pred, tokenizer) -> pd.DataFrame:
    """Compute per-class precision, recall, F1, and support."""
    class_ids = np.unique(y_true)

    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=class_ids,
            average=None,
            zero_division=0,
        )
    )

    class_tokens = tokenizer.convert_ids_to_tokens(class_ids.tolist())

    return pd.DataFrame(
        {
            "token_id": class_ids,
            "token": class_tokens,
            "precision": per_class_precision,
            "recall": per_class_recall,
            "f1": per_class_f1,
            "support": per_class_support,
        }
    ).sort_values("support", ascending=False)


def plot_roc_for_selected_classes(y_true, y_probs, tokenizer, selected_tokens, save_path=None) -> pd.DataFrame:
    """Plot one-vs-rest ROC curves for a selected set of token classes."""
    rows = []
    plt.figure(figsize=(8, 6))

    for token in selected_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)

        if token_id is None or token_id == tokenizer.unk_token_id:
            continue

        binary_true = (y_true == token_id).astype(int)
        if binary_true.sum() == 0:
            continue

        token_scores = y_probs[:, token_id]
        fpr, tpr, _ = roc_curve(binary_true, token_scores)
        token_auc = auc(fpr, tpr)

        rows.append({"token": token, "auc": token_auc})
        plt.plot(fpr, tpr, label=f"{token} (AUC = {token_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves for selected tokens")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return pd.DataFrame(rows)


def plot_pr_for_selected_classes(y_true, y_probs, tokenizer, selected_tokens, save_path=None) -> pd.DataFrame:
    """Plot precision-recall curves for a selected set of token classes."""
    rows = []
    plt.figure(figsize=(8, 6))

    for token in selected_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)

        if token_id is None or token_id == tokenizer.unk_token_id:
            continue

        binary_true = (y_true == token_id).astype(int)
        if binary_true.sum() == 0:
            continue

        token_scores = y_probs[:, token_id]
        precision, recall, _ = precision_recall_curve(binary_true, token_scores)
        avg_precision = average_precision_score(binary_true, token_scores)

        rows.append({"token": token, "average_precision": avg_precision})
        plt.plot(recall, precision, label=f"{token} (AP = {avg_precision:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves for selected tokens")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return pd.DataFrame(rows)


def run_qualitative_probe(model, tokenizer, probe_sequences, device) -> pd.DataFrame:
    """Run fill-mask predictions for a small set of masked glycan examples."""
    unmasker = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=0 if str(device).startswith("cuda") else -1,
    )

    rows = []
    for sequence in probe_sequences:
        results = unmasker(sequence)
        for rank, result in enumerate(results[:5], start=1):
            rows.append(
                {
                    "masked_sequence": sequence,
                    "rank": rank,
                    "predicted_token": result["token_str"],
                    "score": result["score"],
                }
            )

    return pd.DataFrame(rows)


def build_test_summary_row(
    tokenizer_family: str,
    setting_label: str,
    experiment_name: str,
    metrics_dict: dict,
) -> pd.DataFrame:
    """Build a one-row summary table for one evaluated run."""
    row = {
        "tokenizer_family": tokenizer_family,
        "setting_label": setting_label,
        "experiment_name": experiment_name,
    }
    row.update(metrics_dict)
    return pd.DataFrame([row])


def get_core_probe_cases():
    """Return biological probe cases with tokenizer-specific masking targets."""
    return [
        {
            "probe_id": "glcnac_core",
            "concept": "GlcNAc in an N-glycan core context",
            "base_sequence": "Galb1-4GlcNAcb1-2Mana1-3Manb1-4GlcNAcb1-4GlcNAc",
            "biological_target": "GlcNAc",
            "tokenizer_targets": {
                "manual": {
                    "masked_sequence": "Galb1-4<mask>b1-2Mana1-3Manb1-4GlcNAcb1-4GlcNAc",
                    "expected_token": "GlcNAc",
                    "target_token_type": "residue",
                },
                "hybrid_char_bpe": {
                    "masked_sequence": "Galb1-4<mask>b1-2Mana1-3Manb1-4GlcNAcb1-4GlcNAc",
                    "expected_token": "GlcNAc",
                    "target_token_type": "merged_residue",
                },
                "byte_bpe": {
                    "masked_sequence": "Galb1-4<mask>b1-2Mana1-3Manb1-4GlcNAcb1-4GlcNAc",
                    "expected_token": "GlcNAc",
                    "target_token_type": "byte_bpe_residue",
                },
            },
        },
        {
            "probe_id": "gal_terminal",
            "concept": "Terminal Gal in antenna context",
            "base_sequence": "NeuAca2-3Galb1-4GlcNAcb1-2Mana1-3(Galb1-4GlcNAcb1-2Mana1-6)Manb1-4GlcNAc",
            "biological_target": "Gal",
            "tokenizer_targets": {
                "manual": {
                    "masked_sequence": "NeuAca2-3<mask>b1-4GlcNAcb1-2Mana1-3(Galb1-4GlcNAcb1-2Mana1-6)Manb1-4GlcNAc",
                    "expected_token": "Gal",
                    "target_token_type": "residue",
                },
                "hybrid_char_bpe": {
                    "masked_sequence": "NeuAca2-3<mask>b1-4GlcNAcb1-2Mana1-3(Galb1-4GlcNAcb1-2Mana1-6)Manb1-4GlcNAc",
                    "expected_token": "Gal",
                    "target_token_type": "merged_residue",
                },
                "byte_bpe": {
                    "masked_sequence": "NeuAca2-3<mask>b1-4GlcNAcb1-2Mana1-3(Galb1-4GlcNAcb1-2Mana1-6)Manb1-4GlcNAc",
                    "expected_token": "Gal",
                    "target_token_type": "byte_bpe_residue",
                },
            },
        },
        {
            "probe_id": "mannose_core",
            "concept": "Man in the mannose core",
            "base_sequence": "Mana1-6(Mana1-3)Manb1-4GlcNAcb1-4GlcNAc",
            "biological_target": "Man",
            "tokenizer_targets": {
                "manual": {
                    "masked_sequence": "Mana1-6(Mana1-3)<mask>b1-4GlcNAcb1-4GlcNAc",
                    "expected_token": "Man",
                    "target_token_type": "residue",
                },
                "hybrid_char_bpe": {
                    "masked_sequence": "Mana1-6(Mana1-3)<mask>b1-4GlcNAcb1-4GlcNAc",
                    "expected_token": "Man",
                    "target_token_type": "merged_residue",
                },
                "byte_bpe": {
                    "masked_sequence": "Mana1-6(Mana1-3)<mask>1-4GlcNAcb1-4GlcNAc",
                    "expected_token": "Manb",
                    "target_token_type": "byte_bpe_residue_variant",
                },
            },
        },
        {
            "probe_id": "fucose_branch",
            "concept": "Fuc in a fucosylated branch",
            "base_sequence": "Fuca1-3Galb1-4GlcNAcb1-6Man",
            "biological_target": "Fuc",
            "tokenizer_targets": {
                "manual": {
                    "masked_sequence": "<mask>a1-3Galb1-4GlcNAcb1-6Man",
                    "expected_token": "Fuc",
                    "target_token_type": "residue",
                },
                "hybrid_char_bpe": {
                    "masked_sequence": "<mask>a1-3Galb1-4GlcNAcb1-6Man",
                    "expected_token": "Fuc",
                    "target_token_type": "merged_residue",
                },
                "byte_bpe": {
                    "masked_sequence": "<mask>a1-3Galb1-4GlcNAcb1-6Man",
                    "expected_token": "Fuc",
                    "target_token_type": "byte_bpe_residue",
                },
            },
        },
        {
            "probe_id": "sialic_terminal",
            "concept": "Terminal sialic acid context",
            "base_sequence": "NeuAca2-3Galb1-4GlcNAc",
            "biological_target": "sialic_acid",
            "tokenizer_targets": {
                "manual": {
                    "masked_sequence": "<mask>a2-3Galb1-4GlcNAc",
                    "expected_token": "NeuAc",
                    "target_token_type": "residue",
                },
                "hybrid_char_bpe": {
                    "masked_sequence": "<mask>Aca2-3Galb1-4GlcNAc",
                    "expected_token": "Neu",
                    "target_token_type": "subword_residue",
                },
                "byte_bpe": {
                    "masked_sequence": "<mask>a2-3Galb1-4GlcNAc",
                    "expected_token": "NeuAc",
                    "target_token_type": "byte_bpe_residue",
                },
            },
        },
    ]


def get_tokenizer_specific_roc_pr_tokens():
    """Return ROC and PR token candidates that make sense for each tokenizer."""
    return {
        "manual": [
            "GlcNAc",
            "Gal",
            "Man",
            "Fuc",
            "NeuAc",
            "GalNAc",
            "Glc",
            "GlcA",
            "a1-3",
            "b1-4",
        ],
        "hybrid_char_bpe": [
            "GlcNAc",
            "Gal",
            "Man",
            "Fuc",
            "Neu",
            "Glc",
            "NAc",
            "b1-4",
            "b1-4GlcNAc",
            "Galb1-4GlcNAc",
        ],
        "byte_bpe": [
            "GlcNAc",
            "Gal",
            "Man",
            "Fuc",
            "NeuAc",
            "GalNAc",
            "Glc",
            "GlcA",
            "Galb",
            "GlcNAcb",
        ],
    }


def run_structured_qualitative_probe(model, tokenizer, tokenizer_family: str, probe_cases, device) -> pd.DataFrame:
    """Run fill-mask predictions for tokenizer-specific probe strings."""
    unmasker = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=0 if str(device).startswith("cuda") else -1,
    )

    family = tokenizer_family.lower().replace("-", "_")
    rows = []

    for case in probe_cases:
        target_info = case.get("tokenizer_targets", {}).get(family)

        if not target_info:
            continue

        masked_sequence = target_info["masked_sequence"]
        expected_token = target_info["expected_token"]
        target_token_type = target_info.get("target_token_type", "")

        results = unmasker(masked_sequence)

        for rank, result in enumerate(results[:5], start=1):
            predicted_token = result["token_str"].strip()

            rows.append(
                {
                    "probe_id": case["probe_id"],
                    "concept": case["concept"],
                    "base_sequence": case["base_sequence"],
                    "biological_target": case["biological_target"],
                    "masked_sequence": masked_sequence,
                    "expected_token": expected_token,
                    "target_token_type": target_token_type,
                    "rank": rank,
                    "predicted_token": predicted_token,
                    "score": result["score"],
                    "is_expected": predicted_token == expected_token,
                }
            )

    return pd.DataFrame(rows)
