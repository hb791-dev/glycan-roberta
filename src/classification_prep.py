"""Helpers for preparing multi-label glycan classification datasets.

This module keeps the accession/label joining work out of notebook 09 so the
notebook can stay focused on configuration, interpretation, and quick displays.

The downstream task described by the professor is multi-label classification:
- sequences come from the accession-aware compact-IUPAC corpus
- labels come from the PyGly ``classification.tsv`` export
- only ``Source == GlycoMotif`` and ``Level == GlycanSubtype`` rows are used
- one glycan accession may carry multiple subtype labels

The helpers below build a clean labeled dataframe, attach the existing
train/validation/test split assignment by exact sequence match, summarize label
coverage, and save the resulting tables for later fine-tuning notebooks.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping


ACCESSION_COLUMN = "glycan_id"
SEQUENCE_COLUMN = "sequence"
LABEL_LIST_COLUMN = "labels"
LABEL_COUNT_COLUMN = "num_labels"
SPLIT_COLUMN = "split"


def _require_columns(dataframe: "pd.DataFrame", required_columns: list[str], frame_name: str) -> None:
    """Raise a readable error when a dataframe is missing required columns."""
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{frame_name} is missing required columns: {missing_columns}")


def _clean_string_series(series: "pd.Series") -> "pd.Series":
    """Return one string-normalized pandas Series for predictable joins."""
    return series.fillna("").map(str).map(str.strip)


def _serialize_labels(label_values: list[str]) -> str:
    """Return a JSON string so saved CSV rows keep label order and punctuation."""
    return json.dumps(label_values, ensure_ascii=True)


def load_accession_reference_corpus(accession_reference_path: str | Path) -> "pd.DataFrame":
    """Load the accession-aware sequence corpus and normalize its key columns.

    The accession-aware corpus is the bridge between the original no-``+aldi``
    training data and the GlyTouCan-keyed classification labels. This loader
    keeps the most important columns predictable so downstream joins stay simple.
    """
    accession_reference_path = Path(accession_reference_path)
    accession_df = pd.read_csv(accession_reference_path)

    required_columns = [ACCESSION_COLUMN]
    if "normalized_iupac" in accession_df.columns:
        sequence_source_column = "normalized_iupac"
    elif SEQUENCE_COLUMN in accession_df.columns:
        sequence_source_column = SEQUENCE_COLUMN
    else:
        raise ValueError(
            "Accession reference corpus must include either 'normalized_iupac' or 'sequence'."
        )

    _require_columns(accession_df, required_columns, "accession_df")

    cleaned_df = accession_df.copy()
    cleaned_df[ACCESSION_COLUMN] = _clean_string_series(cleaned_df[ACCESSION_COLUMN])
    cleaned_df[SEQUENCE_COLUMN] = _clean_string_series(cleaned_df[sequence_source_column])

    if cleaned_df[ACCESSION_COLUMN].eq("").any():
        raise ValueError("Accession reference corpus contains blank glycan IDs.")

    if cleaned_df[SEQUENCE_COLUMN].eq("").any():
        raise ValueError("Accession reference corpus contains blank sequences.")

    # Keep each sequence/accession pair once so later joins and summaries reflect
    # real glycans rather than accidental duplicated rows.
    cleaned_df = cleaned_df.drop_duplicates(subset=[ACCESSION_COLUMN, SEQUENCE_COLUMN]).reset_index(drop=True)
    return cleaned_df


def load_filtered_classification_table(classification_tsv_path: str | Path) -> "pd.DataFrame":
    """Load ``classification.tsv`` and keep only the professor-requested rows.

    The raw export mixes multiple label levels and sources together. For this
    task we only want GlycoMotif-backed glycan subtype labels, because that is
    the exact target definition described in the project feedback.
    """
    classification_tsv_path = Path(classification_tsv_path)
    classification_df = pd.read_csv(classification_tsv_path, sep="\t")
    _require_columns(
        classification_df,
        ["GlyTouCanAccession", "Level", "Classification", "Source"],
        "classification_df",
    )

    filtered_df = classification_df.copy()
    filtered_df["GlyTouCanAccession"] = _clean_string_series(filtered_df["GlyTouCanAccession"])
    filtered_df["Level"] = _clean_string_series(filtered_df["Level"])
    filtered_df["Classification"] = _clean_string_series(filtered_df["Classification"])
    filtered_df["Source"] = _clean_string_series(filtered_df["Source"])

    filtered_df = filtered_df.loc[
        (filtered_df["Source"] == "GlycoMotif")
        & (filtered_df["Level"] == "GlycanSubtype")
        & filtered_df["GlyTouCanAccession"].ne("")
        & filtered_df["Classification"].ne("")
    ].copy()

    # The exported table can include repeated rows for the same accession/label
    # pair. Removing duplicates here keeps label counts honest.
    filtered_df = filtered_df.drop_duplicates(
        subset=["GlyTouCanAccession", "Classification"]
    ).reset_index(drop=True)
    return filtered_df


def aggregate_subtype_labels_by_accession(classification_df: "pd.DataFrame") -> "pd.DataFrame":
    """Collapse one-row-per-label data into one-row-per-accession label sets."""
    _require_columns(
        classification_df,
        ["GlyTouCanAccession", "Classification"],
        "classification_df",
    )

    grouped_rows: list[dict[str, object]] = []

    for accession, group_df in classification_df.groupby("GlyTouCanAccession", sort=True):
        # Sorting label names makes the saved CSV deterministic and easier to diff.
        label_values = sorted(group_df["Classification"].tolist())
        grouped_rows.append(
            {
                ACCESSION_COLUMN: accession,
                LABEL_LIST_COLUMN: label_values,
                "labels_json": _serialize_labels(label_values),
                LABEL_COUNT_COLUMN: len(label_values),
            }
        )

    return pd.DataFrame(grouped_rows)


def join_sequences_with_labels(
    accession_df: "pd.DataFrame",
    accession_label_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Attach subtype label sets to the accession-aware sequence table.

    This join happens by GlyTouCan accession, not by sequence text. The whole
    reason we built the accession-aware corpus is that the label source is keyed
    by accession rather than by raw compact-IUPAC string alone.
    """
    _require_columns(accession_df, [ACCESSION_COLUMN, SEQUENCE_COLUMN], "accession_df")
    _require_columns(accession_label_df, [ACCESSION_COLUMN, LABEL_LIST_COLUMN], "accession_label_df")

    joined_df = accession_df.merge(accession_label_df, on=ACCESSION_COLUMN, how="left")
    joined_df[LABEL_LIST_COLUMN] = joined_df[LABEL_LIST_COLUMN].apply(
        lambda value: value if isinstance(value, list) else []
    )
    joined_df["labels_json"] = joined_df[LABEL_LIST_COLUMN].map(_serialize_labels)
    joined_df[LABEL_COUNT_COLUMN] = joined_df[LABEL_LIST_COLUMN].map(len)
    joined_df["has_labels"] = joined_df[LABEL_COUNT_COLUMN] > 0
    return joined_df


def _load_one_split_file(split_path: str | Path) -> list[str]:
    """Load one split text file as an ordered list of non-empty sequences."""
    split_path = Path(split_path)
    with split_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_split_sequence_lookup(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
) -> dict[str, str]:
    """Build a sequence-to-split lookup from the existing Drive split files.

    Split assignment is sequence-based because the original project splits were
    created before the accession-aware table existed. Matching on exact sequence
    text lets the downstream classification branch reuse the same held-out sets.
    """
    split_lookup: dict[str, str] = {}

    for split_name, split_path in (
        ("train", train_path),
        ("val", val_path),
        ("test", test_path),
    ):
        split_sequences = _load_one_split_file(split_path)
        for sequence in split_sequences:
            if sequence in split_lookup:
                existing_split = split_lookup[sequence]
                raise ValueError(
                    f"Sequence appears in multiple split files: {existing_split!r} and {split_name!r}."
                )
            split_lookup[sequence] = split_name

    return split_lookup


def assign_splits_by_sequence(
    labeled_df: "pd.DataFrame",
    split_lookup: "Mapping[str, str]",
) -> "pd.DataFrame":
    """Attach train/validation/test split labels by exact sequence match."""
    _require_columns(labeled_df, [SEQUENCE_COLUMN], "labeled_df")

    split_df = labeled_df.copy()
    split_df[SPLIT_COLUMN] = split_df[SEQUENCE_COLUMN].map(split_lookup).fillna("")
    split_df["matched_existing_split"] = split_df[SPLIT_COLUMN].ne("")
    return split_df


def build_label_vocabulary(
    labeled_with_split_df: "pd.DataFrame",
    training_split_name: str = "train",
) -> "pd.DataFrame":
    """Create a stable label vocabulary plus support counts across splits.

    The vocabulary is derived from every labeled glycan in the prepared dataset,
    while split-specific support columns make it easy to spot labels that never
    appear in training and therefore cannot be learned by a classifier.
    """
    _require_columns(
        labeled_with_split_df,
        [LABEL_LIST_COLUMN, SPLIT_COLUMN, LABEL_COUNT_COLUMN],
        "labeled_with_split_df",
    )

    total_counter: Counter[str] = Counter()
    per_split_counters: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    for row in labeled_with_split_df.itertuples(index=False):
        label_values = list(getattr(row, LABEL_LIST_COLUMN))
        split_name = str(getattr(row, SPLIT_COLUMN))
        total_counter.update(label_values)
        if split_name in per_split_counters:
            per_split_counters[split_name].update(label_values)

    ordered_labels = sorted(
        total_counter,
        key=lambda label_name: (-total_counter[label_name], label_name),
    )

    vocabulary_rows = []
    for label_index, label_name in enumerate(ordered_labels):
        support_train = int(per_split_counters["train"][label_name])
        vocabulary_rows.append(
            {
                "label_id": label_index,
                "label_name": label_name,
                "support_total": int(total_counter[label_name]),
                "support_train": support_train,
                "support_val": int(per_split_counters["val"][label_name]),
                "support_test": int(per_split_counters["test"][label_name]),
                "present_in_training_split": support_train > 0,
                "missing_from_training_split": support_train == 0,
                "training_split_name": training_split_name,
            }
        )

    return pd.DataFrame(vocabulary_rows)


def summarize_classification_dataset(
    labeled_with_split_df: "pd.DataFrame",
    label_vocabulary_df: "pd.DataFrame",
) -> dict[str, object]:
    """Build compact summary tables for notebook display and saved reports."""
    _require_columns(
        labeled_with_split_df,
        [ACCESSION_COLUMN, SEQUENCE_COLUMN, LABEL_COUNT_COLUMN, SPLIT_COLUMN, "matched_existing_split"],
        "labeled_with_split_df",
    )
    _require_columns(
        label_vocabulary_df,
        ["label_name", "support_total", "present_in_training_split"],
        "label_vocabulary_df",
    )

    dataset_summary_rows = [
        {
            "metric": "total_accession_rows",
            "value": int(len(labeled_with_split_df)),
        },
        {
            "metric": "rows_with_at_least_one_label",
            "value": int((labeled_with_split_df[LABEL_COUNT_COLUMN] > 0).sum()),
        },
        {
            "metric": "rows_without_labels",
            "value": int((labeled_with_split_df[LABEL_COUNT_COLUMN] == 0).sum()),
        },
        {
            "metric": "rows_matched_to_existing_split",
            "value": int(labeled_with_split_df["matched_existing_split"].sum()),
        },
        {
            "metric": "rows_not_matched_to_existing_split",
            "value": int((~labeled_with_split_df["matched_existing_split"]).sum()),
        },
        {
            "metric": "unique_subtype_labels",
            "value": int(len(label_vocabulary_df)),
        },
        {
            "metric": "labels_missing_from_train",
            "value": int(label_vocabulary_df["missing_from_training_split"].sum()),
        },
    ]
    dataset_summary_df = pd.DataFrame(dataset_summary_rows)

    split_summary_df = (
        labeled_with_split_df.groupby(SPLIT_COLUMN, dropna=False, sort=False)
        .agg(
            num_rows=(ACCESSION_COLUMN, "count"),
            num_labeled_rows=(LABEL_COUNT_COLUMN, lambda values: int((values > 0).sum())),
            num_unlabeled_rows=(LABEL_COUNT_COLUMN, lambda values: int((values == 0).sum())),
            mean_labels_per_row=(LABEL_COUNT_COLUMN, "mean"),
            median_labels_per_row=(LABEL_COUNT_COLUMN, "median"),
        )
        .reset_index()
    )

    label_coverage_summary_df = pd.DataFrame(
        [
            {
                "coverage_group": "all_labels",
                "num_labels": int(len(label_vocabulary_df)),
            },
            {
                "coverage_group": "labels_present_in_train",
                "num_labels": int(label_vocabulary_df["present_in_training_split"].sum()),
            },
            {
                "coverage_group": "labels_missing_from_train",
                "num_labels": int(label_vocabulary_df["missing_from_training_split"].sum()),
            },
        ]
    )

    missing_train_label_df = label_vocabulary_df.loc[
        label_vocabulary_df["missing_from_training_split"]
    ].copy()

    summary_json = {
        row["metric"]: int(row["value"])
        for row in dataset_summary_rows
    }
    summary_json["split_counts"] = {
        str(row[SPLIT_COLUMN]): int(row["num_rows"])
        for _, row in split_summary_df.iterrows()
    }

    return {
        "dataset_summary_df": dataset_summary_df,
        "split_summary_df": split_summary_df,
        "label_coverage_summary_df": label_coverage_summary_df,
        "missing_train_label_df": missing_train_label_df,
        "summary_json": summary_json,
    }


def save_classification_prep_outputs(
    labeled_df: "pd.DataFrame",
    labeled_with_split_df: "pd.DataFrame",
    label_vocabulary_df: "pd.DataFrame",
    dataset_summary_df: "pd.DataFrame",
    split_summary_df: "pd.DataFrame",
    label_coverage_summary_df: "pd.DataFrame",
    missing_train_label_df: "pd.DataFrame",
    summary_json: dict[str, object],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write the derived CSV and JSON outputs used by notebook 09 and later steps."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_output_df = labeled_df.copy()
    labeled_output_df["labels_json"] = labeled_output_df[LABEL_LIST_COLUMN].map(_serialize_labels)

    labeled_with_split_output_df = labeled_with_split_df.copy()
    labeled_with_split_output_df["labels_json"] = labeled_with_split_output_df[LABEL_LIST_COLUMN].map(
        _serialize_labels
    )

    train_output_df = labeled_with_split_output_df.loc[
        labeled_with_split_output_df[SPLIT_COLUMN] == "train"
    ].copy()
    val_output_df = labeled_with_split_output_df.loc[
        labeled_with_split_output_df[SPLIT_COLUMN] == "val"
    ].copy()
    test_output_df = labeled_with_split_output_df.loc[
        labeled_with_split_output_df[SPLIT_COLUMN] == "test"
    ].copy()

    output_paths = {
        "labeled_glycans_path": str(output_dir / "labeled_glycans.csv"),
        "labeled_glycans_with_split_path": str(output_dir / "labeled_glycans_with_split.csv"),
        "train_classification_path": str(output_dir / "train_classification.csv"),
        "val_classification_path": str(output_dir / "val_classification.csv"),
        "test_classification_path": str(output_dir / "test_classification.csv"),
        "label_vocabulary_path": str(output_dir / "label_vocabulary.csv"),
        "dataset_summary_path": str(output_dir / "dataset_summary.csv"),
        "split_summary_path": str(output_dir / "split_summary.csv"),
        "label_coverage_summary_path": str(output_dir / "label_coverage_summary.csv"),
        "missing_train_labels_path": str(output_dir / "missing_train_labels.csv"),
        "classification_prep_summary_path": str(output_dir / "classification_prep_summary.json"),
    }

    labeled_output_df.to_csv(output_paths["labeled_glycans_path"], index=False)
    labeled_with_split_output_df.to_csv(output_paths["labeled_glycans_with_split_path"], index=False)
    train_output_df.to_csv(output_paths["train_classification_path"], index=False)
    val_output_df.to_csv(output_paths["val_classification_path"], index=False)
    test_output_df.to_csv(output_paths["test_classification_path"], index=False)
    label_vocabulary_df.to_csv(output_paths["label_vocabulary_path"], index=False)
    dataset_summary_df.to_csv(output_paths["dataset_summary_path"], index=False)
    split_summary_df.to_csv(output_paths["split_summary_path"], index=False)
    label_coverage_summary_df.to_csv(output_paths["label_coverage_summary_path"], index=False)
    missing_train_label_df.to_csv(output_paths["missing_train_labels_path"], index=False)
    Path(output_paths["classification_prep_summary_path"]).write_text(
        json.dumps(summary_json, indent=2),
        encoding="utf-8",
    )

    return output_paths


def run_classification_prep_pipeline(
    accession_reference_path: str | Path,
    classification_tsv_path: str | Path,
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run the full notebook-09 classification prep workflow end to end.

    This is the main entry point the Colab notebook can call. It returns the
    important dataframes for immediate display and also writes the derived files
    needed by the later fine-tuning and evaluation notebooks.
    """
    accession_df = load_accession_reference_corpus(accession_reference_path)
    filtered_classification_df = load_filtered_classification_table(classification_tsv_path)
    accession_label_df = aggregate_subtype_labels_by_accession(filtered_classification_df)

    joined_df = join_sequences_with_labels(accession_df, accession_label_df)
    labeled_df = joined_df.loc[joined_df["has_labels"]].copy().reset_index(drop=True)

    split_lookup = load_split_sequence_lookup(train_path, val_path, test_path)
    labeled_with_split_df = assign_splits_by_sequence(labeled_df, split_lookup)

    label_vocabulary_df = build_label_vocabulary(labeled_with_split_df)
    summary_bundle = summarize_classification_dataset(labeled_with_split_df, label_vocabulary_df)
    output_paths = save_classification_prep_outputs(
        labeled_df=labeled_df,
        labeled_with_split_df=labeled_with_split_df,
        label_vocabulary_df=label_vocabulary_df,
        dataset_summary_df=summary_bundle["dataset_summary_df"],
        split_summary_df=summary_bundle["split_summary_df"],
        label_coverage_summary_df=summary_bundle["label_coverage_summary_df"],
        missing_train_label_df=summary_bundle["missing_train_label_df"],
        summary_json=summary_bundle["summary_json"],
        output_dir=output_dir,
    )

    return {
        "accession_df": accession_df,
        "filtered_classification_df": filtered_classification_df,
        "accession_label_df": accession_label_df,
        "labeled_df": labeled_df,
        "labeled_with_split_df": labeled_with_split_df,
        "label_vocabulary_df": label_vocabulary_df,
        "dataset_summary_df": summary_bundle["dataset_summary_df"],
        "split_summary_df": summary_bundle["split_summary_df"],
        "label_coverage_summary_df": summary_bundle["label_coverage_summary_df"],
        "missing_train_label_df": summary_bundle["missing_train_label_df"],
        "summary_json": summary_bundle["summary_json"],
        "output_paths": output_paths,
    }
