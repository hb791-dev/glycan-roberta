# Glycan RoBERTa Project

This repository contains a workflow for pretraining and evaluating
RoBERTa-style masked-language models on glycan sequences.

The project compares four tokenizer strategies:

- `byte_bpe`
- `glyberta`
- `manual`
- `hybrid_char_bpe`

The workflow is organized as a notebook pipeline with helper scripts in
`src/`. Large artifacts such as dataset splits, trained tokenizers,
tokenized datasets, checkpoints, and evaluation outputs are stored outside the
repository in Google Drive.

## Project Split

This rebuild uses a split-storage workflow.

- GitHub repository:
  - notebooks
  - helper scripts
  - templates
- Google Drive:
  - raw data
  - train-validation-test splits
  - tokenizer artifacts
  - tokenized datasets
  - training checkpoints
  - validation outputs
  - test-set evaluation outputs

The intended Drive root is:

- `MyDrive/ProjectRoot/`

The notebooks assume a single Drive project root at `MyDrive/ProjectRoot/`.

## Repository Layout

```text
glycan-roberta/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_data_splitting.ipynb
│   ├── 02_tokenizer_generation/
│   │   ├── 02a_byte_bpe_gen.ipynb
│   │   ├── 02b_glyberta_gen.ipynb
│   │   ├── 02c_manual_gen.ipynb
│   │   └── 02d_hybrid_char_bpe_gen.ipynb
│   ├── 03_dataset_preprocessing.ipynb
│   ├── 04_roberta_pretraining.ipynb
│   ├── 05_validation_diagnostics.ipynb
│   ├── 06_test_set_evaluation.ipynb
│   ├── 06b_rarity_analysis.ipynb
│   ├── 07_similarity_analysis.ipynb
│   ├── 08_similarity_scaleup.ipynb
│   ├── 09_classification_dataset_prep.ipynb
│   └── 10_classification_finetuning.ipynb
├── public_reports/
│   ├── README.md
│   └── <tokenizer_family>/
│       └── <experiment_name>/
│           └── <run_label>/
├── src/
│   ├── classification_prep.py
│   ├── classification_training.py
│   ├── glycan_cartoons.py
│   ├── similarity.py
│   ├── similarity_core.py
│   ├── similarity_variants.py
│   ├── similarity_scaleup.py
│   ├── test_evaluation.py
│   ├── tokenizer_utils.py
│   └── training_diagnostics.py
└── templates/
    ├── experiment_metadata.example.json
    └── run_index.csv
```

## Expected Drive Layout

```text
MyDrive/ProjectRoot/
├── data/
│   ├── raw/
│   │   ├── raw_glycans_dataset_no_aldi.txt
│   │   ├── accession_reference_corpus.csv
│   │   └── classification.tsv
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│       ├── split_summary.csv
│       └── split_preview.csv
├── tokenizers/
│   ├── byte_bpe/
│   ├── glyberta/
│   ├── hybrid_char_bpe/
│   └── manual/
├── tokenized_datasets/
│   ├── byte_bpe/
│   ├── glyberta/
│   ├── hybrid_char_bpe/
│   └── manual/
├── checkpoints/
│   ├── byte_bpe/
│   ├── classification/
│   ├── glyberta/
│   ├── hybrid_char_bpe/
│   └── manual/
├── results/
│   ├── exploration/
│   ├── validation/
│   ├── test_evaluation/
│   ├── classification_finetuning/
│   ├── similarity/
│   ├── similarity_scaleup/
│   └── classification_prep/
└── registry/
    └── run_index.csv
```

## Notebook Workflow

### `00_data_exploration.ipynb`

Purpose:
- inspect the raw glycan dataset
- summarize sequence lengths
- save lightweight exploration outputs

Main outputs:
- dataset summary CSV
- example sequences CSV
- sequence-length distribution plot

### `01_data_splitting.ipynb`

Purpose:
- create the train-validation-test split from the raw dataset

Main outputs:
- `train.txt`
- `val.txt`
- `test.txt`
- `split_summary.csv`
- `split_preview.csv`

### `02a_byte_bpe_gen.ipynb`

Purpose:
- train the byte-level BPE tokenizer on the training split

Main outputs:
- tokenizer files
- merges and vocab
- inspection preview
- tokenizer configuration summary

### `02b_glyberta_gen.ipynb`

Purpose:
- train the GlyBERTa-style WordLevel tokenizer on the training split
- adapt the GlyBERTa glyco-letter idea to this project's compact glycan format
- isolate inline linkage text and branch markers before learning the vocabulary

Main outputs:
- tokenizer files
- vocab
- inspection preview
- tokenizer configuration summary

### `02c_manual_gen.ipynb`

Purpose:
- build the manual glycan tokenizer from the training split
- save a fixed vocabulary and matching Hugging Face tokenizer

Main outputs:
- tokenizer files
- vocab
- inspection preview
- tokenizer configuration summary

### `02d_hybrid_char_bpe_gen.ipynb`

Purpose:
- train a hybrid char-BPE tokenizer restricted to the character inventory
  present in the dataset
- leave more vocabulary space available for learned merges

Main outputs:
- tokenizer files
- merges and vocab
- inspection preview
- tokenizer configuration summary

### `03_dataset_preprocessing.ipynb`

Purpose:
- tokenize the train, validation, and test splits with one selected tokenizer
- save padded tensor datasets for training and evaluation

Main outputs:
- `train_dataset.pt`
- `val_dataset.pt`
- `test_dataset.pt`
- `tokenization_preview.csv`
- `preprocessing_summary.json`

### `04_roberta_pretraining.ipynb`

Purpose:
- pretrain a RoBERTa masked-language model on the tokenized training split
- evaluate on the validation split during training
- support fresh runs, checkpoint resumes, and continuation runs

Main outputs:
- checkpoint folders
- `best_model/`
- `trainer_state.json`
- `experiment_metadata.json`
- run-index updates

Notes:
- masking is dynamic and applied on the fly by the MLM data collator
- validation in this notebook is for training-time diagnostics, not final
  model comparison

### `05_validation_diagnostics.ipynb`

Purpose:
- review training and validation loss after notebook 4
- summarize best-epoch and final validation behavior for a run

Main outputs:
- `loss_curves.png`
- `loss_history.csv`
- `validation_summary.json`

### `06_test_set_evaluation.ipynb`

Purpose:
- evaluate a saved `best_model` on the held-out test split
- compute token-level, sequence-level, per-class, ROC, PR, and qualitative
  probe outputs

Main outputs:
- `masking_summary.csv`
- `test_summary.json`
- `test_summary_row.csv`
- `per_class_metrics.csv`
- `roc_curves.png`
- `roc_auc_summary.csv`
- `pr_curves.png`
- `pr_auc_summary.csv`
- `qualitative_probe_results.csv`

Notes:
- ROC and precision-recall plots use tokenizer-specific class selections
- qualitative probes are tokenizer-specific because token boundaries differ
  across tokenizers

### `06b_rarity_analysis.ipynb`

Purpose:
- inspect token-frequency behavior in the training data
- summarize which tokens are common versus rare
- support interpretation of why `macro` and `weighted` metrics can differ

Main outputs:
- token-frequency summary tables
- rare-token diagnostic tables
- rarity-oriented distribution plots

### `07_similarity_analysis.ipynb`

Purpose:
- compare glycan sequence embeddings from one saved model checkpoint at a time
- choose one `best_model/` folder in Drive in a single `MODEL_DIR` cell
- define small manual anchor-and-variant review sets directly in the notebook
- score anchor-to-variant cosine similarity and rank variants within each set
- inspect tokenization alongside similarity results
- summarize similarity distributions with histograms across all 9 variants in each anchor group
- make the overall within-anchor order explicit with a single 1-to-9 rank
- build HTML reports with glycan cartoons for anchors and variants

Main outputs:
- `variant_similarity_results.csv`
- `variant_tokenization_preview.csv`
- `variant_cartoon_manifest.csv`
- `variant_distribution_summary.csv`
- `variant_similarity_config.json`
- `anchor_matrices/`
- `histograms/`
- `html/index.html`
- `html/<anchor_id>_variant_similarity.html`

Notes:
- the reusable embedding and similarity logic lives in `src/similarity.py`
- glycan cartoon lookup and generic cartoon HTML helpers live in `src/glycan_cartoons.py`
- the notebook saves outputs under `results/similarity/`
- similarity results follow the same nested pattern as the other evaluation notebooks:
  `results/similarity/<tokenizer_family>/<experiment_name>/`
- this notebook is not split-specific evaluation: it does not automatically load
  only the train, validation, or test set
- the selected checkpoint may come from a run associated with a dataset split,
  but the similarity inputs are the user-defined glycans configured in the
  notebook
- the notebook is currently set up for manual review sets, not exhaustive
  whole-test-set similarity sweeps
- the current notebook emphasizes all-variant histograms and one overall 1-to-9
  anchor ranking rather than heatmap-style visualization

### `08_similarity_scaleup.ipynb`

Purpose:
- run a broader test-set similarity analysis after the smaller manual variant review
- load one saved checkpoint and compare the full held-out test set against itself
- run `all vs all` similarity across the test set to get the background distribution
- run `specific vs all` similarity for professor-selected glycans against the test set
- build threshold-based similarity clouds and HTML reports for the selected glycans

Main outputs:
- `test_corpus_sequences.csv`
- `selected_glycans.csv`
- `all_vs_all_similarity_matrix.csv`
- `all_vs_all_summary.csv`
- `all_vs_all_top_neighbors.csv`
- `specific_vs_all_ranked.csv`
- `specific_vs_all_distribution_summary.csv`
- `specific_vs_all_threshold_clouds.csv`
- `specific_vs_all_threshold_summary.csv`
- `scaleup_cartoon_manifest.csv`
- `histograms/`
- `html/index.html`
- `html/<accession>_specific_vs_all.html`
- PCA report images and coordinate tables
- clean public HTML export folder in Drive

Notes:
- this notebook uses the real held-out `test.txt` split in Drive and assigns
  stable internal test-row IDs to the corpus side because the current Drive
  project does not include an accession-aware test-set table
- the four professor-selected GlyTouCan accessions are currently configured
  directly inside the notebook together with their compact IUPAC sequences, and
  they may be external query glycans rather than members of the held-out split
- the reusable mechanics live in `src/similarity.py`
- the notebook saves outputs under `results/similarity_scaleup/`
- the notebook can also prepare a clean browser-facing export in Drive that is
  intended to be copied into a run-specific folder such as
  `public_reports/manual/mlm15_L6_H512_A8_lr00001_ep100_setv1_train_only/live_extended/`
  before pushing to GitHub

### `09_classification_dataset_prep.ipynb`

Purpose:
- build the labeled dataset for the downstream glycan classification task
- join the accession-aware compact-IUPAC corpus to the professor's
  `classification.tsv` file
- keep only `Source == GlycoMotif` and `Level == GlycanSubtype` labels
- reuse the existing train, validation, and test split assignment by exact
  sequence match
- save clean CSV outputs for the later multi-label classification notebook

Main outputs:
- `labeled_glycans.csv`
- `labeled_glycans_with_split.csv`
- `train_classification.csv`
- `val_classification.csv`
- `test_classification.csv`
- `label_vocabulary.csv`
- `dataset_summary.csv`
- `split_summary.csv`
- `label_coverage_summary.csv`
- `missing_train_labels.csv`
- `classification_prep_summary.json`

Notes:
- the reusable dataset-prep logic lives in `src/classification_prep.py`
- the notebook expects an accession-aware raw file in Drive because the
  subtype label source is keyed by GlyTouCan accession rather than by sequence
  text alone
- the prepared classification dataset is expected to be smaller than the full
  accession-aware corpus because notebook 09 keeps only glycans that have at
  least one `GlycoMotif` / `GlycanSubtype` label in `classification.tsv`
- split assignment is done by exact sequence match so this classification
  branch stays aligned with the existing `train.txt`, `val.txt`, and `test.txt`
  files used elsewhere in the project
- this notebook is the handoff point between raw labeled data and the later
  multi-label fine-tuning workflow

### `10_classification_finetuning.ipynb`

Purpose:
- fine-tune one sequence classifier from one saved pretrained MLM checkpoint
- load the prepared train and validation classification tables from notebook 09
- train `RobertaForSequenceClassification` for multi-label subtype prediction
- review training and validation loss
- scan a few global validation thresholds before final test-set evaluation

Main outputs:
- classifier checkpoints under
  `checkpoints/classification/<tokenizer_family>/<experiment_name>/<classifier_run_label>/`
- `training_config.json`
- `trainer_state.json`
- `loss_history.csv`
- `loss_curves.png`
- `validation_metrics.csv`
- `validation_threshold_scan.csv`
- `validation_prediction_table.csv`
- `best_threshold.json`
- `label_vocabulary_snapshot.csv`

Notes:
- the reusable classifier-training logic lives in `src/classification_training.py`
- this notebook keeps classifier outputs nested under the tokenizer family and
  pretrained experiment name so downstream comparisons stay aligned with the
  original four-tokenizer workflow
- threshold selection is done on the validation split only; the test split is
  reserved for the later final evaluation notebook
- the saved `best_threshold.json` file is intended to be reused unchanged in
  the test-set evaluation step

## Public HTML Sharing Workflow

Notebook `08_similarity_scaleup.ipynb` now includes a final export step for
preparing a small static-report folder that is easier to share publicly than
the full notebook output tree.

Recommended flow:

1. Run notebook 8 in Colab.
2. Run the final public-export cell.
3. Review the clean export folder in Google Drive.
4. Copy the reviewed export into the run-specific `public_reports/...` folder in
   this repository.
5. Commit only that report folder.
6. Push to GitHub and open the printed `raw.githack.com` URL.

This keeps the public report path explicit and avoids mixing browser-facing HTML
with the larger Drive-only artifact tree.

## Tokenizer Settings In This Workflow

Current tokenizer settings in this rebuild:

- `byte_bpe`: `v300_m2`
- `glyberta`: `v1_train_only`
- `manual`: `v1_train_only`
- `hybrid_char_bpe`: `v70_m2`

These labels are used consistently across:
- tokenizer folders
- tokenized-dataset folders
- training checkpoints
- validation outputs
- test-set evaluation outputs

## Training Setup

The current training notebook supports:

- fresh runs
- checkpoint-resume runs
- continuation runs from `best_model`

The default notebook configuration currently shown in the training workflow is:

- MLM probability: `0.15`
- hidden layers: `6`
- hidden size: `512`
- attention heads: `8`
- learning rate: `1e-4`
- batch size: `32`

Training diagnostics and test outputs are written with experiment-specific
names so runs can be compared later without overwriting older results.

## Architectures Run So Far

Based on the Drive-side run registry in
`MyDrive/ProjectRoot/registry/run_index.csv`, this project has already been
used to train more than one model architecture.

Recorded architecture families:

- `L4_H384_A6`:
  4 layers, hidden size 384, 6 attention heads, intermediate size 1536
- `L6_H512_A8`:
  6 layers, hidden size 512, 8 attention heads, intermediate size 2048
- `L8_H512_A8`:
  8 layers, hidden size 512, 8 attention heads, intermediate size 2048

Architectures observed by tokenizer family:

- `manual`:
  runs recorded for `L4_H384_A6`, `L6_H512_A8`, and `L8_H512_A8`
- `hybrid_char_bpe`:
  runs recorded for `L4_H384_A6`, `L6_H512_A8`, and `L8_H512_A8`
- `byte_bpe`:
  runs recorded for `L6_H512_A8`
- `glyberta`:
  no runs recorded in the current registry yet

Run modes seen in the registry:

- fresh training runs
- continuation runs from `best_model`

Notes:

- the Drive run index also contains test-only rows where the architecture
  fields are not fully populated, so the architecture summary above is based on
  rows with recorded model dimensions
- at the time this README was updated, the registry shows completed runs for
  the original three tokenizer families and at least one in-progress
  historical run in the `L4_H384_A6` family

## Evaluation Workflow

The current evaluation workflow separates:

- notebook 4:
  - training-time validation diagnostics
  - overfitting checks
  - checkpoint behavior
- notebook 6:
  - held-out test-set evaluation
  - top-1 and top-3 token accuracy
  - top-1 and top-3 sequence accuracy
  - macro and weighted precision/recall/F1
  - per-class metrics
  - tokenizer-specific ROC and precision-recall plots
  - qualitative-probe examples

## Reproducibility Notes

To reproduce this workflow, a new user will need:

- this repository
- the raw glycan dataset
- a Drive folder matching the expected `FolderName` structure
- access to a Colab or Python environment with the required packages installed

Most notebook paths assume Google Drive mounting in Colab and a project root
of:

- `/content/drive/MyDrive/FolderName`

If this path changes, the notebook configuration cells should be updated
accordingly.

## Current Status

This rebuild currently includes:

- all six main notebooks
- tokenizer generation for all three tokenizer families
- preprocessing for all three tokenizer families
- pretraining runs for all three tokenizer families
- validation diagnostics for all three tokenizer families
- test-set evaluation outputs for the tokenizer comparison workflow

## Notes

This repository is a work-in-progress

Earlier iterations of this project have been preserved separately as a legacy record while
this version was reorganized to make:
- run-tracking clearer
- artifact storage more consistent
- notebook roles easier to follow
- reproduction easier
