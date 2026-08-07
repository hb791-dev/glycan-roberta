# Glycan RoBERTa Project

This repository contains a workflow for pretraining and evaluating
RoBERTa-style masked-language models on glycan sequences.

The project compares seven tokenizer strategies:

- `byte_bpe`
- `glyberta`
- `manual`
- `hybrid_char_bpe`
- `linkage_block`
- `donor_bound`
- `semi_atomic`

The workflow is organized as a notebook pipeline with helper scripts in
`src/`. Large artifacts such as dataset splits, trained tokenizers,
tokenized datasets, checkpoints, and evaluation outputs are stored outside the
repository in Google Drive.

## Project Split

This rebuild uses a split-storage workflow.

- GitHub repository:
  - notebooks
  - helper scripts in `src/`
  - lightweight templates and public-report assets
- Google Drive:
  - raw data
  - train-validation-test splits
  - tokenizer artifacts
  - tokenized datasets
  - checkpoints
  - evaluation outputs

## Drive Root Setting

Each notebook keeps `PROJECT_ROOT` as an editable user setting near the top.
You must update that path to match the Drive folder you are actually using.

Examples in this README use a placeholder such as:

- `MyDrive/ProjectRoot/`

If your real folder is different, for example `MyDrive/GlycanProject/`, update
`PROJECT_ROOT` in the notebook before running it. The cleaned notebooks do not
assume that the literal folder name must be `ProjectRoot`.

## Notebook Conventions

The cleaned notebooks in this repository follow a shared structure so they are
easier to read, rerun, and maintain.

- each notebook keeps a clearly labeled user settings cell near the top for
  values you may need to edit before running, such as `PROJECT_ROOT`, input
  filenames, run labels, tokenizer families, or overwrite flags
- each code cell is preceded by professional markdown that explains what the
  cell does, why it is being run, what output to expect, and how to interpret
  that output
- repeated setup, path handling, validation, overwrite checks, and other
  runtime logic are pushed into shared helpers under `src/` rather than being
  re-implemented inline in every notebook
- notebook code cells still include descriptive inline comments so the workflow
  stays beginner-friendly even when logic has been moved into helpers

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
│   │   ├── 02d_hybrid_char_bpe_gen.ipynb
│   │   ├── 02e_linkage_block_gen.ipynb
│   │   ├── 02f_donor_bound_gen.ipynb
│   │   └── 02g_semi_atomic_gen.ipynb
│   ├── 03_dataset_preprocessing.ipynb
│   ├── 04_roberta_pretraining.ipynb
│   ├── 05_validation_diagnostics.ipynb
│   ├── 06_test_set_evaluation.ipynb
│   ├── 06b_rarity_analysis.ipynb
│   ├── 07_similarity_analysis.ipynb
│   ├── 08_similarity_scaleup.ipynb
│   ├── 09_classification_dataset_prep.ipynb
│   ├── 10_classification_finetuning.ipynb
│   ├── 11_classification_evaluation.ipynb
│   ├── 11b_classification_embedding_umap.ipynb
│   ├── 12_glyberta_similarity_model_comparison.ipynb
│   └── 13_pooling_metric_comparison.ipynb
├── public_reports/
├── src/
│   ├── exploration.py
│   ├── data_utils.py
│   ├── dataset_preprocessing.py
│   ├── notebook_setup.py
│   ├── notebook_utils.py
│   ├── pretraining.py
│   ├── rarity_analysis.py
│   ├── run_index.py
│   ├── tokenizer_notebook_utils.py
│   ├── tokenizer_utils.py
│   ├── classification_embedding_umap.py
│   ├── classification_prep.py
│   ├── classification_evaluation.py
│   ├── classification_training.py
│   ├── glycan_cartoons.py
│   ├── similarity.py
│   ├── similarity_core.py
│   ├── similarity_model_comparison.py
│   ├── similarity_pooling_comparison.py
│   ├── similarity_variants.py
│   ├── similarity_scaleup.py
│   ├── test_evaluation.py
│   └── training_diagnostics.py
└── templates/
```

## Expected Drive Layout

```text
MyDrive/<PROJECT_ROOT>/
├── data/
│   ├── raw/
│   │   ├── raw_glycans_dataset_no_aldi.txt
│   │   ├── accession_reference_corpus.csv
│   │   └── classification.tsv
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│       └── split_summary.csv
├── tokenizers/
│   ├── byte_bpe/
│   ├── glyberta/
│   ├── manual/
│   ├── hybrid_char_bpe/
│   ├── linkage_block/
│   ├── donor_bound/
│   └── semi_atomic/
├── tokenized_datasets/
├── checkpoints/
├── results/
│   ├── exploration/
│   ├── validation/
│   ├── test_evaluation/
│   ├── classification_finetuning/
│   ├── classification_evaluation/
│   ├── similarity/
│   ├── similarity_scaleup/
│   └── classification_prep/
├── registry/
└── public_reports/
```

## Notebook Workflow

### `00_data_exploration.ipynb`

Purpose:
- inspect the raw glycan sequence file before any splitting or tokenizer work
- summarize dataset size and sequence-length behavior
- save a lightweight reference snapshot of the raw corpus

Inputs:
- `data/raw/raw_glycans_dataset_no_aldi.txt`

Main outputs:
- `results/exploration/dataset_summary.csv`
- `results/exploration/example_sequences.csv`
- `results/exploration/sequence_length_distribution.png`

### `01_data_splitting.ipynb`

Purpose:
- create the reusable train, validation, and test split used by later notebooks
- make the active seed explicit so the split can be reproduced later

Inputs:
- `data/raw/raw_glycans_dataset_no_aldi.txt`

Main outputs:
- `data/splits/train.txt`
- `data/splits/val.txt`
- `data/splits/test.txt`
- `data/splits/split_summary.csv`

### `02a_byte_bpe_gen.ipynb`

Purpose:
- train the byte-level BPE baseline tokenizer on the training split

Inputs:
- `data/splits/train.txt`

Main outputs:
- `tokenizers/byte_bpe/<setting_label>/vocab.json`
- `tokenizers/byte_bpe/<setting_label>/merges.txt`
- Hugging Face tokenizer files in the same folder
- `tokenizer_config_summary.json`
- optionally `inspection_preview.csv`

### `02b_glyberta_gen.ipynb`

Purpose:
- train a GlyBERTa-style WordLevel tokenizer on compact glycan strings
- isolate inline linkage text and branch markers before vocabulary learning

Inputs:
- `data/splits/train.txt`

Main outputs:
- `tokenizers/glyberta/<setting_label>/vocab.json`
- Hugging Face tokenizer files in the same folder
- `tokenizer_config_summary.json`
- optionally `inspection_preview.csv`

### `02c_manual_gen.ipynb`

Purpose:
- build a fixed manual-tokenizer vocabulary from the hand-defined parser
- save a matching Hugging Face tokenizer that follows the same token boundaries

Inputs:
- `data/splits/train.txt`

Main outputs:
- `tokenizers/manual/<setting_label>/vocab.json`
- Hugging Face tokenizer files in the same folder
- `tokenizer_config_summary.json`
- optionally `inspection_preview.csv`

### `02d_hybrid_char_bpe_gen.ipynb`

Purpose:
- train a character-level BPE tokenizer on compact glycan strings
- provide a middle ground between byte-level BPE and the structured tokenizers

Inputs:
- `data/splits/train.txt`

Main outputs:
- `tokenizers/hybrid_char_bpe/<setting_label>/vocab.json`
- `tokenizers/hybrid_char_bpe/<setting_label>/merges.txt`
- Hugging Face tokenizer files in the same folder
- `tokenizer_config_summary.json`
- optionally `inspection_preview.csv`

### `02e_linkage_block_gen.ipynb`

Purpose:
- build a fixed vocabulary where a residue plus its inline linkage can stay
  bundled together as one token

Inputs:
- `data/splits/train.txt`

Main outputs:
- `tokenizers/linkage_block/<setting_label>/vocab.json`
- Hugging Face tokenizer files in the same folder
- `tokenizer_config_summary.json`
- optionally `inspection_preview.csv`

### `02f_donor_bound_gen.ipynb`

Purpose:
- build a fixed vocabulary where donor-side information stays grouped while the
  acceptor carbon remains separate

Inputs:
- `data/splits/train.txt`

Main outputs:
- `tokenizers/donor_bound/<setting_label>/vocab.json`
- Hugging Face tokenizer files in the same folder
- `tokenizer_config_summary.json`
- optionally `inspection_preview.csv`

### `02g_semi_atomic_gen.ipynb`

Purpose:
- build the most modular fixed vocabulary in this tokenizer family
- keep residues, donor markers, and acceptor markers as smaller reusable units

Inputs:
- `data/splits/train.txt`

Main outputs:
- `tokenizers/semi_atomic/<setting_label>/vocab.json`
- Hugging Face tokenizer files in the same folder
- `tokenizer_config_summary.json`
- optionally `inspection_preview.csv`

## Tokenizer Families

### `byte_bpe`

- Learns byte-level BPE merges directly from the training split.
- This is the most generic baseline and makes the fewest glycan-specific assumptions.

### `glyberta`

- Adapts the GlyBERTa glyco-letter idea to compact IUPAC glycans.
- Inline linkages such as `b1-4` and branch markers such as `(` and `)` are isolated before WordLevel vocabulary learning.

### `manual`

- Uses the project's hand-written glycan parser to define tokens directly.
- This is the main biologically structured fixed-vocabulary baseline.

### `hybrid_char_bpe`

- Learns BPE merges from the compact glycan character inventory rather than from a full byte representation.
- This sits between fully learned BPE and fully rule-based tokenization.

### `linkage_block`

- Bundles a residue and its inline linkage together when they occur as a unit.
- Example behavior: tokens such as `Galb1-4` can stay intact.

### `donor_bound`

- Keeps donor-side residue and donor-linkage information together while separating the acceptor carbon.
- Example behavior: `Galb1` and `-4` can become separate tokens.

### `semi_atomic`

- Splits glycans into smaller reusable pieces such as residue identity, donor information, and acceptor information.
- This is the most modular structured tokenizer in the current set.

## Key Helper Scripts

### `src/notebook_setup.py`

- mounts Google Drive when needed
- syncs the GitHub repo into the Colab runtime
- adds the repo to `sys.path`
- builds a shared notebook context with standard project directories

### `src/notebook_utils.py`

- validates required input paths
- checks overwrite policy for predictable output files
- resolves reproducible or generated random seeds

### `src/exploration.py`

- loads raw glycan text data
- builds summary tables for notebook `00`
- plots and saves the sequence-length distribution
- saves the exploration CSV outputs

### `src/data_utils.py`

- loads raw sequence text
- creates train, validation, and test splits
- saves split files and `split_summary.csv`
- keeps notebook `01` focused on workflow rather than split mechanics

### `src/tokenizer_notebook_utils.py`

- builds standard tokenizer input and output paths
- validates optional output files such as inspection previews and merges
- saves tokenizer summary JSON files
- builds small tokenizer inspection tables
- centralizes repeated fixed-vocabulary tokenizer artifact creation

### `src/tokenizer_utils.py`

- contains the actual tokenization logic and training helpers
- defines the manual glycan parser
- defines the GlyBERTa-style compact split rule
- trains BPE and WordLevel tokenizer variants
- supports vocabulary construction for the structured tokenizers

### `src/dataset_preprocessing.py`

- loads saved tokenizer artifacts together with train, validation, and test
  split files
- tokenizes the three splits with one consistent preprocessing configuration
- saves the PyTorch dataset artifacts used by notebook `03`
- centralizes preview-table and summary-file creation for tokenized datasets

### `src/pretraining.py`

- builds standard pretraining paths, configs, and training arguments
- supports fresh runs, checkpoint resumes, and continuation runs
- saves experiment metadata and keeps notebook `04` focused on the run setup
  rather than Trainer boilerplate

### `src/run_index.py`

- standardizes reading and writing of the Drive-side run registry
- keeps registry updates consistent across training and evaluation notebooks
- helps prevent ad hoc run-tracking logic from drifting between notebooks

### `src/training_diagnostics.py`

- loads Trainer history from saved checkpoints
- reshapes the loss history into notebook-5-friendly tables
- saves the validation summary used for continuation decisions

### `src/test_evaluation.py`

- runs the held-out MLM test evaluation used by notebook `06`
- saves summary tables, per-class metrics, ROC and PR outputs, and qualitative
  probe artifacts
- centralizes tokenizer-family-specific evaluation settings

### `src/rarity_analysis.py`

- aggregates notebook-6 outputs by token support
- summarizes rare-token versus common-token performance patterns
- saves the rarity tables and plots used by notebook `06b`

### `src/classification_embedding_umap.py`

- builds pooled sequence embeddings for the classification dataset
- projects them with UMAP and saves the coordinate tables and plots used by
  notebook `11b`
- supports multiple color-label views over the same embedding space

### `src/similarity_pooling_comparison.py`

- compares notebook-8 outputs across `cls`, `mean`, and `max` pooling rules
- creates aligned tables, overlap summaries, and report assets for notebook
  `13`
- keeps pooling-comparison logic separate from the larger notebook-8 workflow

## Optional Diagnostic Outputs

- `inspection_preview.csv` is a convenience file for checking how a tokenizer
  split a small sample of glycans. It is useful for review, but downstream
  notebooks do not require it.
- `tokenization_preview.csv` plays the same role after dataset preprocessing.
  It is helpful for spot-checking token IDs, masks, and truncation behavior,
  but it is not a required training artifact.
- `split_preview.csv` came from an older version of notebook `01`. The cleaned
  notebook no longer depends on it or treats it as a standard output.

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

Notes:
- this notebook can review runs from any of the seven tokenizer families
- the selected tokenizer family and experiment name determine the checkpoint,
  metadata, validation-results folder, and run-index update paths

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
- the evaluation helpers now include explicit token selections and probe
  targets for `byte_bpe`, `glyberta`, `manual`, `hybrid_char_bpe`,
  `linkage_block`, `donor_bound`, and `semi_atomic`

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
- choose one tokenizer family and pretrained experiment in Drive, then build one
  run suite across the saved MLM checkpoint plus the saved classifier runs
- define small manual anchor-and-variant review sets directly in the notebook
- choose a pooling strategy for sequence embeddings (`mean` baseline or `max`
  comparison) and keep that choice attached to saved output names
- score anchor-to-variant cosine similarity and rank variants within each set
- inspect tokenization alongside similarity results
- summarize similarity distributions with histograms across all 9 variants in each anchor group
- make the overall within-anchor order explicit with a single 1-to-9 rank
- build HTML reports with glycan cartoons for anchors and variants
- optionally prepare a clean browser-facing public HTML export in Drive before
  copying it into `public_reports/`

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
  `results/similarity/<tokenizer_family>/<experiment_name>/<run_label>/`
- classification-backed variant runs are nested under:
  `results/similarity/classification/<tokenizer_family>/<experiment_name>/<classifier_run_label>/<run_label>/`
- the notebook now exposes `POOLING_STRATEGY`, and the default run label is
  expected to include a suffix such as `mean_pool` or `max_pool` so both
  variants can be saved side by side without overwriting one another
- `variant_similarity_config.json` records the pooling choice for each run
- this notebook is not split-specific evaluation: it does not automatically load
  only the train, validation, or test set
- the selected checkpoint may come from a run associated with a dataset split,
  but the similarity inputs are the user-defined glycans configured in the
  notebook
- the notebook is currently set up for manual review sets, not exhaustive
  whole-test-set similarity sweeps
- the current notebook emphasizes all-variant histograms and one overall 1-to-9
  anchor ranking rather than heatmap-style visualization
- the default notebook-7 suite is designed to run the same manual review across
  the pretrained MLM, the classifier fine-tuned from MLM init, and the
  classifier fine-tuned from random init for the chosen tokenizer family
- when public export is enabled, the notebook writes browser-facing HTML into a
  Drive review folder before you copy the final subset into
  `public_reports/07_similarity_analysis/`

### `08_similarity_scaleup.ipynb`

Purpose:
- run a broader test-set similarity analysis after the smaller manual variant review
- load one saved checkpoint and compare the full held-out test set against itself
- run `all vs all` similarity across the test set to get the background distribution
- run `specific vs all` similarity for professor-selected glycans against the test set
- build threshold-based similarity clouds and HTML reports for the selected glycans
- compare alternate embedding pooling strategies by changing one notebook setting
  and saving each run under its own pooling-labeled folder

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
- notebook 8 can now load either a pretraining MLM checkpoint or a
  classification-finetuned checkpoint, using a `CHECKPOINT_SOURCE` toggle plus
  `CLASSIFIER_RUN_LABEL` when the classification checkpoint layout is used
- the notebook saves outputs under `results/similarity_scaleup/`
- the notebook now exposes `POOLING_STRATEGY`, and the saved `OUTPUT_RUN_LABEL`
  should carry a suffix such as `mean_pool` or `max_pool` so comparison runs
  stay separated in both Drive and `public_reports/`
- `similarity_scaleup_config.json` records the pooling choice for each run
- the notebook can also prepare a clean browser-facing export in Drive that is
  intended to be copied into a run-specific folder such as
  `public_reports/08_similarity_scaleup/manual/mlm15_L6_H512_A8_lr00001_ep100_setv1_train_only/pretrained_mlm/live_extended/`
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
- fine-tune one sequence classifier from either one saved pretrained MLM
  checkpoint or a random-init baseline with the same tokenizer/config
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
- notebook 10 now has an `INITIALIZATION_MODE` toggle so the same training
  pipeline can be used for either an MLM-initialized run or a random-init
  baseline while keeping the tokenizer vocabulary fixed
- when loading a saved `RobertaForMaskedLM` checkpoint into
  `RobertaForSequenceClassification`, Hugging Face will normally report
  `UNEXPECTED` `lm_head.*` weights and `MISSING` `classifier.*` weights; this
  is expected because the shared encoder is being reused, the MLM head is being
  dropped, and a fresh classifier head is being initialized for downstream
  training
- when using the random-init baseline, the model weights start from scratch but
  the tokenizer and architecture are still recovered from the saved checkpoint
  folder so the comparison stays controlled
- threshold selection is done on the validation split only; the test split is
  reserved for the later final evaluation notebook
- the saved `best_threshold.json` file is intended to be reused unchanged in
  the test-set evaluation step

### `11_classification_evaluation.ipynb`

Purpose:
- run the final locked test-set evaluation for one trained glycan classifier
- load the saved threshold selected on the validation split in notebook 10
- compute final multi-label metrics on the held-out test set
- save per-label ROC and PR summaries for all subtype labels
- plot ROC and monotonic PR curves for the top supported labels
- add one glycan-level exact-match ROC and monotonic PR view

Main outputs:
- `test_metrics.csv`
- `test_metrics.json`
- `per_label_metrics.csv`
- `support_weighted_error_summary.csv`
- `roc_auc_per_label.csv`
- `average_precision_per_label.csv`
- `curve_aggregate_summary.csv`
- `exact_match_summary.csv`
- `top10_supported_roc_summary.csv`
- `top10_supported_pr_summary.csv`
- `exact_match_roc_curve.png`
- `exact_match_pr_curve.png`
- `top10_supported_roc_curves.png`
- `top10_supported_pr_curves.png`
- `test_prediction_table.csv`
- `evaluation_config.json`

Notes:
- the reusable evaluation logic lives in `src/classification_evaluation.py`
- this notebook is the final reporting step for the classification branch and
  is meant to keep the test split separate from the earlier training and
  threshold-selection decisions
- all labels are evaluated and saved in tables, but the default first-pass
  plots show only the top 10 labels by support so the figures stay readable
- the notebook also saves a support-aware weak-label table based on
  `support * (1 - F1)` so labels with both meaningful support and weaker
  performance are easier to spot
- the exact-match view treats each glycan as fully correct or not fully
  correct, then scores confidence using the smallest label-decision margin
  from the saved threshold
- the PR plots use monotonic precision envelopes for the same general reason as
  the earlier MLM evaluation notebook: the saved curves are easier to interpret
  visually

### `11b_classification_embedding_umap.ipynb`

Purpose:
- project one saved model state into a 2D UMAP view for qualitative
  classification-side embedding inspection
- support `pretrained_mlm`, `classification_mlm_init`, and
  `classification_random_init` model states for the same tokenizer family
- compare alternate sequence-pooling rules such as `mean` and `max`
- color the same embedding space by subtype, broad glycan class, branching, or
  `N` versus `O` views

Main outputs:
- pooled embedding coordinate tables
- UMAP projection tables
- color-view-specific PNG plots
- notebook configuration summary files

Notes:
- the reusable projection and plotting logic lives in
  `src/classification_embedding_umap.py`
- this notebook is intended for qualitative interpretation rather than final
  benchmark scoring
- it works from the prepared classification dataset plus one selected saved
  model state

### `12_glyberta_similarity_model_comparison.ipynb`

Purpose:
- compare notebook-8 similarity-scaleup outputs across any set of model runs
  listed in `RUN_SPECS`
- use the current `manual` tokenizer comparison as the default example
- focus on the professor's question about whether embeddings produce different
  similarity distributions before and after classification fine-tuning
- compare pretrained MLM embeddings, classifier embeddings from MLM init, and
  classifier embeddings from random init for the current default run
- run as a no-GPU analysis step because it only reads saved CSV outputs
- generate a visual HTML gallery report so cartoons and neighbor clouds can be
  inspected side by side

Main outputs:
- `similarity_model_comparison_config.json`
- `similarity_model_comparison_manifest.json`
- `all_vs_all_model_comparison.csv`
- `specific_vs_all_model_comparison.csv`
- `threshold_cloud_size_model_comparison.csv`
- `top_neighbor_overlap_model_comparison.csv`
- `threshold_cloud_overlap_model_comparison.csv`
- `three_way_cloud_overlap_summary.csv`
- `cloud_label_overlap_model_comparison.csv`
- `classification_exact_match_summary.csv`
- `html_neighbor_gallery_table.csv`
- `all_vs_all_model_comparison.png`
- `specific_vs_all_query_medians.png`
- `threshold_cloud_size_model_comparison_<threshold>.png`
- `similarity_model_comparison_report.html`
- `html_assets/cartoons/`

Notes:
- the reusable comparison logic lives in `src/similarity_model_comparison.py`
- this notebook does not recompute embeddings; it expects notebook 8 to have
  already produced matching `live_extended` similarity-scaleup folders
- to compare another tokenizer or model group, edit `TOKENIZER_FAMILY`,
  `EXPERIMENT_NAME`, `OUTPUT_DIR`, and the `RUN_SPECS` list in the user settings
  cell
- by default, the HTML report embeds plots and cartoons directly into the HTML
  file so the report can be downloaded/shared as one standalone file
- the notebook can also prepare a clean public export folder intended for
  `public_reports/12_glyberta_similarity_model_comparison/`
- when classifier evaluation summaries from notebook 11 are available, the HTML
  report header shows exact-match, macro F1, and weighted F1 for those
  classifier runs
- notebook 12 intentionally saves only one shareable HTML report; the CSVs,
  PNGs, config, and manifest stay as support files in the same output folder
- plots in the HTML report are clickable and open larger in an in-page modal;
  if a browser blocks the modal, the plot link can still be opened directly
- when exactly three model runs are compared, the HTML report includes a
  Venn-style accession-overlap diagram for each query similarity cloud
- each query section also reports cloud label-match rates, including exact same
  label-set rate, partial-only shared-label rate, any shared-label rate, labeled
  neighbor count, and unavailable-label count
- if `EMBED_HTML_IMAGES` is set to `False`, the HTML report instead reuses
  relative image files from `html_assets/cartoons/`, so the whole output folder
  must be kept together
- missing labels in cloud-label summaries are counted separately instead of
  being treated as negative examples

### `13_pooling_metric_comparison.ipynb`

Purpose:
- compare notebook-8 similarity outputs across `cls`, `mean`, and `max`
  pooling for one fixed checkpoint
- isolate pooling choice as the only intended difference between compared runs
- summarize how pooling affects score distributions, nearest neighbors, and
  threshold-cloud behavior

Main outputs:
- matched pooling summary tables
- merged comparison CSV outputs
- top-neighbor overlap summaries
- pooling comparison plot matrices
- a self-contained HTML report and optional clean public-export folder

Notes:
- the reusable comparison logic lives in
  `src/similarity_pooling_comparison.py`
- the three notebook-8 input folders should differ only by pooling rule if you
  want a controlled comparison
- this notebook is analysis-only and does not recompute embeddings itself

## Public HTML Sharing Workflow

Notebooks `07_similarity_analysis.ipynb`, `08_similarity_scaleup.ipynb`, and
`12_glyberta_similarity_model_comparison.ipynb` now include final export steps
for preparing small static-report folders that are easier to share publicly
than the full notebook output tree.

Recommended flow:

1. Run the notebook in Colab.
2. Run the final public-export cell.
3. Review the clean export folder in Google Drive.
4. Copy the reviewed export into the notebook-specific `public_reports/...`
   folder in this repository.
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
- `linkage_block`: `v1_train_only`
- `donor_bound`: `v1_train_only`
- `semi_atomic`: `v1_train_only`

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

Continuation runs should be saved as new sibling experiment folders under
`checkpoints/<tokenizer_family>/`. They should not create nested continuation
directories inside the source run, because downstream notebooks expect a flat
`.../<experiment_name>/best_model` layout.

The default notebook configuration currently shown in the training workflow is:

- MLM probability: `0.15`
- hidden layers: `4`
- hidden size: `384`
- attention heads: `6`
- learning rate: `1e-4`
- batch size: `32`

Training diagnostics and test outputs are written with experiment-specific
names so runs can be compared later without overwriting older results.

## Architecture Naming Convention

Experiment names use a compact architecture label so the core model settings
are visible directly in the checkpoint and results folder names.

Examples:

- `L4_H384_A6`:
  4 layers, hidden size 384, 6 attention heads, intermediate size 1536
- `L6_H512_A8`:
  6 layers, hidden size 512, 8 attention heads, intermediate size 2048
- `L8_H512_A8`:
  8 layers, hidden size 512, 8 attention heads, intermediate size 2048

This naming pattern is used alongside other run fields such as MLM probability,
learning rate, epochs, and tokenizer setting label so multiple related runs can
be compared without opening the metadata file first.

## Evaluation Workflow

The current evaluation workflow separates:

- notebook 4:
  - MLM pretraining with on-the-fly validation during training
  - fresh, checkpoint-resume, and continuation run modes
- notebook 5:
  - post-run validation diagnostics
  - best-epoch and late-training continuation review
- notebook 6:
  - held-out test-set evaluation
  - top-1 and top-3 token accuracy
  - top-1 and top-3 sequence accuracy
  - macro and weighted precision/recall/F1
  - per-class metrics
  - tokenizer-specific ROC and precision-recall plots
  - qualitative-probe examples
  - support for all seven tokenizer families
- notebook 6b:
  - rarity-oriented follow-up analysis on notebook-6 class outputs
  - support-bucket summaries for interpreting macro versus weighted metrics
- notebook 11b:
  - qualitative UMAP inspection for saved classification-side embedding states
- notebook 13:
  - pooling-only comparison for notebook-8 similarity runs

## Reproducibility Notes

To reproduce this workflow, a new user will need:

- this repository
- the raw glycan dataset
- a Drive folder matching the expected project-root structure
- access to a Colab or Python environment with the required packages installed

Most notebook paths assume Google Drive mounting in Colab and a user-edited
project root such as:

- `/content/drive/MyDrive/ProjectRoot`
- `/content/drive/MyDrive/GlycanProject`

The exact folder name is not fixed by the repository. Update `PROJECT_ROOT` in
each notebook's user settings cell so it matches the Drive folder you are
actually using.

## Current Status

This rebuild currently includes:

- the full notebook pipeline through notebooks `00` to `13`
- tokenizer generation notebooks for all seven tokenizer strategies
- dataset preprocessing support for all seven tokenizer families
- pretraining notebook support for all seven tokenizer families
- validation-diagnostics notebook support for all seven tokenizer families
- test-set evaluation support for all seven tokenizer families, including
  tokenizer-specific ROC/PR token sets and qualitative probes for the three
  new compact tokenizers
- classification-side UMAP exploration for saved embedding states
- pooling-only comparison support for notebook-8 similarity outputs

## Notes

This repository is still a work in progress.

Earlier iterations of this project have been preserved separately as a legacy record while
this version was reorganized to make:
- run-tracking clearer
- artifact storage more consistent
- notebook roles easier to follow
- reproduction easier
