# glycan-roberta-project2

Clean rebuild of the glycan RoBERTa workflow.

This repository is the rebuilt, colleague-facing version of the project. The
original `glycan-roberta-project` repo and `MyDrive/ProjectRoot` Drive folder
are preserved as legacy records and are not modified by this workflow.

## Project Split

This rebuild uses a split-storage workflow.

- GitHub repo: code, notebooks, templates, and documentation
- Google Drive: raw data, generated splits, tokenizer artifacts, tokenized
  datasets, checkpoints, plots, and evaluation outputs

Current repo:

- `hb791-dev/glycan-roberta-sandbox2`

Current Drive root:

- `MyDrive/ProjectRoot2/`

## Repository Layout

```text
glycan-roberta-project2/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── legacy_notes.md
│   ├── project_workflow.md
│   └── run_index_schema.md
├── notebooks/
│   ├── 00_data_exploration2.ipynb
│   ├── 01_data_splitting2.ipynb
│   ├── 02_tokenizer_generation/
│   │   ├── 02a_byte_bpe_gen2.ipynb
│   │   ├── 02c_manual_gen2.ipynb
│   │   └── 02d_hybrid_char_bpe_gen2.ipynb
│   ├── 03_dataset_preprocessing2.ipynb
│   ├── 04_roberta_pretraining2.ipynb
│   ├── 05_validation_diagnostics2.ipynb
│   └── 06_test_set_evaluation2.ipynb
├── src/
│   ├── data_utils.py
│   ├── run_index.py
│   ├── test_evaluation.py
│   ├── tokenizer_utils.py
│   └── training_diagnostics.py
└── templates/
    ├── experiment_metadata.example.json
    └── run_index.csv
```

## Drive Layout

```text
MyDrive/ProjectRoot2/
├── data/
│   ├── raw/
│   │   └── raw_glycans_dataset_no_aldi.txt
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│       ├── split_summary.csv
│       └── split_preview.csv
├── tokenizers/
│   ├── byte_bpe/
│   │   └── v300_m2/
│   │       ├── vocab.json
│   │       ├── merges.txt
│   │       ├── tokenizer.json
│   │       ├── tokenizer_config.json
│   │       ├── special_tokens_map.json
│   │       ├── inspection_preview.csv
│   │       └── tokenizer_config_summary.json
│   ├── hybrid_char_bpe/
│   │   └── v70_m2/
│   │       ├── vocab.json
│   │       ├── merges.txt
│   │       ├── tokenizer.json
│   │       ├── tokenizer_config.json
│   │       ├── special_tokens_map.json
│   │       ├── inspection_preview.csv
│   │       └── tokenizer_config_summary.json
│   └── manual/
│       └── v1_train_only/
│           ├── vocab.json
│           ├── tokenizer.json
│           ├── tokenizer_config.json
│           ├── special_tokens_map.json
│           ├── inspection_preview.csv
│           └── tokenizer_config_summary.json
├── tokenized_datasets/
│   ├── byte_bpe/
│   │   └── v300_m2/
│   │       ├── train_dataset.pt
│   │       ├── val_dataset.pt
│   │       ├── test_dataset.pt
│   │       ├── tokenization_preview.csv
│   │       └── preprocessing_summary.json
│   ├── hybrid_char_bpe/
│   │   └── v70_m2/
│   │       ├── train_dataset.pt
│   │       ├── val_dataset.pt
│   │       ├── test_dataset.pt
│   │       ├── tokenization_preview.csv
│   │       └── preprocessing_summary.json
│   └── manual/
│       └── v1_train_only/
│           ├── train_dataset.pt
│           ├── val_dataset.pt
│           ├── test_dataset.pt
│           ├── tokenization_preview.csv
│           └── preprocessing_summary.json
├── checkpoints/
│   ├── byte_bpe/
│   │   └── <experiment_name>/
│   │       ├── best_model/
│   │       ├── checkpoint-*/
│   │       ├── logs/
│   │       ├── experiment_metadata.json
│   │       └── trainer_state.json
│   ├── hybrid_char_bpe/
│   │   └── mlm15_L6_H512_A8_lr00001_ep100_setv70_m2/
│   │       ├── best_model/
│   │       ├── checkpoint-*/
│   │       ├── logs/
│   │       ├── experiment_metadata.json
│   │       └── trainer_state.json
│   └── manual/
│       └── <experiment_name>/
│           ├── best_model/
│           ├── checkpoint-*/
│           ├── logs/
│           ├── experiment_metadata.json
│           └── trainer_state.json
├── results/
│   ├── exploration/
│   ├── validation/
│   ├── test_evaluation/
│   └── qualitative_probes/
└── registry/
    └── run_index.csv
```

## Notebook Workflow

### `00_data_exploration2.ipynb`

Purpose:
- inspect the raw dataset before any splits are created
- summarize raw sequence lengths
- save lightweight reference outputs for later comparison

Input:
- `MyDrive/ProjectRoot2/data/raw/raw_glycans_dataset_no_aldi.txt`

Outputs:
- `MyDrive/ProjectRoot2/results/exploration/dataset_summary.csv`
- `MyDrive/ProjectRoot2/results/exploration/example_sequences.csv`
- `MyDrive/ProjectRoot2/results/exploration/sequence_length_distribution.png`

### `01_data_splitting2.ipynb`

Purpose:
- create the train, validation, and test splits from the raw dataset

Input:
- `MyDrive/ProjectRoot2/data/raw/raw_glycans_dataset_no_aldi.txt`

Outputs:
- `MyDrive/ProjectRoot2/data/splits/train.txt`
- `MyDrive/ProjectRoot2/data/splits/val.txt`
- `MyDrive/ProjectRoot2/data/splits/test.txt`
- `MyDrive/ProjectRoot2/data/splits/split_summary.csv`
- `MyDrive/ProjectRoot2/data/splits/split_preview.csv`

### `02a_byte_bpe_gen2.ipynb`

Purpose:
- train the byte-level BPE baseline tokenizer

Input:
- `MyDrive/ProjectRoot2/data/splits/train.txt`

Outputs:
- `MyDrive/ProjectRoot2/tokenizers/byte_bpe/v300_m2/vocab.json`
- `MyDrive/ProjectRoot2/tokenizers/byte_bpe/v300_m2/merges.txt`
- `MyDrive/ProjectRoot2/tokenizers/byte_bpe/v300_m2/tokenizer.json`
- `MyDrive/ProjectRoot2/tokenizers/byte_bpe/v300_m2/tokenizer_config.json`
- `MyDrive/ProjectRoot2/tokenizers/byte_bpe/v300_m2/special_tokens_map.json`
- `MyDrive/ProjectRoot2/tokenizers/byte_bpe/v300_m2/inspection_preview.csv`
- `MyDrive/ProjectRoot2/tokenizers/byte_bpe/v300_m2/tokenizer_config_summary.json`

### `02c_manual_gen2.ipynb`

Purpose:
- build the manual tokenizer from the training split using the glycan parser
- save a fixed word-level vocabulary and matching Hugging Face tokenizer

Input:
- `MyDrive/ProjectRoot2/data/splits/train.txt`

Outputs:
- `MyDrive/ProjectRoot2/tokenizers/manual/v1_train_only/vocab.json`
- `MyDrive/ProjectRoot2/tokenizers/manual/v1_train_only/tokenizer.json`
- `MyDrive/ProjectRoot2/tokenizers/manual/v1_train_only/tokenizer_config.json`
- `MyDrive/ProjectRoot2/tokenizers/manual/v1_train_only/special_tokens_map.json`
- `MyDrive/ProjectRoot2/tokenizers/manual/v1_train_only/inspection_preview.csv`
- `MyDrive/ProjectRoot2/tokenizers/manual/v1_train_only/tokenizer_config_summary.json`

### `02d_hybrid_char_bpe_gen2.ipynb`

Purpose:
- train the hybrid character-BPE tokenizer on the training split
- save raw merge files and the matching Hugging Face tokenizer

Input:
- `MyDrive/ProjectRoot2/data/splits/train.txt`

Outputs:
- `MyDrive/ProjectRoot2/tokenizers/hybrid_char_bpe/v70_m2/vocab.json`
- `MyDrive/ProjectRoot2/tokenizers/hybrid_char_bpe/v70_m2/merges.txt`
- `MyDrive/ProjectRoot2/tokenizers/hybrid_char_bpe/v70_m2/tokenizer.json`
- `MyDrive/ProjectRoot2/tokenizers/hybrid_char_bpe/v70_m2/tokenizer_config.json`
- `MyDrive/ProjectRoot2/tokenizers/hybrid_char_bpe/v70_m2/special_tokens_map.json`
- `MyDrive/ProjectRoot2/tokenizers/hybrid_char_bpe/v70_m2/inspection_preview.csv`
- `MyDrive/ProjectRoot2/tokenizers/hybrid_char_bpe/v70_m2/tokenizer_config_summary.json`

### `02_tokenizer_generation/*2.ipynb`

Tokenizer families covered in this stage:
- `byte_bpe`
- `hybrid_char_bpe`
- `manual`

### `03_dataset_preprocessing2.ipynb`

Purpose:
- tokenize the train, validation, and test splits for one tokenizer setting
- export PyTorch-ready dataset tensors

Inputs:
- `MyDrive/ProjectRoot2/data/splits/train.txt`
- `MyDrive/ProjectRoot2/data/splits/val.txt`
- `MyDrive/ProjectRoot2/data/splits/test.txt`
- tokenizer files from one folder in `MyDrive/ProjectRoot2/tokenizers/`

Outputs:
- `MyDrive/ProjectRoot2/tokenized_datasets/byte_bpe/v300_m2/train_dataset.pt`
- `MyDrive/ProjectRoot2/tokenized_datasets/byte_bpe/v300_m2/val_dataset.pt`
- `MyDrive/ProjectRoot2/tokenized_datasets/byte_bpe/v300_m2/test_dataset.pt`
- `MyDrive/ProjectRoot2/tokenized_datasets/byte_bpe/v300_m2/tokenization_preview.csv`
- `MyDrive/ProjectRoot2/tokenized_datasets/byte_bpe/v300_m2/preprocessing_summary.json`
- `MyDrive/ProjectRoot2/tokenized_datasets/hybrid_char_bpe/v70_m2/train_dataset.pt`
- `MyDrive/ProjectRoot2/tokenized_datasets/hybrid_char_bpe/v70_m2/val_dataset.pt`
- `MyDrive/ProjectRoot2/tokenized_datasets/hybrid_char_bpe/v70_m2/test_dataset.pt`
- `MyDrive/ProjectRoot2/tokenized_datasets/hybrid_char_bpe/v70_m2/tokenization_preview.csv`
- `MyDrive/ProjectRoot2/tokenized_datasets/hybrid_char_bpe/v70_m2/preprocessing_summary.json`
- `MyDrive/ProjectRoot2/tokenized_datasets/manual/v1_train_only/train_dataset.pt`
- `MyDrive/ProjectRoot2/tokenized_datasets/manual/v1_train_only/val_dataset.pt`
- `MyDrive/ProjectRoot2/tokenized_datasets/manual/v1_train_only/test_dataset.pt`
- `MyDrive/ProjectRoot2/tokenized_datasets/manual/v1_train_only/tokenization_preview.csv`
- `MyDrive/ProjectRoot2/tokenized_datasets/manual/v1_train_only/preprocessing_summary.json`

### `04_roberta_pretraining2.ipynb`

Purpose:
- train `RobertaForMaskedLM` models from scratch
- save checkpoints, `best_model`, trainer state, and experiment metadata
- support fresh runs and continuation-style runs

Inputs:
- tokenizer files from one folder in `MyDrive/ProjectRoot2/tokenizers/`
- tokenized datasets from one folder in `MyDrive/ProjectRoot2/tokenized_datasets/`
- optional checkpoint or `best_model` folder for continuation runs

Outputs:
- `MyDrive/ProjectRoot2/checkpoints/byte_bpe/<experiment_name>/best_model/`
- `MyDrive/ProjectRoot2/checkpoints/byte_bpe/<experiment_name>/checkpoint-*/`
- `MyDrive/ProjectRoot2/checkpoints/byte_bpe/<experiment_name>/logs/`
- `MyDrive/ProjectRoot2/checkpoints/byte_bpe/<experiment_name>/experiment_metadata.json`
- `MyDrive/ProjectRoot2/checkpoints/byte_bpe/<experiment_name>/trainer_state.json`
- `MyDrive/ProjectRoot2/checkpoints/hybrid_char_bpe/mlm15_L6_H512_A8_lr00001_ep100_setv70_m2/best_model/`
- `MyDrive/ProjectRoot2/checkpoints/hybrid_char_bpe/mlm15_L6_H512_A8_lr00001_ep100_setv70_m2/checkpoint-*/`
- `MyDrive/ProjectRoot2/checkpoints/hybrid_char_bpe/mlm15_L6_H512_A8_lr00001_ep100_setv70_m2/logs/`
- `MyDrive/ProjectRoot2/checkpoints/hybrid_char_bpe/mlm15_L6_H512_A8_lr00001_ep100_setv70_m2/experiment_metadata.json`
- `MyDrive/ProjectRoot2/checkpoints/hybrid_char_bpe/mlm15_L6_H512_A8_lr00001_ep100_setv70_m2/trainer_state.json`
- `MyDrive/ProjectRoot2/checkpoints/manual/<experiment_name>/best_model/`
- `MyDrive/ProjectRoot2/checkpoints/manual/<experiment_name>/checkpoint-*/`
- `MyDrive/ProjectRoot2/checkpoints/manual/<experiment_name>/logs/`
- `MyDrive/ProjectRoot2/checkpoints/manual/<experiment_name>/experiment_metadata.json`
- `MyDrive/ProjectRoot2/checkpoints/manual/<experiment_name>/trainer_state.json`
- `MyDrive/ProjectRoot2/registry/run_index.csv`

### `05_validation_diagnostics2.ipynb`

Purpose:
- inspect training and validation loss after a run finishes
- support continuation decisions
- save validation summaries

### `06_test_set_evaluation2.ipynb`

Purpose:
- evaluate the saved best model on held-out test data
- compute masked-token prediction metrics
- save test metrics, plots, and qualitative probe outputs

## Colab and GitHub Workflow

Each notebook is designed around the same Colab pattern.

1. Mount Google Drive.
2. Clone or pull `glycan-roberta-sandbox2` into the Colab runtime.
3. Add the cloned repo to `sys.path` so `src/` imports work.
4. Read and write heavy artifacts in `MyDrive/ProjectRoot2/`.
5. Save the notebook back to GitHub with the final sync cell.

Notes:
- the repo is private, so Colab uses a `GITHUB_TOKEN` secret for clone and pull
- notebooks are easiest to work with as Drive-backed copies in Colab
- the final notebook cell copies the Drive notebook into the repo clone, commits
  it, pulls remote changes, and pushes back to GitHub

## Comments and Notebook Notes

This rebuild separates two styles of explanation.

- `src/` files use professional comments and docstrings
- notebooks use shorter, more informal notes explaining what a cell does, why
  it exists, what parameters matter, and what the outputs mean

The goal is to keep the code stable while leaving the notebooks readable as
working research notes.

## Run Tracking

Future runs are intended to be recorded in:

- `MyDrive/ProjectRoot2/registry/run_index.csv`

The repo includes:

- `src/run_index.py` for notebook-side updates
- `templates/run_index.csv` as the starter schema
- `templates/experiment_metadata.example.json` as a metadata reference

## Requirements

Current dependencies:

- `transformers`
- `tokenizers`
- `torch`
- `scikit-learn`
- `matplotlib`
- `pandas`
- `numpy`
- `jupyter`

Install with:

```bash
pip install -r requirements.txt
```

## Current Status

The rebuilt workflow is in progress.

What is already in place:
- new repo initialized and pushed
- new Drive root created
- `src/` migrated and cleaned
- notebook `00_data_exploration2.ipynb` rebuilt and tested
- notebook `01_data_splitting2.ipynb` rebuilt and tested
- notebook `02a_byte_bpe_gen2.ipynb` rebuilt and tested
- notebook `02c_manual_gen2.ipynb` rebuilt and tested
- notebook `02d_hybrid_char_bpe_gen2.ipynb` rebuilt and tested
- notebook `03_dataset_preprocessing2.ipynb` rebuilt and tested for all three tokenizers
- notebook `04_roberta_pretraining2.ipynb` rebuilt and tested for hybrid character-BPE

What is next:
- continue notebook-by-notebook through preprocessing,
  training, validation, and test evaluation
