# Public Reports

This folder is the GitHub-facing home for small static report exports that are
safe to share in a browser.

The public layout is notebook-based on purpose. Each top-level folder maps back
to the notebook that produced the report, which keeps new report families easy
to place without inventing a second taxonomy for HTML outputs.

During large rerun cycles, the notebook-specific folders may be intentionally
emptied between report batches so stale published HTML does not get confused
with the current run set.

Current notebook roots:

- `public_reports/07_similarity_analysis/`
- `public_reports/08_similarity_scaleup/`
- `public_reports/13_pooling_metric_comparison/`
- `public_reports/12_glyberta_similarity_model_comparison/`

The intended workflow is:

1. Generate the clean HTML export from the notebook.
2. Let that notebook save the export to Google Drive first.
3. Review the exported files in Drive and make sure the scan is clean.
4. Copy the final shareable folder into this directory.
5. Commit only the report folder you want to publish.

Expected repo destinations are notebook-specific, for example:

- notebook 7:
  `public_reports/07_similarity_analysis/manual/mlm15_L6_H512_A8_lr00001_ep100_setv1_train_only/pretrained_mlm/live_extended/`
- notebook 7 classifier run:
  `public_reports/07_similarity_analysis/manual/mlm15_L6_H512_A8_lr00001_ep100_setv1_train_only/classification_mlm_init/live_extended__manual__classification_mlm_init__mean_pool/`
- notebook 8:
  `public_reports/08_similarity_scaleup/manual/mlm15_L6_H512_A8_lr00001_ep100_setv1_train_only/pretrained_mlm/live_extended__manual__pretrained_mlm__mean_pool/`
- notebook 8 classifier run:
  `public_reports/08_similarity_scaleup/manual/mlm15_L6_H512_A8_lr00001_ep100_setv1_train_only/classification_random_init/live_extended__manual__classification_random_init__max_pool/`
- notebook 12:
  `public_reports/12_glyberta_similarity_model_comparison/manual/mlm15_L6_H512_A8_lr00001_ep100_setv1_train_only/pretrain_vs_classifier_mlm_vs_randominit/`
- notebook 13:
  `public_reports/13_pooling_metric_comparison/manual/mlm15_L6_H512_A8_lr00001_ep100_setv1_train_only/classification_mlm_init/classification_mlm_init__live_extended__cls_mean_max/`

New public exports should use the clean report-facing model IDs:

- `pretrained_mlm`
- `classification_mlm_init`
- `classification_random_init`

For classification runs, the notebooks may still load checkpoints from legacy
Drive folder names such as `cls_lr2e-5_ep10_bs16_mlm` or
`cls_lr2e-5_ep10_bs16_randominit`. That legacy label is only the checkpoint
folder name used for loading. It should not be reused as the public-facing
report folder name for new exports.

This folder is meant for browser-facing HTML only, not full notebook output
trees, Drive-only artifacts, or intermediate CSV/debug files unless you
explicitly want them public.
