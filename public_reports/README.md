# Public Reports

This folder is the GitHub-facing home for small static report exports that are
safe to share in a browser.

The intended workflow is:

1. Generate the clean HTML export from notebook `07_similarity_analysis.ipynb`
   or notebook `08_similarity_scaleup.ipynb`.
2. Let that notebook save the export to Google Drive first.
3. Review the exported files in Drive and make sure the scan is clean.
4. Copy the final shareable folder into this directory.
5. Commit only the report folder you want to publish.

For notebook 7 manual variant-review reports, the expected repo destination is
run-specific, for example:

- `public_reports/similarity/manual/mlm15_L6_H512_A8_lr00001_ep100_setv1_train_only/live_extended/`
- `public_reports/similarity/classification/manual/mlm15_L6_H512_A8_lr00001_ep100_setv1_train_only/cls_lr2e-5_ep100_bs16_mlm/live_extended/`

For notebook 8 similarity scale-up reports, the expected repo destination is
run-specific, for example:

- `public_reports/manual/mlm15_L6_H512_A8_lr00001_ep100_setv1_train_only/live_extended/`

That folder is meant for browser-facing HTML only, not full notebook output
trees, Drive-only artifacts, or intermediate CSV/debug files unless you
explicitly want them public.
