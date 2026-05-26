# Pre-SNS Meaningful Training V2

PRE_SNS_MEANINGFUL_TRAINING_V2_OK

This v2 path exists because the earlier pre-SNS baseline was good enough for smoke testing but not good enough to support proposal-style claims. A balanced validation subset showed weak class accuracy and macro-F1, overlapping real/tampered tampered-score distributions, and unstable localization activation around tau.

The scope is intentionally pre-SNS only. SNS augmentation, screenshot simulation, text overlays, stickers, and recompression robustness are excluded here and must be evaluated later in a separate SNS task.

## Inputs

V2 requires separate `train_manifest_path` and `val_manifest_path`. The runner reads only explicit samples from those manifests and does not recursively scan directories.

Approved local configs must also declare `approved_input_roots`. Absolute train/validation manifest paths, and actual image or mask paths used during approved training, must stay under those input roots.

The class head uses:

- `real`
- `full_synthetic`
- `tampered`

The family head uses:

- `LatDiff`
- `PixDiff`
- `GAN`
- `Other`
- `Real-or-N/A`

Family loss is applied only when a sample has a meaningful family label. Localization loss is applied only for tampered samples with masks.

## Modes

`no_write_dry_run=true` validates the config and exercises deterministic dry-run training/evaluation behavior without creating run artifacts or checkpoints.

`no_write_dry_run=false` is allowed only in approved local mode. Run artifacts must be written under `approved_run_root`, and checkpoints must be written under `approved_checkpoint_root`. Both roots must be outside the repository, such as under `~/cvf_runs` and `~/cvf_checkpoints`.

Repository `outputs/` and `checkpoints/` paths are rejected, as are protected paths such as `.env`, `secrets`, `data`, and `datasets`.

## Outputs

Actual approved runs write:

- `run_summary.json`
- `train_metrics.jsonl`
- `val_metrics.json`
- `threshold_calibration.json`
- `confusion_matrix.json`
- `artifact_manifest.json`
- best checkpoint
- latest checkpoint when available

Validation reports include class accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, family accuracy on meaningful labels, tampered-score summaries by ground-truth class, tau sweep, selected tau, localization activation recall, false activation rate, and localization mean/median IoU.

Tau selection is deterministic: it maximizes `tampered_recall - 0.5 * false_activation_rate + 0.1 * localization_iou`, with ties choosing the lower tau to preserve tampered recall.
