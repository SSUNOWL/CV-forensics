# Pre-SNS v3 Hard-Negative Refinement

PRE_SNS_V3_REFINEMENT_OK

This task adds a guarded pre-SNS hard-negative refinement stage for the clean v3 model. It is not SNS augmentation and must not execute screenshot, sticker, text-overlay, recompression, or other SNS transforms.

The refinement uses hard cases mined from the clean TRAIN split only. Validation hard-mining outputs are useful for diagnosis, but they must not be used as training examples or sampling weights.

## Inputs

The config kind for approved local execution is `approved_pre_sns_v3_refinement`, with execution mode `approved_local_pre_sns_v3_refinement` and approval text `I_APPROVE_PRE_SNS_V3_REFINEMENT`.

Required inputs include:

- clean train manifest
- clean validation manifest
- base v3 checkpoint, normally clean long256 `best_checkpoint.pt`
- train hard-mining JSON files:
  - `hard_negative_real.json`
  - `hard_negative_non_tampered.json`
  - `hard_positive_tampered_low_iou.json`
  - `class_mask_inconsistent_cases.json`

`hard_cases_split` must be `train`. The validator rejects validation hard-mining paths and unsafe guardrails.

## Refinement

Normal train samples remain available. Hard cases from the train split are oversampled with configurable defaults:

- real hard negatives: `5`
- non-tampered hard negatives: `4`
- class-mask inconsistent cases: `3`
- tampered low-IoU positives: `3`

The loss starts from pre-SNS v3 and shifts weight toward the observed failure mode:

- class loss: `1.0`
- tamper binary loss: `1.5`
- family loss: `0.2`
- localization loss: `12.0`
- non-tampered empty-mask loss: `0.8`

Tau calibration remains FPR-constrained.

## Artifacts

Artifacts are written only under an approved run root outside the repository:

- `run_summary.json`
- `val_metrics.json`
- `threshold_calibration.json`
- `confusion_matrix.json`
- `per_source_confusion_matrix.json`
- `refinement_hard_case_summary.json`
- `train_metrics.jsonl`
- `artifact_manifest.json`
- `config_snapshot.json`

Checkpoints are written only under an approved checkpoint root outside the repository:

- `best_checkpoint.pt`
- `latest_checkpoint.pt`

## Added Metrics

The refinement reports the normal v3 metrics plus:

- `hard_negative_real_count_used`
- `hard_negative_non_tampered_count_used`
- `hard_inconsistent_count_used`
- `hard_positive_low_iou_count_used`
- `class_mask_inconsistency_proxy`
- `non_tampered_mask_activation_rate`
