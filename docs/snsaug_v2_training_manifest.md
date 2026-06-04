# SNSAug V2 Training Manifest

`SNSAUG_V2_TRAINING_MANIFEST_AND_WRAPPER_OK`

This task prepares SNS-aware training data selection only. It does not train models. It builds train-only manifests from the `0052` train-only mining output and provides an on-the-fly dataset wrapper for clean, basic, and SNSAug V2 views.

## Manifest Builder

The builder consumes:

- a base train manifest
- a train-only mining manifest from `0052`
- an external output root

It writes:

- `snsaug_v2_training_manifest.jsonl`
- `snsaug_v2_sampling_summary.json`
- `snsaug_v2_class_balance_summary.json`
- `artifact_manifest.json`

Non-train rows are dropped. Validation and test samples must not appear in the training manifest.

## Sampling Policy

Tampered-target group mix:

- `stable_correct_anchor`: 40%
- fragile bundle (`fragile_correct_to_fail`, `confidence_fragile`, `mask_iou_fragile`): 35%
- `clean_fail`: 15%
- `random_tampered_coverage`: 10%

Default class-balance targets:

- real: 25-30%
- synthetic: 25-30%
- tampered: 40-50%

Default augmentation mode ratio:

- clean: 30%
- basic augmentation: 30%
- sns augmentation: 40%

## Curriculum

The wrapper supports a deterministic curriculum:

- epoch 1-3: clean 50%, basic 30%, sns light 20%
- epoch 4-8: clean 35%, basic 30%, sns medium 35%
- epoch 9+: clean 25%, basic 25%, sns medium or heavy 50%

## Dataset Wrapper

`SNSAugV2DatasetWrapper` wraps an existing dataset and returns:

- image
- label
- tamper mask
- ignore mask
- view
- profile
- augmentation metadata
- base id
- optional family label
- `family_loss_mask`

Clean samples return a zero `ignore_mask`. Overlay profiles return non-zero `ignore_mask` regions.

## Loss Policy

`masked_localization_loss(pred_mask, tamper_mask, ignore_mask)` applies BCE only on valid pixels:

- `valid_region = 1 - ignore_mask`
- ignored benign overlay regions do not contribute to localization loss

Family supervision policy:

- missing `family_label` -> `family_loss_mask = 0`
- SID-Set-like rows without family labels do not contribute to family loss
- rows with family labels may contribute
