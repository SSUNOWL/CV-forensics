# Task 0069: Geometry-normalized and Degradation-aware Inference for SNSAug Recovery

## Context

0067a Masked Nuisance Inference showed that local ignore_mask-based pixel masking does not sufficiently recover tampered recall or valid IoU.

0068 Global Degradation and Geometry Shift Analysis showed that global degradation / geometry is the dominant factor:

- global_degradation_geometry score is much larger than local_nuisance score
- crop_scale_proxy is strongly correlated with delta_valid_iou
- blockiness_delta_abs is strongly correlated with activation_flip_off, delta_p_tampered, and delta_valid_iou
- geometry family has zero ignore_mask area and zero tamper occlusion, but high pred_flip_rate and large valid_iou drop

Therefore the next recovery experiment should test geometry normalization and degradation-aware preprocessing, not more local overlay masking.

## Goal

Evaluate whether inference-time geometry normalization and mild degradation correction recover baseline tampered classification/localization on SNSAug profiles.

## Files Codex May Modify

- `tasks/0069-geometry-normalized-degradation-aware-inference.md`
- `src/cv_forensics/snsaug_v2_geometry_degradation_recovery.py`
- `scripts/evaluation/run_snsaug_v2_geometry_degradation_recovery.py`
- `scripts/agent/validate_snsaug_v2_geometry_degradation_recovery_config.py`
- `configs/evaluation/snsaug_v2_geometry_degradation_recovery.example.json`
- `tests/test_snsaug_v2_geometry_degradation_recovery.py`
- `docs/snsaug_v2_geometry_degradation_recovery.md`

## Files Claude May Modify

- `tasks/0069-geometry-normalized-degradation-aware-inference.md`
- `src/cv_forensics/snsaug_v2_geometry_degradation_recovery.py`
- `scripts/evaluation/run_snsaug_v2_geometry_degradation_recovery.py`
- `scripts/agent/validate_snsaug_v2_geometry_degradation_recovery_config.py`
- `configs/evaluation/snsaug_v2_geometry_degradation_recovery.example.json`
- `tests/test_snsaug_v2_geometry_degradation_recovery.py`
- `docs/snsaug_v2_geometry_degradation_recovery.md`

## Required Behavior

1. No training.

- no_training = true
- no_finetune = true
- no_network = true
- no_download = true

2. Inputs.

Use:

- frozen pre-SNS best bundle
- fixed SNSAug pair root
- meta.jsonl
- images directory
- tamper_masks directory
- optional ignore_masks directory

3. Recovery policies.

Implement inference-time variants:

- original
- content_box_crop_resize
- geometry_normalized
- deblock_mild
- resize_to_clean_proxy
- geometry_plus_deblock
- oracle_clean_geometry_diagnostic

The oracle policy is evaluation-only and must be clearly marked as diagnostic, not deployment-ready.

4. Geometry normalization.

Use metadata if available. If not available, estimate content box using border/padding detection and non-background region detection.

For canvas/crop/zoom/screenshot profiles, attempt to recover the main content region and resize it to the model input geometry.

5. Degradation correction.

Implement simple inference-time corrections:

- mild deblocking using median or bilateral-like approximation
- optional JPEG re-encode quality sweep if PIL supports it
- optional resize-to-clean-scale proxy

Do not introduce heavy external dependencies.

6. Evaluation.

For each policy/profile/label, run frozen pre-SNS baseline and write:

- geometry_degradation_recovery_records.jsonl
- per_policy_per_profile_metrics.json
- recovery_delta_summary.json
- geometry_degradation_recovery_report.md
- visual_gallery_manifest.json
- artifact_manifest.json

Metrics:

- accuracy
- macro_f1
- real_fpr
- synthetic_recall
- tampered_recall
- localization_activation_recall
- tampered_valid_mean_iou
- mean_p_tampered_on_tampered
- non_tampered_high_mask_rate
- recovery over original

7. Success criteria.

A policy is promising if:

- tampered_recall improves by at least 0.10 on at least two degradation/geometry profiles, or
- valid_iou improves by at least 0.05 on at least two degradation/geometry profiles, and
- synthetic_recall does not collapse relative to original, and
- real_fpr does not increase severely.

Focus profiles:

- canvas_9x16_only
- resize_crop_pad
- zoom_crop
- resize_jpeg
- screenshot_recapture_light
- combined_sns_realistic

8. Tests.

- config validator rejects training flags
- content box crop modifies geometry profiles
- deblock_mild keeps image size
- policy output paths are written
- metrics handle missing tampered rows safely
- artifact manifest is written
- dry-run writes no records

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_geometry_degradation_recovery_config.py configs/evaluation/snsaug_v2_geometry_degradation_recovery.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_geometry_degradation_recovery.py
- python3 scripts/evaluation/run_snsaug_v2_geometry_degradation_recovery.py --help
- grep -q SNSAUG_V2_GEOMETRY_DEGRADATION_RECOVERY_OK docs/snsaug_v2_geometry_degradation_recovery.md
- python3 scripts/agent/check_agent_changes.py tasks/0069-geometry-normalized-degradation-aware-inference.md

## Marker

SNSAUG_V2_GEOMETRY_DEGRADATION_RECOVERY_OK
