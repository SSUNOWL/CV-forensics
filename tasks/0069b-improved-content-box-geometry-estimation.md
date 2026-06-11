# Task 0069b: Improved Content-box and Geometry Estimation for SNSAug Recovery

## Context

0067a Masked Nuisance Inference showed that local ignore_mask pixel masking was not sufficient.

0068 Global Degradation and Geometry Shift Analysis showed that global degradation / geometry is the dominant factor.

0069 Geometry/Degradation Recovery showed:

- actual run completed successfully
- record_count = 4200
- oracle_clean_geometry_diagnostic recovered all focus profiles
- non-oracle policies mostly recovered only canvas_9x16_only
- decision = oracle_only_geometry_recovery

Interpretation:

Geometry/degradation recovery is possible in principle, but current non-oracle content-box and geometry heuristics are too weak.

## Goal

Improve deployable content-box and geometry estimation so that non-oracle recovery closes part of the gap to oracle_clean_geometry_diagnostic.

This is still an inference-only diagnostic task. Do not train.

## Files Codex May Modify

- `tasks/0069b-improved-content-box-geometry-estimation.md`
- `src/cv_forensics/snsaug_v2_geometry_degradation_recovery.py`
- `scripts/evaluation/run_snsaug_v2_geometry_degradation_recovery.py`
- `scripts/agent/validate_snsaug_v2_geometry_degradation_recovery_config.py`
- `configs/evaluation/snsaug_v2_geometry_degradation_recovery.example.json`
- `tests/test_snsaug_v2_geometry_degradation_recovery.py`
- `docs/snsaug_v2_geometry_degradation_recovery.md`

## Files Claude May Modify

- `tasks/0069b-improved-content-box-geometry-estimation.md`
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

2. Keep existing 0069 behavior.

Do not remove existing policies:

- original
- content_box_crop_resize
- geometry_normalized
- deblock_mild
- resize_to_clean_proxy
- geometry_plus_deblock
- oracle_clean_geometry_diagnostic

3. Add improved non-oracle policies.

Add at least these policies:

- border_trim_v2
- edge_density_content_box_v2
- letterbox_unpad_resize_v2
- screenshot_frame_trim_v2
- multi_candidate_geometry_v2
- multi_candidate_geometry_plus_deblock_v2

Optional diagnostic-only policy:

- score_guided_geometry_diagnostic

The score-guided policy must be marked diagnostic-only and not deployment-ready.

4. Content-box estimation.

Implement multiple content-box candidates using image-only heuristics:

- border color / padding detection from image margins
- low-gradient uniform border detection
- edge-density bounding box
- foreground/content bounding box based on gradient and color contrast
- aspect-ratio constrained crop candidates
- central crop candidates for zoom/crop recovery
- screenshot frame trim candidates

Do not use ground-truth label or tamper mask to select deployable candidates.

5. Candidate selection.

For deployable policies, select candidates using only image statistics:

- content area ratio
- border uniformity
- edge density inside crop
- aspect-ratio plausibility
- blockiness / high-pass residual stability
- avoid extreme crop area loss

For diagnostic policies, model score-guided selection may be allowed only if clearly labeled diagnostic.

6. Mask transform correctness.

When an image crop/resize transform is applied, transform the tamper mask with the same operation before computing valid IoU.

This is critical. Do not compare recovered-image predictions against the untransformed original mask.

7. Oracle gap closure.

Compute per policy and profile:

- oracle_tampered_recall_gain
- oracle_valid_iou_gain
- policy_tampered_recall_gain
- policy_valid_iou_gain
- tampered_recall_oracle_gap_closure
- valid_iou_oracle_gap_closure

Definition:

gap_closure = policy_gain / oracle_gain, clipped to [0, 1] when oracle_gain > 0.

8. Success criteria.

A non-oracle policy is promising if:

- it improves tampered_recall by at least 0.10 on at least two focus profiles, or
- it improves valid_iou by at least 0.05 on at least two focus profiles, or
- it closes at least 40 percent of the oracle valid_iou gap on at least two focus profiles,

and:

- synthetic_recall does not decrease by more than 0.10 on average
- real_fpr does not increase by more than 0.20 on average

9. Focus profiles.

- canvas_9x16_only
- resize_crop_pad
- zoom_crop
- resize_jpeg
- screenshot_recapture_light
- combined_sns_realistic

10. Outputs.

Write all existing 0069 outputs plus:

- oracle_gap_closure_summary.json
- content_box_candidate_diagnostics.jsonl
- 0069b_geometry_estimation_report.md

Existing outputs must still include:

- geometry_degradation_recovery_records.jsonl
- per_policy_per_profile_metrics.json
- recovery_delta_summary.json
- geometry_degradation_recovery_report.md
- visual_gallery_manifest.json
- artifact_manifest.json

11. Tests.

Tests must cover:

- border_trim_v2 removes synthetic padding on a toy image
- edge_density_content_box_v2 finds a central content region
- letterbox_unpad_resize_v2 keeps image output size valid
- mask transform follows image transform
- oracle gap closure calculation handles zero oracle gain
- non-oracle candidate selection does not use content_label or tamper_mask
- dry-run writes no records
- actual tiny run writes artifact_manifest and oracle_gap_closure_summary

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_geometry_degradation_recovery_config.py configs/evaluation/snsaug_v2_geometry_degradation_recovery.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_geometry_degradation_recovery.py
- python3 scripts/evaluation/run_snsaug_v2_geometry_degradation_recovery.py --help
- grep -q SNSAUG_V2_GEOMETRY_DEGRADATION_RECOVERY_OK docs/snsaug_v2_geometry_degradation_recovery.md
- python3 scripts/agent/check_agent_changes.py tasks/0069b-improved-content-box-geometry-estimation.md

## Marker

SNSAUG_V2_GEOMETRY_DEGRADATION_RECOVERY_OK
