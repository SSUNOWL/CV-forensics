# Task 0067: Masked Nuisance Inference for SNSAug Forensic Distraction

## Context

SNSAug v2 training variants 0060b, 0063b, 0064f, and 0064g produced valid checkpoints and evaluation records, but did not yield a robust operating point.

Observed failures:

- Frozen pre-SNS baseline keeps relatively better real/synthetic discrimination, but SNS platform profiles suppress tampered activation and localization.
- Fine-tuned SNSAug models recover tampered activation, but over-predict tampered for real/synthetic and collapse synthetic recall.
- Threshold sweep and score fusion did not find a strict candidate operating point.
- 0066 tendency analysis suggests SNSAug should be treated as forensic distraction / nuisance layer, not simple augmentation.

The WACV 2026 paper Ignoring the Decoy motivates this task: benign visual elements such as logos, watermarks, and captions can mislead image forgery localization models, and masked convolution can make models ignore masked distractions without retraining.

## Goal

Evaluate whether masking local SNS nuisance regions using SNSAug ignore_mask can restore baseline tampered classification/localization.

Main question:

If local SNS overlay is ignored at inference time, does the frozen pre-SNS baseline recover tampered recall and valid IoU on SNSAug profiles?

## Files Codex May Modify

- `tasks/0067-masked-nuisance-inference-for-snsaug-distraction.md`
- `src/cv_forensics/snsaug_v2_masked_nuisance_inference.py`
- `scripts/evaluation/run_snsaug_v2_masked_nuisance_inference.py`
- `scripts/agent/validate_snsaug_v2_masked_nuisance_inference_config.py`
- `configs/evaluation/snsaug_v2_masked_nuisance_inference.example.json`
- `tests/test_snsaug_v2_masked_nuisance_inference.py`
- `docs/snsaug_v2_masked_nuisance_inference.md`

## Files Claude May Modify

- `tasks/0067-masked-nuisance-inference-for-snsaug-distraction.md`
- `src/cv_forensics/snsaug_v2_masked_nuisance_inference.py`
- `scripts/evaluation/run_snsaug_v2_masked_nuisance_inference.py`
- `scripts/agent/validate_snsaug_v2_masked_nuisance_inference_config.py`
- `configs/evaluation/snsaug_v2_masked_nuisance_inference.example.json`
- `tests/test_snsaug_v2_masked_nuisance_inference.py`
- `docs/snsaug_v2_masked_nuisance_inference.md`

## Required Behavior

1. No training.

- no_training must be true
- no_finetune must be true
- no_download must be true
- no_network must be true

2. Inputs.

Use:

- frozen pre-SNS best bundle
- fixed SNSAug pair root from 0058c or later
- meta.jsonl
- ignore_masks directory
- tamper_masks directory
- images directory

3. Masking policies.

Implement inference-time variants:

- original: no masking
- gray_fill: fill ignore_mask region with neutral gray
- blur_fill: blur image and paste blurred pixels only inside ignore_mask
- mean_fill: fill ignore_mask region with image mean color
- black_fill: fill ignore_mask region with black
- dilated_gray_fill: dilate ignore_mask then gray-fill
- valid_region_crop_optional: optional crop/pad around non-ignore region if easy

This is not true masked convolution yet; it is a practical approximation to test whether local nuisance pixels are the cause.

4. Evaluation.

For each row/profile/masking policy, run the frozen pre-SNS baseline and write:

- model_eval_records_masked.jsonl
- per_policy_per_profile_metrics.json
- clean_vs_sns_masked_delta.json
- masked_nuisance_inference_summary.json
- masked_nuisance_inference_report.md
- visual_gallery_manifest.json

Metrics:

- accuracy
- macro_f1
- real_fpr
- synthetic_recall
- tampered_recall
- localization_activation_recall
- tampered_valid_mean_iou
- non_tampered_high_mask_rate
- mean_p_tampered_on_tampered
- mask_area_ratio
- recovery over original SNS view

5. Recovery metrics.

Compute per profile and per policy:

- tampered_recall_recovery = masked_tampered_recall - original_tampered_recall
- valid_iou_recovery = masked_valid_iou - original_valid_iou
- p_tampered_recovery_on_tampered = masked_mean_p_tampered_on_tampered - original_mean_p_tampered_on_tampered
- synthetic_recall_change
- real_fpr_change

6. Success criteria.

A masked policy is promising if:

- tampered_recall improves on at least two SNS platform profiles
- valid_iou improves on at least two SNS platform profiles
- synthetic_recall does not collapse relative to original
- real_fpr does not increase severely

7. Guardrails.

- Do not use fixed validation pair root for training.
- Do not write outputs inside repo.
- Do not overwrite pair_root.
- Do not modify existing checkpoints.
- Do not download.
- Do not use network.

8. Tests.

Tests must cover:

- ignore_mask path resolution
- gray_fill modifies only ignore region
- blur_fill modifies only ignore region
- mask dilation increases mask area
- metrics handle no tampered rows safely
- config validator rejects training flags
- dry-run writes no inference records
- tiny run writes summary and records

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_masked_nuisance_inference_config.py configs/evaluation/snsaug_v2_masked_nuisance_inference.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_masked_nuisance_inference.py
- python3 scripts/evaluation/run_snsaug_v2_masked_nuisance_inference.py --help
- grep -q SNSAUG_V2_MASKED_NUISANCE_INFERENCE_OK docs/snsaug_v2_masked_nuisance_inference.md
- python3 scripts/agent/check_agent_changes.py tasks/0067-masked-nuisance-inference-for-snsaug-distraction.md

## Marker

SNSAUG_V2_MASKED_NUISANCE_INFERENCE_OK
