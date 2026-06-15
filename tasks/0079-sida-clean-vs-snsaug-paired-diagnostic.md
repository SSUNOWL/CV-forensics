# Task 0079: SIDA Clean-vs-SNSAug Paired Diagnostic

## Context

The project has evaluated SIDA-7B on an SNSAug subset and compared it against the final mixed_feature_gate.

However, the current SIDA result only answers how SIDA behaves on SNSAug views.
To determine whether SNSAug itself causes degradation, we need a paired comparison between clean/basic views and SNSAug views for the same base samples.

This directly supports the project goal of evaluating social-media perturbation robustness.

## Goal

Evaluate whether SIDA-7B performance changes from clean/basic images to SNSAug-transformed images on the same base_id/content_label samples.

This is an evaluation/export/cached-analysis task. Do not train.

## Files Codex May Modify

- `tasks/0079-sida-clean-vs-snsaug-paired-diagnostic.md`
- `src/cv_forensics/snsaug_v2_sida_clean_vs_snsaug.py`
- `scripts/evaluation/run_snsaug_v2_sida_clean_vs_snsaug.py`
- `scripts/agent/validate_snsaug_v2_sida_clean_vs_snsaug_config.py`
- `configs/evaluation/snsaug_v2_sida_clean_vs_snsaug.example.json`
- `tests/test_snsaug_v2_sida_clean_vs_snsaug.py`
- `docs/snsaug_v2_sida_clean_vs_snsaug.md`

## Files Claude May Modify

- `tasks/0079-sida-clean-vs-snsaug-paired-diagnostic.md`
- `src/cv_forensics/snsaug_v2_sida_clean_vs_snsaug.py`
- `scripts/evaluation/run_snsaug_v2_sida_clean_vs_snsaug.py`
- `scripts/agent/validate_snsaug_v2_sida_clean_vs_snsaug_config.py`
- `configs/evaluation/snsaug_v2_sida_clean_vs_snsaug.example.json`
- `tests/test_snsaug_v2_sida_clean_vs_snsaug.py`
- `docs/snsaug_v2_sida_clean_vs_snsaug.md`

## Required Behavior

1. Modes.

- export_clean_counterpart_mode
- cached_clean_vs_snsaug_eval_mode

2. Inputs.

Use:

- SNSAug pair root from 0058c
- meta.jsonl
- existing 0074 SIDA SNSAug export manifest
- existing SIDA SNSAug cached outputs with masks
- optional SIDA clean cached outputs
- optional 0072 mixed_feature_gate output

3. Export clean counterpart subset.

For each SNSAug exported row, find the matching clean/basic image by:

- base_id
- content_label
- source_dataset if available
- clean profile or original image path in meta.jsonl

Write:

- sida_clean_counterpart_manifest.jsonl
- sida_clean_prompt_list.jsonl
- run_sida_clean_external_template.sh

4. Prompt.

Use the same SIDA prompt style as the SNSAug run:

- [CLS] classification
- [SEG] mask if tampered

5. Cached paired evaluation.

When both clean and SNSAug cached outputs exist, join them by:

- base_id
- content_label
- target sns profile
- image_id pair metadata

6. Metrics.

Compute paired deltas:

- clean_pred_class
- sns_pred_class
- class_flip
- clean_synthetic_recall
- sns_synthetic_recall
- clean_tampered_recall
- sns_tampered_recall
- clean_real_fpr
- sns_real_fpr
- clean_mask_available
- sns_mask_available
- clean_tamper_iou
- sns_tamper_iou
- delta_iou
- mask_missing_increase
- ignore_capture_increase

7. Family summaries.

Compute summaries for:

- clean_reference
- Type A local overlay
- Type B global geometry/degradation
- strict Type A+B

8. Synthetic handling.

Keep synthetic for 3-way classification and synthetic-to-tampered false positive analysis.
Do not compute tampered mask IoU for synthetic samples.
For synthetic, report mask false-positive rate.

9. Outputs.

Write:

- sida_clean_vs_snsaug_pairs.jsonl
- sida_clean_vs_snsaug_metrics.json
- sida_clean_vs_snsaug_type_summary.json
- sida_clean_vs_snsaug_report.md
- artifact_manifest.json

10. Interpretation.

Report must distinguish:

- SIDA clean weakness
- SNSAug-induced degradation
- Type A local overlay degradation
- Type B geometry/degradation degradation

11. Tests.

- clean counterpart matching works
- export mode writes prompt and manifest
- cached eval mode joins clean and SNS rows
- synthetic is retained for 3-way but excluded from mask IoU
- Type A/B summary computes deltas
- artifact manifest is written
- dry-run writes no records

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_sida_clean_vs_snsaug_config.py configs/evaluation/snsaug_v2_sida_clean_vs_snsaug.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_sida_clean_vs_snsaug.py
- python3 scripts/evaluation/run_snsaug_v2_sida_clean_vs_snsaug.py --help
- grep -q SNSAUG_V2_SIDA_CLEAN_VS_SNSAUG_OK docs/snsaug_v2_sida_clean_vs_snsaug.md
- python3 scripts/agent/check_agent_changes.py tasks/0079-sida-clean-vs-snsaug-paired-diagnostic.md

## Marker

SNSAUG_V2_SIDA_CLEAN_VS_SNSAUG_OK
