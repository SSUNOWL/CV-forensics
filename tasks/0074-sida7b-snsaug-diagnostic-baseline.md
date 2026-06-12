# Task 0074: SIDA-7B SNSAug Diagnostic Baseline

## Context

The project completed SNSAug v2 analysis and finalized a deployable mixed_feature_gate policy.

Known project findings:

- single SNSAug fine-tuned models did not produce a balanced operating point
- local ignore_mask masking was insufficient
- global geometry/degradation and residual/DCT signals explained much of the failure
- final 0072 mixed_feature_gate is deployable and non-oracle
- selected_selector = mixed_feature_gate
- mean_tampered_recall_gain around 0.408
- mean_valid_iou_gain around 0.262
- mean_synthetic_recall_change around +0.100
- mean_real_fpr_change around -0.025

SIDA is a large multimodal model framework for social-media image deepfake detection, localization, and explanation.

We have not yet evaluated the open-source SIDA-7B implementation on our SNSAug fixed-pair samples.

## Goal

Evaluate whether SIDA-7B/SIDA-style large VLMs experience the same two SNSAug failure types:

- Type A: local overlay / nuisance failure
- Type B: global degradation / geometry failure

This task must support export-only and cached-output evaluation. Do not download or run SIDA inside the main cv-forensics experiment by default.

## Files Codex May Modify

- `tasks/0074-sida7b-snsaug-diagnostic-baseline.md`
- `src/cv_forensics/snsaug_v2_sida7b_diagnostic_baseline.py`
- `scripts/evaluation/run_snsaug_v2_sida7b_diagnostic_baseline.py`
- `scripts/agent/validate_snsaug_v2_sida7b_diagnostic_baseline_config.py`
- `configs/evaluation/snsaug_v2_sida7b_diagnostic_baseline.example.json`
- `tests/test_snsaug_v2_sida7b_diagnostic_baseline.py`
- `docs/snsaug_v2_sida7b_diagnostic_baseline.md`

## Files Claude May Modify

- `tasks/0074-sida7b-snsaug-diagnostic-baseline.md`
- `src/cv_forensics/snsaug_v2_sida7b_diagnostic_baseline.py`
- `scripts/evaluation/run_snsaug_v2_sida7b_diagnostic_baseline.py`
- `scripts/agent/validate_snsaug_v2_sida7b_diagnostic_baseline_config.py`
- `configs/evaluation/snsaug_v2_sida7b_diagnostic_baseline.example.json`
- `tests/test_snsaug_v2_sida7b_diagnostic_baseline.py`
- `docs/snsaug_v2_sida7b_diagnostic_baseline.md`

## Required Behavior

1. Two modes.

- export_sida_eval_subset_mode: export SNSAug image subset and prompt list for an external SIDA runner
- cached_sida_output_mode: read cached SIDA outputs and evaluate them

Default mode must not call network or download weights.

2. Inputs.

Use:

- fixed SNSAug pair root
- meta.jsonl
- images directory
- tamper_masks directory
- optional ignore_masks directory
- 0072 final mixed_feature_gate records and metrics
- optional cached SIDA outputs

3. Export subset.

Export a balanced subset across:

- content_label: real, synthetic, tampered
- Type A local overlay profiles: news_meme_overlay, platform_ui_same_size, tiktok_like, instagram_story_like, youtube_shorts_like, combined_sns_realistic
- Type B geometry/degradation profiles: canvas_9x16_only, resize_crop_pad, zoom_crop, resize_jpeg, screenshot_recapture_light, recompression_light
- clean

Write:

- sida_eval_subset_manifest.jsonl
- sida_prompt_list.jsonl
- run_sida_external_template.sh
- sida_external_output_schema.md

4. Prompt.

Prompt must request SIDA-style output:

- [CLS] classification
- [SEG] mask if tampered
- optional explanation only if description model is used

5. Cached SIDA output schema.

Support JSONL rows with:

- image_id
- image_path
- base_id
- profile
- profile_family
- content_label
- sida_text_output
- sida_pred_class
- sida_p_real optional
- sida_p_synthetic optional
- sida_p_tampered optional
- sida_mask_path optional
- parse_error optional
- mask_missing optional

6. Evaluation.

If cached outputs exist, compute:

- accuracy
- macro_f1
- real_fpr
- synthetic_recall
- tampered_recall
- tampered_valid_mean_iou if masks exist
- mask_missing_rate
- parse_error_rate
- Type A local overlay metrics
- Type B global geometry/degradation metrics
- clean-to-SNS drops
- comparison against 0072 mixed_feature_gate

7. Report.

Write:

- sida7b_diagnostic_records.jsonl
- sida7b_per_profile_metrics.json
- sida7b_type_a_type_b_summary.json
- sida7b_vs_mixed_gate_comparison.json
- sida7b_diagnostic_report.md
- artifact_manifest.json

8. Interpretation rules.

The report must explicitly distinguish:

- SIDA was not run yet: export-only result; no performance conclusion allowed
- SIDA cached outputs were evaluated: performance conclusion allowed

Decision cases:

- sida_handles_both_types
- sida_fails_local_overlay_only
- sida_fails_global_degradation_only
- sida_fails_both_types
- sida_cached_outputs_missing

9. Guardrails.

- no_training = true
- no_finetune = true
- no_network = true in cv-forensics default
- no_download = true in cv-forensics default
- do not modify SIDA weights
- do not modify cv-forensics checkpoints

10. Tests.

- export manifest is balanced
- prompt contains [CLS] and [SEG]
- cached output parser handles missing masks
- class parser maps text to real/synthetic/tampered
- mask IoU handles missing masks
- mixed_gate comparison handles missing profiles
- artifact manifest is written
- export-only report forbids performance conclusion

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_sida7b_diagnostic_baseline_config.py configs/evaluation/snsaug_v2_sida7b_diagnostic_baseline.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_sida7b_diagnostic_baseline.py
- python3 scripts/evaluation/run_snsaug_v2_sida7b_diagnostic_baseline.py --help
- grep -q SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE_OK docs/snsaug_v2_sida7b_diagnostic_baseline.md
- python3 scripts/agent/check_agent_changes.py tasks/0074-sida7b-snsaug-diagnostic-baseline.md

## Marker

SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE_OK
