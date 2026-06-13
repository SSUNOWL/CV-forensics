# Task 0074c: Clean-safe SIDA vs Mixed Gate Comparison

## Context

0074b fixed most SIDA vs mixed_feature_gate comparison issues, but audit still reports missing_mixed_gate_profiles = ['clean'].

This is not a real Type A/B comparison failure.

Reason:

- SIDA exported subset includes clean reference rows.
- 0072 final mixed_feature_gate records contain only the 12 SNSAug perturbation profiles.
- Therefore clean is unavailable for mixed_feature_gate but should be treated as reference-only, not fatal.

## Goal

Make SIDA vs mixed_feature_gate comparison clean-safe.

Strict comparison must use Type A and Type B SNSAug profiles only. Clean should be reported separately.

## Files Codex May Modify

- `tasks/0074c-clean-safe-sida-vs-gate-comparison.md`
- `src/cv_forensics/snsaug_v2_sida7b_diagnostic_baseline.py`
- `scripts/evaluation/run_snsaug_v2_sida7b_diagnostic_baseline.py`
- `scripts/agent/validate_snsaug_v2_sida7b_diagnostic_baseline_config.py`
- `configs/evaluation/snsaug_v2_sida7b_diagnostic_baseline.example.json`
- `tests/test_snsaug_v2_sida7b_diagnostic_baseline.py`
- `docs/snsaug_v2_sida7b_diagnostic_baseline.md`

## Files Claude May Modify

- `tasks/0074c-clean-safe-sida-vs-gate-comparison.md`
- `src/cv_forensics/snsaug_v2_sida7b_diagnostic_baseline.py`
- `scripts/evaluation/run_snsaug_v2_sida7b_diagnostic_baseline.py`
- `scripts/agent/validate_snsaug_v2_sida7b_diagnostic_baseline_config.py`
- `configs/evaluation/snsaug_v2_sida7b_diagnostic_baseline.example.json`
- `tests/test_snsaug_v2_sida7b_diagnostic_baseline.py`
- `docs/snsaug_v2_sida7b_diagnostic_baseline.md`

## Required Behavior

1. Clean-safe comparison.

- Exclude clean from strict missing_mixed_gate_profiles check.
- Report clean as reference-only.
- Do not mark comparison incomplete if only clean is missing.

2. Required strict profiles.

Type A local overlay:
- news_meme_overlay
- platform_ui_same_size
- tiktok_like
- instagram_story_like
- youtube_shorts_like
- combined_sns_realistic

Type B global geometry/degradation:
- canvas_9x16_only
- resize_crop_pad
- zoom_crop
- resize_jpeg
- screenshot_recapture_light
- recompression_light

3. Mixed gate metric loading.

Read 0072 final_policy_gate_metrics.json using:

- comparison_against_fixed_original[profile].selected
- per_profile[profile]
- fallback to final_policy_gate_records.jsonl if needed

4. Output.

Write existing outputs plus:

- sida7b_clean_safe_vs_mixed_gate_comparison.json
- sida7b_clean_safe_vs_mixed_gate_comparison.md

5. Decision findings.

Use:

- sida_classification_collapse_synthetic
- sida_local_overlay_worse_than_global
- sida_localization_unavailable_mask_missing
- sida_vs_mixed_gate_strict_comparison_available
- clean_missing_only_expected_for_mixed_gate

6. Tests.

- clean missing does not fail strict comparison
- Type A/B missing still fails strict comparison
- comparison_against_fixed_original schema loads selected metrics
- per_profile fallback works
- clean reference is reported separately

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_sida7b_diagnostic_baseline_config.py configs/evaluation/snsaug_v2_sida7b_diagnostic_baseline.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_sida7b_diagnostic_baseline.py
- python3 scripts/evaluation/run_snsaug_v2_sida7b_diagnostic_baseline.py --help
- grep -q SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE_OK docs/snsaug_v2_sida7b_diagnostic_baseline.md
- python3 scripts/agent/check_agent_changes.py tasks/0074c-clean-safe-sida-vs-gate-comparison.md

## Marker

SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE_OK
