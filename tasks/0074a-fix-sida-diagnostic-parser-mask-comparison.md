# Task 0074a: Fix SIDA Diagnostic Parser, Mask Accounting, and Mixed-gate Comparison

## Context

0074 SIDA-7B SNSAug diagnostic baseline ran successfully on the 78-image exported SNSAug subset.

Observed current outputs:

- record_count = 78
- metadata_missing_count = 0
- profile_family counts: type_a_local_overlay = 36, type_b_global_geometry_degradation = 36, clean = 6
- label counts: real = 26, synthetic = 26, tampered = 26
- sida_pred_class: real = 58, tampered = 19, unknown/None = 1
- synthetic_recall = 0.0 for clean, type_a, and type_b
- mask_missing_rate = 1.0 for all groups
- current decision = sida_fails_local_overlay_only
- mixed_feature_gate comparison failed because all mixed_gate profile metrics are null

Interpretation:

The SIDA run succeeded as a classification-only diagnostic, but the current report is incomplete.
We must inspect and fix class parsing, mask accounting, and mixed_feature_gate comparison before using this as final report evidence.

## Goal

Produce a corrected SIDA diagnostic report that separates model behavior from parser/wrapper limitations.

## Files Codex May Modify

- `tasks/0074a-fix-sida-diagnostic-parser-mask-comparison.md`
- `src/cv_forensics/snsaug_v2_sida7b_diagnostic_baseline.py`
- `scripts/evaluation/run_snsaug_v2_sida7b_diagnostic_baseline.py`
- `scripts/agent/validate_snsaug_v2_sida7b_diagnostic_baseline_config.py`
- `configs/evaluation/snsaug_v2_sida7b_diagnostic_baseline.example.json`
- `tests/test_snsaug_v2_sida7b_diagnostic_baseline.py`
- `docs/snsaug_v2_sida7b_diagnostic_baseline.md`

## Files Claude May Modify

- `tasks/0074a-fix-sida-diagnostic-parser-mask-comparison.md`
- `src/cv_forensics/snsaug_v2_sida7b_diagnostic_baseline.py`
- `scripts/evaluation/run_snsaug_v2_sida7b_diagnostic_baseline.py`
- `scripts/agent/validate_snsaug_v2_sida7b_diagnostic_baseline_config.py`
- `configs/evaluation/snsaug_v2_sida7b_diagnostic_baseline.example.json`
- `tests/test_snsaug_v2_sida7b_diagnostic_baseline.py`
- `docs/snsaug_v2_sida7b_diagnostic_baseline.md`

## Required Behavior

1. Raw output audit.

Add raw text summary fields:

- raw_contains_cls_count
- raw_contains_seg_count
- raw_contains_real_count
- raw_contains_synthetic_count
- raw_contains_fully_synthetic_count
- raw_contains_generated_count
- raw_contains_tampered_count
- raw_contains_manipulated_count

2. Robust class parser.

Improve class parsing from sida_text_output:

- recognize real
- recognize synthetic
- recognize fully synthetic
- recognize generated / ai-generated as synthetic when not paired with tampered/manipulated
- recognize tampered / manipulated / altered as tampered
- prefer explicit [CLS] sentence if present
- mark ambiguous as unknown with parse_error or parse_warning

3. Mask accounting.

Report whether raw text contains [SEG].

If raw text contains [SEG] but sida_mask_path is missing, set:

- mask_extraction_failed = true

If raw text does not contain [SEG], set:

- mask_not_requested_or_not_generated = true

Do not claim localization failure when masks are missing.
Report localization as unavailable.

4. Mixed-gate comparison fix.

Read 0072 final_policy_gate_metrics.json and compare by profile.

The current comparison has missing_mixed_gate_profiles for all profiles, which is wrong.
Fix schema handling so mixed_feature_gate per-profile metrics are loaded.

Required comparison fields per profile:

- sida_tampered_recall
- sida_synthetic_recall
- sida_real_fpr
- sida_valid_iou if available
- mixed_gate_tampered_recall
- mixed_gate_synthetic_recall
- mixed_gate_real_fpr
- mixed_gate_valid_iou
- delta_sida_minus_mixed_gate for available metrics

5. Decision refinement.

Current decision sida_fails_local_overlay_only is too narrow if synthetic_recall is 0.0 and masks are missing.

Add decision labels:

- sida_classification_collapse_synthetic
- sida_local_overlay_worse_than_global
- sida_global_degradation_partially_failed
- sida_localization_unavailable_mask_missing
- sida_vs_mixed_gate_incomplete
- sida_vs_mixed_gate_available

The final report may include multiple findings rather than one decision string.

6. Outputs.

Write existing outputs plus:

- sida7b_raw_output_audit.json
- sida7b_corrected_type_a_type_b_summary.json
- sida7b_corrected_vs_mixed_gate_comparison.json
- sida7b_corrected_diagnostic_report.md

7. Tests.

Tests must cover:

- parser recognizes fully synthetic
- parser recognizes ai-generated as synthetic when not tampered
- parser recognizes tampered/manipulated
- ambiguous parse becomes unknown
- [SEG] with missing mask sets mask_extraction_failed
- missing [SEG] sets mask_not_requested_or_not_generated
- 0072 mixed gate metrics load by profile
- corrected report does not claim localization failure when masks missing

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_sida7b_diagnostic_baseline_config.py configs/evaluation/snsaug_v2_sida7b_diagnostic_baseline.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_sida7b_diagnostic_baseline.py
- python3 scripts/evaluation/run_snsaug_v2_sida7b_diagnostic_baseline.py --help
- grep -q SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE_OK docs/snsaug_v2_sida7b_diagnostic_baseline.md
- python3 scripts/agent/check_agent_changes.py tasks/0074a-fix-sida-diagnostic-parser-mask-comparison.md

## Marker

SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE_OK
