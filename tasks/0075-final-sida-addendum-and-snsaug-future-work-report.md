# Task 0075: Final SIDA Addendum and SNSAug Future Work Report

## Context

0072 finalized the deployable `mixed_feature_gate`.

0074, 0074a, and 0074c evaluated SIDA-7B/SIDA-style cached outputs on an SNSAug subset and fixed parser, mask-accounting, and clean-safe mixed-gate comparison issues.

Known SIDA constraints:

- The cached SIDA run is classification-only because masks are missing.
- Do not claim SIDA localization performance.
- SIDA synthetic recall is 0.0 because raw output contains no synthetic/generated phrases.
- Type A local overlay has lower SIDA tampered recall than Type B global geometry/degradation.
- Clean is reference-only for SIDA and is excluded from strict mixed-gate comparison.

## Goal

Generate final addendum artifacts:

- `final_sida_addendum_report.md`
- `final_sida_addendum_notion_summary.md`
- `final_sida_vs_mixed_gate_table.tsv`
- `future_work_snsaware_model.md`
- `artifact_manifest.json`

This is a reporting-only task. Do not train, download, or use the network.

## Files Codex May Modify

- `tasks/0075-final-sida-addendum-and-snsaug-future-work-report.md`
- `src/cv_forensics/snsaug_v2_final_sida_addendum.py`
- `scripts/evaluation/run_snsaug_v2_final_sida_addendum.py`
- `scripts/agent/validate_snsaug_v2_final_sida_addendum_config.py`
- `configs/evaluation/snsaug_v2_final_sida_addendum.example.json`
- `tests/test_snsaug_v2_final_sida_addendum.py`
- `docs/snsaug_v2_final_sida_addendum.md`

## Files Claude May Modify

- `tasks/0075-final-sida-addendum-and-snsaug-future-work-report.md`
- `src/cv_forensics/snsaug_v2_final_sida_addendum.py`
- `scripts/evaluation/run_snsaug_v2_final_sida_addendum.py`
- `scripts/agent/validate_snsaug_v2_final_sida_addendum_config.py`
- `configs/evaluation/snsaug_v2_final_sida_addendum.example.json`
- `tests/test_snsaug_v2_final_sida_addendum.py`
- `docs/snsaug_v2_final_sida_addendum.md`

## Required Behavior

1. Inputs.

Read configured 0072 final mixed-gate output root:

- `final_policy_gate_metrics.json`
- `final_policy_gate_report.md`
- `artifact_manifest.json`

Read configured 0074/0074c SIDA diagnostic output root:

- `sida7b_corrected_type_a_type_b_summary.json`
- `sida7b_clean_safe_vs_mixed_gate_comparison.json`
- `sida7b_raw_output_audit.json`
- `artifact_manifest.json`

2. Safety guardrails.

Require:

- `no_training = true`
- `no_finetune = true`
- `no_network = true`
- `no_download = true`

Do not run SIDA. Do not train. Do not download. Do not access the network.

3. Final addendum report.

The report must include:

- What SIDA-7B was used for.
- What was actually measured.
- What was not measured because masks are missing.
- Type A vs Type B SIDA classification result.
- Comparison with 0072 `mixed_feature_gate` on Type A/B profiles.
- Clean handling: clean is SIDA reference-only and excluded from strict comparison.
- Future work summary.

The report must not claim SIDA localization performance.

4. Notion summary.

Write a compact final summary suitable for project notes, emphasizing:

- 0072 deployable gate remains the practical offline recovery artifact.
- SIDA cached run is a classification diagnostic only.
- Local overlays remain harder for SIDA than global geometry/degradation.
- Synthetic recall collapse is a parser/output limitation observed in the cached text outputs.

5. TSV comparison table.

Write one row per strict Type A/B profile with:

- `profile`
- `profile_family`
- `sida_tampered_recall`
- `sida_synthetic_recall`
- `sida_real_fpr`
- `sida_valid_iou`
- `mixed_gate_tampered_recall`
- `mixed_gate_synthetic_recall`
- `mixed_gate_real_fpr`
- `mixed_gate_valid_iou`
- `tampered_recall_delta_sida_minus_gate`
- `valid_iou_delta_sida_minus_gate`

Do not include `clean` as a strict comparison row.

6. Future work report.

Write `future_work_snsaware_model.md` with:

- SIDA-13B or SIDA-description rerun for better mask output.
- VLM-guided nuisance preprocessor.
- SNS-aware dual-branch model with local nuisance branch plus geometry/degradation residual branch.
- Distillation from SIDA-like VLM into lightweight mixed gate / residual model.

7. Artifact manifest.

Write `artifact_manifest.json` containing:

- marker
- input roots
- output paths
- safety flags
- decision findings from SIDA summary
- strict comparison availability
- no training/download/network indicators

## Tests

Tests must cover:

- example config validates
- report contains all required interpretation sections
- mask-missing language forbids localization claims
- TSV excludes clean and includes Type A/B profiles
- future-work report includes all required directions
- artifact manifest is written with safety flags
- dry-run writes no output artifacts

## Validation Commands

- `python3 scripts/agent/validate_snsaug_v2_final_sida_addendum_config.py configs/evaluation/snsaug_v2_final_sida_addendum.example.json`
- `CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_final_sida_addendum.py`
- `python3 scripts/evaluation/run_snsaug_v2_final_sida_addendum.py --help`
- `grep -q SNSAUG_V2_FINAL_SIDA_ADDENDUM_OK docs/snsaug_v2_final_sida_addendum.md`
- `python3 scripts/agent/check_agent_changes.py tasks/0075-final-sida-addendum-and-snsaug-future-work-report.md`

## Marker

SNSAUG_V2_FINAL_SIDA_ADDENDUM_OK
