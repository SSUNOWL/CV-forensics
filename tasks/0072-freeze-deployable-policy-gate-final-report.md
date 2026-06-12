# Task 0072: Freeze Deployable Policy Gate and Generate Final Report

## Context

0071a policy gate comparison fixed report/provenance issues and produced a deployable policy gate result.

Observed 0071a result:

- marker = SNSAUG_V2_POLICY_GATE_COMPARISON_OK
- record_count = 3600
- no_training = true
- no_finetune = true
- no_network = true
- no_download = true
- policy_gate_report.md exists
- policy_gate_comparison_report.md exists
- chosen_policy is no longer null
- decision = deployable_policy_gate_promising
- deployable candidate good_profile_count = 8
- mean_tampered_recall_gain around 0.408
- mean_valid_iou_gain around 0.262
- mean_synthetic_recall_change around +0.100
- mean_real_fpr_change around -0.025

The project has moved from model fine-tuning failure analysis to a deployable feature-gated preprocessing/inference policy.

## Goal

Freeze the best deployable policy gate as the final practical SNSAug recovery policy and generate report-ready artifacts.

This is an evaluation/reporting task. Do not train.

## Files Codex May Modify

- `tasks/0072-freeze-deployable-policy-gate-final-report.md`
- `src/cv_forensics/snsaug_v2_deployable_policy_gate_report.py`
- `scripts/evaluation/run_snsaug_v2_deployable_policy_gate_report.py`
- `scripts/agent/validate_snsaug_v2_deployable_policy_gate_report_config.py`
- `configs/evaluation/snsaug_v2_deployable_policy_gate_report.example.json`
- `tests/test_snsaug_v2_deployable_policy_gate_report.py`
- `docs/snsaug_v2_deployable_policy_gate_report.md`

## Files Claude May Modify

- `tasks/0072-freeze-deployable-policy-gate-final-report.md`
- `src/cv_forensics/snsaug_v2_deployable_policy_gate_report.py`
- `scripts/evaluation/run_snsaug_v2_deployable_policy_gate_report.py`
- `scripts/agent/validate_snsaug_v2_deployable_policy_gate_report_config.py`
- `configs/evaluation/snsaug_v2_deployable_policy_gate_report.example.json`
- `tests/test_snsaug_v2_deployable_policy_gate_report.py`
- `docs/snsaug_v2_deployable_policy_gate_report.md`

## Required Behavior

1. No training.

- no_training = true
- no_finetune = true
- no_network = true
- no_download = true

2. Inputs.

Read the latest or configured 0071a policy gate output root:

- policy_gate_records.jsonl
- policy_gate_metrics.json
- policy_gate_oracle_gap_summary.json
- policy_gate_report.md
- policy_gate_comparison_report.md
- artifact_manifest.json

3. Select best deployable candidate.

From policy_gate_metrics.json decision.deployable_candidates:

- choose the highest score candidate if score exists
- otherwise choose the first candidate
- reject diagnostic_only candidates
- reject selectors whose selected rows use oracle-only chosen_policy

4. Freeze deployable policy spec.

Write deployable_policy_gate_spec.json containing:

- selected_selector
- selected_selector_score
- good_profile_count
- good_profiles
- mean_tampered_recall_gain
- mean_valid_iou_gain
- mean_tampered_gap_closure
- mean_valid_iou_gap_closure
- mean_synthetic_recall_change
- mean_real_fpr_change
- chosen_policy_distribution for selected selector
- feature_gate_summary if available
- diagnostic_only = false

5. Export final selected records.

Write final_policy_gate_records.jsonl containing only records from selected_selector.

Each final record must include:

- selector
- chosen_policy
- profile
- content_label
- pred_class or equivalent
- p_real
- p_synthetic
- p_tampered
- valid_iou if available
- diagnostic_only_selector = false

6. Final metrics.

Write final_policy_gate_metrics.json with:

- per-profile metrics for selected selector
- comparison against fixed_original
- comparison against oracle_best_policy_diagnostic when available
- focus profile average
- all profile average

7. Report outputs.

Write:

- final_policy_gate_report.md
- final_policy_gate_notion_summary.md
- final_policy_gate_tables.tsv
- final_policy_gate_records.jsonl
- final_policy_gate_metrics.json
- deployable_policy_gate_spec.json
- artifact_manifest.json

8. Report interpretation.

The report must explain:

- why naive SNSAug fine-tuning failed
- why local nuisance masking was insufficient
- why global geometry/degradation was dominant
- how policy gate combines geometry and residual/DCT signals
- which profiles recovered
- which profiles remain unresolved
- why this is deployable and not oracle

9. Tests.

Tests must cover:

- best deployable candidate selection
- diagnostic candidate rejection
- oracle chosen_policy rejection for deployable export
- final records contain no null chosen_policy
- final records are selected selector only
- artifact manifest written
- dry-run writes no final records

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_deployable_policy_gate_report_config.py configs/evaluation/snsaug_v2_deployable_policy_gate_report.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_deployable_policy_gate_report.py
- python3 scripts/evaluation/run_snsaug_v2_deployable_policy_gate_report.py --help
- grep -q SNSAUG_V2_DEPLOYABLE_POLICY_GATE_REPORT_OK docs/snsaug_v2_deployable_policy_gate_report.md
- python3 scripts/agent/check_agent_changes.py tasks/0072-freeze-deployable-policy-gate-final-report.md

## Marker

SNSAUG_V2_DEPLOYABLE_POLICY_GATE_REPORT_OK
