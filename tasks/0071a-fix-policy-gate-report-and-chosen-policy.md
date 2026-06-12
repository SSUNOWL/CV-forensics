# Task 0071a: Fix Policy Gate Report Name and Chosen Policy Provenance

## Context

0071 policy gate comparison ran successfully and produced valid metrics.

Observed outputs:

- artifact_manifest.json exists
- policy_gate_records.jsonl exists
- policy_gate_metrics.json exists
- policy_gate_oracle_gap_summary.json exists
- record_count = 3600
- decision = deployable_policy_gate_promising
- deployable candidate has good_profile_count = 8

Observed issues:

- audit expected policy_gate_comparison_report.md
- runner wrote policy_gate_report.md or did not write the alias
- records have chosen_policy = None for all rows

These are provenance/reporting issues, not metric failures.

## Goal

Fix 0071 output schema so audits and final reporting are stable.

## Files Codex May Modify

- `tasks/0071a-fix-policy-gate-report-and-chosen-policy.md`
- `src/cv_forensics/snsaug_v2_policy_gate_comparison.py`
- `scripts/evaluation/run_snsaug_v2_policy_gate_comparison.py`
- `scripts/agent/validate_snsaug_v2_policy_gate_comparison_config.py`
- `configs/evaluation/snsaug_v2_policy_gate_comparison.example.json`
- `tests/test_snsaug_v2_policy_gate_comparison.py`
- `docs/snsaug_v2_policy_gate_comparison.md`

## Files Claude May Modify

- `tasks/0071a-fix-policy-gate-report-and-chosen-policy.md`
- `src/cv_forensics/snsaug_v2_policy_gate_comparison.py`
- `scripts/evaluation/run_snsaug_v2_policy_gate_comparison.py`
- `scripts/agent/validate_snsaug_v2_policy_gate_comparison_config.py`
- `configs/evaluation/snsaug_v2_policy_gate_comparison.example.json`
- `tests/test_snsaug_v2_policy_gate_comparison.py`
- `docs/snsaug_v2_policy_gate_comparison.md`

## Required Behavior

1. Report filenames.

Successful actual runs must write both:

- policy_gate_report.md
- policy_gate_comparison_report.md

If both contain the same content, that is acceptable.

2. Artifact manifest.

artifact_manifest.json must include both output paths:

- policy_gate_report
- policy_gate_comparison_report

3. Record provenance.

Every row in policy_gate_records.jsonl must include:

- selector
- chosen_policy
- selector_reason
- diagnostic_only_selector
- geometry_score if applicable
- residual_score if applicable
- selected_by_profile_family if applicable

For fixed_original, chosen_policy should be original.
For geometry_feature_gate, chosen_policy should be the actual policy selected for the sample.
For residual_dct_feature_gate, chosen_policy should be the actual policy selected for the sample.
For mixed_feature_gate, chosen_policy should be the actual policy selected for the sample.
For oracle_best_policy_diagnostic, chosen_policy should be the oracle-selected policy and diagnostic_only_selector must be true.

4. Metrics should remain unchanged or numerically equivalent.

5. Tests.

Tests must cover:

- both report files are written
- artifact manifest lists both report paths
- chosen_policy is not null in policy_gate_records
- fixed_original sets chosen_policy=original
- oracle selector is marked diagnostic_only_selector=true
- dry-run writes no records

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_policy_gate_comparison_config.py configs/evaluation/snsaug_v2_policy_gate_comparison.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_policy_gate_comparison.py
- python3 scripts/evaluation/run_snsaug_v2_policy_gate_comparison.py --help
- grep -q SNSAUG_V2_POLICY_GATE_COMPARISON_OK docs/snsaug_v2_policy_gate_comparison.md
- python3 scripts/agent/check_agent_changes.py tasks/0071a-fix-policy-gate-report-and-chosen-policy.md

## Marker

SNSAUG_V2_POLICY_GATE_COMPARISON_OK
