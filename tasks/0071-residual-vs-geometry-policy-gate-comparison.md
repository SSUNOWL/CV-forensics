# Task 0071: Residual-vs-Geometry Policy Gate Comparison

## Context

0070a corrected residual/DCT analysis produced `mixed_low_level_and_geometry_signal`.
Geometry/crop-scale features ranked highest, but residual DCT and histogram features were also high-ranking.

## Goal

Implement a no-training policy gate comparison using 0070a residual degradation records.

## Files Codex May Modify

- `tasks/0071-residual-vs-geometry-policy-gate-comparison.md`
- `src/cv_forensics/snsaug_v2_policy_gate_comparison.py`
- `scripts/evaluation/run_snsaug_v2_policy_gate_comparison.py`
- `scripts/agent/validate_snsaug_v2_policy_gate_comparison_config.py`
- `configs/evaluation/snsaug_v2_policy_gate_comparison.example.json`
- `tests/test_snsaug_v2_policy_gate_comparison.py`
- `docs/snsaug_v2_policy_gate_comparison.md`

## Files Claude May Modify

- `tasks/0071-residual-vs-geometry-policy-gate-comparison.md`
- `src/cv_forensics/snsaug_v2_policy_gate_comparison.py`
- `scripts/evaluation/run_snsaug_v2_policy_gate_comparison.py`
- `scripts/agent/validate_snsaug_v2_policy_gate_comparison_config.py`
- `configs/evaluation/snsaug_v2_policy_gate_comparison.example.json`
- `tests/test_snsaug_v2_policy_gate_comparison.py`
- `docs/snsaug_v2_policy_gate_comparison.md`

## Required Behavior

1. No training, no finetune, no network, no download.
2. Read 0070a `residual_degradation_records.jsonl`.
3. Compare `fixed_original`, `geometry_feature_gate`, `residual_dct_feature_gate`, `mixed_feature_gate`, and `oracle_best_policy_diagnostic`.
4. Write `policy_gate_records.jsonl`, `policy_gate_metrics.json`, `policy_gate_oracle_gap_summary.json`, `policy_gate_report.md`, and `artifact_manifest.json`.
5. Decide whether deployable gates are promising, only diagnostic recovery exists, or policy gates are insufficient.

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_policy_gate_comparison_config.py configs/evaluation/snsaug_v2_policy_gate_comparison.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_policy_gate_comparison.py
- python3 scripts/evaluation/run_snsaug_v2_policy_gate_comparison.py --help
- grep -q SNSAUG_V2_POLICY_GATE_COMPARISON_OK docs/snsaug_v2_policy_gate_comparison.md
- python3 scripts/agent/check_agent_changes.py tasks/0071-residual-vs-geometry-policy-gate-comparison.md

## Marker

SNSAUG_V2_POLICY_GATE_COMPARISON_OK
