# Task 0077: Final Cross-model Comparison Matrix

## Context

We now have:

- 0072 final `mixed_feature_gate` output.
- 0076 unified SIDA-7B vs mixed gate comparison.
- Optional pre-SNS baseline and failed fine-tune outputs from earlier runs.

The goal is to generate final report-ready comparison artifacts without training, downloading, or network access.

## Role

Codex is the task writer, implementation worker, strict reviewer, and limited repair manager because Claude Code is unavailable.

## Goal

Generate:

- `final_cross_model_comparison_matrix.json`
- `final_cross_model_comparison_matrix.md`
- `final_cross_model_comparison_table.tsv`
- `final_claims_for_report.md`
- `artifact_manifest.json`

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0072-freeze-deployable-policy-gate-final-report.md`
- `tasks/0074-sida7b-snsaug-diagnostic-baseline.md`
- `tasks/0074c-clean-safe-sida-vs-gate-comparison.md`
- `tasks/0075-final-sida-addendum-and-snsaug-future-work-report.md`
- `tasks/0077-final-cross-model-comparison-matrix.md`
- `docs/snsaug_v2_deployable_policy_gate_report.md`
- `docs/snsaug_v2_final_sida_addendum.md`
- `configs/evaluation/snsaug_v2_deployable_policy_gate_report.example.json`
- `configs/evaluation/snsaug_v2_final_sida_addendum.example.json`
- `src/cv_forensics/snsaug_v2_deployable_policy_gate_report.py`
- `src/cv_forensics/snsaug_v2_final_sida_addendum.py`
- `scripts/evaluation/run_snsaug_v2_deployable_policy_gate_report.py`
- `scripts/evaluation/run_snsaug_v2_final_sida_addendum.py`
- `scripts/agent/validate_snsaug_v2_deployable_policy_gate_report_config.py`
- `scripts/agent/validate_snsaug_v2_final_sida_addendum_config.py`
- `tests/test_snsaug_v2_deployable_policy_gate_report.py`
- `tests/test_snsaug_v2_final_sida_addendum.py`

## Files Codex May Modify

- `tasks/0077-final-cross-model-comparison-matrix.md`
- `src/cv_forensics/snsaug_v2_final_cross_model_comparison.py`
- `scripts/evaluation/run_snsaug_v2_final_cross_model_comparison.py`
- `scripts/agent/validate_snsaug_v2_final_cross_model_comparison_config.py`
- `configs/evaluation/snsaug_v2_final_cross_model_comparison.example.json`
- `tests/test_snsaug_v2_final_cross_model_comparison.py`
- `docs/snsaug_v2_final_cross_model_comparison.md`

## Files Claude May Modify

- `tasks/0077-final-cross-model-comparison-matrix.md`
- `src/cv_forensics/snsaug_v2_final_cross_model_comparison.py`
- `scripts/evaluation/run_snsaug_v2_final_cross_model_comparison.py`
- `scripts/agent/validate_snsaug_v2_final_cross_model_comparison_config.py`
- `configs/evaluation/snsaug_v2_final_cross_model_comparison.example.json`
- `tests/test_snsaug_v2_final_cross_model_comparison.py`
- `docs/snsaug_v2_final_cross_model_comparison.md`

## Forbidden Actions

- Do not train or fine-tune models.
- Do not download datasets, checkpoints, or packages.
- Do not access the network.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not write checkpoints, dataset files, or real evaluation outputs.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not install packages.
- Do not create large files.
- Do not claim `mixed_feature_gate` is a drop-in replacement for SIDA.

## Important Project Facts

- SIDA and `mixed_feature_gate` are not identical systems.
- SIDA emits 3-way labels and masks.
- `mixed_feature_gate` is a deployable policy-gated recovery system.
- The fair comparison is split across synthetic preservation, `real_fpr`, `tampered_recall`, and tampered localization.
- Baseline/fine-tune rows are context for the failure path and must not become the primary claim.
- Missing or unparsable baseline/fine-tune metrics must be represented as `null` in JSON and `NA` in Markdown/TSV.

## Required Behavior

1. Safety guardrails.

- `no_training = true`
- `no_finetune = true`
- `no_network = true`
- `no_download = true`
- Do not train, download, or access the network.

2. Inputs.

Read configured 0076 unified SIDA/gate output root:

- `unified_sida_gate_comparison_summary.json`
- `unified_sida_gate_profile_table.tsv`
- `unified_sida_gate_comparison_report.md`
- `artifact_manifest.json`

Read configured 0072 final mixed-gate output root when needed:

- `final_policy_gate_metrics.json`
- `deployable_policy_gate_spec.json`
- `artifact_manifest.json`

Optionally read baseline/fine-tune roots if configured and robustly parseable. If metrics cannot be loaded with confidence, keep their metric fields as `null` in JSON and `NA` in Markdown/TSV.

3. Main comparison table.

The main table must contain only:

- `SIDA-7B`
- `mixed_feature_gate`

Compare by:

- synthetic preservation / `synthetic_recall`
- `real_fpr`
- `tampered_recall`
- tampered localization / `valid_iou`

Include SIDA mask-specific context when available:

- `synthetic_tampered_fpr`
- `detected_only_iou`
- `ignore_capture`

Use family rows:

- `type_a_local_overlay`
- `type_b_global_geometry_degradation`
- `strict_type_a_plus_type_b`

4. Ablation/context table.

The ablation/context table must contain:

- `pre_sns_baseline`
- `failed_single_model_finetune`
- `mixed_feature_gate`

Use baseline/fine-tune only to explain the failure path and why the gate was needed. Do not make them the primary claim.

5. Interpretation guardrails.

Reports must state:

- SIDA and `mixed_feature_gate` are not identical systems.
- SIDA emits 3-way labels and masks.
- `mixed_feature_gate` is a deployable policy-gated recovery system.
- The fair comparison is split across synthetic preservation, real FPR, tampered recall, and tampered localization.
- Do not overclaim `mixed_feature_gate` is a drop-in replacement for SIDA.

6. Outputs.

Write all outputs under configured `output_root`:

- `final_cross_model_comparison_matrix.json`
- `final_cross_model_comparison_matrix.md`
- `final_cross_model_comparison_table.tsv`
- `final_claims_for_report.md`
- `artifact_manifest.json`

The JSON must include:

- marker
- source roots
- model entries for all main and ablation rows
- recommendations with `main_table`, `ablation_table`, and `do_not_overclaim`
- safety flags

The Markdown must include:

- recommendation
- main table
- ablation/context table
- interpretation
- do-not-overclaim bullets

The TSV must include the main table rows in report-ready form.

## Tests

Tests must cover:

- example config validates
- safety flags are required
- main table contains only SIDA-7B and `mixed_feature_gate`
- ablation table contains baseline, failed fine-tune, and mixed gate
- baseline/fine-tune missing or unparsable metrics become NA/null rather than failing
- Markdown includes non-drop-in replacement language
- artifact manifest is written
- dry-run writes no output artifacts

## Validation Commands

- `python3 scripts/agent/validate_snsaug_v2_final_cross_model_comparison_config.py configs/evaluation/snsaug_v2_final_cross_model_comparison.example.json`
- `python3 tests/test_snsaug_v2_final_cross_model_comparison.py`
- `python3 scripts/evaluation/run_snsaug_v2_final_cross_model_comparison.py --help`
- `grep -q SNSAUG_V2_0077_FINAL_CROSS_MODEL_COMPARISON_OK docs/snsaug_v2_final_cross_model_comparison.md`
- `python3 scripts/agent/check_agent_changes.py tasks/0077-final-cross-model-comparison-matrix.md`

## Acceptance Criteria

- The example config validates.
- Plain Python tests pass without third-party dependencies.
- CLI help exits successfully.
- Documentation includes the marker.
- Changed files are limited to the allowed list.
- Generated JSON includes source roots, main and ablation entries, recommendations, do-not-overclaim guidance, and safety flags.
- Generated Markdown includes recommendation, main table, ablation/context table, interpretation, and do-not-overclaim bullets.
- Generated TSV includes report-ready main table rows.
- Dry-run returns a plan without writing output artifacts.

## Stop Condition

Stop after validation and strict Codex review. Do not commit automatically.

## Marker

SNSAUG_V2_0077_FINAL_CROSS_MODEL_COMPARISON_OK
