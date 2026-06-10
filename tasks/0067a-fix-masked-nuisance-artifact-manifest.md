# Task 0067a: Fix Masked Nuisance Inference Artifact Manifest

## Context

0067 masked nuisance tiny inference ran successfully after config schema fix.

Observed successful outputs:

- model_eval_records_masked.jsonl exists
- per_policy_per_profile_metrics.json exists
- clean_vs_sns_masked_delta.json exists
- masked_nuisance_inference_summary.json exists
- masked_nuisance_inference_report.md exists
- visual_gallery_manifest.json exists
- records = 3600
- status = 0
- no_training = true

Observed failure:

- artifact_manifest.json is missing
- audit fails only because artifact_manifest.json is expected but not written

Therefore this is not an inference/data/model failure. It is an output manifest bug.

## Goal

Update masked nuisance inference so every successful actual run writes artifact_manifest.json.

## Files Codex May Modify

- `tasks/0067a-fix-masked-nuisance-artifact-manifest.md`
- `src/cv_forensics/snsaug_v2_masked_nuisance_inference.py`
- `scripts/evaluation/run_snsaug_v2_masked_nuisance_inference.py`
- `scripts/agent/validate_snsaug_v2_masked_nuisance_inference_config.py`
- `configs/evaluation/snsaug_v2_masked_nuisance_inference.example.json`
- `tests/test_snsaug_v2_masked_nuisance_inference.py`
- `docs/snsaug_v2_masked_nuisance_inference.md`

## Files Claude May Modify

- `tasks/0067a-fix-masked-nuisance-artifact-manifest.md`
- `src/cv_forensics/snsaug_v2_masked_nuisance_inference.py`
- `scripts/evaluation/run_snsaug_v2_masked_nuisance_inference.py`
- `scripts/agent/validate_snsaug_v2_masked_nuisance_inference_config.py`
- `configs/evaluation/snsaug_v2_masked_nuisance_inference.example.json`
- `tests/test_snsaug_v2_masked_nuisance_inference.py`
- `docs/snsaug_v2_masked_nuisance_inference.md`

## Required Behavior

1. Successful actual run must write artifact_manifest.json.

The artifact manifest must include:

- marker
- config_path
- output_root
- pair_root
- best_bundle_path
- policies
- profiles
- subset_only
- max_rows
- no_training
- no_finetune
- no_network
- no_download
- inference_started
- training_started
- record_count
- row_count
- policy_counts
- profile_counts
- label_counts
- output_paths
- warning_count
- warnings

2. artifact_manifest.json must be listed in the summary output_paths if the code reports output paths.

3. Dry-run must not write artifact_manifest.json unless existing dry-run behavior explicitly writes no files.

4. Tests must cover:

- tiny actual run writes artifact_manifest.json
- artifact manifest contains record_count
- artifact manifest contains output_paths
- dry-run does not write inference records
- output files listed in artifact manifest exist

5. Do not change masking policies or model inference logic in this task.

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_masked_nuisance_inference_config.py configs/evaluation/snsaug_v2_masked_nuisance_inference.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_masked_nuisance_inference.py
- python3 scripts/evaluation/run_snsaug_v2_masked_nuisance_inference.py --help
- grep -q SNSAUG_V2_MASKED_NUISANCE_INFERENCE_OK docs/snsaug_v2_masked_nuisance_inference.md
- python3 scripts/agent/check_agent_changes.py tasks/0067a-fix-masked-nuisance-artifact-manifest.md

## Marker

SNSAUG_V2_MASKED_NUISANCE_INFERENCE_OK
