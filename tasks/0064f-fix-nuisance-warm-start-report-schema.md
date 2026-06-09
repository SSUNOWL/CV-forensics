# Task 0064f: Fix SNSAug v2 Nuisance Warm-start Report Schema

## Context

0064e nuisance tiny 5x3 actual run executed correctly, but audit failed because warm_start_report.json omitted required provenance fields.

Observed successful parts:

- artifact_manifest.json exists
- training_log.jsonl exists
- loss_breakdown.json exists
- per_phase_metrics.json exists
- clean_validation_metrics.json exists
- snsaug_0058c_metrics.json exists
- warm_start_report.json exists
- training_started = true
- checkpoint_written = true
- global_step = 15
- tensor_total_numel = 5403589
- training_log rows = 15
- phase_counts = {1:5, 2:5, 3:5}
- best/last checkpoints exist and both have global_step = 15

Observed failure:

- warm_start_report.warm_start_checkpoint_path = null
- warm_start_report.loaded_numel = 5399224
- warm_start_report.total_numel = 5403589
- warm_start_report.loaded_ratio = null
- audit fails with: warm_start_report.loaded_ratio missing

This is a reporting/schema bug, not a training failure. loaded_ratio should be computed as loaded_numel / total_numel, approximately 0.9992 in the observed run.

## Goal

Fix 0064e nuisance fine-tune warm-start report so audits can verify reproducibility and warm-start coverage.

## Files Codex May Modify

- `tasks/0064f-fix-nuisance-warm-start-report-schema.md`
- `src/cv_forensics/snsaug_v2_nuisance_finetune.py`
- `scripts/training/run_snsaug_v2_nuisance_finetune.py`
- `scripts/agent/validate_snsaug_v2_nuisance_finetune_config.py`
- `configs/training/snsaug_v2_nuisance_finetune.example.json`
- `tests/test_snsaug_v2_nuisance_finetune.py`
- `docs/snsaug_v2_nuisance_finetune.md`

## Files Claude May Modify

- `tasks/0064f-fix-nuisance-warm-start-report-schema.md`
- `src/cv_forensics/snsaug_v2_nuisance_finetune.py`
- `scripts/training/run_snsaug_v2_nuisance_finetune.py`
- `scripts/agent/validate_snsaug_v2_nuisance_finetune_config.py`
- `configs/training/snsaug_v2_nuisance_finetune.example.json`
- `tests/test_snsaug_v2_nuisance_finetune.py`
- `docs/snsaug_v2_nuisance_finetune.md`

## Required Behavior

1. warm_start_report.json must always include:

- warm_start_checkpoint_path
- warm_start_checkpoint_exists
- loaded_numel
- total_numel
- loaded_ratio
- missing_keys
- unexpected_keys
- loaded_key_count
- total_key_count

2. loaded_ratio must be finite when total_numel > 0.

Formula:

    loaded_ratio = loaded_numel / total_numel

3. warm_start_checkpoint_path must be copied from config when configured.

If the run uses base_model_bundle_path instead of warm_start_checkpoint_path, report that path as base_model_bundle_path and set warm_start_source accordingly.

4. artifact_manifest.json should also include a compact warm_start_summary:

- warm_start_checkpoint_path
- loaded_numel
- total_numel
- loaded_ratio
- missing_key_count
- unexpected_key_count

5. The tiny audit should pass when loaded_ratio >= require_warm_start_loaded_numel_ratio_gte.

6. Do not change comparison evaluator in this task.

7. Do not train long. Do not download. Do not use network.

## Tests

Update tests/test_snsaug_v2_nuisance_finetune.py:

- warm_start_report contains warm_start_checkpoint_path
- warm_start_report contains finite loaded_ratio
- loaded_ratio equals loaded_numel / total_numel
- artifact_manifest contains warm_start_summary
- tiny actual run audit passes when loaded_ratio is high enough
- dry-run still writes no checkpoint

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_nuisance_finetune_config.py configs/training/snsaug_v2_nuisance_finetune.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_nuisance_finetune.py
- python3 scripts/training/run_snsaug_v2_nuisance_finetune.py --help
- grep -q SNSAUG_V2_NUISANCE_MASK_FINETUNE_OK docs/snsaug_v2_nuisance_finetune.md
- python3 scripts/agent/check_agent_changes.py tasks/0064f-fix-nuisance-warm-start-report-schema.md

## Marker

SNSAUG_V2_NUISANCE_MASK_FINETUNE_OK
