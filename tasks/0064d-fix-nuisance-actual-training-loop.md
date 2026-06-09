# Task 0064d: Fix SNSAug v2 Nuisance Actual Training Loop

## Context

0064 nuisance-mask finetune implementation validates and writes checkpoints, and 0064c evaluator now compares nuisance checkpoints successfully.

However, the supposed 0064 nuisance 30x3 run is not a valid 30x3 training run.

Observed evidence:

- artifact marker: SNSAUG_V2_NUISANCE_MASK_FINETUNE_OK
- training_started = true
- checkpoint_written = true
- model_version = snsaug_aware_nuisance_multihead_forensics_v1
- but training_log rows = 1
- phase_counts = {1: 1}
- checkpoint global_step = 1
- checkpoint phase = 1

The 0064 comparison result is therefore an internal implementation/training-loop issue, not a valid model result.

## Goal

Fix the nuisance actual training loop so max_steps_per_phase and phase_1/2/3 step counts are honored.

The runner must perform real training steps for all requested phases and write checkpoints with correct global_step.

## Files Codex May Modify

- `tasks/0064d-fix-nuisance-actual-training-loop.md`
- `src/cv_forensics/snsaug_v2_nuisance_finetune.py`
- `scripts/training/run_snsaug_v2_nuisance_finetune.py`
- `scripts/agent/validate_snsaug_v2_nuisance_finetune_config.py`
- `configs/training/snsaug_v2_nuisance_finetune.example.json`
- `tests/test_snsaug_v2_nuisance_finetune.py`
- `docs/snsaug_v2_nuisance_finetune.md`

## Files Claude May Modify

- `tasks/0064d-fix-nuisance-actual-training-loop.md`
- `src/cv_forensics/snsaug_v2_nuisance_finetune.py`
- `scripts/training/run_snsaug_v2_nuisance_finetune.py`
- `scripts/agent/validate_snsaug_v2_nuisance_finetune_config.py`
- `configs/training/snsaug_v2_nuisance_finetune.example.json`
- `tests/test_snsaug_v2_nuisance_finetune.py`
- `docs/snsaug_v2_nuisance_finetune.md`

## Required Behavior

1. Dry-run remains unchanged.

When --dry-run is passed:
- no training
- no checkpoint writes
- training_started = false
- checkpoint_written = false

2. Actual run must honor phase step config.

For config:
- phase_1_max_steps = A
- phase_2_max_steps = B
- phase_3_max_steps = C

Expected:
- training_log rows >= A+B+C
- phase_counts[1] = A
- phase_counts[2] = B
- phase_counts[3] = C
- artifact global_step >= A+B+C
- checkpoint global_step >= A+B+C

3. Tiny 5x3 run must produce at least 15 rows.

Expected tiny audit:
- training_log rows >= 15
- phase_counts = {1:5, 2:5, 3:5}
- best/last checkpoint exist
- checkpoint has model_state_dict
- checkpoint global_step >= 15

4. 30x3 run must produce at least 90 rows.

Expected 30x3 audit:
- training_log rows >= 90
- phase_counts = {1:30, 2:30, 3:30}
- checkpoint global_step >= 90

5. Loss logging must include:

- class_loss
- tamper_mask_loss
- sns_nuisance_mask_loss
- global_degradation_loss
- hardneg_loss
- clean_sns_class_consistency_loss
- total_loss

6. Preserve guardrails.

- no network
- no download
- train-only manifest
- no validation samples for training
- outputs outside repo

7. Do not change checkpoint comparison evaluator in this task.

0064c already fixed evaluation schema. This task is only about nuisance finetune training loop.

## Tests

Update tests/test_snsaug_v2_nuisance_finetune.py:

- actual tiny config with 1/1/1 writes exactly 3 or more step rows
- actual tiny config with 2/2/2 writes at least 6 step rows
- phase_counts match requested phase steps
- checkpoint global_step matches total executed steps
- dry-run writes no checkpoint
- train split only guard still works
- checkpoint contains model_state_dict

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_nuisance_finetune_config.py configs/training/snsaug_v2_nuisance_finetune.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_nuisance_finetune.py
- python3 scripts/training/run_snsaug_v2_nuisance_finetune.py --help
- grep -q SNSAUG_V2_NUISANCE_MASK_FINETUNE_OK docs/snsaug_v2_nuisance_finetune.md
- python3 scripts/agent/check_agent_changes.py tasks/0064d-fix-nuisance-actual-training-loop.md

## Marker

SNSAUG_V2_NUISANCE_MASK_FINETUNE_OK
