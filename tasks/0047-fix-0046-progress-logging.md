# Task 0047: Fix 0046 Progress Logging for Tile Localizer v2 Training

## Task Title

Repair long-run progress visibility for `0046-pre-sns-highres-forensic-tile-localizer-v2` by adding periodic progress logging for pre-SNS v3 tile localizer v2 training.

## Role

Codex-only implementation worker, reviewer, and limited repair manager operating under the repository contract and Codex-only workflow.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0046-pre-sns-highres-forensic-tile-localizer-v2.md`
- `tasks/0047-fix-0046-progress-logging.md`
- `configs/training/pre_sns_v3_tile_localizer_v2_train.example.json`
- `scripts/agent/validate_pre_sns_v3_tile_localizer_v2_train_config.py`
- `scripts/training/train_pre_sns_v3_tile_localizer_v2.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_training.py`
- `tests/test_pre_sns_v3_tile_localizer_v2_training.py`
- `docs/pre_sns_v3_tile_localizer_v2_training.md`
- `src/cv_forensics/__init__.py`

## Files Codex May Modify

- `configs/training/pre_sns_v3_tile_localizer_v2_train.example.json`
- `scripts/agent/validate_pre_sns_v3_tile_localizer_v2_train_config.py`
- `scripts/training/train_pre_sns_v3_tile_localizer_v2.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_training.py`
- `tests/test_pre_sns_v3_tile_localizer_v2_training.py`
- `docs/pre_sns_v3_tile_localizer_v2_training.md`
- `src/cv_forensics/__init__.py`

## Forbidden Actions

- Do not modify any file outside the allowed modify list.
- Do not modify `tasks/0046-pre-sns-highres-forensic-tile-localizer-v2.md`.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not use network resources.
- Do not install packages.
- Do not download datasets.
- Do not run real training beyond the tiny existing local test fixture path.
- Do not add SNS augmentation.
- Do not write outputs or checkpoints inside the repository.
- Do not create large files.
- Do not commit changes.

## Important Project Facts

- This repository is operating in Codex-only mode, but the task/review workflow must remain auditable.
- The project contract requires lightweight, reversible changes and forbids protected path access and network actions.
- `PRE_SNS_V3_TILE_LOCALIZER_V2_TRAINING_OK` remains the completion marker for successful training and must stay documented.
- Pre-SNS tile localizer v2 training writes only to approved external roots and must remain outside the repository.
- Long-running training needs visible progress before epoch completion so users can monitor percent complete, ETA, current epoch/step, and recent loss.

## Implementation Requirements

- In `src/cv_forensics/pre_sns_v3_tile_localizer_v2_training.py`, add periodic progress logging to:
  - `approved_run_root/progress.json`
  - `approved_run_root/progress.jsonl`
- Emit progress updates at minimum:
  - at training start
  - every `progress_log_interval_steps`
  - every epoch end
  - before validation
  - after validation
  - at training completion
  - on handled failure when possible
- Progress payload must include:
  - `marker` set to `PRE_SNS_V3_TILE_LOCALIZER_V2_TRAINING_PROGRESS`
  - `train_name`
  - `pid`
  - `device`
  - `epoch`
  - `epochs`
  - `step_in_epoch`
  - `steps_per_epoch`
  - `global_step`
  - `total_steps`
  - `percent_complete`
  - `elapsed_sec`
  - `eta_sec`
  - `latest_loss`
  - `latest_bce_loss` when available
  - `latest_dice_loss` when available
  - `latest_tversky_loss` when available
  - `latest_boundary_loss` when available
  - `latest_empty_mask_loss` when available
  - `learning_rate`
  - `batch_size`
  - `tile_size`
  - `max_tiles_train`
  - `max_tiles_val`
  - `gpu_memory_allocated_mb` when CUDA is active
  - `gpu_memory_reserved_mb` when CUDA is active
- Add config support with defaults:
  - `progress_log_interval_steps`: `100`
  - `progress_write_json`: `true`
  - `progress_write_jsonl`: `true`
  - `stdout_progress_interval_steps`: `100`
- Add the progress fields to the example training config.
- Update validation so the progress fields are accepted and validated as positive integers or booleans as appropriate.
- Ensure `scripts/training/train_pre_sns_v3_tile_localizer_v2.py` preserves progress output and any progress `print(...)` calls use `flush=True`.
- Extend tests to cover:
  - `progress.json` written during tiny actual fixture training
  - `progress.jsonl` contains at least one line
  - `percent_complete` exists and is numeric
  - no-write dry-run does not write progress files
  - validator accepts the progress fields
  - validator rejects an invalid negative interval
- Update documentation to describe `progress.json` and `progress.jsonl` while keeping the success marker documentation unchanged.

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_v3_tile_localizer_v2_train_config.py configs/training/pre_sns_v3_tile_localizer_v2_train.example.json
CUDA_VISIBLE_DEVICES="" python3 tests/test_pre_sns_v3_tile_localizer_v2_training.py
grep -q PRE_SNS_V3_TILE_LOCALIZER_V2_TRAINING_OK docs/pre_sns_v3_tile_localizer_v2_training.md
python3 scripts/agent/check_agent_changes.py tasks/0047-fix-0046-progress-logging.md
```

## Acceptance Criteria

- The allowed training path writes externally visible progress snapshots and append-only progress history during real non-dry-run training.
- Progress records contain the required marker and required numeric/context fields, with CUDA memory metrics populated only when applicable.
- Dry-run mode does not create progress files.
- The example config and validator support the new progress controls with the required defaults and guardrails.
- CLI output remains flushed so progress lines are visible during long runs.
- Tests and validation commands pass without touching protected paths or writing repository-local outputs/checkpoints.

## Stop Condition

- Stop after implementing only the allowed-file changes, running the validation commands, and running `python3 scripts/agent/check_agent_changes.py tasks/0047-fix-0046-progress-logging.md`.
- Review the result in Korean and report `PASS` or `NEEDS_FIX`.
- Do not commit; wait for the user to review and commit manually.
