# Task 0028: Pre-SNS Scaled Training Run

## Task title
Create a scaled pre-SNS training run controller and config generator.

## Role
Codex is the implementation worker in Codex-only mode. Do not delegate to Claude Code for this task.

Do not implement this task while creating this task file. Stop after creating this task file.

## Goal
Create a scaled pre-SNS training run controller and config generator.

## Purpose
Task 0025 proved actual GPU pilot training with artifacts. Task 0027 evaluates a checkpoint. This task prepares controlled scaled pre-SNS training so the user can increase `max_samples`, `epochs`, `batch_size`, and artifact roots safely without editing JSON by hand.

This is still pre-SNS. It must not implement SNS augmentation.

## Files Codex May Read
- AGENTS.md
- CLAUDE.md
- docs/project_brief.md
- docs/project_contract.md
- configs/project_contract.json
- docs/agent_workflow.md
- docs/pre_sns_training_artifacts.md
- docs/pre_sns_evaluation.md
- tasks/0025-pre-sns-training-artifact-writer.md
- tasks/0027-pre-sns-evaluation-runner.md
- tasks/0028-pre-sns-scaled-training-run.md
- scripts/training/train_pre_sns_baseline.py
- scripts/agent/validate_pre_sns_baseline_train_config.py
- scripts/agent/validate_pre_sns_training_artifacts.py
- scripts/agent/validate_pre_sns_dataset_manifest.py
- .local/pre_sns_dataset_manifest.local.json, if it exists
- .local/pre_sns_baseline_train.local.json, if it exists

## Files Codex May Modify
- scripts/training/prepare_pre_sns_scaled_train_config.py
- scripts/agent/validate_pre_sns_scaled_training_plan.py
- tests/test_pre_sns_scaled_training_plan.py
- docs/pre_sns_scaled_training.md
- configs/training/pre_sns_scaled_train.example.json

If a parent directory for an allowed file does not exist, Codex may create it. Codex must not create or modify any other files.

## Forbidden actions
Codex must not:

- download datasets
- access network resources
- install packages
- train during implementation
- write checkpoints during implementation
- use SNS augmentation
- run SNS perturbation evaluation
- access `.env`, `.env.*`, secrets, credentials, data, datasets, outputs, or checkpoints
- recursively scan directories
- run `git push`, `git pull`, or `git fetch`
- use `rg`
- use `rm -rf`
- create large files

## Allowed runtime behavior
- The config generator may read `.local/pre_sns_dataset_manifest.local.json` only when user runs it manually.
- The config generator may create `.local/pre_sns_baseline_scaled.local.json`.
- It may create approved run/checkpoint root names under `~/cvf_runs` and `~/cvf_checkpoints`, but it should not run training.

## Implementation requirements

### 1. Scaled config generator
Create `scripts/training/prepare_pre_sns_scaled_train_config.py`.

CLI:

```bash
python3 scripts/training/prepare_pre_sns_scaled_train_config.py \
  --base-config .local/pre_sns_baseline_train.local.json \
  --manifest .local/pre_sns_dataset_manifest.local.json \
  --out .local/pre_sns_baseline_scaled.local.json \
  --scale scaled_pilot \
  --max-samples 8192 \
  --epochs 2 \
  --batch-size 8 \
  --max-image-size 160 \
  --device cuda
```

It must:

- validate base config exists
- validate manifest exists
- not read images
- not read masks
- not recursively scan directories
- not train
- generate fresh `approved_run_root` and `approved_checkpoint_root` outside repo
- set `no_write_dry_run` false by default only when explicit approval text is included
- keep `no_download` true, `no_network` true, `no_sns_augmentation` true
- set `result_scope` to scaled pre-SNS baseline training
- print `PRE_SNS_SCALED_TRAIN_CONFIG_CREATED_OK`

### 2. Example config
Create `configs/training/pre_sns_scaled_train.example.json`.

It must be symbolic and safe:

- no real paths
- no URLs
- no secrets
- contains scale knobs and guardrail fields
- includes marker `PRE_SNS_SCALED_TRAINING_PLAN_OK`

### 3. Scaled training plan validator
Create `scripts/agent/validate_pre_sns_scaled_training_plan.py`.

CLI:

```bash
python3 scripts/agent/validate_pre_sns_scaled_training_plan.py <config.json>
```

It must:

- validate example symbolic config
- validate approved local generated config
- ensure scaled values are bounded:
  - `max_samples > 0 and <= 50000`
  - `epochs > 0 and <= 10`
  - `batch_size > 0 and <= 64`
  - `max_image_size > 0 and <= 512`
- ensure `device` is `cuda` or `cpu`
- ensure `no_download` true
- ensure `no_network` true
- ensure `no_sns_augmentation` true
- ensure run/checkpoint roots are outside repo in approved local mode
- print `PRE_SNS_SCALED_TRAINING_PLAN_OK` on success

### 4. Tests
Create `tests/test_pre_sns_scaled_training_plan.py`.

Tests must be runnable with:

```bash
python3 tests/test_pre_sns_scaled_training_plan.py
```

Tests must cover:

- example config passes
- approved local generated config passes
- excessive `max_samples` rejected
- excessive `epochs` rejected
- excessive `batch_size` rejected
- excessive `max_image_size` rejected
- repo `outputs/` or `checkpoints/` rejected
- `no_download=false` rejected
- `no_network=false` rejected
- `no_sns_augmentation=false` rejected
- generator produces fresh roots and does not train

### 5. Documentation
Create `docs/pre_sns_scaled_training.md`.

It must include marker:

```text
PRE_SNS_SCALED_TRAINING_PLAN_OK
```

It must explain:

- scaling path from pilot to larger pre-SNS baseline
- suggested increments:
  - 2560 samples, 1 epoch
  - 8192 samples, 2 epochs
  - 20000 samples, 3 epochs
  - 50000 samples max for this guarded phase
- artifacts outside repo
- no SNS augmentation
- not final full-dataset performance unless separately approved

## Validation commands

```bash
python3 scripts/agent/check_agent_changes.py tasks/0028-pre-sns-scaled-training-run.md
```

```bash
python3 scripts/agent/validate_pre_sns_scaled_training_plan.py configs/training/pre_sns_scaled_train.example.json
```

```bash
python3 tests/test_pre_sns_scaled_training_plan.py
```

```bash
grep -q PRE_SNS_SCALED_TRAINING_PLAN_OK docs/pre_sns_scaled_training.md
```

```bash
git status --short --untracked-files=all
```

## Acceptance criteria
- Config generator exists and is safe.
- Validator passes.
- Tests pass.
- Documentation marker exists.
- No actual training occurs during implementation.
- No SNS augmentation is implemented.
- Changed files are limited to allowed files.

## Stop condition
After creating `tasks/0028-pre-sns-scaled-training-run.md`, stop and show:

- task file path
- summary
- git status
- git add command
- git commit command
