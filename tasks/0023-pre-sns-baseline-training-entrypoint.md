# Task 0023: Pre-SNS Baseline Training Entrypoint

## Task title
Create a guarded pre-SNS baseline training entrypoint.

## Role
Codex is the implementation worker in Codex-only mode. Codex must implement only the files allowed by this task file after this task file is committed and after the user explicitly says to implement.

## Files Codex may read
- AGENTS.md
- CLAUDE.md
- docs/project_brief.md
- docs/project_contract.md
- configs/project_contract.json
- docs/agent_workflow.md
- docs/pre_sns_integrated_smoke.md
- docs/pre_sns_dataset_manifest.md
- docs/pre_sns_training_preflight.md
- src/cv_forensics/pre_sns_integrated_model.py
- src/cv_forensics/pre_sns_manifest.py
- src/cv_forensics/pre_sns_preflight.py
- src/cv_forensics/local_data_gate.py
- tasks/0023-pre-sns-baseline-training-entrypoint.md

## Files Codex may modify
- configs/training/pre_sns_baseline_train.example.json
- scripts/agent/validate_pre_sns_baseline_train_config.py
- scripts/training/train_pre_sns_baseline.py
- tests/test_pre_sns_baseline_train_gate.py
- docs/pre_sns_baseline_training.md

If a parent directory for an allowed file does not exist, Codex may create that parent directory. Codex must not create or modify any other files.

## Forbidden actions
During implementation, Codex must not:
- download datasets
- run real training on real data
- inspect dataset directories recursively
- access .env, .env.*, secrets, data, datasets, outputs, or checkpoints
- install packages
- access network resources from shell commands
- run git push, git pull, or git fetch
- use rg
- use rm -rf
- create large files
- implement SNS augmentation
- run SNS evaluation
- use danger-full-access
- bypass permissions

## Important project facts
- This is the final task before actual pre-SNS baseline training.
- This task creates a training script and validation gate, but implementation and tests must not run real training on real data.
- The script must require explicit approval before real training.
- Outputs and checkpoints may be written only to approved local run roots outside protected repository paths.
- Actual training should start only after tasks 0020, 0021, 0022, and 0023 are PASS and committed.
- SNS augmentation is not part of this task.

## Implementation requirements

1. Create `configs/training/pre_sns_baseline_train.example.json`
   - The tracked config must be a safe symbolic example only.
   - It must not contain real paths, URLs, secrets, protected paths, outputs, or checkpoints.
   - It must include explicit approval fields, but set `approved_training_run` to `false`.
   - Required real training approval text must be exactly:
     - `I_APPROVE_PRE_SNS_BASELINE_TRAINING`
   - It must include expected local fields:
     - `unified_manifest_path`
     - `approved_run_root`
     - `approved_checkpoint_root`
     - `device`
     - `batch_size`
     - `epochs`
     - `max_samples`
     - `seed`
     - `class_loss_weight`
     - `family_loss_weight`
     - `localization_loss_weight`
   - It must document that `approved_run_root` and `approved_checkpoint_root` must be outside the repository and must not be named `outputs` or `checkpoints` inside the repository.

2. Create `scripts/agent/validate_pre_sns_baseline_train_config.py`
   - Use only the Python standard library.
   - Accept exactly one positional config path.
   - Validate config kinds:
     - `example_symbolic`
     - `approved_pre_sns_baseline_training`
   - For approved training mode:
     - require `approved_training_run` to be `true`
     - require approval text exactly `I_APPROVE_PRE_SNS_BASELINE_TRAINING`
   - Validate `unified_manifest_path` is explicit and under approved local roots.
   - Validate `approved_run_root` and `approved_checkpoint_root` are explicit, local, and outside protected repository paths.
   - Reject URLs, remote schemes, secret-like values, recursive scan flags, path traversal, repo `outputs/`, repo `checkpoints/`, `data/`, `datasets/`, `.env`, and secrets.
   - Print `PRE_SNS_BASELINE_TRAIN_CONFIG_OK` on success.
   - Do not read images or masks.
   - Do not write files.

3. Create `scripts/training/train_pre_sns_baseline.py`
   - This is the actual future training entrypoint, but tests must not run real training on real data.
   - Validate config before loading images or masks.
   - Exit safely if `approved_training_run` is `false`.
   - Require explicit approval text before any training that writes run artifacts.
   - Use CPU by default unless config explicitly requests CUDA and CUDA is available.
   - Use torch and PIL if available.
   - Read only explicitly listed image and mask paths from the approved unified manifest.
   - Do not recursively scan directories.
   - Do not download anything.
   - Do not access `.env` or secrets.
   - When approved, create only the approved run root and checkpoint root outside protected repository paths.
   - Support a `no_write_dry_run` mode that does not write artifacts.
   - Print a JSON summary with marker `PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK` for dry-run checks.
   - Print `PRE_SNS_BASELINE_TRAINING_STARTED` only when actual approved training begins.

4. Create `tests/test_pre_sns_baseline_train_gate.py`
   - Use a Python standard library test harness.
   - Do not import pytest.
   - Tests must use temporary configs and temporary fixture manifests only.
   - Tests must not use real datasets.
   - Tests must cover:
     - example config passes
     - no approval exits before training and before file writes
     - bad approval is rejected
     - protected run root is rejected
     - protected checkpoint root is rejected
     - URLs are rejected
     - recursive scan flags are rejected
     - `no_write_dry_run` prints `PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK` without creating outputs or checkpoints
   - If torch or PIL is missing, skip model execution with a clear message but still validate gates.

5. Create `docs/pre_sns_baseline_training.md`
   - Explain this is the guarded final entrypoint before real pre-SNS baseline training.
   - Explain the exact approval text.
   - Explain required local artifacts:
     - unified manifest
     - approved run root
     - approved checkpoint root
   - Explain that SNS augmentation is not part of this task.
   - Explain that actual training should start only after 0020, 0021, 0022, and 0023 are PASS and committed.
   - Include marker:
     - `PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK`

## Validation commands

```bash
python3 scripts/agent/validate_pre_sns_baseline_train_config.py configs/training/pre_sns_baseline_train.example.json
```

```bash
python3 tests/test_pre_sns_baseline_train_gate.py
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0023-pre-sns-baseline-training-entrypoint.md
```

```bash
test -f configs/training/pre_sns_baseline_train.example.json
```

```bash
test -f scripts/agent/validate_pre_sns_baseline_train_config.py
```

```bash
test -f scripts/training/train_pre_sns_baseline.py
```

```bash
test -f tests/test_pre_sns_baseline_train_gate.py
```

```bash
test -f docs/pre_sns_baseline_training.md
```

```bash
grep -q PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK docs/pre_sns_baseline_training.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- configs/training/pre_sns_baseline_train.example.json scripts/agent/validate_pre_sns_baseline_train_config.py scripts/training/train_pre_sns_baseline.py tests/test_pre_sns_baseline_train_gate.py docs/pre_sns_baseline_training.md
```

Optional validation command:

```bash
pytest -q tests/test_pre_sns_baseline_train_gate.py
```

## Acceptance criteria
- `python3 scripts/agent/validate_pre_sns_baseline_train_config.py configs/training/pre_sns_baseline_train.example.json` passes.
- `python3 tests/test_pre_sns_baseline_train_gate.py` passes.
- `python3 scripts/agent/check_agent_changes.py tasks/0023-pre-sns-baseline-training-entrypoint.md` passes.
- If pytest exists, `pytest -q tests/test_pre_sns_baseline_train_gate.py` passes.
- `docs/pre_sns_baseline_training.md` contains `PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK`.
- Changed files are limited to:
  - configs/training/pre_sns_baseline_train.example.json
  - scripts/agent/validate_pre_sns_baseline_train_config.py
  - scripts/training/train_pre_sns_baseline.py
  - tests/test_pre_sns_baseline_train_gate.py
  - docs/pre_sns_baseline_training.md
- The training script refuses to train without explicit approval.
- The training script rejects protected paths and unsafe roots.
- The training script supports `no_write_dry_run`.
- No dataset download, real training, network access, protected path access, output writing, checkpoint writing, or SNS augmentation occurs during implementation.
- After this task is PASS and committed, the user can create an approved local training config and start pre-SNS baseline training manually.

## Stop condition
Stop after implementing the allowed files, running safe validation commands, running `python3 scripts/agent/check_agent_changes.py tasks/0023-pre-sns-baseline-training-entrypoint.md`, and reviewing the result in Korean as PASS or NEEDS_FIX.
