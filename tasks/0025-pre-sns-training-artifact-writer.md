# Task 0025: Pre-SNS Training Artifact Writer

## Task title
Implement a guarded pre-SNS training artifact writer and minimal real pilot training loop.

## Role
Codex is the implementation worker in Codex-only mode. Codex must implement only the files allowed by this task file after this task file is committed and after the user explicitly says to implement.

Do not implement this task while creating this task file.

## Context
The current pre-SNS baseline training entrypoint reaches CUDA and validates the approved local manifest/config, but it is intentionally minimal. The latest run output only reports `PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK` and does not include finite loss values, completed training steps, metrics, run artifacts, or checkpoints.

The next step is to turn the guarded entrypoint into a real but still small pre-SNS pilot training run with artifact writing.

## Project goal
Build a lightweight multi-head image forensics prototype before SNS augmentation:

- 3-way class prediction: `real / full_synthetic / tampered`
- generator-family provenance prediction for Community Forensics-Small samples
- conditional tampered localization for SID-Set tampered samples with masks
- evidence/template explanation compatibility from prior schema tasks
- no SNS augmentation yet

## Task objective
Implement a guarded pre-SNS training artifact writer and minimal real pilot training loop so that approved local GPU training produces inspectable artifacts outside the repository.

## Files Codex may read
- AGENTS.md
- CLAUDE.md
- docs/project_brief.md
- docs/project_contract.md
- configs/project_contract.json
- docs/agent_workflow.md
- tasks/0020-pre-sns-integrated-model-smoke.md
- tasks/0021-pre-sns-dataset-manifest-gate.md
- tasks/0022-pre-sns-training-preflight-dry-run.md
- tasks/0023-pre-sns-baseline-training-entrypoint.md
- tasks/0024-fix-pre-sns-manifest-approved-paths.md
- scripts/training/train_pre_sns_baseline.py
- scripts/agent/validate_pre_sns_baseline_train_config.py
- scripts/agent/validate_pre_sns_dataset_manifest.py
- src/cv_forensics/pre_sns_integrated_model.py
- src/cv_forensics/pre_sns_manifest.py
- src/cv_forensics/pre_sns_preflight.py
- src/cv_forensics/model_output_schema.py
- src/cv_forensics/explanation_templates.py
- docs/pre_sns_dataset_manifest.md
- configs/training/pre_sns_baseline_train.example.json
- .local/pre_sns_dataset_manifest.local.json
- .local/pre_sns_baseline_train.local.json

## Files Codex may modify
- scripts/training/train_pre_sns_baseline.py
- src/cv_forensics/pre_sns_training_artifacts.py
- src/cv_forensics/__init__.py
- scripts/agent/validate_pre_sns_training_artifacts.py
- tests/test_pre_sns_training_artifacts.py
- docs/pre_sns_training_artifacts.md
- configs/training/pre_sns_baseline_train.example.json

Codex must not create or modify any other files.

## Forbidden actions
Codex must not:

- download datasets
- access network resources
- install packages
- use SNS augmentation
- run SNS perturbation evaluation
- access `.env`, `.env.*`, secrets, or credentials
- read files outside explicit manifest `image_path` and `mask_path` entries
- recursively scan dataset directories
- write to repository `outputs/` or `checkpoints/`
- write generated model artifacts inside the git repository
- run `git push`, `git pull`, or `git fetch`
- create large files
- use `rg`
- use `rm -rf`
- modify unrelated files

## Allowed runtime writes
During approved actual training only, the training script may write small artifacts to:

- `approved_run_root`
- `approved_checkpoint_root`

These roots must be outside the repository and must come from the approved training config.

## Important project facts
- This is pre-SNS baseline pilot training, not final full-dataset training or a final performance claim.
- The implementation must keep the existing safety gates from the guarded training entrypoint.
- Actual artifact writes are allowed only for approved actual training runs and only outside the repository.
- `no_write_dry_run=true` must still exercise a minimal forward/backward smoke loop and report finite losses without creating artifacts.
- SNS augmentation is not part of this phase.

## Implementation requirements

### 1. Artifact module
Create `src/cv_forensics/pre_sns_training_artifacts.py`.

It must provide pure-Python helper functions for:

- validating that artifact roots are outside the repository
- creating approved run/checkpoint directories
- allowing directories that do not exist or exist but are empty
- refusing non-empty directories unless an explicit overwrite flag is present
- writing JSON atomically where practical
- writing:
  - `config_snapshot.json`
  - `manifest_snapshot.json`
  - `metrics_summary.json`
  - `run_summary.json`
  - `artifact_manifest.json`
- recording artifact paths and file sizes
- rejecting writes to repository `outputs/` and `checkpoints/`
- rejecting writes to `.env`, secrets, data, datasets, outputs, checkpoints, or hidden credential-like paths

### 2. Minimal real pre-SNS pilot training loop
Update `scripts/training/train_pre_sns_baseline.py`.

It must keep all existing safety gates but add a real minimal training loop for `approved_training_run=true`.

The training loop must:

- read only explicit samples from the approved unified manifest
- accept either `samples` or `sample_manifest`, but prefer `samples` if present
- never recursively scan directories
- use GPU when `device=cuda` and `torch.cuda.is_available()`
- fail clearly if `device=cuda` is requested but CUDA is unavailable
- load images from explicit `image_path` only
- load masks only for tampered samples with explicit `mask_path`
- not use SNS augmentation
- not download anything
- compute class loss for all samples
- compute family loss only for samples with valid `family_label`
- compute localization/mask loss only for tampered samples with masks
- run at least one epoch when `epochs >= 1`
- respect `max_samples`, `batch_size`, `max_image_size`, and `seed`
- produce finite loss values
- report `samples_seen`, `masks_seen`, `steps_completed`, `epochs_completed`
- report `class_smoke_accuracy`
- report `family_smoke_accuracy` if family samples exist
- report `localization_smoke_iou` if tampered masks exist
- keep the model small enough for pilot training
- avoid large checkpoint files

The result JSON printed to stdout after a successful actual run must include:

- `marker`: `PRE_SNS_BASELINE_TRAINING_RUN_OK`
- `entrypoint_marker`: `PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK`
- `training_started`: `true`
- `training_completed`: `true`
- `device`
- `cuda_device_name` when CUDA is used
- `epochs_completed`
- `steps_completed`
- `samples_seen`
- `masks_seen`
- `initial_total_loss`
- `final_total_loss`
- `total_loss_finite`
- `class_loss_finite`
- `family_loss_finite`
- `localization_loss_finite`
- `class_smoke_accuracy`
- `family_smoke_accuracy` if available
- `localization_smoke_iou` if available
- `no_download`: `true`
- `no_network`: `true`
- `no_sns_augmentation`: `true`
- `no_write_dry_run`
- `approved_run_root`
- `approved_checkpoint_root`
- `artifact_manifest_path` when actual writes are enabled
- `checkpoint_path` when checkpoint writing is enabled
- `result_scope`

### 3. `no_write_dry_run` behavior
When `no_write_dry_run=true`:

- perform config/manifest validation and a minimal forward/backward smoke loop
- print finite loss and metric fields
- do not create run/checkpoint artifact files
- do not write checkpoint
- result marker may be `PRE_SNS_BASELINE_TRAINING_DRY_RUN_OK`
- include `no_write_dry_run: true`
- include `total_loss_finite: true`

### 4. Actual training artifact behavior
When `no_write_dry_run=false`:

- create `approved_run_root` and `approved_checkpoint_root` safely
- allow roots that already exist only if they are empty
- write `config_snapshot.json`
- write `manifest_snapshot.json`
- write `metrics_summary.json`
- write `run_summary.json`
- write `artifact_manifest.json`
- write a small model checkpoint under `approved_checkpoint_root`
- stdout result JSON must reference the artifact paths
- artifacts must be outside the repository

### 5. Artifact validator
Create `scripts/agent/validate_pre_sns_training_artifacts.py`.

CLI:

```bash
python3 scripts/agent/validate_pre_sns_training_artifacts.py <training-config.json> <training-result-json>
```

It must:

- read the training config
- read the training result JSON, including files that may contain extra log text before JSON
- verify marker `PRE_SNS_BASELINE_TRAINING_RUN_OK` for actual runs
- verify `total_loss_finite` is `true`
- verify `training_completed` is `true`
- verify `no_download`, `no_network`, and `no_sns_augmentation` are `true`
- verify `approved_run_root` and `approved_checkpoint_root` are outside the repository
- verify required artifacts exist
- verify `artifact_manifest.json` lists the expected files
- verify `checkpoint_path` exists for actual write runs
- verify artifact files are small
- verify no artifact path is inside repository `outputs/` or `checkpoints/`
- print `PRE_SNS_TRAINING_ARTIFACTS_OK` on success
- exit non-zero with actionable errors on failure

### 6. Tests
Create `tests/test_pre_sns_training_artifacts.py`.

Tests must be runnable with:

```bash
python3 tests/test_pre_sns_training_artifacts.py
```

Tests should use temporary directories and tiny fake configs/manifests only. Do not use real datasets in tests.

Tests must cover:

- artifact roots outside repo are accepted
- repo `outputs/` and `checkpoints/` roots are rejected
- non-empty run root is rejected unless explicitly allowed
- empty pre-existing run root is accepted
- artifact manifest contains required files
- validator rejects missing `run_summary.json`
- validator rejects missing checkpoint for actual run
- validator accepts a valid small artifact set
- parsing a result JSON with leading log text works
- `no_write_dry_run` result does not require artifacts
- actual training result requires artifacts

### 7. Documentation
Create `docs/pre_sns_training_artifacts.md`.

It must include:

- marker `PRE_SNS_TRAINING_ARTIFACTS_OK`
- explanation of pre-SNS artifact policy
- list of generated artifacts
- distinction between `no_write_dry_run` and actual training
- note that artifacts go to `~/cvf_runs` and `~/cvf_checkpoints`, not repository `outputs/` or `checkpoints/`
- note that SNS augmentation is not part of this phase
- note that this is pilot pre-SNS baseline training, not a final full-dataset performance claim

## Validation commands

```bash
python3 scripts/agent/check_agent_changes.py tasks/0025-pre-sns-training-artifact-writer.md
```

```bash
python3 scripts/agent/validate_pre_sns_baseline_train_config.py configs/training/pre_sns_baseline_train.example.json
```

```bash
python3 tests/test_pre_sns_training_artifacts.py
```

```bash
grep -q PRE_SNS_TRAINING_ARTIFACTS_OK docs/pre_sns_training_artifacts.md
```

```bash
git status --short --untracked-files=all
```

## Manual local validation commands
Document these commands for local approved validation. Do not run actual training unless the user explicitly requests it after implementation.

```bash
python3 scripts/agent/validate_pre_sns_dataset_manifest.py .local/pre_sns_dataset_manifest.local.json
```

```bash
python3 scripts/agent/validate_pre_sns_baseline_train_config.py .local/pre_sns_baseline_train.local.json
```

```bash
python3 scripts/training/train_pre_sns_baseline.py .local/pre_sns_baseline_train.local.json
```

```bash
python3 scripts/agent/validate_pre_sns_training_artifacts.py .local/pre_sns_baseline_train.local.json <training-result-json>
```

## Acceptance criteria
- Task file exists.
- The implementation produces finite loss fields in `no_write_dry_run` mode.
- The implementation produces `PRE_SNS_BASELINE_TRAINING_RUN_OK` in actual training mode.
- Actual training writes `run_summary.json`, `metrics_summary.json`, `config_snapshot.json`, `manifest_snapshot.json`, `artifact_manifest.json`, and a small checkpoint outside the repository.
- Artifact validator prints `PRE_SNS_TRAINING_ARTIFACTS_OK` for a valid actual training result.
- No repository `outputs/` or `checkpoints/` directories are touched.
- No datasets are downloaded.
- No network access is used.
- No SNS augmentation is used.
- Changed source files are limited to the allowed files.
- Tests pass.

## Stop condition
After creating `tasks/0025-pre-sns-training-artifact-writer.md`, stop and show:

- task file path
- short summary
- git status
- exact `git add` / `git commit` command for the task file

Do not implement yet. Do not run training yet.
