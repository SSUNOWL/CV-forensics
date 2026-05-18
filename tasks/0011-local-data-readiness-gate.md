# Task 0011: Local Data Readiness Gate

## Task Title

Implement a local data readiness gate and checklist with no dataset download.

## Role of Claude Code

Claude Code is the implementation worker. Codex is the supervisor, task writer, delegation controller, reviewer, and limited repair manager.

Implement only this task. Do not reinterpret the project scope beyond this task file.

## Purpose

Before real dataset training, add a dry-run-safe local data readiness gate that validates whether local paths, dataset manifest references, output/checkpoint policies, and protected-path exclusions are explicitly approved and safe.

This task prepares the repository for a later user-approved local-data workflow. It must not download datasets, inspect actual dataset directories recursively, train models, write outputs, or write checkpoints.

## Files Claude May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `docs/config_schema.md`
- `docs/dataset_manifest.md`
- `docs/model_output_schema.md`
- `docs/inference_stub.md`
- `docs/metrics.md`
- `docs/training_dry_run.md`
- `configs/datasets.example.json`
- `configs/experiments/smoke_baseline.json`
- `configs/training/dry_run_training.example.json`
- `configs/manifests/community_forensics_small.example.json`
- `configs/manifests/sid_set.example.json`
- `configs/manifests/combined_smoke_manifest.example.json`
- `src/cv_forensics/__init__.py`
- `src/cv_forensics/config_schema.py`
- `src/cv_forensics/dataset_manifest.py`
- `src/cv_forensics/model_output_schema.py`
- `src/cv_forensics/inference_stub.py`
- `src/cv_forensics/metrics.py`
- `src/cv_forensics/training_dry_run.py`
- `scripts/agent/validate_config_schema.py`
- `scripts/agent/validate_dataset_manifest.py`
- `scripts/agent/validate_training_dry_run.py`
- `tasks/0011-local-data-readiness-gate.md`
- `docs/repo_structure.md`, only if it exists
- `docs/planning/pre_sns_implementation_plan.md`, only if it exists
- `tests/test_training_dry_run.py`, only if it exists
- any file Claude is allowed to create or modify in this task, if it already exists

If proposal PDFs or slides are present, Claude may ignore them for this task. If they are absent, do not fail. Rely on `docs/project_brief.md`, the project contract, planning docs, and existing schema docs.

## Files Claude May Modify

- `src/cv_forensics/__init__.py`
- `src/cv_forensics/local_data_gate.py`
- `configs/local_data/readiness.example.json`
- `scripts/agent/validate_local_data_readiness.py`
- `tests/test_local_data_gate.py`
- `docs/local_data_readiness.md`

If a parent directory for an allowed file does not exist, Claude may create that parent directory. Claude must not create any other files.

## Forbidden Actions

Claude must not:

- download datasets
- train models
- run real optimization over real data
- inspect actual dataset directories recursively
- read images
- read masks
- read or inspect `.env`, `.env.*`, `secrets`, `data`, `datasets`, `outputs`, or `checkpoints`
- access `.env`, secrets, data, datasets, outputs, or checkpoints without explicit user-approved symbolic config
- write checkpoints
- write prediction outputs
- create `outputs/`
- create `checkpoints/`
- install packages
- access network resources
- run `git push`, `git pull`, or `git fetch`
- run `curl`, `wget`, `ssh`, `scp`, `rsync`, `pip`, `npm`, `yarn`, `pnpm`, `kaggle`, `huggingface-cli`, or `wandb`
- use `rg`
- use `rm -rf`
- import `torch`, `torchvision`, `PIL`, `cv2`, `numpy`, `pandas`, or `sklearn`
- create large files
- implement SNS augmentation
- use `--permission-mode bypassPermissions`
- use `--dangerously-skip-permissions`

## Important Project Facts

- Final model direction is a lightweight multi-head image forensics research prototype.
- Target outputs are class, mask/localization, family/provenance, and reason.
- Class labels are `real`, `synthetic`, and `tampered`.
- Family labels are coarse family-level labels: `LatDiff`, `PixDiff`, `GAN`, `Other`, `Real-or-N/A`.
- Community Forensics-Small is for shared backbone learning and coarse provenance planning.
- SID-Set is for 3-way classification and tampered mask localization planning.
- Task 0010 established dry-run-only training-loop preparation.
- This task is a readiness gate only. It must not implement actual Community Forensics-Small or SID-Set training.
- This task must not implement SNS augmentation.
- SNS augmentation begins only after pre-SNS baseline evaluation is committed.

## Implementation Requirements

### 1. Create `src/cv_forensics/local_data_gate.py`

- Use only the Python standard library.
- Reuse existing validation style where appropriate, especially safety checks from config, manifest, and dry-run modules.
- Implement lightweight structures such as:
  - `LocalDataReadinessConfig`
  - `LocalDataReadinessResult`
  - `ReadinessIssue`
- Implement deterministic functions such as:
  - `load_local_data_readiness_config`
  - `validate_local_data_readiness_config`
  - `evaluate_local_data_readiness`
  - `summarize_local_data_readiness`
- Validate symbolic readiness configuration only.
- Do not inspect dataset directories recursively.
- Do not read files from protected directories.
- Do not write outputs or checkpoints.
- Do not train or launch training code.
- Preserve fields needed for later real local-data approval:
  - `schema_version`
  - `dry_run`
  - `no_download`
  - `no_training`
  - `no_network`
  - `no_outputs`
  - `no_checkpoints`
  - `dataset_plan`
  - `manifest_refs`
  - `local_path_policy`
  - `output_policy`
  - `checkpoint_policy`
  - `approval`
  - `protected_path_exclusions`

### 2. Safety validation

The readiness validator must recursively reject unsafe keys and values unless the value is an explicitly approved symbolic placeholder that does not resolve to a real protected path.

It must reject:

- `.env`
- `.env.*`
- `secrets`
- `data`
- `datasets`
- `outputs`
- `checkpoints`
- `/home/`
- `/mnt/`
- `/root/`
- `/Users/`
- Windows drive paths
- `http://`
- `https://`
- `s3://`
- `gs://`
- `hf://`
- `token`
- `api_key`
- `password`
- `secret`
- `credential`
- `bearer`
- `auth` as a standalone secret-like token or key

It must not reject ordinary prose such as `authoritative guidance` or `authentication policy` merely because those words contain `auth` as a substring.

Real local paths must require explicit approval fields. For this example config, keep all path references symbolic and dry-run safe.

### 3. Create `configs/local_data/readiness.example.json`

- It must be symbolic and dry-run safe.
- It must not contain real local paths.
- It must not contain protected paths.
- It must not contain URLs.
- It must not contain secrets.
- It must include:
  - `schema_version`
  - `dry_run: true`
  - `no_download: true`
  - `no_training: true`
  - `no_network: true`
  - `no_outputs: true`
  - `no_checkpoints: true`
  - `dataset_plan`
  - `manifest_refs` as symbolic names only
  - `local_path_policy`
  - `output_policy`
  - `checkpoint_policy`
  - `approval`
  - `protected_path_exclusions`
- It may reference existing safe config files in `configs/` as config refs only.
- It must not reference `data/`, `datasets/`, `outputs/`, `checkpoints/`, `.env`, or secret-like values.

### 4. Create `scripts/agent/validate_local_data_readiness.py`

- Use only the Python standard library.
- Accept exactly one positional argument: readiness config JSON path.
- Add repo `src` path to `sys.path` if needed.
- Load and validate the readiness config.
- Run the readiness evaluation.
- Validate that the config is dry-run safe.
- Validate that no download, training, network, output, or checkpoint action is requested.
- Validate that protected path exclusions are present.
- Print a concise success message on pass.
- Exit non-zero with actionable errors on fail.
- Do not write files.
- Do not create outputs or checkpoints.
- Do not access real datasets.

### 5. Create `tests/test_local_data_gate.py`

- Use only the Python standard library.
- Do not import `pytest`.
- It must be runnable with:

```bash
python3 tests/test_local_data_gate.py
```

- It may also be pytest-compatible if pytest is installed.
- Tests should cover:
  - valid symbolic readiness config passes
  - `dry_run: false` is rejected
  - `no_download: false` is rejected
  - `no_training: false` is rejected
  - `no_outputs: false` is rejected
  - `no_checkpoints: false` is rejected
  - protected paths are rejected
  - URLs are rejected
  - secret-like keys or values are rejected
  - standalone `auth` key/value is rejected while ordinary prose such as `authoritative guidance` or `authentication policy` is allowed
  - real local path examples require explicit approval fields
  - readiness evaluation is deterministic
  - no files are written by the validator or evaluator

### 6. Create `docs/local_data_readiness.md`

- Explain that this gate does not download datasets.
- Explain that this gate does not train models.
- Explain that this gate does not inspect dataset directories recursively.
- Explain how the readiness gate checks:
  - symbolic local-data plan
  - manifest references
  - approval policy
  - output/checkpoint policy
  - protected-path exclusions
- Explain what must happen before real training:
  - explicit user approval
  - local data readiness approval
  - validated manifests
  - output/checkpoint policy decision
  - small local subset smoke test
- Explain that SNS augmentation is not implemented here.
- Include marker string:
  - `LOCAL_DATA_READINESS_OK`

### 7. Update `src/cv_forensics/__init__.py` only if needed

- Add a lightweight export for `local_data_gate` only if useful.
- Do not add heavy imports.
- Do not import third-party packages.

## Validation Commands

```bash
python3 scripts/agent/validate_local_data_readiness.py configs/local_data/readiness.example.json
```

```bash
python3 tests/test_local_data_gate.py
```

```bash
grep -q LOCAL_DATA_READINESS_OK docs/local_data_readiness.md
```

```bash
git status --short --untracked-files=all
```

```bash
test -f src/cv_forensics/local_data_gate.py
```

```bash
test -f configs/local_data/readiness.example.json
```

```bash
test -f scripts/agent/validate_local_data_readiness.py
```

```bash
test -f tests/test_local_data_gate.py
```

```bash
test -f docs/local_data_readiness.md
```

```bash
git diff -- src/cv_forensics/__init__.py src/cv_forensics/local_data_gate.py configs/local_data/readiness.example.json scripts/agent/validate_local_data_readiness.py tests/test_local_data_gate.py docs/local_data_readiness.md
```

## Acceptance Criteria

- `python3 scripts/agent/validate_local_data_readiness.py configs/local_data/readiness.example.json` passes.
- `python3 tests/test_local_data_gate.py` passes.
- `grep -q LOCAL_DATA_READINESS_OK docs/local_data_readiness.md` passes.
- Readiness config is symbolic and dry-run safe.
- Real local paths require explicit approval fields.
- No dataset download or training occurs.
- No dataset directories are inspected recursively.
- Protected paths, URLs, secrets, output paths, and checkpoint paths are rejected.
- Ordinary prose such as `authoritative guidance` or `authentication policy` is allowed when not used as a standalone secret-like key or value.
- Changed files are limited to:
  - `src/cv_forensics/__init__.py`
  - `src/cv_forensics/local_data_gate.py`
  - `configs/local_data/readiness.example.json`
  - `scripts/agent/validate_local_data_readiness.py`
  - `tests/test_local_data_gate.py`
  - `docs/local_data_readiness.md`
- No real data, images, masks, outputs, checkpoints, secrets, package installs, downloads, network access, or real training are used.
- No SNS augmentation is implemented.

## Stop Condition

Stop after implementation and validation. Report:

- files changed
- validation commands run and results
- any skipped optional validation and why
- confirmation that forbidden paths and actions were not touched
- confirmation that no dataset download, model training, recursive dataset inspection, output writing, checkpoint writing, package installation, network access, or SNS augmentation occurred
