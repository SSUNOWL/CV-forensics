# Task 0012: CF-Small Local Subset Smoke Plan

## Task Title

Create a Community Forensics-Small local subset smoke plan.

## Role

Codex is the task writer, implementation worker, reviewer, and limited repair manager for this task because Claude Code is currently unavailable due to usage limits.

Implement only this task. Do not reinterpret the project scope beyond this task file.

## Purpose

Prepare a tiny local subset smoke workflow for Community Forensics-Small, gated by user-approved local paths and validated manifests. This is a planning and validation step only.

This task must not download datasets, run full training, inspect actual dataset directories recursively, read real images, write outputs, or write checkpoints. It should define a symbolic, safe, local-path-gated plan that can later be converted into a real local-data smoke test only after explicit user approval.

## Files Codex May Read

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
- `docs/local_data_readiness.md`, only if it exists
- `configs/datasets.example.json`
- `configs/experiments/smoke_baseline.json`
- `configs/training/dry_run_training.example.json`
- `configs/local_data/readiness.example.json`, only if it exists
- `configs/manifests/community_forensics_small.example.json`
- `configs/manifests/sid_set.example.json`
- `configs/manifests/combined_smoke_manifest.example.json`
- `src/cv_forensics/config_schema.py`
- `src/cv_forensics/dataset_manifest.py`
- `src/cv_forensics/model_output_schema.py`
- `src/cv_forensics/inference_stub.py`
- `src/cv_forensics/metrics.py`
- `src/cv_forensics/training_dry_run.py`, only if it exists
- `src/cv_forensics/local_data_gate.py`, only if it exists
- `scripts/agent/validate_config_schema.py`
- `scripts/agent/validate_dataset_manifest.py`
- `scripts/agent/validate_training_dry_run.py`, only if it exists
- `scripts/agent/validate_local_data_readiness.py`, only if it exists
- `tasks/0012-cf-small-local-subset-smoke-plan.md`
- `docs/repo_structure.md`, only if it exists
- `docs/planning/pre_sns_implementation_plan.md`, only if it exists
- any file Claude is allowed to create or modify in this task, if it already exists

If proposal PDFs or slides are absent, do not fail. Rely on `docs/project_brief.md`, the project contract, planning docs, and existing schema docs.

## Files Codex May Modify

- `configs/local_data/cf_small_subset_smoke.example.json`
- `scripts/agent/validate_cf_small_subset_smoke.py`
- `tests/test_cf_small_subset_smoke.py`
- `docs/cf_small_subset_smoke.md`

## Files Claude May Modify

This Codex-only task preserves this section name so `scripts/agent/check_agent_changes.py` can enforce the allowed modification list.

- `configs/local_data/cf_small_subset_smoke.example.json`
- `scripts/agent/validate_cf_small_subset_smoke.py`
- `tests/test_cf_small_subset_smoke.py`
- `docs/cf_small_subset_smoke.md`

If a parent directory for an allowed file does not exist, Codex may create that parent directory. Codex must not create any other files.

## Forbidden Actions

Codex must not:

- download datasets
- train models
- run full training
- run real optimization over real data
- inspect actual dataset directories recursively
- read real images
- read real masks
- access `.env`, `.env.*`, `secrets`, `data`, `datasets`, `outputs`, or `checkpoints`
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
- Community Forensics-Small metadata of interest includes `architecture`, `model_name`, and `subset`.
- Generator-holdout or model-name-holdout planning is preferred over random split because unseen generator generalization is central to the project.
- SID-Set is not the target of this task.
- This task is a local subset smoke plan only. It must not implement actual CF-Small training.
- This task must not implement SNS augmentation.
- SNS augmentation begins only after pre-SNS baseline evaluation is committed.

## Implementation Requirements

### 1. Create `configs/local_data/cf_small_subset_smoke.example.json`

- Use a symbolic, dry-run-safe configuration.
- Do not include real local paths.
- Do not include protected paths.
- Do not include URLs.
- Do not include secrets.
- Include:
  - `schema_version`
  - `dry_run: true`
  - `no_download: true`
  - `no_training: true`
  - `no_network: true`
  - `no_outputs: true`
  - `no_checkpoints: true`
  - `dataset: Community Forensics-Small`
  - `dataset_alias: CF-Small`
  - `local_path_gate`
  - `manifest_ref`
  - `subset_plan`
  - `sample_limits`
  - `required_metadata`
  - `holdout_plan`
  - `validation_plan`
  - `approval`
  - `forbidden_actions`
- `local_path_gate` must make clear that real local paths require explicit user approval later.
- `manifest_ref` must be symbolic or reference an existing safe config file under `configs/`.
- `subset_plan` must describe a tiny smoke subset plan without enumerating actual dataset files.
- `sample_limits` should keep the planned subset small.
- `required_metadata` must include `architecture`, `model_name`, and `subset`.
- `holdout_plan` must represent generator/model-name holdout planning and must explicitly reject random-only validation as sufficient.
- `validation_plan` must be planning-only and must not read images or launch training.

### 2. Create `scripts/agent/validate_cf_small_subset_smoke.py`

- Use only the Python standard library.
- Accept exactly one positional argument: CF-Small subset smoke config JSON path.
- Add repo `src` path to `sys.path` if needed.
- Load and validate the config.
- Recursively reject unsafe keys and values, including:
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
- Do not reject ordinary prose such as `authoritative guidance` or `authentication policy` merely because it contains `auth` as a substring.
- Validate that all guardrail booleans are true.
- Validate that no download, training, network, output, or checkpoint action is requested.
- Validate that the smoke plan is local-path gated.
- Validate that generator/model-name holdout planning is represented.
- Validate that required CF-Small metadata fields are represented.
- Validate that no real image reading is requested.
- Print a concise success message on pass.
- Exit non-zero with actionable errors on fail.
- Do not write files.
- Do not create outputs or checkpoints.
- Do not access real datasets.

### 3. Create `tests/test_cf_small_subset_smoke.py`

- Use only the Python standard library.
- Do not import `pytest`.
- It must be runnable with:

```bash
python3 tests/test_cf_small_subset_smoke.py
```

- It may also be pytest-compatible if pytest is installed.
- Tests should cover:
  - valid symbolic smoke config passes
  - `dry_run: false` is rejected
  - `no_download: false` is rejected
  - `no_training: false` is rejected
  - `no_outputs: false` is rejected
  - `no_checkpoints: false` is rejected
  - protected paths are rejected
  - URLs are rejected
  - secret-like keys or values are rejected
  - standalone `auth` key/value is rejected while ordinary prose such as `authoritative guidance` or `authentication policy` is allowed
  - missing local path gate is rejected
  - missing holdout plan is rejected
  - random-only validation planning is rejected
  - missing required metadata fields are rejected
  - image reading or training requests are rejected
  - validator does not write files

### 4. Create `docs/cf_small_subset_smoke.md`

- Explain that this is a planning and validation step, not full training.
- Explain that no dataset download occurs.
- Explain that no real images or masks are read.
- Explain that no dataset directories are inspected recursively.
- Explain that no outputs or checkpoints are written.
- Explain the local-path approval gate.
- Explain the tiny subset smoke plan for CF-Small.
- Explain why generator-holdout or model-name-holdout planning is required.
- Explain that random-only validation is insufficient for the project goal.
- Explain what must happen before any real local subset smoke run:
  - explicit user approval
  - validated local data readiness gate
  - validated CF-Small manifest
  - approved local paths
  - explicit output/checkpoint policy if later needed
- Explain that SNS augmentation is not implemented here.
- Include marker string:
  - `CF_SMALL_SUBSET_SMOKE_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_cf_small_subset_smoke.py configs/local_data/cf_small_subset_smoke.example.json
```

```bash
python3 tests/test_cf_small_subset_smoke.py
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0012-cf-small-local-subset-smoke-plan.md
```

```bash
grep -q CF_SMALL_SUBSET_SMOKE_OK docs/cf_small_subset_smoke.md
```

```bash
git status --short --untracked-files=all
```

```bash
test -f configs/local_data/cf_small_subset_smoke.example.json
```

```bash
test -f scripts/agent/validate_cf_small_subset_smoke.py
```

```bash
test -f tests/test_cf_small_subset_smoke.py
```

```bash
test -f docs/cf_small_subset_smoke.md
```

```bash
git diff -- configs/local_data/cf_small_subset_smoke.example.json scripts/agent/validate_cf_small_subset_smoke.py tests/test_cf_small_subset_smoke.py docs/cf_small_subset_smoke.md
```

## Acceptance Criteria

- `python3 scripts/agent/validate_cf_small_subset_smoke.py configs/local_data/cf_small_subset_smoke.example.json` passes.
- `python3 tests/test_cf_small_subset_smoke.py` passes.
- `python3 scripts/agent/check_agent_changes.py tasks/0012-cf-small-local-subset-smoke-plan.md` passes.
- `grep -q CF_SMALL_SUBSET_SMOKE_OK docs/cf_small_subset_smoke.md` passes.
- Smoke plan is local-path gated.
- Readiness config is symbolic and dry-run safe.
- No dataset download occurs.
- No training occurs.
- No real image or mask reading occurs.
- No actual dataset directories are inspected recursively.
- No outputs or checkpoints are written.
- Generator/model-name holdout planning is represented.
- Random-only validation is not treated as sufficient.
- Required CF-Small metadata fields `architecture`, `model_name`, and `subset` are represented.
- Protected paths, URLs, secrets, output paths, and checkpoint paths are rejected.
- Ordinary prose such as `authoritative guidance` or `authentication policy` is allowed when not used as a standalone secret-like key or value.
- Changed files are limited to:
  - `configs/local_data/cf_small_subset_smoke.example.json`
  - `scripts/agent/validate_cf_small_subset_smoke.py`
  - `tests/test_cf_small_subset_smoke.py`
  - `docs/cf_small_subset_smoke.md`
- No SNS augmentation is implemented.

## Stop Condition

Stop after implementation and validation. Report:

- files changed
- validation commands run and results
- any skipped optional validation and why
- confirmation that forbidden paths and actions were not touched
- confirmation that no dataset download, model training, recursive dataset inspection, image reading, output writing, checkpoint writing, package installation, network access, or SNS augmentation occurred
