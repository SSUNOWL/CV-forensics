# Task 0010: Pure-Python Training Loop Skeleton with Dry-Run Only Behavior

## Role of Claude Code

Claude Code is the implementation worker. Codex is the supervisor, task writer, delegation controller, reviewer, and limited repair manager.

Implement this task exactly as written. Do not reinterpret the project scope beyond this task file. Stop after implementation and validation, then report the requested results.

## Task Objective

Implement a pure-Python training-loop skeleton that connects the existing config schema, dataset manifest schema, model output schema, fake inference stub, and toy metrics into a dry-run training/evaluation flow.

This task prepares the project for later real local-data training, but it must remain dry-run only.

This task must not:

- perform real training
- access real datasets
- read images or masks
- write checkpoints
- write outputs
- install packages
- use network access
- implement SNS augmentation

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
- `configs/datasets.example.json`
- `configs/experiments/smoke_baseline.json`
- `configs/inference/fake_inputs.example.json`
- `configs/metrics/toy_metrics.example.json`
- `configs/manifests/community_forensics_small.example.json`
- `configs/manifests/sid_set.example.json`
- `configs/manifests/combined_smoke_manifest.example.json`
- `src/cv_forensics/config_schema.py`
- `src/cv_forensics/dataset_manifest.py`
- `src/cv_forensics/model_output_schema.py`
- `src/cv_forensics/explanation_templates.py`
- `src/cv_forensics/inference_stub.py`
- `src/cv_forensics/metrics.py`
- `scripts/agent/validate_config_schema.py`
- `scripts/agent/validate_dataset_manifest.py`
- `scripts/agent/validate_model_output_schema.py`
- `scripts/agent/validate_inference_stub.py`
- `scripts/agent/validate_metrics.py`
- `scripts/agent/run_fake_inference.py`
- `scripts/agent/run_toy_metrics.py`
- `tasks/0010-training-loop-dry-run.md`
- any file Claude is allowed to create or modify in this task, if it already exists

## Files Claude May Modify

- `src/cv_forensics/__init__.py`
- `src/cv_forensics/training_dry_run.py`
- `configs/training/dry_run_training.example.json`
- `scripts/agent/run_training_dry_run.py`
- `scripts/agent/validate_training_dry_run.py`
- `tests/test_training_dry_run.py`
- `docs/training_dry_run.md`

If a parent directory for an allowed file does not exist, Claude may create that parent directory. Claude must not create any other files.

## Forbidden Actions

Claude must not:

- download datasets
- train a real model
- run real optimization over real data
- load real images
- load real masks
- read or inspect `data`, `datasets`, `outputs`, `checkpoints`, `.env`, `.env.*`, or `secrets`
- inspect protected directories or protected files recursively
- write checkpoints
- write prediction outputs
- create `outputs/`
- create `checkpoints/`
- install packages
- access network resources
- run `git push`
- run `git pull`
- run `git fetch`
- run `curl`, `wget`, `ssh`, `scp`, `rsync`, `pip`, `pip3`, `npm`, `yarn`, `pnpm`, `kaggle`, `huggingface-cli`, or `wandb`
- use `rg`
- use `rm -rf`
- use `--permission-mode bypassPermissions`
- use `--dangerously-skip-permissions`
- import `torch`, `torchvision`, `PIL`, `cv2`, `numpy`, `pandas`, `sklearn`, `pytest`, or any third-party package
- create large files
- implement actual Community Forensics-Small training
- implement actual SID-Set training
- implement SNS augmentation
- modify unrelated files

## Important Project Facts

- The final model direction is a lightweight multi-head image forensics prototype.
- Target outputs are class, mask/localization, family/provenance, and reason.
- Class labels are:
  - `real`
  - `synthetic`
  - `tampered`
- Family labels are coarse family-level labels:
  - `LatDiff`
  - `PixDiff`
  - `GAN`
  - `Other`
  - `Real-or-N/A`
- Architecture direction:
  - shared visual backbone
  - 3-way classification head
  - generator-family provenance head
  - conditional localization head
  - evidence aggregation
  - deterministic template-based explanation
- Community Forensics-Small is for shared backbone learning and coarse provenance planning.
- SID-Set is for 3-way classification and tampered mask localization planning.
- This task must not implement actual Community Forensics-Small or SID-Set training yet.
- This task must not implement SNS augmentation.
- SNS augmentation begins only after pre-SNS baseline evaluation is committed.
- Task 0007 defines the model output schema and template explanation schema.
- Task 0008 defines the lightweight inference stub with fake inputs.
- Task 0009 defines metric calculators with toy arrays.

## Implementation Requirements

### 1. Create `src/cv_forensics/training_dry_run.py`

- Use only Python standard library.
- Do not import third-party packages.
- Reuse existing modules where appropriate:
  - `config_schema`
  - `dataset_manifest`
  - `model_output_schema`
  - `inference_stub`
  - `metrics`
- Implement dry-run structures such as:
  - `DryRunTrainingConfig`
  - `DryRunEpochSummary`
  - `DryRunTrainingResult`
- Implement deterministic functions such as:
  - `load_dry_run_training_config`
  - `validate_dry_run_training_config`
  - `build_fake_batches`
  - `run_dry_epoch`
  - `run_dry_training`
  - `summarize_dry_run`
- The dry-run loop must simulate epochs and batches using fake input records only.
- It must not update real model parameters.
- It must not perform real optimization.
- It must not write checkpoints.
- It must not write outputs.
- It must not access data directories.
- It must not read images or masks.
- It must call fake inference and toy metrics to produce dry-run summaries.
- It must keep all behavior deterministic for a fixed config.
- It must preserve fields needed for later real training:
  - `stage`
  - `dataset_plan`
  - `family_policy`
  - `threshold_tau`
  - `epochs`
  - `batch_size`
  - `seed`
  - `metric_names`
  - `no_download`
  - `no_training`
  - `no_network`
  - `no_outputs`
  - `no_checkpoints`
- It must validate dry-run safety recursively and reject:
  - `.env`
  - `.env.*`
  - `secrets`
  - `data`
  - `datasets`
  - `outputs`
  - `checkpoints`
  - absolute local paths such as `/home/`, `/mnt/`, `/root/`, `/Users/`
  - Windows drive paths
  - URLs such as `http://`, `https://`, `ftp://`, `s3://`, `gs://`, `hf://`
  - secret-looking keys or values such as `token`, `api_key`, `password`, `secret`, `credential`, `auth`, or `bearer`

### 2. Create `configs/training/dry_run_training.example.json`

- It must be a dry-run example only.
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
  - `stage: dry_run`
  - `epochs`
  - `batch_size`
  - `seed`
  - `threshold_tau`
  - `fake_input_config_ref`
  - `toy_metrics_config_ref`
  - `dataset_manifest_refs`
  - `metric_names`
  - `family_policy`
- `dataset_manifest_refs` must use symbolic names only, not file paths to real data.
- It may reference existing safe config files in `configs/` as config refs only.
- It must not reference `data/`, `datasets/`, `outputs/`, `checkpoints/`, `.env`, or secret-like values.

### 3. Create `scripts/agent/run_training_dry_run.py`

- Use only Python standard library.
- Be import-safe without package installation.
- Add the repository `src` path to `sys.path` if needed.
- Accept exactly one positional argument:
  - dry-run training config JSON path
- Run the dry-run training simulation.
- Print a JSON summary to stdout.
- Do not write files.
- Do not create outputs or checkpoints.
- Do not access real datasets.
- Do not read images or masks.
- Exit non-zero with actionable errors on invalid config.

### 4. Create `scripts/agent/validate_training_dry_run.py`

- Use only Python standard library.
- Be import-safe without package installation.
- Add the repository `src` path to `sys.path` if needed.
- Accept exactly one positional argument:
  - dry-run training config JSON path
- Validate the dry-run training config.
- Run a dry-run training simulation.
- Validate that no checkpoint, output, data, image, or mask access is requested.
- Validate that the produced summary includes expected metrics.
- Validate that `dry_run` is true and `no_training` is true.
- Validate that `no_outputs` is true and `no_checkpoints` is true.
- Print a concise success message on pass.
- Exit non-zero with actionable errors on fail.

### 5. Create `tests/test_training_dry_run.py`

- Use only Python standard library.
- Do not import `pytest`.
- Keep pytest-style test functions using plain `assert`.
- Make the file runnable directly with:

```bash
python3 tests/test_training_dry_run.py
```

- Keep it pytest-compatible if pytest is installed.
- Tests should cover:
  - valid dry-run config passes
  - `dry_run: false` is rejected
  - `no_training: false` is rejected
  - `no_outputs: false` is rejected
  - `no_checkpoints: false` is rejected
  - protected paths are rejected
  - URL values are rejected
  - secret-like keys or values are rejected
  - `run_dry_training` returns deterministic results
  - output summary includes metric names from task 0009
  - no files are written by the runner
  - fake batches are deterministic

### 6. Create `docs/training_dry_run.md`

- Explain that this is not real training.
- Explain how this dry-run skeleton connects:
  - config schema
  - dataset manifests
  - fake inference
  - toy metrics
- Explain what will be needed before real training:
  - explicit user approval
  - local data readiness gate
  - validated manifests
  - output/checkpoint policy
  - small local subset smoke test
- Explain that SNS augmentation is not implemented here.
- Include marker string:
  - `TRAINING_DRY_RUN_OK`
- Keep it concise but useful.

### 7. Update `src/cv_forensics/__init__.py` Only If Needed

- Add a lightweight export for `training_dry_run` only if needed.
- Do not add heavy imports.
- Do not import third-party packages.

## Validation Commands

```bash
python3 scripts/agent/validate_training_dry_run.py configs/training/dry_run_training.example.json
```

```bash
python3 scripts/agent/run_training_dry_run.py configs/training/dry_run_training.example.json
```

```bash
python3 tests/test_training_dry_run.py
```

```bash
test -f src/cv_forensics/training_dry_run.py
```

```bash
test -f configs/training/dry_run_training.example.json
```

```bash
test -f scripts/agent/run_training_dry_run.py
```

```bash
test -f scripts/agent/validate_training_dry_run.py
```

```bash
test -f docs/training_dry_run.md
```

```bash
grep -q TRAINING_DRY_RUN_OK docs/training_dry_run.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- src/cv_forensics/__init__.py src/cv_forensics/training_dry_run.py configs/training/dry_run_training.example.json scripts/agent/run_training_dry_run.py scripts/agent/validate_training_dry_run.py tests/test_training_dry_run.py docs/training_dry_run.md
```

Optional validation command:

```bash
pytest -q tests/test_training_dry_run.py
```

## Acceptance Criteria

- `python3 scripts/agent/validate_training_dry_run.py configs/training/dry_run_training.example.json` passes.
- `python3 scripts/agent/run_training_dry_run.py configs/training/dry_run_training.example.json` prints a JSON summary.
- `python3 tests/test_training_dry_run.py` passes.
- If pytest exists, `pytest -q tests/test_training_dry_run.py` also passes.
- Changed files are limited to the allowed task files.
- The dry-run training loop is deterministic.
- The dry-run loop reuses fake inference and toy metrics.
- No real data, images, masks, outputs, checkpoints, secrets, package installs, training, downloads, or network access are used.
- No SNS augmentation is implemented.
- `docs/training_dry_run.md` contains `TRAINING_DRY_RUN_OK`.

## Stop Condition

After implementation, Claude must stop and report:

- files changed
- validation commands run and their results
- confirmation that no real datasets, images, masks, checkpoints, outputs, secrets, package installation, training, downloads, network access, or SNS augmentation were used
- any skipped optional validation, with the reason

Claude must not commit changes.
