# Task 0017: CF-Small Local Tiny Training Smoke

## Task Title

Create a Community Forensics-Small local tiny training smoke workflow for non-SNS baseline verification.

## Role

Codex is the implementation worker in Codex-only mode. Codex must implement only the files allowed by this task file after the task file is committed and after the user explicitly says to implement.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `docs/local_data_readiness.md`
- `docs/cf_small_subset_smoke.md`
- `docs/cf_small_baseline_plan.md`
- `docs/pre_sns_baseline_report.md`
- `configs/local_data/readiness.example.json`
- `configs/local_data/cf_small_subset_smoke.example.json`
- `configs/training/cf_small_baseline.example.json`
- `configs/reports/pre_sns_baseline_report.example.json`
- `src/cv_forensics/local_data_gate.py`
- `src/cv_forensics/metrics.py`, only if it exists
- `src/cv_forensics/training_dry_run.py`, only if it exists
- `tasks/0017-cf-small-local-tiny-training-smoke.md`
- `tests/test_metrics.py`, only if useful
- `tests/test_training_dry_run.py`, only if useful
- `tests/test_cf_small_baseline_plan.py`, only if useful

## Files Codex May Modify

- `.gitignore`
- `configs/training/cf_small_tiny_train_smoke.example.json`
- `scripts/agent/validate_cf_small_tiny_train_smoke_config.py`
- `scripts/training/run_cf_small_tiny_train_smoke.py`
- `tests/test_cf_small_tiny_train_smoke.py`
- `docs/cf_small_tiny_training_smoke.md`

If a parent directory for an allowed file does not exist, Codex may create that parent directory. Codex must not create or modify any other files.

## Forbidden Actions During Implementation

Codex must not:

- download datasets
- download Community Forensics-Small
- train a model on real data
- run real optimization on real data
- inspect actual dataset directories recursively
- read real images
- read real masks
- validate actual real image file existence outside temporary test fixtures
- access `.env`, `.env.*`, secrets, data, datasets, outputs, or checkpoints
- create `outputs/`
- create `checkpoints/`
- write generated predictions
- write checkpoints
- install packages
- access network resources from shell commands
- run `git push`, `git pull`, or `git fetch`
- run `curl`, `wget`, `ssh`, `scp`, `rsync`, `pip`, `npm`, `yarn`, `pnpm`, `kaggle`, `huggingface-cli`, or `wandb`
- use `rg`
- use `rm -rf`
- create large files
- implement SNS augmentation
- run SNS perturbation evaluation
- use `danger-full-access`
- bypass permissions

## Explicitly Allowed Future User-Run Behavior

The implemented smoke runner may support a future manual command that reads tiny, user-approved, non-SNS Community Forensics-Small sample paths from an untracked local config file. This future user-run mode may:

- read only explicitly listed image files in the local config
- reject recursive directory traversal
- reject download URLs
- reject protected repo paths
- reject outputs and checkpoints unless explicitly disabled or approval-gated
- run a tiny CPU-only training smoke with strict limits
- print a JSON summary to stdout

The future user-run mode must not:

- download data
- scan full dataset directories recursively
- run full training
- write checkpoints
- write outputs by default
- train beyond tiny limits
- use SNS augmentation

## Important Project Facts

- The user wants to train on non-SNS original data first, then later evaluate robustness on SNS-transformed data.
- Community Forensics-Small is the first baseline stage for real-vs-synthetic classification and backbone/provenance planning.
- This tiny training smoke is not a performance claim.
- This tiny training smoke only verifies data connection, loader, model forward/backward, finite loss, and metric plumbing.
- Full baseline training must remain gated behind explicit user approval after this smoke stage.
- SNS augmentation is not implemented here.
- SNS robustness evaluation is not run here.
- SID-Set 3-way and localization smoke will be a later task.

## Implementation Requirements

1. Update `.gitignore` if needed
   - Ensure local real-data configs are not accidentally committed.
   - Add ignore rules only if they are missing.
   - Suggested ignore patterns:
     - `.local/`
     - `*.local.json`
     - `configs/**/*.local.json`
   - Do not remove existing `.gitignore` content.

2. Create `configs/training/cf_small_tiny_train_smoke.example.json`
   - The tracked example config must be symbolic and safe to commit.
   - It must not contain real local dataset paths.
   - It must not contain URLs.
   - It must not contain secrets.
   - It must not contain `data/`, `datasets/`, `outputs/`, `checkpoints/`, `.env`, or secret-like values.
   - It must include:
     - `schema_version`
     - `config_kind: example_symbolic`
     - `smoke_name`
     - `dataset_name: Community Forensics-Small`
     - `stage: non_sns_cf_small_tiny_training_smoke`
     - `execution_mode: example_only`
     - `dry_run: true`
     - `no_download: true`
     - `no_network: true`
     - `no_outputs: true`
     - `no_checkpoints: true`
     - `no_sns_augmentation: true`
     - `no_sns_perturbation_eval: true`
     - `requires_user_approval_for_real_data: true`
     - `local_config_template_path`
     - `approved_real_mode_policy`
     - `tiny_limits`
     - `class_policy`
     - `sample_manifest_policy`
     - `training_smoke_policy`
     - `metric_policy`
     - `output_policy`
     - `checkpoint_policy`
     - `next_stage_policy`
     - `validation_notes`
   - `class_policy` must represent real-vs-synthetic binary classification.
   - `tiny_limits` must include conservative caps:
     - `max_samples`
     - `max_steps`
     - `max_epochs`
     - `max_image_size`
     - `cpu_only`
   - `sample_manifest_policy` must require explicit listed sample paths, not recursive directory scanning.
   - `output_policy` must state stdout summary only by default.
   - `checkpoint_policy` must state no checkpoint writing by default.
   - `next_stage_policy` must state that SID-Set tiny multi-head smoke and full baseline training are later explicit-approval stages.

3. Create `scripts/agent/validate_cf_small_tiny_train_smoke_config.py`
   - Use only Python standard library.
   - Accept exactly one positional argument: config JSON path.
   - Add repo `src` path to `sys.path` if needed.
   - Reuse `src/cv_forensics/local_data_gate.py` safety checks where appropriate.
   - Support two config kinds:
     - `example_symbolic`
     - `approved_local_smoke`
   - For `example_symbolic`:
     - reject real paths
     - reject absolute paths
     - reject URLs
     - reject protected paths
     - require `execution_mode: example_only`
     - require `dry_run true`
     - require `no_download true`
     - require `no_network true`
     - require `no_outputs true`
     - require `no_checkpoints true`
     - require `no_sns_augmentation true`
     - require `no_sns_perturbation_eval true`
     - require `requires_user_approval_for_real_data true`
   - For `approved_local_smoke`:
     - require `approved_real_data_access true`
     - require `user_approval_text` exactly `I_APPROVE_LOCAL_NON_SNS_TINY_TRAINING_SMOKE`
     - require `no_download true`
     - require `no_network true`
     - require `no_outputs true` by default
     - require `no_checkpoints true` by default
     - require `no_sns_augmentation true`
     - require `no_sns_perturbation_eval true`
     - allow local absolute paths only when they are explicitly listed sample files and are under `approved_local_roots`
     - reject paths under the repository protected directories: `.env`, secrets, data, datasets, outputs, checkpoints
     - reject URLs and remote schemes
     - reject recursive scan flags
     - reject sample counts above `tiny_limits`
     - reject `max_steps` above `tiny_limits`
     - reject `max_epochs` above `tiny_limits`
   - Validate class labels are real and synthetic only.
   - Validate sample manifest entries include:
     - `sample_id`
     - `image_path`
     - `label`
   - Validate labels are only real or synthetic.
   - Validate both classes are represented for `approved_local_smoke`.
   - Validate all paths are explicit file paths; no directory-only manifest entries.
   - Print actionable errors and exit non-zero on fail.
   - Print a concise success message containing `CF_SMALL_TINY_TRAIN_SMOKE_CONFIG_OK` on pass.
   - Do not write files.
   - Do not read images.
   - Do not inspect dataset directories recursively.

4. Create `scripts/training/run_cf_small_tiny_train_smoke.py`
   - This script is for future manual user execution with an `approved_local_smoke` config.
   - Use Python standard library plus `torch` and `PIL` only if available.
   - Do not install packages.
   - Add repo `src` path to `sys.path` if needed.
   - Import and call the validator before any image loading or training.
   - Fail with an actionable message if `torch` or `PIL` is unavailable.
   - Use CPU by default.
   - Read only explicitly listed image paths from the approved local config.
   - Do not recursively scan directories.
   - Do not download anything.
   - Do not write checkpoints.
   - Do not write outputs by default.
   - Do not use SNS augmentation.
   - Implement a minimal tiny binary classifier training smoke:
     - deterministic seed
     - image load and resize to `max_image_size`
     - simple tensor conversion
     - tiny linear or small CNN model
     - cross-entropy loss
     - strict `max_samples`, `max_steps`, and `max_epochs` caps
     - finite loss checks
     - one or more optimizer steps only within tiny limits
   - Print a JSON summary to stdout containing:
     - `marker: CF_SMALL_TINY_TRAIN_SMOKE_OK`
     - `dataset_name`
     - `samples_seen`
     - `classes_seen`
     - `steps_completed`
     - `epochs_completed`
     - `initial_loss`
     - `final_loss`
     - `finite_loss`
     - `smoke_accuracy`
     - `no_download`
     - `no_network`
     - `no_outputs`
     - `no_checkpoints`
     - `no_sns_augmentation`
   - Do not claim real baseline performance.
   - The summary must state this is a tiny smoke result, not a final metric.

5. Create `tests/test_cf_small_tiny_train_smoke.py`
   - Use Python standard library for assertions and test harness.
   - Do not import `pytest`.
   - Tests must be runnable with:
     `python3 tests/test_cf_small_tiny_train_smoke.py`
   - Tests may create temporary tiny image fixtures outside the repository using `tempfile`.
   - Tests may use `PIL` and `torch` only for testing the training runner if available.
   - If `torch` or `PIL` is unavailable, tests must still validate config behavior and skip runner execution with a clear message rather than failing the entire test suite.
   - Tests should cover:
     - tracked example config passes validation
     - `approved_local_smoke` config with temporary tiny images passes validation
     - `approved_local_smoke` config without approval text is rejected
     - URL `image_path` is rejected
     - protected repo path is rejected
     - absolute path outside `approved_local_roots` is rejected
     - directory-only sample path is rejected
     - recursive scan flag is rejected
     - `max_samples` above cap is rejected
     - `max_steps` above cap is rejected
     - `max_epochs` above cap is rejected
     - missing real class is rejected
     - missing synthetic class is rejected
     - label other than real or synthetic is rejected
     - `no_download false` is rejected
     - `no_network false` is rejected
     - `no_outputs false` is rejected
     - `no_checkpoints false` is rejected
     - `no_sns_augmentation false` is rejected
     - `no_sns_perturbation_eval false` is rejected
     - secret-like keys and values are rejected
     - ordinary prose containing words like authoritative or authentication is not rejected merely because it contains auth
     - training runner prints `CF_SMALL_TINY_TRAIN_SMOKE_OK` when `torch` and `PIL` are available
     - training runner does not write outputs or checkpoints by default
   - Tests must not access real datasets.
   - Tests must not access `.env`, secrets, data, datasets, outputs, or checkpoints.
   - Tests must not download anything.
   - Tests must not run full training.

6. Create `docs/cf_small_tiny_training_smoke.md`
   - Explain the purpose of the Community Forensics-Small local tiny training smoke.
   - Explain that this is the first controlled bridge from dry-run planning to approved local non-SNS data.
   - Explain that this verifies:
     - local manifest connection
     - tiny image loading
     - binary real-vs-synthetic label handling
     - minimal model forward/backward
     - finite loss
     - smoke accuracy plumbing
   - Explain that this is not full training and not a performance claim.
   - Explain that no SNS augmentation is used.
   - Explain that no SNS perturbation evaluation is run.
   - Explain that no outputs or checkpoints are written by default.
   - Explain that actual user-run mode requires an untracked local config with:
     - `config_kind: approved_local_smoke`
     - `approved_real_data_access: true`
     - `user_approval_text: I_APPROVE_LOCAL_NON_SNS_TINY_TRAINING_SMOKE`
     - explicit `approved_local_roots`
     - explicit `sample_manifest` entries
   - Explain that after this passes, the next natural task is SID-Set local tiny multi-head smoke.
   - Include marker string:
     - `CF_SMALL_TINY_TRAIN_SMOKE_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_cf_small_tiny_train_smoke_config.py configs/training/cf_small_tiny_train_smoke.example.json
```

```bash
python3 tests/test_cf_small_tiny_train_smoke.py
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0017-cf-small-local-tiny-training-smoke.md
```

```bash
test -f configs/training/cf_small_tiny_train_smoke.example.json
```

```bash
test -f scripts/agent/validate_cf_small_tiny_train_smoke_config.py
```

```bash
test -f scripts/training/run_cf_small_tiny_train_smoke.py
```

```bash
test -f tests/test_cf_small_tiny_train_smoke.py
```

```bash
test -f docs/cf_small_tiny_training_smoke.md
```

```bash
grep -q CF_SMALL_TINY_TRAIN_SMOKE_OK docs/cf_small_tiny_training_smoke.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- .gitignore configs/training/cf_small_tiny_train_smoke.example.json scripts/agent/validate_cf_small_tiny_train_smoke_config.py scripts/training/run_cf_small_tiny_train_smoke.py tests/test_cf_small_tiny_train_smoke.py docs/cf_small_tiny_training_smoke.md
```

Optional validation command:

```bash
pytest -q tests/test_cf_small_tiny_train_smoke.py
```

## Acceptance Criteria

- `python3 scripts/agent/validate_cf_small_tiny_train_smoke_config.py configs/training/cf_small_tiny_train_smoke.example.json` passes.
- `python3 tests/test_cf_small_tiny_train_smoke.py` passes.
- `python3 scripts/agent/check_agent_changes.py tasks/0017-cf-small-local-tiny-training-smoke.md` passes.
- If pytest exists, `pytest -q tests/test_cf_small_tiny_train_smoke.py` passes.
- `docs/cf_small_tiny_training_smoke.md` contains `CF_SMALL_TINY_TRAIN_SMOKE_OK`.
- Changed files are limited to:
  - `.gitignore`
  - `configs/training/cf_small_tiny_train_smoke.example.json`
  - `scripts/agent/validate_cf_small_tiny_train_smoke_config.py`
  - `scripts/training/run_cf_small_tiny_train_smoke.py`
  - `tests/test_cf_small_tiny_train_smoke.py`
  - `docs/cf_small_tiny_training_smoke.md`
- The tracked example config is symbolic and safe to commit.
- Local real-data configs are ignored by git.
- The validator supports approved local tiny smoke configs with explicit approval.
- The future runner can train only on tiny explicitly listed local samples.
- The future runner does not recursively scan dataset directories.
- The future runner does not write checkpoints.
- The future runner does not write outputs by default.
- The future runner does not use SNS augmentation.
- Implementation does not access real datasets, real images, real masks, protected paths, network resources, outputs, or checkpoints.

## Stop Condition

Stop after creating the task file. Report:

- task file path
- short summary
- git status
- exact git add command for the task file
- exact git commit command for the task file
