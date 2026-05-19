# Task 0018: CF-Small Multi-Head Provenance Tiny Smoke

## Task Title

Create a Community Forensics-Small multi-head tiny training smoke workflow for non-SNS pre-baseline model verification.

## Role

Codex is the implementation worker in Codex-only mode. Codex must implement only the files allowed by this task file after the task file is committed and after the user explicitly says to implement.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `docs/cf_small_tiny_training_smoke.md`
- `docs/cf_small_baseline_plan.md`
- `docs/pre_sns_baseline_report.md`
- `configs/training/cf_small_tiny_train_smoke.example.json`
- `scripts/agent/validate_cf_small_tiny_train_smoke_config.py`
- `scripts/training/run_cf_small_tiny_train_smoke.py`
- `tests/test_cf_small_tiny_train_smoke.py`
- `src/cv_forensics/local_data_gate.py`
- `src/cv_forensics/metrics.py`, if it exists
- `tasks/0018-cf-small-multihead-provenance-tiny-smoke.md`

## Files Codex May Modify

- `configs/training/cf_small_multihead_tiny_smoke.example.json`
- `scripts/agent/validate_cf_small_multihead_tiny_smoke_config.py`
- `scripts/training/run_cf_small_multihead_tiny_smoke.py`
- `tests/test_cf_small_multihead_tiny_smoke.py`
- `docs/cf_small_multihead_tiny_smoke.md`

If a parent directory for an allowed file does not exist, Codex may create that parent directory. Codex must not create or modify any other files.

## Forbidden Actions During Implementation

Codex must not:

- download datasets
- download Community Forensics-Small
- train on real data
- run real optimization on real data
- inspect actual dataset directories recursively
- read real images outside temporary test fixtures
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

The implemented runner may support a future manual command that reads a user-approved local config containing explicitly listed Community Forensics-Small sample image paths and metadata. Future user-run mode may read only explicitly listed local image files and may run a strict tiny CPU-only multi-head smoke. It must not recursively scan directories, download data, write outputs, write checkpoints, or use SNS augmentation.

## Important Project Facts

- The proposal target is not just real/fake classification.
- The pre-SNS model must eventually provide class, mask/localization, family/provenance, and reason.
- Community Forensics-Small is used for shared backbone and generator-family provenance learning.
- Generator-family provenance is not exact model attribution.
- The coarse family labels are architecture-centered:
  - `LatDiff`
  - `PixDiff`
  - `GAN`
  - `Other`
  - `Real-or-N/A`
- `model_name` and `subset` are analysis metadata and must be preserved in local manifests/configs when available.
- SID-Set 3-way classification and tampered localization will be a later task.
- SNS augmentation is not part of this task.

## Implementation Requirements

1. Create `configs/training/cf_small_multihead_tiny_smoke.example.json`
   - Tracked example config must be symbolic and safe to commit.
   - It must not contain real paths, URLs, secrets, data, datasets, outputs, checkpoints, or `.env`.
   - It must include `schema_version`, `config_kind`, `smoke_name`, `dataset_name`, `stage`, `execution_mode`.
   - It must include guardrails:
     - `dry_run: true`
     - `no_download: true`
     - `no_network: true`
     - `no_outputs: true`
     - `no_checkpoints: true`
     - `no_sns_augmentation: true`
     - `no_sns_perturbation_eval: true`
     - `requires_user_approval_for_real_data: true`
   - It must define class labels:
     - `real`
     - `synthetic`
   - It must define family labels:
     - `LatDiff`
     - `PixDiff`
     - `GAN`
     - `Other`
     - `Real-or-N/A`
   - It must document that real samples use `Real-or-N/A`.
   - It must document that synthetic family labels are derived from architecture metadata.
   - It must require `model_name` and `subset` metadata to be preserved when available.
   - It must include `tiny_limits`:
     - `max_samples`
     - `max_steps`
     - `max_epochs`
     - `max_image_size`
     - `cpu_only`

2. Create `scripts/agent/validate_cf_small_multihead_tiny_smoke_config.py`
   - Use only Python standard library.
   - Accept exactly one positional config path.
   - Add repo `src` path to `sys.path` if needed.
   - Reuse `src/cv_forensics/local_data_gate.py` safety checks where appropriate.
   - Support `config_kind`:
     - `example_symbolic`
     - `approved_local_multihead_smoke`
   - For `example_symbolic`:
     - validate symbolic safe structure and guardrails.
   - For `approved_local_multihead_smoke`:
     - require `approved_real_data_access true`
     - require `user_approval_text` exactly `I_APPROVE_LOCAL_NON_SNS_CF_SMALL_MULTIHEAD_TINY_SMOKE`
     - require `no_download`, `no_network`, `no_outputs`, `no_checkpoints`, `no_sns_augmentation`, `no_sns_perturbation_eval` true
     - allow local absolute image paths only if explicitly listed and under `approved_local_roots`
     - reject URLs, remote schemes, protected repo paths, recursive scan flags, directory-only sample paths, and secret-like values
     - validate `sample_manifest` entries include:
       - `sample_id`
       - `image_path`
       - `label`
       - `architecture`
       - `family_label`
       - `model_name`, optional but preserve if available
       - `subset`, optional but preserve if available
     - validate `label` is real or synthetic
     - validate `family_label` is one of `LatDiff`, `PixDiff`, `GAN`, `Other`, `Real-or-N/A`
     - validate real samples have `family_label` `Real-or-N/A`
     - validate synthetic samples do not use `Real-or-N/A` unless explicitly configured as `unknown_synthetic_family_allowed`
     - validate at least both class labels are present
     - validate at least one synthetic family label is present
     - validate `tiny_limits` are conservative
   - Print `CF_SMALL_MULTIHEAD_TINY_SMOKE_CONFIG_OK` on success.
   - Print actionable errors and exit non-zero on failure.
   - Do not write files.
   - Do not read real images.
   - Do not scan dataset directories.

3. Create `scripts/training/run_cf_small_multihead_tiny_smoke.py`
   - For future manual execution with an `approved_local_multihead_smoke` config.
   - Use Python standard library plus `torch` and `PIL` only if available.
   - Do not install packages.
   - Import and call the validator before image loading.
   - Use CPU by default and respect `cpu_only`.
   - Read only explicitly listed image paths.
   - Do not recursively scan directories.
   - Do not download anything.
   - Do not write checkpoints.
   - Do not write outputs by default.
   - Do not use SNS augmentation.
   - Implement a tiny shared-backbone multi-head model:
     - shared feature extractor
     - binary class head
     - family/provenance head
   - Compute:
     - class cross-entropy loss
     - family cross-entropy loss
     - `total_loss = class_loss + family_loss_weight * family_loss`
   - Use deterministic seed.
   - Enforce `max_samples`, `max_steps`, `max_epochs`.
   - Check all losses are finite.
   - Print JSON summary containing:
     - `marker: CF_SMALL_MULTIHEAD_TINY_SMOKE_OK`
     - `dataset_name`
     - `samples_seen`
     - `class_labels_seen`
     - `family_labels_seen`
     - `steps_completed`
     - `epochs_completed`
     - `initial_total_loss`
     - `final_total_loss`
     - `class_loss_finite`
     - `family_loss_finite`
     - `total_loss_finite`
     - `class_smoke_accuracy`
     - `family_smoke_accuracy`
     - `no_download`
     - `no_network`
     - `no_outputs`
     - `no_checkpoints`
     - `no_sns_augmentation`
     - `result_scope: tiny smoke only; not a final metric or performance claim`

4. Create `tests/test_cf_small_multihead_tiny_smoke.py`
   - Use Python standard library for test harness and assertions.
   - Do not import `pytest`.
   - Runnable with:
     `python3 tests/test_cf_small_multihead_tiny_smoke.py`
   - Tests may create temporary tiny image fixtures outside the repository using `tempfile`.
   - Tests may use `PIL` and `torch` only for runner smoke if available.
   - If `PIL` or `torch` is unavailable, skip runner execution with a clear message, but still test config validation.
   - Tests must cover:
     - example config passes
     - approved local multihead config with temporary tiny images passes
     - missing approval text rejected
     - URL path rejected
     - protected path rejected
     - directory-only path rejected
     - recursive scan flag rejected
     - label outside real/synthetic rejected
     - `family_label` outside allowed set rejected
     - real sample with non `Real-or-N/A` family rejected
     - synthetic sample with `Real-or-N/A` rejected unless explicitly allowed
     - missing architecture rejected
     - missing both class labels rejected
     - missing synthetic family rejected
     - `max_samples` above cap rejected
     - `max_steps` above cap rejected
     - `no_download false` rejected
     - `no_network false` rejected
     - `no_outputs false` rejected
     - `no_checkpoints false` rejected
     - `no_sns_augmentation false` rejected
     - `no_sns_perturbation_eval false` rejected
     - ordinary prose containing authoritative or authentication not rejected merely because it contains auth
     - runner prints `CF_SMALL_MULTIHEAD_TINY_SMOKE_OK` when `torch` and `PIL` are available
     - runner does not write outputs or checkpoints
   - Tests must not access real datasets, `.env`, secrets, data, datasets, outputs, or checkpoints.
   - Tests must not download anything.
   - Tests must not run full training.

5. Create `docs/cf_small_multihead_tiny_smoke.md`
   - Explain the purpose of the CF-Small multi-head tiny smoke.
   - Explain that it extends binary real/synthetic smoke with a generator-family provenance head.
   - Explain that family labels are coarse architecture labels, not exact model attribution.
   - Explain the family labels:
     - `LatDiff`
     - `PixDiff`
     - `GAN`
     - `Other`
     - `Real-or/N-A`
   - Explain that `model_name` and `subset` are analysis metadata.
   - Explain that this is not full training and not a performance claim.
   - Explain that SID-Set 3-way and localization are the next stage.
   - Explain that SNS augmentation is not implemented here.
   - Include marker:
     `CF_SMALL_MULTIHEAD_TINY_SMOKE_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_cf_small_multihead_tiny_smoke_config.py configs/training/cf_small_multihead_tiny_smoke.example.json
```

```bash
python3 tests/test_cf_small_multihead_tiny_smoke.py
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0018-cf-small-multihead-provenance-tiny-smoke.md
```

```bash
test -f configs/training/cf_small_multihead_tiny_smoke.example.json
```

```bash
test -f scripts/agent/validate_cf_small_multihead_tiny_smoke_config.py
```

```bash
test -f scripts/training/run_cf_small_multihead_tiny_smoke.py
```

```bash
test -f tests/test_cf_small_multihead_tiny_smoke.py
```

```bash
test -f docs/cf_small_multihead_tiny_smoke.md
```

```bash
grep -q CF_SMALL_MULTIHEAD_TINY_SMOKE_OK docs/cf_small_multihead_tiny_smoke.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- configs/training/cf_small_multihead_tiny_smoke.example.json scripts/agent/validate_cf_small_multihead_tiny_smoke_config.py scripts/training/run_cf_small_multihead_tiny_smoke.py tests/test_cf_small_multihead_tiny_smoke.py docs/cf_small_multihead_tiny_smoke.md
```

Optional validation command:

```bash
pytest -q tests/test_cf_small_multihead_tiny_smoke.py
```

## Acceptance Criteria

- Validator passes for the tracked example config.
- Standalone tests pass.
- `check_agent_changes` passes.
- If pytest exists, pytest passes.
- Docs contain `CF_SMALL_MULTIHEAD_TINY_SMOKE_OK`.
- Changed files are limited to the allowed files.
- The implementation supports a future approved local config with explicit image paths and metadata.
- The future runner computes finite class, family, and total losses.
- The future runner prints `CF_SMALL_MULTIHEAD_TINY_SMOKE_OK`.
- No dataset download, full training, network access, output writing, checkpoint writing, protected path access, SNS augmentation, or SNS perturbation evaluation occurs during implementation.

## Stop Condition

Stop after creating the task file. Report:

- task file path
- short summary
- git status
- exact git add command
- exact git commit command
