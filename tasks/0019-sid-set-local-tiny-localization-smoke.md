# Task 0019: SID-Set Local Tiny Localization Smoke

## Role

Codex is the implementation worker in Codex-only mode. Codex must implement only the files allowed by this task file after the task file is committed and after the user explicitly says to implement.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `docs/model_output_schema.md`
- `docs/inference_stub.md`
- `docs/cf_small_tiny_training_smoke.md`
- `docs/cf_small_multihead_tiny_smoke.md`
- `configs/training/cf_small_multihead_tiny_smoke.example.json`
- `scripts/agent/validate_cf_small_multihead_tiny_smoke_config.py`
- `scripts/training/run_cf_small_multihead_tiny_smoke.py`
- `tests/test_cf_small_multihead_tiny_smoke.py`
- `src/cv_forensics/local_data_gate.py`
- `src/cv_forensics/metrics.py`, only if it exists
- `tasks/0019-sid-set-local-tiny-localization-smoke.md`

## Files Codex May Modify

- `configs/training/sid_set_tiny_localization_smoke.example.json`
- `scripts/agent/validate_sid_set_tiny_localization_smoke_config.py`
- `scripts/training/run_sid_set_tiny_localization_smoke.py`
- `tests/test_sid_set_tiny_localization_smoke.py`
- `docs/sid_set_tiny_localization_smoke.md`

If a parent directory for an allowed file does not exist, Codex may create that parent directory. Codex must not create or modify any other files.

## Forbidden Actions During Implementation

Codex must not:

- download datasets
- download SID-Set
- train on real data
- run real optimization on real data
- inspect actual dataset directories recursively
- read real images outside temporary test fixtures
- read real masks outside temporary test fixtures
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
- use danger-full-access
- bypass permissions

## Explicitly Allowed Future User-Run Behavior

The implemented runner may support a future manual command that reads a user-approved local config containing explicitly listed SID-Set sample image paths and optional mask paths.

Future user-run mode may read only explicitly listed local image files and explicitly listed local mask files. It may run a strict tiny CPU-only 3-way classification and localization smoke.

It must not recursively scan directories, download data, write outputs, write checkpoints, or use SNS augmentation.

## Important Project Facts

- The pre-SNS model must eventually provide class, mask/localization, family/provenance, and reason.
- Community Forensics-Small covers binary/provenance preparation.
- SID-Set covers 3-way classification and tampered localization.
- SID-Set labels are:
  - `0`: real
  - `1`: full synthetic
  - `2`: tampered
- Tampered samples have binary masks highlighting manipulated regions.
- SID-Set samples may not have Community Forensics-Small-style family/provenance labels.
- For this task, family/provenance loss is not applied to SID-Set samples.
- This task does not implement SNS augmentation or SNS perturbation evaluation.
- This task does not train the final model.
- This task must not download SID-Set.

## Implementation Requirements

1. Create `configs/training/sid_set_tiny_localization_smoke.example.json`
   - The tracked example config must be symbolic and safe to commit.
   - It must not contain real paths, URLs, secrets, data, datasets, outputs, checkpoints, or `.env`.
   - It must include `schema_version`, `config_kind`, `smoke_name`, `dataset_name`, `stage`, and `execution_mode`.
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
     - `full_synthetic`
     - `tampered`
   - It must define label ID mapping:
     - `0 -> real`
     - `1 -> full_synthetic`
     - `2 -> tampered`
   - It must document that only tampered samples require masks.
   - It must document that real and full_synthetic samples must not require masks.
   - It must document that family/provenance labels are not required for SID-Set samples in this task.
   - It must include tiny limits:
     - `max_samples`
     - `max_steps`
     - `max_epochs`
     - `max_image_size`
     - `cpu_only`

2. Create `scripts/agent/validate_sid_set_tiny_localization_smoke_config.py`
   - Use only Python standard library.
   - Accept exactly one positional config path.
   - Add repo `src` path to `sys.path` if needed.
   - Reuse `src/cv_forensics/local_data_gate.py` safety checks where appropriate.
   - Support `config_kind`:
     - `example_symbolic`
     - `approved_local_sid_tiny_localization_smoke`
   - For `example_symbolic`:
     - validate symbolic safe structure and guardrails.
   - For `approved_local_sid_tiny_localization_smoke`:
     - require `approved_real_data_access: true`
     - require `user_approval_text` exactly:
       `I_APPROVE_LOCAL_NON_SNS_SID_SET_TINY_LOCALIZATION_SMOKE`
     - require `no_download`, `no_network`, `no_outputs`, `no_checkpoints`, `no_sns_augmentation`, and `no_sns_perturbation_eval` true
     - allow local absolute image paths only if explicitly listed and under `approved_local_roots`
     - allow local absolute mask paths only if explicitly listed and under `approved_local_roots`
     - reject URLs, remote schemes, protected repo paths, recursive scan flags, directory-only sample paths, secret-like values, and path traversal
     - validate `sample_manifest` entries include:
       - `sample_id`
       - `image_path`
       - `label`
       - `label_id`
       - `split`, optional but preserve if available
       - `img_id`, optional but preserve if available
       - `mask_path` only for tampered samples
     - validate label is `real`, `full_synthetic`, or `tampered`
     - validate label ID mapping:
       - real: `0`
       - full_synthetic: `1`
       - tampered: `2`
     - validate tampered samples have `mask_path`
     - validate non-tampered samples do not require `mask_path`
     - validate at least one sample for each class is present
     - validate at least one tampered sample with `mask_path` is present
     - validate tiny limits are conservative
   - Print `SID_SET_TINY_LOCALIZATION_SMOKE_CONFIG_OK` on success.
   - Print actionable errors and exit non-zero on failure.
   - Do not write files.
   - Do not read real images.
   - Do not read real masks.
   - Do not scan dataset directories.

3. Create `scripts/training/run_sid_set_tiny_localization_smoke.py`
   - For future manual execution with an `approved_local_sid_tiny_localization_smoke` config.
   - Use Python standard library plus torch and PIL only if available.
   - Do not install packages.
   - Import and call the validator before image/mask loading.
   - Use CPU by default and respect `cpu_only`.
   - Read only explicitly listed image paths.
   - Read only explicitly listed mask paths for tampered samples.
   - Do not recursively scan directories.
   - Do not download anything.
   - Do not write checkpoints.
   - Do not write outputs by default.
   - Do not use SNS augmentation.
   - Implement a tiny shared-backbone model:
     - shared image feature extractor
     - 3-way class head
     - localization head producing a small mask/logit map
   - Compute:
     - class cross-entropy loss for all samples
     - mask binary loss only for tampered samples with masks
     - `total_loss = class_loss + mask_loss_weight * mask_loss`
   - For non-tampered samples:
     - localization loss must be false or skipped
     - mask target is not required
   - Use deterministic seed.
   - Enforce `max_samples`, `max_steps`, and `max_epochs`.
   - Check all relevant losses are finite.
   - Print JSON summary containing:
     - `marker: SID_SET_TINY_LOCALIZATION_SMOKE_OK`
     - `dataset_name`
     - `samples_seen`
     - `class_labels_seen`
     - `tampered_samples_seen`
     - `masks_seen`
     - `steps_completed`
     - `epochs_completed`
     - `initial_total_loss`
     - `final_total_loss`
     - `class_loss_finite`
     - `mask_loss_finite`
     - `total_loss_finite`
     - `class_smoke_accuracy`
     - `localization_smoke_iou`
     - `localization_loss_applied_to_tampered_only`
     - `no_download`
     - `no_network`
     - `no_outputs`
     - `no_checkpoints`
     - `no_sns_augmentation`
     - `result_scope: tiny smoke only; not a final metric or performance claim`

4. Create `tests/test_sid_set_tiny_localization_smoke.py`
   - Use Python standard library for test harness and assertions.
   - Do not import pytest.
   - Runnable with:
     `python3 tests/test_sid_set_tiny_localization_smoke.py`
   - Tests may create temporary tiny image and mask fixtures outside the repository using `tempfile`.
   - Tests may use PIL and torch only for runner smoke if available.
   - If PIL or torch is unavailable, skip runner execution with a clear message, but still test config validation.
   - Tests must cover:
     - example config passes
     - approved local config with temporary tiny images and tampered mask passes
     - missing approval text rejected
     - URL path rejected
     - protected path rejected
     - directory-only path rejected
     - recursive scan flag rejected
     - label outside real/full_synthetic/tampered rejected
     - label_id mismatch rejected
     - missing real class rejected
     - missing full_synthetic class rejected
     - missing tampered class rejected
     - tampered sample without `mask_path` rejected
     - non-tampered samples do not require `mask_path`
     - `mask_path` on non-tampered sample is either rejected or explicitly ignored according to documented policy
     - `max_samples` above cap rejected
     - `max_steps` above cap rejected
     - `no_download` false rejected
     - `no_network` false rejected
     - `no_outputs` false rejected
     - `no_checkpoints` false rejected
     - `no_sns_augmentation` false rejected
     - `no_sns_perturbation_eval` false rejected
     - ordinary prose containing authoritative or authentication not rejected merely because it contains auth
     - runner prints `SID_SET_TINY_LOCALIZATION_SMOKE_OK` when torch and PIL are available
     - runner reports `localization_loss_applied_to_tampered_only` true
     - runner does not write outputs or checkpoints
   - Tests must not access real datasets, `.env`, secrets, data, datasets, outputs, or checkpoints.
   - Tests must not download anything.
   - Tests must not run full training.

5. Create `docs/sid_set_tiny_localization_smoke.md`
   - Explain the purpose of the SID-Set tiny localization smoke.
   - Explain that it adds 3-way classification and conditional tampered localization.
   - Explain labels:
     - real
     - full_synthetic
     - tampered
   - Explain label IDs:
     - `0` real
     - `1` full_synthetic
     - `2` tampered
   - Explain that masks are required only for tampered samples.
   - Explain that this is not full training and not a performance claim.
   - Explain that CF-Small provenance and SID-Set localization will later be combined into a pre-SNS integrated model.
   - Explain that SNS augmentation is not implemented here.
   - Include marker:
     `SID_SET_TINY_LOCALIZATION_SMOKE_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_sid_set_tiny_localization_smoke_config.py configs/training/sid_set_tiny_localization_smoke.example.json
```

```bash
python3 tests/test_sid_set_tiny_localization_smoke.py
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0019-sid-set-local-tiny-localization-smoke.md
```

```bash
test -f configs/training/sid_set_tiny_localization_smoke.example.json
```

```bash
test -f scripts/agent/validate_sid_set_tiny_localization_smoke_config.py
```

```bash
test -f scripts/training/run_sid_set_tiny_localization_smoke.py
```

```bash
test -f tests/test_sid_set_tiny_localization_smoke.py
```

```bash
test -f docs/sid_set_tiny_localization_smoke.md
```

```bash
grep -q SID_SET_TINY_LOCALIZATION_SMOKE_OK docs/sid_set_tiny_localization_smoke.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- configs/training/sid_set_tiny_localization_smoke.example.json scripts/agent/validate_sid_set_tiny_localization_smoke_config.py scripts/training/run_sid_set_tiny_localization_smoke.py tests/test_sid_set_tiny_localization_smoke.py docs/sid_set_tiny_localization_smoke.md
```

Optional validation command:

```bash
pytest -q tests/test_sid_set_tiny_localization_smoke.py
```

## Acceptance Criteria

- Validator passes for the tracked example config.
- Standalone tests pass.
- `check_agent_changes` passes.
- If pytest exists, pytest passes.
- Docs contain `SID_SET_TINY_LOCALIZATION_SMOKE_OK`.
- Changed files are limited to the allowed files.
- The implementation supports a future approved local config with explicit SID-Set image and mask paths.
- The future runner computes finite class, mask, and total losses.
- The future runner applies localization loss only to tampered samples.
- The future runner prints `SID_SET_TINY_LOCALIZATION_SMOKE_OK`.
- No dataset download, full training, network access, output writing, checkpoint writing, protected path access, SNS augmentation, or SNS perturbation evaluation occurs during implementation.

## Stop Condition

Stop after creating the task file. Report:

- task file path
- short summary
- git status
- exact git add command
- exact git commit command
