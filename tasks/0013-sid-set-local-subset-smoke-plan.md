# Task 0013: SID-Set Local Subset Smoke Plan

## Task Title

Create a SID-Set local subset smoke plan.

## Role

Codex is the implementation worker in Codex-only mode. Codex must implement only the files allowed by this task file after the task file is committed and after the user explicitly says to implement.

Codex must not implement this task while creating this task file.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `docs/local_data_readiness.md`
- `docs/cf_small_subset_smoke.md`
- `configs/local_data/readiness.example.json`
- `configs/local_data/cf_small_subset_smoke.example.json`
- `src/cv_forensics/local_data_gate.py`
- `tasks/0013-sid-set-local-subset-smoke-plan.md`
- `configs/manifests/sid_set.example.json`, only if it exists
- `configs/manifests/combined_smoke_manifest.example.json`, only if it exists
- `tests/test_local_data_gate.py`, only if useful
- `tests/test_cf_small_subset_smoke.py`, only if useful

## Files Codex May Modify

- `configs/local_data/sid_set_subset_smoke.example.json`
- `scripts/agent/validate_sid_set_subset_smoke.py`
- `tests/test_sid_set_subset_smoke.py`
- `docs/sid_set_subset_smoke.md`

If a parent directory for an allowed file does not exist, Codex may create that parent directory. Codex must not create or modify any other files.

## Files Claude May Modify

This Codex-only task preserves this section name so `scripts/agent/check_agent_changes.py` can enforce the same allowed modification list.

- `configs/local_data/sid_set_subset_smoke.example.json`
- `scripts/agent/validate_sid_set_subset_smoke.py`
- `tests/test_sid_set_subset_smoke.py`
- `docs/sid_set_subset_smoke.md`

## Forbidden Actions

Codex must not:

- download datasets
- download SID-Set
- train a model
- run real optimization
- inspect actual dataset directories recursively
- read real images
- read real masks
- validate actual image or mask file existence
- access `.env`, `.env.*`, `secrets`, `data`, `datasets`, `outputs`, or `checkpoints`
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
- import `torch`, `torchvision`, `PIL`, `cv2`, `numpy`, `pandas`, `sklearn`, or `pytest`
- create large files
- implement SNS augmentation
- use danger-full-access
- bypass permissions

## Important Project Facts

- The final project target outputs are class, mask/localization, family/provenance, and reason.
- SID-Set is intended for 3-way image classification and tampered localization planning.
- Class labels are `real`, `synthetic`, and `tampered`.
- Localization is relevant mainly for tampered samples.
- Synthetic samples may not have tampered masks.
- Real samples should not have tampered masks.
- Family/provenance labels may be missing or coarse for SID-Set, so the plan must define a policy for missing family/provenance labels.
- Conditional localization activates when `tampered_score >= threshold_tau`.
- Localization activation recall is important because high `threshold_tau` can cause false negatives.
- Task 0011 provides the local data readiness gate.
- Task 0012 provides the Community Forensics-Small local subset smoke plan and a separate precedent for symbolic local-path-gated planning.
- This task must not implement SNS augmentation.
- SNS augmentation begins only after pre-SNS baseline evaluation is committed.

## Implementation Requirements

### 1. Create `configs/local_data/sid_set_subset_smoke.example.json`

- The config must be symbolic and dry-run safe.
- It must not contain real local dataset paths.
- It must not contain URLs.
- It must not contain secrets.
- It must not contain `data/`, `datasets/`, `outputs/`, `checkpoints/`, `.env`, or secret-like values.
- It must include:
  - `schema_version`
  - `dataset_name: SID-Set`
  - `dry_run: true`
  - `no_download: true`
  - `no_training: true`
  - `no_network: true`
  - `no_outputs: true`
  - `no_checkpoints: true`
  - `no_real_image_reading: true`
  - `no_real_mask_reading: true`
  - `local_data_gate_ref`
  - `readiness_policy`
  - `subset_policy`
  - `split_policy`
  - `class_policy`
  - `mask_policy`
  - `family_policy`
  - `localization_policy`
  - `threshold_tau`
  - `expected_task_outputs`
  - `validation_notes`
- The subset policy should be tiny and symbolic, for example planned class-balanced samples for `real`, `synthetic`, and `tampered`.
- The class policy must represent 3-way classification:
  - `real`
  - `synthetic`
  - `tampered`
- The mask policy must represent:
  - tampered samples require mask/localization planning
  - real samples must not require tampered masks
  - synthetic samples may not require tampered masks unless explicitly annotated
  - no real mask files are read in this task
- The family policy must represent:
  - missing family/provenance labels are allowed in SID-Set planning
  - missing family labels should map to `Unknown` or `Real-or-N/A` depending on class
  - provenance/family head training remains gated until labels are confirmed
- The localization policy must represent:
  - conditional localization activation
  - `threshold_tau`
  - localization activation recall
  - mask IoU planned metric
- The config may use symbolic refs to existing safe config or manifest files, but must not use real dataset paths.

### 2. Create `scripts/agent/validate_sid_set_subset_smoke.py`

- Use only Python standard library.
- Accept exactly one positional argument: SID-Set subset smoke config JSON path.
- Add repo `src` path to `sys.path` if needed.
- Reuse `src/cv_forensics/local_data_gate.py` safety checks where appropriate.
- Validate that the config is symbolic and dry-run safe.
- Validate required guardrail flags:
  - `dry_run` is true
  - `no_download` is true
  - `no_training` is true
  - `no_network` is true
  - `no_outputs` is true
  - `no_checkpoints` is true
  - `no_real_image_reading` is true
  - `no_real_mask_reading` is true
- Validate class policy includes exactly the required 3-way labels:
  - `real`
  - `synthetic`
  - `tampered`
- Validate mask policy represents tampered localization without reading masks.
- Validate family/provenance missing-label policy is explicit.
- Validate localization policy includes:
  - `threshold_tau`
  - conditional activation
  - mask IoU
  - localization activation recall
- Validate `threshold_tau` is between `0.0` and `1.0`.
- Validate the config recursively rejects protected paths, URLs, absolute paths, Windows drive paths, and secret-like keys/values.
- Print actionable errors and exit non-zero on fail.
- Print a concise success message containing `SID_SET_SUBSET_SMOKE_OK` on pass.
- Do not write files.
- Do not read images or masks.
- Do not inspect real dataset directories.

### 3. Create `tests/test_sid_set_subset_smoke.py`

- Use only Python standard library.
- Do not import `pytest`.
- It must be runnable with:

```bash
python3 tests/test_sid_set_subset_smoke.py
```

- It may also be pytest-compatible if pytest is installed.
- Tests should cover:
  - valid example config passes
  - `dry_run: false` is rejected
  - `no_download: false` is rejected
  - `no_training: false` is rejected
  - `no_outputs: false` is rejected
  - `no_checkpoints: false` is rejected
  - `no_real_image_reading: false` is rejected
  - `no_real_mask_reading: false` is rejected
  - missing `real`, `synthetic`, or `tampered` class label is rejected
  - invalid extra class label is rejected
  - missing mask policy is rejected
  - missing family/provenance policy is rejected
  - missing localization activation recall plan is rejected
  - missing mask IoU plan is rejected
  - `threshold_tau` outside `0.0` to `1.0` is rejected
  - protected paths are rejected
  - URLs are rejected
  - absolute paths are rejected
  - Windows drive paths are rejected
  - secret-like keys and values are rejected
  - ordinary prose containing words like `authoritative` or `authentication` is not rejected merely because it contains `auth`
  - validator does not write files
- Tests must not access `data`, `datasets`, `outputs`, `checkpoints`, `secrets`, or `.env`.
- Tests must not read real images or masks.

### 4. Create `docs/sid_set_subset_smoke.md`

- Explain the purpose of the SID-Set local subset smoke plan.
- Explain that this is not dataset download, not training, and not real image/mask reading.
- Explain how SID-Set maps to project goals:
  - 3-way classification
  - tampered localization
  - mask IoU
  - localization activation recall
  - family/provenance policy when labels are missing or coarse
- Explain how it depends on the local data readiness gate from task 0011.
- Explain how it differs from the Community Forensics-Small subset smoke plan from task 0012.
- Explain what must happen before real SID-Set smoke execution:
  - explicit user approval
  - validated local path policy
  - validated manifest
  - output/checkpoint policy
  - tiny subset size confirmation
- State that SNS augmentation is not implemented here.
- Include marker string:
  - `SID_SET_SUBSET_SMOKE_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_sid_set_subset_smoke.py configs/local_data/sid_set_subset_smoke.example.json
```

```bash
python3 tests/test_sid_set_subset_smoke.py
```

```bash
test -f configs/local_data/sid_set_subset_smoke.example.json
```

```bash
test -f scripts/agent/validate_sid_set_subset_smoke.py
```

```bash
test -f tests/test_sid_set_subset_smoke.py
```

```bash
test -f docs/sid_set_subset_smoke.md
```

```bash
grep -q SID_SET_SUBSET_SMOKE_OK docs/sid_set_subset_smoke.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0013-sid-set-local-subset-smoke-plan.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- configs/local_data/sid_set_subset_smoke.example.json scripts/agent/validate_sid_set_subset_smoke.py tests/test_sid_set_subset_smoke.py docs/sid_set_subset_smoke.md
```

Optional validation command:

```bash
pytest -q tests/test_sid_set_subset_smoke.py
```

## Acceptance Criteria

- `python3 scripts/agent/validate_sid_set_subset_smoke.py configs/local_data/sid_set_subset_smoke.example.json` passes.
- `python3 tests/test_sid_set_subset_smoke.py` passes.
- If pytest exists, `pytest -q tests/test_sid_set_subset_smoke.py` passes.
- `docs/sid_set_subset_smoke.md` contains `SID_SET_SUBSET_SMOKE_OK`.
- `python3 scripts/agent/check_agent_changes.py tasks/0013-sid-set-local-subset-smoke-plan.md` passes.
- Changed files are limited to:
  - `configs/local_data/sid_set_subset_smoke.example.json`
  - `scripts/agent/validate_sid_set_subset_smoke.py`
  - `tests/test_sid_set_subset_smoke.py`
  - `docs/sid_set_subset_smoke.md`
- The config is symbolic and dry-run safe.
- SID-Set 3-way classification requirements are represented.
- Tampered localization and mask policy are represented.
- Family/provenance missing-label policy is represented.
- Conditional localization threshold and activation recall planning are represented.
- No dataset download, model training, package installation, network access, real image reading, real mask reading, output writing, checkpoint writing, SNS augmentation, or protected path access occurs.

## Stop Condition

Stop after implementation and validation. Report:

- files changed
- validation commands run and results
- whether optional pytest validation was run or skipped
- confirmation that changed files are limited to the allowed list
- confirmation that no dataset download, model training, package installation, network access, real image reading, real mask reading, output writing, checkpoint writing, SNS augmentation, or protected path access occurred
