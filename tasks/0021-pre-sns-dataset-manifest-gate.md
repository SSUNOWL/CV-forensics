# Task 0021: Pre-SNS Dataset Manifest Gate

## Role

Codex is the implementation worker in Codex-only mode. Codex must implement only the files allowed by this task file after the task file is committed and after the user explicitly says to implement.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `docs/cf_small_multihead_tiny_smoke.md`
- `docs/sid_set_tiny_localization_smoke.md`
- `docs/pre_sns_integrated_smoke.md`
- `src/cv_forensics/local_data_gate.py`
- `tasks/0021-pre-sns-dataset-manifest-gate.md`

## Files Codex May Modify

- `src/cv_forensics/pre_sns_manifest.py`
- `configs/training/pre_sns_dataset_manifest.example.json`
- `scripts/agent/validate_pre_sns_dataset_manifest.py`
- `scripts/data/build_pre_sns_dataset_manifest.py`
- `tests/test_pre_sns_dataset_manifest.py`
- `docs/pre_sns_dataset_manifest.md`

If a parent directory for an allowed file does not exist, Codex may create that parent directory. Codex must not create or modify any other files.

## Forbidden Actions During Implementation

Codex must not:

- download datasets
- train models
- inspect real dataset directories recursively
- read real images outside temporary fixtures
- read real masks outside temporary fixtures
- access `.env`, `.env.*`, secrets, data, datasets, outputs, or checkpoints
- install packages
- access network resources from shell commands
- use `rg`
- use `rm -rf`
- create large files
- implement SNS augmentation
- run SNS perturbation evaluation
- run `git push`, `git pull`, or `git fetch`
- run `curl`, `wget`, `ssh`, `scp`, `rsync`, `pip`, `npm`, `yarn`, `pnpm`, `kaggle`, `huggingface-cli`, or `wandb`
- use danger-full-access
- bypass permissions

## Important Project Facts

- The project target outputs are class, mask/localization, family/provenance, and reason.
- CF-Small supplies binary/provenance preparation and metadata including `architecture`, `model_name`, and `subset`.
- SID-Set supplies 3-way classification and tampered localization preparation, including `label_id`, optional `split`, optional `img_id`, and tampered masks.
- Before real training, the repository needs one auditable manifest schema that represents which supervision each sample can provide.
- The manifest gate combines approved local CF-Small and SID-Set sample manifests without downloading data, training, scanning directories, or reading real images/masks during implementation.
- This task must not implement SNS augmentation or run SNS perturbation evaluation.

## Implementation Requirements

1. Create `src/cv_forensics/pre_sns_manifest.py`
   - Use only Python standard library.
   - Define a unified manifest schema and validation helpers.
   - Support `source_dataset` values:
     - `community_forensics_small`
     - `sid_set`
   - For CF-Small samples, preserve:
     - `sample_id`
     - `image_path`
     - `class_label`
     - `family_label`
     - `architecture`
     - `model_name`
     - `subset`
   - For SID-Set samples, preserve:
     - `sample_id`
     - `image_path`
     - `class_label`
     - `label_id`
     - `mask_path` for tampered
     - `split`
     - `img_id`
   - Add `tasks_available` fields:
     - `class`
     - `family`
     - `localization`
   - Add loss routing fields so training knows which losses apply.
   - Reject unsafe paths, URLs, protected paths, secret-like fields, and path traversal.

2. Create `configs/training/pre_sns_dataset_manifest.example.json`
   - The config must be safe and symbolic only.
   - It must not contain real paths, URLs, protected paths, secrets, outputs, or checkpoints.
   - It must include guardrails:
     - `no_download: true`
     - `no_network: true`
     - `no_training: true`
     - `no_outputs: true`
     - `no_checkpoints: true`
     - `no_sns_augmentation: true`
   - It must include class labels, family labels, and source dataset policies.

3. Create `scripts/agent/validate_pre_sns_dataset_manifest.py`
   - Use only Python standard library.
   - Validate `example_symbolic` and `approved_local_pre_sns_manifest`.
   - For approved local mode, require approval text exactly:
     `I_APPROVE_LOCAL_NON_SNS_PRE_SNS_DATASET_MANIFEST`
   - Validate explicit local paths under `approved_local_roots` only.
   - Do not read images or masks.
   - Do not scan directories.
   - Print `PRE_SNS_DATASET_MANIFEST_OK` on success.

4. Create `scripts/data/build_pre_sns_dataset_manifest.py`
   - Use only Python standard library.
   - Accept an approved local config.
   - Validate before building.
   - Combine explicitly listed CF-Small and SID-Set samples.
   - May write a manifest only to an explicitly approved path under `.local/manifests` or another approved local output root outside protected repo paths.
   - Must not write `outputs/` or `checkpoints/`.
   - Must not download or scan directories.
   - Print JSON summary with marker `PRE_SNS_DATASET_MANIFEST_OK`.

5. Create `tests/test_pre_sns_dataset_manifest.py`
   - Use a standard library test harness.
   - Do not import pytest.
   - Use temporary JSON fixtures only.
   - Test valid symbolic config, valid approved config, unsafe paths, missing approval, missing class coverage, bad family labels, tampered without mask, protected output path, and no file writes except approved temporary manifest output.

6. Create `docs/pre_sns_dataset_manifest.md`
   - Explain manifest purpose and how CF-Small and SID-Set supervision is combined.
   - Explain this is still pre-training and no real training occurs.
   - Include marker:
     `PRE_SNS_DATASET_MANIFEST_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_dataset_manifest.py configs/training/pre_sns_dataset_manifest.example.json
```

```bash
python3 tests/test_pre_sns_dataset_manifest.py
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0021-pre-sns-dataset-manifest-gate.md
```

```bash
test -f src/cv_forensics/pre_sns_manifest.py
```

```bash
test -f configs/training/pre_sns_dataset_manifest.example.json
```

```bash
test -f scripts/agent/validate_pre_sns_dataset_manifest.py
```

```bash
test -f scripts/data/build_pre_sns_dataset_manifest.py
```

```bash
test -f tests/test_pre_sns_dataset_manifest.py
```

```bash
test -f docs/pre_sns_dataset_manifest.md
```

```bash
grep -q PRE_SNS_DATASET_MANIFEST_OK docs/pre_sns_dataset_manifest.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- src/cv_forensics/pre_sns_manifest.py configs/training/pre_sns_dataset_manifest.example.json scripts/agent/validate_pre_sns_dataset_manifest.py scripts/data/build_pre_sns_dataset_manifest.py tests/test_pre_sns_dataset_manifest.py docs/pre_sns_dataset_manifest.md
```

Optional validation command:

```bash
pytest -q tests/test_pre_sns_dataset_manifest.py
```

## Acceptance Criteria

- Validator passes for example config.
- Standalone tests pass.
- `check_agent_changes` passes.
- Docs contain `PRE_SNS_DATASET_MANIFEST_OK`.
- Changed files are limited to the allowed files.
- Unified manifest schema supports class, family, and localization routing.
- No dataset download, training, network access, output/checkpoint writing, protected path access, or SNS augmentation occurs.

## Stop Condition

Stop after creating the task file. Report:

- task file path
- short summary
- git status
- exact git add command
- exact git commit command
