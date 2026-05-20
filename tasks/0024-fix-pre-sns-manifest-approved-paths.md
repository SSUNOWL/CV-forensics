# Task 0024 Fix: Pre-SNS Manifest Approved Local Paths

## Task title
Fix approved local `image_path` and `mask_path` validation in the pre-SNS unified manifest validator.

## Role
Codex is the implementation worker in Codex-only mode. Codex must implement only the files allowed by this task file after this task file is committed and after the user explicitly says to implement.

## Files Codex May Read
- AGENTS.md
- CLAUDE.md
- docs/project_brief.md
- docs/project_contract.md
- configs/project_contract.json
- docs/agent_workflow.md
- tasks/0024-fix-pre-sns-manifest-approved-paths.md
- tasks/0021-pre-sns-dataset-manifest-gate.md
- scripts/agent/validate_pre_sns_dataset_manifest.py
- tests/test_pre_sns_dataset_manifest.py
- docs/pre_sns_dataset_manifest.md
- src/cv_forensics/pre_sns_manifest.py
- configs/training/pre_sns_dataset_manifest.example.json

## Files Codex May Modify
- scripts/agent/validate_pre_sns_dataset_manifest.py
- tests/test_pre_sns_dataset_manifest.py
- docs/pre_sns_dataset_manifest.md

Codex must not create or modify any other files.

## Forbidden actions
Codex must not:
- download datasets
- train models
- install packages
- access .env, .env.*, secrets, data, datasets, outputs, or checkpoints
- modify unrelated files
- run git push, git pull, or git fetch
- create large files
- use rg
- use rm -rf
- access network resources from shell commands
- implement SNS augmentation
- run SNS perturbation evaluation
- use danger-full-access
- bypass permissions

## Important project facts
- The pre-SNS unified manifest combines approved local CF-Small and SID-Set sample manifests before training.
- Approved local manifest entries must use explicit listed local file paths.
- In approved local mode, `image_path` and tampered `mask_path` are expected to be absolute local file paths under `approved_local_roots`.
- These paths must still be rejected if they are outside `approved_local_roots`, URL or remote scheme references, Windows drive paths, path traversal, protected repository paths, protected directory references, or non-files.
- The current failure is repeated validation errors such as:
  - `sample_manifest[...].image_path: absolute path rejected`
  - `sample_manifest[...].mask_path: absolute path rejected`
- The validator already has explicit local file validation logic for `image_path` and `mask_path`.
- Generic recursive safety scanning must not reject approved local sample file paths solely because they are absolute paths.

## Implementation requirements

1. Update `scripts/agent/validate_pre_sns_dataset_manifest.py`.
   - In approved local mode, exclude sample manifest `image_path` and `mask_path` values from generic unsafe absolute-path scanning.
   - Continue to validate each `image_path` and `mask_path` through the explicit local file path checker.
   - Require `image_path` and `mask_path` values to be under `approved_local_roots`.
   - Require `image_path` and `mask_path` values to exist as files.
   - Reject directory-only paths.
   - Reject paths outside `approved_local_roots`.
   - Reject URLs and remote schemes.
   - Reject Windows drive paths.
   - Reject path traversal.
   - Reject protected repo paths and protected path segments including `data`, `datasets`, `outputs`, `checkpoints`, `.env`, and `secrets`.
   - Do not weaken protected-path checks for any other config field.

2. Update `tests/test_pre_sns_dataset_manifest.py`.
   - Use only Python standard library.
   - Do not import pytest.
   - Use temporary files outside protected repo paths.
   - Add tests proving:
     - absolute `image_path` under `approved_local_roots` passes
     - absolute tampered `mask_path` under `approved_local_roots` passes
     - `image_path` outside `approved_local_roots` fails
     - `mask_path` outside `approved_local_roots` fails
     - URL paths fail
     - protected paths fail
     - non-tampered samples with `mask_path` fail
     - tampered samples without `mask_path` fail
   - Keep existing symbolic example validation passing.
   - Tests must not access real datasets, .env, secrets, data, datasets, outputs, or checkpoints.

3. Update `docs/pre_sns_dataset_manifest.md`.
   - Document that approved local mode accepts explicit absolute sample file paths only when they are under `approved_local_roots`.
   - Document that the validator checks file existence for approved sample `image_path` and tampered `mask_path`.
   - Document that generic safety checks still apply to all other fields.
   - Preserve marker `PRE_SNS_DATASET_MANIFEST_OK`.

## Validation commands

```bash
python3 scripts/agent/check_agent_changes.py tasks/0024-fix-pre-sns-manifest-approved-paths.md
```

```bash
python3 scripts/agent/validate_pre_sns_dataset_manifest.py configs/training/pre_sns_dataset_manifest.example.json
```

```bash
python3 scripts/agent/validate_pre_sns_dataset_manifest.py .local/pre_sns_dataset_manifest.local.json
```

```bash
python3 tests/test_pre_sns_dataset_manifest.py
```

```bash
git status --short --untracked-files=all
```

## Acceptance criteria
- The existing symbolic example still passes.
- The approved local manifest with absolute `image_path` and `mask_path` under `approved_local_roots` passes.
- Unsafe absolute paths are rejected.
- URLs are rejected.
- Protected paths are rejected.
- Paths outside `approved_local_roots` are rejected.
- Non-file sample paths are rejected.
- Non-tampered samples with `mask_path` are rejected.
- Tampered samples without `mask_path` are rejected.
- Changed files are limited to:
  - scripts/agent/validate_pre_sns_dataset_manifest.py
  - tests/test_pre_sns_dataset_manifest.py
  - docs/pre_sns_dataset_manifest.md
- No datasets, training, installs, network, outputs, checkpoints, or secrets are touched.

## Stop condition
Stop after implementing the allowed files, running safe validation commands, and reviewing the result in Korean as PASS or NEEDS_FIX. Do not commit automatically.
