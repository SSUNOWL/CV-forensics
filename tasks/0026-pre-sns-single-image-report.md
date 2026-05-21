# Task 0026: Pre-SNS Single-Image Report

## Task title
Implement a pre-SNS single-image inference report runner.

## Role
Codex is the implementation worker in Codex-only mode. Codex must implement only the files allowed by this task file after this task file is committed and after the user explicitly says to implement.

Do not implement this task while creating this task file.

## Goal
Implement a pre-SNS single-image inference report runner.

## Purpose
The project now has a guarded pre-SNS pilot training loop and actual checkpoint artifact. This task adds a single-image inference/report path so a user can provide one local image and a trained checkpoint, then receive a JSON forensic report with:

- 3-way class prediction: `real / full_synthetic / tampered`
- generator-family provenance prediction
- conditional localization summary
- template-based evidence reason
- latency/FPS estimate
- safety metadata

## Project alignment
The proposal target output is not a simple yes/no detector. It should provide Class / Mask / Family / Reason together. The project uses a lightweight multi-head structure with shared visual backbone, 3-way classification head, generator-family provenance head, conditional localization head, evidence aggregation, and template-based explanation. SNS augmentation is not part of this task.

## Files Codex May Read
- AGENTS.md
- CLAUDE.md
- docs/project_brief.md
- docs/project_contract.md
- configs/project_contract.json
- docs/agent_workflow.md
- docs/model_output_schema.md
- docs/model_output_schema.md
- docs/pre_sns_training_artifacts.md
- docs/inference_stub.md
- tasks/0025-pre-sns-training-artifact-writer.md
- tasks/0026-pre-sns-single-image-report.md
- scripts/training/train_pre_sns_baseline.py
- scripts/agent/validate_pre_sns_baseline_train_config.py
- scripts/agent/validate_pre_sns_training_artifacts.py
- src/cv_forensics/pre_sns_integrated_model.py
- src/cv_forensics/pre_sns_training_artifacts.py
- src/cv_forensics/model_output_schema.py
- src/cv_forensics/explanation_templates.py
- configs/inference/fake_inputs.example.json
- configs/training/pre_sns_baseline_train.example.json
- .local/pre_sns_baseline_train.local.json
- .local/pre_sns_dataset_manifest.local.json

## Files Codex May Modify
- src/cv_forensics/pre_sns_inference_report.py
- src/cv_forensics/__init__.py
- scripts/inference/run_pre_sns_report.py
- scripts/agent/validate_pre_sns_inference_report.py
- tests/test_pre_sns_inference_report.py
- docs/pre_sns_inference_report.md
- configs/inference/pre_sns_report.example.json

If a parent directory for an allowed file does not exist, Codex may create that parent directory. Codex must not create or modify any other files.

## Forbidden actions
Codex must not:

- download datasets
- access network resources
- install packages
- train a model
- update checkpoints
- write to repository `outputs/` or `checkpoints/`
- access `.env`, `.env.*`, secrets, credentials, data, datasets, outputs, or checkpoints
- recursively scan directories
- use SNS augmentation
- run SNS perturbation evaluation
- run `git push`, `git pull`, or `git fetch`
- use `rg`
- use `rm -rf`
- create large files

## Allowed runtime behavior
- The inference runner may read exactly one explicit local `image_path` from config.
- The inference runner may read exactly one explicit `checkpoint_path` from config.
- The inference runner may write a small JSON report and optional mask summary file only under an explicitly approved `report_root` outside the repository.
- If `write_report=false`, the runner must print JSON only and write nothing.

## Implementation requirements

### 1. Inference report module
Create `src/cv_forensics/pre_sns_inference_report.py`.

It must provide helpers to:

- load and validate a report config
- validate explicit `image_path`
- validate explicit `checkpoint_path`
- validate approved `report_root` outside repository
- prepare an image tensor
- load the trained pre-SNS model checkpoint
- run inference on one image
- compute softmax confidence dictionaries
- compute `tampered_score`
- apply conditional localization threshold `tau`
- compute `mask_area_pct` from predicted mask probabilities
- create evidence-template reason
- measure `latency_ms` and `fps_estimate`
- produce a stable JSON-safe report

It must not recursively scan directories. It must reject URL-like paths, secret-like keys, protected paths, and repository `outputs/` or `checkpoints/` destinations. It must use standard library plus torch and PIL when needed, and it should degrade clearly if torch or PIL is missing.

### 2. Inference runner CLI
Create `scripts/inference/run_pre_sns_report.py`.

CLI:

```bash
python3 scripts/inference/run_pre_sns_report.py <config.json>
```

It must:

- validate the config
- run one-image inference
- print JSON report to stdout
- write report JSON only if `write_report=true`
- never download data
- never train
- never write checkpoints
- never use SNS augmentation

### 3. Example config
Create `configs/inference/pre_sns_report.example.json`.

The config must be symbolic and safe to commit. It must contain no real paths, URLs, or secrets.

It must contain expected fields:

- `schema_version`
- `config_kind`
- `execution_mode`
- `no_download`
- `no_network`
- `no_training`
- `no_checkpoint_writes`
- `no_sns_augmentation`
- `checkpoint_path`
- `image_path`
- `report_root`
- `write_report`
- `device`
- `cuda_device_index`
- `max_image_size`
- `threshold_tau`
- `class_labels`
- `family_labels`
- `required_approval_text`
- `user_approval_text`
- `result_scope`

### 4. Inference report validator
Create `scripts/agent/validate_pre_sns_inference_report.py`.

CLI:

```bash
python3 scripts/agent/validate_pre_sns_inference_report.py <config.json> [report-json]
```

It must:

- validate the example config safely
- if `report-json` is provided, parse JSON even if logs precede it
- verify marker `PRE_SNS_SINGLE_IMAGE_REPORT_OK`
- verify `class`, `class_conf`, `family`, `family_conf`, `tampered_score`, `localization_head`, `mask_area_pct`, `reason`, `latency_ms`, `fps_estimate`
- verify `no_download`, `no_network`, `no_training`, `no_sns_augmentation`
- verify `checkpoint_path` and `image_path` are explicit local paths in approved local mode
- verify report artifacts are outside the repository when `write_report=true`
- print `PRE_SNS_INFERENCE_REPORT_CONFIG_OK` for config-only validation
- print `PRE_SNS_SINGLE_IMAGE_REPORT_VALIDATED_OK` for report validation

### 5. Tests
Create `tests/test_pre_sns_inference_report.py`.

Tests must be runnable with:

```bash
python3 tests/test_pre_sns_inference_report.py
```

Tests must use temporary directories and tiny fake image/checkpoint artifacts only. Tests must not use real datasets.

Tests must cover:

- example config validates
- unsafe URL paths rejected
- protected paths rejected
- repo `outputs/` or `checkpoints/` `report_root` rejected
- report JSON with required fields accepted
- missing `class_conf` rejected
- missing `family_conf` rejected
- missing `reason` rejected
- `threshold_tau` activates localization when `tampered_score >= tau`
- `threshold_tau` skips localization when `tampered_score < tau`
- parser handles leading log text before JSON
- `write_report=false` writes nothing
- ordinary prose with words like authoritative is not rejected as a secret

### 6. Documentation
Create `docs/pre_sns_inference_report.md`.

It must include marker:

```text
PRE_SNS_SINGLE_IMAGE_REPORT_OK
```

It must explain:

- one-image pre-SNS inference report purpose
- Class / Mask / Family / Reason output
- conditional localization threshold `tau`
- report output fields
- artifact/write policy
- not SNS augmentation
- not final performance claim

## Validation commands

```bash
python3 scripts/agent/check_agent_changes.py tasks/0026-pre-sns-single-image-report.md
```

```bash
python3 scripts/agent/validate_pre_sns_inference_report.py configs/inference/pre_sns_report.example.json
```

```bash
python3 tests/test_pre_sns_inference_report.py
```

```bash
grep -q PRE_SNS_SINGLE_IMAGE_REPORT_OK docs/pre_sns_inference_report.md
```

```bash
git status --short --untracked-files=all
```

## Acceptance criteria
- Changed files are limited to the allowed files.
- Example config validates.
- Standalone tests pass.
- Documentation marker exists.
- Single-image report schema supports `class`, `class_conf`, `family`, `family_conf`, `tampered_score`, `localization_head`, `mask_area_pct`, `reason`, `latency_ms`, and `fps_estimate`.
- No dataset download, network access, training, checkpoint writing, SNS augmentation, protected path access, or unrelated modification occurs.

## Stop condition
After creating `tasks/0026-pre-sns-single-image-report.md`, stop and show:

- task file path
- short summary
- git status
- exact `git add` command
- exact `git commit` command

Do not implement yet. Do not run inference yet. Do not commit.
