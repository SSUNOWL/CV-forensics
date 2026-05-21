# Task 0027: Pre-SNS Evaluation Runner

## Task title
Implement a pre-SNS evaluation runner for the trained pilot checkpoint and approved local unified manifest.

## Role
Codex is the implementation worker in Codex-only mode. Do not delegate to Claude Code for this task.

Do not implement this task while creating this task file. Stop after creating this task file.

## Goal
Implement a pre-SNS evaluation runner for the trained pilot checkpoint and approved local unified manifest.

## Purpose
After task 0026 adds single-image inference, this task evaluates a trained pre-SNS checkpoint on an approved local manifest subset and produces metrics aligned with the proposal:

- 3-way classification accuracy
- macro-F1
- generator-family accuracy
- tampered mask IoU
- localization activation recall
- latency_ms_mean
- fps_estimate
- sample counts by dataset and class
- no SNS augmentation

## Project alignment
The proposal evaluates 3-way classification Accuracy/Macro-F1, tampered mask IoU, generator-family accuracy, localization activation recall, and RTX 4090 inference time/FPS. SNS perturbation robustness comes later and must not be implemented in this task.

## Files Codex May Read
- AGENTS.md
- CLAUDE.md
- docs/project_brief.md
- docs/project_contract.md
- configs/project_contract.json
- docs/agent_workflow.md
- docs/pre_sns_inference_report.md
- docs/pre_sns_training_artifacts.md
- tasks/0026-pre-sns-single-image-report.md
- tasks/0027-pre-sns-evaluation-runner.md
- src/cv_forensics/pre_sns_inference_report.py
- src/cv_forensics/pre_sns_integrated_model.py
- src/cv_forensics/pre_sns_manifest.py
- src/cv_forensics/metrics.py
- scripts/inference/run_pre_sns_report.py
- scripts/agent/validate_pre_sns_inference_report.py
- .local/pre_sns_dataset_manifest.local.json, if it exists
- .local/pre_sns_baseline_train.local.json, if it exists

## Files Codex May Modify
- src/cv_forensics/pre_sns_evaluation.py
- src/cv_forensics/__init__.py
- scripts/evaluation/evaluate_pre_sns_baseline.py
- scripts/agent/validate_pre_sns_evaluation.py
- tests/test_pre_sns_evaluation.py
- docs/pre_sns_evaluation.md
- configs/evaluation/pre_sns_evaluation.example.json

If a parent directory for an allowed file does not exist, Codex may create it. Codex must not create or modify any other files.

## Forbidden actions
Codex must not:

- download datasets
- access network resources
- install packages
- train a model
- write checkpoints
- use SNS augmentation
- run SNS perturbation evaluation
- access `.env`, `.env.*`, secrets, credentials, data, datasets, outputs, or checkpoints
- recursively scan directories
- write to repository `outputs/` or `checkpoints/`
- run `git push`, `git pull`, or `git fetch`
- use `rg`
- use `rm -rf`
- create large files

## Allowed runtime behavior
- The evaluation runner may read explicit `image_path` and `mask_path` values listed in the approved local manifest.
- The evaluation runner may read one explicit `checkpoint_path` from config.
- The evaluation runner may write a small evaluation JSON artifact under `approved_eval_root` outside the repository.
- It must never recursively scan directories.

## Implementation requirements

### 1. Evaluation module
Create `src/cv_forensics/pre_sns_evaluation.py`.

It must provide helpers to:

- validate evaluation config
- load approved local manifest
- select up to `max_samples` samples without recursion
- load checkpoint
- run inference using existing pre-SNS inference code
- compute 3-way accuracy
- compute macro-F1 without sklearn
- compute family accuracy for samples with family labels
- compute tampered mask IoU for tampered samples with masks
- compute localization activation recall for tampered samples
- compute `latency_ms_mean` and `fps_estimate`
- produce JSON-safe metrics summary

### 2. Evaluation runner CLI
Create `scripts/evaluation/evaluate_pre_sns_baseline.py`.

CLI:

```bash
python3 scripts/evaluation/evaluate_pre_sns_baseline.py <config.json>
```

It must:

- validate config
- evaluate explicit manifest samples
- print JSON result
- write `evaluation_summary.json` only if `write_eval_artifact=true`
- use GPU if `device=cuda` and available
- fail clearly if CUDA requested but unavailable
- not train
- not download
- not use SNS augmentation

### 3. Example config
Create `configs/evaluation/pre_sns_evaluation.example.json`.

The config must be a safe symbolic example:

- no real paths
- no URLs
- no secrets

It must contain fields:

- `schema_version`
- `config_kind`
- `execution_mode`
- `required_approval_text`
- `user_approval_text`
- `manifest_path`
- `checkpoint_path`
- `approved_eval_root`
- `write_eval_artifact`
- `device`
- `cuda_device_index`
- `max_samples`
- `batch_size`
- `max_image_size`
- `threshold_tau`
- `no_download`
- `no_network`
- `no_training`
- `no_checkpoint_writes`
- `no_sns_augmentation`
- `result_scope`

### 4. Evaluation validator
Create `scripts/agent/validate_pre_sns_evaluation.py`.

CLI:

```bash
python3 scripts/agent/validate_pre_sns_evaluation.py <config.json> [evaluation-result-json]
```

It must:

- validate example config
- parse result JSON even with leading logs
- verify marker `PRE_SNS_EVALUATION_OK`
- verify metric fields:
  - `accuracy_3way`
  - `macro_f1_3way`
  - `family_accuracy`
  - `mask_iou_mean`
  - `localization_activation_recall`
  - `latency_ms_mean`
  - `fps_estimate`
  - `samples_evaluated`
- verify `no_download`, `no_network`, `no_training`, `no_sns_augmentation`
- verify artifact paths are outside repo when `write_eval_artifact=true`
- print `PRE_SNS_EVALUATION_CONFIG_OK` for config-only validation
- print `PRE_SNS_EVALUATION_RESULT_OK` for result validation

### 5. Tests
Create `tests/test_pre_sns_evaluation.py`.

Tests must be runnable with:

```bash
python3 tests/test_pre_sns_evaluation.py
```

Tests must use tiny synthetic arrays and temporary files only. Tests must not use real datasets.

Tests must cover:

- macro-F1 calculation
- accuracy calculation
- family accuracy with missing family labels ignored
- mask IoU calculation
- localization activation recall
- JSON result parser with leading logs
- example config validation
- unsafe paths rejected
- eval root inside repo `outputs/` or `checkpoints/` rejected
- result validator rejects missing metrics
- result validator accepts a valid result

### 6. Documentation
Create `docs/pre_sns_evaluation.md`.

It must include marker:

```text
PRE_SNS_EVALUATION_OK
```

It must explain:

- proposal-aligned metrics
- no-SNS scope
- artifact policy
- relation to pre-SNS baseline

## Validation commands

```bash
python3 scripts/agent/check_agent_changes.py tasks/0027-pre-sns-evaluation-runner.md
```

```bash
python3 scripts/agent/validate_pre_sns_evaluation.py configs/evaluation/pre_sns_evaluation.example.json
```

```bash
python3 tests/test_pre_sns_evaluation.py
```

```bash
grep -q PRE_SNS_EVALUATION_OK docs/pre_sns_evaluation.md
```

```bash
git status --short --untracked-files=all
```

## Acceptance criteria
- Changed files are limited to allowed files.
- Config validator passes.
- Standalone tests pass.
- Documentation marker exists.
- Evaluation runner can compute proposal-aligned pre-SNS metrics.
- No dataset download, network access, training, checkpoint writing, SNS augmentation, protected path access, or unrelated modification occurs.

## Stop condition
After creating `tasks/0027-pre-sns-evaluation-runner.md`, stop and show:

- task file path
- summary
- git status
- exact git add command
- exact git commit command
