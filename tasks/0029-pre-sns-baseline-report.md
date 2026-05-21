# Task 0029: Pre-SNS Baseline Report

## Task title
Implement a pre-SNS baseline report generator.

## Role
Codex is the implementation worker in Codex-only mode. Do not delegate to Claude Code for this task.

Do not implement this task while creating this task file. Stop after creating this task file.

## Goal
Implement a pre-SNS baseline report generator.

## Purpose
Collect the results from:

- task 0025 actual pilot training artifact
- task 0026 single-image inference report
- task 0027 pre-SNS evaluation
- task 0028 scaled training run

Then generate a concise Markdown and JSON summary of the pre-SNS baseline before SNS augmentation begins.

## Project alignment
The proposal requires pre-SNS model outputs and metrics before evaluating social-media perturbation robustness. The report must clearly state that SNS augmentation has not been applied yet.

## Files Codex May Read
- AGENTS.md
- CLAUDE.md
- docs/project_brief.md
- docs/project_contract.md
- configs/project_contract.json
- docs/agent_workflow.md
- docs/pre_sns_training_artifacts.md
- docs/pre_sns_inference_report.md
- docs/pre_sns_evaluation.md
- docs/pre_sns_scaled_training.md
- tasks/0025-pre-sns-training-artifact-writer.md
- tasks/0026-pre-sns-single-image-report.md
- tasks/0027-pre-sns-evaluation-runner.md
- tasks/0028-pre-sns-scaled-training-run.md
- tasks/0029-pre-sns-baseline-report.md
- .local/run_logs/*.json, if explicitly selected by the user
- ~/cvf_runs/*/artifact_manifest.json, if explicitly selected by config
- ~/cvf_runs/*/metrics_summary.json, if explicitly selected by config
- ~/cvf_runs/*/run_summary.json, if explicitly selected by config

## Files Codex May Modify
- src/cv_forensics/pre_sns_baseline_report.py
- src/cv_forensics/__init__.py
- scripts/reporting/generate_pre_sns_baseline_report.py
- scripts/agent/validate_pre_sns_baseline_report.py
- tests/test_pre_sns_baseline_report.py
- docs/pre_sns_baseline_report.md
- configs/reporting/pre_sns_baseline_report.example.json

If a parent directory for an allowed file does not exist, Codex may create it. Codex must not create or modify any other files.

## Forbidden actions
Codex must not:

- download datasets
- access network resources
- install packages
- train
- run inference
- run evaluation
- write checkpoints
- use SNS augmentation
- run SNS perturbation evaluation
- access `.env`, `.env.*`, secrets, credentials, data, datasets, outputs, or checkpoints
- recursively scan directories
- write generated report into repository `outputs/`
- run `git push`, `git pull`, or `git fetch`
- use `rg`
- use `rm -rf`
- create large files

## Allowed runtime behavior
- The report generator may read only explicit result JSON paths listed in config.
- The report generator may write a small Markdown report and JSON summary under `approved_report_root` outside the repository.
- It may also update tracked `docs/pre_sns_baseline_report.md` with a template and marker, but not with local private paths unless sanitized.

## Implementation requirements

### 1. Baseline report module
Create `src/cv_forensics/pre_sns_baseline_report.py`.

It must:

- validate report config
- load explicit JSON result paths:
  - `training_result_path`
  - `inference_report_path`
  - `evaluation_result_path`
  - `scaled_training_result_path`
- parse JSON even when leading logs exist
- extract key metrics:
  - training samples/steps/epochs/losses
  - checkpoint path sanitized to basename or outside-repo root category
  - inference class/family/localization/reason example
  - evaluation `accuracy_3way`, `macro_f1_3way`, `family_accuracy`, `mask_iou_mean`, `localization_activation_recall`, `latency_ms_mean`, `fps_estimate`
  - scaled training samples/steps/epochs/losses
- produce Markdown report
- produce JSON summary
- include `PRE_SNS_BASELINE_REPORT_OK` marker
- clearly state no SNS augmentation has been applied

### 2. Baseline report CLI
Create `scripts/reporting/generate_pre_sns_baseline_report.py`.

CLI:

```bash
python3 scripts/reporting/generate_pre_sns_baseline_report.py <config.json>
```

It must:

- validate config
- read explicit paths only
- generate report artifacts outside repo
- print JSON summary to stdout
- not train, infer, evaluate, download, or use SNS augmentation

### 3. Example config
Create `configs/reporting/pre_sns_baseline_report.example.json`.

It must be a symbolic safe config:

- no real local paths
- no URLs
- no secrets

It must contain fields:

- `schema_version`
- `config_kind`
- `execution_mode`
- `required_approval_text`
- `user_approval_text`
- `training_result_path`
- `inference_report_path`
- `evaluation_result_path`
- `scaled_training_result_path`
- `approved_report_root`
- `write_report`
- `no_download`
- `no_network`
- `no_training`
- `no_inference`
- `no_evaluation`
- `no_checkpoint_writes`
- `no_sns_augmentation`
- `result_scope`

### 4. Baseline report validator
Create `scripts/agent/validate_pre_sns_baseline_report.py`.

CLI:

```bash
python3 scripts/agent/validate_pre_sns_baseline_report.py <config.json> [report-result-json]
```

It must:

- validate example config
- validate generated report result
- verify marker `PRE_SNS_BASELINE_REPORT_OK`
- verify `report_markdown_path` and `report_summary_path` exist when `write_report=true`
- verify required sections:
  - Training summary
  - Single-image report summary
  - Evaluation metrics
  - Scaled training summary
  - Pre-SNS limitations
  - Next SNS augmentation step
- verify `no_sns_augmentation` true
- verify report root outside repo
- print `PRE_SNS_BASELINE_REPORT_CONFIG_OK` for config-only
- print `PRE_SNS_BASELINE_REPORT_RESULT_OK` for result validation

### 5. Tests
Create `tests/test_pre_sns_baseline_report.py`.

Tests must be runnable with:

```bash
python3 tests/test_pre_sns_baseline_report.py
```

Tests must use temporary fake JSON result files only.

Tests must cover:

- example config validates
- result parser handles leading logs
- Markdown contains required sections
- summary JSON contains key metrics
- missing evaluation metrics rejected
- missing inference report rejected
- report root inside repo `outputs/` or `checkpoints/` rejected
- `no_sns_augmentation=false` rejected
- valid report artifacts accepted

### 6. Documentation
Create `docs/pre_sns_baseline_report.md`.

It must include marker:

```text
PRE_SNS_BASELINE_REPORT_OK
```

It must explain:

- what the pre-SNS baseline report contains
- why it is created before SNS augmentation
- expected inputs
- generated artifacts
- no-SNS status
- not a final full-dataset claim unless scaled/full evaluation is separately completed

## Validation commands

```bash
python3 scripts/agent/check_agent_changes.py tasks/0029-pre-sns-baseline-report.md
```

```bash
python3 scripts/agent/validate_pre_sns_baseline_report.py configs/reporting/pre_sns_baseline_report.example.json
```

```bash
python3 tests/test_pre_sns_baseline_report.py
```

```bash
grep -q PRE_SNS_BASELINE_REPORT_OK docs/pre_sns_baseline_report.md
```

```bash
git status --short --untracked-files=all
```

## Acceptance criteria
- Report generator exists.
- Validator passes.
- Tests pass.
- Documentation marker exists.
- Report reads explicit result paths only.
- Report clearly states no SNS augmentation yet.
- Changed files are limited to allowed files.

## Stop condition
After creating `tasks/0029-pre-sns-baseline-report.md`, stop and show:

- task file path
- summary
- git status
- git add command
- git commit command
