# Task 0033: Pre-SNS Meaningful Training V2

## Task title
Implement a meaningful guarded pre-SNS training and evaluation loop.

## Role
Codex is the supervisor and reviewer. Claude Code is the implementation worker.

Codex must not implement this task directly. Claude Code must implement only the files and requirements in this task file after the user commits this task file and says `delegate`.

## Goal
Upgrade the pre-SNS baseline from a smoke/toy training loop into a meaningful guarded training/evaluation loop that can support proposal-style pre-SNS claims before SNS augmentation.

## Context
The current pre-SNS pipeline can create class, family, and localization reports, but a tau sweep on a balanced 180-sample validation subset produced weak metrics:

- `class_accuracy ~= 0.522`
- `macro_f1 ~= 0.505`
- real and tampered `tampered_score` distributions overlap heavily
- `tau=0.5` almost never activates localization
- `tau=0.4` improves recall but creates too many false activations

This task must make pre-SNS-only training/evaluation robust enough to judge whether the pre-SNS baseline is meaningful. It must not add SNS augmentation.

## Project scope
This is still pre-SNS only.

The model/task surface remains:

- class head: `real / full_synthetic / tampered`
- family head: `LatDiff / PixDiff / GAN / Other / Real-or-N/A`
- localization head for SID-Set tampered samples with masks
- family loss applied only when family labels are meaningful
- localization loss applied only to tampered samples with masks

## Files Claude May Read
- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0033-pre-sns-meaningful-training-v2.md`
- `docs/pre_sns_baseline_training.md`
- `docs/pre_sns_training_artifacts.md`
- `docs/pre_sns_evaluation.md`
- `docs/pre_sns_scaled_training.md`
- `docs/pre_sns_dataset_manifest.md`
- `configs/training/pre_sns_baseline_train.example.json`
- `configs/training/pre_sns_scaled_train.example.json`
- `configs/evaluation/pre_sns_evaluation.example.json`
- `scripts/training/train_pre_sns_baseline.py`
- `scripts/training/prepare_pre_sns_scaled_train_config.py`
- `scripts/evaluation/evaluate_pre_sns_baseline.py`
- `scripts/agent/validate_pre_sns_baseline_train_config.py`
- `scripts/agent/validate_pre_sns_training_artifacts.py`
- `scripts/agent/validate_pre_sns_evaluation.py`
- `scripts/agent/validate_pre_sns_dataset_manifest.py`
- `src/cv_forensics/pre_sns_integrated_model.py`
- `src/cv_forensics/pre_sns_manifest.py`
- `src/cv_forensics/pre_sns_preflight.py`
- `src/cv_forensics/pre_sns_training_artifacts.py`
- `src/cv_forensics/pre_sns_evaluation.py`
- `src/cv_forensics/pre_sns_inference_report.py`
- `src/cv_forensics/metrics.py`
- `src/cv_forensics/model_output_schema.py`
- `src/cv_forensics/__init__.py`
- Existing tests under `tests/` that are directly related to pre-SNS training, artifacts, evaluation, metrics, and manifest validation

Claude may read explicit local manifest files only when supplied through a config and only for validation/runtime behavior of the new v2 code. Claude must not read image, mask, data, dataset, output, or checkpoint files while implementing or testing this task unless they are temporary files created inside the test's temporary directory.

## Files Claude May Modify
- `tasks/0033-pre-sns-meaningful-training-v2.md`
- `configs/training/pre_sns_meaningful_train_v2.example.json`
- `scripts/agent/validate_pre_sns_meaningful_train_v2_config.py`
- `scripts/training/train_pre_sns_meaningful_v2.py`
- `src/cv_forensics/pre_sns_meaningful_training_v2.py`
- `tests/test_pre_sns_meaningful_training_v2.py`
- `docs/pre_sns_meaningful_training_v2.md`
- `src/cv_forensics/__init__.py`

If a parent directory for an allowed file does not exist, Claude may create it. Claude must not create or modify any other files.

## Forbidden Actions
Claude must not:

- download datasets
- access network resources
- install packages
- train on real/local datasets during implementation or tests
- add SNS augmentation
- run SNS perturbation evaluation
- access `.env`, `.env.*`, secrets, credentials, `data/`, `datasets/`, `outputs/`, or `checkpoints/`
- read files from protected paths, even if a config points there
- write tracked prediction artifacts or checkpoints into the repository
- write generated artifacts into repository `outputs/` or `checkpoints/`
- recursively scan dataset directories
- run `git push`, `git pull`, or `git fetch`
- use `rg`
- use `rm -rf`
- create large files
- modify files outside `## Files Claude May Modify`

## Approved runtime roots
Actual approved run mode may write only under approved local roots outside the repository, for example:

- `~/cvf_runs`
- `~/cvf_checkpoints`

The implementation must reject repository output/checkpoint roots and protected paths. The implementation must also reject absolute paths unless they are explicitly approved and under approved roots.

## Implementation Requirements

### 1. V2 training module
Create `src/cv_forensics/pre_sns_meaningful_training_v2.py`.

It must provide importable helpers for config validation, safe path handling, dataset/sample loading, deterministic setup, training, evaluation metrics, threshold calibration, artifact writing, and JSON-safe summaries.

The implementation should reuse existing repository patterns where practical, but must keep this task self-contained enough for standalone tests to run with `python3`.

### 2. V2 trainer CLI
Create `scripts/training/train_pre_sns_meaningful_v2.py`.

CLI:

```bash
python3 scripts/training/train_pre_sns_meaningful_v2.py <config.json>
```

It must support:

- separate `train_manifest_path` and `val_manifest_path`
- `device` values `cpu` and `cuda`
- CUDA training when `device` is `cuda`
- clear failure if `device=cuda` is requested but CUDA is unavailable
- `no_write_dry_run` mode
- actual approved run mode with artifact writing
- checkpoint writing only under `approved_checkpoint_root`
- run artifacts only under `approved_run_root`
- `no_download=true`
- `no_network=true`
- `no_sns_augmentation=true`
- deterministic `seed`
- `max_samples_train`
- `max_samples_val`
- `max_image_size`
- `batch_size`
- `epochs`
- `learning_rate`
- class-balanced or loss-weighted training to reduce real/tampered collapse
- optional checkpoint resume if simple and safe

Training behavior:

- read only explicit samples from the train and validation manifests
- never recursively scan directories
- load images only from explicit `image_path` values during actual approved local runs
- load masks only for tampered samples with explicit `mask_path`
- compute class loss for all samples
- compute family loss only when the sample has a meaningful family label
- compute localization loss only for tampered samples with masks
- avoid SNS transforms entirely
- avoid package downloads and network calls entirely
- keep the default/example config safe and symbolic

### 3. Config validator
Create `scripts/agent/validate_pre_sns_meaningful_train_v2_config.py`.

CLI:

```bash
python3 scripts/agent/validate_pre_sns_meaningful_train_v2_config.py <config.json>
```

It must:

- validate `configs/training/pre_sns_meaningful_train_v2.example.json`
- reject `no_download=false`
- reject `no_network=false`
- reject `no_sns_augmentation=false`
- require separate `train_manifest_path` and `val_manifest_path`
- validate `train_manifest_path` and `val_manifest_path`
- require approved local roots for approved actual local mode
- allow absolute paths only if explicitly approved and under approved roots
- reject repository output/checkpoint roots
- reject protected paths such as `data/`, `datasets/`, `outputs/`, `checkpoints/`, `.env`, and `secrets`
- validate bounds for `max_samples_train`, `max_samples_val`, `max_image_size`, `batch_size`, `epochs`, and `learning_rate`
- validate class label names and family label names
- print `PRE_SNS_MEANINGFUL_TRAINING_V2_CONFIG_OK` on success

The validator must not read images or masks.

### 4. Example config
Create `configs/training/pre_sns_meaningful_train_v2.example.json`.

It must be safe and symbolic:

- no real local dataset paths
- no URLs
- no secrets
- no repository `outputs/` or `checkpoints/` artifact roots
- contains `config_kind` identifying pre-SNS meaningful training v2
- contains `no_download=true`
- contains `no_network=true`
- contains `no_sns_augmentation=true`
- contains `no_write_dry_run=true`
- contains separate `train_manifest_path` and `val_manifest_path`
- contains approved root fields for actual local mode examples
- contains all training knobs required by this task

### 5. Validation/evaluation metrics
The v2 implementation must compute and expose:

- `class_accuracy`
- `class_macro_f1`
- per-class `precision`, `recall`, and `f1`
- `confusion_matrix`
- `family_accuracy` on samples with meaningful family labels
- `tampered_score_summary_by_gt_class`
- `threshold_sweep` for tau values `[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]`
- `selected_tau` based on validation trade-off
- `localization_activation_recall`
- `false_activation_rate`
- `localization_mean_iou` on tampered samples with masks
- `localization_median_iou` on tampered samples with masks

Latency/FPS estimation may be omitted because the report runner already supports it.

Threshold selection should favor tampered recall while penalizing excessive false activations. The exact scoring function may be simple, but it must be deterministic, documented in code or docs, and covered by tests.

### 6. Actual run artifacts
When `no_write_dry_run=false`, the trainer must write only outside the repository under approved roots:

- `run_summary.json`
- `train_metrics.jsonl`
- `val_metrics.json`
- `threshold_calibration.json`
- `confusion_matrix.json`
- `artifact_manifest.json`
- best checkpoint file
- latest checkpoint file if easy

Artifact paths must be recorded in `artifact_manifest.json`. Checkpoint files must be under `approved_checkpoint_root`; run artifacts must be under `approved_run_root`.

When `no_write_dry_run=true`, tests and CLI dry-runs must not create run/checkpoint artifacts.

### 7. Tests
Create `tests/test_pre_sns_meaningful_training_v2.py`.

Tests must be runnable with:

```bash
python3 tests/test_pre_sns_meaningful_training_v2.py
```

Tests must use temporary files and synthetic tiny images/arrays only. They must not use real datasets, network, package installs, or repository artifact directories.

Tests must cover:

- example config validation
- rejection of network/download/SNS flags
- rejection of repository `outputs/` and `checkpoints/` roots
- rejection of protected paths
- validation of separate train and validation manifests
- no-write dry-run creates no repository artifacts
- metric computation
- per-class precision/recall/f1
- confusion matrix
- family accuracy ignoring non-meaningful family labels
- tampered score summaries by ground-truth class
- threshold sweep
- deterministic tau selection
- localization activation recall
- false activation rate
- localization mean and median IoU
- no repository writes from dry-run/test paths

### 8. Documentation
Create `docs/pre_sns_meaningful_training_v2.md`.

It must include marker:

```text
PRE_SNS_MEANINGFUL_TRAINING_V2_OK
```

It must explain:

- why v2 exists after weak pre-SNS validation metrics
- that this is pre-SNS only
- that SNS augmentation is intentionally excluded and will be evaluated later
- train/validation manifest separation
- dry-run versus approved actual local run behavior
- artifact root policy
- checkpoint root policy
- metric and tau calibration outputs
- how this task supports judging whether pre-SNS-only training is meaningful

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_meaningful_train_v2_config.py configs/training/pre_sns_meaningful_train_v2.example.json
```

```bash
python3 tests/test_pre_sns_meaningful_training_v2.py
```

```bash
grep -q PRE_SNS_MEANINGFUL_TRAINING_V2_OK docs/pre_sns_meaningful_training_v2.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0033-pre-sns-meaningful-training-v2.md
```

## Acceptance Criteria
- Changed files are limited to `## Files Claude May Modify`.
- Config validator passes for the example config.
- Standalone tests pass with `python3`.
- Documentation marker exists.
- Dry-run mode performs meaningful guarded validation/training/evaluation logic without writing run/checkpoint artifacts.
- Actual approved mode writes artifacts only outside the repository under approved roots.
- Checkpoints are written only under `approved_checkpoint_root`.
- Evaluation metrics include all required class, family, localization, and tau calibration fields.
- Training uses class-balanced or loss-weighted behavior to reduce real/tampered collapse.
- SNS augmentation is not implemented.
- No dataset download, network access, package installation, protected path access, repository artifact writes, or unrelated file modifications occur.

## Stop Condition
Claude must stop after implementation and validation. Claude must report:

- files changed
- validation commands run and results
- artifact/write behavior for dry-run tests
- whether any validation command was skipped
- any remaining limitations
