# Task 0037: Pre-SNS v3 Hard-Negative Refinement

## Role

Codex-only task writer, implementation worker, strict self-reviewer, and limited repair manager.

Claude Code is unavailable for this task. Codex must implement directly only after this task file is committed and the user explicitly says `implement`.

## Goal

Refine the clean pre-SNS v3 model using hard cases mined from the clean TRAIN split, while evaluating only on the clean VAL split.

This task addresses remaining clean pre-SNS v3 issues:

- real/non-tampered hard negatives
- class-mask consistency

This task must not train on validation hard cases.

This task is pre-SNS only. Do not add SNS augmentation.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0037-pre-sns-v3-hard-negative-refinement.md`
- `tasks/0036-pre-sns-v3-dual-scale-calibrated-report-and-mining.md`
- `tasks/0035-pre-sns-v3-strong-model.md`
- `configs/training/pre_sns_v3_train.example.json`
- `configs/evaluation/pre_sns_v3_hard_mining.example.json`
- `scripts/agent/check_agent_changes.py`
- `scripts/agent/validate_pre_sns_v3_train_config.py`
- `scripts/agent/validate_pre_sns_v3_hard_mining_config.py`
- `scripts/evaluation/mine_pre_sns_v3_hard_cases.py`
- `scripts/training/train_pre_sns_v3.py`
- `src/cv_forensics/pre_sns_v3_training.py`
- `src/cv_forensics/pre_sns_v3_model.py`
- `src/cv_forensics/pre_sns_v3_metrics.py`
- `src/cv_forensics/pre_sns_v3_hard_mining.py`
- `src/cv_forensics/pre_sns_manifest.py`
- `src/cv_forensics/outputs.py`
- `src/cv_forensics/__init__.py`
- `tests/test_pre_sns_v3_training.py`
- `tests/test_pre_sns_v3_dual_scale_report_and_mining.py`
- `docs/pre_sns_v3_training.md`
- `docs/pre_sns_v3_dual_scale_report_and_mining.md`

## Files Codex May Modify

- `tasks/0037-pre-sns-v3-hard-negative-refinement.md`
- `configs/training/pre_sns_v3_refinement.example.json`
- `scripts/agent/validate_pre_sns_v3_refinement_config.py`
- `scripts/training/refine_pre_sns_v3_hard_cases.py`
- `src/cv_forensics/pre_sns_v3_refinement.py`
- `tests/test_pre_sns_v3_refinement.py`
- `docs/pre_sns_v3_refinement.md`
- `src/cv_forensics/__init__.py`

## Forbidden Actions

- Do not add SNS augmentation.
- Do not execute SNS augmentation.
- Do not train on validation hard cases.
- Do not use validation hard-mining outputs as training data.
- Do not download datasets.
- Do not use network resources from shell commands.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not write outputs or checkpoints into the repository.
- Do not modify training data.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not use `rg`.
- Do not create large files.
- Do not commit unless the user explicitly says `commit this`.

## Important Project Facts

- The project is a lightweight multi-head image forensics prototype for social-media-robust image forensics.
- Target outputs are `Class / Mask / Family / Reason`.
- This task remains before SNS augmentation.
- Clean v3 long256 is already strong, with approximate observed metrics:
  - `class_macro_f1 ~= 0.978`
  - `tampered_f1 ~= 0.971`
  - `false_activation_rate ~= 0.051`
- Full hard mining on clean VAL found:
  - `hard_negative_real = 88`
  - `hard_negative_non_tampered = 93`
  - `class_mask_inconsistent_cases = 522`
  - `hard_positive_tampered_low_iou = 0`
  - `failed_cases = 0`
- Clean TRAIN hard mining has been run separately and is the only allowed hard-case source for refinement training.
- Latest observed clean TRAIN hard-mining summary:
  - `manifest_path = /home/rlatjswo/cvf_local_store/pre_sns_manifests_0035_clean/pre_sns_large_manifest_train_clean.local.json`
  - `max_samples = 31994`
  - `hard_negative_real = 517`
  - `hard_negative_non_tampered = 525`
  - `hard_positive_tampered_low_iou = 0`
  - `class_mask_inconsistent_cases = 1514`
  - `failed_cases = 0`
  - marker `PRE_SNS_V3_HARD_MINING_OK`
- The remaining issue is mainly hard negatives and class-mask consistency, not broad low-IoU tampered failure.
- v3 checkpoints produce class logits, tamper binary logits, family logits, and localization logits.
- Selected tau must remain FPR-constrained.

## Implementation Requirements

### 1. Refinement Trainer Entrypoint

Create `scripts/training/refine_pre_sns_v3_hard_cases.py`.

The script must:

- Load a JSON config.
- Validate the config before execution.
- Load a base v3 checkpoint, typically a clean long256 `best_checkpoint.pt`.
- Use `train_manifest_path` for training.
- Use `val_manifest_path` only for evaluation.
- Use only TRAIN hard-mining JSON files for refinement sampling:
  - `hard_negative_real.json`
  - `hard_negative_non_tampered.json`
  - `hard_positive_tampered_low_iou.json`
  - `class_mask_inconsistent_cases.json`
- Refuse to train if hard-case config paths indicate validation mining outputs.
- Save `best_checkpoint.pt` and `latest_checkpoint.pt` only under `approved_checkpoint_root` outside the repository.
- Write run artifacts only under `approved_run_root` outside the repository.

### 2. Refinement Library

Create `src/cv_forensics/pre_sns_v3_refinement.py`.

It must provide reusable, testable functions for:

- Loading configs.
- Validating configs and guardrails.
- Loading train and val manifests.
- Loading hard-case JSON records.
- Mapping hard cases to train samples by `sample_id` and/or `image_path`.
- Rejecting hard cases that are not from the configured train split.
- Building weighted or oversampled training records.
- Running guarded refinement from a base v3 checkpoint.
- Computing and writing the required metric and artifact schema.

Reuse existing v3 model, dataset, transform, loss, and metric code where practical instead of duplicating large blocks.

### 3. Sampling and Weighting

Normal train samples must remain available.

Oversample:

- `hard_negative_real`
- `hard_negative_non_tampered`
- `class_mask_inconsistent_cases` when they are from the train split
- `hard_positive_tampered_low_iou` if present

Default configurable oversampling weights:

- `hard_negative_real_oversample_weight`: `5`
- `hard_negative_non_tampered_oversample_weight`: `4`
- `hard_inconsistent_oversample_weight`: `3`
- `hard_positive_low_iou_oversample_weight`: `3`

The implementation must make the oversampling logic testable without requiring CUDA or a real dataset.

### 4. Loss Refinement

Start from the v3 loss design and refine weights:

- `class_loss_weight`: default `1.0`
- `non_tampered_empty_mask_loss_weight`: default `0.8`
- `tamper_binary_loss_weight`: default `1.5`
- `localization_loss_weight`: default `12.0`
- `family_loss_weight`: default `0.2`
- keep `dice_loss_weight` configurable if inherited from v3

Optional consistency behavior is allowed when narrow and testable:

- If the ground-truth class is non-tampered, penalize active mask area.
- If the ground-truth class is `real` or `full_synthetic`, binary tamper target must be `0`.
- Optional mask-area regularization for non-tampered samples.

### 5. Fine-Tuning Controls

Support:

- CUDA when available and configured.
- `max_image_size` including `256`.
- `epochs`.
- `batch_size`.
- `learning_rate`.
- Optional lower backbone learning rate if simple.
- Optional `freeze_backbone_epochs` if simple.

Default refinement config:

- `epochs`: `20`
- `batch_size`: `2`
- `learning_rate`: `0.00003`
- `max_image_size`: `256`

Tests must be able to run tiny CPU-only fixture refinement or dry-run refinement without third-party downloads.

### 6. Metrics

Report the same v3 metrics:

- `class_accuracy`
- `class_macro_f1`
- `real_recall`
- `tampered_precision`
- `tampered_recall`
- `tampered_f1`
- `family_accuracy`
- `selected_tau`
- `false_activation_rate`
- `localization_activation_recall`
- `localization_mean_iou`
- `localization_median_iou`

Add refinement metrics:

- `hard_negative_real_count_used`
- `hard_negative_non_tampered_count_used`
- `hard_inconsistent_count_used`
- `hard_positive_low_iou_count_used`
- `class_mask_inconsistency_proxy` when available
- `non_tampered_mask_activation_rate` when available

Tau selection must remain FPR-constrained.

### 7. Artifacts

Under `approved_run_root`, write:

- `run_summary.json`
- `val_metrics.json`
- `threshold_calibration.json`
- `confusion_matrix.json`
- `per_source_confusion_matrix.json` when available
- `refinement_hard_case_summary.json`
- `train_metrics.jsonl`
- `artifact_manifest.json`
- `config_snapshot.json`

Under `approved_checkpoint_root`, write:

- `best_checkpoint.pt`
- `latest_checkpoint.pt`

All artifact and checkpoint roots must be outside the repository.

### 8. Config Validator

Create `configs/training/pre_sns_v3_refinement.example.json`.

Create `scripts/agent/validate_pre_sns_v3_refinement_config.py`.

Approved config requirements:

- `config_kind`: `approved_pre_sns_v3_refinement`
- `execution_mode`: `approved_local_pre_sns_v3_refinement`
- `required_approval_text`: `I_APPROVE_PRE_SNS_V3_REFINEMENT`
- approved runs require `user_approval_text` to equal `I_APPROVE_PRE_SNS_V3_REFINEMENT`
- `hard_cases_split`: `train`
- `no_download`: `true`
- `no_network`: `true`
- `no_sns_augmentation`: `true`
- approved train and val manifest paths must be under approved input roots outside the repository
- approved hard-case JSON paths must be under approved input roots outside the repository
- approved base checkpoint path must be under approved input roots or approved checkpoint roots outside the repository
- approved run and checkpoint roots must be outside the repository

The validator must reject:

- `no_download: false`
- `no_network: false`
- `no_sns_augmentation: false`
- `hard_cases_split: val`
- validation hard-mining output paths for training
- paths containing `hard_mining_full_repaired` with validation indicators
- repo-local run or checkpoint roots
- protected path references
- remote URL schemes
- secret-like keys or values

Example configs may use symbolic paths and must not execute training.

### 9. Tests

Create `tests/test_pre_sns_v3_refinement.py`.

The test file must be plain `python3` runnable and must not require pytest as the test runner.

Cover:

- validator accepts the example config.
- validator rejects validation hard cases.
- validator rejects missing guardrails.
- validator rejects repo-local output/checkpoint writes.
- no-write guard or dry-run behavior if implemented.
- tiny actual fixture refinement or tiny dry-run refinement.
- hard-case oversampling logic.
- artifact writing outside the repository.
- result schema completeness.
- tests run safely on CPU without requiring CUDA.

### 10. Docs

Create `docs/pre_sns_v3_refinement.md`.

The document must:

- Include marker `PRE_SNS_V3_REFINEMENT_OK`.
- Explain that this is pre-SNS hard-negative refinement, not SNS augmentation.
- Explain that train hard-mining outputs may be used for refinement.
- Explain that validation hard-mining outputs must not be used for training.
- Document expected config fields.
- Document artifact outputs.
- Document the added hard-case metrics.

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_v3_refinement_config.py configs/training/pre_sns_v3_refinement.example.json
```

```bash
python3 tests/test_pre_sns_v3_refinement.py
```

```bash
grep -q PRE_SNS_V3_REFINEMENT_OK docs/pre_sns_v3_refinement.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0037-pre-sns-v3-hard-negative-refinement.md
```

## Acceptance Criteria

- The task adds a guarded pre-SNS v3 refinement path based on clean TRAIN hard cases only.
- The validator rejects validation hard cases and unsafe guardrail settings.
- Hard-case oversampling is configurable and test-covered.
- Refinement artifacts and checkpoints are written only outside the repository.
- Reported metrics include required v3 metrics plus refinement hard-case usage counts.
- Selected tau remains FPR-constrained.
- Tests are plain `python3` runnable and safe without CUDA.
- Documentation includes `PRE_SNS_V3_REFINEMENT_OK` and clearly states this is not SNS augmentation.
- `python3 scripts/agent/check_agent_changes.py tasks/0037-pre-sns-v3-hard-negative-refinement.md` passes after implementation.

## Stop Condition

After creating this task file, stop.

Do not implement until:

1. The user commits this task file.
2. The user explicitly says `implement`.
