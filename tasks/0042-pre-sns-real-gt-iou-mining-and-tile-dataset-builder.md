# Task 0042: Pre-SNS Real GT-IoU Mining and Tile Dataset Builder

## Role

Codex-only task writer, implementation worker, strict self-reviewer, and limited repair manager.

Claude Code is unavailable for this task. Codex must implement directly only after this task file is committed and the user explicitly says `implement`.

## Goal

Clean long256 is strong as a detector, but red-mask localization still fails on some tampered samples. Prior hard-negative refinement only modestly improved IoU and did not solve cases where all candidate masks fail. This task must run or prepare a real GT-IoU mining pass over the clean TRAIN split and build a tile/crop localization dataset manifest for a future lightweight high-resolution localizer.

This task is analysis and dataset-manifest building only. It must not train a model, add SNS augmentation, download datasets, use network resources, install packages, or write outputs inside the repository.

The project should differentiate itself from SIDA's large multimodal framework as a lightweight conditional pipeline:

```text
clean long256 global detector -> high-resolution tile/crop localizer
```

The purpose of this task is to identify localization failure cases and prepare tile/crop records for future training.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0042-pre-sns-real-gt-iou-mining-and-tile-dataset-builder.md`
- `tasks/0041-pre-sns-long256-tile-report-local-run.md`
- `tasks/0040-clean-long256-tile-localization-integrated-report.md`
- `tasks/0039-high-resolution-crop-tile-localization-model.md`
- `tasks/0038-pre-sns-v3-gt-iou-localization-mining.md`
- `configs/evaluation/pre_sns_v3_gt_iou_localization_mining.example.json`
- `configs/training/pre_sns_v3_tile_localization.example.json`
- `configs/inference/pre_sns_v3_long256_tile_report.example.json`
- `configs/inference/pre_sns_v3_long256_tile_local_run.example.json`
- `scripts/agent/check_agent_changes.py`
- `scripts/agent/validate_pre_sns_v3_gt_iou_localization_mining_config.py`
- `scripts/agent/validate_pre_sns_v3_tile_localization_config.py`
- `scripts/agent/validate_pre_sns_v3_long256_tile_report_config.py`
- `scripts/agent/validate_pre_sns_v3_long256_tile_local_run_config.py`
- `scripts/evaluation/mine_pre_sns_v3_gt_iou_localization_cases.py`
- `scripts/evaluation/evaluate_pre_sns_v3_tile_localization.py`
- `scripts/inference/run_pre_sns_v3_report.py`
- `scripts/inference/run_pre_sns_v3_long256_tile_report.py`
- `scripts/inference/run_pre_sns_v3_long256_tile_local_run.py`
- `src/cv_forensics/pre_sns_v3_gt_iou_mining.py`
- `src/cv_forensics/pre_sns_v3_tile_localization.py`
- `src/cv_forensics/pre_sns_v3_long256_tile_report.py`
- `src/cv_forensics/pre_sns_v3_long256_tile_local_run.py`
- `src/cv_forensics/pre_sns_v3_report.py`
- `src/cv_forensics/pre_sns_v3_model.py`
- `src/cv_forensics/pre_sns_visualization.py`
- `src/cv_forensics/__init__.py`
- `tests/test_pre_sns_v3_gt_iou_localization_mining.py`
- `tests/test_pre_sns_v3_tile_localization.py`
- `tests/test_pre_sns_v3_long256_tile_report.py`
- `tests/test_pre_sns_v3_long256_tile_local_run.py`
- `docs/pre_sns_v3_gt_iou_localization_mining.md`
- `docs/pre_sns_v3_tile_localization.md`
- `docs/pre_sns_v3_long256_tile_report.md`
- `docs/pre_sns_v3_long256_tile_local_run.md`

Codex may read approved local config files under `.local/` only when needed to understand path conventions. Codex must not read `.env`, `.env.*`, secrets, protected repo `data/`, `datasets/`, `outputs/`, or `checkpoints/`.

## Files Codex May Modify

- `tasks/0042-pre-sns-real-gt-iou-mining-and-tile-dataset-builder.md`
- `configs/evaluation/pre_sns_v3_gt_iou_tile_builder.example.json`
- `scripts/agent/validate_pre_sns_v3_gt_iou_tile_builder_config.py`
- `scripts/evaluation/build_pre_sns_v3_gt_iou_tile_dataset.py`
- `src/cv_forensics/pre_sns_v3_gt_iou_tile_builder.py`
- `tests/test_pre_sns_v3_gt_iou_tile_builder.py`
- `docs/pre_sns_v3_gt_iou_tile_builder.md`
- `src/cv_forensics/__init__.py`

## Forbidden Actions

- Do not train a model.
- Do not add SNS augmentation.
- Do not execute SNS augmentation.
- Do not download datasets.
- Do not use network resources from shell commands.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, repo-local `data/`, repo-local `datasets/`, repo-local `outputs/`, or repo-local `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not write outputs, generated manifests, reports, visual audit files, cropped images, checkpoints, or large artifacts inside the repository.
- Do not write cropped image files by default.
- Do not use validation hard cases as training data.
- Do not modify training data.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not use `rg`.
- Do not create large files.
- Do not commit unless the user explicitly says `commit this`.
- Do not modify files outside the allowed list.

## Important Project Facts

- The project is a lightweight multi-head image forensics prototype for social-media-robust image forensics.
- Target outputs are `Class / Mask / Family / Reason`.
- Clean long256 is the current primary detector candidate.
- Current focus is pre-SNS red-mask localization quality.
- Task 0038 added GT-IoU localization mining infrastructure.
- Task 0039 prepared the high-resolution crop/tile localization workflow.
- Task 0040 integrated clean long256 with conditional tile localization reporting.
- Task 0041 produced existing-input GT-vs-prediction comparison artifacts and showed refined localization only modestly improved IoU.
- SIDA uses a large multimodal framework, but this project should remain a lightweight conditional pipeline.
- This task prepares a future high-resolution localizer; it does not train it.
- Outputs must be written only under approved external roots outside the repository.

## Implementation Requirements

### 1. Config Validator

Create `scripts/agent/validate_pre_sns_v3_gt_iou_tile_builder_config.py`.

Create `configs/evaluation/pre_sns_v3_gt_iou_tile_builder.example.json`.

Approved config rules:

- `config_kind` must be `approved_pre_sns_v3_gt_iou_tile_builder`.
- `execution_mode` must be `approved_local_pre_sns_v3_gt_iou_tile_builder`.
- `approved_real_data_access` must be `true`.
- `no_download` must be `true`.
- `no_network` must be `true`.
- `no_sns_augmentation` must be `true`.
- `no_training` must be `true`.
- Clean train manifest path is required.
- Clean train manifest path must be under approved input roots outside the repository.
- Long256 checkpoint path is required.
- Long256 checkpoint path must be under approved input roots or approved checkpoint roots outside the repository.
- Optional long224/refined checkpoint paths may be accepted for analysis only.
- Optional hard-negative mining records path may be accepted only under approved input roots.
- `output_root` must be outside the repository and must not be under approved input roots.
- `approved_input_roots` must be a non-empty list of absolute external roots.
- `approved_checkpoint_roots`, when present, must be a list of absolute external roots.
- `max_samples` must be a positive integer.
- `tile_size` must be a positive integer and should default to `512`.
- Jitter/oversampling/count parameters must be non-negative integers where applicable.
- Visual audit settings must be optional and bounded.
- Protected path segments, URLs, path traversal, repo-local outputs, missing guardrails, training flags, SNS flags, network/download flags, and validation hard cases marked as training records must be rejected.

### 2. Real GT-IoU Mining

Create `src/cv_forensics/pre_sns_v3_gt_iou_tile_builder.py`.

The module must provide reusable, testable functions for:

- Loading and validating configs.
- Loading clean train manifests or sample lists.
- Selecting tampered TRAIN samples with GT masks.
- Running the long256 report or using predicted masks already present in local manifest/report-compatible records.
- Collecting predicted masks.
- Computing:
  - IoU
  - Dice
  - GT area percent
  - predicted area percent
  - GT component count
  - predicted component count
  - largest GT component area percent
  - largest predicted component area percent
  - predicted/GT area ratio
- Classifying localization failure types:
  - `severe_iou_fail`: IoU < `0.05`
  - `low_iou`: IoU < `0.15`
  - `weak_iou`: IoU < `0.25`
  - `good_iou`: IoU >= `0.40`
  - `empty_prediction`
  - `undersegmented`
  - `oversegmented`
  - `wrong_region`
  - `tiny_gt_mask`
  - `fragmented_prediction`
- Summarizing model-level and bucket-level localization quality.

Pure metric, component, bucketing, crop-box, and manifest-building functions must be testable without CUDA, real checkpoints, real datasets, downloads, or network.

### 3. Tile/Crop Dataset Builder

The builder input is:

- clean train manifest
- GT-IoU mining records
- optional hard-negative mining records

The builder output is an explicit tile localization manifest JSON outside the repository.

Do not write cropped image files by default. Prefer manifest-only crop records.

Positive tile records must:

- Use GT-centered `512` crops by default.
- Use jittered GT-centered crops.
- Oversample `severe_iou_fail` and `low_iou` samples.
- Include crop boxes that keep the GT mask inside the tile as much as possible.
- Handle image boundaries safely.
- Include:
  - `source_image_path`
  - `source_mask_path`
  - `crop_box` as `[x1, y1, x2, y2]`
  - `tile_class`: `positive_tampered`
  - `mining_bucket`
  - `gt_area_pct_in_crop`
  - `expected_mask_type`: `cropped_gt_mask`

Negative tile records must:

- Use real/full_synthetic random crops with empty masks.
- If hard-negative real records exist, create crops around false-positive predicted mask areas.
- Include:
  - `source_image_path`
  - `crop_box` as `[x1, y1, x2, y2]`
  - `tile_class`: `negative_real`, `negative_synthetic`, or `hard_negative`
  - `expected_mask_type`: `empty_mask`
  - `mining_bucket` when applicable

The manifest must make future training reproducible by recording source manifest path, source mining record path, tile size, crop policy, counts by tile class, and guardrail flags.

### 4. Entrypoint

Create `scripts/evaluation/build_pre_sns_v3_gt_iou_tile_dataset.py`.

The script must:

- Load and validate config.
- Refuse unsafe paths before writing outputs.
- Run or load GT-IoU mining records for clean TRAIN tampered samples up to `max_samples`.
- Build manifest-only positive/negative/hard-negative tile records.
- Write all outputs under approved external `output_root`.
- Print compact JSON summary to stdout.
- Preserve guardrails in all summaries:
  - no download
  - no network
  - no SNS augmentation
  - no training

### 5. Required Output Files

Write these files under approved external `output_root`:

- `gt_iou_train_records.jsonl`
- `gt_iou_train_summary.json`
- `severe_iou_fail_cases.json`
- `low_iou_cases.json`
- `weak_iou_cases.json`
- `empty_prediction_cases.json`
- `wrong_region_cases.json`
- `undersegmented_cases.json`
- `oversegmented_cases.json`
- `tiny_gt_mask_cases.json`
- `fragmented_prediction_cases.json`
- `tile_localization_manifest.json`
- `tile_manifest_summary.json`
- `artifact_manifest.json`

### 6. Visual Audit Outputs

Visual audit outputs are optional and controlled by config.

For top N severe/low-IoU cases, write comparison sheets outside the repository:

- input image
- GT red overlay
- long256 red overlay
- GT/pred agreement overlay

Clean red overlays must contain no text. Comparison sheets may contain labels.

Visual audit must be skipped gracefully when image files are unavailable or PIL is unavailable.

### 7. Tests

Create `tests/test_pre_sns_v3_gt_iou_tile_builder.py`.

The test file must be plain `python3` runnable and must not require pytest as the test runner.

Cover:

- IoU and Dice computation.
- Empty mask behavior.
- Severe/low/weak/good IoU bucket classification.
- Undersegmented and oversegmented classification.
- Crop box generation near image boundaries.
- Positive tile record generation.
- Negative tile record generation.
- Hard-negative tile generation when provided.
- Validator rejects SNS/network/download/training.
- Validator rejects repo-local output roots.
- Artifact manifest schema.
- No repository writes during dry-run/fixture execution.

Use tiny in-memory fixtures and temporary directories outside the repository.

### 8. Docs

Create `docs/pre_sns_v3_gt_iou_tile_builder.md`.

The document must:

- Include marker `PRE_SNS_V3_GT_IOU_TILE_BUILDER_OK`.
- Explain that this is pre-SNS analysis and tile-manifest building only.
- Explain that this is not SNS augmentation.
- Explain that this is not training.
- Explain that clean long256 remains the primary detector.
- Explain that this task prepares a lightweight high-resolution tile localizer path.
- Document config fields.
- Document output files.
- Document mining buckets.
- Document tile manifest fields.
- State that outputs must be outside the repository.

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_v3_gt_iou_tile_builder_config.py configs/evaluation/pre_sns_v3_gt_iou_tile_builder.example.json
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_pre_sns_v3_gt_iou_tile_builder.py
```

```bash
grep -q PRE_SNS_V3_GT_IOU_TILE_BUILDER_OK docs/pre_sns_v3_gt_iou_tile_builder.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0042-pre-sns-real-gt-iou-mining-and-tile-dataset-builder.md
```

## Acceptance Criteria

- The config validator accepts the example config and rejects unsafe configs.
- The evaluator/builder entrypoint validates config before execution.
- Real-data execution requires explicit approved real data access in config.
- Reports, tile manifests, and visual audit artifacts are written only under approved external `output_root`.
- The mining pass writes all required GT-IoU records and bucket files.
- Mining records contain IoU, Dice, area, component, largest component, and predicted/GT ratio metrics.
- Failure buckets are deterministic and follow the configured thresholds.
- The tile manifest contains positive, negative, and hard-negative record schemas where inputs are available.
- Positive records are GT-centered or jittered GT-centered and keep GT inside the tile as much as possible.
- Negative records use real/full_synthetic random crops with empty masks.
- Optional hard-negative records use false-positive predicted mask areas.
- No cropped image files are written by default.
- No training, SNS augmentation, network, download, package installation, or checkpoint writing occurs.
- The docs marker check passes.
- `scripts/agent/check_agent_changes.py` reports that all changed files are within the allowed list.

## Stop Condition

After implementation, stop and report PASS or NEEDS_FIX in Korean.

If PASS:

- Show changed files.
- Show validation results.
- Show protected path status.
- Show exact `git add` command.
- Show exact `git commit` command.
- Wait for the user to commit.

If NEEDS_FIX:

- Analyze the exact failure cause from validation output, git diff, relevant source files, relevant tests, and task requirements.
- Apply a small targeted direct repair only when it stays within the allowed file list and does not require training, SNS augmentation, network, protected data access, or a new architecture decision.
- Repeat at most 2 repair cycles.
- If still failing after 2 cycles, stop and summarize root cause, files changed, validations passed/failed, and recommended manual decision.
