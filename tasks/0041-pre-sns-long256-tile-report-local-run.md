# Task 0041: Pre-SNS Long256 Tile Report Local Run

## Role

Codex-only task writer, implementation worker, strict self-reviewer, and limited repair manager.

Claude Code is unavailable for this task. Codex must implement directly only after this task file is committed and the user explicitly says `implement`.

## Goal

Run the clean long256 + tile localization integrated report on approved local pre-SNS samples/checkpoints and inspect whether the final red mask output is visually and quantitatively usable.

This task is local evaluation/reporting only. It must not train a model, add SNS augmentation, download datasets, use network resources, install packages, write checkpoints, or write outputs into the repository.

The purpose is to produce concrete local artifacts that let a human inspect whether red-mask localization is good enough to proceed toward SNS robustness evaluation, or whether the next step should be real 512 crop/tile localization training.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0041-pre-sns-long256-tile-report-local-run.md`
- `tasks/0040-clean-long256-tile-localization-integrated-report.md`
- `tasks/0039-high-resolution-crop-tile-localization-model.md`
- `tasks/0038-pre-sns-v3-gt-iou-localization-mining.md`
- `configs/inference/pre_sns_v3_long256_tile_report.example.json`
- `configs/training/pre_sns_v3_tile_localization.example.json`
- `configs/evaluation/pre_sns_v3_gt_iou_localization_mining.example.json`
- `scripts/agent/check_agent_changes.py`
- `scripts/agent/validate_pre_sns_v3_long256_tile_report_config.py`
- `scripts/agent/validate_pre_sns_v3_tile_localization_config.py`
- `scripts/agent/validate_pre_sns_v3_gt_iou_localization_mining_config.py`
- `scripts/inference/run_pre_sns_v3_long256_tile_report.py`
- `scripts/evaluation/evaluate_pre_sns_v3_tile_localization.py`
- `scripts/evaluation/mine_pre_sns_v3_gt_iou_localization_cases.py`
- `src/cv_forensics/pre_sns_v3_long256_tile_report.py`
- `src/cv_forensics/pre_sns_v3_tile_localization.py`
- `src/cv_forensics/pre_sns_v3_gt_iou_mining.py`
- `src/cv_forensics/pre_sns_v3_report.py`
- `src/cv_forensics/pre_sns_v3_model.py`
- `src/cv_forensics/pre_sns_visualization.py`
- `src/cv_forensics/__init__.py`
- `tests/test_pre_sns_v3_long256_tile_report.py`
- `tests/test_pre_sns_v3_tile_localization.py`
- `tests/test_pre_sns_v3_gt_iou_localization_mining.py`
- `docs/pre_sns_v3_long256_tile_report.md`
- `docs/pre_sns_v3_tile_localization.md`
- `docs/pre_sns_v3_gt_iou_localization_mining.md`

## Files Codex May Modify

- `tasks/0041-pre-sns-long256-tile-report-local-run.md`
- `configs/inference/pre_sns_v3_long256_tile_local_run.example.json`
- `scripts/agent/validate_pre_sns_v3_long256_tile_local_run_config.py`
- `scripts/inference/run_pre_sns_v3_long256_tile_local_run.py`
- `src/cv_forensics/pre_sns_v3_long256_tile_local_run.py`
- `tests/test_pre_sns_v3_long256_tile_local_run.py`
- `docs/pre_sns_v3_long256_tile_local_run.md`
- `src/cv_forensics/__init__.py`

## Forbidden Actions

- Do not train a model.
- Do not add SNS augmentation.
- Do not execute SNS augmentation.
- Do not download datasets.
- Do not use network resources from shell commands.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not write reports, overlays, galleries, generated artifacts, or checkpoints into the repository.
- Do not modify training data.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not use `rg`.
- Do not create large files.
- Do not commit unless the user explicitly says `commit this`.

## Important Project Facts

- The project is a lightweight multi-head image forensics prototype for social-media-robust image forensics.
- Target outputs are `Class / Mask / Family / Reason`.
- This task remains pre-SNS and must not add SNS augmentation.
- Clean long256 is the current primary detector candidate.
- Task 0038 mines GT-IoU localization failure cases.
- Task 0039 prepares the tile/crop localization workflow.
- Task 0040 integrates clean long256 and conditional tile localization into a report path.
- Task 0041 must run or prepare a guarded local run of the 0040 integrated report over approved local samples/checkpoints and produce inspectable red-mask artifacts.
- Outputs must be written only under approved external roots outside the repository.
- If red-mask output is usable, the next likely task is SNS robustness evaluation.
- If red-mask output is not usable, the next likely task is real high-resolution crop/tile localization training with explicit approval.

## Implementation Requirements

### 1. Local Run Config Validator

Create `scripts/agent/validate_pre_sns_v3_long256_tile_local_run_config.py`.

Create `configs/inference/pre_sns_v3_long256_tile_local_run.example.json`.

Approved config rules:

- `config_kind` must be `approved_pre_sns_v3_long256_tile_local_run`.
- `execution_mode` must be `approved_local_pre_sns_v3_long256_tile_local_run`.
- `no_download` must be `true`.
- `no_network` must be `true`.
- `no_sns_augmentation` must be `true`.
- `no_training` must be `true`.
- `no_checkpoint_writes` must be `true`.
- `manifest_path` or `sample_list_path` must be under approved input roots outside the repository.
- Optional `gt_iou_mining_records_path` must be under approved input roots outside the repository.
- `long256_checkpoint_path` must be under approved input roots or approved checkpoint roots outside the repository.
- `output_root` must be outside the repository and must not be under approved input roots.
- `max_samples` must be a positive integer.
- `tile_size`, `tile_stride`, `crop_context_px`, and `high_res_max_size` must be positive integers.
- `tampered_suspect_threshold` must be numeric and non-negative.
- `visual_top_n` must be a non-negative integer.
- Protected repo paths, URLs, path traversal, repo-local outputs, missing guardrails, training flags, SNS flags, and checkpoint-writing flags must be rejected.

### 2. Local Run Module

Create `src/cv_forensics/pre_sns_v3_long256_tile_local_run.py`.

It must provide reusable, testable functions for:

- Loading and validating local-run configs.
- Loading a manifest or selected sample list.
- Building a per-sample execution plan for the 0040 integrated report.
- Calling or wrapping the 0040 integrated report logic.
- Writing JSON reports under approved external `output_root`.
- Writing clean red overlays and comparison sheets under approved external `output_root`.
- Comparing baseline long256 masks against tile-localized final masks when GT masks exist.
- Summarizing IoU, Dice, area, component count, largest component, activation, improvement, regression, and skipped counts.
- Listing good red-mask cases and failed red-mask cases.
- Producing a human-inspection gallery manifest.
- Producing a final recommendation:
  - `ready_for_sns_robustness_evaluation`
  - `needs_real_tile_localization_training`
  - `needs_manual_review`

Pure summary, bucketing, recommendation, and schema functions must be testable without CUDA, real checkpoints, or real datasets.

### 3. Local Run Entrypoint

Create `scripts/inference/run_pre_sns_v3_long256_tile_local_run.py`.

The script must:

- Load and validate the config.
- Refuse unsafe paths before writing outputs.
- Run the local report plan over up to `max_samples`.
- Use the existing task 0040 integrated report path where practical.
- Write only under approved external `output_root`.
- Print a compact JSON summary to stdout.

Required output files:

- `local_run_records.jsonl`
- `local_run_summary.json`
- `good_red_mask_cases.json`
- `failed_red_mask_cases.json`
- `red_mask_gallery_manifest.json`
- `artifact_manifest.json`

### 4. Red-Mask Usability Buckets

Implement deterministic case buckets:

- `good_red_mask_cases`: GT exists and final mask IoU/Dice meets configured thresholds, or no GT exists but final mask is non-empty and class/mask status is consistent.
- `failed_red_mask_cases`: empty final mask for tampered/suspect samples, low final IoU when GT exists, serious oversegmentation, serious undersegmentation, or missing visual artifact.
- `needs_manual_review_cases`: no GT and ambiguous class/mask/tile status.

Configurable defaults:

- `good_iou_threshold`: `0.4`
- `good_dice_threshold`: `0.5`
- `low_iou_threshold`: `0.15`
- `overseg_ratio_threshold`: `2.0`
- `underseg_ratio_threshold`: `0.5`

### 5. Visual Output

For selected cases, write clean red overlays and comparison sheets under approved external `output_root`.

Visuals should include:

- input image
- baseline long256 red overlay
- tile-localized red overlay when active
- final red overlay
- GT overlay when GT mask is available
- agreement/difference overlay when applicable

Clean red overlay images must not contain text. Comparison sheets may contain labels outside clean overlay panels.

### 6. Tests

Create `tests/test_pre_sns_v3_long256_tile_local_run.py`.

The test file must be plain `python3` runnable and must not require pytest as the test runner.

Cover:

- config validator guardrails
- sample-plan construction
- red-mask usability bucketing
- recommendation logic
- summary schema
- gallery manifest schema
- no repository writes
- dry-run/fixture local run without real checkpoints, datasets, CUDA, downloads, or network

Use tiny in-memory fixtures and temporary directories outside the repository.

### 7. Docs

Create `docs/pre_sns_v3_long256_tile_local_run.md`.

The document must:

- Include marker `PRE_SNS_V3_LONG256_TILE_LOCAL_RUN_OK`.
- Explain that this is pre-SNS local evaluation/reporting only.
- Explain that the task checks whether the final red mask is visually and quantitatively usable.
- Explain the relationship to task 0038, task 0039, and task 0040.
- Document config fields.
- Document output files.
- Document red-mask usability buckets.
- Document recommendation meanings.
- State that this task does not train, add SNS augmentation, write checkpoints, download datasets, use network resources, install packages, or write outputs into the repository.

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_v3_long256_tile_local_run_config.py configs/inference/pre_sns_v3_long256_tile_local_run.example.json
```

```bash
python3 tests/test_pre_sns_v3_long256_tile_local_run.py
```

```bash
grep -q PRE_SNS_V3_LONG256_TILE_LOCAL_RUN_OK docs/pre_sns_v3_long256_tile_local_run.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0041-pre-sns-long256-tile-report-local-run.md
```

## Acceptance Criteria

- The local-run config validator accepts the example config and rejects unsafe configs.
- The local-run entrypoint validates config before execution.
- Reports and visual artifacts are written only under approved external `output_root`.
- The run writes all required JSON output files.
- Good, failed, and manual-review red-mask buckets are deterministic.
- The summary includes IoU, Dice, area, component, activation, improvement, regression, skipped, and recommendation fields.
- A human-inspection gallery manifest is produced.
- The docs marker check passes.
- The plain Python test passes.
- `check_agent_changes.py` passes for this task file.
- No protected paths, data, datasets, outputs, checkpoints, secrets, network, package installation, model training, checkpoint writing, or SNS augmentation are touched.

## Stop Condition

After creating this task file, stop and report:

- task file path
- short summary
- git status
- exact `git add` command for the task file
- exact `git commit` command for the task file
- next recommended task after 0041 depending on the local-run result

Do not implement until the task file is committed and the user explicitly says `implement`.
