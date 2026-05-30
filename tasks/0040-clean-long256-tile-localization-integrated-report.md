# Task 0040: Clean Long256 + Tile Localization Integrated Report

## Role

Codex-only task writer, implementation worker, strict self-reviewer, and limited repair manager.

Claude Code is unavailable for this task. Codex must implement directly only after this task file is committed and the user explicitly says `implement`.

## Goal

Create an integrated pre-SNS report path that combines the clean long256 detector with the crop/tile localization workflow from task 0039.

The intended final flow is:

1. Run clean long256 as the primary class/tampered detector.
2. If long256 predicts `tampered` or tampered-suspect, activate tile/crop localization.
3. Merge tile masks back into full-image coordinates.
4. Produce the final report with `Class / Mask / Family / Reason`.
5. Compare baseline long256 localization against tile-localized mask when GT mask is available.

This task is report/integration/evaluation only. It must not train a model, write checkpoints, add SNS augmentation, download datasets, use network resources, or install packages.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0040-clean-long256-tile-localization-integrated-report.md`
- `tasks/0038-pre-sns-v3-gt-iou-localization-mining.md`
- `tasks/0039-high-resolution-crop-tile-localization-model.md`
- `tasks/0036-pre-sns-v3-dual-scale-calibrated-report-and-mining.md`
- `configs/evaluation/pre_sns_v3_gt_iou_localization_mining.example.json`
- `configs/training/pre_sns_v3_tile_localization.example.json`
- `configs/inference/pre_sns_v3_dual_report.example.json`
- `scripts/agent/check_agent_changes.py`
- `scripts/agent/validate_pre_sns_v3_gt_iou_localization_mining_config.py`
- `scripts/agent/validate_pre_sns_v3_tile_localization_config.py`
- `scripts/evaluation/mine_pre_sns_v3_gt_iou_localization_cases.py`
- `scripts/evaluation/evaluate_pre_sns_v3_tile_localization.py`
- `scripts/training/prepare_pre_sns_v3_tile_localization.py`
- `scripts/inference/run_pre_sns_v3_report.py`
- `scripts/inference/run_pre_sns_v3_dual_scale_report.py`
- `src/cv_forensics/pre_sns_v3_gt_iou_mining.py`
- `src/cv_forensics/pre_sns_v3_tile_localization.py`
- `src/cv_forensics/pre_sns_v3_dual_scale_report.py`
- `src/cv_forensics/pre_sns_v3_report.py`
- `src/cv_forensics/pre_sns_v3_model.py`
- `src/cv_forensics/pre_sns_v3_metrics.py`
- `src/cv_forensics/pre_sns_visualization.py`
- `src/cv_forensics/__init__.py`
- `tests/test_pre_sns_v3_gt_iou_localization_mining.py`
- `tests/test_pre_sns_v3_tile_localization.py`
- `tests/test_pre_sns_v3_dual_scale_report_and_mining.py`
- `tests/test_pre_sns_clean_red_overlay.py`
- `docs/pre_sns_v3_gt_iou_localization_mining.md`
- `docs/pre_sns_v3_tile_localization.md`
- `docs/pre_sns_v3_dual_scale_report_and_mining.md`
- `docs/pre_sns_clean_red_overlay.md`

## Files Codex May Modify

- `tasks/0040-clean-long256-tile-localization-integrated-report.md`
- `configs/inference/pre_sns_v3_long256_tile_report.example.json`
- `scripts/agent/validate_pre_sns_v3_long256_tile_report_config.py`
- `scripts/inference/run_pre_sns_v3_long256_tile_report.py`
- `src/cv_forensics/pre_sns_v3_long256_tile_report.py`
- `tests/test_pre_sns_v3_long256_tile_report.py`
- `docs/pre_sns_v3_long256_tile_report.md`
- `src/cv_forensics/__init__.py`

## Forbidden Actions

- Do not add SNS augmentation.
- Do not execute SNS augmentation.
- Do not train a model.
- Do not write checkpoints.
- Do not download datasets.
- Do not use network resources from shell commands.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not write reports, visual sheets, generated artifacts, or checkpoints into the repository.
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
- Full-image 384/512 training may be a useful future experiment, but this task does not train. It integrates and evaluates long256 + tile localization only.
- Task 0038 is the GT-IoU mining step that identifies localization failures.
- Task 0039 is the crop/tile localization workflow step. It is dry-run-safe and conditional; it does not replace long256.
- Tile localization must activate only for `tampered` or tampered-suspect cases.
- Tile localization must never change the primary long256 class label by itself.
- All outputs must be written under approved external roots outside the repository.

## Implementation Requirements

### 1. Config Validator

Create `scripts/agent/validate_pre_sns_v3_long256_tile_report_config.py`.

Create `configs/inference/pre_sns_v3_long256_tile_report.example.json`.

Approved config rules:

- `config_kind` must be `approved_pre_sns_v3_long256_tile_report`.
- `execution_mode` must be `approved_local_pre_sns_v3_long256_tile_report`.
- `no_download` must be `true`.
- `no_network` must be `true`.
- `no_sns_augmentation` must be `true`.
- `no_training` must be `true`.
- `no_checkpoint_writes` must be `true`.
- `manifest_path` or `image_path` must be under approved input roots outside the repository.
- Optional `gt_iou_mining_records_path` must be under approved input roots outside the repository.
- `long256_checkpoint_path` must be under approved input roots or approved checkpoint roots outside the repository.
- `output_root` must be outside the repository and must not be under approved input roots.
- `max_samples` must be a positive integer when a manifest is used.
- `tile_size`, `tile_stride`, `crop_context_px`, and `high_res_max_size` must be positive integers.
- `tampered_suspect_threshold` must be numeric and non-negative.
- Protected repo paths, URLs, path traversal, repo-local outputs, missing guardrails, training flags, SNS flags, and checkpoint-writing flags must be rejected.

### 2. Integrated Report Module

Create `src/cv_forensics/pre_sns_v3_long256_tile_report.py`.

It must provide reusable, testable functions for:

- Loading and validating integrated report configs.
- Running or accepting clean long256 report data.
- Applying tile activation rules from task 0039.
- Creating tile grids and merging tile masks into full-image masks.
- Selecting the final mask:
  - baseline long256 mask when tile localization is skipped
  - tile-localized mask when tile localization activates and produces a usable mask
  - baseline long256 mask with uncertainty when tile localization activates but is empty
- Preserving the primary long256 class and family outputs.
- Generating a template-based reason from class score, tampered score, mask status, and tile status.
- Computing baseline-vs-tile IoU/Dice/component metrics when GT mask is available.
- Producing JSON-safe report and summary payloads.

Pure gating, merge, mask selection, metric, and explanation functions must be testable without CUDA, real checkpoints, or real datasets.

### 3. Inference Entrypoint

Create `scripts/inference/run_pre_sns_v3_long256_tile_report.py`.

The script must:

- Load and validate the config.
- Support single-image mode via `image_path`.
- Support manifest mode via `manifest_path` and `max_samples`.
- Use clean long256 as the primary detector.
- Activate tile localization only for tampered/tampered-suspect cases.
- Support a fixture/dry-run-safe mode for tests without real checkpoints.
- Write only under approved external `output_root`.

Required output files:

- `long256_tile_integrated_records.jsonl` in manifest mode.
- `long256_tile_integrated_report.json` in single-image mode.
- `long256_tile_integrated_summary.json`.
- `long256_tile_artifact_manifest.json`.

### 4. Final Report Fields

Each report record must include:

- `marker`
- `image_path`
- `primary_detector`
- `class`
- `class_conf`
- `family`
- `family_conf`
- `tampered_score`
- `tile_localization_status`
- `primary_class_unchanged`
- `baseline_long256_mask_stats`
- `tile_mask_stats`
- `final_mask_source`
- `final_mask_stats`
- `baseline_vs_tile_metrics` when GT mask is available
- `localization_delta`
- `reason`
- `visual_artifacts`
- `no_download`
- `no_network`
- `no_training`
- `no_checkpoint_writes`
- `no_sns_augmentation`

### 5. Optional Visual Output

Support optional visual comparison sheets under approved external `output_root`.

Sheets should include:

- input image
- baseline long256 red overlay
- tile-localized red overlay when active
- final red overlay
- GT overlay when GT mask is available
- agreement/difference overlay when applicable

Clean red overlays must not contain text. Comparison sheets may contain labels only outside clean overlay panels.

### 6. Tests

Create `tests/test_pre_sns_v3_long256_tile_report.py`.

The test file must be plain `python3` runnable and must not require pytest as the test runner.

Cover:

- config validator guardrails
- tile activation gating
- final mask selection
- preservation of primary long256 class
- reason generation
- baseline-vs-tile metrics
- manifest summary schema
- single-image report schema
- no repository writes

Use tiny in-memory fixtures and temporary directories outside the repository. Do not require third-party downloads, real checkpoints, real datasets, or CUDA.

### 7. Docs

Create `docs/pre_sns_v3_long256_tile_report.md`.

The document must:

- Include marker `PRE_SNS_V3_LONG256_TILE_REPORT_OK`.
- Explain that this is pre-SNS only.
- Explain that task 0038 mined GT-IoU failures and task 0039 prepared tile localization.
- Explain that clean long256 remains the primary detector.
- Explain that tile localization is conditional and cannot change the primary class.
- Explain that full-image 384/512 is a plausible future experiment, but this task focuses on tile/crop integration because small red-mask failures are likely caused by full-image resizing.
- Document config fields.
- Document output files and report fields.
- State that this task does not add SNS augmentation, train, write checkpoints, download datasets, use network resources, or write outputs into the repository.

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_v3_long256_tile_report_config.py configs/inference/pre_sns_v3_long256_tile_report.example.json
```

```bash
python3 tests/test_pre_sns_v3_long256_tile_report.py
```

```bash
grep -q PRE_SNS_V3_LONG256_TILE_REPORT_OK docs/pre_sns_v3_long256_tile_report.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0040-clean-long256-tile-localization-integrated-report.md
```

## Acceptance Criteria

- The task implements a pre-SNS long256 + tile localization integrated report path.
- Clean long256 remains the primary detector.
- Tile localization activates only for tampered or tampered-suspect cases.
- Tile localization never changes the primary class label.
- Reports include `Class / Mask / Family / Reason` information.
- Manifest mode and single-image mode are supported.
- Fixture/dry-run-safe tests run without real checkpoints, datasets, CUDA, downloads, or network.
- JSON summaries include activation counts, skipped counts, baseline-vs-tile metrics, improvement/regression counts, component stats, guardrail flags, and output paths.
- Optional visual comparison sheets are written only under approved external output roots.
- Clean red overlays contain no text.
- Config validator accepts the example config and rejects unsafe configs.
- The plain Python test passes.
- The docs marker check passes.
- `check_agent_changes.py` passes for this task file.
- No protected paths, data, datasets, outputs, checkpoints, secrets, network, package installation, training, checkpoint writing, or SNS augmentation are touched.

## Stop Condition

After creating this task file, stop and report:

- task file path
- short summary
- git status
- exact `git add` command for the task file
- exact `git commit` command for the task file

Do not implement until the task file is committed and the user explicitly says `implement`.
