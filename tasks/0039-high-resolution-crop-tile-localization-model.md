# Task 0039: High-Resolution Crop/Tile Localization Model

## Role

Codex-only task writer, implementation worker, strict self-reviewer, and limited repair manager.

Claude Code is unavailable for this task. Codex must implement directly only after this task file is committed and the user explicitly says `implement`.

## Goal

Build a pre-SNS high-resolution crop/tile localization path to improve tampered mask localization using the GT-IoU failure patterns mined in task 0038.

This task prepares and implements a controlled crop/tile localization training/evaluation workflow, but it must not add SNS augmentation. Long256 remains the primary detector, and the tile/crop localization path must improve localization only when a sample is tampered or tampered-suspect. It must not replace the long256 detector.

This task does not approve real training. Implement a dry-run-safe training/preparation entrypoint and evaluation path. Real training, checkpoint writing, or large run execution still requires explicit user approval after this task is implemented and reviewed.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0039-high-resolution-crop-tile-localization-model.md`
- `tasks/0038-pre-sns-v3-gt-iou-localization-mining.md`
- `tasks/0036-pre-sns-v3-dual-scale-calibrated-report-and-mining.md`
- `tasks/0037-pre-sns-v3-hard-negative-refinement.md`
- `configs/evaluation/pre_sns_v3_gt_iou_localization_mining.example.json`
- `configs/evaluation/pre_sns_v3_hard_mining.example.json`
- `configs/inference/pre_sns_v3_dual_report.example.json`
- `scripts/agent/check_agent_changes.py`
- `scripts/agent/validate_pre_sns_v3_gt_iou_localization_mining_config.py`
- `scripts/agent/validate_pre_sns_v3_hard_mining_config.py`
- `scripts/evaluation/mine_pre_sns_v3_gt_iou_localization_cases.py`
- `scripts/evaluation/mine_pre_sns_v3_hard_cases.py`
- `scripts/inference/run_pre_sns_v3_report.py`
- `scripts/inference/run_pre_sns_v3_dual_scale_report.py`
- `src/cv_forensics/pre_sns_v3_gt_iou_mining.py`
- `src/cv_forensics/pre_sns_v3_hard_mining.py`
- `src/cv_forensics/pre_sns_v3_dual_scale_report.py`
- `src/cv_forensics/pre_sns_v3_report.py`
- `src/cv_forensics/pre_sns_v3_model.py`
- `src/cv_forensics/pre_sns_v3_metrics.py`
- `src/cv_forensics/pre_sns_manifest.py`
- `src/cv_forensics/pre_sns_visualization.py`
- `src/cv_forensics/__init__.py`
- `tests/test_pre_sns_v3_gt_iou_localization_mining.py`
- `tests/test_pre_sns_v3_dual_scale_report_and_mining.py`
- `tests/test_pre_sns_clean_red_overlay.py`
- `docs/pre_sns_v3_gt_iou_localization_mining.md`
- `docs/pre_sns_v3_dual_scale_report_and_mining.md`
- `docs/pre_sns_clean_red_overlay.md`

## Files Codex May Modify

- `tasks/0039-high-resolution-crop-tile-localization-model.md`
- `configs/training/pre_sns_v3_tile_localization.example.json`
- `scripts/agent/validate_pre_sns_v3_tile_localization_config.py`
- `scripts/training/prepare_pre_sns_v3_tile_localization.py`
- `scripts/evaluation/evaluate_pre_sns_v3_tile_localization.py`
- `src/cv_forensics/pre_sns_v3_tile_localization.py`
- `tests/test_pre_sns_v3_tile_localization.py`
- `docs/pre_sns_v3_tile_localization.md`
- `src/cv_forensics/__init__.py`

## Forbidden Actions

- Do not add SNS augmentation.
- Do not execute SNS augmentation.
- Do not train a real model in this task.
- Do not write checkpoints in this task.
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
- Clean pre-SNS v3 long256 remains the primary detector.
- The tile/crop localization path is a conditional localization improvement path only.
- Tile/crop localization must activate only when the long256 class is `tampered`, the long256 report is tampered-suspect, or the tampered score crosses a configured threshold.
- Task 0038 produces GT-IoU localization mining outputs that identify failed mask patterns. Those outputs are analysis inputs only.
- Real training, checkpoint writing, dataset download, package installation, network access, and protected path access are outside this task.
- All output roots must be approved external roots outside the repository.

## Implementation Requirements

### 1. Config Validator

Create `scripts/agent/validate_pre_sns_v3_tile_localization_config.py`.

Create `configs/training/pre_sns_v3_tile_localization.example.json`.

Approved config rules:

- `config_kind` must be `approved_pre_sns_v3_tile_localization`.
- `execution_mode` must be `approved_local_pre_sns_v3_tile_localization`.
- `no_download` must be `true`.
- `no_network` must be `true`.
- `no_sns_augmentation` must be `true`.
- `no_training` must be `true` for this task.
- `dry_run_only` must be `true` for this task.
- `no_checkpoint_writes` must be `true` for this task.
- `manifest_path` and optional `gt_iou_mining_records_path` must be under approved input roots outside the repository.
- `long256_checkpoint_path` must be under approved input roots or approved checkpoint roots outside the repository.
- `output_root` must be outside the repository and must not be under approved input roots.
- `max_samples` must be a positive integer.
- `tile_size`, `tile_stride`, `crop_context_px`, and high-resolution limits must be positive integers where applicable.
- Protected repo paths, URLs, path traversal, repo-local outputs, missing guardrails, training flags, SNS flags, and checkpoint-writing flags must be rejected.

### 2. Crop/Tile Localization Module

Create `src/cv_forensics/pre_sns_v3_tile_localization.py`.

It must provide reusable, testable functions for:

- Loading and validating tile-localization configs.
- Reading task 0038 GT-IoU mining records as optional analysis hints.
- Selecting tampered or tampered-suspect samples for tile localization.
- Computing tile grids over high-resolution images.
- Expanding GT-IoU failure boxes or mask-derived regions with configurable context.
- Merging tile-level mask predictions back into full-image coordinates.
- Comparing baseline long256 masks against tile-localized masks.
- Computing IoU, Dice, area, component count, and largest-component metrics.
- Building JSON summaries and optional visual comparison sheets under approved external output roots.

The implementation must keep pure tiling, merging, metric, and gating functions testable without CUDA, real checkpoints, or real datasets.

### 3. Dry-Run-Safe Training Preparation Entrypoint

Create `scripts/training/prepare_pre_sns_v3_tile_localization.py`.

The script must:

- Load and validate the config.
- Refuse real training when `dry_run_only` is true.
- Build a dry-run training plan from manifest metadata and optional 0038 mining outputs.
- Report selected sample counts, failure buckets used, tile counts, estimated pixel coverage, and intended external roots.
- Write only small JSON planning artifacts under approved external `output_root`.
- Not train, not write checkpoints, and not access protected paths.

Required dry-run output files:

- `tile_training_plan_summary.json`
- `tile_training_sample_plan.json`

### 4. Evaluation Entrypoint

Create `scripts/evaluation/evaluate_pre_sns_v3_tile_localization.py`.

The script must:

- Load and validate the config.
- Run a guarded evaluation comparing baseline long256 localization against tile-localized masks.
- Activate tile localization only for tampered or tampered-suspect samples.
- Support a dry-run or fixture mode that can run without real checkpoints.
- Write only under approved external `output_root`.

Required evaluation output files:

- `tile_localization_records.jsonl`
- `tile_vs_long256_summary.json`
- `tile_failure_bucket_summary.json`
- `tile_localization_artifact_manifest.json`

### 5. Tile Activation and Merge Rules

Implement deterministic activation and merge logic:

- If long256 class is `tampered`, tile localization may activate.
- If long256 class is non-tampered but tampered score is above a configured suspect threshold, tile localization may activate and must mark uncertainty.
- If long256 class is non-tampered and tampered score is below threshold, tile localization must skip.
- Tile masks must be merged into full-image coordinates by max/union or weighted averaging according to config.
- Tile localization must never change the primary class label by itself.
- The output must identify whether localization improved, regressed, or was unchanged compared with long256 baseline.

### 6. JSON Summary Requirements

Summaries must include:

- marker `PRE_SNS_V3_TILE_LOCALIZATION_OK`
- config guardrail flags
- tile configuration
- activation counts
- skipped counts
- baseline long256 mean/median IoU and Dice where GT masks exist
- tile-localized mean/median IoU and Dice where GT masks exist
- improvement/regression counts
- component statistics
- output paths
- explicit `no_training`, `no_checkpoint_writes`, and `no_sns_augmentation` flags

### 7. Optional Visual Comparison Sheets

Support optional visual comparison sheets under approved external `output_root`.

Sheets should include:

- input image
- GT red overlay if available
- baseline long256 red overlay
- tile-localized red overlay
- agreement or difference overlay

Clean red overlays must not contain text. Comparison sheets may contain labels only outside clean overlay panels.

### 8. Tests

Create `tests/test_pre_sns_v3_tile_localization.py`.

The test file must be plain `python3` runnable and must not require pytest as the test runner.

Cover:

- config validator guardrails
- tile grid generation
- crop context clipping at image boundaries
- tile mask merge behavior
- tile activation gating
- baseline vs tile metric comparison
- dry-run training plan schema
- evaluation summary schema
- no repository writes

Use tiny in-memory fixtures and temporary directories outside the repository. Do not require third-party downloads or real checkpoints.

### 9. Docs

Create `docs/pre_sns_v3_tile_localization.md`.

The document must:

- Include marker `PRE_SNS_V3_TILE_LOCALIZATION_OK`.
- Explain that this is pre-SNS only.
- Explain why task 0038 GT-IoU mining informs this task.
- Explain that long256 remains the primary detector.
- Explain that crop/tile localization is conditional and cannot change the primary class.
- Document config fields.
- Document dry-run training preparation outputs.
- Document evaluation outputs.
- State that this task does not add SNS augmentation, train a real model, write checkpoints, download datasets, use network resources, or write outputs into the repository.

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_v3_tile_localization_config.py configs/training/pre_sns_v3_tile_localization.example.json
```

```bash
python3 tests/test_pre_sns_v3_tile_localization.py
```

```bash
grep -q PRE_SNS_V3_TILE_LOCALIZATION_OK docs/pre_sns_v3_tile_localization.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0039-high-resolution-crop-tile-localization-model.md
```

## Acceptance Criteria

- The task implements a pre-SNS crop/tile localization path without SNS augmentation.
- Long256 remains the primary detector.
- Tile localization activates only for tampered or tampered-suspect cases.
- Tile localization does not change the primary class label.
- The dry-run-safe training preparation entrypoint writes small planning JSON files only under approved external output roots.
- The evaluation entrypoint compares baseline long256 masks with tile-localized masks.
- JSON summaries include IoU, Dice, area, component, activation, improvement, and regression metrics.
- Optional visual comparison sheets are written only under approved external output roots.
- Clean red overlays contain no text.
- Config validator accepts the example config and rejects unsafe configs.
- The plain Python test passes.
- The docs marker check passes.
- `check_agent_changes.py` passes for this task file.
- No protected paths, data, datasets, outputs, checkpoints, secrets, network, package installation, real training, checkpoint writing, or SNS augmentation are touched.

## Stop Condition

After creating this task file, stop and report:

- task file path
- short summary
- git status
- exact `git add` command for the task file
- exact `git commit` command for the task file

Do not implement until the task file is committed and the user explicitly says `implement`.
