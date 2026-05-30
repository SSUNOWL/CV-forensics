# Task 0038: Pre-SNS v3 GT-IoU Localization Mining

## Role

Codex-only task writer, implementation worker, strict self-reviewer, and limited repair manager.

Claude Code is unavailable for this task. Codex must implement directly only after this task file is committed and the user explicitly says `implement`.

## Goal

Clean long256 is the best current detector, but red-mask localization still fails on some tampered samples. Long224 and refinement are not reliable enough to replace long256. Add GT-IoU-based localization mining to identify exactly which tampered samples and mask patterns fail before building a new high-resolution or tile localization model.

This task is analysis/mining only. It must not train, add SNS augmentation, download datasets, use network resources, or install packages.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0038-pre-sns-v3-gt-iou-localization-mining.md`
- `tasks/0036-pre-sns-v3-dual-scale-calibrated-report-and-mining.md`
- `tasks/0037-pre-sns-v3-hard-negative-refinement.md`
- `configs/evaluation/pre_sns_v3_hard_mining.example.json`
- `configs/inference/pre_sns_v3_dual_report.example.json`
- `scripts/agent/check_agent_changes.py`
- `scripts/agent/validate_pre_sns_v3_hard_mining_config.py`
- `scripts/agent/validate_pre_sns_v3_dual_report_config.py`
- `scripts/evaluation/mine_pre_sns_v3_hard_cases.py`
- `scripts/inference/run_pre_sns_v3_dual_scale_report.py`
- `scripts/inference/run_pre_sns_v3_report.py`
- `src/cv_forensics/pre_sns_v3_hard_mining.py`
- `src/cv_forensics/pre_sns_v3_dual_scale_report.py`
- `src/cv_forensics/pre_sns_v3_report.py`
- `src/cv_forensics/pre_sns_v3_model.py`
- `src/cv_forensics/pre_sns_v3_metrics.py`
- `src/cv_forensics/pre_sns_manifest.py`
- `src/cv_forensics/pre_sns_visualization.py`
- `src/cv_forensics/__init__.py`
- `tests/test_pre_sns_v3_dual_scale_report_and_mining.py`
- `tests/test_pre_sns_clean_red_overlay.py`
- `docs/pre_sns_v3_dual_scale_report_and_mining.md`
- `docs/pre_sns_clean_red_overlay.md`

## Files Codex May Modify

- `tasks/0038-pre-sns-v3-gt-iou-localization-mining.md`
- `configs/evaluation/pre_sns_v3_gt_iou_localization_mining.example.json`
- `scripts/agent/validate_pre_sns_v3_gt_iou_localization_mining_config.py`
- `scripts/evaluation/mine_pre_sns_v3_gt_iou_localization_cases.py`
- `src/cv_forensics/pre_sns_v3_gt_iou_mining.py`
- `tests/test_pre_sns_v3_gt_iou_localization_mining.py`
- `docs/pre_sns_v3_gt_iou_localization_mining.md`
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
- Do not write mining outputs, visual sheets, checkpoints, or generated artifacts into the repository.
- Do not modify training data.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not use `rg`.
- Do not create large files.
- Do not commit unless the user explicitly says `commit this`.

## Important Project Facts

- The project is a lightweight multi-head image forensics prototype for social-media-robust image forensics.
- Target outputs are `Class / Mask / Family / Reason`.
- This task remains pre-SNS and must not add SNS augmentation.
- Clean pre-SNS v3 long256 is the primary detector because current evidence shows it is the best classification and tampered detector.
- Long224 and refined checkpoints are optional comparison models only; they must not replace long256 in this task.
- The purpose is localization failure analysis against ground-truth masks, not model improvement.
- v3 checkpoints produce class logits, tamper binary logits, family logits, and localization logits.
- Mining must focus on tampered samples with ground-truth masks.
- Output roots must be approved external roots outside the repository.

## Implementation Requirements

### 1. Localization IoU Miner Entrypoint

Create `scripts/evaluation/mine_pre_sns_v3_gt_iou_localization_cases.py`.

The script must:

- Load a JSON config.
- Validate the config before mining.
- Use `manifest_path`.
- Use required `long256_checkpoint_path`.
- Use optional `long224_checkpoint_path`.
- Use optional `refined_checkpoint_path`.
- Use `output_root`.
- Use positive integer `max_samples`.
- Use `threshold_tau`.
- Use per-model `max_image_size` settings.
- Run only on local files under approved input/checkpoint roots.
- Write only under an approved external output root outside the repository.
- Refuse unsafe configs before model loading or output writing.

### 2. Localization Mining Library

Create `src/cv_forensics/pre_sns_v3_gt_iou_mining.py`.

It must provide reusable, testable functions for:

- Loading configs.
- Validating configs and guardrails.
- Loading manifests.
- Selecting only tampered samples with ground-truth masks.
- Running long256 reports.
- Optionally running long224 and refined reports if checkpoints are provided.
- Collecting predicted masks.
- Computing per-sample localization metrics.
- Classifying failure types.
- Building bucketed JSON outputs.
- Writing optional visual sheets under the approved external output root.

Reuse existing v3 report, model, mask, visualization, and validation helpers where practical. Keep pure metric and bucket logic testable without CUDA or real checkpoints.

### 3. Per-Sample Metrics

For every tampered sample with a ground-truth mask, compute and write a JSONL record containing at least:

- `sample_id`
- `image_path`
- `mask_path` or equivalent ground-truth mask identifier when present
- per-model prediction availability
- `iou`
- `dice`
- `gt_area_pct`
- `pred_area_pct`
- `gt_component_count`
- `pred_component_count`
- `largest_gt_component_area_pct`
- `largest_pred_component_area_pct`
- `mask_area_ratio_pred_over_gt`
- `failure_types`

Metrics must be computed for long256 and for optional long224/refined models when present.

### 4. Failure Type Classification

Classify records into these failure types where applicable:

- `empty_prediction`
- `low_iou_wrong_region`
- `undersegmented`
- `oversegmented`
- `tiny_gt_mask`
- `fragmented_prediction`
- `all_models_failed`
- `long256_failed_long224_succeeded`
- `long256_succeeded`

The classification logic must be deterministic and covered by tests. It should use configurable thresholds where needed, with safe defaults.

### 5. Required Output JSON Files

Write these files under `output_root`:

- `localization_iou_records.jsonl`
- `low_iou_long256.json`
- `empty_prediction_tampered.json`
- `all_models_failed.json`
- `long256_failed_long224_succeeded.json`
- `tiny_gt_mask_cases.json`
- `oversegmented_cases.json`
- `undersegmented_cases.json`
- `model_iou_summary.json`
- `localization_mining_summary.json`

`localization_mining_summary.json` must include marker `PRE_SNS_V3_GT_IOU_LOCALIZATION_MINING_OK`, config guardrail flags, thresholds, counts per bucket, model availability, and output paths.

### 6. Optional Visual Output

Support optional visual sheets for top N cases per bucket under the approved external output root.

Visual sheets should include:

- input image
- GT red overlay
- long256 red overlay
- long224 red overlay if available
- refined red overlay if available
- GT/pred agreement overlay

Clean red overlays must not contain text. Comparison sheets may contain labels only outside clean overlay panels.

### 7. Config Validator

Create `configs/evaluation/pre_sns_v3_gt_iou_localization_mining.example.json`.

Create `scripts/agent/validate_pre_sns_v3_gt_iou_localization_mining_config.py`.

Approved config rules:

- `config_kind` must be `approved_pre_sns_v3_gt_iou_localization_mining` for real execution.
- `execution_mode` must be `approved_local_pre_sns_v3_gt_iou_localization_mining` for real execution.
- Example configs may use a clearly non-executing symbolic mode if consistent with existing validators.
- `no_download` must be `true`.
- `no_network` must be `true`.
- `no_sns_augmentation` must be `true`.
- `no_training` must be `true`.
- `manifest_path` must be under approved input roots outside the repository.
- checkpoint paths must be under approved input roots or approved checkpoint roots outside the repository.
- `output_root` must be outside the repository.
- `max_samples` must be a positive integer.
- Protected repo paths must be rejected.
- URLs, path traversal, repo-local outputs, missing guardrails, training/SNS flags, and package/network assumptions must be rejected.

### 8. Tests

Create `tests/test_pre_sns_v3_gt_iou_localization_mining.py`.

The test file must be plain `python3` runnable and must not require pytest as the test runner.

Cover:

- IoU and Dice calculations.
- Empty mask handling.
- Oversegmented and undersegmented classification.
- `all_models_failed` bucket.
- `long256_failed_long224_succeeded` bucket.
- Validator guardrails.
- No repository writes.

Use temporary directories outside the repository for write tests. Keep fixtures tiny and standard-library-only unless existing project code already requires a local dependency.

### 9. Docs

Create `docs/pre_sns_v3_gt_iou_localization_mining.md`.

The document must:

- Include marker `PRE_SNS_V3_GT_IOU_LOCALIZATION_MINING_OK`.
- Explain that this is pre-SNS analysis only.
- Explain that this miner identifies GT-IoU localization failure cases before high-resolution or tile localization training.
- Document required inputs.
- Document required output JSON files.
- Document failure type definitions.
- Document visual output behavior and the rule that clean red overlays contain no text.
- State that the task does not train, add SNS augmentation, download datasets, use network resources, or write outputs into the repository.

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_v3_gt_iou_localization_mining_config.py configs/evaluation/pre_sns_v3_gt_iou_localization_mining.example.json
```

```bash
python3 tests/test_pre_sns_v3_gt_iou_localization_mining.py
```

```bash
grep -q PRE_SNS_V3_GT_IOU_LOCALIZATION_MINING_OK docs/pre_sns_v3_gt_iou_localization_mining.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0038-pre-sns-v3-gt-iou-localization-mining.md
```

## Acceptance Criteria

- The task implements an analysis/mining-only GT-IoU localization miner.
- The miner processes tampered samples with ground-truth masks and writes `localization_iou_records.jsonl`.
- Long256 is always supported and is treated as the primary model.
- Long224 and refined checkpoints are optional comparison models.
- All required metrics are computed per applicable model.
- All required failure buckets are deterministic and written to the required JSON files.
- `model_iou_summary.json` summarizes per-model IoU/Dice and failure counts.
- `localization_mining_summary.json` includes the required marker, counts, thresholds, guardrails, model availability, and output paths.
- Optional visual sheets are written only under the approved external output root.
- Clean red overlays contain no text.
- The config validator accepts the example config and rejects unsafe configs.
- The plain Python test passes.
- The docs marker check passes.
- `check_agent_changes.py` passes for this task file.
- No protected paths, data, datasets, outputs, checkpoints, secrets, network, package installation, model training, or SNS augmentation are touched.

## Stop Condition

After creating this task file, stop and report:

- task file path
- short summary
- git status
- exact `git add` command for the task file
- exact `git commit` command for the task file

Do not implement until the task file is committed and the user explicitly says `implement`.
