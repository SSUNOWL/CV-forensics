# Task 0049: Pre-SNS v2 Replay Medium Training

## Task Title

Build a guarded replay tile-manifest and recommended medium training config for tile localizer v2 using train-set 0042 tile records, with explicit exclusion of 0048 validation audit case IDs and no validation hard-case leakage.

## Role

Codex-only implementation worker, reviewer, and limited repair manager operating under the repository contract and Codex-only workflow.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0049-pre-sns-v2-replay-medium-training.md`
- `tasks/0048-pre-sns-v2-policy-gated-integrated-report.md`
- `tasks/0046-pre-sns-highres-forensic-tile-localizer-v2.md`
- `tasks/0042-pre-sns-real-gt-iou-mining-and-tile-dataset-builder.md`
- `configs/evaluation/pre_sns_v3_v2_replay_tile_manifest.example.json`
- `scripts/agent/validate_pre_sns_v3_v2_replay_tile_manifest_config.py`
- `scripts/evaluation/build_pre_sns_v3_v2_replay_tile_manifest.py`
- `src/cv_forensics/pre_sns_v3_v2_replay_tile_manifest.py`
- `tests/test_pre_sns_v3_v2_replay_tile_manifest.py`
- `docs/pre_sns_v3_v2_replay_tile_manifest.md`
- `src/cv_forensics/__init__.py`
- Relevant existing manifest-builder and v2/reporting/training files needed to mirror established local patterns, including:
  - `src/cv_forensics/pre_sns_v3_gt_iou_tile_builder.py`
  - `src/cv_forensics/pre_sns_v3_tile_localizer_v2_training.py`
  - `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
  - `scripts/evaluation/build_pre_sns_v3_gt_iou_tile_dataset.py`
  - `docs/pre_sns_v3_gt_iou_tile_builder.md`
  - `docs/pre_sns_v3_tile_localizer_v2_training.md`
  - `tests/test_pre_sns_v3_gt_iou_tile_builder.py`
  - `tests/test_pre_sns_v3_v2_policy_gated_report.py`

## Files Codex May Modify

- `tasks/0049-pre-sns-v2-replay-medium-training.md`
- `configs/evaluation/pre_sns_v3_v2_replay_tile_manifest.example.json`
- `scripts/agent/validate_pre_sns_v3_v2_replay_tile_manifest_config.py`
- `scripts/evaluation/build_pre_sns_v3_v2_replay_tile_manifest.py`
- `src/cv_forensics/pre_sns_v3_v2_replay_tile_manifest.py`
- `tests/test_pre_sns_v3_v2_replay_tile_manifest.py`
- `docs/pre_sns_v3_v2_replay_tile_manifest.md`
- `src/cv_forensics/__init__.py`

## Forbidden Actions

- Do not modify any file outside the allowed modify list.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not add SNS augmentation.
- Do not train models.
- Do not train directly on validation hard cases or validation replay case IDs.
- Do not download datasets.
- Do not use network resources.
- Do not install packages.
- Do not write outputs or checkpoints inside the repository.
- Do not create large files.
- Do not commit changes.

## Important Project Facts

- This is a pre-SNS manifest-building and training-config generation task only.
- `0048` validation audit failures are diagnostic signals only and must not be used as direct replay training inputs.
- Replay training must be built from train-set 0042 tile manifest records.
- The goal is to emphasize hard tampered localization failures and hard negatives while guarding against validation leakage.
- The resulting recommended config is for a future `0046` v2 medium replay run, but this task must not launch training.

## Implementation Requirements

### 1. Replay Tile Manifest Builder

- Add module:
  - `src/cv_forensics/pre_sns_v3_v2_replay_tile_manifest.py`
- Add script:
  - `scripts/evaluation/build_pre_sns_v3_v2_replay_tile_manifest.py`
- Input support:
  - `base_tile_manifest_path` from `0042`
  - optional `0048` policy replay summary/files for exclusion and diagnostics
  - `output_root`
- Output files under external `output_root`:
  - `replay_tile_manifest.json`
  - `replay_tile_manifest_summary.json`
  - `excluded_validation_case_ids.json`
  - `artifact_manifest.json`

### 2. Replay Weighting

- Build replay records from train tile records only.
- Apply replay weights:
  - `severe_iou_fail`: `12`
  - `low_iou`: `10`
  - `weak_iou`: `5`
  - `empty_prediction` / `wrong_region`: `12`
  - `undersegmented` / `oversegmented`: `4`
  - normal `positive_tampered`: `1` to `2`
  - `hard_negative`: `8`
  - `negative_real` / `negative_synthetic`: `3`
  - `good_iou`: `1`
- Replay weighting must be explicit and recorded in manifest summary metadata.

### 3. Leakage Guard

- Load sample IDs from optional `0048` policy replay files when provided:
  - `policy_still_failed_cases.json`
  - `policy_worse_cases.json`
  - `policy_non_tampered_high_mask_cases.json`
  - `policy_fixed_cases.json`
  - `policy_better_cases.json`
- Treat these IDs as validation audit IDs only.
- Exclude matching IDs from replay training records if encountered in the base tile manifest.
- Write the final excluded ID set to `excluded_validation_case_ids.json`.
- Document counts for excluded IDs and retained replay records.

### 4. Recommended Medium v2 Training Config

- Generate `recommended_v2_medium_train_config.json` under external `output_root`.
- The generated config must use:
  - `config_kind`: `approved_pre_sns_v3_tile_localizer_v2_training`
  - `execution_mode`: `approved_local_pre_sns_v3_tile_localizer_v2_training`
  - `tile_manifest_path`: generated `replay_tile_manifest.json`
  - `tile_size`: `768`
  - `max_tiles_train`: `24000`
  - `max_tiles_val`: `3000`
  - `epochs`: `60`
  - `batch_size`: `2`
  - `learning_rate`: `3e-5`
  - `input_feature_mode`: `rgb_edge_residual`
  - `empty_mask_loss_weight`: `3.0`
  - `false_activation_area_loss_weight`: `2.0`
  - `dice_loss_weight`: `2.0`
  - `tversky_loss_weight`: `1.0`
  - `progress_log_interval_steps`: `100`
  - `stdout_progress_interval_steps`: `100`
  - `progress_write_json`: `true`
  - `progress_write_jsonl`: `true`
  - `no_sns_augmentation`: `true`
  - `no_network`: `true`
  - `no_download`: `true`
- `approved_run_root` and `approved_checkpoint_root` must be external paths outside the repository.

### 5. Validator

- Add:
  - `configs/evaluation/pre_sns_v3_v2_replay_tile_manifest.example.json`
  - `scripts/agent/validate_pre_sns_v3_v2_replay_tile_manifest_config.py`
- Approved config rules:
  - `config_kind` must be `approved_pre_sns_v3_v2_replay_tile_manifest`
  - `execution_mode` must be `approved_local_pre_sns_v3_v2_replay_tile_manifest`
  - `no_download` must be `true`
  - `no_network` must be `true`
  - `no_sns_augmentation` must be `true`
  - `no_training` must be `true`
  - `output_root` must be outside the repository
  - approved input roots must be absolute external roots outside the repository
  - reject repo-local output roots, missing guardrails, path traversal, remote URLs, and protected path segments

### 6. Tests

- Add:
  - `tests/test_pre_sns_v3_v2_replay_tile_manifest.py`
- Cover:
  - replay weights
  - leakage exclusion
  - output manifest schema
  - recommended training config fields
  - validator rejects repo-local output
  - validator rejects SNS/network/download violations

### 7. Docs

- Add:
  - `docs/pre_sns_v3_v2_replay_tile_manifest.md`
- Include marker:
  - `PRE_SNS_V3_V2_REPLAY_TILE_MANIFEST_OK`
- Explain clearly that validation failure cases are diagnostic only and are not directly trained on.

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_v3_v2_replay_tile_manifest_config.py configs/evaluation/pre_sns_v3_v2_replay_tile_manifest.example.json
CUDA_VISIBLE_DEVICES="" python3 tests/test_pre_sns_v3_v2_replay_tile_manifest.py
grep -q PRE_SNS_V3_V2_REPLAY_TILE_MANIFEST_OK docs/pre_sns_v3_v2_replay_tile_manifest.md
python3 scripts/agent/check_agent_changes.py tasks/0049-pre-sns-v2-replay-medium-training.md
```

## Acceptance Criteria

- The replay builder creates a manifest-only replay dataset from `0042` train tile records without using validation audit images as direct training inputs.
- Replay weighting emphasizes severe and low-IoU tampered failures plus hard negatives and false-activation controls as specified.
- Validation audit IDs from optional `0048` case files are excluded from replay records and written to `excluded_validation_case_ids.json`.
- The generated recommended medium training config uses the required `0046` v2 training fields and enables progress logging.
- All outputs are written only under external approved roots outside the repository.
- Tests and validation commands pass without training, downloads, SNS augmentation, network use, or repository-local outputs/checkpoints.

## Stop Condition

- Stop after implementing only the allowed-file changes, running the validation commands, and running `python3 scripts/agent/check_agent_changes.py tasks/0049-pre-sns-v2-replay-medium-training.md`.
- Review the result in Korean and report `PASS` or `NEEDS_FIX`.
- Do not commit; wait for the user to review and commit manually.
