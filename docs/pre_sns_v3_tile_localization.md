# Pre-SNS v3 Tile Localization

Marker: PRE_SNS_V3_TILE_LOCALIZATION_OK

This task is pre-SNS only. It does not add SNS augmentation, train a real model, write checkpoints, download datasets, use network resources, install packages, or write outputs into the repository.

## Purpose

Task 0038 mines GT-IoU localization failures for tampered samples. This task uses those failure records as analysis hints for a high-resolution crop/tile localization path. The goal is to improve mask localization patterns before any SNS robustness evaluation.

## Detector Boundary

Clean long256 remains the primary detector. The crop/tile localization path is conditional and cannot change the primary class label. It may activate when long256 predicts `tampered`, or when a non-tampered class has a tampered score above the configured suspect threshold. Low-score non-tampered samples are skipped.

## Config

The config requires:

- `config_kind`: `approved_pre_sns_v3_tile_localization`
- `execution_mode`: `approved_local_pre_sns_v3_tile_localization`
- `manifest_path`
- optional `gt_iou_mining_records_path`
- `long256_checkpoint_path`
- `approved_input_roots`
- optional `approved_checkpoint_roots`
- `output_root`
- `max_samples`
- `tile_size`
- `tile_stride`
- `crop_context_px`
- `high_res_max_size`
- `merge_mode`
- `tampered_suspect_threshold`
- guardrails: `no_download`, `no_network`, `no_training`, `dry_run_only`, `no_checkpoint_writes`, `no_sns_augmentation`

All input and checkpoint paths must be under approved external roots. `output_root` must be outside the repository and outside approved input roots.

## Dry-Run Training Preparation

`scripts/training/prepare_pre_sns_v3_tile_localization.py` builds a dry-run plan only. It writes:

- `tile_training_plan_summary.json`
- `tile_training_sample_plan.json`

The summary includes selected sample count, total tile count, failure bucket counts from 0038 records, tile configuration, and guardrail flags.

## Evaluation

`scripts/evaluation/evaluate_pre_sns_v3_tile_localization.py` compares baseline long256 masks with tile-localized masks. In this task it supports fixture/dry-run-safe evaluation so tests do not need real checkpoints.

It writes:

- `tile_localization_records.jsonl`
- `tile_vs_long256_summary.json`
- `tile_failure_bucket_summary.json`
- `tile_localization_artifact_manifest.json`

Summaries include activation counts, skipped counts, baseline long256 IoU/Dice, tile-localized IoU/Dice, improvement/regression counts, component statistics, output paths, and guardrail flags.

## Visuals

Optional visual comparison sheets must be written only under the approved external `output_root`. Clean red overlay panels must not contain text. Comparison sheets may use labels outside clean overlay panels.
