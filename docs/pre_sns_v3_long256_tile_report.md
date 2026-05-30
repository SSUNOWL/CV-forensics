# Pre-SNS v3 Long256 Tile Report

Marker: PRE_SNS_V3_LONG256_TILE_REPORT_OK

This task is pre-SNS only. It does not add SNS augmentation, train, write checkpoints, download datasets, use network resources, install packages, or write outputs into the repository.

## Purpose

Task 0038 mined GT-IoU localization failures for tampered masks. Task 0039 prepared a dry-run-safe tile/crop localization workflow. This task integrates those directions into a report path where clean long256 remains the primary detector and tile localization is used only as a conditional mask refinement path.

## Flow

1. Clean long256 provides class, family, tampered score, and baseline mask evidence.
2. Tile localization activates only when long256 predicts `tampered` or the tampered score crosses the suspect threshold.
3. Tile masks are merged back into full-image coordinates.
4. The final mask is selected from tile-localized output when usable, otherwise from baseline long256.
5. The primary class is never changed by tile localization.
6. A template reason summarizes class, tile status, mask source, and localization delta.

Full-image 384/512 remains a plausible future experiment. This task focuses on tile/crop integration because small red-mask failures are likely caused by full-image resizing.

## Config

The config requires:

- `config_kind`: `approved_pre_sns_v3_long256_tile_report`
- `execution_mode`: `approved_local_pre_sns_v3_long256_tile_report`
- `manifest_path` or `image_path`
- optional `gt_iou_mining_records_path`
- `long256_checkpoint_path`
- `approved_input_roots`
- optional `approved_checkpoint_roots`
- `output_root`
- `max_samples` for manifest mode
- `tile_size`
- `tile_stride`
- `crop_context_px`
- `high_res_max_size`
- `merge_mode`
- `tampered_suspect_threshold`
- guardrails: `no_download`, `no_network`, `no_training`, `no_checkpoint_writes`, `no_sns_augmentation`

All inputs and checkpoints must be under approved external roots. `output_root` must be outside the repository and outside approved input roots.

## Outputs

Manifest mode writes:

- `long256_tile_integrated_records.jsonl`
- `long256_tile_integrated_summary.json`
- `long256_tile_artifact_manifest.json`

Single-image mode writes:

- `long256_tile_integrated_report.json`
- `long256_tile_integrated_summary.json`
- `long256_tile_artifact_manifest.json`

## Report Fields

Each record includes class, class confidence, family, family confidence, tampered score, tile localization status, baseline long256 mask stats, tile mask stats, final mask source, final mask stats, baseline-vs-tile metrics, localization delta, template reason, visual artifact metadata, and guardrail flags.

## Visuals

Optional visual comparison sheets must be written only under an approved external output root. Clean red overlay panels must not contain text. Comparison sheets may include labels outside clean overlay panels.
