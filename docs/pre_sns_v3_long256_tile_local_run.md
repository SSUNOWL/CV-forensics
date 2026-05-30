# Pre-SNS v3 Long256 Tile Local Run

Marker: PRE_SNS_V3_LONG256_TILE_LOCAL_RUN_OK

This task is pre-SNS local evaluation/reporting only. It does not train, add SNS augmentation, write checkpoints, download datasets, use network resources, install packages, or write outputs into the repository.

## Purpose

Task 0038 mines GT-IoU localization failures. Task 0039 prepares the crop/tile localization workflow. Task 0040 integrates clean long256 and conditional tile localization into a report path. This task runs that integrated path on approved local samples/checkpoints and produces artifacts for checking whether the final red mask is visually and quantitatively usable.

## Config

The config requires:

- `config_kind`: `approved_pre_sns_v3_long256_tile_local_run`
- `execution_mode`: `approved_local_pre_sns_v3_long256_tile_local_run`
- `manifest_path` or `sample_list_path`
- optional `gt_iou_mining_records_path`
- `long256_checkpoint_path`
- `approved_input_roots`
- optional `approved_checkpoint_roots`
- `output_root`
- `max_samples`
- tile fields: `tile_size`, `tile_stride`, `crop_context_px`, `high_res_max_size`
- thresholds: `tampered_suspect_threshold`, `good_iou_threshold`, `good_dice_threshold`, `low_iou_threshold`, `overseg_ratio_threshold`, `underseg_ratio_threshold`
- `visual_top_n`
- guardrails: `no_download`, `no_network`, `no_training`, `no_checkpoint_writes`, `no_sns_augmentation`

All outputs must go under an approved external `output_root`.

## Outputs

- `local_run_records.jsonl`
- `local_run_summary.json`
- `good_red_mask_cases.json`
- `failed_red_mask_cases.json`
- `needs_manual_review_cases.json`
- `red_mask_gallery_manifest.json`
- `artifact_manifest.json`

## Red-Mask Buckets

- `good_red_mask_cases`: GT exists and final mask IoU/Dice meet thresholds, or no GT exists but final mask is non-empty and class/mask status is consistent.
- `failed_red_mask_cases`: empty final mask for tampered/suspect samples, low final IoU when GT exists, serious oversegmentation, or serious undersegmentation.
- `needs_manual_review_cases`: no GT and ambiguous class/mask/tile status.

## Recommendations

- `ready_for_sns_robustness_evaluation`: red masks are usable enough to proceed.
- `needs_real_tile_localization_training`: red-mask failures are frequent enough to justify explicit real tile/crop training.
- `needs_manual_review`: results are mixed or lack GT evidence and need human inspection.
