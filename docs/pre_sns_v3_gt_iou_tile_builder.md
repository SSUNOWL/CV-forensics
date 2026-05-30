# Pre-SNS v3 GT-IoU Tile Builder

Marker: PRE_SNS_V3_GT_IOU_TILE_BUILDER_OK

This task is pre-SNS analysis and tile-manifest building only. It is not training, not SNS augmentation, not dataset download, and not checkpoint writing.

## Purpose

Clean long256 remains the primary detector. This task mines real TRAIN tampered samples with GT masks and prepares a manifest-only 512 crop/tile dataset for a future lightweight high-resolution localizer.

The intended pipeline is:

```text
clean long256 global detector -> high-resolution tile/crop localizer
```

## Config

Required fields include:

- `config_kind`: `approved_pre_sns_v3_gt_iou_tile_builder`
- `execution_mode`: `approved_local_pre_sns_v3_gt_iou_tile_builder`
- `approved_real_data_access`: `true`
- `train_manifest_path`
- `long256_checkpoint_path`
- `approved_input_roots`
- optional `approved_checkpoint_roots`
- optional `gt_iou_records_path`
- optional `hard_negative_records_path`
- `output_root`
- `max_samples`
- `tile_size`
- guardrails: `no_download`, `no_network`, `no_training`, `no_sns_augmentation`

`output_root` must be outside the repository. Validation hard cases must not be used as training tile records.

## Outputs

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

## Mining Buckets

- `severe_iou_fail`: IoU < 0.05
- `low_iou`: IoU < 0.15
- `weak_iou`: IoU < 0.25
- `good_iou`: IoU >= 0.40
- `empty_prediction`
- `undersegmented`
- `oversegmented`
- `wrong_region`
- `tiny_gt_mask`
- `fragmented_prediction`

## Tile Manifest

Positive records are GT-centered or jittered GT-centered crops with:

- `source_image_path`
- `source_mask_path`
- `crop_box`
- `tile_class`: `positive_tampered`
- `mining_bucket`
- `gt_area_pct_in_crop`
- `expected_mask_type`: `cropped_gt_mask`

Negative records use real/full_synthetic random crops or hard-negative false-positive crops with empty masks. Cropped image files are not written by default.
