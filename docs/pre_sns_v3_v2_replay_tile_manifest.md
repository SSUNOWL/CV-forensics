# Pre-SNS v2 Replay Tile Manifest

Marker: PRE_SNS_V3_V2_REPLAY_TILE_MANIFEST_OK

This task is pre-SNS manifest building and medium-training config preparation only. It is not training, not SNS augmentation, not dataset download, and not checkpoint writing.

## Purpose

The replay manifest is built from train-set `0042` tile records only. Validation audit failure cases from `0048` are diagnostic signals only and must not be used directly as replay training inputs.

The replay path emphasizes severe and low-IoU tampered localization failures, wrong-region and empty-prediction style failures, and hard negatives that help reduce false activation.

## Leakage Guard

Optional `0048` policy replay case files may be provided only for exclusion and diagnostics:

- `policy_still_failed_cases.json`
- `policy_worse_cases.json`
- `policy_non_tampered_high_mask_cases.json`
- `policy_fixed_cases.json`
- `policy_better_cases.json`

Sample IDs from these files are treated as validation audit IDs. Matching tile records are excluded from the replay manifest and written to `excluded_validation_case_ids.json`.

## Outputs

Under external `output_root`, the builder writes:

- `replay_tile_manifest.json`
- `replay_tile_manifest_summary.json`
- `excluded_validation_case_ids.json`
- `recommended_v2_medium_train_config.json`
- `artifact_manifest.json`

## Replay Weighting

Replay weighting is explicit and materialized into the replay manifest records. The builder increases replay frequency for:

- `severe_iou_fail`
- `low_iou`
- `weak_iou`
- `empty_prediction`
- `wrong_region`
- `undersegmented`
- `oversegmented`
- `hard_negative`
- `negative_real`
- `negative_synthetic`

This keeps the medium replay path focused on train-set hard localization behavior without leaking validation audit images into training.
