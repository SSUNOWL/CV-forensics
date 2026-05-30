# Pre-SNS v3 GT-IoU Localization Mining

Marker: PRE_SNS_V3_GT_IOU_LOCALIZATION_MINING_OK

This is a pre-SNS analysis-only miner. It does not train models, add SNS augmentation, download datasets, use network resources, install packages, write checkpoints, or write mining outputs into the repository.

## Purpose

Clean long256 is the current primary detector, but red-mask localization can still fail on tampered samples. This miner compares predicted localization masks against ground-truth masks so the project can identify failed tampered cases before designing a high-resolution or tile localization model.

## Inputs

- `manifest_path`: local manifest under an approved input root.
- `long256_checkpoint_path`: required primary v3 checkpoint.
- `long224_checkpoint_path`: optional comparison checkpoint.
- `refined_checkpoint_path`: optional comparison checkpoint.
- `output_root`: approved external output root outside the repository.
- `max_samples`: positive integer limit for tampered samples with GT masks.
- `threshold_tau`: tampered-score threshold used to decide whether a predicted mask is active.
- per-model max image-size settings for recording intended evaluation scale.

The config must set `no_download`, `no_network`, `no_training`, and `no_sns_augmentation` to `true`.

## Outputs

The miner writes these files under `output_root`:

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

Each JSONL record includes sample identity, image and mask paths, long256 top-level metrics, per-model metrics, component statistics, mask area ratio, and failure types.

## Failure Types

- `empty_prediction`: the model produces no active predicted mask.
- `low_iou_wrong_region`: predicted mask is non-empty but below the low-IoU threshold.
- `undersegmented`: predicted area is much smaller than GT area.
- `oversegmented`: predicted area is much larger than GT area.
- `tiny_gt_mask`: GT mask area is below the tiny-mask threshold.
- `fragmented_prediction`: predicted mask has many components.
- `all_models_failed`: all available models are below the success IoU threshold.
- `long256_failed_long224_succeeded`: long256 is below success IoU while long224 reaches it.
- `long256_succeeded`: long256 reaches the success IoU threshold.

## Visual Output

Optional visual sheets can be enabled with `visual_top_n_per_bucket`. They are written only under the approved external `output_root`. Clean red overlays must not contain text; comparison sheets may use labels outside clean overlay panels.

## Intended Use

Use this miner to inspect GT-IoU failure patterns, including tiny GT masks, empty predictions, oversegmentation, undersegmentation, and cross-model disagreements. The resulting buckets should inform a later high-resolution or tile localization training task, but this task itself is only mining and review.
