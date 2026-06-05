# SNSAug V2 Fixed Pairs Eval

`SNSAUG_V2_FIXED_PAIRS_EVAL_OK`

`SNSAUG_V2_PRED_REDMASK_EXPORT_OK`

This workflow is evaluation only. It does not train, fine-tune, run `0054`, modify checkpoints, download assets, or use network resources.

## Input Pair Dataset

The evaluator expects an SNSAug v2 fixed-pair root containing:

- `meta.jsonl`
- `pair_index.json`
- `images/`
- `tamper_masks/`
- `ignore_masks/`
- `debug_overlays/`
- `artifact_manifest.json`

Rows with `view == "clean"` define the clean baseline. Rows with `view == "sns_aug"` and a non-clean `profile` are evaluated as SNSAug variants. Duplicate `sns_aug/profile=clean` rows are ignored with a warning.

## Clean vs SNS Comparison

Records are grouped by `base_id`. Each SNSAug row is joined with its clean counterpart to compute class flips, confidence drops, localization drops, and fragile-case flags.

Supported profiles:

- `clean`
- `tiktok_like`
- `instagram_story_like`
- `youtube_shorts_like`
- `news_meme_overlay`
- `combined_sns_realistic`

## Raw IoU vs Valid IoU

For tampered samples:

- `raw_iou` and `raw_dice` are computed over the full image
- `valid_iou` and `valid_dice` exclude SNS overlay regions using `ignore_mask`

`valid_iou` is the primary localization robustness metric because benign overlay pixels should not count as tamper-localization errors.

## Visual Mask Export

For report-quality visual comparison, the evaluator can export per-record mask visualizations under the configured external `output_root`:

- `pred_masks/`
- `pred_red_overlays/`
- `gt_red_overlays/`
- `ignore_blue_overlays/`
- `overlap_overlays/`

`snsaug_v2_eval_records.jsonl` includes these fields:

- `pred_mask_path`
- `pred_red_overlay_path`
- `gt_red_overlay_path`
- `ignore_blue_overlay_path`
- `overlap_overlay_path`
- `pred_mask_available`
- `localization_activated`

When localization is activated, the evaluator writes the predicted binary mask and a red overlay. When `write_empty_pred_mask` is true, records without localization activation receive deterministic all-zero predicted masks and matching red overlays. GT red overlays, ignore-mask blue overlays, and tamper-ignore overlap overlays are written when the corresponding masks are available.

The valid-IoU computation is unchanged: prediction and GT masks are still evaluated after excluding `ignore_mask` pixels from the valid region.

## Metrics

Per profile:

- 3-way accuracy
- macro-F1
- confusion matrix
- real FPR
- synthetic recall
- tampered recall
- tampered raw mean/median IoU
- tampered valid mean/median IoU
- tampered valid mean Dice
- localization activation recall
- non-tampered high-mask rate
- mean latency and FPS when available

Drop metrics compare each SNS profile against clean:

- accuracy drop
- macro-F1 drop
- real FPR increase
- synthetic recall drop
- tampered recall drop
- valid IoU drop
- raw IoU drop
- localization activation recall drop
- mean `p_tampered` drop on tampered samples

## Output Schema

The evaluator writes:

- `snsaug_v2_eval_records.jsonl`
- `snsaug_v2_eval_comparisons.jsonl`
- `snsaug_v2_per_profile_metrics.json`
- `snsaug_v2_robustness_drop_metrics.json`
- `snsaug_v2_eval_summary.json`
- `snsaug_v2_worst_samples.json`
- `snsaug_v2_fragile_candidates.jsonl`
- `visual_gallery_manifest.json`
- `artifact_manifest.json`

All outputs must be written outside the repository.
