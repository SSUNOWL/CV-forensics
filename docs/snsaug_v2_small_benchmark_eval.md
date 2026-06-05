# SNSAug V2 Small Benchmark Eval

`SNSAUG_V2_SMALL_GEOMETRY_POSTPROCESS_BENCHMARK_OK`

This workflow is evaluation and generation only. It does not train, fine-tune, run `0054`, modify checkpoints, download assets, or use network resources.

## Scope

`0058` expands the fixed SNSAug validation benchmark beyond overlay layouts so the frozen pre-SNS bundle can be audited against:

- clean
- geometry/canvas perturbations
- compression/postprocess perturbations
- layout/overlay perturbations
- combined realistic SNS perturbations

## Required Profiles

- `clean`
- `resize_crop_pad`
- `zoom_crop`
- `recompression_light`
- `resize_jpeg`
- `screenshot_recapture_light`
- `canvas_9x16_only`
- `platform_ui_same_size`
- `news_meme_overlay`
- `tiktok_like`
- `instagram_story_like`
- `youtube_shorts_like`
- `combined_sns_realistic`

## Mask Policy

- `tamper_mask` tracks only malicious manipulation regions
- UI bars, frames, text, stickers, banners, and badges must go to `ignore_mask`
- geometric transforms apply to image and `tamper_mask` together
- recompression and color/blur postprocess apply to image only
- mask resizing must use nearest-neighbor

## Reports

The evaluator writes benchmark-oriented reports outside the repository:

- `small_benchmark_summary.md`
- `small_benchmark_metrics_table.csv`
- `small_benchmark_drop_table.csv`
- `small_benchmark_interpretation.json`
- `artifact_manifest.json`

The interpretation JSON identifies:

- main collapse cause
- profiles safe for training
- profiles too severe for early curriculum
- recommended SNSAug fine-tuning mix
