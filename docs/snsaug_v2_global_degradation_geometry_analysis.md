# SNSAug V2 Global Degradation Geometry Analysis

SNSAUG_V2_GLOBAL_DEGRADATION_GEOMETRY_ANALYSIS_OK

0068 tests the hypothesis that SNSAug failure is driven more by global degradation or geometry shift than by local overlay nuisance. It follows 0067a, where local `ignore_mask` pixel masking produced no promising policies.

The analysis is evaluation-only. It reads a fixed SNSAug pair root, joins clean/SNS pairs from `meta.jsonl`, computes image-level shift features, and optionally joins pre-SNS baseline response records. It does not train, fine-tune, download, use network access, or modify checkpoints.

## Profile Families

Profiles are grouped as:

- `clean`
- `local_overlay`: `platform_ui_same_size`, `news_meme_overlay`, `tiktok_like`, `instagram_story_like`, `youtube_shorts_like`, `combined_sns_realistic`
- `geometry`: `canvas_9x16_only`, `resize_crop_pad`, `zoom_crop`
- `postprocess`: `recompression_light`, `resize_jpeg`
- `screenshot`: `screenshot_recapture_light`

## Features

For each clean/SNS pair, the analyzer computes width, height, aspect ratio, area ratio, image mean/std, grayscale histogram L1 distance, edge energy, high-pass residual energy, Laplacian variance, JPEG-like blockiness, crop/scale proxy, ignore-mask area, tamper area, and tamper occlusion by ignore mask.

When baseline records are configured, the output also includes `p_tampered` drop, valid IoU drop, activation flip-off, and prediction flip. Correlations are computed between shift features and response collapse indicators.

## Outputs

Successful actual runs write:

- `global_degradation_geometry_records.jsonl`
- `profile_family_shift_summary.json`
- `feature_response_correlation.json`
- `global_degradation_geometry_report.md`
- `artifact_manifest.json`

If geometry/postprocess features correlate more strongly with response collapse than ignore-mask area, the report points toward global degradation/geometry correction. If ignore-mask area or tamper occlusion dominates, the report points back toward masked nuisance or masked-convolution work.
