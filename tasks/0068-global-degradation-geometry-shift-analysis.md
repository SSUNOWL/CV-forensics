# Task 0068: SNSAug v2 Global Degradation and Geometry Shift Analysis

## Context

0067a Masked Nuisance Inference successfully ran with artifact manifest and balanced records.

Observed 0067a result:

- record_count = 3600
- row_count = 600
- policies: original, gray_fill, blur_fill, mean_fill, black_fill, dilated_gray_fill
- profiles and labels are balanced
- promising_policies = []

Interpretation:

Local ignore_mask-based pixel masking does not sufficiently recover tampered recall or valid IoU. Therefore local SNS overlay is not the only or dominant cause.

0066 tendency analysis showed high difficulty for geometry and degradation profiles such as zoom_crop and resize_jpeg. The next step is to quantify global degradation and geometry shift.

## Goal

Analyze whether SNSAug failure is driven by global degradation or geometry/layout shift rather than local overlay nuisance.

## Files Codex May Modify

- `tasks/0068-global-degradation-geometry-shift-analysis.md`
- `src/cv_forensics/snsaug_v2_global_degradation_geometry_analysis.py`
- `scripts/evaluation/run_snsaug_v2_global_degradation_geometry_analysis.py`
- `scripts/agent/validate_snsaug_v2_global_degradation_geometry_analysis_config.py`
- `configs/evaluation/snsaug_v2_global_degradation_geometry_analysis.example.json`
- `tests/test_snsaug_v2_global_degradation_geometry_analysis.py`
- `docs/snsaug_v2_global_degradation_geometry_analysis.md`

## Files Claude May Modify

- `tasks/0068-global-degradation-geometry-shift-analysis.md`
- `src/cv_forensics/snsaug_v2_global_degradation_geometry_analysis.py`
- `scripts/evaluation/run_snsaug_v2_global_degradation_geometry_analysis.py`
- `scripts/agent/validate_snsaug_v2_global_degradation_geometry_analysis_config.py`
- `configs/evaluation/snsaug_v2_global_degradation_geometry_analysis.example.json`
- `tests/test_snsaug_v2_global_degradation_geometry_analysis.py`
- `docs/snsaug_v2_global_degradation_geometry_analysis.md`

## Required Behavior

1. No training.

- no_training = true
- no_finetune = true
- no_network = true
- no_download = true

2. Inputs.

Use:

- fixed SNSAug pair root from 0058c
- meta.jsonl
- images directory
- tamper_masks directory
- ignore_masks directory if available
- existing baseline/fixed-pair evaluation records if configured

3. Analyze profile families.

Group profiles into:

- clean
- local_overlay: platform_ui_same_size, news_meme_overlay, tiktok_like, instagram_story_like, youtube_shorts_like, combined_sns_realistic
- geometry: canvas_9x16_only, resize_crop_pad, zoom_crop
- postprocess: recompression_light, resize_jpeg
- screenshot: screenshot_recapture_light

4. Compute image-level shift features.

For each clean/SNS pair:

- width, height, aspect_ratio
- image mean/std
- grayscale histogram distance
- edge energy
- high-pass residual energy
- Laplacian variance
- JPEG-like blockiness score
- crop/scale proxy if dimensions or metadata allow
- ignore_mask_area if available
- tamper_area
- tamper_occluded_by_ignore ratio

5. Join with model response.

Join image shift features with pre-SNS baseline records:

- p_tampered_clean
- p_tampered_sns
- delta_p_tampered
- clean_valid_iou
- sns_valid_iou
- delta_valid_iou
- activation_flip_off
- pred_flip

6. Output.

Write:

- global_degradation_geometry_records.jsonl
- profile_family_shift_summary.json
- feature_response_correlation.json
- global_degradation_geometry_report.md
- artifact_manifest.json

7. Decision criteria.

If geometry/postprocess features correlate more strongly with p_tampered/IoU collapse than ignore_mask_area, conclude global degradation/geometry is dominant.

If ignore_mask_area or tamper_occlusion correlates most strongly, return to masked nuisance / masked convolution path.

8. Tests.

- feature extraction handles missing masks
- blockiness score finite
- high-pass residual finite
- clean/SNS pair join works
- correlation handles constant features
- artifact manifest written

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_global_degradation_geometry_analysis_config.py configs/evaluation/snsaug_v2_global_degradation_geometry_analysis.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_global_degradation_geometry_analysis.py
- python3 scripts/evaluation/run_snsaug_v2_global_degradation_geometry_analysis.py --help
- grep -q SNSAUG_V2_GLOBAL_DEGRADATION_GEOMETRY_ANALYSIS_OK docs/snsaug_v2_global_degradation_geometry_analysis.md
- python3 scripts/agent/check_agent_changes.py tasks/0068-global-degradation-geometry-shift-analysis.md

## Marker

SNSAUG_V2_GLOBAL_DEGRADATION_GEOMETRY_ANALYSIS_OK
