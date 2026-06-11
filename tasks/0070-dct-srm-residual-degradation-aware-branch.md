# Task 0070: DCT/SRM Residual Degradation-aware Analysis for SNSAug Failure

## Context

0067a Masked Nuisance Inference showed local ignore_mask pixel masking was not sufficient.

0068 Global Degradation and Geometry Shift Analysis showed global_degradation_geometry is dominant over local nuisance.

0069 Geometry/Degradation Recovery showed oracle geometry can recover all focus profiles, but deployable non-oracle policies are weak.

0069b Improved Geometry Estimation confirmed:

- actual run completed successfully
- record_count = 8400
- oracle_clean_geometry_diagnostic recovered all focus profiles
- deployable policies mostly helped only canvas_9x16_only
- decision = diagnostic_only_geometry_recovery
- next recommendation = move to residual/DCT branch if estimator remains weak

Interpretation:

Geometry is recoverable in principle, but image-only crop/content-box heuristics are not enough. The remaining failure likely involves low-level forensic trace disruption: blockiness, resampling, high-pass residual shift, JPEG/DCT artifact shift, edge/boundary degradation, and screenshot recapture artifacts.

## Goal

Implement an evaluation-first DCT/SRM/high-pass residual degradation analysis for SNSAug robustness failure.

This is not a long training task. First, diagnose whether residual/DCT/blockiness features explain p_tampered drop, valid_iou drop, activation flip-off, and prediction flip.

## Files Codex May Modify

- `tasks/0070-dct-srm-residual-degradation-aware-branch.md`
- `src/cv_forensics/snsaug_v2_residual_degradation_analysis.py`
- `scripts/evaluation/run_snsaug_v2_residual_degradation_analysis.py`
- `scripts/agent/validate_snsaug_v2_residual_degradation_analysis_config.py`
- `configs/evaluation/snsaug_v2_residual_degradation_analysis.example.json`
- `tests/test_snsaug_v2_residual_degradation_analysis.py`
- `docs/snsaug_v2_residual_degradation_analysis.md`

## Files Claude May Modify

- `tasks/0070-dct-srm-residual-degradation-aware-branch.md`
- `src/cv_forensics/snsaug_v2_residual_degradation_analysis.py`
- `scripts/evaluation/run_snsaug_v2_residual_degradation_analysis.py`
- `scripts/agent/validate_snsaug_v2_residual_degradation_analysis_config.py`
- `configs/evaluation/snsaug_v2_residual_degradation_analysis.example.json`
- `tests/test_snsaug_v2_residual_degradation_analysis.py`
- `docs/snsaug_v2_residual_degradation_analysis.md`

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
- pre-SNS baseline records from 0064g comparison or a filtered baseline records file
- optional 0068 output root
- optional 0069b output root

3. Feature extraction.

For each clean/SNS pair, compute:

- grayscale high-pass residual energy
- residual energy delta
- SRM-like fixed filter responses
- SRM residual mean/std/energy
- Laplacian variance
- Sobel edge energy
- edge energy delta
- JPEG-like blockiness score
- blockiness delta
- approximate 8x8 DCT energy summary
- low-frequency DCT energy
- high-frequency DCT energy
- high/low DCT ratio
- histogram L1 distance
- aspect/crop/scale proxy from metadata or image dimensions
- ignore_mask_area
- tamper_area
- tamper_occluded_by_ignore ratio

Use only standard libraries plus numpy/PIL/OpenCV if already available. Do not introduce heavy dependencies.

4. Join with model response.

Join features with pre-SNS baseline response records and compute:

- p_tampered_clean
- p_tampered_sns
- delta_p_tampered
- clean_valid_iou
- sns_valid_iou
- delta_valid_iou
- activation_flip_off
- pred_flip
- correct_to_wrong
- label-specific rows for real/synthetic/tampered

5. Profile family grouping.

Use:

- clean
- geometry: canvas_9x16_only, resize_crop_pad, zoom_crop
- postprocess: recompression_light, resize_jpeg
- screenshot: screenshot_recapture_light
- local_overlay: news_meme_overlay, platform_ui_same_size, tiktok_like, instagram_story_like, youtube_shorts_like, combined_sns_realistic

6. Correlation / ranking.

Compute robust correlations between feature deltas and response failures:

- abs Pearson correlation where valid
- Spearman approximation if simple
- point-biserial style correlation for binary targets
- per-label and per-profile-family summaries

Targets:

- delta_p_tampered
- delta_valid_iou
- activation_flip_off
- pred_flip
- correct_to_wrong

7. Output.

Write:

- residual_degradation_records.jsonl
- residual_feature_summary.json
- residual_feature_response_correlation.json
- residual_degradation_report.md
- artifact_manifest.json

8. Decision criteria.

If DCT/SRM/high-pass/blockiness features correlate strongly with response collapse, recommend 0071 lightweight residual branch training.

If geometry features dominate again, recommend one final geometry estimator refinement or geometry-normalized preprocessing.

If neither explains collapse, recommend revisiting model calibration / class-head robustness.

9. Tests.

Tests must cover:

- high-pass residual feature is finite
- SRM-like feature extraction is finite
- blockiness score is finite
- DCT energy summary is finite
- correlation handles constant values
- clean/SNS pair join works
- output artifact manifest is written
- dry-run writes no records

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_residual_degradation_analysis_config.py configs/evaluation/snsaug_v2_residual_degradation_analysis.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_residual_degradation_analysis.py
- python3 scripts/evaluation/run_snsaug_v2_residual_degradation_analysis.py --help
- grep -q SNSAUG_V2_RESIDUAL_DEGRADATION_ANALYSIS_OK docs/snsaug_v2_residual_degradation_analysis.md
- python3 scripts/agent/check_agent_changes.py tasks/0070-dct-srm-residual-degradation-aware-branch.md

## Marker

SNSAUG_V2_RESIDUAL_DEGRADATION_ANALYSIS_OK
