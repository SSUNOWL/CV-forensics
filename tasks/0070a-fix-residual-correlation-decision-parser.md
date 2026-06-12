# Task 0070a: Fix Residual/DCT Correlation Decision Parser

## Context

0070 residual degradation analysis executed successfully and wrote residual_feature_response_correlation.json.

However, the decision report produced:

- Decision = no_single_low_level_factor_dominant
- Top Correlations table empty

This appears to be a parser/schema bug. The correlation JSON contains nested entries like:

- dct_high_low_ratio_delta_abs -> activation_flip_off -> abs_pearson_r
- dct_low_energy_delta_abs -> correct_to_wrong -> abs_pearson_r

But the decision parser expected flat rows with an explicit feature key.

## Goal

Fix residual degradation decision reporting so nested correlation schemas are flattened correctly.

## Files Codex May Modify

- `tasks/0070a-fix-residual-correlation-decision-parser.md`
- `src/cv_forensics/snsaug_v2_residual_degradation_analysis.py`
- `scripts/evaluation/run_snsaug_v2_residual_degradation_analysis.py`
- `scripts/agent/validate_snsaug_v2_residual_degradation_analysis_config.py`
- `configs/evaluation/snsaug_v2_residual_degradation_analysis.example.json`
- `tests/test_snsaug_v2_residual_degradation_analysis.py`
- `docs/snsaug_v2_residual_degradation_analysis.md`

## Files Claude May Modify

- `tasks/0070a-fix-residual-correlation-decision-parser.md`
- `src/cv_forensics/snsaug_v2_residual_degradation_analysis.py`
- `scripts/evaluation/run_snsaug_v2_residual_degradation_analysis.py`
- `scripts/agent/validate_snsaug_v2_residual_degradation_analysis_config.py`
- `configs/evaluation/snsaug_v2_residual_degradation_analysis.example.json`
- `tests/test_snsaug_v2_residual_degradation_analysis.py`
- `docs/snsaug_v2_residual_degradation_analysis.md`

## Required Behavior

1. Support both correlation schemas.

Flat schema:
- {feature, target, abs_pearson_r, abs_spearman_r, ...}

Nested schema:
- {family: {feature: {target: {abs_pearson_r, abs_spearman_r, pair_count, ...}}}}
- {feature: {target: {abs_pearson_r, abs_spearman_r, pair_count, ...}}}

2. Decision report must never show an empty Top Correlations table when residual_feature_response_correlation.json contains valid correlation values.

3. Output top correlations with:

- feature
- target
- score
- pearson_r
- spearman_r
- pair_count
- feature_group
- path

4. Feature groups.

Classify features into:

- residual_dct: dct, srm, residual, highpass, high_pass, blockiness, laplacian, sobel, edge
- geometry: crop, scale, aspect, area_ratio, width, height, geometry
- local_nuisance: ignore, occlud, mask_area, tamper_occluded
- color_histogram: histogram, mean, std, color
- other

5. Decision.

Use corrected top correlations:

- residual_dct_signal_dominant
- geometry_still_dominant_or_mixed
- mixed_low_level_and_geometry_signal
- weak_low_level_signal
- correlation_parser_failed_or_empty

6. Tests.

Tests must cover:

- nested correlation schema flattening
- flat correlation schema flattening
- non-empty top correlations when nested data exists
- feature group classification
- decision changes away from parser-failure when valid correlations exist

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_residual_degradation_analysis_config.py configs/evaluation/snsaug_v2_residual_degradation_analysis.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_residual_degradation_analysis.py
- python3 scripts/evaluation/run_snsaug_v2_residual_degradation_analysis.py --help
- grep -q SNSAUG_V2_RESIDUAL_DEGRADATION_ANALYSIS_OK docs/snsaug_v2_residual_degradation_analysis.md
- python3 scripts/agent/check_agent_changes.py tasks/0070a-fix-residual-correlation-decision-parser.md

## Marker

SNSAUG_V2_RESIDUAL_DEGRADATION_ANALYSIS_OK
