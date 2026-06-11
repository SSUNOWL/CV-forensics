# SNSAug V2 Residual Degradation Analysis

SNSAUG_V2_RESIDUAL_DEGRADATION_ANALYSIS_OK

0070 is an evaluation-only DCT/SRM/high-pass residual degradation analysis. It follows 0067a, 0068, 0069, and 0069b: local ignore-mask masking was insufficient, global geometry/degradation dominated the failure, and oracle geometry recovered profiles that deployable geometry estimators did not.

This task does not train, fine-tune, download, use network access, or modify checkpoints.

## Inputs

The analysis reads:

- fixed SNSAug pair root from 0058c
- `meta.jsonl`
- image, tamper mask, and ignore mask paths from the pair rows
- pre-SNS baseline response records, usually filtered to the `original` policy
- optional 0068 and 0069b output roots for provenance in the artifact manifest

Response records are joined by `base_id`, `profile`, `view`, and `content_label`. Non-original policy rows are ignored when loading baseline responses.

## Features

For each clean/SNS pair, the analyzer computes:

- grayscale high-pass residual energy
- SRM-like fixed-filter mean, standard deviation, and energy
- Laplacian variance
- Sobel edge energy
- JPEG-like 8x8 blockiness score
- approximate 8x8 DCT total, low-frequency, high-frequency, and high/low ratio
- histogram L1 distance
- aspect, area, and crop-scale geometry proxies
- ignore mask area
- tamper mask area
- tamper occluded by ignore-mask ratio

Clean/SNS feature deltas are joined with `delta_p_tampered`, `delta_valid_iou`, `activation_flip_off`, `pred_flip`, and `correct_to_wrong`.

## Outputs

Successful actual runs write:

- `residual_degradation_records.jsonl`
- `residual_feature_summary.json`
- `residual_feature_response_correlation.json`
- `residual_degradation_report.md`
- `artifact_manifest.json`

The correlation file includes overall, per-label, and per-profile-family Pearson/Spearman summaries. Constant feature or target vectors produce `null` correlations rather than failing.

## Decision

The decision rule is diagnostic:

- Strong residual/DCT/SRM correlations recommend `train_0071_lightweight_residual_dct_branch`.
- Strong geometry correlations recommend one more geometry-normalized preprocessing refinement.
- Weak correlations recommend revisiting calibration or class-head robustness.

This task should decide whether 0071 should train a lightweight residual/DCT branch; it does not start that training.
