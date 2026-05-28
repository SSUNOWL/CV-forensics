# Pre-SNS v3 Dual-Scale Report and Mining

Marker: `PRE_SNS_V3_DUAL_SCALE_REPORT_AND_MINING_OK`

## Why This Exists

The clean long256 pre-SNS v3 checkpoint improves classification and tampered detection over long224, but localization is not uniformly better. Some samples are localized better by long224, some long256 false-positive real samples have empty masks, and some true tampered samples have low-IoU masks. This task calibrates the report layer before any SNS augmentation or additional training.

## Dual-Scale Report

`scripts/inference/run_pre_sns_v3_dual_scale_report.py` runs long224 and long256 on the same image. long256 is the primary class-decision model. The report includes:

- `marker`
- `image_path`
- `primary_class_model`
- `long224`
- `long256`
- `class_mask_consistency`
- `localized_evidence_status`
- `final_decision`
- `final_decision_confidence`
- `classification_confidence`
- `localization_confidence`
- `localization_disagreement_reason`
- `mask_selection_reason`
- `explanation_reason`
- `final_mask_area_pct`
- `final_mask_stats`
- `final_mask_source`
- `dual_scale_mask_agreement_iou`
- `visual_artifacts`

Each model block includes class confidence, tampered score, family confidence, raw mask stats, processed mask stats, latency, device, and checkpoint path.

## Mask Selection

Masks are thresholded and post-processed by removing tiny components, optionally keeping top-k components, and optionally applying simple open/close morphology. The final mask rules are:

- When both masks are active and overlap enough, use a union mask.
- When only long256 is active and long256 predicts tampered, use long256.
- When long224 is active and long256 is empty, use long224 only if long256 predicts tampered or has an uncertain tampered score.
- When long256 predicts tampered and both masks are empty, report `localized_evidence_status: not_found`.
- When long256 predicts real or full synthetic, suppress active masks unless the tampered score is very high and mark the report uncertain.
- When both models predict tampered but processed mask agreement is below `cross_scale_disagreement_iou_threshold` (default `0.05`), report `localized_evidence_status: disputed` or `weak_disputed`, set `class_mask_consistency: cross_scale_mask_disagreement`, and cap decision confidence at medium.
- When only one scale produces a mask, localization confidence is downgraded unless the selected mask has strong component quality.
- When the final mask is extremely tiny, below `tiny_mask_area_pct_threshold` (default `0.15%`), and cross-scale agreement is low, the report must not call localization high confidence.

Classification confidence is reported separately from localization confidence. long256 remains the primary class-decision model, but low cross-scale mask agreement reduces only localization confidence and the final localized-evidence decision.

Clean red overlay images do not draw text. The comparison sheet may contain labels.

## Hard-Case Mining

`scripts/evaluation/mine_pre_sns_v3_hard_cases.py` writes only under an approved external output root. It produces:

- `hard_negative_real.json`
- `hard_negative_non_tampered.json`
- `hard_positive_tampered_low_iou.json`
- `class_mask_inconsistent_cases.json`
- `failed_cases.json`
- `mining_summary.json`

The miner supports positive integer `max_samples` and does not modify training data. `max_samples=0` is rejected; do not use zero to mean all samples. If a local wrapper wants "all samples", it must count the manifest samples first and pass that positive count into the approved config.

Checkpoint-backed mining defaults to `fail_fast: false`. If one sample fails during report generation, the miner appends `sample_id`, `class_label`, `image_path`, exception text, and traceback tail to `failed_cases.json`, continues with the remaining samples, and reports `failed_case_count` plus the failed-cases path in `mining_summary.json`. Set `fail_fast: true` only when debugging a specific crash. The miner can also mine from manifest-provided predictions for guarded dry analysis.

## Guardrails

This remains pre-SNS only. It does not add SNS augmentation, does not train, does not download datasets, does not use network resources, and does not write outputs or checkpoints into the repository.
