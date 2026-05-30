# Pre-SNS v3 Final Visual Audit

Marker: PRE_SNS_V3_FINAL_VISUAL_AUDIT_OK

This is the final pre-SNS audit before SNS robustness evaluation. It evaluates class quality and red-mask quality for the lightweight conditional pipeline:

```text
clean long256 detector + 512 tile localizer
```

The audit is evaluation/reporting only. It does not train, does not add SNS augmentation, does not download datasets, and does not use network access.

## Inputs

- clean validation manifest or explicit sample list
- clean long256 checkpoint
- tile localizer checkpoint
- approved external `output_root`
- class balance and max sample settings

Tampered samples with GT masks are preferred when balanced sampling is enabled. Optional hard-case lists can be supplied for severe IoU failures, low-IoU cases, false-positive real cases, or class-mask inconsistent cases.

## Outputs

Under `output_root`, the runner writes:

- `final_audit_records.jsonl`
- `final_audit_summary.json`
- `confusion_matrix.json`
- `good_red_mask_cases.json`
- `acceptable_red_mask_cases.json`
- `failed_red_mask_cases.json`
- `false_positive_real_cases.json`
- `false_negative_tampered_cases.json`
- `needs_manual_review_cases.json`
- `red_mask_gallery_manifest.json`
- `artifact_manifest.json`

Each case directory contains clean red overlays with no text, agreement overlays when GT exists, a labeled comparison sheet, and `case_summary.json`.

## Recommendation

The summary recommendation is one of:

- `ready_for_sns_robustness_evaluation`
- `needs_more_tile_localization_training`
- `needs_manual_review`

The policy considers class macro F1, tampered recall, real false positive rate, and failed red-mask rate.

