# Pre-SNS Long256 + Tile Integrated Report

Marker: PRE_SNS_V3_LONG256_TILE_INTEGRATED_REPORT_OK

This is pre-SNS reporting only. It is not training and it is not SNS augmentation.

## Pipeline

Clean long256 remains the primary detector. It provides the final class, family, tampered score, reason basis, and rough baseline mask.

The 512 tile localizer is a conditional high-resolution localizer. It runs only when clean long256 predicts `tampered` or when the tampered score is at least `tile_activation_tau`. For real or full_synthetic cases, the final mask is suppressed by default.

This is the lightweight alternative path to SIDA-style large VLM reporting:

```text
clean long256 detector -> conditional 512 tile localizer -> final red mask report
```

## Outputs

Under external `output_root`, the runner writes:

- `integrated_report.json`
- `integrated_report_summary.json`
- `final_mask.png`
- `final_clean_red_overlay.png`
- `artifact_manifest.json`
- optional `baseline_agreement_overlay.png`
- optional `tile_agreement_overlay.png`
- optional `gt_comparison_sheet.jpg`

Clean red overlays contain no text. Comparison sheets may contain labels.

## Report Fields

The report records:

- primary class/family/tampered score from clean long256
- `tile_localization_activated`
- `tile_count`
- `final_mask_area_pct`
- `baseline_mask_area_pct`
- `final_vs_baseline_mask_iou`
- `localized_evidence_status`
- `localization_confidence`
- `final_decision`
- `reason`

When a GT mask is provided, it also reports baseline IoU/Dice, tile-final IoU/Dice, and IoU/Dice deltas.

