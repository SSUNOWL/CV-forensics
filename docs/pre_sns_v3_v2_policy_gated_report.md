# Pre-SNS v2 Policy-Gated Integrated Report

Marker: PRE_SNS_V3_V2_POLICY_GATED_REPORT_OK

This is pre-SNS reporting only. It is not training and it is not SNS augmentation.

## Pipeline

Clean long256 remains the primary detector. It determines the class gate and remains the fallback localization source when the tile-localizer-v2 output is unreliable.

Tile localizer v2 is a conditional high-resolution localizer. It is not used blindly. The report first runs clean long256, then only runs v2 localization for `tampered` cases. Non-tampered cases suppress localization by default.

```text
clean long256 detector -> policy gate -> conditional tile localizer v2 -> final red mask report
```

## Policy Gate

The policy gate applies configurable checks to the v2 mask:

- threshold the v2 probability mask with `mask_threshold`
- compute `mask_area_pct`
- compute `component_count`
- compute `largest_component_area_pct`
- reject masks that are too small or too large
- optionally reject masks with too many components or too-small dominant components
- optionally fall back to the clean long256 baseline mask when the v2 output is unreliable

For non-tampered long256 outputs, the final `localized_evidence_status` is `suppressed_non_tampered` by default.

## Outputs

Under external `output_root`, the runner writes:

- `policy_gated_report.json`
- `final_mask.png`
- `final_clean_red_overlay.png`
- `baseline_clean_red_overlay.png` when available
- `tile_v2_clean_red_overlay.png`
- `tile_v2_probability_mask.png`
- `comparison_sheet.jpg` when GT exists
- `artifact_manifest.json`

Clean red overlays contain no text. Comparison sheets may contain labels.

## Metrics

When GT exists, the report records:

- `baseline_iou`
- `baseline_dice`
- `v2_raw_iou`
- `v2_raw_dice`
- `final_iou`
- `final_dice`
- `final_iou_delta_vs_baseline`
- `final_mask_source`
