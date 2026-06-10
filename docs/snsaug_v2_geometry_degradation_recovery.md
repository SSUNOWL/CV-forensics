# SNSAug V2 Geometry Degradation Recovery

SNSAUG_V2_GEOMETRY_DEGRADATION_RECOVERY_OK

0069 evaluates inference-time geometry normalization and mild degradation correction after 0068 identified global geometry/degradation as the dominant SNSAug failure factor. This branch uses the frozen pre-SNS best bundle and a fixed SNSAug pair root. It does not train, fine-tune, download, use network access, or modify checkpoints.

## Policies

The evaluator runs each fixed-pair row under:

- `original`: unchanged input
- `content_box_crop_resize`: estimate non-background content box, crop it, and resize back to the current image geometry
- `geometry_normalized`: crop estimated content box and resize toward the paired clean geometry
- `deblock_mild`: median-filter deblocking approximation
- `resize_to_clean_proxy`: resize SNS view to the paired clean image size
- `geometry_plus_deblock`: geometry normalization followed by mild deblocking
- `oracle_clean_geometry_diagnostic`: diagnostic-only clean-geometry oracle, not deployment-ready

The content box is estimated from border/background color when explicit metadata is unavailable.

## Outputs

Successful actual runs write:

- `geometry_degradation_recovery_records.jsonl`
- `per_policy_per_profile_metrics.json`
- `recovery_delta_summary.json`
- `geometry_degradation_recovery_report.md`
- `visual_gallery_manifest.json`
- `artifact_manifest.json`

Metrics include accuracy, macro-F1, real FPR, synthetic recall, tampered recall, localization activation recall, valid IoU, mean `p_tampered` on tampered rows, non-tampered high-mask rate, and recovery over `original`.

## Success Criteria

A policy is marked promising if it improves tampered recall by at least `0.10` on at least two focus profiles, or valid IoU by at least `0.05` on at least two focus profiles, while avoiding synthetic recall collapse and severe real-FPR increase. Focus profiles are `canvas_9x16_only`, `resize_crop_pad`, `zoom_crop`, `resize_jpeg`, `screenshot_recapture_light`, and `combined_sns_realistic`.
