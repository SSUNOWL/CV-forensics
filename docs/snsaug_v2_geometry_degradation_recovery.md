# SNSAug V2 Geometry Degradation Recovery

SNSAUG_V2_GEOMETRY_DEGRADATION_RECOVERY_OK

0069 evaluates inference-time geometry normalization and mild degradation correction after 0068 identified global geometry/degradation as the dominant SNSAug failure factor. This branch uses the frozen pre-SNS best bundle and a fixed SNSAug pair root. It does not train, fine-tune, download, use network access, or modify checkpoints.

0069b extends the non-oracle geometry estimation path after the initial 0069 run found `oracle_only_geometry_recovery`: the clean-geometry oracle recovered focus profiles, but deployable heuristics mostly helped only `canvas_9x16_only`. The 0069b policies test whether better image-only content-box estimation can close that oracle gap before moving to a model-level residual or DCT branch.

## Policies

The evaluator runs each fixed-pair row under:

- `original`: unchanged input
- `content_box_crop_resize`: estimate non-background content box, crop it, and resize back to the current image geometry
- `geometry_normalized`: crop estimated content box and resize toward the paired clean geometry
- `deblock_mild`: median-filter deblocking approximation
- `resize_to_clean_proxy`: resize SNS view to the paired clean image size
- `geometry_plus_deblock`: geometry normalization followed by mild deblocking
- `border_trim_v2`: robust border/background trim resized to the paired clean geometry when available
- `edge_density_content_box_v2`: edge-density content-box estimation without labels, tamper masks, or model outputs
- `letterbox_unpad_resize_v2`: letterbox/pad removal using image-only border and edge candidates
- `screenshot_frame_trim_v2`: outer screenshot-frame trim for recapture-like margins
- `multi_candidate_geometry_v2`: deterministic image-only candidate selection across the v2 box estimators
- `multi_candidate_geometry_plus_deblock_v2`: selected v2 geometry candidate followed by mild deblocking
- `score_guided_geometry_diagnostic`: optional diagnostic policy; it is not treated as deployable
- `oracle_clean_geometry_diagnostic`: diagnostic-only clean-geometry oracle, not deployment-ready

The deployable v2 candidate selector uses only image geometry, edge density, and border statistics. It must not use content labels, tamper masks, ignore masks, or oracle clean geometry to choose candidates. When a policy crops or resizes an image, tamper and ignore masks are transformed with the same crop and nearest-neighbor resize before IoU is computed.

## Outputs

Successful actual runs write:

- `geometry_degradation_recovery_records.jsonl`
- `per_policy_per_profile_metrics.json`
- `recovery_delta_summary.json`
- `oracle_gap_closure_summary.json`
- `content_box_candidate_diagnostics.jsonl`
- `geometry_degradation_recovery_report.md`
- `0069b_geometry_estimation_report.md`
- `visual_gallery_manifest.json`
- `artifact_manifest.json`

Metrics include accuracy, macro-F1, real FPR, synthetic recall, tampered recall, localization activation recall, valid IoU, mean `p_tampered` on tampered rows, non-tampered high-mask rate, and recovery over `original`.

`oracle_gap_closure_summary.json` compares each deployable policy against both `original` and `oracle_clean_geometry_diagnostic`. Gap closure is `policy_gain / oracle_gain`, clipped to `[0, 1]`; when the oracle has no positive gain, closure is reported as `0.0`. `content_box_candidate_diagnostics.jsonl` records the candidate boxes, image-only scores, selected candidate, and `used_labels_or_masks=false` for auditability.

## Success Criteria

A policy is marked promising if it improves tampered recall by at least `0.10` on at least two focus profiles, or valid IoU by at least `0.05` on at least two focus profiles, while avoiding synthetic recall collapse and severe real-FPR increase. Focus profiles are `canvas_9x16_only`, `resize_crop_pad`, `zoom_crop`, `resize_jpeg`, `screenshot_recapture_light`, and `combined_sns_realistic`.

Diagnostic policies are excluded from deployable promising-policy decisions. The 0069b decision point is whether non-oracle policies close enough of the oracle gap to justify a deployable inference path, or whether the project should proceed to a model-level residual/DCT branch.
