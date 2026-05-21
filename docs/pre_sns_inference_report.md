# Pre-SNS Inference Report

`PRE_SNS_SINGLE_IMAGE_REPORT_OK`

`PRE_SNS_VISUAL_ARTIFACTS_OK`

This runner produces a one-image pre-SNS forensic report from an explicitly
provided local image and an explicitly provided trained pre-SNS checkpoint. It
is intended for inspection of the pilot model path, not as a final performance
claim.

Use `config_kind: example_symbolic` for the tracked safe example. Use
`config_kind: approved_pre_sns_single_image_report` for an approved local run.
The legacy alias `approved_local_pre_sns_single_image_report` is accepted for
older local configs, but new configs should use the canonical value.

## Output

The report keeps the project target structure together:

- Class: `real / full_synthetic / tampered`
- Mask: conditional localization summary
- Family: coarse generator-family provenance
- Reason: deterministic template-based evidence explanation

The JSON report includes `class`, `class_conf`, `family`, `family_conf`,
`tampered_score`, `localization_head`, `mask_area_pct`, `reason`,
`latency_ms`, and `fps_estimate`.

When `write_visual_artifacts=true` and localization is `activated`, the report
also links `predicted_mask_path`, `heatmap_path`, `overlay_path`, and
`visual_artifacts_manifest_path`. These files are visualizations of the model's
predicted localization probabilities, not ground-truth masks.

## Localization

The localization head is conditional. If `tampered_score >= threshold_tau`, the
report marks localization as `activated` and computes `mask_area_pct` from the
predicted mask probabilities. If the score is below `threshold_tau`, the report
uses `skipped_below_threshold`.

## Write Policy

The runner reads exactly one configured `image_path` and exactly one configured
`checkpoint_path`. It never scans directories, downloads data, trains, updates
checkpoints, or applies SNS augmentation.

In approved local mode, `image_path`, `checkpoint_path`, and `report_root` may
be absolute local paths only after the approval phrase is present. The optional
root allow-lists are:

- `approved_image_roots`
- `approved_checkpoint_roots`
- `approved_report_roots`
- `approved_local_roots`

If a specific root list is omitted, `approved_local_roots` may be used for that
path type. When root lists are provided, the configured path must be under the
matching approved root. `report_root` may not exist yet, but its parent must
already be a valid local directory.

When `write_report=false`, the runner only prints JSON. When
`write_report=true`, it may write a small report JSON under the approved
`report_root`, which must be outside the repository and not under repository
`outputs/` or `checkpoints/`.

Visual artifacts, when requested, are written under the same approved
`report_root`. If localization is skipped or not applicable, the runner records
`visual_artifacts_written=false` and does not invent a suspicious mask.

SNS augmentation and SNS perturbation evaluation are not part of this phase.
