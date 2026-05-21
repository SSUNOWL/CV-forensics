# Pre-SNS Visual Artifacts

`PRE_SNS_VISUAL_ARTIFACTS_OK`

Pre-SNS visual artifacts make the localization head visible for a single-image
report. When the model activates localization and the approved config sets
`write_visual_artifacts=true`, the runner writes:

- `predicted_mask.png`
- `heatmap.png`
- `overlay.png`
- `visual_artifacts_manifest.json`

The mask and heatmap are derived from the model's predicted localization
probability map. The overlay blends the predicted mask over the original image.
Ground-truth masks are not used for report visualization.

Artifacts are written only under the approved `report_root`, which must be
outside the repository and not under repository `outputs/` or `checkpoints/`.
If localization is not activated, no visual suspicion artifact is invented.

This is pre-SNS visualization only. It is not SNS augmentation and not SNS
perturbation evaluation.
