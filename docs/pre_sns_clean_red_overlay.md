# Pre-SNS Clean Red Overlay

`PRE_SNS_CLEAN_RED_OVERLAY_OK`

The clean red overlay is a presentation-friendly visualization for a single
pre-SNS inference report. When the model activates localization and the
approved config sets `write_clean_red_overlay=true`, the runner writes:

- `clean_focused_mask.png`
- `clean_red_overlay.png`
- `clean_red_overlay_manifest.json`

The focused mask is derived only from the model-predicted localization output.
It may keep the strongest suspicious pixels by top-percentile thresholding and
optionally retain the largest connected component. Ground-truth masks are not
used for prediction visualization.

`clean_red_overlay.png` contains only the original image plus semi-transparent
red blending where `clean_focused_mask.png` is active. Pixels outside the active
mask remain unchanged. The clean overlay must not contain text, bounding boxes,
arrows, labels, legends, titles, captions, watermarks, panel layouts, or any
annotation outside the red mask region.

This artifact is for qualitative inspection and presentation. It is not a new
training step, not SNS augmentation, and not SNS perturbation evaluation.

Artifacts are written only under the approved `report_root`, which must be
outside the repository and not under repository `outputs/` or `checkpoints/`.
