# Community Forensics-Small Multi-Head Tiny Smoke

This document defines the CF-Small multi-head tiny smoke workflow for non-SNS pre-baseline model verification.

The smoke extends the earlier binary real/synthetic local tiny smoke with a generator-family provenance head. It is still a tiny smoke test only. It is not full training, not a performance claim, and not a replacement for the later approved baseline training stage.

The intended model shape is a shared feature extractor with two supervised heads:

- binary class head for `real` and `synthetic`
- generator-family provenance head for `LatDiff`, `PixDiff`, `GAN`, `Other`, and `Real-or-N/A`

The family labels are coarse architecture labels, not exact model attribution. Real samples use `Real-or-N/A`. Synthetic family labels are derived from architecture metadata when it is available. `model_name` and `subset` remain analysis metadata and should be preserved in approved local manifests when present.

The future manual runner may read only explicitly listed image paths from an approved local config. It must not recursively scan dataset directories, download data, write outputs, write checkpoints, use SNS augmentation, or run SNS perturbation evaluation.

This smoke verifies:

- local manifest metadata preservation
- binary real/synthetic label handling
- architecture-derived family label handling
- shared feature extractor execution
- finite class loss, family loss, and total loss
- class and family smoke accuracy plumbing

SID-Set 3-way classification and tampered localization are the next stage. SNS augmentation is not implemented here.

Marker: CF_SMALL_MULTIHEAD_TINY_SMOKE_OK
