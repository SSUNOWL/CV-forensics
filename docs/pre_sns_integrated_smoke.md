# Pre-SNS Integrated Model Smoke

This document defines the first guarded pre-SNS integrated model smoke. It combines the earlier CF-Small provenance smoke and SID-Set localization smoke into one tiny non-SNS model interface.

The smoke model contains:

- a shared image backbone
- a 3-way class head for `real`, `full_synthetic`, and `tampered`
- a generator-family provenance head for `LatDiff`, `PixDiff`, `GAN`, `Other`, and `Real-or-N/A`
- a conditional localization head for tampered samples with masks
- a deterministic evidence and reason summary compatible with the existing model output schema

Loss routing is conditional. Class loss applies to all labeled samples. Family loss applies only to samples with family labels. Localization loss applies only to tampered samples with explicit masks. Samples without a family label or without a relevant mask do not force those losses.

This is a smoke test only. It is not full training, not a performance claim, not dataset download, not SNS augmentation, and not SNS perturbation evaluation. Future approved local execution may read only explicit image and mask paths from an approved local config and prints a JSON summary instead of writing predictions, outputs, or checkpoints.

Marker: PRE_SNS_INTEGRATED_SMOKE_OK
