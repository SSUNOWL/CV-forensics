# SID-Set Tiny Localization Smoke

This document defines the SID-Set local tiny localization smoke for non-SNS pre-baseline model verification.

The smoke adds SID-Set-style 3-way classification and conditional tampered localization after the Community Forensics-Small binary and provenance smokes. It verifies that a future approved local config can preserve SID-Set sample metadata, connect explicitly listed images and masks, run a tiny shared-backbone model, and compute finite class, mask, and total losses.

This is not full training and not a performance claim. It does not download SID-Set, scan dataset directories, write outputs, write checkpoints, implement SNS augmentation, or run SNS perturbation evaluation.

The class labels are:

- `real`
- `full_synthetic`
- `tampered`

The label IDs are:

- `0` real
- `1` full_synthetic
- `2` tampered

Masks are required only for tampered samples. Real and full_synthetic samples do not require masks, and localization for those samples is treated as not applicable. The smoke runner applies mask loss only to tampered samples that have explicit mask paths in an approved local config.

SID-Set samples may not have Community Forensics-Small-style family/provenance labels, so family/provenance loss is not applied to SID-Set samples in this task. CF-Small provenance and SID-Set localization will later be combined into a pre-SNS integrated model.

Actual local execution requires an untracked approved local config with:

- `config_kind: approved_local_sid_tiny_localization_smoke`
- `approved_real_data_access: true`
- `user_approval_text: I_APPROVE_LOCAL_NON_SNS_SID_SET_TINY_LOCALIZATION_SMOKE`
- explicit `approved_local_roots`
- explicit `sample_manifest` entries for image paths
- explicit `mask_path` entries for tampered samples only

Marker: SID_SET_TINY_LOCALIZATION_SMOKE_OK
