# Pre-SNS Dataset Manifest Gate

This document defines the pre-SNS unified local dataset manifest gate.

The manifest combines approved local Community Forensics-Small and SID-Set sample records into one auditable schema before any real training. It records what each sample can supervise:

- class supervision
- family/provenance supervision
- localization/mask supervision

Community Forensics-Small records preserve `sample_id`, `image_path`, `class_label`, `family_label`, `architecture`, `model_name`, and `subset`. These records provide class and family/provenance supervision, but not localization supervision.

SID-Set records preserve `sample_id`, `image_path`, `class_label`, `label_id`, optional `split`, optional `img_id`, and `mask_path` for tampered samples. These records provide class supervision and localization supervision for tampered samples with masks. SID-Set family/provenance labels are not required.

The generated unified manifest adds `tasks_available` and `loss_routing` fields so later training can route class, family, and localization losses without guessing from dataset names.

In approved local mode, sample `image_path` values and tampered SID-Set `mask_path` values must be explicit absolute file paths under `approved_local_roots`. The validator checks these sample paths through the explicit local file checker and confirms that they exist as files. Directory-only paths, paths outside the approved roots, URL or remote-scheme paths, Windows drive paths, traversal paths, and protected path segments remain rejected.

Generic recursive safety checks still apply to all other config fields. Only approved sample file path values are exempt from generic absolute-path rejection, and they are still subject to the stricter explicit file validation.

This remains a pre-training gate. It does not download datasets, train models, scan directories, read images, read masks, write outputs or checkpoints, implement SNS augmentation, or run SNS perturbation evaluation.

Marker: PRE_SNS_DATASET_MANIFEST_OK
