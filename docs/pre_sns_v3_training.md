# Pre-SNS v3 Training

Marker: `PRE_SNS_V3_TRAINING_OK`

v2 was a weak baseline: low 3-way accuracy/F1, poor real recall, high false activation, near-zero localization IoU, overlapping tampered-score distributions, central-blob masks, and family collapse toward LatDiff.

v3 stays pre-SNS only. It does not add SNS augmentation. The model adds a fixed high-pass branch beside RGB input so the CNN sees residual cues such as edges and local discontinuities without pretrained weights. It also adds a binary tamper head so tampered activation is learned directly instead of relying only on the 3-way class probability.

Localization uses BCEWithLogits plus Dice on tampered samples with masks. Non-tampered samples receive an empty-mask penalty to reduce false red overlays. The default weights prioritize class/tamper/localization over provenance family: class `1.0`, binary tamper `1.0`, family `0.3`, localization `10.0`, empty-mask `0.2`, Dice `1.0`.

Validation reports class metrics, binary tamper metrics, family accuracy on meaningful family labels, score summaries, threshold sweep, FPR-constrained `selected_tau`, localization activation/IoU, mask area summaries, and per-source/per-family confusion matrices. Tau selection chooses the threshold with highest tampered activation recall among rows with `false_activation_rate <= max_false_activation_rate` and falls back to the lowest false activation threshold if needed.

Actual runs require `config_kind: approved_pre_sns_v3_training`, `execution_mode: approved_local_pre_sns_v3_training`, explicit local manifests under approved input roots, and external approved run/checkpoint roots.

