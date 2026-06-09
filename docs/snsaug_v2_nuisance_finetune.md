# SNSAug V2 Nuisance Mask Fine-Tune

SNSAUG_V2_NUISANCE_MASK_FINETUNE_OK

0063b proved that the guarded real training branch can write real checkpoints, but balanced hard-negative suppression did not remove the over-tampered SNS bias. The next branch separates benign local SNS artifacts from malicious tamper evidence instead of relying only on thresholds or class-level penalties.

The 0064 model adds:

- `class_head` for real / synthetic / tampered.
- `tamper_mask_head` for malicious manipulation regions.
- `sns_nuisance_mask_head` supervised by local `ignore_mask` regions from SNSAug pair generation.
- `global_degradation_head` for JPEG, resize, screenshot, platform layout, overlay, sticker, news/meme, combined SNS, and severity metadata.
- optional reliability logits around regions where nuisance evidence reduces forensic reliability.

Mask-guided tamper feature gating uses:

```text
F_tamper = F * (1.0 - alpha * downsample(sns_mask_or_pred_sns_mask))
```

The gate is soft. It reduces the tendency to treat platform UI bars, text overlays, stickers, watermarks, and meme/news banners as tamper evidence without erasing the underlying image evidence completely.

## Stabilization

0064e warm-starts the shared backbone, class head, decoder, and tamper mask head from the configured pre-SNS bundle or `warm_start_checkpoint_path`. Newly added `sns_nuisance_mask_head`, `global_degradation_head`, and `reliability_head` remain fresh. Each real run writes `warm_start_report.json` with warm-start provenance, loaded/missing keys, loaded/total tensor counts, and `loaded_ratio = loaded_numel / total_numel`.

`artifact_manifest.json` also includes `warm_start_summary` with the source path, loaded/total numel, loaded ratio, and missing/unexpected key counts so audits can verify warm-start coverage without opening the full report.

0064g changes the warm-start policy to a hybrid source:

- class backbone and `class_head` load from `class_backbone_warm_start_path`, `pre_sns_best_bundle_path`, or the configured base bundle
- tamper decoder and `tamper_mask_head` load from `tamper_head_warm_start_path`, `warm_start_checkpoint_path`, or the configured base bundle
- `sns_nuisance_mask_head`, `global_degradation_head`, and `reliability_head` are always initialized from scratch

The real run writes `hybrid_warm_start_report.json` next to the compatibility `warm_start_report.json`. The hybrid report records loaded/missing keys per component so an audit can distinguish class preservation from tamper-mask transfer.

Gating is staged:

- Phase 1 uses `gating_alpha_effective = 0.0` and trains only new nuisance/degradation/reliability heads.
- Phase 2 ramps gating up to `min(gating_alpha, phase_2_gating_alpha_max)`.
- Phase 3 caps gating at `min(gating_alpha, phase_3_gating_alpha_max)` unless joint tuning is explicitly enabled.

The loss includes non-tampered tamper-mask suppression and mask-area regularization so real/synthetic SNS overlays do not become all-one tamper masks. Training logs include prediction counts, mean class probabilities, per-class tamper mask area, SNS-mask IoU, and effective gating alpha. Checkpoints record tensor counts and collapse-guard status; a collapsed checkpoint is not selected as best unless no guardrail-passing checkpoint exists.

0064g adds synthetic-preservation pressure because prior 0064 runs recovered tamper activation largely through over-tampered bias while `synthetic_recall` remained zero on key SNS profiles. The preservation term penalizes synthetic rows when `p_synthetic` falls below `synthetic_probability_floor`, and class-balanced SNS sampling cycles real, synthetic, and tampered rows so synthetic examples are not starved during short corrective runs. Logs now include `synthetic_preservation_loss`, synthetic-recall proxy, real-FPR proxy, tampered-recall proxy, and non-tampered high-mask-rate proxy fields.

Best-checkpoint selection uses the balanced score:

```text
balanced_score =
  + 1.0 * tampered_recall
  + 1.0 * tampered_valid_mean_iou
  + 0.8 * synthetic_recall
  - 2.0 * real_fpr
  - 1.0 * non_tampered_high_mask_rate
```

The guard rejects checkpoints with `synthetic_recall = 0`, high real false positives, high non-tampered mask activation, weak tampered recall, or single-class collapse. If no checkpoint passes, the branch still writes the last real-weight checkpoint and marks selection as `fallback_last_no_guardrail_pass`.

Training remains guarded. A dry-run validates the config and prints planned outputs without creating checkpoint files. A real run requires:

- `approval_text = I_APPROVE_SNSAUG_V2_NUISANCE_MASK_FINETUNE`
- `no_network = true`
- `no_download = true`
- train-only manifest rows
- output and checkpoint roots outside the repository

Validation or fixed-pair evaluation roots must not be used as training input. Checkpoints must contain real `model_state_dict` weights; proxy-only `trainable_state` payloads are invalid.

## Phase Steps

Actual training honors the three phase controls:

- `phase_1_max_steps`
- `phase_2_max_steps`
- `phase_3_max_steps`

If a phase-specific value is omitted, `max_steps_per_phase` is used as the default. A 5x3 audit must write at least 15 `training_log.jsonl` rows with phase counts `{1: 5, 2: 5, 3: 5}`. A 30x3 run must write at least 90 rows with phase counts `{1: 30, 2: 30, 3: 30}`. The artifact manifest and best/last checkpoints record the final `global_step` and `phase_counts`.
