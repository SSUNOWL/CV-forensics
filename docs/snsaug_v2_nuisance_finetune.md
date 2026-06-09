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

0064e warm-starts the shared backbone, class head, decoder, and tamper mask head from the configured pre-SNS bundle or `warm_start_checkpoint_path`. Newly added `sns_nuisance_mask_head`, `global_degradation_head`, and `reliability_head` remain fresh. Each real run writes `warm_start_report.json` with loaded/missing keys and loaded/total tensor counts.

Gating is staged:

- Phase 1 uses `gating_alpha_effective = 0.0` and trains only new nuisance/degradation/reliability heads.
- Phase 2 ramps gating up to `min(gating_alpha, phase_2_gating_alpha_max)`.
- Phase 3 caps gating at `min(gating_alpha, phase_3_gating_alpha_max)` unless joint tuning is explicitly enabled.

The loss includes non-tampered tamper-mask suppression and mask-area regularization so real/synthetic SNS overlays do not become all-one tamper masks. Training logs include prediction counts, mean class probabilities, per-class tamper mask area, SNS-mask IoU, and effective gating alpha. Checkpoints record tensor counts and collapse-guard status; a collapsed checkpoint is not selected as best unless no guardrail-passing checkpoint exists.

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
