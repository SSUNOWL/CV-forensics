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

Training remains guarded. A dry-run validates the config and prints planned outputs without creating checkpoint files. A real run requires:

- `approval_text = I_APPROVE_SNSAUG_V2_NUISANCE_MASK_FINETUNE`
- `no_network = true`
- `no_download = true`
- train-only manifest rows
- output and checkpoint roots outside the repository

Validation or fixed-pair evaluation roots must not be used as training input. Checkpoints must contain real `model_state_dict` weights; proxy-only `trainable_state` payloads are invalid.
