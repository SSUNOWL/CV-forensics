# Pre-SNS v3 Tile Localizer v2 Training

Marker: PRE_SNS_V3_TILE_LOCALIZER_V2_TRAINING_OK

This is pre-SNS high-resolution forensic tile localization training. It is not SNS augmentation and it does not replace clean long256 as the primary detector.

The model remains part of the lightweight conditional pipeline:

```text
clean long256 detector -> high-resolution forensic tile localizer v2
```

Unlike a SIDA-style large multimodal model, this path keeps inference lightweight and only activates localization after the primary detector reports tampered or tampered-suspect evidence.

## Model

The v2 localizer is torch-only and uses no external pretrained weights. It supports configurable tile sizes such as 768 or 1024 and feature modes:

- `rgb_only`
- `rgb_residual`
- `rgb_edge_residual`

The edge/residual mode adds deterministic Sobel X/Y, Laplacian, fixed high-pass residual, and residual magnitude cues. The model can emit `mask_logits`, `boundary_logits`, and tile confidence logits.

## Training

The trainer reads a 0042 tile manifest using `records`, `tiles`, or `tile_records`. Crops are loaded on the fly. If the configured tile size is larger than the manifest crop box, the crop is expanded around the same center and clamped to image boundaries.

Supported loss terms include BCE, Dice, Tversky, boundary loss, empty-mask loss, and false-activation area penalty for negatives. Oversampling can target severe/low/weak IoU buckets, hard negatives, and negative tiles.

All outputs and checkpoints must be outside the repository.

