# Pre-SNS v3 Tile Localizer Training

Marker: PRE_SNS_V3_TILE_LOCALIZER_TRAINING_OK

This is guarded pre-SNS tile localization training. It is not SNS augmentation, it does not replace clean long256 as the primary detector, and it does not train unless the config contains the explicit approval phrase and external output roots.

## Purpose

Clean long256 remains the global detector. The tile localizer is a conditional high-resolution red-mask model used after clean long256 predicts `tampered` or a tampered-suspect case.

The expected input is the 0042 `tile_localization_manifest.json`, containing positive GT-centered or jittered crops, low-IoU hard crops, negative real/full_synthetic crops, and optional hard-negative false-positive crops.

## Guardrails

- `config_kind`: `approved_pre_sns_v3_tile_localizer_training`
- `execution_mode`: `approved_local_pre_sns_v3_tile_localizer_training`
- `user_approval_text`: `I_APPROVE_PRE_SNS_V3_TILE_LOCALIZER_TRAINING`
- `no_download`, `no_network`, and `no_sns_augmentation` must be `true`
- `tile_manifest_path` must be under `approved_input_roots`
- `approved_run_root` and `approved_checkpoint_root` must be outside the repository and outside input roots
- protected path segments are rejected

`no_write_dry_run` validates the config and returns a summary without writing artifacts or checkpoints.

## Model

The model is lightweight and self-contained:

- RGB branch
- fixed high-pass/residual filter branch
- small U-Net-like encoder-decoder
- one-channel tampered mask logits
- optional tile confidence logits

No external pretrained weights are used.

## Outputs

Under `approved_run_root`:

- `run_summary.json`
- `val_metrics.json`
- `threshold_calibration.json`
- `tile_metrics.jsonl`
- `artifact_manifest.json`
- `config_snapshot.json`

Under `approved_checkpoint_root`:

- `best_tile_localizer.pt`
- `latest_tile_localizer.pt`

## Metrics

Validation reports `tile_mean_iou`, `tile_median_iou`, `tile_mean_dice`, `positive_tile_iou`, `negative_tile_false_activation_rate`, `empty_mask_precision_proxy`, a mask area percent summary, threshold sweep, and `selected_mask_threshold`.

