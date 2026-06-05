# SNSAug V2 Forced Localization Oracle Gate

`SNSAUG_V2_FORCED_LOCALIZATION_ORACLE_GATE_OK`

This workflow is evaluation only. It does not train, fine-tune, use validation failures for training, modify checkpoints, download assets, or use network resources.

## Goal

The diagnostic compares normal conditional localization against an oracle gate that forces localization for ground-truth tampered samples. It separates two failure modes:

- `class_activation_bottleneck`: classification or tampered-score collapse prevents localization from running
- `mask_decoder_geometry_bottleneck`: localization still fails even when the gate is forced open

## Modes

- `normal_gate`: preserves the current policy-gated behavior.
- `oracle_tampered_gate`: forces the tile localization path for rows with `content_label == "tampered"`.
- `threshold_sweep`: reports activation and false-positive tradeoffs for tampered-score thresholds `0.00`, `0.01`, `0.05`, `0.10`, `0.25`, and `0.50`.

## Mask Policy

`valid_iou` excludes SNS nuisance pixels with:

```text
valid_region = 1 - ignore_mask
```

SNS overlay pixels must not count as tamper ground truth for valid-IoU interpretation.

## Outputs

The diagnostic writes under the configured external output root:

- `oracle_gate_eval_records.jsonl`
- `oracle_gate_eval_comparisons.jsonl`
- `oracle_gate_per_profile_metrics.json`
- `oracle_gate_drop_metrics.json`
- `threshold_sweep_metrics.json`
- `activation_bottleneck_summary.json`
- `forced_localization_worst_samples.json`
- `visual_gallery_manifest.json`
- `artifact_manifest.json`
- `pred_masks/`
- `pred_red_overlays/`
- `gt_red_overlays/`
- `ignore_blue_overlays/`
- `overlap_overlays/`

All outputs must be written outside the repository.
