# Metrics — Task 0009

<!-- METRICS_OK -->

## Purpose

This module (`src/cv_forensics/metrics.py`) provides pure-Python metric calculators
for the lightweight multi-head image forensics prototype. All functions operate on
toy labels, toy nested-list binary masks, toy localization states, toy latency values,
and toy perturbation group records. The module uses only the Python standard library.

## Connection to Project Proposal

The metrics map directly to the evaluation table in `docs/project_brief.md` §10:

| Metric | Module function |
|---|---|
| 3-way Classification Accuracy | `classification_accuracy`, `classification_metrics` |
| Macro-F1 | `macro_f1` |
| Tampered Mask IoU | `mask_iou` |
| Generator-Family Accuracy | `family_accuracy` |
| Perturbation Robustness Drop | `perturbation_robustness_drop`, `robustness_summary` |
| Latency (ms/image) | `latency_summary` |
| FPS | `fps_summary` |
| Localization Activation Recall | `localization_activation_recall` |

## Connection to Task 0007 (Model Output Schema)

`metrics.py` imports label constants from `src/cv_forensics/model_output_schema.py`:
`CLASS_LABELS`, `FAMILY_LABELS`, `FAMILY_REAL_OR_NA`, `LOCALIZATION_ACTIVATED`,
`LOCALIZATION_SKIPPED_BELOW_THRESHOLD`, `LOCALIZATION_STATES`. This ensures metric
functions always use the same label vocabulary as the forensic output schema.

## Connection to Task 0008 (Fake Inference Outputs)

The localization and classification metric functions accept exactly the kinds of
outputs that the task 0008 inference stub produces (class labels, family labels,
localization states). Future evaluation tasks can pipe `run_fake_inference` outputs
directly into these calculators without conversion.

## Why Toy Arrays

Real datasets (`data/`, `datasets/`), images, masks, and checkpoints are not
accessed in this task. Toy labels and nested-list binary masks allow the metric
logic to be validated deterministically with no external dependencies, no downloads,
and no training loops. Real evaluation will replace toy inputs with actual model
outputs once training scaffolding (Stage 1–3) is implemented.

## Robustness Drop — Placeholder / Calculator

`perturbation_robustness_drop` and `robustness_summary` are metric calculators only.
SNS augmentation (JPEG, resize, crop, screenshot, text overlay, sticker overlay,
recompression chain) is **not** implemented here. These functions accept pre-computed
toy clean and perturbed scores and compute `drop = clean_score - perturbed_score`.
Actual SNS augmentation will be added in a later task (Stage 4).

## What This Task Does NOT Do

- No real data, images, masks, outputs, checkpoints, or training.
- No dataset downloads or network access.
- No third-party package installation (pure Python standard library only).
- No SNS augmentation implementation.
- No file writes inside metric functions.

## Validation Commands

```bash
python3 scripts/agent/validate_metrics.py configs/metrics/toy_metrics.example.json
```

```bash
python3 scripts/agent/run_toy_metrics.py configs/metrics/toy_metrics.example.json
```

```bash
python3 tests/test_metrics.py
```

Optional (if pytest is installed):

```bash
pytest -q tests/test_metrics.py
```

## Mask IoU Convention

For this toy metric, if both the predicted mask and the ground-truth mask are
all-zero (no tampered region predicted or annotated), IoU is defined as `1.0`.
This avoids a `0/0` undefined case and expresses agreement that no region is
tampered. This convention applies only to toy evaluation; production evaluation
should document its own handling.

## Localization Activation Recall Note

`localization_activation_recall` counts true tampered examples and reports the
fraction where `localization_head == "activated"`. Tampered examples where
`localization_head == "skipped_below_threshold"` are explicitly counted as missed
activations (false negatives from a high `threshold_tau`). This reflects the project
design principle: `tau` must be tuned to prioritise tampered recall over compute savings.
