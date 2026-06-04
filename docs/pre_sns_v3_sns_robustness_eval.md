# Pre-SNS Basic SNS Robustness Evaluation

Marker: PRE_SNS_V3_SNS_ROBUSTNESS_EVAL_OK

This task is evaluation only. No training, fine-tuning, or SNS augmentation training occurs here.

## Purpose

The frozen pre-SNS best bundle is evaluated under basic social-media-like degradation before any SNS augmentation training. This task measures robustness under JPEG compression, resize, resize plus JPEG, mild center crop, and optional WebP when the local PIL build supports it.

## Guardrails

- This task does not train.
- This task does not fine-tune.
- This task does not download datasets.
- This task does not use network access.
- This task does not install packages.

Validation and test failure samples are evaluation artifacts only. They must not be used directly for training. Train-split mining for future training manifests must be done separately.

## Outputs

Under external `output_root`, the evaluator writes:

- `transformed_inputs/`
- `sns_robustness_records.jsonl`
- `sns_robustness_summary.json`
- `per_perturbation_metrics.json`
- `robustness_drop_metrics.json`
- `worst_samples.json`
- `worst_perturbations.json`
- `fragile_candidates.jsonl`
- `visual_gallery_manifest.json`
- `artifact_manifest.json`

Optional comparison sheets may be added for top worst cases when enough localization information is available.
