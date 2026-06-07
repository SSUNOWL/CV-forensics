# SNSAug V2 Checkpoint Comparison Evaluation

SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL_OK

This evaluation compares the frozen pre-SNS bundle and SNSAug fine-tuned checkpoints on the same fixed 0058c SNSAug v2 pair root. It is evaluation-only: no training, no fine-tuning, no network, and no downloads.

## Purpose

0060b training summaries can be proxy or subset metrics. This evaluator writes fixed-pair inference records and per-profile metrics so the `30x3` and `150x3` checkpoints can be compared against the frozen baseline on the same examples.

## Outputs

The external output root contains:

- `model_eval_records.jsonl`
- `model_eval_comparisons.jsonl`
- `per_model_per_profile_metrics.json`
- `robustness_drop_by_model.json`
- `checkpoint_comparison_summary.json`
- `checkpoint_comparison_report.md`
- `worst_samples_by_model.json`
- `visual_gallery_manifest.json`
- `artifact_manifest.json`

Full fixed-pair runs must set `full_fixed_pair_evaluation_ran=true` and `eval_subset_only=false`. Tests may use subset fixtures, but must mark `eval_subset_only=true`.

## Guardrails

The config requires `no_training=true`, `no_finetune=true`, `no_network=true`, and `no_download=true`. `pair_root` must be an evaluation or fixed-pair root, and training manifests are rejected as evaluation input. Outputs must be outside the repository.
