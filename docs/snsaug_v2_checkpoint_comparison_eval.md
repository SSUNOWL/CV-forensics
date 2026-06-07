# SNSAug V2 Checkpoint Comparison Evaluation

SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL_OK

This evaluation compares the frozen pre-SNS bundle and SNSAug fine-tuned checkpoints on the same fixed 0058c SNSAug v2 pair root. It is evaluation-only: no training, no fine-tuning, no network, and no downloads.

## Purpose

0060b training summaries can be proxy or subset metrics. This evaluator writes fixed-pair inference records and per-profile metrics so the `30x3` and `150x3` checkpoints can be compared against the frozen baseline on the same examples.

## Real Inference Requirement

The evaluator must load the frozen pre-SNS bundle and each SNSAug fine-tuned `.pt` checkpoint before scoring records. Fine-tuned checkpoints must provide real model weights or an explicit derived checkpoint path; `trainable_state` proxy values alone are rejected because they cannot prove real fixed-pair inference.

Preferred 0060b checkpoints use `checkpoint_kind=snsaug_v2_real_model_weights` and contain `model_state_dict` plus optimizer/config metadata. The evaluator records checkpoint SHA-256, byte size, tensor count, and total tensor parameter count for each fine-tuned checkpoint.

Every `model_eval_records.jsonl` row must contain `pred_class`, `p_real`, `p_synthetic`, and `p_tampered`. Metrics are computed from these records only, not from labels, profile names, or synthetic score tables.

The three required model IDs are:

- `pre_sns_baseline`
- `snsaug_guarded_short_30x3`
- `snsaug_medium_150x3`

`checkpoint_comparison_summary.json` must include:

- `pre_sns_baseline_vs_snsaug_guarded_short_30x3`
- `pre_sns_baseline_vs_snsaug_medium_150x3`
- `snsaug_guarded_short_30x3_vs_snsaug_medium_150x3`

If `comparisons=[]`, the run is invalid even when the script exits successfully.

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

## Sanity Guards

The evaluator fails the run when comparisons are empty, all model/profile accuracies are exactly `1.0`, all profile metrics are identical across all models, required probability fields are missing, checkpoint hashes are missing for fine-tuned checkpoints, or a fine-tuned checkpoint produces byte-identical predictions to the baseline without an explicit `no_weight_delta` exemption.

It also emits a warning when `non_tampered_high_mask_rate=1.0` appears with `real_fpr=0.0`, because that combination is usually a sign that mask and class metrics are being interpreted inconsistently.

## Troubleshooting

If 0061B says `fine-tuned checkpoint has only trainable_state proxy values`, rerun 0060b after the real checkpoint saving fix. Existing proxy checkpoints are not valid for this evaluator and should not be used for the real fixed-pair comparison.
