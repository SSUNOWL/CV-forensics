# SNSAug V2 Checkpoint Comparison Evaluation

SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL_OK

This evaluation compares the frozen pre-SNS bundle and SNSAug fine-tuned checkpoints on the same fixed 0058c SNSAug v2 pair root. It is evaluation-only: no training, no fine-tuning, no network, and no downloads.

## Purpose

0060b training summaries can be proxy or subset metrics. This evaluator writes fixed-pair inference records and per-profile metrics so the `30x3` and `150x3` checkpoints can be compared against the frozen baseline on the same examples.

## Real Inference Requirement

The evaluator must load the frozen pre-SNS bundle and each SNSAug fine-tuned `.pt` checkpoint before scoring records. Fine-tuned checkpoints must provide real model weights or an explicit derived checkpoint path; `trainable_state` proxy values alone are rejected because they cannot prove real fixed-pair inference.

Preferred 0060b checkpoints use `checkpoint_kind=snsaug_v2_real_model_weights` and contain `model_state_dict` plus optimizer/config metadata. The evaluator records checkpoint SHA-256, byte size, tensor count, and total tensor parameter count for each fine-tuned checkpoint.

Every `model_eval_records.jsonl` row must contain `model_id`, sample identifiers when available (`row_id`, `record_id`, `sample_id`), `base_id`, `profile`, `view`, `content_label`, `image_path`, `pred_class`, `p_real`, `p_synthetic`, `p_tampered`, `localization_activated`, and tampered localization fields (`valid_iou`/`tampered_valid_iou`, `raw_iou`) where applicable. Metrics and comparisons are computed from these records only, not from labels, profile names, or synthetic score tables.

Model IDs are not fixed. The historical IDs remain valid examples:

- `pre_sns_baseline`
- `snsaug_guarded_short_30x3`
- `snsaug_medium_150x3`

If `comparison_pairs` is present in the config, the evaluator compares those explicit model pairs. Otherwise it compares all pairwise combinations of `config["models"]`.

## Comparison Join Keys

`model_eval_comparisons.jsonl` is built by joining records across models, not by matching hard-coded model IDs. The evaluator tries these keys in order and uses the first key that creates at least one joinable group for the configured comparison pairs:

- `row_id`
- `record_id`
- `sample_id`
- `base_id + profile + view + content_label`
- `base_id + profile + content_label`
- `image_path`
- `image_relpath`

Each comparison row includes the left/right model IDs, `join_key_type`, `join_key`, sample metadata, left/right predictions, correctness delta, tampered probability delta, and valid-IoU delta.

`checkpoint_comparison_summary.json` must contain non-empty `comparisons`. Each entry includes `comparison_id`, left/right model IDs, `comparison_count`, and per-profile deltas for accuracy, tampered recall, localization activation recall, valid IoU, real false-positive rate, and synthetic recall.

If `comparisons=[]`, the run is invalid even when the script exits successfully.

## Join Diagnostics

The evaluator writes `comparison_join_diagnostics.json` with record counts, per-model record counts, available record keys, configured model IDs, comparison pairs, and candidate join-key stats. Candidate stats include group count, missing records, pair-joinable group counts, and full-model group counts.

When comparisons are empty, the evaluator writes `comparison_join_diagnostics.json`, `model_eval_records.jsonl`, and `model_eval_comparisons.jsonl` before failing. This is the expected failure mode for unjoinable records.

## Outputs

The external output root contains:

- `model_eval_records.jsonl`
- `model_eval_comparisons.jsonl`
- `comparison_join_diagnostics.json`
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
