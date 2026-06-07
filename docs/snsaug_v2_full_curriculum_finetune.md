# SNSAug V2 Full Curriculum Fine-Tune

SNSAUG_V2_FULL_CURRICULUM_FINETUNE_OK

This document describes the guarded full-curriculum fine-tuning path for `SNSAug-aware Multi-head Forensics Model v1`.

## Scope

The run starts from `.local/pre_sns_current_best_model_bundle.json`; it must not train from scratch. Training data comes only from the 0059 train-only SNSAug V2 curriculum manifest, schedule, and profile weights. Validation data is evaluation-only and includes a clean validation manifest plus the fixed 0058c balanced SNSAug benchmark. Optional 0058e oracle diagnostics are analysis-only and never training input.

## Curriculum

The full run uses the 0059 phases:

- Phase 1: activation recovery and safe geometry adaptation
- Phase 2: geometry plus light platform layout
- Phase 3: platform layout plus combined SNS

SNSAug labels are preserved as `real`, `synthetic`, and `tampered`.

## Loss And Selection

The training objective is:

```text
L_total =
  L_class
+ lambda_mask * L_tamper_mask_valid
+ lambda_score * L_tampered_score_consistency
+ lambda_consistency * L_clean_sns_class_consistency
+ lambda_hardneg * L_real_synthetic_sns_hard_negative
+ optional family loss where family_loss_mask == 1
```

`L_tamper_mask_valid` uses `valid_region = 1 - ignore_mask`, so SNS nuisance pixels are excluded from tamper localization loss.

`L_tampered_score_consistency` compares clean/SNS paired tampered logits for the same curriculum sample. Training logs and phase metrics include `mean_p_tampered_clean`, `mean_p_tampered_sns`, `tampered_score_consistency_loss`, and `tampered_pair_count`. The loss should be positive when SNS lowers `p_tampered` below the clean score or configured floor; it should be zero only when no tampered pairs are present, in which case `tampered_score_consistency_skip_reason` is recorded.

Best checkpoint selection uses SNSAug tampered recall plus valid IoU as the primary score, clean macro-F1 as the secondary tie-break, and a configured real-FPR limit as the guardrail.

## Required Outputs

Approved real runs must write outputs only under external roots:

- best checkpoint
- last checkpoint
- training logs
- per-phase metrics
- clean validation metrics
- 0058c SNSAug fixed benchmark metrics
- robustness drop metrics
- comparison vs frozen pre-SNS baseline
- report markdown
- artifact manifest

The CLI also supports a dry run that validates guardrails and prints the planned training/evaluation outputs without training or writing checkpoints.

## Dry-Run And Actual Run

`--dry-run` validates the config and prints the plan only. It must report `training_started=false` and `checkpoint_written=false`, and it must not create checkpoint files.

Running without `--dry-run` enters the guarded actual training branch after validation. The branch loads the pre-SNS bundle metadata, the 0059 train-only curriculum manifest, the curriculum schedule, and profile sampling weights. It executes the three curriculum phases, writes `training_log.jsonl`, and writes both checkpoint files:

- `snsaug_aware_multihead_forensics_v1_best.pt`
- `snsaug_aware_multihead_forensics_v1_last.pt`

## Real Checkpoint Format

Full-curriculum checkpoints must be real PyTorch model checkpoints, not scalar training summaries. Each `.pt` file uses:

```text
checkpoint_format = snsaug_v2_real_state_dict_v1
```

Required checkpoint fields include:

- `model_state_dict`
- `optimizer_state_dict`
- `global_step`
- `phase`
- `config`
- `metrics`
- `trainable_components`

The `model_state_dict` is loaded from the real pre-SNS v3 architecture and updated by the guarded non-dry-run branch. Proxy-only checkpoint payloads containing only `trainable_state`, scalar biases, or synthetic state are invalid because the 0061B fixed-pair evaluator cannot run real checkpoint inference from them.

Small local sanity runs can use `max_steps_per_phase`, `phase_1_max_steps`, `phase_2_max_steps`, and `phase_3_max_steps` with tiny values such as 30 or less. Approved medium runs may raise those phase steps up to `max_allowed_steps_per_phase`, which defaults to 500.

`max_allowed_steps_per_phase` values above 500 require `allow_long_run_after_medium_pass=true` and are capped at 2000. Values such as 5000 are rejected. Unit validation should keep actual runner execution tiny and use validator-only checks for medium or long-run limits.

Subset evaluation summaries must be marked with `eval_subset_only=true`, `full_evaluation_ran=false`, and `sample_count`. Do not treat these as full benchmark metrics.

## Troubleshooting

If a tmux session disappears quickly and the log or printed JSON shows `training_started=false` for a non-dry-run invocation, the actual training branch did not run. Validate dry-run separately from the actual run, then inspect `artifact_manifest.json`.

For a successful non-dry-run actual branch, `artifact_manifest.json` must include:

- `training_started=true`
- `checkpoint_written=true`
- `best_checkpoint_path`
- `last_checkpoint_path`
- `best_checkpoint_sha256`
- `last_checkpoint_sha256`
- `checkpoint_format=snsaug_v2_real_state_dict_v1`
- `real_weight_checkpoint=true`

If 0061B reports `fine-tuned checkpoint has only trainable_state proxy values`, the checkpoint came from an old proxy branch or an invalid writer. Rerun the guarded full-curriculum branch after this fix and verify that `validate_real_weight_checkpoint` passes on both best and last checkpoints.
