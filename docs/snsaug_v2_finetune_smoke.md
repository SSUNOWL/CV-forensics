# SNSAug V2 Fine-Tune Smoke

SNSAUG_V2_FINETUNE_SMOKE_OK

This task adds guarded infrastructure for the first SNSAug-aware fine-tuning smoke run. It is not full training. The smoke path is restricted to `SNSAug-aware Multi-head Forensics Model v1`, which keeps the existing architecture as much as practical and targets the class head plus tamper localization head.

## Guardrails

Real smoke training requires the exact approval text `I_APPROVE_SNSAUG_V2_FINETUNE_SMOKE`. Config validation also requires `smoke_only=true` and `no_full_training=true`. It rejects validation or test rows in the training manifest, evaluation/fixed-pair roots as training input, repo-local output roots, repo-local checkpoint roots, protected paths, `max_steps` above the configured smoke limit, and `epochs > 1`.

Outputs must go under an approved external run root, and checkpoints must go under an approved external checkpoint root. Network access and downloads are disabled by config guardrails.

## Dry-Run vs Actual Smoke Run

`--dry-run` validates the config and prints the smoke plan only. It must report `training_started=false` and `checkpoint_written=false`, and it must not create a checkpoint directory.

Running without `--dry-run` enters the actual short smoke branch after guardrails pass. The smoke branch loads the base bundle metadata and train-only curriculum manifest, samples a small class-balanced subset, runs CPU-safe class-head and tamper-localization-head updates for the configured smoke step count, and writes a checkpoint under the external checkpoint root.

If a non-dry-run artifact reports `training_started=false`, the actual training branch did not run. Treat that as a runner failure or guardrail-path regression, not as a successful smoke fine-tune.

## Losses

The smoke loss design is:

```text
L_total =
  L_class
+ lambda_mask * L_tamper_mask_valid
+ lambda_score * L_tampered_score_consistency
+ lambda_consistency * L_clean_sns_class_consistency
+ lambda_hardneg * L_real_synthetic_sns_hard_negative
+ optional family loss where family_loss_mask == 1
```

`L_tamper_mask_valid` uses `valid_region = 1 - ignore_mask`, so SNS nuisance pixels are excluded from tamper localization loss. Real and synthetic SNSAug samples contribute as hard negatives by penalizing high tampered probability.

## Smoke Outputs

An approved real smoke run is expected to produce:

- `smoke_train_log.jsonl`
- `smoke_train_summary.json`
- `loss_breakdown.json`
- `snsaug_sampling_summary.json`
- `smoke_eval_clean_summary.json`
- `smoke_eval_0058c_summary.json`
- `artifact_manifest.json`

The smoke checkpoint is written under `checkpoint_root` and records that only `class_head` and `tamper_localization_head` were trainable. `artifact_manifest.json` must include `training_started=true` and `checkpoint_written=true` for non-dry-run execution.

Evaluation is planned for a clean validation subset and the fixed 0058c SNSAug benchmark. The repair path may write lightweight smoke placeholders marked `eval_subset_only=true` and `full_evaluation_ran=false`; it must not claim that full evaluation ran unless the full evaluator actually executed. Reported metrics include clean accuracy, macro-F1, tampered recall, valid IoU, SNSAug per-profile accuracy, SNSAug tampered recall, localization activation recall, real FPR, synthetic recall, and non-tampered high mask rate.
