# SNSAug V2 Fine-Tune Smoke

SNSAUG_V2_FINETUNE_SMOKE_OK

This task adds guarded infrastructure for the first SNSAug-aware fine-tuning smoke run. It is not full training. The smoke path is restricted to `SNSAug-aware Multi-head Forensics Model v1`, which keeps the existing architecture as much as practical and targets the class head plus tamper localization head.

## Guardrails

Real smoke training requires the exact approval text `I_APPROVE_SNSAUG_V2_FINETUNE_SMOKE`. Config validation rejects validation or test rows in the training manifest, evaluation/fixed-pair roots as training input, repo-local output roots, repo-local checkpoint roots, protected paths, `max_steps` outside `1..500`, and `epochs > 1`.

Outputs must go under an approved external run root, and checkpoints must go under an approved external checkpoint root. Network access and downloads are disabled by config guardrails.

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

Evaluation is planned for a clean validation subset and the fixed 0058c SNSAug benchmark. Reported metrics include clean accuracy, macro-F1, tampered recall, valid IoU, SNSAug per-profile accuracy, SNSAug tampered recall, localization activation recall, real FPR, synthetic recall, and non-tampered high mask rate.
