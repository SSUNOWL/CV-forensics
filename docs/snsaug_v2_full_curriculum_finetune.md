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
