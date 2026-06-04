# SNSAug V2 Fine-Tuning

`SNSAUG_V2_AWARE_FINETUNING_OK`

This module adds guarded SNS-aware fine-tuning support on top of the existing detector/localizer stack. It is designed to use train-only manifests, clean/basic/SNS curricula, `ignore_mask`-aware localization loss, and masked family supervision.

## Data Safety

- training input must come from `snsaug_v2_training_manifest.jsonl`
- training manifest rows must be `split == train`
- validation and test remain evaluation-only
- validation failure samples must not be used directly for training

## Approval Guard

Actual training is only allowed when the config contains the exact approval text:

`I_APPROVE_SNSAUG_V2_FINETUNE`

Without that exact value, config validation fails.

## Curriculum

Supported curriculum:

- epoch 1-3: clean 50%, basic 30%, sns light 20%
- epoch 4-8: clean 35%, basic 30%, sns medium 35%
- epoch 9+: clean 25%, basic 25%, sns medium or heavy 50%

## Loss Structure

Total loss follows:

- class loss
- localization loss with `ignore_mask`
- optional family loss only when family supervision exists
- optional consistency loss between clean and SNS-aware paired views

## Validation Reporting

Validation hooks are expected for:

- clean validation
- basic SNS validation
- realistic SNSAug validation

Reported metrics include:

- 3-way accuracy
- macro-F1
- real FPR
- synthetic recall
- tampered recall
- tampered mask IoU
- localization activation recall
- latency and FPS when available

## Output Restrictions

- save checkpoints only under approved external output roots
- save config copy and artifact manifest
- do not write outputs or checkpoints inside repository
