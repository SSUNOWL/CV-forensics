# Task 0054: SNSAug V2 Aware Detector/Localizer Fine-Tuning

## Task Title

Implement SNS-aware fine-tuning for the existing detector/localizer using clean, basic augmentation, and SNSAug V2 views while preserving clean performance and improving robustness.

## Role

Codex-only implementation worker, reviewer, and limited repair manager operating under the repository contract and Codex-only workflow.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0054-snsaug-v2-aware-finetuning.md`
- `tasks/0053-snsaug-v2-training-manifest-and-dataset-wrapper.md`
- `tasks/0052-realistic-snsaug-v2-robustness-eval-and-mining.md`
- `tasks/0051-snsaug-v2-realistic-overlay-module-and-pair-generator.md`
- `docs/snsaug_v2_training_manifest.md`
- `docs/snsaug_v2_robustness_eval.md`
- `docs/snsaug_v2.md`
- `.local/pre_sns_current_best_model_bundle.json`
- `src/cv_forensics/snsaug_v2_training_manifest.py`
- `src/cv_forensics/snsaug_v2_dataset_wrapper.py`
- `src/cv_forensics/snsaug_v2/__init__.py`
- `src/cv_forensics/snsaug_v2/sns_augmentor.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_training.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_model.py`
- `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
- `tests/test_snsaug_v2_training_manifest.py`
- Any existing safe training, checkpoint-loading, loss, metrics, and reporting helpers already in `src/cv_forensics/`, `scripts/`, `tests/`, and `docs/` that are directly needed for finetuning implementation, config validation, and artifact writing

## Files Codex May Modify

- `tasks/0054-snsaug-v2-aware-finetuning.md`
- `configs/training/snsaug_v2_finetune.example.json`
- `scripts/agent/validate_snsaug_v2_finetune_config.py`
- `scripts/training/run_snsaug_v2_finetune.py`
- `src/cv_forensics/snsaug_v2_finetune.py`
- `tests/test_snsaug_v2_finetune.py`
- `docs/snsaug_v2_finetune.md`

## Forbidden Actions

- Do not modify any file outside the allowed modify list.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, or unrelated protected directories.
- Do not inspect protected directories recursively.
- Do not use validation or test samples for training.
- Do not use network resources.
- Do not download datasets or assets.
- Do not install packages.
- Do not write outputs or checkpoints inside the repository.
- Do not create large files beyond normal config/log/test artifacts.
- Do not modify unrelated model training code.
- Do not commit changes.

## Important Project Facts

- This task implements SNS-aware fine-tuning logic and configuration, not data downloading.
- Training data must come from train split only through `snsaug_v2_training_manifest.jsonl`.
- Validation and test remain evaluation-only.
- Validation failure samples must not be used directly for training.
- The pre-SNS current-best bundle is the initialization source.
- Family loss must be masked when family supervision is unavailable.
- Localization loss must respect `ignore_mask`.
- Actual training is allowed only when explicit approval text is present:
  - `I_APPROVE_SNSAUG_V2_FINETUNE`

## Implementation Requirements

### 1. Add Config, Validator, and Runner

Add:

- `configs/training/snsaug_v2_finetune.example.json`
- `scripts/agent/validate_snsaug_v2_finetune_config.py`
- `scripts/training/run_snsaug_v2_finetune.py`

The config validator must:

- require train manifest split == `train`
- reject validation or test samples in the training manifest
- require output root outside repository
- require approved checkpoint input roots
- require `no_network == true`
- require `no_download == true`
- allow training only when the exact user approval text is present:
  - `I_APPROVE_SNSAUG_V2_FINETUNE`

### 2. Add Fine-Tuning Module

Add:

- `src/cv_forensics/snsaug_v2_finetune.py`

This module must load the frozen or pre-SNS best bundle as initialization and save checkpoints only under approved external output roots.

It must also save:

- training config copy
- artifact manifest

### 3. Training Data

- use `snsaug_v2_training_manifest.jsonl`
- use train split only
- do not use validation failure samples for training
- validation and test are evaluation only

### 4. Training Mix

Support default mix:

- clean 30%
- basic_aug 30%
- sns_aug 40%

Support curriculum:

- Epoch 1-3:
  - clean 50%
  - basic_aug 30%
  - sns_light 20%
- Epoch 4-8:
  - clean 35%
  - basic_aug 30%
  - sns_medium 35%
- Epoch 9+:
  - clean 25%
  - basic_aug 25%
  - sns_medium_or_heavy 50%

### 5. Loss

Implement total loss:

```text
L_total =
  L_class
+ lambda_mask * L_mask
+ lambda_family * L_family
+ lambda_consistency * L_consistency optional
```

Mask loss:

- use `ignore_mask`
- `valid_region = 1 - ignore_mask`
- `mask_loss = BCE(pred_mask, tamper_mask) * valid_region`
- optional Dice-style component may be included if narrow and well-justified

Family loss:

- apply only when `family_label` exists and `family_loss_mask == 1`
- SID-Set samples without family labels must not contribute to family loss

Consistency loss:

- optional
- clean and SNSAug views with same `base_id` should have similar class logits
- apply lower or zero family consistency weight for heavy SNSAug if implemented

### 6. Validation During Training

Implement validation reporting for:

- clean validation
- basic SNS validation
- realistic SNSAug validation

Report:

- 3-way accuracy
- macro-F1
- real FPR
- synthetic recall
- tampered recall
- tampered mask IoU
- localization activation recall
- latency and FPS if available

### 7. Checkpoints and Artifacts

- save checkpoints under approved external `output_root` only
- save training config copy
- save artifact manifest
- do not write outputs or checkpoints inside repository

### 8. Tests

Add:

- `tests/test_snsaug_v2_finetune.py`

Tests must cover:

- config validator rejects validation or test training samples
- loss masking with `ignore_mask`
- family loss mask-out
- curriculum schedule
- output schema

Tests must remain CPU-safe and must not require real training runs.

### 9. Docs

Add:

- `docs/snsaug_v2_finetune.md`

Include:

- fine-tuning overview
- train-only data requirement
- approval-text requirement
- curriculum and loss structure
- validation reporting expectations
- checkpoint/output restrictions
- marker `SNSAUG_V2_AWARE_FINETUNING_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_snsaug_v2_finetune_config.py configs/training/snsaug_v2_finetune.example.json
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_finetune.py
```

```bash
grep -q SNSAUG_V2_AWARE_FINETUNING_OK docs/snsaug_v2_finetune.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0054-snsaug-v2-aware-finetuning.md
```

## Acceptance Criteria

- Fine-tuning config and validator enforce approval text and train-only manifest safety.
- SNS-aware training mix and curriculum are explicit in code.
- Mask loss respects `ignore_mask`.
- Family loss is masked when family supervision is unavailable.
- Validation reporting hooks exist for clean/basic/SNSAug evaluation.
- Checkpoints and artifacts are written only outside repository.
- Tests and validator pass without real training.
- Docs include the required marker and guardrails.

## Stop Condition

Stop after implementing only the allowed files, running the validation commands, running `python3 scripts/agent/check_agent_changes.py tasks/0054-snsaug-v2-aware-finetuning.md`, and reviewing the result in Korean as `PASS` or `NEEDS_FIX`. Do not start real training unless the required approval text is present and the task has been committed under the repository workflow.
