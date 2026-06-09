# Task 0064e: Warm-start Nuisance Model and Stabilize Gated Training

## Context

0064d fixed the actual training loop. Tiny 5x3 and 30x3 now execute the requested number of steps.

However, 0064d 30x3 comparison shows degenerate behavior:

- accuracy around 0.333 and macro_f1 around 0.167
- synthetic_recall = 0.0
- tampered_recall = 0.0
- localization_activation_recall = 1.0
- non_tampered_high_mask_rate = 1.0

This means the model is not robust. It is collapsing in classification while over-activating masks.

Do not run 0064d 150x3. Fix warm-start, model capacity, gating schedule, and loss balancing first.

## Goal

Implement 0064e nuisance training stabilization.

The nuisance model must:

1. warm-start the full shared backbone / class head / tamper mask head from an existing pre-SNS or SNSAug checkpoint
2. initialize only the new SNS/nuisance, degradation, and reliability heads from scratch
3. avoid early gating collapse by using a staged gating schedule
4. penalize non-tampered high masks
5. preserve 3-way classification behavior while learning nuisance masks

## Files Codex May Modify

- `tasks/0064e-warmstart-nuisance-gated-training-fix.md`
- `src/cv_forensics/snsaug_v2_nuisance_model.py`
- `src/cv_forensics/snsaug_v2_nuisance_losses.py`
- `src/cv_forensics/snsaug_v2_nuisance_finetune.py`
- `scripts/training/run_snsaug_v2_nuisance_finetune.py`
- `scripts/agent/validate_snsaug_v2_nuisance_finetune_config.py`
- `configs/training/snsaug_v2_nuisance_finetune.example.json`
- `tests/test_snsaug_v2_nuisance_finetune.py`
- `docs/snsaug_v2_nuisance_finetune.md`

## Files Claude May Modify

- `tasks/0064e-warmstart-nuisance-gated-training-fix.md`
- `src/cv_forensics/snsaug_v2_nuisance_model.py`
- `src/cv_forensics/snsaug_v2_nuisance_losses.py`
- `src/cv_forensics/snsaug_v2_nuisance_finetune.py`
- `scripts/training/run_snsaug_v2_nuisance_finetune.py`
- `scripts/agent/validate_snsaug_v2_nuisance_finetune_config.py`
- `configs/training/snsaug_v2_nuisance_finetune.example.json`
- `tests/test_snsaug_v2_nuisance_finetune.py`
- `docs/snsaug_v2_nuisance_finetune.md`

## Required Behavior

1. Full-model warm start

- Load backbone, class_head, and tamper_mask_head from base_model_bundle_path, pre_sns_best_bundle_path, or a configured warm_start_checkpoint_path.
- Support warm-start from snsaug_0063b_hardneg_30x3 checkpoint if configured.
- New heads may be randomly initialized: sns_nuisance_mask_head, global_degradation_head, reliability_head.
- Save warm_start_report.json with loaded keys, missing keys, unexpected keys, loaded_numel, total_numel.
- Fail if loaded_numel is suspiciously small unless allow_partial_warm_start=true.

2. Checkpoint size guard

- Best/last checkpoints must contain real model_state_dict.
- Record tensor_total_numel in artifact_manifest.json.
- Fail audit if tensor_total_numel is far smaller than the warm-start checkpoint.

3. Staged gating schedule

Phase 1:
- gating_alpha_effective = 0.0
- freeze backbone/class/tamper heads by default
- train sns_nuisance_mask_head and global_degradation_head

Phase 2:
- gating_alpha_effective ramp from 0.0 to min(config.gating_alpha, 0.2)
- unfreeze class head with low LR or keep frozen if unstable
- keep tamper mask head stable

Phase 3:
- gating_alpha_effective <= 0.3 by default
- allow joint tuning only if classification collapse guard passes

4. Loss balancing

Keep existing losses and add/ensure:

- L_non_tampered_mask_suppression: real/synthetic should not have large tamper masks
- L_class_preservation: clean/SNS paired views should preserve real/synthetic/tampered class
- L_tamper_mask_valid only for tampered valid regions
- L_sns_nuisance_mask uses ignore_mask local regions
- L_global_degradation for global augmentation metadata
- L_mask_area_regularization to prevent all-one masks

5. Collapse guards during training

Log per phase:

- pred_class_counts
- mean_p_real
- mean_p_synthetic
- mean_p_tampered
- mean_tamper_mask_area_real
- mean_tamper_mask_area_synthetic
- mean_tamper_mask_area_tampered
- sns_mask_iou if available
- gating_alpha_effective

If all predictions collapse to one class for multiple checkpoints, mark checkpoint as failed and do not select it as best.

6. Best checkpoint policy

Prefer checkpoints satisfying:

- synthetic_recall > 0.2 in eval subset
- tampered_recall > 0.2 in eval subset
- non_tampered_high_mask_rate < 0.5
- no single-class prediction collapse

If no checkpoint passes, still write last checkpoint but mark best_checkpoint_selected_by=fallback_last_no_guardrail_pass.

7. Tests

- warm-start report is written
- model_state_dict tensor_total_numel is comparable to warm-start checkpoint
- phase 1 gating alpha is 0
- phase counts match requested steps
- non-tampered mask suppression loss is positive for all-one masks
- dry-run writes no checkpoints
- tiny actual run writes best/last checkpoints
- collapse guard detects all-real or all-tampered fake records

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_nuisance_finetune_config.py configs/training/snsaug_v2_nuisance_finetune.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_nuisance_finetune.py
- python3 scripts/training/run_snsaug_v2_nuisance_finetune.py --help
- grep -q SNSAUG_V2_NUISANCE_MASK_FINETUNE_OK docs/snsaug_v2_nuisance_finetune.md
- python3 scripts/agent/check_agent_changes.py tasks/0064e-warmstart-nuisance-gated-training-fix.md

## Marker

SNSAUG_V2_NUISANCE_MASK_FINETUNE_OK
