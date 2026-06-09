# Task 0064g: Hybrid Warm-start and Synthetic Preservation for SNSAug Nuisance Model

## Context

0064f fixed training loop, warm-start provenance, and real-weight checkpoint audit.
The 0064f 30x3 run is valid, but it is not a successful final model.

Observed 0064f behavior:

- 30x3 audit passes
- global_step = 90
- warm_start loaded_ratio around 0.999
- real-weight checkpoint exists
- threshold sweep candidate_count = 0
- SNS profiles still have high real_fpr
- synthetic_recall is 0.0 on key profiles
- tampered_recall is partially recovered but mostly through over-tampered bias

Interpretation:

0064f solved infrastructure and provenance, but it inherits too much over-tampered class bias from the 0063b warm-start. The next fix should separate warm-start sources and explicitly preserve synthetic classification.

## Goal

Implement 0064g hybrid warm-start and synthetic-preservation training.

The model should preserve synthetic recognition while keeping tampered localization robust under SNS perturbations.

## Files Codex May Modify

- `tasks/0064g-hybrid-warmstart-synthetic-preservation.md`
- `src/cv_forensics/snsaug_v2_nuisance_model.py`
- `src/cv_forensics/snsaug_v2_nuisance_losses.py`
- `src/cv_forensics/snsaug_v2_nuisance_finetune.py`
- `scripts/training/run_snsaug_v2_nuisance_finetune.py`
- `scripts/agent/validate_snsaug_v2_nuisance_finetune_config.py`
- `configs/training/snsaug_v2_nuisance_finetune.example.json`
- `tests/test_snsaug_v2_nuisance_finetune.py`
- `docs/snsaug_v2_nuisance_finetune.md`

## Files Claude May Modify

- `tasks/0064g-hybrid-warmstart-synthetic-preservation.md`
- `src/cv_forensics/snsaug_v2_nuisance_model.py`
- `src/cv_forensics/snsaug_v2_nuisance_losses.py`
- `src/cv_forensics/snsaug_v2_nuisance_finetune.py`
- `scripts/training/run_snsaug_v2_nuisance_finetune.py`
- `scripts/agent/validate_snsaug_v2_nuisance_finetune_config.py`
- `configs/training/snsaug_v2_nuisance_finetune.example.json`
- `tests/test_snsaug_v2_nuisance_finetune.py`
- `docs/snsaug_v2_nuisance_finetune.md`

## Required Behavior

1. Hybrid warm-start

Support separate checkpoint sources:

- class_backbone_warm_start_path: preferably pre-SNS/current best bundle or baseline model
- tamper_head_warm_start_path: optionally 0063b/0060b checkpoint for tamper localization recovery
- nuisance heads remain newly initialized

Default policy:

- backbone and class_head should not inherit 0063b over-tampered class bias if pre-SNS baseline is available
- tamper_mask_head may inherit from 0063b or 0060b
- sns_nuisance_mask_head, global_degradation_head, reliability_head are initialized from scratch

Write hybrid_warm_start_report.json with:

- class_backbone_warm_start_path
- tamper_head_warm_start_path
- loaded keys by component
- loaded_numel by component
- missing/unexpected keys by component
- total model numel

2. Synthetic preservation

Add or strengthen losses:

- L_synthetic_preservation: synthetic samples should keep synthetic probability above configured floor
- L_non_tampered_tampered_suppression: real/synthetic should not become tampered
- L_clean_sns_class_consistency: clean/SNS paired views should preserve class distribution
- L_class_balanced_ce: class-balanced CE over real/synthetic/tampered

Recommended defaults:

- lambda_synthetic_preservation = 2.0
- synthetic_probability_floor = 0.35
- lambda_non_tampered_tampered_suppression = 2.0
- lambda_class_preservation = 1.0

3. Balanced sampling

For SNS-heavy phases, enforce class-balanced sampling:

- real:syn:tampered approximately 1:1:1
- within non-tampered, preserve enough synthetic SNS samples
- do not oversample tampered SNS more than non-tampered

4. Gating schedule

Keep 0064f staged gating, but prevent class collapse:

- Phase 1: gating_alpha_effective = 0.0
- Phase 2: gating_alpha_effective <= 0.15
- Phase 3: gating_alpha_effective <= 0.25 unless synthetic_recall guard passes

5. Best checkpoint policy

Do not select best checkpoint if:

- synthetic_recall == 0 on eval subset
- real_fpr > 0.60
- single-class prediction collapse occurs
- non_tampered_high_mask_rate > 0.50

Balanced score should reward:

- synthetic_recall
- tampered_recall
- valid_iou
- low real_fpr
- low high_mask

6. Logging

training_log.jsonl must include per phase:

- pred_class_counts
- mean_p_real
- mean_p_synthetic
- mean_p_tampered
- synthetic_recall_proxy
- real_fpr_proxy
- tampered_recall_proxy
- non_tampered_high_mask_rate_proxy
- gating_alpha_effective
- synthetic_preservation_loss

7. Tests

- hybrid_warm_start_report is written
- class/backbone keys can load from baseline path
- tamper head keys can load from 0063b path
- synthetic_preservation_loss is positive when p_synthetic is below floor
- class-balanced sampler includes synthetic samples
- best checkpoint guard rejects synthetic_recall=0
- tiny actual run writes best/last checkpoints
- dry-run writes no checkpoints

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_nuisance_finetune_config.py configs/training/snsaug_v2_nuisance_finetune.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_nuisance_finetune.py
- python3 scripts/training/run_snsaug_v2_nuisance_finetune.py --help
- grep -q SNSAUG_V2_NUISANCE_MASK_FINETUNE_OK docs/snsaug_v2_nuisance_finetune.md
- python3 scripts/agent/check_agent_changes.py tasks/0064g-hybrid-warmstart-synthetic-preservation.md

## Marker

SNSAUG_V2_NUISANCE_MASK_FINETUNE_OK
