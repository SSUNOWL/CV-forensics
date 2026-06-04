# SNSAug V2 Robustness Evaluation

`SNSAUG_V2_ROBUSTNESS_EVAL_AND_MINING_OK`

This module evaluates the frozen pre-SNS current-best bundle against realistic SNSAug V2 profile variants. It is evaluation and mining only. It does not train, fine-tune, download, or use network resources.

## Overview

The pipeline runs:

- clean input
- `jpeg_resize`
- `tiktok_like`
- `instagram_story_like`
- `youtube_shorts_like`
- `news_meme_overlay`
- `ai_badge_overlay`
- `annotation_sticker`
- `combined_sns_realistic`

Each profile is generated deterministically with SNSAug V2, then scored by the frozen policy-gated report bundle.

## Record Join

For each sample, the evaluation joins clean and profile outputs and writes:

- clean/profile class outputs
- clean/profile IoU and Dice
- `p_tampered` drop
- localization activation flip
- ignore-mask and overlay area percentages
- latency and final mask source

## Fragile Mining Groups

Implemented mining groups:

- `stable_correct_anchor`
- `fragile_correct_to_fail`
- `confidence_fragile`
- `mask_iou_fragile`
- `threshold_flip`
- `clean_fail`
- `noisy_candidate`

`noisy_candidate` is deterministic: a tampered sample is marked noisy when clean IoU is very low (`< 0.10`) and predictions become unstable across non-clean profiles.

## Split Safety

- Validation outputs are diagnostic only.
- Validation failure cases must not be used directly for training.
- If a train manifest is provided, the pipeline writes a separate train-only mining manifest.
- The train-only mining manifest excludes validation sample IDs.

## Outputs

The pipeline writes external-only artifacts including:

- `snsaug_v2_robustness_records.jsonl`
- `snsaug_v2_per_profile_metrics.json`
- `snsaug_v2_robustness_summary.json`
- `snsaug_v2_worst_profiles.json`
- `snsaug_v2_worst_samples.json`
- `snsaug_v2_fragile_candidates_val.jsonl`
- `snsaug_v2_fragile_candidates_train.jsonl`
- `snsaug_v2_training_mining_manifest_train_only.jsonl`
- `visual_gallery_manifest.json`
- `artifact_manifest.json`

## Usage

```bash
python3 scripts/evaluation/run_snsaug_v2_robustness_eval.py configs/evaluation/snsaug_v2_robustness_eval.example.json
```
