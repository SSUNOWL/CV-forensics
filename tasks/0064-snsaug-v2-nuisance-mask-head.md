# Task 0064: SNSAug v2 Nuisance Mask Head and Mask-Guided Tamper Gating

## Context

0063b balanced hard-negative 30x3 real-run completed successfully, but it did not solve the core over-tampered failure.

Observed direction from 0063b:

- The real-run wrote real checkpoints.
- Hard-negative loss was logged.
- The repaired checkpoint comparison evaluator can compare baseline / 0060b / 0063b.
- However, SNS platform profiles still show excessive tampered predictions or poor real/synthetic separation.
- Threshold sweep does not provide a good operating point.

Conclusion:

Threshold calibration and hard-negative suppression alone are insufficient. The model needs to explicitly separate benign SNS/local overlay artifacts from malicious tamper evidence.

This task follows the project proposal direction: a lightweight multi-head image forensics prototype using real / synthetic / tampered classification, tampered localization, and social-media perturbation robustness.

## Goal

Implement an SNS/Nuisance mask head and mask-guided tamper feature gating.

Core phrase: Mask-guided tamper feature gating.

The model should learn:

1. class_head: real / synthetic / tampered
2. tamper_mask_head: malicious manipulation region
3. sns_nuisance_mask_head: local SNS/UI/text/sticker/watermark/overlay region
4. global_degradation_head: JPEG/resize/screenshot/recompression/global degradation type
5. optional reliability_head: reduced reliability around nuisance regions

## Key Design

### Local SNS/Nuisance mask

Use ignore_mask from SNSAug pair generation as supervision for local overlay artifacts.

Examples:

- platform UI bars
- text blocks
- sticker / emoji
- speech bubbles
- visible watermark
- red circles / arrows
- news / meme banners
- platform buttons / icons

Target:

- sns_nuisance_mask = ignore_mask_local

Important:

- Do not treat full-image JPEG / compression / resize as a local mask.
- For global degradation-only profiles, local SNS mask can be all-zero while degradation label is non-zero.

### Global degradation type

Use metadata/profile to create a multi-hot vector:

- jpeg
- resize
- crop
- screenshot_resampling
- platform_layout
- overlay_text
- sticker
- news_meme
- combined_sns
- severity_light
- severity_medium

### Mask-guided tamper feature gating

Implement soft gating into the tamper decoder.

Preferred simple form:

    F_tamper = F * (1.0 - alpha * downsample(sns_mask_or_pred_sns_mask))

Default:

- alpha = 0.5
- train-time teacher forcing probability using GT sns mask = 0.5
- otherwise use predicted SNS mask, detached initially for stability

The point is not to erase image evidence completely. The point is to reduce the tendency to interpret SNS overlay as malicious tamper.

## Required Files

Add or update:

- src/cv_forensics/snsaug_v2_nuisance_model.py
- src/cv_forensics/snsaug_v2_nuisance_losses.py
- src/cv_forensics/snsaug_v2_nuisance_finetune.py
- scripts/training/run_snsaug_v2_nuisance_finetune.py
- scripts/agent/validate_snsaug_v2_nuisance_finetune_config.py
- configs/training/snsaug_v2_nuisance_finetune.example.json
- tests/test_snsaug_v2_nuisance_finetune.py
- docs/snsaug_v2_nuisance_finetune.md

## Losses

Implement:

    L_total =
        L_class
      + lambda_tamper_mask * L_tamper_mask_valid
      + lambda_sns_mask * L_sns_nuisance_mask
      + lambda_degradation * L_global_degradation
      + lambda_gating_consistency * L_clean_sns_class_consistency
      + lambda_hardneg * L_non_tampered_tampered_suppression

Defaults:

- lambda_sns_mask = 1.0
- lambda_degradation = 0.3
- lambda_hardneg = 1.0
- lambda_tamper_mask = 1.0
- lambda_gating_consistency = 0.5
- gating_alpha = 0.5

## Training Data Policy

Use train-only SNSAug curriculum manifest.

Allowed training inputs:

- train curriculum manifest from 0059
- generated train-only SNSAug samples
- train split only

Forbidden:

- fixed validation pair root as training input
- 0058c eval pair root as training input
- validation manifest as training input
- network/download

## Evaluation

Use the repaired 0061 checkpoint comparison evaluator.

Required comparison:

- pre_sns_baseline
- snsaug_0060b_over_tampered_30x3
- snsaug_0063b_hardneg_30x3
- snsaug_0064_nuisance_30x3

Metrics:

- accuracy
- macro_f1
- real_fpr
- synthetic_recall
- tampered_recall
- localization_activation_recall
- tampered_valid_mean_iou
- non_tampered_high_mask_rate
- sns_nuisance_mask_iou
- degradation_type_macro_f1
- threshold_sweep_candidate_count

Success criteria for first 30x3:

- tiktok_like real_fpr <= 0.40
- instagram_story_like real_fpr <= 0.40
- youtube_shorts_like real_fpr <= 0.40
- combined_sns_realistic real_fpr <= 0.40
- synthetic_recall >= 0.30 on SNS profiles
- tampered_recall >= 0.50 on SNS profiles
- candidate_count > 0 in threshold sweep

## Guardrails

- no network
- no download
- no validation data for training
- train-only manifest only
- outputs outside repository
- checkpoints must contain real model_state_dict
- reject proxy trainable_state-only checkpoints
- preserve tamper_mask and sns_nuisance_mask separation
- do not mark SNS local overlay as tamper

## Tests

Tests must cover:

- SNS nuisance mask target generated from ignore_mask
- global degradation profiles produce degradation labels
- JPEG-only profile has no local SNS mask unless local overlay exists
- tamper mask loss excludes ignore_mask region
- SNS mask loss uses local ignore-mask region
- gating reduces tamper feature magnitude in SNS areas
- real checkpoint includes model_state_dict
- train split only
- no network / no download
- dry-run does not train
- tiny real-run writes best/last checkpoints

## Expected Output Marker

SNSAUG_V2_NUISANCE_MASK_FINETUNE_OK
