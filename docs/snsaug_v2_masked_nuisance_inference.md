# SNSAug V2 Masked Nuisance Inference

SNSAUG_V2_MASKED_NUISANCE_INFERENCE_OK

0067 treats SNSAug overlays as an inference-time forensic distraction layer. Prior SNSAug fine-tuning recovered tampered activation mainly by over-predicting tampered, while the frozen pre-SNS baseline preserved real/synthetic discrimination but lost tampered activation under platform overlays. This branch tests whether local `ignore_mask` regions are the immediate cause before implementing full masked convolution.

The evaluator uses the frozen pre-SNS best bundle and the fixed SNSAug pair root. It does not train, fine-tune, download, modify checkpoints, or write inside the repository. A dry-run validates the configuration and returns planned outputs without creating records.

## Masking Policies

Each fixed-pair row is evaluated under:

- `original`: no pixel masking
- `gray_fill`: fill `ignore_mask` pixels with neutral gray
- `blur_fill`: blur the image and paste blurred pixels only inside `ignore_mask`
- `mean_fill`: fill `ignore_mask` pixels with image mean color
- `black_fill`: fill `ignore_mask` pixels with black
- `dilated_gray_fill`: dilate `ignore_mask`, then gray-fill

These policies are an approximation inspired by Ignoring the Decoy. They test whether removing local nuisance pixels restores the frozen baseline before adding architecture-level masked convolution.

## Outputs

Real runs write:

- `model_eval_records_masked.jsonl`
- `per_policy_per_profile_metrics.json`
- `clean_vs_sns_masked_delta.json`
- `masked_nuisance_inference_summary.json`
- `masked_nuisance_inference_report.md`
- `visual_gallery_manifest.json`

Metrics are grouped by masking policy and profile: accuracy, macro-F1, real FPR, synthetic recall, tampered recall, localization activation recall, valid IoU, non-tampered high-mask rate, mean `p_tampered` on tampered rows, and nuisance mask area ratio.

Recovery metrics compare each masked policy against `original` for the same profile:

- `tampered_recall_recovery`
- `valid_iou_recovery`
- `p_tampered_recovery_on_tampered`
- `synthetic_recall_change`
- `real_fpr_change`

A policy is marked promising only when tampered recall and valid IoU improve on at least two SNS platform profiles, synthetic recall does not collapse, and real FPR does not increase severely.
