# SNSAug V2 Train Curriculum Manifest

SNSAUG_V2_TRAIN_CURRICULUM_MANIFEST_OK

This document describes the train-only SNSAug V2 curriculum manifest builder. The builder creates manifest and schedule artifacts only; it does not train, fine-tune, download data, touch checkpoints, or write generated outputs inside the repository.

## Purpose

SNSAug V2 evaluation showed that the pre-SNS model remains strong on clean samples but often loses p_tampered under SNS/crop/resize/canvas profiles. Because localization is conditional on the tampered score, predicted redmasks become empty when the gate turns off. Oracle-gate diagnostics also showed that some geometry and screenshot profiles recover IoU when localization is forced, while platform-layout profiles remain hard for the mask decoder.

The curriculum is designed to improve both activation robustness and mask decoder robustness using train split samples only.

## Outputs

The builder writes these files under an approved external output root:

- `snsaug_v2_train_curriculum_manifest.jsonl`
- `snsaug_v2_curriculum_schedule.json`
- `snsaug_v2_profile_sampling_weights.json`
- `snsaug_v2_class_balance_summary.json`
- `snsaug_v2_training_guardrails.json`
- `snsaug_v2_training_intent_and_references.md`
- `artifact_manifest.json`

Each JSONL record includes `base_id`, `source_dataset`, `split`, `image_path`, optional `tamper_mask_path`, `content_label`, optional `family_label`, `family_loss_mask`, `allowed_profile_groups`, `profile_sampling_weights_by_phase`, `severity_schedule`, `sampling_weight`, `label_preserved`, and `training_notes`.

## Guardrails

Only `split=train` rows are accepted. Validation/test fixed pairs and evaluation result paths are rejected as training sources. The output root must be outside the repository and under an approved output root. The generated guardrails file records `no_training`, `no_finetune`, `no_network`, `no_download`, `train_only`, `output_root_outside_repository`, `validation_samples_rejected`, and `evaluation_outputs_rejected_as_training_images`.

The manifest keeps labels unchanged after SNSAug. `family_loss_mask` is `1` only when a non-empty family label exists; otherwise it is `0`. SNS nuisance regions must be excluded from tamper loss through `valid_region = 1 - ignore_mask`.

## Curriculum

Profile groups are fixed:

- `clean`: no augmentation
- `geometry_light`: `canvas_9x16_only`, `resize_crop_pad`, `zoom_crop`
- `postprocess_light`: `recompression_light`, `resize_jpeg`
- `screenshot_light`: `screenshot_recapture_light`
- `overlay_light`: `platform_ui_same_size`, `news_meme_overlay`
- `platform_layout`: `tiktok_like`, `instagram_story_like`, `youtube_shorts_like`
- `combined`: `combined_sns_realistic`

The three phases are:

- Phase 1: clean 45%, geometry_light 25%, postprocess_light 15%, overlay_light 10%, screenshot_light 5%, platform_layout 0%, combined 0%
- Phase 2: clean 35%, geometry_light 25%, postprocess_light 15%, screenshot_light 10%, overlay_light 5%, platform_layout 10%, combined 0%
- Phase 3: clean 25%, geometry_light 25%, postprocess_light 15%, screenshot_light 10%, overlay_light 5%, platform_layout 15%, combined 5%

Each phase sums to 1.0.

## Compatibility

The manifest is intended for the guarded 0054 fine-tuning infrastructure and SNSAug V2 DatasetWrapper. It supports on-the-fly SNSAug generation, ignore-mask valid-region localization loss, clean/SNS class consistency, tampered-score consistency, real/synthetic SNS hard negatives, and class-balanced sampling.
