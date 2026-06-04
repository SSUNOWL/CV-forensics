## Task Title
0055 SNSAug V2 Source Manifest Audit, Overlay Generator, and Tiny Fixed Pairs

## Role
Codex-only task writer and later implementation worker/reviewer for SNSAug v2 data generation, preview, and tiny paired benchmark creation. This task is generation-only and must not train or fine-tune models.

## Files Codex May Read
- AGENTS.md
- CLAUDE.md
- docs/project_brief.md
- docs/project_contract.md
- configs/project_contract.json
- docs/agent_workflow.md
- tasks/0051-snsaug-v2-realistic-overlay-module-and-pair-generator.md
- tasks/0052-realistic-snsaug-v2-robustness-eval-and-mining.md
- tasks/0053-snsaug-v2-training-manifest-and-dataset-wrapper.md
- tasks/0054-snsaug-v2-aware-finetuning.md
- docs/snsaug_v2.md
- docs/snsaug_v2_robustness_eval.md
- docs/snsaug_v2_training_manifest.md
- docs/snsaug_v2_finetune.md
- configs/snsaug_v2/snsaug_v2.example.json
- src/cv_forensics/snsaug_v2/__init__.py
- src/cv_forensics/snsaug_v2/configs.py
- src/cv_forensics/snsaug_v2/transforms.py
- src/cv_forensics/snsaug_v2/ui_renderers.py
- src/cv_forensics/snsaug_v2/sticker_packs.py
- src/cv_forensics/snsaug_v2/platform_templates.py
- src/cv_forensics/snsaug_v2/sns_augmentor.py
- src/cv_forensics/snsaug_v2/pair_generator.py
- tests/test_snsaug_v2.py
- relevant validators, scripts, and current git status

## Files Codex May Modify
- tasks/0055-snsaug-v2-source-manifest-audit-overlay-generator-and-tiny-fixed-pairs.md
- configs/snsaug_v2/snsaug_v2_source_audit.example.json
- configs/snsaug_v2/snsaug_v2_overlay_preview.example.json
- configs/snsaug_v2/snsaug_v2_tiny_fixed_pairs.example.json
- scripts/agent/validate_snsaug_v2_generation_config.py
- scripts/snsaug_v2/preview_snsaug_v2_overlays.py
- scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py
- src/cv_forensics/snsaug_v2/__init__.py
- src/cv_forensics/snsaug_v2/configs.py
- src/cv_forensics/snsaug_v2/transforms.py
- src/cv_forensics/snsaug_v2/ui_renderers.py
- src/cv_forensics/snsaug_v2/sticker_packs.py
- src/cv_forensics/snsaug_v2/platform_templates.py
- src/cv_forensics/snsaug_v2/sns_augmentor.py
- src/cv_forensics/snsaug_v2/source_manifest_audit.py
- src/cv_forensics/snsaug_v2/pair_generator.py
- tests/test_snsaug_v2_generation.py
- docs/snsaug_v2_generation.md

## Forbidden Actions
- Do not train.
- Do not fine-tune.
- Do not run 0054.
- Do not use network.
- Do not download assets.
- Do not install packages.
- Do not use exact platform logos.
- Do not write generated outputs inside the repository.
- Do not modify unrelated model code.
- Do not access `.env`, `.env.*`, secrets, datasets, outputs, checkpoints, or protected directories beyond what is explicitly required for safe manifest validation.
- Do not assume a prebuilt SNSAug dataset exists.

## Important Project Facts
- The project is a lightweight multi-head image forensics system using Community Forensics-Small and SID-Set.
- It performs real / synthetic / tampered 3-way classification, tampered mask localization, generator-family provenance when available, and template-based evidence reporting.
- SNSAug v2 is benign degradation and overlay only. It must never change the underlying content label:
  - Real remains real.
  - Synthetic remains synthetic.
  - Tampered remains tampered.
- `tamper_mask` contains only malicious manipulation regions.
- UI, text, stickers, watermark, AI badges, news banners, emoji, red circle, arrows, platform frame, and screenshot bars are not tamper regions and must be written to `ignore_mask`.
- Geometric transforms must be applied identically to image and `tamper_mask`.
- Compression, blur, color shift, sharpen, and recompression affect image only.
- Mask resizing must always use nearest-neighbor interpolation.
- Existing `src/cv_forensics/snsaug_v2/*` modules from 0051 already exist and must be audited and extended rather than duplicated.
- All generated artifacts must be written outside the repository.

## Implementation Requirements
- Implement four phases within the allowed files only.

- Phase A: Source manifest audit
  - Add `src/cv_forensics/snsaug_v2/source_manifest_audit.py`.
  - Audit source manifests without assuming an existing SNSAug dataset.
  - Support JSONL or project-supported manifest formats.
  - Verify manifest exists and records can be parsed.
  - Each valid record must have or derive:
    - `base_id`
    - `image_path`
    - `content_label`
    - optional `tamper_mask_path`
    - optional `family_label`
    - `split`
    - `source_dataset`
  - Verify `image_path` exists and is readable.
  - Verify `tamper_mask_path` exists and is readable when present.
  - Map labels into:
    - `real`
    - `synthetic`
    - `tampered`
  - For tampered records, report mask availability.
  - Summarize records by class.
  - Inaccessible or invalid records must be reported and must not crash when `fail_fast` is false.
  - Required audit outputs:
    - `source_manifest_audit_summary.json`
    - `source_manifest_valid_records.jsonl`
    - `source_manifest_invalid_records.jsonl`
    - `source_manifest_class_counts.json`

- Phase B: SNSAug v2 overlay generator
  - Audit and extend, without duplication:
    - `src/cv_forensics/snsaug_v2/__init__.py`
    - `src/cv_forensics/snsaug_v2/configs.py`
    - `src/cv_forensics/snsaug_v2/transforms.py`
    - `src/cv_forensics/snsaug_v2/ui_renderers.py`
    - `src/cv_forensics/snsaug_v2/sticker_packs.py`
    - `src/cv_forensics/snsaug_v2/platform_templates.py`
    - `src/cv_forensics/snsaug_v2/sns_augmentor.py`
    - `src/cv_forensics/snsaug_v2/pair_generator.py`
  - Maintain backward compatibility with 0053 and 0054 code.
  - Ensure these core APIs exist and remain usable:
    - `SNSAugV2Config`
    - `SNSAugV2Result`
    - `SNSAugV2Augmentor`
  - `SNSAugV2Config` must support:
    - `profile`
    - `severity="medium"`
    - `output_size=None`
    - `seed=None`
    - `platform_template=None`
    - `p_recompression=0.7`
    - `p_platform_ui=0.8`
    - `p_ai_badge=0.4`
    - `p_news_banner=0.3`
    - `p_annotation_sticker=0.4`
    - `p_emoji_sticker=0.3`
    - `p_color_shift=0.2`
    - `p_blur_pixelation=0.2`
    - `font_path=None`
  - `SNSAugV2Result` must return:
    - augmented image
    - transformed `tamper_mask`
    - `ignore_mask`
    - augmentation metadata
  - Required profiles:
    - `clean`
    - `jpeg_resize`
    - `screenshot_basic`
    - `tiktok_like`
    - `instagram_story_like`
    - `youtube_shorts_like`
    - `news_meme_overlay`
    - `ai_badge_overlay`
    - `annotation_sticker`
    - `combined_sns_realistic`
  - Implement research-safe, PIL-drawn platform-like templates only. Do not use exact platform logos or downloaded assets.
  - Required platform/template behavior:
    - `tiktok_like`
      - portrait 9:16 or similar canvas
      - source image fit/crop/pad into main content area
      - top tabs like Following / For You or Korean equivalent
      - right-side action bar with simple drawn icons: profile, heart, comment, bookmark, share
      - bottom username/caption/music area
      - optional bottom navigation bar
      - optional AI badge
      - every UI/text/icon/badge box must be added to `ignore_mask`
    - `instagram_story_like`
      - portrait 9:16 canvas
      - top progress bars
      - account row and timestamp
      - optional translucent text sticker
      - optional poll/question/location sticker
      - bottom message input bar
      - every UI/text/sticker box must be added to `ignore_mask`
    - `youtube_shorts_like`
      - portrait 9:16 canvas
      - right-side action bar: like, dislike, comment, share, remix
      - bottom channel/title/subscribe area
      - optional bottom navigation
      - optional AI label chip
      - every UI/text/icon/badge box must be added to `ignore_mask`
    - `news_meme_overlay`
      - headline banner
      - lower-third caption
      - optional `속보`, `BREAKING`, `NEWS`, `LIVE`, `단독`, `확인 필요`
      - optional red circle / arrow / highlight box
      - every overlay box must be added to `ignore_mask`
    - `annotation_sticker`
      - red circle
      - red arrow
      - red rectangle
      - semi-transparent highlight box
      - speech bubble
      - emoji-like reaction sticker
      - every sticker box must be added to `ignore_mask`
    - `combined_sns_realistic`
      - randomly choose one platform template and combine optional AI badge, news/meme banner, annotation sticker, emoji/reaction, recompression, resize, or screenshot recapture
      - deterministic with seed
  - Use PIL / NumPy only unless existing project utilities already provide OpenCV safely.
  - Use optional local font paths only when available; otherwise use PIL default font. Do not add font files to repo.

- Phase C: Preview generation
  - Add `scripts/snsaug_v2/preview_snsaug_v2_overlays.py`.
  - It must accept:
    - input image path
    - optional mask path
    - output_root outside repo
    - profiles list
    - severity
    - seed
    - `--use_demo_image` if no input image is provided
  - For each profile, save:
    - augmented image
    - transformed `tamper_mask`
    - `ignore_mask`
    - metadata JSON
    - overlay debug image with boxes
  - Save a preview grid containing:
    - original
    - augmented
    - `tamper_mask`
    - `ignore_mask`
    - overlay debug

- Phase D: Tiny fixed paired benchmark generation
  - Add `scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py`.
  - It must accept:
    - `source_manifest_path`
    - `output_root` outside repo
    - `profiles`
    - `severity`
    - `seed`
    - `max_samples_per_class`
  - Selection must be deterministic.
  - It must support a tiny first run with approximately 5 to 20 records per class.
  - Save clean and SNSAug views with the same `base_id`.
  - Required outputs:
    - `images/`
    - `tamper_masks/`
    - `ignore_masks/`
    - `debug_overlays/`
    - `meta.jsonl`
    - `pair_index.json`
    - `source_manifest_audit_summary.json`
    - `artifact_manifest.json`
  - Each `meta.jsonl` record must include:
    - `base_id`
    - `source_dataset`
    - `split`
    - `source_path`
    - optional `source_mask_path`
    - `content_label`
    - optional `family_label`
    - `view` as `clean` or `sns_aug`
    - `profile`
    - `severity`
    - `seed`
    - `image_path`
    - `tamper_mask_path`
    - `ignore_mask_path`
    - `debug_overlay_path`
    - `aug_meta`
    - `overlay_boxes`
    - `label_preserved=true`

- Configs and validator
  - Add:
    - `configs/snsaug_v2/snsaug_v2_source_audit.example.json`
    - `configs/snsaug_v2/snsaug_v2_overlay_preview.example.json`
    - `configs/snsaug_v2/snsaug_v2_tiny_fixed_pairs.example.json`
    - `scripts/agent/validate_snsaug_v2_generation_config.py`
  - Validator must accept:
    - `approved_snsaug_v2_source_audit`
    - `approved_snsaug_v2_overlay_preview`
    - `approved_snsaug_v2_tiny_fixed_pairs`
  - Validator execution modes must accept:
    - `approved_local_snsaug_v2_source_audit`
    - `approved_local_snsaug_v2_overlay_preview`
    - `approved_local_snsaug_v2_tiny_fixed_pairs`
  - Require:
    - `no_training=true`
    - `no_finetune=true`
    - `no_network=true`
    - `no_download=true`
    - `output_root` outside repository when present
    - approved output roots
  - Reject:
    - training
    - fine-tuning
    - network/download
    - repo-local output
    - protected paths

- Tests
  - Add `tests/test_snsaug_v2_generation.py`.
  - Cover:
    - source manifest audit valid/invalid records
    - same seed reproducibility
    - different seed can change output
    - output image, `tamper_mask`, and `ignore_mask` sizes match
    - `clean` profile creates zero `ignore_mask`
    - label preservation
    - UI/text/sticker/badge/news overlay regions appear in `ignore_mask`
    - overlay regions are not added to `tamper_mask`
    - geometric transforms apply identically to image and `tamper_mask`
    - JPEG/recompression does not alter mask
    - mask resizing uses nearest-neighbor
    - tiny fixed pair generator writes `meta.jsonl` and `pair_index.json`
    - clean and SNSAug views share the same `base_id`
    - output paths are outside repository
    - validator guardrails

- Docs
  - Add `docs/snsaug_v2_generation.md`.
  - Include:
    - this task creates SNSAug data from clean source manifests
    - it does not require a pre-existing SNSAug dataset
    - it is generation / preview / pair-building only
    - it does not train or fine-tune
    - supported profiles
    - mask policy
    - metadata schema
    - source manifest audit usage
    - preview usage
    - tiny fixed pair generation usage
    - limitations
    - marker `SNSAUG_V2_GENERATION_AND_TINY_PAIRS_OK`

## Validation Commands
```bash
python3 scripts/agent/validate_snsaug_v2_generation_config.py configs/snsaug_v2/snsaug_v2_source_audit.example.json
```

```bash
python3 scripts/agent/validate_snsaug_v2_generation_config.py configs/snsaug_v2/snsaug_v2_overlay_preview.example.json
```

```bash
python3 scripts/agent/validate_snsaug_v2_generation_config.py configs/snsaug_v2/snsaug_v2_tiny_fixed_pairs.example.json
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_generation.py
```

```bash
python3 scripts/snsaug_v2/preview_snsaug_v2_overlays.py --help
```

```bash
python3 scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py --help
```

```bash
grep -q SNSAUG_V2_GENERATION_AND_TINY_PAIRS_OK docs/snsaug_v2_generation.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0055-snsaug-v2-source-manifest-audit-overlay-generator-and-tiny-fixed-pairs.md
```

## Acceptance Criteria
- A source manifest can be audited safely from clean source data without assuming any prebuilt SNSAug dataset.
- The audit writes valid/invalid record outputs and class summaries without crashing when `fail_fast=false`.
- SNSAug v2 overlay generation remains deterministic with a fixed seed and preserves labels.
- `ignore_mask` contains benign overlay/UI/sticker/banner regions and these regions are not added to `tamper_mask`.
- Geometric transforms keep image and `tamper_mask` aligned, while image-only degradations do not mutate mask content.
- The preview script can generate per-profile previews and a combined preview grid using a real input image or a demo image.
- The tiny fixed pair generator can build deterministic clean/SNSAug paired outputs from a source manifest and write all required metadata and artifact files outside the repository.
- Config validation, tests, docs marker, and agent change checks all pass.
- Backward compatibility with existing SNSAug v2 training-wrapper and fine-tuning code is preserved.

## Stop Condition
- Stop after implementing only the allowed files, running the listed validation commands, and reviewing the result as PASS or NEEDS_FIX.
- If implementation would require training, fine-tuning, network access, downloading assets, repo-local output writing, or modifying unrelated model code, stop and report the blocker.
