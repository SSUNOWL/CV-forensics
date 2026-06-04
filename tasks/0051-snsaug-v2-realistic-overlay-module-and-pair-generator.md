# Task 0051: SNSAug V2 Realistic Overlay Module and Pair Generator

## Task Title

Implement a deterministic, research-safe, platform-template-based SNS augmentation module that produces realistic social-media-style benign overlays, transformed tamper masks, ignore masks, and paired clean/augmented outputs without changing content labels.

## Role

Codex-only implementation worker, reviewer, and limited repair manager operating under the repository contract and Codex-only workflow.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0051-snsaug-v2-realistic-overlay-module-and-pair-generator.md`
- `tasks/0050-pre-sns-best-basic-sns-robustness-evaluation.md`
- `docs/pre_sns_v3_sns_robustness_eval.md`
- `src/cv_forensics/pre_sns_v3_sns_robustness_eval.py`
- `tests/test_pre_sns_v3_sns_robustness_eval.py`
- `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
- `src/cv_forensics/pre_sns_v3_final_visual_audit.py`
- `src/cv_forensics/__init__.py`
- Any existing safe image, manifest, or visualization helpers already in `src/cv_forensics/`, `scripts/`, `tests/`, and `docs/` that are directly relevant to deterministic image transforms, mask handling, manifest writing, or evaluation-safe artifact generation

## Files Codex May Modify

- `tasks/0051-snsaug-v2-realistic-overlay-module-and-pair-generator.md`
- `src/cv_forensics/snsaug_v2/__init__.py`
- `src/cv_forensics/snsaug_v2/configs.py`
- `src/cv_forensics/snsaug_v2/transforms.py`
- `src/cv_forensics/snsaug_v2/ui_renderers.py`
- `src/cv_forensics/snsaug_v2/sticker_packs.py`
- `src/cv_forensics/snsaug_v2/platform_templates.py`
- `src/cv_forensics/snsaug_v2/sns_augmentor.py`
- `src/cv_forensics/snsaug_v2/pair_generator.py`
- `scripts/snsaug_v2/preview_snsaug_v2.py`
- `scripts/snsaug_v2/build_snsaug_v2_pairs.py`
- `configs/snsaug_v2/snsaug_v2.example.json`
- `tests/test_snsaug_v2.py`
- `docs/snsaug_v2.md`
- `src/cv_forensics/__init__.py`

## Forbidden Actions

- Do not modify any file outside the allowed modify list.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not train or fine-tune models.
- Do not use network resources.
- Do not download assets, fonts, datasets, or templates.
- Do not install packages.
- Do not use exact platform logos or copyrighted sticker images.
- Do not embed or share font files.
- Do not write outputs inside the repository.
- Do not create large files.
- Do not modify unrelated model training code.
- Do not commit changes.

## Important Project Facts

- This task implements augmentation tooling only; it does not train anything.
- SNSAug is benign degradation and overlay only.
- Augmentation never changes the content label:
  - real remains real
  - synthetic remains synthetic
  - tampered remains tampered
- `tamper_mask` contains only malicious manipulation regions.
- UI, text, stickers, watermarks, AI badges, news banners, emoji, red circle, arrows, platform frames, and screenshot bars are benign overlay regions and must never be added to `tamper_mask`.
- Benign overlay regions must be written to `ignore_mask`.
- Geometric transforms must be applied identically to image and `tamper_mask`.
- Compression, blur, pixelation, and color shift apply only to image data.
- Mask resizing must use nearest-neighbor interpolation.
- The implementation must be deterministic for a given seed.
- The design must be research-safe and platform-like, not a copy of proprietary UI assets.

## Implementation Requirements

### 1. Add SNSAug V2 Package

Add:

- `src/cv_forensics/snsaug_v2/__init__.py`
- `src/cv_forensics/snsaug_v2/configs.py`
- `src/cv_forensics/snsaug_v2/transforms.py`
- `src/cv_forensics/snsaug_v2/ui_renderers.py`
- `src/cv_forensics/snsaug_v2/sticker_packs.py`
- `src/cv_forensics/snsaug_v2/platform_templates.py`
- `src/cv_forensics/snsaug_v2/sns_augmentor.py`
- `src/cv_forensics/snsaug_v2/pair_generator.py`

### 2. Core Dataclasses and API

Implement:

```python
@dataclass
class SNSAugV2Config:
    profile: str
    severity: str = "medium"
    output_size: Optional[int] = None
    seed: Optional[int] = None
    platform_template: Optional[str] = None
    p_recompression: float = 0.7
    p_platform_ui: float = 0.8
    p_ai_badge: float = 0.4
    p_news_banner: float = 0.3
    p_annotation_sticker: float = 0.4
    p_emoji_sticker: float = 0.3
    p_color_shift: float = 0.2
    p_blur_pixelation: float = 0.2
```

```python
@dataclass
class SNSAugV2Result:
    image: PIL.Image.Image
    tamper_mask: Optional[PIL.Image.Image]
    ignore_mask: PIL.Image.Image
    meta: Dict[str, Any]
```

```python
class SNSAugV2Augmentor:
    def __init__(self, config: SNSAugV2Config): ...
    def __call__(self, image, tamper_mask=None, label=None, base_id=None, seed=None) -> SNSAugV2Result: ...
```

Every augmentation call must return:

- augmented image
- transformed `tamper_mask`
- `ignore_mask`
- augmentation metadata

### 3. Required Profiles

Implement support for:

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

### 4. Required Platform Templates

Implement drawn, generic platform-like templates with `ignore_mask` coverage for all rendered UI/text/sticker/icon regions.

#### `tiktok_like`

- portrait 9:16 or similar canvas
- centered/cropped/padded source content
- top small tab row
- right-side action bar with simple drawn icons
- bottom username, caption, and music text area
- optional bottom navigation bar
- optional AI badge

#### `instagram_story_like`

- portrait 9:16 canvas
- top progress bars
- account row and time
- optional translucent text sticker
- optional poll, question, or location sticker
- bottom message input bar

#### `youtube_shorts_like`

- portrait 9:16 canvas
- right-side action bar
- bottom channel, title, and subscribe area
- optional bottom navigation
- optional AI label chip

#### `news_meme_overlay`

- headline banner
- lower-third caption
- optional red circle, arrow, or highlight box
- optional terms such as `BREAKING`, `속보`, `단독`, `확인 필요`

### 5. Sticker Packs

Implement all sticker packs using PIL drawing only:

- AI badges
  - `Made with AI`
  - `AI generated`
  - `AI-edited`
  - `Synthetic image`
  - `Generated with AI`
- platform-like watermarks
  - generic short-video watermark
  - `@username` text
- news banners
  - `속보`
  - `BREAKING`
  - `NEWS`
  - `LIVE`
  - `단독`
- annotation stickers
  - red circle
  - red arrow
  - red rectangle
  - highlight box
  - speech bubble
- emoji or reaction stickers
  - simple drawn emoji-like face
  - heart
  - check or cross

Do not depend on external fonts. Use PIL default font when project fonts are unavailable. If a local font path is later provided, support it as optional configuration only without embedding the font.

### 6. Mask and Transform Policy

- Never add benign overlay regions to `tamper_mask`.
- Always mark benign overlay pixels into `ignore_mask`.
- Apply geometric transforms identically to image and `tamper_mask`.
- Apply recompression, blur, pixelation, and color shift only to image.
- Use nearest-neighbor interpolation for any mask resize.
- Ensure `clean` profile returns zero `ignore_mask` and preserves image and mask shape.

### 7. Pair Generator

Implement `SNSAugV2PairGenerator` with support for:

- input manifest
- output root
- profiles
- severity
- seed
- split name
- max samples
- optional samples per class
- deterministic behavior
- saving clean and augmented views with the same `base_id`

Write:

- images
- tamper masks
- ignore masks
- `meta.jsonl`
- `pair_index.json`
- `artifact_manifest.json`

Metadata per sample must include:

- `base_id`
- `source_dataset`
- `split`
- `source_path`
- `content_label`
- `family_label` if available
- `view`
- `profile`
- `severity`
- `seed`
- `image_path`
- `tamper_mask_path`
- `ignore_mask_path`
- `aug_meta`
- `overlay_boxes`
- `geometric_transform_meta`
- `label_preserved` true

### 8. Preview Script

Add:

- `scripts/snsaug_v2/preview_snsaug_v2.py`

It must:

- accept input image path
- accept optional mask path
- accept profile
- accept severity
- accept seed
- accept output root
- save augmented image, tamper mask, ignore mask, and metadata
- save a preview grid showing original, augmented, tamper mask, and ignore mask

### 9. Pair-Build Script

Add:

- `scripts/snsaug_v2/build_snsaug_v2_pairs.py`

It must expose a CLI for deterministic pair generation from an input manifest into an approved external output root.

### 10. Config and Example

Add:

- `configs/snsaug_v2/snsaug_v2.example.json`

The example config should demonstrate:

- deterministic seed use
- one or more realistic profiles
- external output root
- no training
- no network
- no download

### 11. Tests

Add:

- `tests/test_snsaug_v2.py`

Tests must cover:

- same seed reproducibility
- different seed can change output
- output image, `tamper_mask`, and `ignore_mask` sizes match
- label preservation
- text, sticker, and UI regions appear in `ignore_mask`
- text, sticker, and UI regions are not added to `tamper_mask`
- geometric transforms apply identically to image and `tamper_mask`
- JPEG or recompression does not alter mask
- mask resizing uses nearest-neighbor
- `clean` profile returns zero `ignore_mask` and preserves image and mask shape
- pair generator writes `meta.jsonl` and preserves same `base_id` mapping across clean and augmented views

### 12. Docs

Add:

- `docs/snsaug_v2.md`

Include:

- design overview
- supported profiles
- mask policy
- usage examples
- pair generation example
- marker `SNSAUG_V2_REALISTIC_OVERLAY_MODULE_OK`

## Validation Commands

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2.py
```

```bash
python3 scripts/snsaug_v2/preview_snsaug_v2.py --help
```

```bash
python3 scripts/snsaug_v2/build_snsaug_v2_pairs.py --help
```

```bash
grep -q SNSAUG_V2_REALISTIC_OVERLAY_MODULE_OK docs/snsaug_v2.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0051-snsaug-v2-realistic-overlay-module-and-pair-generator.md
```

## Acceptance Criteria

- SNSAug V2 package exists and exposes deterministic augmentation APIs.
- Benign overlay regions are captured in `ignore_mask` and excluded from `tamper_mask`.
- Geometric transforms preserve image and mask alignment.
- Image-only degradations do not alter `tamper_mask`.
- Generic platform-like templates and sticker packs are drawn without external proprietary assets.
- Preview and pair-build scripts expose working CLI help.
- Pair generator writes clean and augmented paired views plus required manifests.
- Tests pass using CPU-only execution.
- Docs include the required marker and explain the mask policy and research-safe constraints.

## Stop Condition

Stop after implementing only the allowed files, running the validation commands, running `python3 scripts/agent/check_agent_changes.py tasks/0051-snsaug-v2-realistic-overlay-module-and-pair-generator.md`, and reviewing the result in Korean as `PASS` or `NEEDS_FIX`. Do not implement training, networked asset retrieval, or any repo-local output writing.
