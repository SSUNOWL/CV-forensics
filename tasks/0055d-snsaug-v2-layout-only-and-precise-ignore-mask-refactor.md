## Task Title
0055d SNSAug V2 Layout-Only and Precise Ignore Mask Refactor

## Role
Codex-only task writer and later implementation worker/reviewer for a narrow SNSAug v2 refactor. This task fixes layout-only quality behavior, alpha-based precise `ignore_mask`, and deterministic placement diversity without training or fine-tuning.

## Files Codex May Read
- AGENTS.md
- CLAUDE.md
- docs/project_brief.md
- docs/project_contract.md
- configs/project_contract.json
- docs/agent_workflow.md
- tasks/0051-snsaug-v2-realistic-overlay-module-and-pair-generator.md
- tasks/0053-snsaug-v2-training-manifest-and-dataset-wrapper.md
- tasks/0054-snsaug-v2-aware-finetuning.md
- tasks/0055-snsaug-v2-source-manifest-audit-overlay-generator-and-tiny-fixed-pairs.md
- docs/snsaug_v2.md
- docs/snsaug_v2_generation.md
- configs/snsaug_v2/snsaug_v2.example.json
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
- tests/test_snsaug_v2.py
- tests/test_snsaug_v2_generation.py
- relevant validators, scripts, and current git status

## Files Codex May Modify
- tasks/0055d-snsaug-v2-layout-only-and-precise-ignore-mask-refactor.md
- scripts/snsaug_v2/preview_snsaug_v2_overlays.py
- scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py
- src/cv_forensics/snsaug_v2/__init__.py
- src/cv_forensics/snsaug_v2/configs.py
- src/cv_forensics/snsaug_v2/transforms.py
- src/cv_forensics/snsaug_v2/ui_renderers.py
- src/cv_forensics/snsaug_v2/sticker_packs.py
- src/cv_forensics/snsaug_v2/platform_templates.py
- src/cv_forensics/snsaug_v2/sns_augmentor.py
- src/cv_forensics/snsaug_v2/pair_generator.py
- tests/test_snsaug_v2_generation.py
- tests/test_snsaug_v2_layout_only_and_precise_ignore.py
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
- Do not assume a prebuilt SNSAug dataset exists.
- Do not modify validators or configs outside the allowed list.

## Important Project Facts
- SNSAug v2 is benign degradation and overlay only. It must never change the underlying content label:
  - Real remains real.
  - Synthetic remains synthetic.
  - Tampered remains tampered.
- `tamper_mask` contains only malicious manipulation regions.
- UI, text, stickers, watermark, AI badges, news banners, emoji, red circle, arrows, platform frame, and screenshot bars are not tamper regions and must be written to `ignore_mask`.
- The current preview revealed that some platform profiles degrade image quality too early. This task separates layout-only rendering from quality-degradation postprocess steps.
- `ignore_mask` must follow actual rendered overlay pixels derived from overlay alpha, not coarse bounding boxes, especially for hollow shapes like outline rectangles and circles.
- Existing preview and tiny-pair generation scripts must remain compatible.
- All generated artifacts must be written outside the repository.

## Implementation Requirements
- Implement a narrow SNSAug v2 refactor within the allowed files only.

- Part A: Separate layout-only profiles from degradation profiles
  - These profiles must be layout-only by default:
    - `tiktok_like`
    - `instagram_story_like`
    - `youtube_shorts_like`
    - `news_meme_overlay`
  - Layout-only means:
    - no JPEG recompression
    - no WebP recompression
    - no blur
    - no pixelation
    - no sharpen
    - no color shift
    - no screenshot recapture
    - no intentional low-quality downscale/upscale
    - only high-quality resize when geometrically required to fit a new platform canvas
    - masks use nearest-neighbor only
  - Add or expose separate postprocess profiles:
    - `recompression_light`
    - `resize_jpeg`
    - `screenshot_recapture`
    - `blur_color_shift`
  - `combined_sns_realistic` may combine:
    - one layout-only profile
    - optional text/sticker/badge/news overlay
    - optional postprocess degradation
  - Metadata must record all applied transforms.
  - Add or expose config flags:
    - `apply_degradation: bool = False`
    - `apply_recompression: bool = False`
    - `apply_blur: bool = False`
    - `apply_color_shift: bool = False`
    - `apply_screenshot_recapture: bool = False`

- Part B: Pixel-accurate `ignore_mask` from overlay alpha
  - Replace bbox-only `ignore_mask` logic with alpha-based `ignore_mask`.
  - Every overlay element must be rendered to a transparent RGBA overlay layer:
    - text block
    - UI icon
    - badge
    - news banner
    - red rectangle
    - red circle
    - arrow
    - emoji
    - speech bubble
    - sticker
    - highlight box
  - Final `ignore_mask` contribution must be derived from the rendered overlay alpha:
    - `alpha > threshold` means ignore
    - `alpha == 0` means do not ignore
  - Required behavior:
    - red outline rectangle marks only outline pixels, not transparent interior
    - red outline circle marks only circular outline pixels
    - arrow marks only arrow body/head pixels
    - filled or semi-transparent highlight box marks filled area
    - text with background box marks the background box and text
    - text without background marks text glyph pixels and optional shadow pixels
  - Keep `overlay_boxes` as coarse debug metadata only. They must not define exact ignore-mask coverage for hollow shapes.

- Part C: Placement diversity
  - Keep structural platform UI mostly fixed:
    - progress bars
    - account row
    - right action bar
    - bottom navigation
    - message input bar
    - channel/title area
  - Make these elements variable:
    - text_block
    - AI badge
    - news banner
    - red rectangle
    - red circle
    - arrow
    - emoji
    - speech bubble
    - annotation sticker
    - poll/question/location sticker
  - Add candidate placement regions per profile:
    - `text_regions`
    - `sticker_regions`
    - `badge_regions`
    - `annotation_regions`
    - `news_banner_regions`
  - For each variable overlay:
    - choose candidate region with seed-controlled RNG
    - apply position jitter
    - apply size jitter
    - optionally apply small rotation for sticker-like elements
    - clamp to canvas bounds
    - retry if `ignore_mask_area_pct` exceeds configured max
  - Suggested default policy:
    - 70% peripheral regions
    - 20% mid regions
    - 10% may overlap salient content area
  - This must be deterministic for the same seed.
  - Metadata must record:
    - `element_type`
    - `chosen_candidate_region`
    - `final_bbox`
    - `size_scale`
    - `jitter`
    - `rotation_deg` if used
    - `alpha_mask_area_px`
    - `ignore_mask_area_pct`

- Part D: Tamper mask policy
  - Overlay pixels must never be added to `tamper_mask`.
  - `tamper_mask` should only be geometrically transformed if the image is placed/resized/cropped into a new canvas.
  - JPEG/recompression/blur/color shift must never be applied to `tamper_mask`.
  - Mask resizing must use nearest-neighbor interpolation.

- Part E: Tests
  - Add `tests/test_snsaug_v2_layout_only_and_precise_ignore.py`.
  - Required tests:
    - `instagram_story_like` metadata shows no recompression/blur/color/postprocess by default
    - `youtube_shorts_like` metadata shows no recompression/blur/color/postprocess by default
    - layout-only profiles preserve image quality except necessary geometric canvas placement
    - red outline rectangle `ignore_mask` marks only outline pixels, not full interior
    - red outline circle `ignore_mask` marks only outline pixels
    - arrow `ignore_mask` marks only rendered arrow pixels
    - filled translucent box marks filled area
    - same seed produces identical overlay placements
    - different seed can produce different text/sticker placement
    - `tamper_mask` is not contaminated by overlay alpha
    - `overlay_boxes` are metadata only and do not define exact ignore mask for hollow shapes
    - `combined_sns_realistic` can still include degradation when configured
  - Update existing SNSAug tests only where needed to preserve compatibility.

- Part F: Documentation
  - Update `docs/snsaug_v2_generation.md`.
  - Add sections covering:
    - layout-only profiles vs postprocess profiles
    - alpha-based `ignore_mask`
    - deterministic placement diversity
    - validation checklist before tiny-pair generation
  - Include marker:
    - `SNSAUG_V2_LAYOUT_ONLY_PRECISE_IGNORE_OK`

- Part G: Compatibility
  - Keep:
    - `scripts/snsaug_v2/preview_snsaug_v2_overlays.py`
    - `scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py`
    compatible with existing arguments and outputs.

- Part H: Real preview rerun
  - After implementation, rerun preview on the same real sample with:
    - `clean`
    - `tiktok_like`
    - `instagram_story_like`
    - `youtube_shorts_like`
    - `news_meme_overlay`
    - `combined_sns_realistic`
  - Expected preview improvements:
    - `instagram_story_like` should not look low-quality
    - `youtube_shorts_like` should not look low-quality
    - red rectangle `ignore_mask` should only cover outline pixels when interior is transparent
    - text/sticker/badge/annotation should move when seed changes
    - same seed should reproduce exact same result
  - If the same real sample path is not available from local task context, stop short of rerun and report that blocker explicitly rather than guessing a path.

## Validation Commands
```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_generation.py
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_layout_only_and_precise_ignore.py
```

```bash
python3 scripts/snsaug_v2/preview_snsaug_v2_overlays.py --help
```

```bash
python3 scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py --help
```

```bash
grep -q SNSAUG_V2_LAYOUT_ONLY_PRECISE_IGNORE_OK docs/snsaug_v2_generation.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0055d-snsaug-v2-layout-only-and-precise-ignore-mask-refactor.md
```

## Acceptance Criteria
- Layout-only profiles no longer apply quality degradation by default.
- Degradation behavior is exposed separately and can still be combined by `combined_sns_realistic`.
- `ignore_mask` is derived from rendered overlay alpha rather than coarse bounding boxes.
- Hollow shapes mark only rendered outline pixels in `ignore_mask`.
- Variable overlays show deterministic seed-based placement diversity.
- `tamper_mask` remains uncontaminated by overlays and only changes under geometric placement transforms.
- Existing preview and tiny-pair generation interfaces remain compatible.
- Tests, docs marker, and agent change checks all pass.

## Stop Condition
- Stop after implementing only the allowed files, running the listed validation commands, and reviewing the result as PASS or NEEDS_FIX.
- If implementation would require training, fine-tuning, network access, downloading assets, repo-local output writing, or modifying unrelated model code, stop and report the blocker.
