# SNSAug V2 Generation

`SNSAUG_V2_GENERATION_AND_TINY_PAIRS_OK`

`SNSAUG_V2_LAYOUT_ONLY_PRECISE_IGNORE_OK`

`SNSAUG_V2_OVERLAY_COLLISION_GUARD_OK`

This task creates SNSAug data from clean source manifests. It does not require a pre-existing SNSAug dataset. It is generation, preview, and pair-building only. It does not train or fine-tune models.

## Scope

- Source manifest audit from clean train/validation/manual manifests
- SNSAug V2 overlay generation from existing source images
- Per-profile preview generation
- Tiny fixed clean/SNSAug paired benchmark generation

## Supported Profiles

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

Postprocess-oriented profiles:

- `recompression_light`
- `resize_jpeg`
- `screenshot_recapture`
- `blur_color_shift`

## Mask Policy

- SNSAug is benign degradation and overlay only.
- It never changes the underlying content label.
- `tamper_mask` contains only malicious manipulation regions.
- UI, text, stickers, watermarks, AI badges, news banners, emoji, red circles, arrows, highlight boxes, and screenshot bars must not be added to `tamper_mask`.
- Those benign overlay areas are written to `ignore_mask`.
- Geometric transforms are applied identically to image and `tamper_mask`.
- JPEG, blur, color shift, sharpen, and recompression affect image only.
- Any mask resize uses nearest-neighbor interpolation.

## Layout vs Postprocess

The platform-layout profiles are layout-only by default:

- `tiktok_like`
- `instagram_story_like`
- `youtube_shorts_like`
- `news_meme_overlay`

By default they do not apply recompression, blur, pixelation, sharpen, color shift, screenshot recapture, or intentional low-quality resampling. Only the geometric canvas placement needed to fit the platform layout is applied.

If degradation is needed, use explicit postprocess profiles or enable degradation flags through `combined_sns_realistic`. The metadata records `transforms_applied` and `postprocess_applied` separately.

## Alpha-Based Ignore Mask

`ignore_mask` is derived from actual rendered overlay alpha, not only from coarse bounding boxes.

- Hollow outline shapes such as red rectangles and circles mark only their visible outline pixels.
- Arrow overlays mark only the rendered shaft and arrow head.
- Filled or translucent highlight boxes mark the filled region because the pixels are changed.
- Text, badges, and chips mark their true rendered background and glyph area.

`overlay_boxes` remain coarse debug metadata only. They are useful for previews and inspection, but they are not the exact ignore-mask definition for hollow shapes.

## Deterministic Placement Diversity

Structural platform UI stays mostly fixed, but variable elements such as text blocks, badges, news banners, stickers, arrows, circles, speech bubbles, and annotation overlays use seed-controlled candidate regions with jitter and size variation.

For the same seed:

- the same placement is reproduced exactly

For a different seed:

- the same profile can move variable overlays within its candidate regions

Metadata records the chosen candidate region, final box, jitter, size scale, and alpha-driven ignore-mask area fields for variable elements.

## Placement Collision Guard

Variable overlays now use a deterministic placement collision guard before they are committed. The guard renders a candidate overlay to a temporary alpha layer, compares that alpha mask against already placed overlays, and retries placement when overlap exceeds the configured thresholds.

Default collision policy includes:

- text vs sticker: no overlap
- badge vs text: no overlap
- variable overlay vs fixed UI: no overlap
- text vs text: very small overlap only
- sticker vs sticker: small overlap only

Metadata records placement retries, skipped optional overlays, overlap ratios, and placement-policy fields so the final augmentation remains auditable.

If all candidate regions are blocked, optional overlays may be skipped instead of forcing a collision.

## Source Manifest Audit

The source manifest audit verifies that records can be parsed and that each record has or can derive:

- `base_id`
- `image_path`
- `content_label`
- optional `tamper_mask_path`
- optional `family_label`
- `split`
- `source_dataset`

It writes:

- `source_manifest_audit_summary.json`
- `source_manifest_valid_records.jsonl`
- `source_manifest_invalid_records.jsonl`
- `source_manifest_class_counts.json`

Example:

```bash
python3 scripts/agent/validate_snsaug_v2_generation_config.py configs/snsaug_v2/snsaug_v2_source_audit.example.json
```

## Preview Usage

The preview script can run on a real source image or a built-in demo image. For each profile it saves the augmented image, transformed mask, ignore mask, metadata JSON, overlay debug image, and a preview grid.

Example:

```bash
python3 scripts/snsaug_v2/preview_snsaug_v2_overlays.py --use_demo_image --profiles clean tiktok_like combined_sns_realistic --severity medium --seed 51 --output-root /home/rlatjswo/.codex/memories/snsaug_v2_preview
```

## Tiny Fixed Pair Generation

The tiny fixed pair generator creates deterministic clean/SNSAug views from a clean source manifest. It is suitable for a first-run benchmark and does not assume any prebuilt SNSAug dataset.

Outputs:

- `images/`
- `tamper_masks/`
- `ignore_masks/`
- `debug_overlays/`
- `meta.jsonl`
- `pair_index.json`
- `source_manifest_audit_summary.json`
- `artifact_manifest.json`

Each `meta.jsonl` record includes the source metadata, `view`, `profile`, `severity`, `seed`, saved artifact paths, `aug_meta`, `overlay_boxes`, and `label_preserved: true`.

## Validation Checklist

Before creating larger tiny/fixed pairs:

- confirm layout-only profiles look visually clean without unintended low-quality degradation
- confirm outline rectangle and outline circle overlays do not fill transparent interiors in `ignore_mask`
- confirm seed reproducibility on the same sample
- confirm different seeds move variable stickers, text, or badges
- confirm `tamper_mask` remains unchanged except for explicit geometric placement transforms

Example:

```bash
python3 scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py --source-manifest-path /abs/source_manifest.jsonl --output-root /abs/tiny_pairs_out --profiles jpeg_resize tiktok_like combined_sns_realistic --severity medium --seed 77 --max-samples-per-class 5
```

## Safety Notes

- This workflow is generation-only.
- No SNS augmentation training occurs here.
- Validation or test failure samples must not be used directly for training.
- Train split mining for later training manifests must be handled separately.
- All generated artifacts must be written outside the repository.

## Limitations

- The overlays are research-safe approximations built from PIL primitives, not exact platform assets.
- If no local font is supplied, PIL default font is used.
- Optional masks remain optional for source audit and pair generation, but tampered localization supervision is only available when a real mask exists.
