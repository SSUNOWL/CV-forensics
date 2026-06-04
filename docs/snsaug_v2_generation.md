# SNSAug V2 Generation

`SNSAUG_V2_GENERATION_AND_TINY_PAIRS_OK`

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

## Mask Policy

- SNSAug is benign degradation and overlay only.
- It never changes the underlying content label.
- `tamper_mask` contains only malicious manipulation regions.
- UI, text, stickers, watermarks, AI badges, news banners, emoji, red circles, arrows, highlight boxes, and screenshot bars must not be added to `tamper_mask`.
- Those benign overlay areas are written to `ignore_mask`.
- Geometric transforms are applied identically to image and `tamper_mask`.
- JPEG, blur, color shift, sharpen, and recompression affect image only.
- Any mask resize uses nearest-neighbor interpolation.

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
