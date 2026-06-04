# SNSAug V2

`SNSAUG_V2_REALISTIC_OVERLAY_MODULE_OK`

SNSAug V2 is a deterministic, research-safe augmentation module for social-media-style robustness work. It draws generic UI overlays, stickers, badges, banners, and screenshot framing with PIL only. It does not train models, download assets, use network resources, or copy exact proprietary platform logos.

## Design Overview

The module separates three concepts:

- image content: receives geometric transforms and benign visual degradation
- `tamper_mask`: keeps only malicious manipulation regions
- `ignore_mask`: captures benign UI, text, sticker, banner, watermark, badge, and frame regions

This ensures social-media overlays do not become false tamper supervision.

## Profiles

Supported profiles:

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
- It never changes the content label.
- `real` remains `real`.
- `synthetic` remains `synthetic`.
- `tampered` remains `tampered`.
- UI, text, stickers, watermarks, AI badges, news banners, emoji, red circles, arrows, highlight boxes, and screenshot bars must not be added to `tamper_mask`.
- Those benign overlay regions are written to `ignore_mask`.
- Geometric transforms are applied identically to image and `tamper_mask`.
- JPEG, blur, pixelation, and color shift are applied only to image data.
- Any mask resize uses nearest-neighbor interpolation.

## Usage

Preview a single augmented sample:

```bash
python3 scripts/snsaug_v2/preview_snsaug_v2.py --image /abs/input.png --mask /abs/mask.png --profile combined_sns_realistic --severity medium --seed 51 --output-root /abs/output_dir
```

Build paired clean and augmented views:

```bash
python3 scripts/snsaug_v2/build_snsaug_v2_pairs.py --input-manifest /abs/manifest.json --output-root /abs/pairs_out --profiles combined_sns_realistic news_meme_overlay --severity medium --seed 51 --split train --max-samples 100
```

## Pair Generation

The pair generator writes:

- `images/`
- `tamper_masks/`
- `ignore_masks/`
- `meta.jsonl`
- `pair_index.json`
- `artifact_manifest.json`

Each record includes the source metadata, profile, severity, deterministic seed, paired `base_id`, saved artifact paths, overlay box metadata, geometric transform metadata, and `label_preserved: true`.

## Safety Notes

- This module is augmentation tooling only.
- No SNS augmentation training occurs here.
- Validation or test failure samples must not be used directly as training data.
- If training manifests are needed later, train split mining must be handled separately from this module.
