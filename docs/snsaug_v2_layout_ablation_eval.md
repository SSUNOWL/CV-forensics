# SNSAug V2 Layout Cause Ablation Eval

`SNSAUG_V2_LAYOUT_CAUSE_ABLATION_EVAL_OK`

This workflow is evaluation and generation only. It does not train, fine-tune, run `0054`, modify checkpoints, download assets, or use network resources.

## Purpose

`0057b` isolates which SNS layout factors break the frozen pre-SNS bundle before any training decision:

- portrait 9:16 canvas placement
- content rescaling inside portrait canvas
- simplified platform UI overlays
- right-side action bars
- story text or sticker blocks
- combined realistic layouts

## Ablation Profiles

- `canvas_9x16_only`
- `canvas_9x16_full_content`
- `platform_ui_same_size`
- `tiktok_like_no_actionbar`
- `instagram_story_no_text_sticker`
- `youtube_shorts_no_actionbar`
- `news_meme_overlay`
- `tiktok_like`
- `instagram_story_like`
- `youtube_shorts_like`
- `combined_sns_realistic`

Canvas-only ablations are layout-only by design:

- no platform UI
- no text or stickers
- no recompression
- no blur or color shift

The UI ablations remove only the named structural cause while keeping the rest of the layout minimal and deterministic.

## Evaluation Scope

The fixed-pair evaluator still compares clean vs SNSAug rows by `base_id` and reports:

- 3-way accuracy
- macro-F1
- real FPR
- synthetic recall
- tampered recall
- localization activation recall
- raw IoU
- valid IoU excluding `ignore_mask`
- valid IoU drop
- `p_tampered` drop
- fragile class flip
- fragile activation flip

## Output Artifacts

Expected outputs remain outside the repository and include:

- tiny fixed-pair ablation dataset with `meta.jsonl` and `pair_index.json`
- evaluation records
- per-profile metrics
- robustness drop metrics
- worst samples
- fragile candidates
- visual gallery manifest

## Validation Checklist

Before using a larger fixed-pair batch:

- confirm `canvas_9x16_only` has no UI and no degradation
- confirm `platform_ui_same_size` preserves original size/aspect as much as practical
- confirm ablation profiles are deterministic for the same seed
- confirm all generated outputs remain outside the repository
