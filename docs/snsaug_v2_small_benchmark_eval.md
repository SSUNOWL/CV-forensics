# SNSAug V2 Small Benchmark Eval

`SNSAUG_V2_SMALL_GEOMETRY_POSTPROCESS_BENCHMARK_OK`

`SNSAUG_V2_SMALL_BENCHMARK_CLASS_BALANCED_OK`

This workflow is evaluation and generation only. It does not train, fine-tune, run `0054`, modify checkpoints, download assets, or use network resources.

## Scope

`0058` expands the fixed SNSAug validation benchmark beyond overlay layouts so the frozen pre-SNS bundle can be audited against:

- clean
- geometry/canvas perturbations
- compression/postprocess perturbations
- layout/overlay perturbations
- combined realistic SNS perturbations

## Required Profiles

- `clean`
- `resize_crop_pad`
- `zoom_crop`
- `recompression_light`
- `resize_jpeg`
- `screenshot_recapture_light`
- `canvas_9x16_only`
- `platform_ui_same_size`
- `news_meme_overlay`
- `tiktok_like`
- `instagram_story_like`
- `youtube_shorts_like`
- `combined_sns_realistic`

## Mask Policy

- `tamper_mask` tracks only malicious manipulation regions
- UI bars, frames, text, stickers, banners, and badges must go to `ignore_mask`
- geometric transforms apply to image and `tamper_mask` together
- recompression and color/blur postprocess apply to image only
- mask resizing must use nearest-neighbor
- tiny fixed-pair benchmark generation requires an explicit `max_samples_per_class`
- base sample selection is balanced across `real`, `synthetic`, and `tampered`
- tampered base samples require a readable `tamper_mask_path` by default
- missing tampered samples or zero tampered masks are hard failures unless an explicit diagnostic override is used

## Sanity Files

Pair generation writes these dataset-level sanity summaries:

- `class_balance_summary.json`
- `mask_availability_summary.json`

Before evaluation, run:

```bash
python3 scripts/snsaug_v2/check_snsaug_v2_pair_dataset.py --pair-root /path/to/pair_root
```

The checker verifies that `meta.jsonl` and `pair_index.json` exist, there is exactly one clean row per `base_id`, every base has every required profile, all three labels are present, tampered rows have readable masks, image and ignore-mask paths exist, `label_preserved` is true, and row count equals `base_count x profile_count`.

## Reports

The evaluator writes benchmark-oriented reports outside the repository:

- `small_benchmark_summary.md`
- `small_benchmark_metrics_table.csv`
- `small_benchmark_drop_table.csv`
- `small_benchmark_interpretation.json`
- `artifact_manifest.json`

Per-profile metrics include denominator fields:

- `class_count_real`
- `class_count_synthetic`
- `class_count_tampered`
- `tampered_mask_eval_count`
- `localization_activation_denominator`

When a denominator is zero, the corresponding metric is reported as `NA`/`null` and a warning is emitted; it must not be interpreted as `0.0`.

The interpretation JSON identifies:

- main collapse cause
- profiles safe for training
- profiles too severe for early curriculum
- recommended SNSAug fine-tuning mix
