# Task: 0058 SNSAug V2 Small Benchmark With Geometry And Postprocess

## Task Title

SNSAug V2 small fixed validation benchmark with geometry, postprocess, layout, and combined perturbation evaluation

## Role

Codex-only implementation worker, reviewer, and limited repair manager for an evaluation/generation-only benchmark task.

## Files Codex May Read

- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0055-snsaug-v2-source-manifest-audit-overlay-generator-and-tiny-fixed-pairs.md`
- `tasks/0055d-snsaug-v2-layout-only-and-precise-ignore-mask-refactor.md`
- `tasks/0055e-snsaug-v2-overlay-placement-collision-guard.md`
- `tasks/0057-snsaug-v2-tiny-fixed-pairs-frozen-model-evaluation.md`
- `tasks/0057b-snsaug-v2-layout-cause-ablation-evaluation.md`
- `docs/snsaug_v2_generation.md`
- `docs/snsaug_v2_fixed_pairs_eval.md`
- `docs/snsaug_v2_layout_ablation_eval.md`
- `configs/snsaug_v2/snsaug_v2_tiny_fixed_pairs.example.json`
- `configs/evaluation/snsaug_v2_fixed_pairs_eval.example.json`
- `scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py`
- `scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py`
- `scripts/agent/validate_snsaug_v2_generation_config.py`
- `scripts/agent/validate_snsaug_v2_fixed_pairs_eval_config.py`
- `src/cv_forensics/snsaug_v2/configs.py`
- `src/cv_forensics/snsaug_v2/transforms.py`
- `src/cv_forensics/snsaug_v2/platform_templates.py`
- `src/cv_forensics/snsaug_v2/sns_augmentor.py`
- `src/cv_forensics/snsaug_v2/pair_generator.py`
- `src/cv_forensics/snsaug_v2/source_manifest_audit.py`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_generation.py`
- `tests/test_snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_layout_cause_ablation_eval.py`
- `.local/pre_sns_current_best_model_bundle.json`
- `/home/rlatjswo/cvf_local_store/pre_sns_manifests_0035_clean/pre_sns_large_manifest_val_clean.local.json`

## Files Codex May Modify

- `tasks/0058-snsaug-v2-small-benchmark-with-geometry-and-postprocess.md`
- `configs/snsaug_v2/snsaug_v2_tiny_fixed_pairs.example.json`
- `configs/evaluation/snsaug_v2_fixed_pairs_eval.example.json`
- `scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py`
- `scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py`
- `src/cv_forensics/snsaug_v2/configs.py`
- `src/cv_forensics/snsaug_v2/transforms.py`
- `src/cv_forensics/snsaug_v2/platform_templates.py`
- `src/cv_forensics/snsaug_v2/sns_augmentor.py`
- `src/cv_forensics/snsaug_v2/pair_generator.py`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_generation.py`
- `tests/test_snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_small_benchmark_eval.py`
- `docs/snsaug_v2_small_benchmark_eval.md`

## Forbidden Actions

- Do not train.
- Do not fine-tune.
- Do not run `0054`.
- Do not use network.
- Do not download assets or datasets.
- Do not install packages.
- Do not write generated outputs inside the repository.
- Do not use validation failures for training.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not modify unrelated model or training code.

## Important Project Facts

- This repository builds a lightweight multi-head image forensics system for `real / synthetic / tampered` classification, tampered localization, family provenance, and evidence reporting.
- `0057` already provides fixed-pair frozen-model evaluation with raw and ignore-mask-excluded valid localization metrics.
- `0057b` showed that portrait canvas / aspect-ratio / content scaling is likely the dominant failure cause, not UI icons alone.
- The project still needs a small benchmark that covers geometry, recompression/postprocess, overlays, and combined realistic SNS perturbations before any SNS-aware fine-tuning decision.
- The frozen evaluation bundle is `.local/pre_sns_current_best_model_bundle.json`.
- The validation manifest for this benchmark is `/home/rlatjswo/cvf_local_store/pre_sns_manifests_0035_clean/pre_sns_large_manifest_val_clean.local.json`.
- This task is evaluation/generation only and must keep all generated artifacts outside the repository.

## Implementation Requirements

1. Build a small fixed paired validation benchmark and frozen-model evaluation path across:
   - clean
   - geometry/canvas perturbations
   - compression/postprocess perturbations
   - overlay/layout perturbations
   - combined SNS realistic perturbation
2. Always include these profiles:
   - `clean`
   - `canvas_9x16_only`
   - `platform_ui_same_size`
   - `news_meme_overlay`
   - `tiktok_like`
   - `instagram_story_like`
   - `youtube_shorts_like`
   - `combined_sns_realistic`
3. Add these profiles if missing:
   - `resize_crop_pad`
   - `zoom_crop`
   - `recompression_light`
   - `resize_jpeg`
   - `screenshot_recapture_light`
4. Profile requirements:
   - `resize_crop_pad`
     - choose from `1:1`, `4:5`, `9:16`, `16:9`
     - crop or pad
     - apply the same geometry to image and `tamper_mask`
     - use nearest-neighbor for masks
     - no image-quality degradation unless explicitly configured
   - `zoom_crop`
     - simulate SNS zoom/crop with scale `1.05~1.30`
     - crop back to output frame
     - apply the same transform to `tamper_mask`
     - keep `ignore_mask` zero unless overlay is added
   - `recompression_light`
     - JPEG quality `70~90`
     - optional one downscale/upscale step
     - image only
     - do not alter `tamper_mask`
     - no overlay
   - `resize_jpeg`
     - resize long side to one of `512`, `720`, `1080`
     - optional JPEG quality `70~90`
     - if image size changes, apply geometric resize to `tamper_mask`
     - JPEG applies only to image
   - `screenshot_recapture_light`
     - place image into phone-like canvas
     - minimal top/bottom UI bars
     - UI bars go to `ignore_mask`
     - apply same placement/resize to `tamper_mask`
     - avoid heavy blur/recompression by default
5. Keep mask policy:
   - `tamper_mask` only for malicious manipulation regions
   - text/sticker/UI/frame/badge/news overlay must go to `ignore_mask`
   - overlays must never be added to `tamper_mask`
   - geometric transforms apply to image and `tamper_mask` together
   - recompression/blur/color shift apply to image only
   - masks use nearest-neighbor only
6. Reuse the existing fixed-pair generator and evaluator where practical.
7. Avoid duplicate clean rows in generated `meta.jsonl`.
8. Save benchmark generation artifacts under `~/cvf_runs` and eval outputs under `~/cvf_eval_outputs`.
9. Ensure eval `output_root` is not under approved input roots.
10. Required generated artifacts:
    - `images/`
    - `tamper_masks/`
    - `ignore_masks/`
    - `debug_overlays/`
    - `meta.jsonl`
    - `pair_index.json`
    - `artifact_manifest.json`
11. Required evaluation metrics per profile:
    - `sample_count`
    - `3-way accuracy`
    - `macro-F1`
    - `confusion matrix`
    - `real FPR`
    - `synthetic recall`
    - `tampered recall`
    - `localization activation recall`
    - `tampered raw mean IoU`
    - `tampered valid mean IoU` excluding `ignore_mask`
    - `tampered valid mean Dice`
    - `non-tampered high mask rate`
    - `mean p_tampered on tampered`
    - `latency / FPS`
12. Required robustness-drop metrics vs clean:
    - `accuracy_drop`
    - `macro_f1_drop`
    - `real_fpr_increase`
    - `synthetic_recall_drop`
    - `tampered_recall_drop`
    - `localization_activation_recall_drop`
    - `valid_iou_drop`
    - `raw_iou_drop`
    - `mean_p_tampered_drop_on_tampered`
13. Create reports:
    - `small_benchmark_summary.md`
    - `small_benchmark_metrics_table.csv`
    - `small_benchmark_drop_table.csv`
    - `small_benchmark_interpretation.json`
    - `artifact_manifest.json`
14. Interpretation logic must identify:
    - main collapse cause
    - profiles safe for training
    - profiles too severe for early curriculum
    - recommended SNSAug fine-tuning mix
15. Expected interpretation rules:
    - if `canvas_9x16_only` collapses, prioritize geometry/canvas robustness in training
    - if `platform_ui_same_size` is robust, UI overlay itself is not the main cause
    - if `recompression_light` collapses, prioritize compression robustness
    - if `news_meme_overlay` preserves class but lowers IoU, emphasize ignore-mask-based localization loss
    - if `combined_sns_realistic` is too severe, schedule it later in curriculum rather than from epoch 1
16. Add tests covering:
    - generation of added profiles
    - mask geometry synchronization
    - recompression does not alter mask
    - screenshot UI goes to `ignore_mask`
    - no duplicate clean rows
    - report generation
    - evaluator support for the new profiles
17. Add or update `docs/snsaug_v2_small_benchmark_eval.md` with marker:
    - `SNSAUG_V2_SMALL_GEOMETRY_POSTPROCESS_BENCHMARK_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_snsaug_v2_generation_config.py configs/snsaug_v2/snsaug_v2_tiny_fixed_pairs.example.json
```

```bash
python3 scripts/agent/validate_snsaug_v2_fixed_pairs_eval_config.py configs/evaluation/snsaug_v2_fixed_pairs_eval.example.json
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_generation.py
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_fixed_pairs_eval.py
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_small_benchmark_eval.py
```

```bash
python3 scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py --help
```

```bash
python3 scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py --help
```

```bash
grep -q SNSAUG_V2_SMALL_GEOMETRY_POSTPROCESS_BENCHMARK_OK docs/snsaug_v2_small_benchmark_eval.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0058-snsaug-v2-small-benchmark-with-geometry-and-postprocess.md
```

## Acceptance Criteria

- The new geometry/postprocess profiles are selectable through SNSAug V2 generation without any training.
- Fixed-pair generation emits deterministic small benchmark pairs for the required profile set with no duplicate clean rows.
- Geometry transforms keep image and `tamper_mask` synchronized and keep `ignore_mask` policy intact.
- Recompression/postprocess profiles alter only the image unless geometry resizing is explicitly required.
- Frozen-model evaluation accepts the new profiles and computes the required metrics and robustness drops.
- Benchmark report files are produced with metrics tables, drop tables, and interpretation JSON.
- All generation and evaluation outputs remain outside the repository.
- Documentation explains the benchmark scope and evaluation-only nature.

## Stop Condition

Stop immediately if completing this task would require:
- training or fine-tuning,
- writing outputs inside the repository,
- using protected data or checkpoint paths beyond the explicitly allowed manifest/bundle references,
- changing unrelated model code,
- or making a design decision that expands beyond generation/evaluation benchmark scope.
