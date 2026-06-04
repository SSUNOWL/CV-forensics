# Task: 0057b SNSAug V2 Layout Cause Ablation Evaluation

## Task Title

SNSAug V2 layout-cause ablation profile generation and frozen-model evaluation

## Role

Codex-only implementation worker, reviewer, and limited repair manager for an evaluation-only ablation task.

## Files Codex May Read

- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0055-snsaug-v2-source-manifest-audit-overlay-generator-and-tiny-fixed-pairs.md`
- `tasks/0055d-snsaug-v2-layout-only-and-precise-ignore-mask-refactor.md`
- `tasks/0055e-snsaug-v2-overlay-placement-collision-guard.md`
- `tasks/0057-snsaug-v2-tiny-fixed-pairs-frozen-model-evaluation.md`
- `docs/snsaug_v2_generation.md`
- `docs/snsaug_v2_fixed_pairs_eval.md`
- `configs/snsaug_v2/snsaug_v2_tiny_fixed_pairs.example.json`
- `configs/evaluation/snsaug_v2_fixed_pairs_eval.example.json`
- `scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py`
- `scripts/snsaug_v2/preview_snsaug_v2_overlays.py`
- `scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py`
- `scripts/agent/validate_snsaug_v2_generation_config.py`
- `scripts/agent/validate_snsaug_v2_fixed_pairs_eval_config.py`
- `src/cv_forensics/snsaug_v2/configs.py`
- `src/cv_forensics/snsaug_v2/platform_templates.py`
- `src/cv_forensics/snsaug_v2/sns_augmentor.py`
- `src/cv_forensics/snsaug_v2/pair_generator.py`
- `src/cv_forensics/snsaug_v2/source_manifest_audit.py`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_generation.py`
- `tests/test_snsaug_v2_layout_only_and_precise_ignore.py`
- `tests/test_snsaug_v2_overlay_collision_guard.py`
- `tests/test_snsaug_v2_fixed_pairs_eval.py`
- `.local/pre_sns_current_best_model_bundle.json`
- `.local/pre_sns_v3_clean_long256.local.json`

## Files Codex May Modify

- `tasks/0057b-snsaug-v2-layout-cause-ablation-evaluation.md`
- `configs/snsaug_v2/snsaug_v2_tiny_fixed_pairs.example.json`
- `configs/evaluation/snsaug_v2_fixed_pairs_eval.example.json`
- `scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py`
- `scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py`
- `src/cv_forensics/snsaug_v2/configs.py`
- `src/cv_forensics/snsaug_v2/platform_templates.py`
- `src/cv_forensics/snsaug_v2/sns_augmentor.py`
- `src/cv_forensics/snsaug_v2/pair_generator.py`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_generation.py`
- `tests/test_snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_layout_cause_ablation_eval.py`
- `docs/snsaug_v2_fixed_pairs_eval.md`
- `docs/snsaug_v2_layout_ablation_eval.md`

## Forbidden Actions

- Do not train.
- Do not fine-tune.
- Do not run 0054 training.
- Do not use network.
- Do not download assets or datasets.
- Do not install packages.
- Do not write outputs inside the repository.
- Do not use validation failures for training.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not modify unrelated model or training code.

## Important Project Facts

- This repository implements a lightweight multi-head image forensics system for `real / synthetic / tampered` classification, tampered localization, coarse family provenance, and evidence reporting.
- `0057` already added frozen-model evaluation for SNSAug V2 tiny fixed pairs and computes both raw mask metrics and ignore-mask-excluded valid metrics.
- `0055d` separated layout-only profiles from degradation postprocess profiles, so portrait-layout ablations can be evaluated without accidental quality degradation.
- `0055e` added collision-safe deterministic overlay placement, which must remain stable for the same seed.
- The current frozen evaluation bundle is `.local/pre_sns_current_best_model_bundle.json`.
- The clean validation manifest reference is `.local/pre_sns_v3_clean_long256.local.json`.
- This task is evaluation and generation only. It exists to isolate failure causes before any training decision.

## Implementation Requirements

1. Add or expose these SNSAug V2 ablation profiles:
   - `canvas_9x16_only`
   - `canvas_9x16_full_content`
   - `platform_ui_same_size`
   - `tiktok_like_no_actionbar`
   - `instagram_story_no_text_sticker`
   - `youtube_shorts_no_actionbar`
   - plus existing:
     - `news_meme_overlay`
     - `tiktok_like`
     - `instagram_story_like`
     - `youtube_shorts_like`
     - `combined_sns_realistic`
2. `canvas_9x16_only` must:
   - place the source image on a 9:16 canvas
   - apply no platform UI
   - apply no text or sticker
   - apply no degradation
3. `canvas_9x16_full_content` must:
   - use a 9:16 canvas
   - maximize source content area
   - apply no UI and no degradation
4. `platform_ui_same_size` must:
   - preserve original size and aspect ratio as much as practical
   - overlay simplified platform UI only
   - avoid portrait-canvas expansion
   - avoid degradation
5. `tiktok_like_no_actionbar`, `instagram_story_no_text_sticker`, and `youtube_shorts_no_actionbar` must remove the named failure-source elements while preserving the remaining minimal structural layout.
6. Keep layout-only semantics for all new ablation profiles unless a degradation-specific profile is explicitly selected.
7. Ensure generation metadata records enough information to confirm whether UI, text, stickers, action bars, portrait canvas placement, and degradation were applied.
8. Extend tiny fixed-pair generation support so the ablation profiles can be selected and emitted into:
   - `meta.jsonl`
   - `pair_index.json`
   - fixed-pair image/mask/ignore outputs
9. Extend the frozen-model evaluator so it accepts the new profiles and reports:
   - 3-way accuracy
   - macro-F1
   - real FPR
   - synthetic recall
   - tampered recall
   - localization activation recall
   - raw IoU
   - valid IoU excluding `ignore_mask`
   - valid IoU drop
   - tampered score drop
   - fragile class flip
   - fragile activation flip
10. Keep guardrails:
    - `no_training true`
    - `no_finetune true`
    - `no_network true`
    - `no_download true`
    - `output_root` outside repo
    - `output_root` must not be under approved input roots
11. Add tests covering:
    - new ablation profiles generate outputs
    - no degradation for canvas-only profiles
    - no UI for `canvas_9x16_only`
    - `platform_ui_same_size` preserves original size/aspect as much as practical
    - fixed-pair evaluator accepts and aggregates the new profiles
    - config validators still pass
12. Update docs by either:
    - extending `docs/snsaug_v2_fixed_pairs_eval.md`, or
    - adding `docs/snsaug_v2_layout_ablation_eval.md`
13. Documentation must include marker:
    - `SNSAUG_V2_LAYOUT_CAUSE_ABLATION_EVAL_OK`

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
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_layout_cause_ablation_eval.py
```

```bash
python3 scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py --help
```

```bash
python3 scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py --help
```

```bash
grep -q SNSAUG_V2_LAYOUT_CAUSE_ABLATION_EVAL_OK docs/snsaug_v2_layout_ablation_eval.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0057b-snsaug-v2-layout-cause-ablation-evaluation.md
```

## Acceptance Criteria

- The new ablation profiles are selectable through SNSAug V2 generation code without requiring training.
- Canvas-only ablations produce no UI, no text/sticker overlays, and no quality degradation.
- UI-only ablations isolate specific layout elements such as action bars and story stickers.
- Tiny fixed-pair generation can emit the ablation profiles deterministically with seed `3407`.
- Frozen-model evaluation accepts the ablation profiles and computes clean-vs-profile comparisons and robustness drops.
- Tests confirm no unintended degradation for canvas-only profiles and evaluator compatibility with the new profiles.
- All output-writing behavior remains outside the repository.
- Documentation clearly explains the ablation purpose and the evaluation-only scope.

## Stop Condition

Stop immediately if implementing the ablation profiles or evaluation path would require:
- training or fine-tuning,
- writing outputs inside the repository,
- using protected data paths,
- changing unrelated model code,
- or making a design decision that expands beyond evaluation/generation ablation scope.
