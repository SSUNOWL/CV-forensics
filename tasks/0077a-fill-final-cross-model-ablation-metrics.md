# Task 0077a: Fill Final Cross-model Ablation Metrics

## Context

Task 0077 generated and audited the final cross-model comparison artifacts. The main `SIDA-7B` vs `mixed_feature_gate` table is valid, but the ablation/context rows still contain `NA`/`null` values for:

- `pre_sns_baseline`
- `failed_single_model_finetune`

The earlier `snsaug_v2_final_decision_report_* / final_decision_report.md` contains a report-ready Markdown metrics table that should fill these ablation rows.

## Role

Codex is the task writer, implementation worker, strict reviewer, and limited repair manager because Claude Code is unavailable.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0077-final-cross-model-comparison-matrix.md`
- `src/cv_forensics/snsaug_v2_final_cross_model_comparison.py`
- `tests/test_snsaug_v2_final_cross_model_comparison.py`
- `/home/rlatjswo/cvf_eval_outputs/snsaug_v2_0077_final_cross_model_comparison_*/final_cross_model_comparison_matrix.json`
- `/home/rlatjswo/cvf_eval_outputs/snsaug_v2_0077_final_cross_model_comparison_*/artifact_manifest.json`
- `/home/rlatjswo/cvf_eval_outputs/snsaug_v2_final_decision_report_*/final_decision_report.md`

## Files Codex May Modify

- `tasks/0077a-fill-final-cross-model-ablation-metrics.md`
- `src/cv_forensics/snsaug_v2_fill_final_cross_model_ablation_metrics.py`
- `scripts/evaluation/run_snsaug_v2_fill_final_cross_model_ablation_metrics.py`
- `scripts/agent/validate_snsaug_v2_fill_final_cross_model_ablation_metrics_config.py`
- `configs/evaluation/snsaug_v2_fill_final_cross_model_ablation_metrics.example.json`
- `tests/test_snsaug_v2_fill_final_cross_model_ablation_metrics.py`
- `docs/snsaug_v2_fill_final_cross_model_ablation_metrics.md`

## Files Claude May Modify

- `tasks/0077a-fill-final-cross-model-ablation-metrics.md`
- `src/cv_forensics/snsaug_v2_fill_final_cross_model_ablation_metrics.py`
- `scripts/evaluation/run_snsaug_v2_fill_final_cross_model_ablation_metrics.py`
- `scripts/agent/validate_snsaug_v2_fill_final_cross_model_ablation_metrics_config.py`
- `configs/evaluation/snsaug_v2_fill_final_cross_model_ablation_metrics.example.json`
- `tests/test_snsaug_v2_fill_final_cross_model_ablation_metrics.py`
- `docs/snsaug_v2_fill_final_cross_model_ablation_metrics.md`

## Forbidden Actions

- Do not train or fine-tune models.
- Do not download datasets, checkpoints, or packages.
- Do not access the network.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, repository `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not write checkpoints, dataset files, or training outputs.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not install packages.
- Do not create large files.
- Do not alter the valid main `SIDA-7B` vs `mixed_feature_gate` metrics except for copying them into filled report artifacts.

## Important Project Facts

- SIDA and `mixed_feature_gate` are not identical systems.
- SIDA emits 3-way labels and masks.
- `mixed_feature_gate` is a deployable policy-gated recovery system.
- The ablation table is context for the failure path and why the gate was needed.
- The final decision report Markdown table columns are: `model`, `profile`, `acc`, `f1`, `real_fpr`, `syn_rec`, `tamp_rec`, `loc_act`, `valid_iou`, `high_mask`.
- Missing family/profile metrics must remain `null` in JSON and `NA` in Markdown/TSV.

## Implementation Requirements

1. Add a pure-Python implementation with no third-party dependencies.
2. Read the latest configured or discovered 0077 final cross-model comparison output root.
3. Read the latest configured or discovered final decision report Markdown file.
4. Parse the Markdown table columns:
   - `model`
   - `profile`
   - `acc`
   - `f1`
   - `real_fpr`
   - `syn_rec`
   - `tamp_rec`
   - `loc_act`
   - `valid_iou`
   - `high_mask`
5. Fill family metrics for `pre_sns_baseline`.
6. Fill family metrics for `failed_single_model_finetune`.
7. Select failed model by preference:
   - model name containing `0064g` and `hybrid`
   - otherwise `0064f`
   - otherwise `0063b`
   - otherwise `0060b`
   - otherwise the model with high tampered recall and low synthetic recall
8. Family groups:
   - Type A: `news_meme_overlay`, `platform_ui_same_size`, `tiktok_like`, `instagram_story_like`, `youtube_shorts_like`, `combined_sns_realistic`
   - Type B: `canvas_9x16_only`, `resize_crop_pad`, `zoom_crop`, `resize_jpeg`, `screenshot_recapture_light`, `recompression_light`
   - Strict: Type A + Type B
9. Output under configured `output_root`, which must be outside the repository:
   - `final_cross_model_comparison_matrix_filled.json`
   - `final_cross_model_comparison_matrix_filled.md`
   - `final_cross_model_comparison_table_filled.tsv`
   - `final_claims_for_report_filled.md`
   - `artifact_manifest.json`

## Validation Commands

```bash
python3 scripts/agent/validate_snsaug_v2_fill_final_cross_model_ablation_metrics_config.py configs/evaluation/snsaug_v2_fill_final_cross_model_ablation_metrics.example.json
```

```bash
python3 tests/test_snsaug_v2_fill_final_cross_model_ablation_metrics.py
```

```bash
python3 scripts/evaluation/run_snsaug_v2_fill_final_cross_model_ablation_metrics.py --help
```

```bash
grep -q SNSAUG_V2_0077A_FILL_FINAL_CROSS_MODEL_ABLATION_METRICS_OK docs/snsaug_v2_fill_final_cross_model_ablation_metrics.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0077a-fill-final-cross-model-ablation-metrics.md
```

## Acceptance Criteria

- The example config validates.
- Tests pass with standard-library Python only.
- Markdown table parsing fills baseline and failed fine-tune family metrics when matching rows exist.
- Missing profile groups remain `null`/`NA`.
- Failed model selection prefers 0064g hybrid when present.
- Main SIDA/gate metrics are preserved.
- Dry-run writes no output artifacts.
- The generated artifact manifest records source files, selected failed model, safety flags, and output paths.
- Changed files are limited to the allowed list.

## Stop Condition

Stop after validation and strict Codex review. Do not commit automatically.

## Marker

SNSAUG_V2_0077A_FILL_FINAL_CROSS_MODEL_ABLATION_METRICS_OK
