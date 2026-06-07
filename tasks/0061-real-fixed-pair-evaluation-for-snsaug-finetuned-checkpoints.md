# Task: 0061 Real Fixed-Pair Evaluation for SNSAug Fine-Tuned Checkpoints

## Task Title

Evaluate frozen pre-SNS and SNSAug fine-tuned checkpoints on the fixed SNSAug v2 0058c paired benchmark using real checkpoint inference rather than proxy training summaries.

## Role

Codex-only task writer, implementation worker, reviewer, and limited repair manager for an evaluation-only checkpoint comparison task.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0058c-snsaug-v2-small-benchmark-class-balanced-and-tampered-mask-fix.md`
- `tasks/0058d-snsaug-v2-export-pred-redmask-for-visual-comparison.md`
- `tasks/0060a-snsaug-v2-activation-mask-robustness-finetune-smoke.md`
- `tasks/0060b-snsaug-v2-full-curriculum-finetune.md`
- `tasks/0060b-fix-full-curriculum-actual-training-branch.md`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_losses.py`
- `src/cv_forensics/pre_sns_v3_sns_robustness_eval.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_model.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_training.py`
- `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
- `scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_fixed_pairs_eval.py`
- `.local/pre_sns_current_best_model_bundle.json`

## Files Codex May Modify

- `tasks/0061-real-fixed-pair-evaluation-for-snsaug-finetuned-checkpoints.md`
- `configs/evaluation/snsaug_v2_checkpoint_comparison_eval.example.json`
- `scripts/agent/validate_snsaug_v2_checkpoint_comparison_eval_config.py`
- `scripts/evaluation/run_snsaug_v2_checkpoint_comparison_eval.py`
- `src/cv_forensics/snsaug_v2_checkpoint_comparison_eval.py`
- `tests/test_snsaug_v2_checkpoint_comparison_eval.py`
- `docs/snsaug_v2_checkpoint_comparison_eval.md`

## Forbidden Actions

- Do not train.
- Do not fine-tune.
- Do not use training samples.
- Do not use validation failures as training input.
- Do not use network resources from shell commands.
- Do not download datasets, assets, or checkpoints.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or protected directories.
- Do not inspect protected directories recursively.
- Do not write generated outputs inside the repository.
- Do not modify model checkpoints.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not use `rg`.
- Do not use Claude Code.

## Important Project Facts

- 0060b guarded short `30x3` and medium `150x3` actual runs both completed and wrote `.pt` checkpoints.
- Their summary metrics are identical:
  - `clean_macro_f1 = 0.850`
  - `clean_tampered_recall = 0.550`
  - `clean_valid_iou = 0.320`
  - `snsaug_tampered_recall = 0.550`
  - `snsaug_localization_activation_recall = 0.550`
  - `snsaug_valid_iou = 0.320`
  - `real_fpr = 0.040`
  - `synthetic_recall = 0.750`
- This suggests the 0060b summaries may be proxy or subset metrics rather than true fixed-pair inference results.
- Required evaluation inputs are:
  - frozen pre-SNS baseline bundle: `.local/pre_sns_current_best_model_bundle.json`
  - guarded short checkpoint: `/home/rlatjswo/cvf_checkpoints/snsaug_v2_0060b_guarded_short_actual_20260607_035027/snsaug_aware_multihead_forensics_v1_best.pt`
  - medium checkpoint: `/home/rlatjswo/cvf_checkpoints/snsaug_v2_0060b_medium_actual_150x3_20260607_041453/snsaug_aware_multihead_forensics_v1_best.pt`
  - fixed SNSAug validation pair root: `/home/rlatjswo/cvf_runs/snsaug_v2_0058c_small_val_pairs_20pc_balanced_20260605_220707`
- The fixed pair root is evaluation-only.
- Outputs must be outside the repository.
- Documentation marker:
  - `SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL_OK`

## Implementation Requirements

1. Add an evaluation config:
   - `configs/evaluation/snsaug_v2_checkpoint_comparison_eval.example.json`
2. Add a config validator:
   - `scripts/agent/validate_snsaug_v2_checkpoint_comparison_eval_config.py`
3. Add an evaluation CLI:
   - `scripts/evaluation/run_snsaug_v2_checkpoint_comparison_eval.py`
4. Add an evaluation module:
   - `src/cv_forensics/snsaug_v2_checkpoint_comparison_eval.py`
5. Add tests:
   - `tests/test_snsaug_v2_checkpoint_comparison_eval.py`
6. Add docs:
   - `docs/snsaug_v2_checkpoint_comparison_eval.md`
7. The evaluator must load multiple model entries from config:
   - frozen pre-SNS best bundle
   - SNSAug fine-tuned `.pt` checkpoint
8. The evaluator must run each model on the exact same fixed pair root.
9. The evaluator must produce per-model and per-profile metrics:
   - `accuracy`
   - `macro_f1`
   - `real_fpr`
   - `synthetic_recall`
   - `tampered_recall`
   - `localization_activation_recall`
   - `tampered_valid_mean_iou`
   - `tampered_raw_mean_iou`
   - `non_tampered_high_mask_rate`
   - `synthetic_to_real_confusion`
   - `synthetic_to_tampered_confusion`
10. The evaluator must produce clean-vs-SNS robustness drop metrics per model.
11. The evaluator must produce direct comparisons:
    - baseline vs `30x3`
    - baseline vs `150x3`
    - `30x3` vs `150x3`
12. The evaluator must export worst samples and visual redmask comparison manifest for the `150x3` model.
13. The evaluator must mark:
    - `full_fixed_pair_evaluation_ran=true`
    - `eval_subset_only=false`
    unless explicitly configured for subset testing.
14. Required outputs under external `output_root`:
    - `model_eval_records.jsonl`
    - `model_eval_comparisons.jsonl`
    - `per_model_per_profile_metrics.json`
    - `robustness_drop_by_model.json`
    - `checkpoint_comparison_summary.json`
    - `checkpoint_comparison_report.md`
    - `worst_samples_by_model.json`
    - `visual_gallery_manifest.json`
    - `artifact_manifest.json`
15. Guardrails:
    - `no_training=true`
    - `no_finetune=true`
    - `no_network=true`
    - `no_download=true`
    - `output_root` outside repository
    - `pair_root` must be evaluation-only
    - reject train manifest paths as evaluation data
    - all compared models must reference the same `pair_root`
16. Tests must cover:
    - evaluator loads a dummy fine-tuned checkpoint
    - same `pair_root` used for all models
    - per-profile metrics are produced
    - synthetic confusion metrics are not null
    - `eval_subset_only` flag is correct
    - no training is triggered
17. Tests may use tiny fixtures and subset mode, but the production config and docs must support full fixed-pair evaluation.
18. Do not run the real full fixed-pair evaluation during implementation validation unless separately approved by the user.

## Validation Commands

```bash
python3 scripts/agent/validate_snsaug_v2_checkpoint_comparison_eval_config.py configs/evaluation/snsaug_v2_checkpoint_comparison_eval.example.json
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_checkpoint_comparison_eval.py
```

```bash
python3 scripts/evaluation/run_snsaug_v2_checkpoint_comparison_eval.py --help
```

```bash
grep -q SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL_OK docs/snsaug_v2_checkpoint_comparison_eval.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0061-real-fixed-pair-evaluation-for-snsaug-finetuned-checkpoints.md
```

## Acceptance Criteria

- Config validator accepts the example config.
- Config validator rejects training manifests or training roots as evaluation input.
- Config validator rejects output roots inside the repository.
- Evaluator can load a dummy `.pt` fine-tuned checkpoint in tests.
- Evaluator evaluates all configured models against the same fixed pair root.
- Per-profile metrics include non-null synthetic confusion metrics.
- Clean-vs-SNS robustness drop metrics are written per model.
- Direct comparison summaries are written for baseline vs `30x3`, baseline vs `150x3`, and `30x3` vs `150x3`.
- Worst-sample and visual-gallery manifests are written.
- Full evaluation flags are correct:
  - full mode: `full_fixed_pair_evaluation_ran=true`, `eval_subset_only=false`
  - test subset mode: explicitly marked subset
- No training or fine-tuning code path is invoked.
- Documentation includes `SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL_OK`.
- All changed files are within the allowed modify list.

## Stop Condition

Stop immediately if completing this task would require:

- training,
- fine-tuning,
- using training samples,
- downloading datasets/assets/checkpoints,
- installing packages,
- network access,
- protected path access,
- writing generated outputs inside the repository,
- modifying existing checkpoints,
- changing files outside the allowed modify list,
- or running the real full fixed-pair evaluation without explicit user approval.
