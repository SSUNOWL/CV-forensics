# 0061b Fix Checkpoint Comparison Evaluator Real Inference

## Title

Fix SNSAug v2 checkpoint comparison evaluator to use real checkpoint inference.

## Role

Codex-only implementation worker, reviewer, and limited repair manager.

Claude Code is unavailable. Codex must implement directly only after this task file is committed and the user explicitly says `implement`.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0061-real-fixed-pair-evaluation-for-snsaug-finetuned-checkpoints.md`
- `tasks/0061b-fix-checkpoint-comparison-evaluator-real-inference.md`
- `configs/evaluation/snsaug_v2_checkpoint_comparison_eval.example.json`
- `scripts/agent/validate_snsaug_v2_checkpoint_comparison_eval_config.py`
- `scripts/evaluation/run_snsaug_v2_checkpoint_comparison_eval.py`
- `src/cv_forensics/snsaug_v2_checkpoint_comparison_eval.py`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `src/cv_forensics/pre_sns_v3_sns_robustness_eval.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_model.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_training.py`
- `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
- `tests/test_snsaug_v2_checkpoint_comparison_eval.py`
- `tests/test_snsaug_v2_fixed_pairs_eval.py`
- `docs/snsaug_v2_checkpoint_comparison_eval.md`
- `.local/pre_sns_current_best_model_bundle.json`

## Files Codex May Modify

- `configs/evaluation/snsaug_v2_checkpoint_comparison_eval.example.json`
- `scripts/agent/validate_snsaug_v2_checkpoint_comparison_eval_config.py`
- `scripts/evaluation/run_snsaug_v2_checkpoint_comparison_eval.py`
- `src/cv_forensics/snsaug_v2_checkpoint_comparison_eval.py`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_checkpoint_comparison_eval.py`
- `docs/snsaug_v2_checkpoint_comparison_eval.md`

## Forbidden Actions

- Do not train.
- Do not fine-tune.
- Do not run long evaluation against the real fixed pair root during implementation validation unless the user explicitly approves it.
- Do not use network.
- Do not download assets, datasets, models, or checkpoints.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or checkpoint directories recursively.
- Do not write generated outputs inside the repository, except small test fixtures under test temporary directories.
- Do not modify model checkpoints.
- Do not use validation samples for training.
- Do not use evaluation pair images as training images.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not commit changes unless the user explicitly says `commit this`.

## Important Project Facts

- The project is a lightweight multi-head image forensics prototype for `real / synthetic / tampered` classification, conditional tamper localization, generator-family provenance, and template evidence.
- SNSAug v2 fixed-pair evaluation already exists and should be reused where practical.
- 0061 currently produces required files but uses proxy or stub predictions, causing untrustworthy perfect metrics and empty comparisons.
- The 0061b evaluator must compare the same fixed pair root across:
  - pre-SNS baseline bundle
  - SNSAug guarded short 30x3 checkpoint
  - SNSAug medium 150x3 checkpoint
- This task is evaluation-only. It must not start or prepare training.

## Implementation Requirements

1. Replace proxy/stub scoring in `src/cv_forensics/snsaug_v2_checkpoint_comparison_eval.py` with real inference.
   - Load the pre-SNS baseline bundle.
   - Load the base architecture and apply each SNSAug fine-tuned `.pt` checkpoint.
   - Fail if a fine-tuned `.pt` checkpoint is not loaded.
   - Fail if a fine-tuned checkpoint produces no detectable parameter/state delta from the baseline unless explicitly configured as `no_weight_delta`.
   - Record `checkpoint_sha256` and `checkpoint_size_bytes` for every fine-tuned checkpoint.

2. Run inference over every row in `pair_root/meta.jsonl`.
   - Use existing fixed-pair evaluator inference and metric helpers where available.
   - Write `model_eval_records.jsonl` with these fields for every model/record:
     - `model_id`
     - `base_id`
     - `profile`
     - `view`
     - `content_label`
     - `pred_class`
     - `p_real`
     - `p_synthetic`
     - `p_tampered`
     - `localization_activated`
     - `pred_mask_area_pct`
     - `gt_mask_area_pct` where available
     - `valid_iou` for tampered records
     - `raw_iou` for tampered records
     - `ignore_mask_path`
     - `tamper_mask_path`
     - `image_path`

3. Compute metrics only from `model_eval_records.jsonl`.
   - Per model and profile:
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
     - `sample_count`

4. Populate `checkpoint_comparison_summary.json`.
   - Include comparisons for:
     - `pre_sns_baseline` vs `snsaug_guarded_short_30x3`
     - `pre_sns_baseline` vs `snsaug_medium_150x3`
     - `snsaug_guarded_short_30x3` vs `snsaug_medium_150x3`
   - `comparisons` must not be empty.

5. Add sanity guards.
   - Fail if comparisons are empty.
   - Fail if all model/profile accuracy values are exactly `1.0`.
   - Fail if all profiles have identical metrics across all models.
   - Warn if `non_tampered_high_mask_rate == 1.0` and `real_fpr == 0.0`.
   - Fail if `model_eval_records.jsonl` records do not include `p_tampered`.
   - Fail if `checkpoint_sha256` is missing for fine-tuned checkpoints.
   - Fail if fine-tuned checkpoint predictions are byte-identical to baseline predictions on all records unless explicitly marked `no_weight_delta`.

6. Update validation/config behavior as needed.
   - Keep guardrails for `no_training`, `no_finetune`, `no_network`, and `no_download`.
   - Reject invalid checkpoint paths.
   - Reject train manifests as evaluation data.
   - Require outputs outside the repository.

7. Update tests.
   - Dummy checkpoint changes predictions and evaluator detects a difference.
   - Comparisons are not empty.
   - `model_eval_records.jsonl` contains probabilities and predictions.
   - Per-profile metrics are computed from records.
   - All-ones metrics trigger a sanity failure.
   - Invalid checkpoint path fails.
   - No training, fine-tuning, network, or download is triggered.

8. Update docs.
   - Explain real inference vs proxy metrics.
   - Explain sanity guards.
   - Explain that `comparisons=[]` invalidates the run.
   - Keep marker `SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL_OK`.

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
python3 scripts/agent/check_agent_changes.py tasks/0061b-fix-checkpoint-comparison-evaluator-real-inference.md
```

## Acceptance Criteria

- The evaluator no longer generates proxy/stub metrics.
- Fine-tuned `.pt` checkpoints are actually loaded and verified.
- Per-record output includes real class probabilities, predicted class, localization activation, mask area, and IoU fields.
- Per-profile metrics are derived from `model_eval_records.jsonl`.
- `checkpoint_comparison_summary.json` contains the three required comparisons.
- Sanity guards reject empty comparisons, all-perfect metrics, identical metrics, missing probabilities, missing checkpoint hashes, invalid checkpoint paths, and unchanged fine-tuned predictions unless explicitly allowed.
- Documentation clearly states that `comparisons=[]` invalidates the run.
- All validation commands pass.
- `python3 scripts/agent/check_agent_changes.py tasks/0061b-fix-checkpoint-comparison-evaluator-real-inference.md` passes.

## Stop Condition

After creating this task file, stop and report the task path, short summary, current git status, and exact git add / git commit commands. Do not implement until the user commits this task file and explicitly says `implement`.
