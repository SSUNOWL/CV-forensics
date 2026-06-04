## Task Title
0057 SNSAug V2 Tiny Fixed Pairs Frozen Model Evaluation

## Role
Codex-only task writer and later implementation worker/reviewer for an evaluation-only SNSAug v2 fixed-pairs robustness audit. This task must not train, fine-tune, or modify model checkpoints.

## Files Codex May Read
- AGENTS.md
- CLAUDE.md
- docs/project_brief.md
- docs/project_contract.md
- configs/project_contract.json
- docs/agent_workflow.md
- tasks/0048-pre-sns-v2-policy-gated-integrated-report.md
- tasks/0050-pre-sns-best-basic-sns-robustness-evaluation.md
- tasks/0051-snsaug-v2-realistic-overlay-module-and-pair-generator.md
- tasks/0052-realistic-snsaug-v2-robustness-eval-and-mining.md
- tasks/0055-snsaug-v2-source-manifest-audit-overlay-generator-and-tiny-fixed-pairs.md
- docs/snsaug_v2_generation.md
- docs/snsaug_v2_robustness_eval.md
- docs/pre_sns_v3_v2_policy_gated_report.md
- .local/pre_sns_current_best_model_bundle.json
- scripts/agent/validate_snsaug_v2_generation_config.py
- scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py
- scripts/snsaug_v2/preview_snsaug_v2_overlays.py
- src/cv_forensics/snsaug_v2_robustness_eval.py
- src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py
- src/cv_forensics/snsaug_v2/__init__.py
- src/cv_forensics/snsaug_v2/configs.py
- src/cv_forensics/snsaug_v2/pair_generator.py
- tests/test_snsaug_v2_generation.py
- relevant validators, scripts, and current git status

## Files Codex May Modify
- tasks/0057-snsaug-v2-tiny-fixed-pairs-frozen-model-evaluation.md
- configs/evaluation/snsaug_v2_fixed_pairs_eval.example.json
- scripts/agent/validate_snsaug_v2_fixed_pairs_eval_config.py
- scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py
- src/cv_forensics/snsaug_v2_fixed_pairs_eval.py
- tests/test_snsaug_v2_fixed_pairs_eval.py
- docs/snsaug_v2_fixed_pairs_eval.md

## Forbidden Actions
- Do not train.
- Do not fine-tune.
- Do not run 0054.
- Do not use network.
- Do not download assets.
- Do not install packages.
- Do not write outputs inside the repository.
- Do not use validation failures for training.
- Do not modify unrelated model code.

## Important Project Facts
- This task is evaluation only.
- The frozen model bundle is the current best pre-SNS bundle at `.local/pre_sns_current_best_model_bundle.json`.
- The input dataset is an SNSAug v2 tiny fixed paired dataset with:
  - `meta.jsonl`
  - `pair_index.json`
  - `images/`
  - `tamper_masks/`
  - `ignore_masks/`
  - `debug_overlays/`
  - `artifact_manifest.json`
- Evaluation must compare clean views against SNSAug views per `base_id`.
- All outputs must be written outside the repository.
- Existing inference/report utilities should be reused where available. The task must not retrain models or alter checkpoints.

## Implementation Requirements
- Implement an evaluation-only fixed-pairs robustness audit within the allowed files only.

- Part A: New module and script
  - Add `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`.
  - Add `scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py`.
  - Add `configs/evaluation/snsaug_v2_fixed_pairs_eval.example.json`.
  - Add `scripts/agent/validate_snsaug_v2_fixed_pairs_eval_config.py`.
  - Add `tests/test_snsaug_v2_fixed_pairs_eval.py`.
  - Add `docs/snsaug_v2_fixed_pairs_eval.md`.

- Part B: Config and guardrails
  - Required config fields:
    - `config_kind: approved_snsaug_v2_fixed_pairs_eval`
    - `execution_mode: approved_local_snsaug_v2_fixed_pairs_eval`
    - `user_approval_text: I_APPROVE_SNSAUG_V2_FIXED_PAIRS_EVAL`
    - `best_bundle_path`
    - `pair_root`
    - `meta_jsonl_path`
    - `output_root`
    - `profiles`
    - `device`
    - optional `max_samples`
    - `no_training: true`
    - `no_finetune: true`
    - `no_network: true`
    - `no_download: true`
  - Guardrails must:
    - reject training
    - reject fine-tuning
    - reject network/download
    - reject `output_root` inside repository
    - require `output_root` outside repository
    - require `pair_root` exists
    - require `meta.jsonl` exists
    - require `best_bundle_path` exists
    - allow only approved input/output roots

- Part C: Input parsing
  - Read `meta.jsonl`.
  - Use rows with `view == "clean"` as clean baseline.
  - Use rows with `view == "sns_aug"` and `profile != "clean"` as SNSAug rows.
  - If rows exist with `view == "sns_aug"` and `profile == "clean"`, ignore them and emit a warning.
  - Group records by `base_id`.
  - Require each `base_id` to have one clean row and zero or more SNSAug rows.
  - Preserve `content_label` as `real`, `synthetic`, or `tampered`.

- Part D: Model inference
  - Run the frozen pre-SNS policy-gated report on each `image_path`.
  - Reuse existing project inference/report utilities where available.
  - Do not retrain or modify checkpoints.
  - Collect when available:
    - `p_real`
    - `p_synthetic`
    - `p_tampered`
    - `pred_class`
    - localization outputs for tampered candidates:
      - `pred_mask_path` or in-memory mask
      - `final_mask_source`
      - `localization_activated`
      - `final_mask_area_pct`
      - `latency_ms`

- Part E: Mask evaluation
  - For tampered samples with `tamper_mask_path`:
    - load transformed `tamper_mask`
    - load `ignore_mask`
    - compute `raw_iou` and `raw_dice` over full image
    - compute `valid_iou` and `valid_dice` excluding `ignore_mask`:
      - `valid_region = 1 - ignore_mask`
      - `pred_valid = pred_mask * valid_region`
      - `gt_valid = tamper_mask * valid_region`
    - `valid_iou` is the primary mask metric
  - For non-tampered samples:
    - compute predicted mask area pct
    - track non-tampered high-mask false positives

- Part F: Required per-record output fields
  - Each record must include:
    - `base_id`
    - `content_label`
    - `view`
    - `profile`
    - `seed`
    - `image_path`
    - `tamper_mask_path`
    - `ignore_mask_path`
    - `pred_class`
    - `class_correct`
    - `p_real`
    - `p_synthetic`
    - `p_tampered`
    - `tampered_score`
    - `localization_activated`
    - `final_mask_source`
    - `final_mask_area_pct`
    - `raw_iou`
    - `raw_dice`
    - `valid_iou`
    - `valid_dice`
    - `latency_ms`
    - `error` if failed

- Part G: Clean-vs-SNS comparisons
  - For each SNSAug row, join with the same `base_id` clean row and compute:
    - `clean_pred_class`
    - `sns_pred_class`
    - `pred_flip`
    - `clean_correct`
    - `sns_correct`
    - `correct_to_wrong`
    - `wrong_to_correct`
    - `clean_p_tampered`
    - `sns_p_tampered`
    - `p_tampered_drop`
    - `clean_valid_iou`
    - `sns_valid_iou`
    - `valid_iou_drop`
    - `clean_raw_iou`
    - `sns_raw_iou`
    - `raw_iou_drop`
    - `clean_localization_activated`
    - `sns_localization_activated`
    - `activation_flip_off`

- Part H: Fragile sample flags
  - Define:
    - `fragile_class_flip`
      - clean correct and SNS incorrect
    - `fragile_confidence_drop`
      - tampered and `p_tampered_drop >= 0.25`
    - `fragile_mask_drop`
      - tampered and `valid_iou_drop >= 0.20`
    - `fragile_activation_flip`
      - clean localization active and SNS localization inactive

- Part I: Aggregate metrics
  - Per profile compute:
    - `sample_count`
    - `3-way accuracy`
    - `macro-F1`
    - `confusion matrix`
    - `real FPR`
    - `synthetic recall`
    - `tampered recall`
    - `tampered raw mean IoU`
    - `tampered raw median IoU`
    - `tampered valid mean IoU`
    - `tampered valid median IoU`
    - `tampered valid mean Dice`
    - `localization activation recall`
    - `non-tampered high mask rate`
    - `mean latency_ms`
    - `FPS` if available

- Part J: Robustness drop metrics
  - Compare each SNS profile to clean:
    - `accuracy_drop`
    - `macro_f1_drop`
    - `real_fpr_increase`
    - `synthetic_recall_drop`
    - `tampered_recall_drop`
    - `valid_iou_drop`
    - `raw_iou_drop`
    - `localization_activation_recall_drop`
    - `mean_p_tampered_drop_on_tampered`

- Part K: Outputs under output_root
  - Write:
    - `snsaug_v2_eval_records.jsonl`
    - `snsaug_v2_eval_comparisons.jsonl`
    - `snsaug_v2_per_profile_metrics.json`
    - `snsaug_v2_robustness_drop_metrics.json`
    - `snsaug_v2_eval_summary.json`
    - `snsaug_v2_worst_samples.json`
    - `snsaug_v2_fragile_candidates.jsonl`
    - `visual_gallery_manifest.json`
    - `artifact_manifest.json`

- Part L: Worst samples
  - Sort by:
    - `fragile_class_flip`
    - `valid_iou_drop`
    - `p_tampered_drop`
    - `activation_flip_off`
    - real false positive under SNSAug
    - synthetic recall failures

- Part M: Documentation
  - `docs/snsaug_v2_fixed_pairs_eval.md` must include:
    - evaluation-only warning
    - input pair dataset format
    - clean vs SNS profile comparison
    - `raw_iou` vs `valid_iou` explanation
    - metrics
    - output schema
    - marker `SNSAUG_V2_FIXED_PAIRS_EVAL_OK`

- Part N: Tests
  - `tests/test_snsaug_v2_fixed_pairs_eval.py` must cover:
    - meta parsing
    - duplicate `sns_aug/profile=clean` filtering
    - clean/SNS join by `base_id`
    - confusion matrix and macro-F1
    - real FPR
    - synthetic recall
    - tampered recall
    - raw IoU/Dice
    - valid IoU/Dice with `ignore_mask`
    - robustness drop metrics
    - fragile sample flags
    - validator guardrails
    - output schema

## Validation Commands
```bash
python3 scripts/agent/validate_snsaug_v2_fixed_pairs_eval_config.py configs/evaluation/snsaug_v2_fixed_pairs_eval.example.json
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_fixed_pairs_eval.py
```

```bash
python3 scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py --help
```

```bash
grep -q SNSAUG_V2_FIXED_PAIRS_EVAL_OK docs/snsaug_v2_fixed_pairs_eval.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0057-snsaug-v2-tiny-fixed-pairs-frozen-model-evaluation.md
```

## Acceptance Criteria
- The evaluator safely parses SNSAug v2 fixed-pair metadata and groups clean/SNS rows by `base_id`.
- Frozen bundle inference runs without training or checkpoint modification.
- `raw_iou` and `valid_iou` are both computed, with `valid_iou` treated as primary.
- Per-profile metrics and clean-vs-SNS drop metrics are produced for real, synthetic, and tampered classes.
- Fragile sample flags and worst-sample outputs are written.
- All outputs are written outside the repository.
- Validator, tests, docs marker, and agent change checks all pass.

## Stop Condition
- Stop after implementing only the allowed files, running the listed validation commands, and reviewing the result as PASS or NEEDS_FIX.
- If implementation would require training, fine-tuning, network access, downloading assets, repo-local output writing, or modifying unrelated model code, stop and report the blocker.
