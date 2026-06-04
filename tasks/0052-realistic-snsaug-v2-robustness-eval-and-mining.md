# Task 0052: Realistic SNSAug V2 Robustness Evaluation and Fragile Mining

## Task Title

Integrate SNSAug V2 realistic overlay augmentation into the frozen pre-SNS robustness evaluation pipeline, then generate fragile-sample mining outputs for evaluation reporting and train-only manifest construction without using validation failures directly for training.

## Role

Codex-only implementation worker, reviewer, and limited repair manager operating under the repository contract and Codex-only workflow.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0052-realistic-snsaug-v2-robustness-eval-and-mining.md`
- `tasks/0051-snsaug-v2-realistic-overlay-module-and-pair-generator.md`
- `tasks/0050-pre-sns-best-basic-sns-robustness-evaluation.md`
- `.local/pre_sns_current_best_model_bundle.json`
- `docs/snsaug_v2.md`
- `docs/pre_sns_v3_sns_robustness_eval.md`
- `src/cv_forensics/snsaug_v2/__init__.py`
- `src/cv_forensics/snsaug_v2/configs.py`
- `src/cv_forensics/snsaug_v2/sns_augmentor.py`
- `src/cv_forensics/snsaug_v2/pair_generator.py`
- `src/cv_forensics/pre_sns_v3_sns_robustness_eval.py`
- `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
- `tests/test_snsaug_v2.py`
- `tests/test_pre_sns_v3_sns_robustness_eval.py`
- `src/cv_forensics/__init__.py`
- Relevant existing evaluation, metrics, and visualization helpers already in `src/cv_forensics/`, `scripts/`, `tests/`, and `docs/` that are directly needed for deterministic robustness reporting, manifest writing, and train/validation split-safe mining outputs

## Files Codex May Modify

- `tasks/0052-realistic-snsaug-v2-robustness-eval-and-mining.md`
- `configs/evaluation/snsaug_v2_robustness_eval.example.json`
- `scripts/agent/validate_snsaug_v2_robustness_eval_config.py`
- `scripts/evaluation/run_snsaug_v2_robustness_eval.py`
- `src/cv_forensics/snsaug_v2_robustness_eval.py`
- `tests/test_snsaug_v2_robustness_eval.py`
- `docs/snsaug_v2_robustness_eval.md`

## Forbidden Actions

- Do not modify any file outside the allowed modify list.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not train or fine-tune models.
- Do not download datasets, assets, or any remote resources.
- Do not use network resources.
- Do not install packages.
- Do not write outputs inside the repository.
- Do not create large files.
- Do not use validation failures directly for training.
- Do not modify unrelated model training code.
- Do not commit changes.

## Important Project Facts

- This task is evaluation, reporting, and mining only.
- The frozen pre-SNS current best bundle remains `.local/pre_sns_current_best_model_bundle.json`.
- `0050` basic SNS degradation evaluation already exists and should be reused where appropriate.
- `0051` SNSAug V2 realistic overlays already exist and should be integrated rather than reimplemented.
- Validation split is for evaluation only.
- If train-split mining is requested through config, train-only outputs must be generated from train manifests only and must not contain validation samples.
- Content labels never change under SNSAug V2.

## Implementation Requirements

### 1. Add Evaluation Module and CLI

Add:

- `src/cv_forensics/snsaug_v2_robustness_eval.py`
- `scripts/evaluation/run_snsaug_v2_robustness_eval.py`

The evaluation must run the frozen pre-SNS best policy-gated report on:

- clean
- SNSAug V2 profiles

### 2. Add Config and Validator

Add:

- `configs/evaluation/snsaug_v2_robustness_eval.example.json`
- `scripts/agent/validate_snsaug_v2_robustness_eval_config.py`

The config should support at least:

- `best_bundle_path`
- `validation_manifest_path`
- optional `train_manifest_path`
- `output_root`
- `profiles`
- `severity`
- `seed`
- `max_samples`
- `samples_per_class`
- `balanced_sampling`
- `device`
- `no_training`
- `no_download`
- `no_network`

The validator must enforce:

- evaluation-only execution
- no training
- no network
- no download
- approved input roots
- approved output roots
- output root outside repository
- protected path rejection

### 3. Evaluation Profiles

Implement evaluation over:

- `clean`
- `jpeg_resize`
- `tiktok_like`
- `instagram_story_like`
- `youtube_shorts_like`
- `news_meme_overlay`
- `ai_badge_overlay`
- `annotation_sticker`
- `combined_sns_realistic`

### 4. Per-Sample Record Fields

Each joined clean/profile record must include:

- `base_id`
- `split`
- `label`
- `profile`
- `seed`
- `clean_pred`
- `profile_pred`
- `clean_correct`
- `profile_correct`
- `p_real_clean`
- `p_synthetic_clean`
- `p_tampered_clean`
- `p_real_profile`
- `p_synthetic_profile`
- `p_tampered_profile`
- `p_tampered_drop`
- `clean_iou`
- `profile_iou`
- `iou_drop`
- `clean_dice`
- `profile_dice`
- `dice_drop`
- `clean_localization_activated`
- `profile_localization_activated`
- `activation_flip_off`
- `pred_flip`
- `correct_to_wrong`
- `ignore_mask_area_pct`
- `overlay_area_pct`
- `final_mask_source`
- `latency_ms`

### 5. Mining Group Rules

Implement group assignment for:

- `stable_correct_anchor`
  - `label == "tampered"`
  - `clean_pred == "tampered"`
  - `clean_correct == true`
  - `p_tampered_clean >= 0.75`
  - `clean_iou >= 0.35`
- `fragile_correct_to_fail`
  - `stable_correct_anchor == true`
  - and `profile_correct == false`
- `confidence_fragile`
  - `stable_correct_anchor == true`
  - and `p_tampered_drop >= 0.25`
- `mask_iou_fragile`
  - `stable_correct_anchor == true`
  - and `iou_drop >= 0.20`
- `threshold_flip`
  - `clean_localization_activated == true`
  - and `profile_localization_activated == false`
- `clean_fail`
  - `label == "tampered"`
  - and `clean_correct == false`
- `noisy_candidate`
  - `label == "tampered"`
  - and `clean_iou` is very low
  - and predictions are unstable across profiles

The implementation must make the `noisy_candidate` rule explicit and deterministic in code and docs.

### 6. Aggregate Metrics

Per profile, compute:

- 3-way accuracy
- macro-F1
- confusion matrix
- real false positive rate
- synthetic recall
- tampered recall
- tampered mean IoU
- tampered median IoU
- tampered mean Dice
- localization activation recall
- mean `p_tampered` drop on tampered
- mean IoU drop
- number of fragile samples
- number of threshold flips
- latency and FPS if available

### 7. Outputs

Write under approved external `output_root`:

- `snsaug_v2_robustness_records.jsonl`
- `snsaug_v2_per_profile_metrics.json`
- `snsaug_v2_robustness_summary.json`
- `snsaug_v2_worst_profiles.json`
- `snsaug_v2_worst_samples.json`
- `snsaug_v2_fragile_candidates_val.jsonl`
- `snsaug_v2_fragile_candidates_train.jsonl` if train manifest provided
- `snsaug_v2_training_mining_manifest_train_only.jsonl` if train manifest provided
- `visual_gallery_manifest.json`
- `artifact_manifest.json`

### 8. Split Safety and Mining Policy

- Validation outputs are diagnostic only.
- Validation fragile cases must not be emitted into the train-only mining manifest.
- If a train manifest is provided, produce a separate train-only fragile output and train-only mining manifest.
- The train-only mining manifest must not include validation sample IDs.

### 9. Tests

Add:

- `tests/test_snsaug_v2_robustness_eval.py`

Tests must cover:

- clean/profile join
- group assignment logic
- metric aggregation
- validator guardrails
- train-only manifest generation does not include validation samples
- output schema

### 10. Docs

Add:

- `docs/snsaug_v2_robustness_eval.md`

Include:

- evaluation-only intent
- use of SNSAug V2 profiles
- fragile mining group definitions
- explicit separation between validation diagnostics and train-only mining
- marker `SNSAUG_V2_ROBUSTNESS_EVAL_AND_MINING_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_snsaug_v2_robustness_eval_config.py configs/evaluation/snsaug_v2_robustness_eval.example.json
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_robustness_eval.py
```

```bash
grep -q SNSAUG_V2_ROBUSTNESS_EVAL_AND_MINING_OK docs/snsaug_v2_robustness_eval.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0052-realistic-snsaug-v2-robustness-eval-and-mining.md
```

## Acceptance Criteria

- SNSAug V2 profiles are integrated into a deterministic robustness evaluation path.
- Clean/profile records contain the required joined fields.
- Fragile group assignment is explicit and reproducible.
- Aggregate per-profile metrics are written.
- Validation outputs remain diagnostic only.
- Train-only mining outputs, when requested, exclude validation samples.
- All artifacts are written outside the repository.
- Tests and validator pass.
- Docs include the required marker and split-safety policy.

## Stop Condition

Stop after implementing only the allowed files, running the validation commands, running `python3 scripts/agent/check_agent_changes.py tasks/0052-realistic-snsaug-v2-robustness-eval-and-mining.md`, and reviewing the result in Korean as `PASS` or `NEEDS_FIX`. Do not implement training, downloading, repo-local outputs, or validation-to-train leakage.
