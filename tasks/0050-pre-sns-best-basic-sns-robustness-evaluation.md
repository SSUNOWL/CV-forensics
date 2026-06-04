# Task 0050: Pre-SNS Best Basic SNS Robustness Evaluation

## Task Title

Implement a basic SNS-like degradation robustness evaluation pipeline for the frozen pre-SNS best bundle using the current best policy-gated report, without any SNS augmentation training.

## Role

Codex-only implementation worker, reviewer, and limited repair manager operating under the repository contract and Codex-only workflow.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0050-pre-sns-best-basic-sns-robustness-evaluation.md`
- `tasks/0048-pre-sns-v2-policy-gated-integrated-report.md`
- `tasks/0046-pre-sns-highres-forensic-tile-localizer-v2.md`
- `tasks/0045-pre-sns-final-visual-audit.md`
- `.local/pre_sns_current_best_model_bundle.json`
- `configs/evaluation/pre_sns_v3_sns_robustness_eval.example.json`
- `scripts/agent/validate_pre_sns_v3_sns_robustness_eval_config.py`
- `scripts/evaluation/run_pre_sns_v3_sns_robustness_eval.py`
- `src/cv_forensics/pre_sns_v3_sns_robustness_eval.py`
- `tests/test_pre_sns_v3_sns_robustness_eval.py`
- `docs/pre_sns_v3_sns_robustness_eval.md`
- `src/cv_forensics/__init__.py`
- Relevant existing report/evaluation modules needed to mirror established local evaluation patterns, including:
  - `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
  - `src/cv_forensics/pre_sns_v3_final_visual_audit.py`
  - `src/cv_forensics/pre_sns_v3_gt_iou_mining.py`
  - `src/cv_forensics/pre_sns_v3_report.py`
  - `scripts/evaluation/run_pre_sns_v3_final_visual_audit.py`
  - `tests/test_pre_sns_v3_v2_policy_gated_report.py`
  - `docs/pre_sns_v3_v2_policy_gated_report.md`

## Files Codex May Modify

- `tasks/0050-pre-sns-best-basic-sns-robustness-evaluation.md`
- `configs/evaluation/pre_sns_v3_sns_robustness_eval.example.json`
- `scripts/agent/validate_pre_sns_v3_sns_robustness_eval_config.py`
- `scripts/evaluation/run_pre_sns_v3_sns_robustness_eval.py`
- `src/cv_forensics/pre_sns_v3_sns_robustness_eval.py`
- `tests/test_pre_sns_v3_sns_robustness_eval.py`
- `docs/pre_sns_v3_sns_robustness_eval.md`
- `src/cv_forensics/__init__.py`

## Forbidden Actions

- Do not modify any file outside the allowed modify list.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not train or fine-tune models.
- Do not add SNS augmentation training.
- Do not download datasets.
- Do not use network resources.
- Do not install packages.
- Do not write outputs inside the repository.
- Do not create large files.
- Do not commit changes.

## Important Project Facts

- This task is evaluation only.
- `0046` phase v2 plus `0048` policy-gated report is frozen as the current best pre-SNS bundle.
- `0049` replay medium was rejected and must not be used as the active best bundle.
- This task evaluates robustness under basic SNS-like degradation only, not the final realistic SNS augmentation overlay pipeline.
- Validation and test failure samples must not be repurposed as training data.

## Implementation Requirements

### 1. Add Evaluation Module and Script

- Add module:
  - `src/cv_forensics/pre_sns_v3_sns_robustness_eval.py`
- Add script:
  - `scripts/evaluation/run_pre_sns_v3_sns_robustness_eval.py`

### 2. Inputs

- Support config inputs:
  - `best_bundle_path`
  - `validation_manifest_path`
  - `output_root`
  - `max_samples`
  - `samples_per_class`
  - `balanced_sampling`
  - `perturbations`
  - `device`
  - `no_training`
  - `no_network`
  - `no_download`
  - `no_sns_augmentation_training`

### 3. Config Validator

- Add:
  - `configs/evaluation/pre_sns_v3_sns_robustness_eval.example.json`
  - `scripts/agent/validate_pre_sns_v3_sns_robustness_eval_config.py`
- Validator rules:
  - `config_kind` must be `approved_pre_sns_v3_sns_robustness_eval`
  - `execution_mode` must be `approved_local_pre_sns_v3_sns_robustness_eval`
  - require `no_training` true
  - require `no_download` true
  - require `no_network` true
  - require `no_sns_augmentation_training` true
  - require `output_root` outside repository
  - require approved input roots
  - require approved output roots
  - reject network/download/training
  - reject repo-local output
  - reject protected paths

### 4. Required Perturbations

- Implement support for:
  - `clean`
  - `jpeg_q95`
  - `jpeg_q85`
  - `jpeg_q75`
  - `jpeg_q60`
  - `resize_long_1080`
  - `resize_long_720`
  - `resize_long_512`
  - `resize_long_1080_jpeg_q85`
  - `resize_long_720_jpeg_q75`
  - `resize_long_512_jpeg_q75`
  - `mild_center_crop_95pct_resize_back`
  - optional `webp_q80` if PIL supports it

### 5. Per-Sample Evaluation Flow

For each selected sample:

1. Load the clean image and label metadata from the validation manifest.
2. Load GT tamper mask if available.
3. Create transformed image copies under `output_root/transformed_inputs/`.
4. For JPEG-only transforms:
   - transform image only
   - keep GT mask unchanged
5. For resize/crop geometric transforms:
   - apply the exact same geometric transform to the GT mask
   - use nearest-neighbor interpolation for masks
6. Run the frozen `0048` policy-gated report on:
   - clean
   - each transformed variant
7. Collect per-sample, per-perturbation records.

### 6. Required Per-Record Fields

- `base_id`
- `sample_id`
- `source_image_path`
- `source_mask_path` if available
- `transformed_image_path`
- `transformed_mask_path` if available
- `label`
- `perturbation`
- `pred_class`
- `class_correct`
- `p_real`
- `p_synthetic`
- `p_tampered`
- `tampered_score`
- `final_mask_source`
- `final_iou`
- `final_dice`
- `final_mask_area_pct`
- `latency_ms` if available
- `fps` if available
- `localization_activated` if available
- `activation_threshold` if available
- `error_message` if failed

### 7. Clean-vs-Perturbed Join Fields

For each non-clean perturbation, join with the same `base_id` clean record and add:

- `clean_pred_class`
- `clean_p_tampered`
- `clean_final_iou`
- `clean_final_dice`
- `clean_localization_activated`
- `pred_flip_from_clean`
- `clean_correct`
- `perturbed_correct`
- `correct_to_wrong`
- `wrong_to_correct`
- `p_tampered_drop`
- `iou_drop`
- `dice_drop`
- `activation_flip_off`
- `fragile_candidate` true if:
  - `label == "tampered"`
  - `clean_correct == true`
  - and one of:
    - `perturbed_correct == false`
    - `p_tampered_drop >= 0.25`
    - `iou_drop >= 0.20`
    - `activation_flip_off == true`

### 8. Required Aggregate Metrics

Per perturbation:

- `sample_count`
- `class_accuracy`
- `macro_f1`
- `confusion_matrix` for `real / synthetic / tampered`
- `real_false_positive_rate`
- `synthetic_recall`
- `tampered_recall`
- `final_mean_iou` over tampered samples with GT mask
- `final_median_iou` over tampered samples with GT mask
- `final_mean_dice` over tampered samples with GT mask
- `final_median_dice` over tampered samples with GT mask
- `failed_red_mask_count` if applicable
- `localization_activation_recall` over tampered samples if available
- `mean_latency_ms` if available
- `fps` if available

Robustness drop metrics versus clean:

- `accuracy_drop`
- `macro_f1_drop`
- `synthetic_recall_drop`
- `tampered_recall_drop`
- `real_fpr_increase`
- `mean_iou_drop`
- `median_iou_drop`
- `mean_dice_drop`
- `localization_activation_recall_drop`
- `mean_p_tampered_drop_on_tampered`

### 9. Worst-Case Outputs

- `worst_perturbations` sorted by:
  - `tampered_recall_drop`
  - `mean_iou_drop`
  - `macro_f1_drop`
- `worst_samples` sorted by:
  - `fragile_candidate`
  - `p_tampered_drop`
  - `iou_drop`
  - `pred_flip_from_clean`
- `visual_gallery_manifest` for top-N worst cases
- optional comparison sheets for top-N worst cases

### 10. Required Outputs Under Approved External `output_root`

- `transformed_inputs/`
- `sns_robustness_records.jsonl`
- `sns_robustness_summary.json`
- `per_perturbation_metrics.json`
- `robustness_drop_metrics.json`
- `worst_samples.json`
- `worst_perturbations.json`
- `fragile_candidates.jsonl`
- `visual_gallery_manifest.json`
- `artifact_manifest.json`
- optional comparison sheets for top-N worst cases

### 11. Tests

- Add:
  - `tests/test_pre_sns_v3_sns_robustness_eval.py`
- Cover:
  - JPEG transform keeps mask unchanged
  - resize transform resizes mask with nearest interpolation
  - mild center crop transforms image and mask consistently
  - per-perturbation metric aggregation
  - macro-F1 and confusion matrix computation
  - clean-vs-perturbed join and drop fields
  - `fragile_candidate` logic
  - validator guardrails
  - output schema

### 12. Docs

- Add:
  - `docs/pre_sns_v3_sns_robustness_eval.md`
- Include marker:
  - `PRE_SNS_V3_SNS_ROBUSTNESS_EVAL_OK`
- Docs must clearly state:
  - this is evaluation only
  - no SNS augmentation training occurs here
  - val/test failure samples must not be used directly for training
  - train split mining must be done separately for training manifests

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_v3_sns_robustness_eval_config.py configs/evaluation/pre_sns_v3_sns_robustness_eval.example.json
CUDA_VISIBLE_DEVICES="" python3 tests/test_pre_sns_v3_sns_robustness_eval.py
grep -q PRE_SNS_V3_SNS_ROBUSTNESS_EVAL_OK docs/pre_sns_v3_sns_robustness_eval.md
python3 scripts/agent/check_agent_changes.py tasks/0050-pre-sns-best-basic-sns-robustness-evaluation.md
```

## Acceptance Criteria

- The evaluation pipeline runs the frozen current-best bundle on clean and basic SNS-like degraded validation variants without training or fine-tuning.
- Image-only perturbations leave GT masks unchanged where appropriate, and geometric perturbations apply the same transform to masks with nearest-neighbor interpolation.
- Records include the required prediction, localization, clean-vs-perturbed comparison, and failure/drop fields.
- Aggregate metrics, robustness drops, worst perturbations, worst samples, fragile candidates, and gallery manifests are produced under external approved roots.
- The validator enforces evaluation-only local safety guardrails and rejects repo-local outputs and protected paths.
- Tests and validation commands pass without network use, downloads, training, installation, or repository-local outputs.

## Stop Condition

- Stop after implementing only the allowed-file changes, running the validation commands, and running `python3 scripts/agent/check_agent_changes.py tasks/0050-pre-sns-best-basic-sns-robustness-evaluation.md`.
- Review the result in Korean and report `PASS` or `NEEDS_FIX`.
- Do not commit; wait for the user to review and commit manually.
