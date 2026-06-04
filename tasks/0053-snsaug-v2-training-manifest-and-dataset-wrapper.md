# Task 0053: SNSAug V2 Training Manifest and Dataset Wrapper

## Task Title

Prepare train-only SNSAug V2 training manifest generation and an on-the-fly dataset wrapper for SNS-aware fine-tuning, without performing any actual training.

## Role

Codex-only implementation worker, reviewer, and limited repair manager operating under the repository contract and Codex-only workflow.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0053-snsaug-v2-training-manifest-and-dataset-wrapper.md`
- `tasks/0052-realistic-snsaug-v2-robustness-eval-and-mining.md`
- `tasks/0051-snsaug-v2-realistic-overlay-module-and-pair-generator.md`
- `docs/snsaug_v2_robustness_eval.md`
- `docs/snsaug_v2.md`
- `src/cv_forensics/snsaug_v2_robustness_eval.py`
- `src/cv_forensics/snsaug_v2/__init__.py`
- `src/cv_forensics/snsaug_v2/configs.py`
- `src/cv_forensics/snsaug_v2/sns_augmentor.py`
- `src/cv_forensics/pre_sns_v3_dual_scale_report.py`
- `tests/test_snsaug_v2_robustness_eval.py`
- `tests/test_snsaug_v2.py`
- Any existing safe dataset, manifest, loss, and wrapper helpers already in `src/cv_forensics/`, `scripts/`, `tests/`, and `docs/` that are directly relevant to manifest generation, deterministic sampling, and mask-loss handling

## Files Codex May Modify

- `tasks/0053-snsaug-v2-training-manifest-and-dataset-wrapper.md`
- `configs/training/snsaug_v2_training_manifest.example.json`
- `scripts/agent/validate_snsaug_v2_training_manifest_config.py`
- `scripts/training/build_snsaug_v2_training_manifest.py`
- `src/cv_forensics/snsaug_v2_training_manifest.py`
- `src/cv_forensics/snsaug_v2_dataset_wrapper.py`
- `tests/test_snsaug_v2_training_manifest.py`
- `docs/snsaug_v2_training_manifest.md`

## Forbidden Actions

- Do not modify any file outside the allowed modify list.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not train or fine-tune models.
- Do not use validation or test samples in the training manifest.
- Do not use network resources.
- Do not download datasets or assets.
- Do not install packages.
- Do not write outputs inside the repository.
- Do not create large files.
- Do not modify unrelated model training code.
- Do not commit changes.

## Important Project Facts

- This task only prepares manifests and dataset-wrapper code.
- The source of fragile mining signals is the train-only mining manifest produced by `0052`.
- Validation and test failure samples remain diagnostic only and must not be used for training.
- SNSAug is benign augmentation only and must preserve labels.
- SID-Set samples without family labels must not contribute to family loss.
- Community Forensics-Small samples may contribute to family loss if family label exists.

## Implementation Requirements

### 1. Add Config, Validator, and Builder Script

Add:

- `configs/training/snsaug_v2_training_manifest.example.json`
- `scripts/agent/validate_snsaug_v2_training_manifest_config.py`
- `scripts/training/build_snsaug_v2_training_manifest.py`

The config should support:

- train manifest path
- train-only mining manifest path from `0052`
- output root outside repository
- sampling ratios
- profiles
- severity schedule
- seed

### 2. Add Training Manifest Module

Add:

- `src/cv_forensics/snsaug_v2_training_manifest.py`

It must generate:

- `snsaug_v2_training_manifest.jsonl`
- `snsaug_v2_sampling_summary.json`
- `snsaug_v2_class_balance_summary.json`
- `artifact_manifest.json`

### 3. Training Sampling Groups

For tampered samples, support the following target group mix:

- `stable_correct_anchor`: 40%
- `fragile_correct_to_fail / confidence_fragile / mask_iou_fragile`: 35%
- `clean_fail` low-weight: 15%
- `random_tampered_coverage`: 10%

### 4. Overall Class Balance

Support overall target balance:

- real SNSAug / clean / basic: 25-30%
- synthetic SNSAug / clean / basic: 25-30%
- tampered SNSAug / clean / basic: 40-50%

### 5. Default Augmentation Mode Ratio

Support default mode ratio:

- clean 30%
- basic_aug 30%
- sns_aug 40%

### 6. Curriculum Support

Support curriculum schedule selection:

- Epoch 1-3:
  - clean 50%
  - basic_aug 30%
  - sns_light 20%
- Epoch 4-8:
  - clean 35%
  - basic_aug 30%
  - sns_medium 35%
- Epoch 9+:
  - clean 25%
  - basic_aug 25%
  - sns_medium_or_heavy 50%

The implementation should make this schedule explicit and deterministic.

### 7. Training Manifest Record Fields

Each record must contain:

- `base_id`
- `source_dataset`
- `split` must be `train`
- `image_path`
- `tamper_mask_path` if available
- `content_label`
- `family_label` if available
- `mining_group`
- `recommended_profiles`
- `sampling_weight`
- `clean_weight`
- `basic_aug_weight`
- `sns_aug_weight`
- `max_severity`
- `allow_family_loss`
- `label_preserved` true

### 8. Dataset Wrapper

Add:

- `src/cv_forensics/snsaug_v2_dataset_wrapper.py`

Implement `SNSAugV2DatasetWrapper` that:

- wraps an existing base dataset
- randomly chooses `clean` / `basic_aug` / `sns_aug` according to ratios or curriculum
- applies `SNSAugV2Augmentor` on-the-fly
- returns:
  - `image`
  - `label`
  - `tamper_mask`
  - `ignore_mask`
  - `view`
  - `profile`
  - `aug_meta`
  - `base_id`
  - `family_label` if available
  - `family_loss_mask`

### 9. Mask Loss Helper

Implement:

```python
masked_localization_loss(pred_mask, tamper_mask, ignore_mask, reduction="mean")
```

using:

- `valid_region = 1 - ignore_mask`
- `loss = BCE(pred_mask, tamper_mask) * valid_region`

An optional dice-style helper may be included if narrow and directly relevant.

### 10. Family Loss Policy

- If a sample has no `family_label`, set `family_loss_mask = 0`
- SID-Set samples without family label must not contribute to family loss
- Community Forensics-Small samples may contribute if family label exists

### 11. Split Safety

- Exclude any non-train sample from the generated training manifest.
- Use the train-only mining manifest from `0052` rather than validation mining outputs.
- If non-train rows are encountered, they must be dropped and reflected in summary output.

### 12. Tests

Add:

- `tests/test_snsaug_v2_training_manifest.py`

Tests must cover:

- training manifest excludes val/test split
- group sampling weights are correct
- dataset wrapper returns zero ignore mask for clean
- dataset wrapper returns nonzero ignore mask for overlay profiles
- labels are preserved
- `masked_localization_loss` ignores `ignore_mask` regions
- `family_loss_mask` works

### 13. Docs

Add:

- `docs/snsaug_v2_training_manifest.md`

Include:

- manifest-builder overview
- train-only mining source requirement
- curriculum schedule
- dataset wrapper behavior
- mask loss and family-loss policy
- marker `SNSAUG_V2_TRAINING_MANIFEST_AND_WRAPPER_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_snsaug_v2_training_manifest_config.py configs/training/snsaug_v2_training_manifest.example.json
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_training_manifest.py
```

```bash
grep -q SNSAUG_V2_TRAINING_MANIFEST_AND_WRAPPER_OK docs/snsaug_v2_training_manifest.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0053-snsaug-v2-training-manifest-and-dataset-wrapper.md
```

## Acceptance Criteria

- The training manifest builder writes only train-split records.
- Manifest sampling weights and class-balance summaries follow the configured policy.
- The dataset wrapper supports deterministic clean/basic/sns selection and returns ignore masks.
- `masked_localization_loss` masks out ignore regions.
- `family_loss_mask` is zero when family supervision is unavailable.
- Validation and test samples are excluded from training outputs.
- All artifacts are written outside the repository.
- Tests and validator pass.
- Docs include the required marker and train-only safety policy.

## Stop Condition

Stop after implementing only the allowed files, running the validation commands, running `python3 scripts/agent/check_agent_changes.py tasks/0053-snsaug-v2-training-manifest-and-dataset-wrapper.md`, and reviewing the result in Korean as `PASS` or `NEEDS_FIX`. Do not implement any actual training run, validation-to-train leakage, or repo-local output writing.
