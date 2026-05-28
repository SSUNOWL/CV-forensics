# Task 0035: Pre-SNS v3 Strong Model

## Goal

Build a stronger pre-SNS model and guarded training pipeline to replace the weak v2 baseline.

## Scope

This task is pre-SNS only. Do not add SNS augmentation, download data, install packages, access the network, or write artifacts/checkpoints inside the repository.

## Files Codex May Modify

- `tasks/0035-pre-sns-v3-strong-model.md`
- `configs/training/pre_sns_v3_train.example.json`
- `scripts/agent/validate_pre_sns_v3_train_config.py`
- `scripts/training/train_pre_sns_v3.py`
- `scripts/inference/run_pre_sns_v3_report.py`
- `src/cv_forensics/pre_sns_v3_training.py`
- `src/cv_forensics/pre_sns_v3_model.py`
- `src/cv_forensics/pre_sns_v3_metrics.py`
- `src/cv_forensics/pre_sns_v3_report.py`
- `tests/test_pre_sns_v3_training.py`
- `docs/pre_sns_v3_training.md`
- `src/cv_forensics/__init__.py`

## Requirements

- Add a self-contained v3 RGB plus fixed high-pass CNN with class, tamper binary, family, and mask decoder heads.
- Add class CE, binary tamper CE/BCE, meaningful-label family CE, localization BCEWithLogits plus Dice, and non-tampered empty-mask penalty.
- Add configurable loss weights with defaults: class `1.0`, tamper binary `1.0`, family `0.3`, localization `10.0`, empty-mask `0.2`, dice `1.0`.
- Add v3 metrics including class/per-class/confusion, binary tamper metrics, meaningful family accuracy, score summaries, FPR-constrained tau selection, localization metrics, mask area summaries, per-source and per-family confusion matrices.
- Support guarded train/validation manifests, max sample caps, image size 224/256, CUDA, no-write dry-run, actual approved external artifact/checkpoint writing, class-balanced sampling or loss, optional gradient clipping, optional AMP default false, deterministic seed.
- Write required artifacts under `approved_run_root` and checkpoints under `approved_checkpoint_root`.
- Final actual run JSON must include marker `PRE_SNS_V3_TRAINING_RUN_OK` and all required status/metric/path fields.
- Add v3 report compatibility with clean red overlay, using the same binary mask for overlay and `mask_area_pct`.
- Add config validator for `approved_pre_sns_v3_training` and `approved_local_pre_sns_v3_training`.
- Add plain `python3` runnable tests and docs marker `PRE_SNS_V3_TRAINING_OK`.

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_v3_train_config.py configs/training/pre_sns_v3_train.example.json
CUDA_VISIBLE_DEVICES="" python3 tests/test_pre_sns_v3_training.py
grep -q PRE_SNS_V3_TRAINING_OK docs/pre_sns_v3_training.md
python3 scripts/agent/check_agent_changes.py tasks/0035-pre-sns-v3-strong-model.md
```
