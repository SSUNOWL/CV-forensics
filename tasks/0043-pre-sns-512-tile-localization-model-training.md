# Task 0043: Pre-SNS 512 Tile Localization Model Training

## Goal

Implement a guarded training pipeline for a lightweight 512 crop/tile localization model. The model is a conditional high-resolution localizer activated after the clean long256 detector predicts tampered or tampered-suspect. It is not a detector replacement.

This task is pre-SNS only. Do not add SNS augmentation, use network access, download datasets, install packages, or write outputs/checkpoints inside the repository.

## Files Codex May Modify

- `tasks/0043-pre-sns-512-tile-localization-model-training.md`
- `configs/training/pre_sns_v3_tile_localizer_train.example.json`
- `scripts/agent/validate_pre_sns_v3_tile_localizer_train_config.py`
- `scripts/training/train_pre_sns_v3_tile_localizer.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_model.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_training.py`
- `tests/test_pre_sns_v3_tile_localizer_training.py`
- `docs/pre_sns_v3_tile_localizer_training.md`
- `src/cv_forensics/__init__.py`

## Requirements

- Add a lightweight torch-only tile localizer model with RGB and fixed high-pass branches.
- Add a manifest-backed dataset loader for 0042 tile manifests.
- Crop source images and GT masks on the fly.
- Use empty masks for negative and hard-negative records.
- Provide BCE, Dice, optional focal, and empty-mask losses.
- Provide guarded no-write dry run and approved actual training.
- Write run artifacts only under `approved_run_root` and checkpoints only under `approved_checkpoint_root`.
- Reject SNS augmentation, network/download, repo-local outputs/checkpoints, and protected path segments.

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_v3_tile_localizer_train_config.py configs/training/pre_sns_v3_tile_localizer_train.example.json
CUDA_VISIBLE_DEVICES="" python3 tests/test_pre_sns_v3_tile_localizer_training.py
grep -q PRE_SNS_V3_TILE_LOCALIZER_TRAINING_OK docs/pre_sns_v3_tile_localizer_training.md
python3 scripts/agent/check_agent_changes.py tasks/0043-pre-sns-512-tile-localization-model-training.md
```

