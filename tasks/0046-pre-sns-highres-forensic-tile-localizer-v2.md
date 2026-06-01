# Task 0046: Pre-SNS Highres Forensic Tile Localizer v2

## Goal

Implement a guarded training path for a lightweight high-resolution forensic tile localizer v2. Clean long256 remains the primary detector. This task is pre-SNS only.

## Files Codex May Modify

- `tasks/0046-pre-sns-highres-forensic-tile-localizer-v2.md`
- `configs/training/pre_sns_v3_tile_localizer_v2_train.example.json`
- `scripts/agent/validate_pre_sns_v3_tile_localizer_v2_train_config.py`
- `scripts/training/train_pre_sns_v3_tile_localizer_v2.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_model.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_training.py`
- `tests/test_pre_sns_v3_tile_localizer_v2_training.py`
- `docs/pre_sns_v3_tile_localizer_v2_training.md`
- `src/cv_forensics/__init__.py`

## Validation

```bash
python3 scripts/agent/validate_pre_sns_v3_tile_localizer_v2_train_config.py configs/training/pre_sns_v3_tile_localizer_v2_train.example.json
CUDA_VISIBLE_DEVICES="" python3 tests/test_pre_sns_v3_tile_localizer_v2_training.py
grep -q PRE_SNS_V3_TILE_LOCALIZER_V2_TRAINING_OK docs/pre_sns_v3_tile_localizer_v2_training.md
python3 scripts/agent/check_agent_changes.py tasks/0046-pre-sns-highres-forensic-tile-localizer-v2.md
```

