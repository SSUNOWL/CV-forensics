# Task 0044: Pre-SNS Clean Long256 Plus Tile Localizer Integrated Report

## Goal

Implement a guarded integrated pre-SNS report runner using clean long256 as the primary detector and the 512 tile localizer as a conditional high-resolution red-mask localizer.

This task is inference/reporting only. Do not train, add SNS augmentation, download data, use network access, install packages, or write outputs inside the repository.

## Files Codex May Modify

- `tasks/0044-pre-sns-clean-long256-plus-tile-localizer-integrated-report.md`
- `configs/inference/pre_sns_v3_long256_tile_integrated_report.example.json`
- `scripts/agent/validate_pre_sns_v3_long256_tile_integrated_report_config.py`
- `scripts/inference/run_pre_sns_v3_long256_tile_integrated_report.py`
- `src/cv_forensics/pre_sns_v3_long256_tile_integrated_report.py`
- `tests/test_pre_sns_v3_long256_tile_integrated_report.py`
- `docs/pre_sns_v3_long256_tile_integrated_report.md`
- `src/cv_forensics/__init__.py`

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_v3_long256_tile_integrated_report_config.py configs/inference/pre_sns_v3_long256_tile_integrated_report.example.json
CUDA_VISIBLE_DEVICES="" python3 tests/test_pre_sns_v3_long256_tile_integrated_report.py
grep -q PRE_SNS_V3_LONG256_TILE_INTEGRATED_REPORT_OK docs/pre_sns_v3_long256_tile_integrated_report.md
python3 scripts/agent/check_agent_changes.py tasks/0044-pre-sns-clean-long256-plus-tile-localizer-integrated-report.md
```

