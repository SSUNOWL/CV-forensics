# Task 0045: Pre-SNS Final Visual Audit

## Goal

Run a final pre-SNS visual audit for the integrated clean long256 + tile localizer report and decide whether the pre-SNS model is ready for SNS robustness evaluation.

This task is evaluation/reporting only. Do not train, add SNS augmentation, download datasets, use network access, install packages, or write outputs inside the repository.

## Files Codex May Modify

- `tasks/0045-pre-sns-final-visual-audit.md`
- `configs/evaluation/pre_sns_v3_final_visual_audit.example.json`
- `scripts/agent/validate_pre_sns_v3_final_visual_audit_config.py`
- `scripts/evaluation/run_pre_sns_v3_final_visual_audit.py`
- `src/cv_forensics/pre_sns_v3_final_visual_audit.py`
- `tests/test_pre_sns_v3_final_visual_audit.py`
- `docs/pre_sns_v3_final_visual_audit.md`
- `src/cv_forensics/__init__.py`

## Validation

```bash
python3 scripts/agent/validate_pre_sns_v3_final_visual_audit_config.py configs/evaluation/pre_sns_v3_final_visual_audit.example.json
CUDA_VISIBLE_DEVICES="" python3 tests/test_pre_sns_v3_final_visual_audit.py
grep -q PRE_SNS_V3_FINAL_VISUAL_AUDIT_OK docs/pre_sns_v3_final_visual_audit.md
python3 scripts/agent/check_agent_changes.py tasks/0045-pre-sns-final-visual-audit.md
```

