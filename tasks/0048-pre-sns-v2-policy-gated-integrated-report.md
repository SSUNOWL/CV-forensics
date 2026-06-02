# Task 0048: Pre-SNS v2 Policy-Gated Integrated Report

## Task Title

Implement a policy-gated integrated pre-SNS report that uses clean long256 as the primary detector and tile localizer v2 as a conditional high-resolution localizer with suppression, reliability gating, and fallback behavior.

## Role

Codex-only implementation worker, reviewer, and limited repair manager operating under the repository contract and Codex-only workflow.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0048-pre-sns-v2-policy-gated-integrated-report.md`
- `configs/inference/pre_sns_v3_v2_policy_gated_report.example.json`
- `scripts/agent/validate_pre_sns_v3_v2_policy_gated_report_config.py`
- `scripts/inference/run_pre_sns_v3_v2_policy_gated_report.py`
- `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
- `tests/test_pre_sns_v3_v2_policy_gated_report.py`
- `docs/pre_sns_v3_v2_policy_gated_report.md`
- `src/cv_forensics/__init__.py`
- Relevant existing pre-SNS report modules, validators, scripts, and tests needed to mirror established local inference/report patterns, including:
  - `src/cv_forensics/pre_sns_v3_long256_tile_integrated_report.py`
  - `src/cv_forensics/pre_sns_v3_long256_tile_report.py`
  - `src/cv_forensics/pre_sns_v3_report.py`
  - `src/cv_forensics/pre_sns_v3_tile_localizer_v2_model.py`
  - `docs/pre_sns_v3_long256_tile_integrated_report.md`
  - `tests/test_pre_sns_v3_long256_tile_integrated_report.py`

## Files Codex May Modify

- `tasks/0048-pre-sns-v2-policy-gated-integrated-report.md`
- `configs/inference/pre_sns_v3_v2_policy_gated_report.example.json`
- `scripts/agent/validate_pre_sns_v3_v2_policy_gated_report_config.py`
- `scripts/inference/run_pre_sns_v3_v2_policy_gated_report.py`
- `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
- `tests/test_pre_sns_v3_v2_policy_gated_report.py`
- `docs/pre_sns_v3_v2_policy_gated_report.md`
- `src/cv_forensics/__init__.py`

## Forbidden Actions

- Do not modify any file outside the allowed modify list.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not add SNS augmentation.
- Do not train models.
- Do not install packages.
- Do not use network resources.
- Do not download datasets.
- Do not write outputs inside the repository.
- Do not create large files.
- Do not commit changes.

## Important Project Facts

- This is a pre-SNS inference/reporting task only.
- Clean long256 remains the primary detector and source of final class gating.
- Tile localizer v2 improves tampered red-mask IoU substantially, but raw masks can over-activate on non-tampered images.
- The new report must be policy-gated, not a blind v2 replacement path.
- The project contract requires local-only, reversible changes and forbids protected path access, training, package installation, and network actions.

## Implementation Requirements

- Add module:
  - `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
- Add script:
  - `scripts/inference/run_pre_sns_v3_v2_policy_gated_report.py`
- The report pipeline must:
  - run clean long256 detector/report first
  - if long256 class is not `tampered`, suppress v2 mask by default and set `localized_evidence_status` to `suppressed_non_tampered`
  - if long256 class is `tampered`, run tile localizer v2
  - threshold v2 probability mask using config `mask_threshold`
  - compute `mask_area_pct`, `component_count`, and `largest_component_area_pct`
  - if `mask_area_pct` is below `min_area_pct` or above `max_area_pct`, fall back to baseline long256 mask when available and set `final_mask_source` to `baseline_fallback`
  - otherwise use the v2 mask
  - write final clean red overlay with no text
- Configurable policy fields must support:
  - `mask_threshold` default `0.4`
  - `min_area_pct` default `0.01`
  - `max_area_pct` default `35.0`
  - `suppress_non_tampered_mask` default `true`
  - `fallback_to_baseline_on_v2_unreliable` default `true`
  - optional `component_filtering`
  - optional `min_largest_component_area_pct`
  - optional `max_component_count`
- Inputs must support:
  - `long256_checkpoint_path`
  - `tile_localizer_v2_checkpoint_path`
  - `image_path` or `sample_list_path`
  - optional `gt_mask_path`
  - `output_root`
- Artifacts written under external `output_root` must include:
  - `policy_gated_report.json`
  - `final_mask.png`
  - `final_clean_red_overlay.png`
  - `baseline_clean_red_overlay.png` when available
  - `tile_v2_clean_red_overlay.png`
  - `tile_v2_probability_mask.png`
  - `comparison_sheet.jpg` when GT exists
  - `artifact_manifest.json`
- When GT exists, report metrics must include:
  - `baseline_iou`
  - `baseline_dice`
  - `v2_raw_iou`
  - `v2_raw_dice`
  - `final_iou`
  - `final_dice`
  - `final_iou_delta_vs_baseline`
  - `final_mask_source`
- Add example config:
  - `configs/inference/pre_sns_v3_v2_policy_gated_report.example.json`
- Add validator:
  - `scripts/agent/validate_pre_sns_v3_v2_policy_gated_report_config.py`
  - require `config_kind` = `approved_pre_sns_v3_v2_policy_gated_report`
  - require `execution_mode` = `approved_local_pre_sns_v3_v2_policy_gated_report`
  - require `no_download`, `no_network`, `no_sns_augmentation`, and `no_training` to be `true`
  - require `output_root` outside repo
  - require approved input roots outside repo
- Add tests:
  - `tests/test_pre_sns_v3_v2_policy_gated_report.py`
  - cover non-tampered suppression
  - cover tampered v2 mask use
  - cover too-small-area fallback
  - cover too-large-area fallback
  - cover final IoU computation
  - cover validator guardrails
  - cover no repository writes
- Add docs:
  - `docs/pre_sns_v3_v2_policy_gated_report.md`
  - include marker `PRE_SNS_V3_V2_POLICY_GATED_REPORT_OK`
  - explain that clean long256 is the primary detector and v2 is a conditional localizer

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_v3_v2_policy_gated_report_config.py configs/inference/pre_sns_v3_v2_policy_gated_report.example.json
CUDA_VISIBLE_DEVICES="" python3 tests/test_pre_sns_v3_v2_policy_gated_report.py
grep -q PRE_SNS_V3_V2_POLICY_GATED_REPORT_OK docs/pre_sns_v3_v2_policy_gated_report.md
python3 scripts/agent/check_agent_changes.py tasks/0048-pre-sns-v2-policy-gated-integrated-report.md
```

## Acceptance Criteria

- The new module and runner produce a guarded integrated report that always treats clean long256 as the primary detector.
- Non-tampered long256 classifications suppress v2 localization by default and report `localized_evidence_status` as `suppressed_non_tampered`.
- Tampered long256 classifications may use the v2 mask only when policy thresholds and reliability checks pass; otherwise baseline fallback is used when available.
- Output artifacts and metrics are written only under external approved roots and include the required report files and overlays.
- The example config and validator enforce the required config kind, execution mode, and local safety guardrails.
- Tests cover suppression, v2 usage, fallback cases, IoU computation, validator behavior, and no-repo-write behavior.

## Stop Condition

- Stop after implementing only the allowed-file changes, running the validation commands, and running `python3 scripts/agent/check_agent_changes.py tasks/0048-pre-sns-v2-policy-gated-integrated-report.md`.
- Review the result in Korean and report `PASS` or `NEEDS_FIX`.
- Do not commit; wait for the user to review and commit manually.
