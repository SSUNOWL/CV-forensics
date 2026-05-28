# Task 0036: Pre-SNS v3 Dual-Scale Calibrated Report and Mining

## Role

Codex-only implementation worker, strict self-reviewer, and limited repair manager.

Claude Code is unavailable for this task. Codex must implement directly only after this task file is committed and the user explicitly says `implement`.

## Goal

Improve final pre-SNS v3 report quality without starting another full training run.

Use the existing strong clean v3 checkpoints at two image scales:

- clean long224
- clean long256

The long256 checkpoint is the primary class-decision model because observed paired comparisons show better classification and tampered detection than long224. Localization remains mixed, so this task adds report-level calibration, mask post-processing, class-mask consistency gating, dual-scale comparison artifacts, and hard-negative / hard-positive mining.

This task is pre-SNS only.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0036-pre-sns-v3-dual-scale-calibrated-report-and-mining.md`
- `tasks/0035-pre-sns-v3-strong-model.md`
- `scripts/inference/run_pre_sns_v3_report.py`
- `scripts/evaluation/evaluate_pre_sns_baseline.py`
- `scripts/agent/check_agent_changes.py`
- `scripts/agent/validate_pre_sns_v3_train_config.py`
- `scripts/agent/validate_pre_sns_inference_report.py`
- `scripts/agent/validate_pre_sns_evaluation.py`
- `src/cv_forensics/pre_sns_v3_report.py`
- `src/cv_forensics/pre_sns_v3_model.py`
- `src/cv_forensics/pre_sns_v3_metrics.py`
- `src/cv_forensics/pre_sns_v3_training.py`
- `src/cv_forensics/pre_sns_visualization.py`
- `src/cv_forensics/pre_sns_evaluation.py`
- `src/cv_forensics/pre_sns_manifest.py`
- `src/cv_forensics/outputs.py`
- `src/cv_forensics/__init__.py`
- `tests/test_pre_sns_v3_training.py`
- `tests/test_pre_sns_visual_artifacts.py`
- `tests/test_pre_sns_evaluation.py`
- `configs/inference/pre_sns_report.example.json`
- `configs/evaluation/pre_sns_evaluation.example.json`
- `docs/pre_sns_v3_training.md`
- `docs/pre_sns_visual_artifacts.md`
- `docs/pre_sns_evaluation.md`

## Files Codex May Modify

- `tasks/0036-pre-sns-v3-dual-scale-calibrated-report-and-mining.md`
- `scripts/inference/run_pre_sns_v3_dual_scale_report.py`
- `scripts/evaluation/mine_pre_sns_v3_hard_cases.py`
- `scripts/agent/validate_pre_sns_v3_dual_report_config.py`
- `scripts/agent/validate_pre_sns_v3_hard_mining_config.py`
- `src/cv_forensics/pre_sns_v3_dual_scale_report.py`
- `src/cv_forensics/pre_sns_v3_hard_mining.py`
- `tests/test_pre_sns_v3_dual_scale_report_and_mining.py`
- `docs/pre_sns_v3_dual_scale_report_and_mining.md`
- `src/cv_forensics/__init__.py`
- `configs/inference/pre_sns_v3_dual_report.example.json`
- `configs/evaluation/pre_sns_v3_hard_mining.example.json`

## Forbidden Actions

- Do not add SNS augmentation.
- Do not execute SNS augmentation.
- Do not train a model.
- Do not download datasets.
- Do not use network resources from shell commands.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not write outputs or checkpoints into the repository.
- Do not modify training data.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not use `rg`.
- Do not create large files.
- Do not commit unless the user explicitly says `commit this`.

## Important Project Facts

- The project is a lightweight multi-head image forensics prototype for social-media-robust pre-SNS and later SNS experiments.
- Target outputs are `Class / Mask / Family / Reason`.
- v3 checkpoints produce class logits, tamper binary logits, family logits, and localization logits.
- Observed results show long256 improves classification and tampered detection over long224.
- Localization quality is mixed: long256 is often better, but some samples are better localized by long224.
- Some long256 false-positive real samples are classified as tampered while mask evidence is empty.
- Some true tampered samples are correctly classified by long256 but have low-IoU masks.
- This task must improve report calibration and mining only; it must not start another training run.

## Implementation Requirements

### 1. Dual-Scale Report Runner

Create `scripts/inference/run_pre_sns_v3_dual_scale_report.py`.

The script must:

- Load a JSON config.
- Use two v3 checkpoints, one long224 and one long256.
- Run both models on the same input image.
- Use long256 primarily for class decision.
- Compute `class_conf`, `tampered_score`, and `family_conf` for each model.
- Emit both model outputs in the final report JSON.
- Use only approved input roots and approved output roots outside the repository.
- Avoid training, checkpoint writing, downloads, network access, and SNS augmentation.

### 2. Dual-Scale Report Library

Create `src/cv_forensics/pre_sns_v3_dual_scale_report.py`.

It must provide reusable, testable functions for:

- Loading and validating dual-scale report configs.
- Running long224 and long256 inference through existing v3 model/report utilities where practical.
- Computing mask statistics:
  - `mask_area_pct`
  - component count
  - largest component area
  - largest component area percentage
- Post-processing masks using PIL/numpy-only logic:
  - remove tiny components
  - optional keep top-k components
  - morphological open/close if implemented without new dependencies
- Selecting the final mask according to these rules:
  - if both masks exist and overlap/agreement is reasonable, use a weighted, union, or agreement mask
  - if only long256 mask exists and long256 class is tampered, use long256 mask
  - if long224 has a valid mask but long256 mask is empty, use long224 only if long256 class is tampered or uncertain
  - if class is tampered but both masks are empty, set `localized_evidence_status` to `not_found`
  - if class is real or non-tampered but masks are active, suppress the mask unless `tampered_score` is very high and mark the case uncertain

The final report must include:

- `marker`
- `image_path`
- `long224`
- `long256`
- `class_mask_consistency`
- `localized_evidence_status`
- `final_decision`
- `final_decision_confidence`
- `explanation_reason`
- `final_mask_area_pct`
- `final_mask_stats`
- `visual_artifacts`
- `no_download`
- `no_network`
- `no_training`
- `no_checkpoint_writes`
- `no_sns_augmentation`

Use a marker such as `PRE_SNS_V3_DUAL_SCALE_REPORT_OK`.

### 3. Class-Mask Consistency Gate

Implement explicit consistency logic:

- If long256 class is tampered but the selected final mask is empty, `final_decision` must be similar to `tampered_suspect_no_localized_evidence`, not a confident localized tampered report.
- If the class is real or full synthetic but masks appear, suppress the visual overlay unless the tampered score is very high, and report uncertainty.
- Include `class_mask_consistency`, `localized_evidence_status`, `final_decision`, `final_decision_confidence`, and `explanation_reason` in the JSON report.
- Explanations must be template-based, concise, and derived only from model scores and mask statistics.

### 4. Visual Artifacts

When writing reports, create visual artifacts only under the approved external output root:

- `final_clean_red_overlay.png`
- `final_mask.png`
- `long224_clean_red_overlay.png` if applicable
- `long256_clean_red_overlay.png` if applicable
- `dual_scale_comparison_sheet.jpg`

The clean red overlay files must not contain drawn text. The comparison sheet may contain labels.

### 5. Hard Negative / Hard Positive Mining

Create `scripts/evaluation/mine_pre_sns_v3_hard_cases.py` and `src/cv_forensics/pre_sns_v3_hard_mining.py`.

The miner must:

- Load a JSON config.
- Run on a manifest and one or both checkpoints.
- Support `max_samples`.
- Not modify training data.
- Write only under an approved output root outside the repository.
- Produce:
  - `hard_negative_real.json`
  - `hard_negative_non_tampered.json`
  - `hard_positive_tampered_low_iou.json`
  - `class_mask_inconsistent_cases.json`
  - `mining_summary.json`
- Include summary counts, thresholds, config guardrail flags, and output paths in `mining_summary.json`.
- Support operation without ground-truth masks by still mining class/mask inconsistency and hard negatives.

### 6. Config Validators

Create:

- `scripts/agent/validate_pre_sns_v3_dual_report_config.py`
- `scripts/agent/validate_pre_sns_v3_hard_mining_config.py`

Add example configs:

- `configs/inference/pre_sns_v3_dual_report.example.json`
- `configs/evaluation/pre_sns_v3_hard_mining.example.json`

Approved configs must require:

- `no_download: true`
- `no_network: true`
- `no_sns_augmentation: true`
- approved input roots outside the repository
- output roots outside the repository
- checkpoint paths under approved input roots or explicit approved checkpoint roots outside the repository
- no protected repo paths

The validators must reject missing guardrails, repo-local outputs, protected paths, and SNS augmentation.

### 7. Tests

Create `tests/test_pre_sns_v3_dual_scale_report_and_mining.py`.

The test file must be plain `python3` runnable and must not require pytest as the test runner.

Cover:

- config validation
- mask component cleanup
- class-mask consistency logic
- dual-scale mask selection rules
- hard mining summary schema
- no repository writes

Use temporary directories outside the repository for write tests. Keep fixtures tiny.

### 8. Docs

Create `docs/pre_sns_v3_dual_scale_report_and_mining.md`.

The document must:

- Include marker `PRE_SNS_V3_DUAL_SCALE_REPORT_AND_MINING_OK`.
- Explain why this task exists: long256 improves detection, but localization and class-mask consistency need calibration.
- Document report JSON fields.
- Document mask selection and consistency gate behavior.
- Document hard-case mining outputs.
- State that this remains pre-SNS only and does not add SNS augmentation.

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_v3_dual_report_config.py configs/inference/pre_sns_v3_dual_report.example.json
```

```bash
python3 scripts/agent/validate_pre_sns_v3_hard_mining_config.py configs/evaluation/pre_sns_v3_hard_mining.example.json
```

```bash
python3 tests/test_pre_sns_v3_dual_scale_report_and_mining.py
```

```bash
grep -q PRE_SNS_V3_DUAL_SCALE_REPORT_AND_MINING_OK docs/pre_sns_v3_dual_scale_report_and_mining.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0036-pre-sns-v3-dual-scale-calibrated-report-and-mining.md
```

## Acceptance Criteria

- The dual-scale report runner produces a JSON report containing both long224 and long256 outputs.
- The long256 output is the primary class decision.
- Final mask selection follows the specified dual-scale and consistency rules.
- Empty-mask tampered decisions are reported as suspect without localized evidence.
- Non-tampered decisions with active masks suppress visual overlays unless the tampered score is very high and the report marks uncertainty.
- Visual artifacts are written only under approved external output roots.
- Clean red overlay images contain no drawn text.
- Hard mining writes only the five required JSON outputs under the approved external output root.
- Config validators reject unsafe configs and accept the example configs.
- The plain Python test passes.
- The docs marker check passes.
- `check_agent_changes.py` passes for this task file.
- No protected paths, data, datasets, outputs, checkpoints, secrets, network, package installation, training, or SNS augmentation are touched.

## Stop Condition

After creating this task file, stop and report:

- task file path
- short summary
- git status
- exact `git add` command for the task file
- exact `git commit` command for the task file

Do not implement until the task file is committed and the user explicitly says `implement`.
