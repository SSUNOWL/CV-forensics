# Task 0031: Pre-SNS Visual Mask Artifacts

## Task title
Add visual localization artifacts to the pre-SNS single-image report pipeline.

## Role
Codex is the implementation agent in Codex-only mode. Do not delegate to Claude. Do not use the Claude wrapper.

Do not implement this task while creating this task file. Stop after creating this task file.

## Context
Task 0030-A produced a valid single-image pre-SNS report JSON from the latest scaled checkpoint. The report contains `class`, `family`, `tampered_score`, `localization_head`, `mask_area_pct`, `reason`, `latency_ms`, `fps_estimate`, and `PRE_SNS_SINGLE_IMAGE_REPORT_OK`. However, the report root contains only `pre_sns_single_image_report.json` and no visual artifacts. The local check printed `MASK_OVERLAY_ARTIFACT_NOT_PRESENT_YET`.

## Goal
Add visual localization artifacts to the pre-SNS single-image report pipeline.

## Purpose
When `localization_head` is activated, the single-image report should save image artifacts showing the suspected tampered region:

- predicted mask PNG
- heatmap PNG
- original image with mask overlay PNG
- visualization manifest JSON

These artifacts should be linked from the report JSON.

## Project alignment
The project target output includes Class / Mask / Family / Reason together. This task makes the Mask part visible for approved pre-SNS single-image reports before any SNS augmentation work begins.

## Files Codex May Read
- AGENTS.md
- CLAUDE.md
- docs/project_brief.md
- docs/project_contract.md
- configs/project_contract.json
- docs/agent_workflow.md
- tasks/0026-pre-sns-single-image-report.md
- tasks/0031-pre-sns-visual-mask-artifacts.md
- src/cv_forensics/pre_sns_inference_report.py
- src/cv_forensics/pre_sns_integrated_model.py
- src/cv_forensics/pre_sns_training_artifacts.py
- scripts/inference/run_pre_sns_report.py
- scripts/agent/validate_pre_sns_inference_report.py
- configs/inference/pre_sns_report.example.json
- tests/test_pre_sns_inference_report.py
- docs/pre_sns_inference_report.md

## Files Codex May Modify
- tasks/0031-pre-sns-visual-mask-artifacts.md
- src/cv_forensics/pre_sns_visualization.py
- src/cv_forensics/pre_sns_inference_report.py
- src/cv_forensics/__init__.py
- scripts/inference/run_pre_sns_report.py
- scripts/agent/validate_pre_sns_inference_report.py
- configs/inference/pre_sns_report.example.json
- tests/test_pre_sns_visual_artifacts.py
- tests/test_pre_sns_inference_report.py
- docs/pre_sns_inference_report.md
- docs/pre_sns_visual_artifacts.md

If a parent directory for an allowed file does not exist, Codex may create it. Codex must not create or modify any other files.

## Forbidden actions
Codex must not:

- download datasets
- train models
- install packages
- access network resources
- access `.env`, `.env.*`, secrets, credentials, data, datasets, outputs, or repo checkpoints directories
- write generated images under repo `outputs/` or repo `checkpoints/`
- write checkpoints
- use SNS augmentation
- run `git push`, `git pull`, or `git fetch`
- use `rm -rf`
- create large files
- use `rg`
- modify unrelated files

## Implementation requirements

### 1. Visual artifact writer module
Create `src/cv_forensics/pre_sns_visualization.py`.

It must provide standard-library/PIL-based helpers to write:

- `predicted_mask.png`
- `heatmap.png`
- `overlay.png`
- `visual_artifacts_manifest.json`

The writer must:

- take an input image and a localization probability map or mask
- resize mask/heatmap to the original image size
- normalize masks safely
- create a binary or probability mask image
- create a simple heatmap image
- create an overlay image by blending a highlighted mask over the original image
- write only inside an approved report root
- return artifact paths as strings
- refuse path traversal or writes outside report root
- not require matplotlib, cv2, numpy, pandas, sklearn, or any new dependency

### 2. Integrate visual artifacts into single-image report
Update `src/cv_forensics/pre_sns_inference_report.py` and/or `scripts/inference/run_pre_sns_report.py` so that approved report configs can request visual artifacts.

Add config option:

- `write_visual_artifacts`: `true` or `false`

If `write_visual_artifacts=true` and `localization_head` is activated:

- save `predicted_mask.png`
- save `heatmap.png`
- save `overlay.png`
- save `visual_artifacts_manifest.json`
- include these paths in the final report JSON:
  - `predicted_mask_path`
  - `heatmap_path`
  - `overlay_path`
  - `visual_artifacts_manifest_path`
  - `visual_artifacts_written: true`

If `localization_head` is not activated:

- do not invent a suspicious mask
- set `visual_artifacts_written: false`
- set `localization_visualization_status` to `not_applicable` or `skipped_below_threshold`

### 3. Use actual model localization output
Prefer the actual localization probability map/logits produced by the current pre-SNS model or inference path. If the current code only computes `mask_area_pct` internally, expose the same underlying map used for that computation.

Do not use ground-truth mask for prediction visualization. Do not generate fake random masks for actual reports. Toy masks are allowed only in tests.

### 4. Report JSON behavior
The final report JSON must preserve existing fields:

- `marker`
- `class`
- `class_conf`
- `family`
- `family_conf`
- `tampered_score`
- `localization_head`
- `mask_area_pct`
- `reason`
- `latency_ms`
- `fps_estimate`
- `device`
- `checkpoint_path`
- `image_path`
- `report_path`
- `no_sns_augmentation`

It must add visual artifact fields when requested:

- `visual_artifacts_written`
- `predicted_mask_path`
- `heatmap_path`
- `overlay_path`
- `visual_artifacts_manifest_path`

### 5. Validator update
Update `scripts/agent/validate_pre_sns_inference_report.py` so it can validate report JSON output with visual artifacts.

It should:

- validate the safe example config
- validate approved local report config
- when a report output is provided and `write_visual_artifacts=true`, require visual artifact paths if `localization_head` is activated
- require artifacts to be under `report_root`
- require files to exist
- require PNG extension for mask, heatmap, and overlay
- reject artifacts outside `report_root`
- reject URLs and secret/protected path references

### 6. Example config
Update `configs/inference/pre_sns_report.example.json`.

It must remain symbolic and safe. It may include `write_visual_artifacts: false` or a symbolic `visual_artifact_policy`. Do not include real local absolute paths in the tracked example.

### 7. Tests
Add `tests/test_pre_sns_visual_artifacts.py`.

Tests must run with:

```bash
python3 tests/test_pre_sns_visual_artifacts.py
```

Tests should cover:

- toy RGB image plus toy mask writes `predicted_mask.png`, `heatmap.png`, `overlay.png`, `visual_artifacts_manifest.json`
- overlay image is created and is non-empty
- artifact paths are under the report root
- path traversal is rejected
- writes outside report root are rejected
- zero mask is handled safely
- mask normalization is deterministic
- visual artifact manifest contains expected keys

Update `tests/test_pre_sns_inference_report.py` if needed to check report JSON visual fields.

### 8. Documentation
Update `docs/pre_sns_inference_report.md` and add `docs/pre_sns_visual_artifacts.md`.

Both documentation paths should explain that this is pre-SNS visualization, not SNS augmentation.

Include marker:

```text
PRE_SNS_VISUAL_ARTIFACTS_OK
```

## Validation commands

```bash
python3 scripts/agent/check_agent_changes.py tasks/0031-pre-sns-visual-mask-artifacts.md
```

```bash
python3 scripts/agent/validate_pre_sns_inference_report.py configs/inference/pre_sns_report.example.json
```

```bash
python3 tests/test_pre_sns_visual_artifacts.py
```

```bash
python3 tests/test_pre_sns_inference_report.py
```

```bash
grep -q PRE_SNS_VISUAL_ARTIFACTS_OK docs/pre_sns_visual_artifacts.md
```

```bash
git status --short --untracked-files=all
```

## Acceptance criteria
- Visual artifact writer exists and uses only standard library plus PIL.
- Approved single-image reports can request visual artifacts.
- Activated localization writes mask, heatmap, overlay, and manifest artifacts under the approved report root.
- Report JSON links visual artifacts when written.
- Non-activated localization does not invent visual artifacts and reports a skipped/not-applicable visualization status.
- Validator accepts valid visual artifact reports and rejects malformed or unsafe artifact paths.
- Tests pass.
- Docs marker exists.
- No dataset download, network access, training, checkpoint writing, SNS augmentation, protected path access, or unrelated modification occurs.

## Stop condition
After creating `tasks/0031-pre-sns-visual-mask-artifacts.md`, stop and show:

- task file path
- summary
- git status
- git add command
- git commit command
