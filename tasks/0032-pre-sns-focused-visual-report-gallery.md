# Task 0032: Pre-SNS Clean Red Mask Overlay

## Role of Codex

Codex is the implementation agent for this task.
Do not delegate to Claude.
Do not run the Claude wrapper.
Do not use `scripts/agent/run_claude_task.sh`.

## Context

Task 0031 added visual artifacts for pre-SNS single-image inference:
- `predicted_mask.png`
- `heatmap.png`
- `overlay.png`
- `visual_artifacts_manifest.json`

The latest long visual report successfully writes visual artifacts and produces:
- class prediction
- family prediction
- tampered_score
- localization_head
- mask_area_pct
- reason
- latency_ms
- fps_estimate
- report JSON

However, the current visual result is not clean enough for presentation. It may include extra text, bounding boxes, titles, panels, or other annotations depending on post-processing. The desired output for task 0032 is simple:

A clean image where only the model-predicted suspicious region is covered by a semi-transparent red mask.

No text.
No bounding box.
No arrows.
No labels.
No legend.
No title.
No demo panel by default.

This task must use the model-predicted localization mask or probability map. It must not use ground-truth masks for prediction visualization.

## Goal

Add a clean red mask overlay mode to pre-SNS single-image inference.

The clean overlay should:
- use the predicted localization mask or probability map
- optionally focus on the strongest suspicious region
- write a clean binary focused mask PNG
- write a clean red overlay PNG
- write a clean overlay manifest JSON
- add clean overlay paths to the final report JSON

## Files Codex May Read

- AGENTS.md
- CLAUDE.md
- docs/project_brief.md
- docs/project_contract.md
- configs/project_contract.json
- docs/agent_workflow.md
- tasks/0031-pre-sns-visual-mask-artifacts.md
- tasks/0032-pre-sns-focused-visual-report-gallery.md
- src/cv_forensics/pre_sns_visualization.py
- src/cv_forensics/pre_sns_inference_report.py
- src/cv_forensics/pre_sns_integrated_model.py
- scripts/inference/run_pre_sns_report.py
- scripts/agent/validate_pre_sns_inference_report.py
- configs/inference/pre_sns_report.example.json
- tests/test_pre_sns_visual_artifacts.py
- tests/test_pre_sns_inference_report.py
- docs/pre_sns_visual_artifacts.md
- docs/pre_sns_inference_report.md

## Files Codex May Modify

- tasks/0032-pre-sns-focused-visual-report-gallery.md
- src/cv_forensics/pre_sns_visualization.py
- src/cv_forensics/pre_sns_inference_report.py
- src/cv_forensics/__init__.py
- scripts/inference/run_pre_sns_report.py
- scripts/agent/validate_pre_sns_inference_report.py
- configs/inference/pre_sns_report.example.json
- tests/test_pre_sns_visual_artifacts.py
- tests/test_pre_sns_clean_red_overlay.py
- tests/test_pre_sns_inference_report.py
- docs/pre_sns_visual_artifacts.md
- docs/pre_sns_clean_red_overlay.md
- docs/pre_sns_inference_report.md

Do not create or modify any other file.

## Forbidden Actions

Codex must not:
- download datasets
- train a model
- install packages
- access network resources
- access `.env`, `.env.*`, secrets, credentials, data, datasets, repo `outputs`, or repo `checkpoints` directories
- write generated images under repo `outputs/` or repo `checkpoints/`
- write checkpoints
- use SNS augmentation
- run `git push`
- run `git pull`
- run `git fetch`
- use `rm -rf`
- create large files
- use `rg`
- modify unrelated files
- use ground-truth masks to create prediction overlays
- draw text on clean overlay images
- draw bounding boxes on clean overlay images
- draw arrows, legends, titles, captions, watermarks, or panel labels on clean overlay images

## Implementation Requirements

1. Add clean red overlay helper functions

Update `src/cv_forensics/pre_sns_visualization.py`.

Add helper functions that can:
- take an original image
- take a predicted mask or localization probability map
- resize the mask to the original image size
- normalize mask values safely
- optionally keep only the strongest suspicious region
- write `clean_focused_mask.png`
- write `clean_red_overlay.png`
- write `clean_red_overlay_manifest.json`

The clean overlay must:
- blend semi-transparent red only where the focused mask is active
- keep all pixels outside the active mask unchanged
- contain no text
- contain no bounding boxes
- contain no arrows
- contain no legends
- contain no titles
- contain no panel layout
- contain no captions

2. Add config options

Update the inference report config handling to support:

- `write_clean_red_overlay`: `true` or `false`
- `clean_overlay_keep_ratio`
- `clean_overlay_component_mode`
- `clean_overlay_alpha`
- `clean_overlay_threshold_mode`
- `clean_overlay_red_rgb`

Default behavior:
- `write_clean_red_overlay` should default to `false` for tracked example configs
- `clean_overlay_alpha` default should be `0.45`
- `clean_overlay_red_rgb` default should be `[255, 0, 0]`
- `clean_overlay_component_mode` default should be `largest_component`
- `clean_overlay_threshold_mode` default should be `top_percentile`

3. Clean focused mask behavior

The clean focused mask should be derived only from the model-predicted localization output.

Allowed focus behavior:
- threshold by top percentile
- largest connected component
- optional small dilation

Disallowed focus behavior:
- using ground-truth mask
- generating random masks
- drawing text
- drawing boxes
- drawing labels
- drawing arrows
- drawing demo sheets by default

4. Update single-image report output

If `write_clean_red_overlay` is `true` and `localization_head` is activated:
- write `clean_focused_mask.png`
- write `clean_red_overlay.png`
- write `clean_red_overlay_manifest.json`
- include these fields in the report JSON:
  - `clean_red_overlay_written`
  - `clean_focused_mask_path`
  - `clean_red_overlay_path`
  - `clean_red_overlay_manifest_path`
  - `clean_overlay_area_pct`
  - `clean_overlay_keep_ratio`
  - `clean_overlay_alpha`
  - `clean_overlay_component_mode`
  - `clean_overlay_threshold_mode`

If `localization_head` is not activated:
- do not invent a suspicious region
- do not write a clean red overlay
- set `clean_red_overlay_written` to `false`
- set `clean_red_overlay_status` to `skipped_below_threshold` or `not_applicable`

5. Preserve existing report behavior

The final report JSON must preserve existing fields:
- marker
- class
- class_conf
- family
- family_conf
- tampered_score
- localization_head
- mask_area_pct
- reason
- latency_ms
- fps_estimate
- device
- checkpoint_path
- image_path
- report_path
- no_sns_augmentation
- visual_artifacts_written
- predicted_mask_path
- heatmap_path
- overlay_path
- visual_artifacts_manifest_path

6. Validator update

Update `scripts/agent/validate_pre_sns_inference_report.py`.

It should validate:
- safe symbolic example config
- approved local report config
- report JSON with clean red overlay fields
- `clean_focused_mask_path` exists when `clean_red_overlay_written` is true
- `clean_red_overlay_path` exists when `clean_red_overlay_written` is true
- `clean_red_overlay_manifest_path` exists when `clean_red_overlay_written` is true
- clean overlay artifacts are under approved report root
- clean overlay artifacts are PNG files where applicable
- clean overlay artifacts are non-empty
- `clean_overlay_area_pct` is numeric and between 0 and 100
- unsafe paths, URLs, secret-like keys, protected paths, and path traversal are rejected

If possible, the validator should verify:
- `clean_red_overlay.png` does not alter pixels outside `clean_focused_mask.png`

This is the main guard against accidental text, bounding boxes, labels, arrows, or UI elements.

7. Tests

Create `tests/test_pre_sns_clean_red_overlay.py`.

Tests must run with:
- `python3 tests/test_pre_sns_clean_red_overlay.py`

Use standard library and PIL only.
Do not require pytest.
Do not require numpy, pandas, sklearn, cv2, matplotlib, or network access.

Tests must cover:
- clean overlay writes `clean_focused_mask.png`
- clean overlay writes `clean_red_overlay.png`
- clean overlay writes `clean_red_overlay_manifest.json`
- pixels outside the active mask are unchanged
- pixels inside the active mask are red-blended
- no text or bounding-box-like changes appear outside the mask
- top-percentile thresholding is deterministic
- largest component selection is deterministic
- zero mask is handled safely
- invalid alpha is rejected
- path traversal is rejected
- output outside report root is rejected
- manifest contains expected clean overlay keys

Update `tests/test_pre_sns_visual_artifacts.py` and `tests/test_pre_sns_inference_report.py` only if needed.

8. Documentation

Create `docs/pre_sns_clean_red_overlay.md`.

It must explain:
- this task creates a clean presentation-friendly red mask overlay
- the overlay contains only red mask blending and no text or boxes
- the overlay uses model-predicted localization output, not ground-truth masks
- it is for qualitative inspection and presentation
- it is not a new training step
- it is not SNS augmentation
- it does not write to repo `outputs/` or repo `checkpoints/`

Include marker:
`PRE_SNS_CLEAN_RED_OVERLAY_OK`

Update `docs/pre_sns_visual_artifacts.md` and `docs/pre_sns_inference_report.md` only if useful.

## Validation Commands

~~~bash
python3 scripts/agent/check_agent_changes.py tasks/0032-pre-sns-focused-visual-report-gallery.md
~~~

~~~bash
python3 scripts/agent/validate_pre_sns_inference_report.py configs/inference/pre_sns_report.example.json
~~~

~~~bash
python3 tests/test_pre_sns_clean_red_overlay.py
~~~

~~~bash
python3 tests/test_pre_sns_visual_artifacts.py
~~~

~~~bash
python3 tests/test_pre_sns_inference_report.py
~~~

~~~bash
grep -q PRE_SNS_CLEAN_RED_OVERLAY_OK docs/pre_sns_clean_red_overlay.md
~~~

~~~bash
grep -q PRE_SNS_VISUAL_ARTIFACTS_OK docs/pre_sns_visual_artifacts.md
~~~

~~~bash
grep -q PRE_SNS_SINGLE_IMAGE_REPORT_OK docs/pre_sns_inference_report.md
~~~

~~~bash
git status --short --untracked-files=all
~~~

## Acceptance Criteria

- Clean red overlay mode is implemented.
- Clean overlay image contains only red mask blending over the original image.
- Clean overlay image contains no text, bounding boxes, labels, arrows, legends, titles, captions, watermarks, or panel layout.
- Pixels outside the focused mask remain unchanged.
- Clean overlay uses model-predicted localization output only.
- Ground-truth masks are not used for prediction visualization.
- Clean focused mask, clean red overlay, and clean overlay manifest are written when enabled and localization is activated.
- Report JSON includes clean overlay paths and metadata.
- Existing report fields remain compatible.
- All artifacts are written only under approved `report_root`.
- Tracked example config remains symbolic and safe.
- Tests run without third-party dependencies other than PIL.
- No dataset download, model training, package installation, network access, checkpoint writing, repo outputs/checkpoints writing, SNS augmentation, or unrelated modification occurs.
- All validation commands pass.

## Stop Condition

After creating the task file, stop and report:
- task file path
- short summary
- git status
- exact git add command
- exact git commit command
