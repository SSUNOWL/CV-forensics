# Task 0007: Model Output Schema and Template Explanation Schema

## Role of Claude Code

Claude Code is the implementation worker. Codex is the supervisor and reviewer.

Implement this task exactly as written. Do not reinterpret the project scope beyond this task file. Stop after implementation and validation, then report the requested results.

## Task Objective

Implement a pure-Python model output schema and deterministic template explanation schema for the lightweight multi-head image forensics prototype.

This task defines the structure of the final forensic report. It must not implement real model inference, train a model, load images, download datasets, install packages, or touch protected paths.

This schema must support future steps:

- 0008 lightweight inference stub with fake inputs
- 0009 metric calculators with toy arrays
- 0010 training loop skeleton, dry-run only
- later real training and evaluation
- later social-media perturbation robustness evaluation

## Files Claude May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `docs/config_schema.md`
- `configs/datasets.example.json`
- `configs/experiments/smoke_baseline.json`
- `docs/dataset_manifest.md`
- `configs/manifests/community_forensics_small.example.json`
- `configs/manifests/sid_set.example.json`
- `configs/manifests/combined_smoke_manifest.example.json`
- `tasks/0007-model-output-schema.md`
- `docs/repo_structure.md`, only if it exists
- `src/cv_forensics/__init__.py`, only if it exists
- `src/cv_forensics/contracts.py`, only if it exists
- `src/cv_forensics/outputs.py`, only if it exists
- `src/cv_forensics/evidence.py`, only if it exists
- `src/cv_forensics/config_schema.py`, only if it exists
- `src/cv_forensics/dataset_manifest.py`, only if it exists
- any file Claude is allowed to create or modify in this task, if it already exists

## Files Claude May Modify

- `src/cv_forensics/__init__.py`
- `src/cv_forensics/outputs.py`
- `src/cv_forensics/evidence.py`
- `src/cv_forensics/model_output_schema.py`
- `src/cv_forensics/explanation_templates.py`
- `configs/model_output_schema.example.json`
- `configs/explanation_templates.example.json`
- `scripts/agent/validate_model_output_schema.py`
- `tests/test_model_output_schema.py`
- `docs/model_output_schema.md`

If a parent directory for an allowed file does not exist, Claude may create that parent directory. Claude must not create any other files.

## Forbidden Actions

Claude must not:

- download datasets
- train a model
- install packages
- access network resources from shell commands
- access `.env`, `.env.*`, `secrets`, `data`, `datasets`, `outputs`, or `checkpoints`
- inspect protected directories or protected files recursively
- read real image files
- read real mask files
- validate actual image or mask file existence
- walk real dataset directories
- modify unrelated files
- run `git push`
- run `git pull`
- run `git fetch`
- run `curl`, `wget`, `ssh`, `scp`, `rsync`, `pip`, `npm`, `yarn`, `pnpm`, `kaggle`, `huggingface-cli`, or `wandb`
- create large files
- use `rg`
- use `rm -rf`
- use `--permission-mode bypassPermissions`
- use `--dangerously-skip-permissions`
- import `torch`, `torchvision`, `PIL`, `cv2`, `numpy`, `pandas`, `sklearn`, or any third-party package

## Important Project Facts to Preserve

- The final prototype should output class, mask, family, and reason together.
- The class output is `real / synthetic / tampered`.
- The mask output is a tampered mask or suspicious-region summary, not necessarily a real image file at this stage.
- The family output is generator family or family-level provenance.
- The reason output is a short evidence-template explanation.
- The model architecture is a lightweight multi-head structure:
  - shared visual backbone
  - 3-way classification head
  - generator-family provenance head
  - conditional localization head
  - evidence aggregation module
  - template-based explanation
- Generator-family provenance is coarse provenance, not exact model attribution.
- Initial family label candidates are:
  - `LatDiff`
  - `PixDiff`
  - `GAN`
  - `Other`
  - `Real-or-N/A`
- SID-Set may not have family labels compatible with Community Forensics-Small.
- SID-Set family/provenance handling should remain compatible with freeze, mask-out, or mixed Community Forensics-Small batch policies.
- In inference, 3-way classification and provenance head run first.
- Conditional localization should activate only when tampered score crosses a threshold tau.
- A high tau can cause localization false negatives, so output schema must preserve enough fields to inspect threshold behavior and localization activation recall later.
- Social-media perturbation context should be representable for future robustness analysis:
  - `jpeg`
  - `resize`
  - `crop`
  - `rotation`
  - `padding`
  - `shear`
  - `screenshot`
  - `text_overlay`
  - `sticker_overlay`
  - `recompression_chain`
- This task defines schemas and deterministic helpers only. It must not perform real inference or training.

## Implementation Requirements

### 1. Create or update `src/cv_forensics/model_output_schema.py`

- Use only Python standard library.
- Do not import third-party packages.
- Do not read images, masks, datasets, outputs, checkpoints, `.env`, or secrets.
- Define constants for:
  - class labels:
    - `real`
    - `synthetic`
    - `tampered`
  - family labels:
    - `LatDiff`
    - `PixDiff`
    - `GAN`
    - `Other`
    - `Real-or-N/A`
  - localization states:
    - `not_applicable`
    - `skipped_below_threshold`
    - `activated`
    - `unavailable`
  - evidence signal ids:
    - `global_synthetic_artifact`
    - `boundary_discontinuity`
    - `texture_inconsistency`
    - `local_mask_activation`
    - `provenance_family_signal`
    - `low_confidence`
    - `social_media_recompression`
    - `screenshot_padding`
    - `text_overlay_occlusion`
    - `sticker_overlay_occlusion`
  - perturbation tags:
    - `none`
    - `jpeg`
    - `resize`
    - `crop`
    - `rotation`
    - `padding`
    - `shear`
    - `screenshot`
    - `text_overlay`
    - `sticker_overlay`
    - `recompression_chain`
- Define pure-Python dataclasses or typed dictionaries for:
  - confidence distributions
  - localization summary
  - evidence signal
  - final forensic output
- The final forensic output must preserve at least these target fields:
  - `schema_version`
  - `class`
  - `class_conf`
  - `family`
  - `family_conf`
  - `localization_head`
  - `mask_area_pct`
  - `threshold_tau`
  - `tampered_score`
  - `evidence`
  - `perturbations`
  - `reason`
- The schema must also allow future optional fields without requiring real data:
  - `mask_ref`
  - `visualization_ref`
  - `model_stage`
  - `family_policy`
  - `latency_ms`
  - `fps`
- Implement validation helpers:
  - `validate_class_label`
  - `validate_family_label`
  - `validate_confidence_map`
  - `validate_localization_summary`
  - `validate_evidence_signals`
  - `validate_forensic_output`
  - `should_activate_localization`
  - `build_minimal_output` or an equivalent helper for fake/dry-run output creation
- Confidence validation should:
  - ensure required class labels are present for `class_conf`
  - ensure family labels are known for `family_conf`
  - ensure values are numeric and between 0 and 1
  - allow a small tolerance for sums near 1.0
- Localization validation should:
  - allow `activated` only when `tampered_score` is greater than or equal to `threshold_tau`
  - allow `skipped_below_threshold` when `tampered_score` is below `threshold_tau`
  - reject negative `mask_area_pct`
  - reject `mask_area_pct` greater than 100
- Protected path detection should reject any optional reference fields containing:
  - `.env`
  - `.env.*`
  - `secrets`
  - `data`
  - `datasets`
  - `outputs`
  - `checkpoints`
  - absolute local paths such as `/home/`, `/mnt/`, `/root/`, `/Users/`
  - Windows drive paths
  - URLs such as `http://`, `https://`, `s3://`, `gs://`, `hf://`
- Placeholder strings such as `<MASK_NOT_MATERIALIZED>` or `<VISUALIZATION_NOT_MATERIALIZED>` may be allowed.
- Keep all helpers deterministic.

### 2. Create or update `src/cv_forensics/explanation_templates.py`

- Use only Python standard library.
- Do not import third-party packages.
- Do not call external APIs.
- Do not use an LLM.
- Implement deterministic template-based explanation generation.
- The explanation generator should consume the validated forensic output dictionary or dataclass.
- It should generate short Korean reason strings using:
  - class label
  - class confidence
  - family label
  - family confidence
  - localization state
  - mask area percentage
  - evidence signal ids
  - perturbation tags
- It should include templates for:
  - real
  - synthetic
  - tampered with localization activated
  - tampered but localization skipped below threshold
  - low-confidence result
  - family unavailable or `Real-or-N/A`
  - social-media perturbation context
- It should avoid claiming certainty.
- It should phrase family as estimated coarse provenance, not exact model attribution.
- It should mention that localization is conditional when relevant.
- It should provide a function such as `generate_reason(output)` or `build_reason(output)`.
- It should provide at least one function that returns a structured evidence summary for later reporting.

### 3. Create `configs/model_output_schema.example.json`

- It must be an example only.
- It must not contain real local paths.
- It must not contain protected paths.
- It must not contain URLs.
- It must not contain secrets.
- It should contain a tampered example similar to the proposal target output:
  - `class`: `tampered`
  - `class_conf` with `real`, `synthetic`, `tampered`
  - `family`: `LatDiff`
  - `family_conf` with `LatDiff`, `PixDiff`, `GAN`, `Other`, `Real-or-N/A`
  - `localization_head`: `activated`
  - `mask_area_pct` around `11.2`
  - `threshold_tau`
  - `tampered_score`
  - evidence signal list
  - perturbations list
  - reason
- It should also include a small `examples` array or notes for:
  - synthetic without localization
  - real with `Real-or-N/A` family
  - tampered below threshold where localization is skipped
- It must clearly mark examples as dry-run schema examples, not model predictions.

### 4. Create `configs/explanation_templates.example.json`

- It must be an example only.
- It must not contain real local paths.
- It must not contain protected paths.
- It must not contain URLs.
- It must not contain secrets.
- It should define template ids and short Korean template strings for:
  - `real`
  - `synthetic`
  - `tampered_localized`
  - `tampered_not_localized`
  - `low_confidence`
  - `family_estimated`
  - `family_unavailable`
  - `social_media_perturbation`
- It should document required placeholders such as:
  - `class_label`
  - `class_confidence`
  - `family_label`
  - `family_confidence`
  - `mask_area_pct`
  - `localization_head`
  - `threshold_tau`
  - `perturbation_tags`
- It must not require an LLM.

### 5. Create `scripts/agent/validate_model_output_schema.py`

- Use only Python standard library.
- Accept one or two positional JSON paths:
  - model output schema example JSON path
  - optional explanation templates example JSON path
- Validate the example JSON using `src/cv_forensics/model_output_schema.py` and `src/cv_forensics/explanation_templates.py`.
- Be import-safe without package installation.
- It may add the repository root or `src` directory to `sys.path`.
- Print a concise success message on pass.
- Print actionable errors and exit non-zero on fail.
- Reject protected paths, URLs, absolute local machine paths, secret-looking keys, and secret-looking values.
- Validate that the generated reason is deterministic and non-empty.
- Validate that tampered examples with `tampered_score >= threshold_tau` have `localization_head` set to `activated`.
- Validate that tampered examples with `tampered_score < threshold_tau` can have `localization_head` set to `skipped_below_threshold`.
- Validate that class/family labels and confidence maps match the project contract.

### 6. Create `docs/model_output_schema.md`

- Explain the purpose of the model output schema.
- Explain that this is a schema and dry-run helper task, not real inference or training.
- Explain how the schema maps to the proposal target outputs:
  - Class
  - Mask
  - Family
  - Reason
- Explain the conditional localization rule:
  - run classification/provenance first
  - activate localization only if tampered score crosses threshold tau
  - preserve tau and tampered_score to analyze false negatives later
- Explain coarse provenance:
  - family is estimated family-level provenance
  - not exact model attribution
- Explain template-based explanation:
  - no free-form LLM generation
  - deterministic templates
  - evidence aggregation from class, family, mask/localization, and perturbation context
- Explain social-media perturbation tags.
- Explain validation commands.
- Include marker string:
  - `MODEL_OUTPUT_SCHEMA_OK`
- Keep it concise but useful.

### 7. Create `tests/test_model_output_schema.py`

- Use pytest-style assertions.
- Do not install pytest.
- Use only Python standard library plus pytest-style tests.
- Keep tests lightweight.
- Do not access external resources.
- Do not access protected paths.
- Tests should cover:
  - valid tampered example passes
  - class confidence required labels are present
  - unknown class labels are rejected
  - unknown family labels are rejected
  - confidence values outside 0..1 are rejected
  - localization activated only when `tampered_score >= threshold_tau`
  - `skipped_below_threshold` allowed when `tampered_score < threshold_tau`
  - `mask_area_pct` outside 0..100 is rejected
  - protected paths in `mask_ref` or `visualization_ref` are rejected
  - URL references are rejected
  - generated reason is deterministic
  - generated reason mentions uncertainty or estimation for family/provenance
  - social-media perturbation tags are preserved

### 8. Update `src/cv_forensics/outputs.py` and `src/cv_forensics/evidence.py` only if useful

- If these files exist, Claude may add lightweight compatibility wrappers or imports.
- Do not duplicate heavy logic unnecessarily.
- Do not add third-party imports.
- Do not break existing validators or tests.

### 9. Update `src/cv_forensics/__init__.py` only if needed

- If the file exists, Claude may add lightweight exports for `model_output_schema` and `explanation_templates`.
- If the file does not exist, Claude may create a minimal package marker.
- Do not add heavy imports.
- Do not import third-party packages.

## Validation Commands

```bash
python3 scripts/agent/validate_model_output_schema.py configs/model_output_schema.example.json configs/explanation_templates.example.json
```

```bash
test -f src/cv_forensics/model_output_schema.py
```

```bash
test -f src/cv_forensics/explanation_templates.py
```

```bash
test -f configs/model_output_schema.example.json
```

```bash
test -f configs/explanation_templates.example.json
```

```bash
test -f scripts/agent/validate_model_output_schema.py
```

```bash
test -f docs/model_output_schema.md
```

```bash
grep -q MODEL_OUTPUT_SCHEMA_OK docs/model_output_schema.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- src/cv_forensics/__init__.py src/cv_forensics/outputs.py src/cv_forensics/evidence.py src/cv_forensics/model_output_schema.py src/cv_forensics/explanation_templates.py configs/model_output_schema.example.json configs/explanation_templates.example.json scripts/agent/validate_model_output_schema.py tests/test_model_output_schema.py docs/model_output_schema.md
```

Optional validation command:

```bash
pytest -q tests/test_model_output_schema.py
```

## Acceptance Criteria

- `python3 scripts/agent/validate_model_output_schema.py configs/model_output_schema.example.json configs/explanation_templates.example.json` passes.
- If pytest is available, `pytest -q tests/test_model_output_schema.py` passes.
- Changed files are limited to:
  - `src/cv_forensics/__init__.py`
  - `src/cv_forensics/outputs.py`
  - `src/cv_forensics/evidence.py`
  - `src/cv_forensics/model_output_schema.py`
  - `src/cv_forensics/explanation_templates.py`
  - `configs/model_output_schema.example.json`
  - `configs/explanation_templates.example.json`
  - `scripts/agent/validate_model_output_schema.py`
  - `tests/test_model_output_schema.py`
  - `docs/model_output_schema.md`
- The schema preserves the final output contract:
  - `class`
  - `class_conf`
  - `family`
  - `family_conf`
  - `localization_head`
  - `mask_area_pct`
  - `reason`
- The schema supports conditional localization based on `tampered_score` and `threshold_tau`.
- The schema supports coarse family-level provenance, not exact model attribution.
- The explanation helper is deterministic and template-based.
- The examples contain no real local paths, no protected paths, no secrets, no network URLs, and no dataset references.
- No `.env`, secrets, data, datasets, outputs, or checkpoints are touched.
- No dataset download, model training, package installation, network access, large file creation, real image reading, or unrelated modification occurs.

## Stop Condition

Stop after implementation and validation. Report:

- files changed
- validation commands run and their results
- any skipped optional validation and why
- confirmation that forbidden paths and actions were not touched
- confirmation that this task did not perform real inference, training, dataset access, or image reading
