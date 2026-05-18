# Task 0008: Lightweight Inference Stub with Fake Inputs

## Role of Claude Code

Claude Code is the implementation worker. Codex is the supervisor and reviewer.

Implement a pure-Python lightweight inference stub that uses fake, symbolic inputs to exercise the model output schema and template explanation pipeline created in task 0007.

This task must not implement real model inference, train a model, load images, load masks, download datasets, install packages, or touch protected paths.

The stub should simulate the proposal inference flow:

- fake image/input record
- fake shared-backbone summary
- fake 3-way classification head
- fake generator-family provenance head
- conditional localization decision based on `tampered_score` and `threshold_tau`
- evidence aggregation
- deterministic template-based reason generation
- final forensic output matching the task 0007 schema

This task supports future steps:

- 0009 metric calculators with toy arrays
- 0010 training loop skeleton, dry-run only
- later real inference and evaluation
- later pre-SNS baseline evaluation
- later SNS augmentation robustness evaluation

## Files Claude May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `docs/config_schema.md`
- `docs/dataset_manifest.md`
- `docs/model_output_schema.md`
- `configs/model_output_schema.example.json`
- `configs/explanation_templates.example.json`
- `src/cv_forensics/model_output_schema.py`
- `src/cv_forensics/explanation_templates.py`
- `tasks/0008-lightweight-inference-stub.md`
- `docs/repo_structure.md`, only if it exists
- `docs/planning/pre_sns_implementation_plan.md`, only if it exists
- `src/cv_forensics/__init__.py`, only if it exists
- `src/cv_forensics/contracts.py`, only if it exists
- `src/cv_forensics/outputs.py`, only if it exists
- `src/cv_forensics/evidence.py`, only if it exists
- `src/cv_forensics/config_schema.py`, only if it exists
- `src/cv_forensics/dataset_manifest.py`, only if it exists
- any file Claude is allowed to create or modify in this task, if it already exists

## Files Claude May Modify

- `src/cv_forensics/__init__.py`
- `src/cv_forensics/inference_stub.py`
- `configs/inference/fake_inputs.example.json`
- `scripts/agent/run_fake_inference.py`
- `scripts/agent/validate_inference_stub.py`
- `tests/test_inference_stub.py`
- `docs/inference_stub.md`

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
- write generated predictions to `outputs/`
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

## Important Project Facts

- The final prototype should output class, mask/localization, family, and reason together.
- The class output is `real` / `synthetic` / `tampered`.
- The family output is coarse generator-family provenance, not exact model attribution.
- Initial family label candidates are:
  - `LatDiff`
  - `PixDiff`
  - `GAN`
  - `Other`
  - `Real-or-N/A`
- The architecture direction is:
  - shared visual backbone
  - 3-way classification head
  - generator-family provenance head
  - conditional localization head
  - evidence aggregation
  - deterministic template-based explanation
- In inference, classification and provenance are computed first.
- Conditional localization activates only when `tampered_score >= threshold_tau`.
- The schema must preserve `threshold_tau` and `tampered_score` because high tau may cause localization false negatives.
- Social-media perturbation tags may be represented as metadata, but this task must not implement SNS augmentation.
- This task defines fake inference only. It must not perform real inference, training, dataset access, or image reading.

## Implementation Requirements

### 1. Create `src/cv_forensics/inference_stub.py`

- Use only Python standard library.
- Do not import third-party packages.
- Do not read images, masks, datasets, outputs, checkpoints, `.env`, or secrets.
- Import and reuse task 0007 helpers from:
  - `src/cv_forensics/model_output_schema.py`
  - `src/cv_forensics/explanation_templates.py`
- Define fake input support using dictionaries or dataclasses.
- The fake input schema should support fields such as:
  - `input_id`
  - `scenario`
  - `class_hint`
  - `family_hint`
  - `tampered_score`
  - `threshold_tau`
  - `mask_area_hint_pct`
  - `perturbations`
  - `evidence_hints`
- Supported scenarios should include at least:
  - `real_clean`
  - `synthetic_latdiff`
  - `tampered_localized`
  - `tampered_below_threshold`
  - `low_confidence`
- Implement deterministic fake head functions, for example:
  - `fake_backbone_summary`
  - `fake_classification_head`
  - `fake_family_head`
  - `fake_localization_head`
  - `fake_evidence_aggregation`
  - `run_fake_inference`
  - `run_fake_batch`
- The fake classification head should produce `class_conf` for:
  - `real`
  - `synthetic`
  - `tampered`
- The fake family head should produce `family_conf` for:
  - `LatDiff`
  - `PixDiff`
  - `GAN`
  - `Other`
  - `Real-or-N/A`
- Conditional localization should:
  - set `localization_head` to `activated` when `tampered_score >= threshold_tau` for tampered-like cases
  - set `localization_head` to `skipped_below_threshold` when `tampered_score < threshold_tau`
  - set `localization_head` to `not_applicable` for real/synthetic cases when appropriate
  - preserve `mask_area_pct` without materializing real masks
- Evidence aggregation should map class/family/localization/perturbation context into evidence signal ids compatible with task 0007.
- The final output must validate with `validate_forensic_output` or equivalent task 0007 validator.
- The reason must be generated by the deterministic template explanation helper.
- Keep all behavior deterministic.

### 2. Create `configs/inference/fake_inputs.example.json`

- This must be an example config only.
- It must not contain real local paths.
- It must not contain protected paths.
- It must not contain URLs.
- It must not contain secrets.
- It must not point to images, masks, data folders, outputs, or checkpoints.
- It should contain:
  - `schema_version`
  - `dry_run: true`
  - `no_download: true`
  - `no_training: true`
  - `no_network: true`
  - `description`
  - `default_threshold_tau`
  - `fake_inputs` array
- The `fake_inputs` array should include at least:
  - `real_clean`
  - `synthetic_latdiff`
  - `tampered_localized`
  - `tampered_below_threshold`
  - `low_confidence`
- Include perturbation tags only as metadata, not as an augmentation implementation.
- If a social-media-like tag is used, clearly mark it as a tag for schema compatibility only, not actual augmentation.

### 3. Create `scripts/agent/run_fake_inference.py`

- Use only Python standard library.
- Accept exactly one positional argument:
  - fake input config JSON path
- Load fake inputs.
- Run the fake inference stub.
- Print a concise JSON result summary to stdout.
- Do not write files.
- Do not create `outputs/`.
- Do not read real image or mask files.
- Do not access protected paths.
- Exit non-zero with actionable errors on invalid config or invalid fake output.

### 4. Create `scripts/agent/validate_inference_stub.py`

- Use only Python standard library.
- Accept exactly one positional argument:
  - fake input config JSON path
- Validate the fake input config.
- Run the fake inference stub on all fake inputs.
- Validate every generated final forensic output using task 0007 schema validation.
- Verify that generated reasons are non-empty and deterministic.
- Verify that all required scenarios are present.
- Verify conditional localization behavior:
  - `tampered_localized` activates localization when `tampered_score >= threshold_tau`
  - `tampered_below_threshold` skips localization when `tampered_score < threshold_tau`
  - `real_clean` does not activate tampered localization
- Verify that no generated output contains protected paths, URLs, real image paths, outputs paths, dataset paths, or checkpoint paths.
- Print a concise success message on pass.
- Print actionable errors and exit non-zero on fail.

### 5. Create `docs/inference_stub.md`

- Explain the purpose of the fake inference stub.
- Explain that this is not real model inference and not training.
- Explain how it maps to the proposal architecture:
  - fake backbone summary
  - fake classification head
  - fake family/provenance head
  - conditional localization head
  - evidence aggregation
  - deterministic template explanation
- Explain how it reuses task 0007 model output schema.
- Explain how it prepares for:
  - 0009 toy metric calculators
  - 0010 dry-run training skeleton
  - later real inference
  - later SNS augmentation evaluation
- Explain that SNS augmentation is not implemented here.
- Explain validation commands.
- Include marker string:
  - `INFERENCE_STUB_OK`
- Keep it concise but useful.

### 6. Create `tests/test_inference_stub.py`

- Use only Python standard library.
- Do not import `pytest` or third-party packages.
- Keep pytest-style test functions using plain `assert`.
- Make the file runnable directly with:

```bash
python3 tests/test_inference_stub.py
```

- Keep it pytest-compatible if pytest is later installed.
- Tests should cover:
  - fake input config loads
  - all required scenarios are present
  - `run_fake_inference` returns validated final forensic output
  - `real_clean` produces class `real` and localization `not_applicable` or skipped safely
  - `synthetic_latdiff` produces class `synthetic` and family `LatDiff` or a valid family confidence distribution
  - `tampered_localized` activates localization
  - `tampered_below_threshold` skips localization
  - generated reasons are deterministic and non-empty
  - evidence signal ids are present
  - protected paths in fake input config are rejected
  - URL-like references are rejected
  - no output file is written

### 7. Update `src/cv_forensics/__init__.py` only if needed

- If the file exists, Claude may add a lightweight export for `inference_stub`.
- If the file does not exist, Claude may create a minimal package marker.
- Do not add heavy imports.
- Do not import third-party packages.

## Validation Commands

```bash
python3 scripts/agent/validate_inference_stub.py configs/inference/fake_inputs.example.json
```

```bash
python3 scripts/agent/run_fake_inference.py configs/inference/fake_inputs.example.json
```

```bash
python3 tests/test_inference_stub.py
```

```bash
test -f src/cv_forensics/inference_stub.py
```

```bash
test -f configs/inference/fake_inputs.example.json
```

```bash
test -f scripts/agent/run_fake_inference.py
```

```bash
test -f scripts/agent/validate_inference_stub.py
```

```bash
test -f docs/inference_stub.md
```

```bash
grep -q INFERENCE_STUB_OK docs/inference_stub.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- src/cv_forensics/__init__.py src/cv_forensics/inference_stub.py configs/inference/fake_inputs.example.json scripts/agent/run_fake_inference.py scripts/agent/validate_inference_stub.py tests/test_inference_stub.py docs/inference_stub.md
```

Optional validation command:

```bash
pytest -q tests/test_inference_stub.py
```

## Acceptance Criteria

- `python3 scripts/agent/validate_inference_stub.py configs/inference/fake_inputs.example.json` passes.
- `python3 scripts/agent/run_fake_inference.py configs/inference/fake_inputs.example.json` runs and prints a JSON summary.
- `python3 tests/test_inference_stub.py` passes.
- If pytest is available, `pytest -q tests/test_inference_stub.py` passes.
- Changed files are limited to:
  - `src/cv_forensics/__init__.py`
  - `src/cv_forensics/inference_stub.py`
  - `configs/inference/fake_inputs.example.json`
  - `scripts/agent/run_fake_inference.py`
  - `scripts/agent/validate_inference_stub.py`
  - `tests/test_inference_stub.py`
  - `docs/inference_stub.md`
- The stub reuses task 0007 model output schema and explanation template helpers.
- The final fake inference outputs preserve:
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
- Conditional localization behavior is deterministic and correct for fake inputs.
- The stub does not read real images, real masks, datasets, outputs, checkpoints, `.env`, or secrets.
- The stub does not write prediction outputs to `outputs/`.
- The stub does not download datasets, train models, install packages, access network resources, create large files, or modify unrelated files.
- SNS augmentation is not implemented in this task.

## Stop Condition

Stop after implementation and validation. Report:

- files changed
- validation commands run and their results
- any skipped optional validation and why
- confirmation that forbidden paths and actions were not touched
- confirmation that this task did not perform real inference, training, dataset access, image reading, output writing, or SNS augmentation
