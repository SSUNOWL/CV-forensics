# Task 0020: Pre-SNS Integrated Model Smoke

## Role

Codex is the implementation worker in Codex-only mode. Codex must implement only the files allowed by this task file after the task file is committed and after the user explicitly says to implement.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `docs/model_output_schema.md`
- `docs/inference_stub.md`
- `docs/cf_small_multihead_tiny_smoke.md`
- `docs/sid_set_tiny_localization_smoke.md`, only if it exists
- `configs/training/cf_small_multihead_tiny_smoke.example.json`
- `configs/training/sid_set_tiny_localization_smoke.example.json`, only if it exists
- `scripts/training/run_cf_small_multihead_tiny_smoke.py`
- `scripts/training/run_sid_set_tiny_localization_smoke.py`, only if it exists
- `src/cv_forensics/model_output_schema.py`
- `src/cv_forensics/explanation_templates.py`
- `src/cv_forensics/local_data_gate.py`
- `tasks/0020-pre-sns-integrated-model-smoke.md`

## Files Codex May Modify

- `src/cv_forensics/__init__.py`
- `src/cv_forensics/pre_sns_integrated_model.py`
- `configs/training/pre_sns_integrated_smoke.example.json`
- `scripts/agent/validate_pre_sns_integrated_smoke_config.py`
- `scripts/training/run_pre_sns_integrated_smoke.py`
- `tests/test_pre_sns_integrated_smoke.py`
- `docs/pre_sns_integrated_smoke.md`

If a parent directory for an allowed file does not exist, Codex may create that parent directory. Codex must not create or modify any other files.

## Forbidden Actions During Implementation

Codex must not:

- download datasets
- train on real data
- run real optimization on real data
- inspect actual dataset directories recursively
- read real images outside temporary test fixtures
- read real masks outside temporary test fixtures
- access `.env`, `.env.*`, secrets, data, datasets, outputs, or checkpoints
- create `outputs/`
- create `checkpoints/`
- write generated predictions
- write checkpoints
- install packages
- access network resources from shell commands
- run `git push`, `git pull`, or `git fetch`
- run `curl`, `wget`, `ssh`, `scp`, `rsync`, `pip`, `npm`, `yarn`, `pnpm`, `kaggle`, `huggingface-cli`, or `wandb`
- use `rg`
- use `rm -rf`
- create large files
- implement SNS augmentation
- run SNS perturbation evaluation
- use danger-full-access
- bypass permissions

## Important Project Facts

- The project target outputs are class, mask/localization, family/provenance, and reason.
- Community Forensics-Small smoke workflows cover binary classification and generator-family provenance preparation.
- SID-Set smoke workflows cover 3-way classification and tampered localization preparation.
- This task creates the first integrated pre-SNS smoke interface combining:
  - shared image backbone
  - 3-way class head: real, full_synthetic, tampered
  - generator-family provenance head: LatDiff, PixDiff, GAN, Other, Real-or-N/A
  - conditional localization head for tampered samples
  - simple evidence/reason summary compatible with the existing model output schema
- This is a smoke test only, not full training and not a performance claim.
- This task must not implement SNS augmentation.
- This task must not run SNS perturbation evaluation.

## Implementation Requirements

1. Create `src/cv_forensics/pre_sns_integrated_model.py`
   - Use Python standard library plus torch only where needed.
   - Do not import PIL in the model module.
   - Provide a tiny deterministic integrated model suitable for smoke tests.
   - Model outputs must include class logits, family logits, localization logits, and evidence fields.
   - Expose labels for class and family.
   - Include helper functions for routing losses by source/task availability.
   - Localization loss must be conditional and apply only to tampered samples with masks.
   - Family loss must be conditional and apply only to samples with family labels.
   - Include a function that returns a schema-compatible JSON-like output summary.

2. Create `configs/training/pre_sns_integrated_smoke.example.json`
   - The config must be safe and symbolic only.
   - It must not contain real paths, URLs, secrets, data, datasets, outputs, checkpoints, or `.env`.
   - It must include guardrails:
     - `dry_run: true`
     - `no_download: true`
     - `no_network: true`
     - `no_outputs: true`
     - `no_checkpoints: true`
     - `no_sns_augmentation: true`
     - `no_sns_perturbation_eval: true`
   - It must include class labels:
     - `real`
     - `full_synthetic`
     - `tampered`
   - It must include family labels:
     - `LatDiff`
     - `PixDiff`
     - `GAN`
     - `Other`
     - `Real-or-N/A`
   - It must include loss routing policy for class, family, and localization.
   - It must include tiny limits:
     - `max_samples`
     - `max_steps`
     - `max_epochs`
     - `max_image_size`
     - `cpu_only`

3. Create `scripts/agent/validate_pre_sns_integrated_smoke_config.py`
   - Use only Python standard library.
   - Accept exactly one config path.
   - Validate `example_symbolic` and `approved_local_pre_sns_integrated_smoke`.
   - For approved local mode, require approval text exactly:
     `I_APPROVE_LOCAL_NON_SNS_PRE_SNS_INTEGRATED_SMOKE`
   - Reject protected paths, URLs, remote schemes, secrets, recursive scan flags, outputs, checkpoints, and SNS flags.
   - Print `PRE_SNS_INTEGRATED_SMOKE_CONFIG_OK` on success.
   - Do not read images or masks.
   - Do not write files.

4. Create `scripts/training/run_pre_sns_integrated_smoke.py`
   - Validate config before loading any image or mask.
   - Use CPU by default.
   - Use torch and PIL only if available.
   - Tests may use temporary fixtures.
   - Do not scan directories.
   - Do not write outputs or checkpoints.
   - Compute finite class, family, localization, and total losses when relevant labels/masks exist.
   - Print JSON summary with marker `PRE_SNS_INTEGRATED_SMOKE_OK`.

5. Create `tests/test_pre_sns_integrated_smoke.py`
   - Use a standard library test harness.
   - Do not import pytest.
   - Must be runnable with:
     `python3 tests/test_pre_sns_integrated_smoke.py`
   - Use temporary image/mask fixtures only.
   - If torch or PIL is unavailable, skip runner execution with a clear message but still test validators.
   - Test safe example config, local approved config, rejected unsafe paths, missing approvals, bad labels, bad family labels, missing tampered masks, and conditional loss routing.

6. Create `docs/pre_sns_integrated_smoke.md`
   - Explain purpose, class head, family head, conditional localization head, evidence/reason output, and why this is not a performance claim.
   - Include marker:
     `PRE_SNS_INTEGRATED_SMOKE_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_integrated_smoke_config.py configs/training/pre_sns_integrated_smoke.example.json
```

```bash
python3 tests/test_pre_sns_integrated_smoke.py
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0020-pre-sns-integrated-model-smoke.md
```

```bash
test -f src/cv_forensics/pre_sns_integrated_model.py
```

```bash
test -f configs/training/pre_sns_integrated_smoke.example.json
```

```bash
test -f scripts/agent/validate_pre_sns_integrated_smoke_config.py
```

```bash
test -f scripts/training/run_pre_sns_integrated_smoke.py
```

```bash
test -f tests/test_pre_sns_integrated_smoke.py
```

```bash
test -f docs/pre_sns_integrated_smoke.md
```

```bash
grep -q PRE_SNS_INTEGRATED_SMOKE_OK docs/pre_sns_integrated_smoke.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- src/cv_forensics/__init__.py src/cv_forensics/pre_sns_integrated_model.py configs/training/pre_sns_integrated_smoke.example.json scripts/agent/validate_pre_sns_integrated_smoke_config.py scripts/training/run_pre_sns_integrated_smoke.py tests/test_pre_sns_integrated_smoke.py docs/pre_sns_integrated_smoke.md
```

Optional validation command:

```bash
pytest -q tests/test_pre_sns_integrated_smoke.py
```

## Acceptance Criteria

- Validator passes for the tracked example config.
- Standalone tests pass.
- `check_agent_changes` passes.
- Docs contain `PRE_SNS_INTEGRATED_SMOKE_OK`.
- Changed files are limited to the allowed files.
- The future runner can compute finite class, family, localization, and total losses on approved local fixtures.
- The future runner prints `PRE_SNS_INTEGRATED_SMOKE_OK`.
- No dataset download, full training, network access, output writing, checkpoint writing, protected path access, SNS augmentation, or SNS perturbation evaluation occurs.

## Stop Condition

Stop after creating the task file. Report:

- task file path
- short summary
- git status
- exact git add command
- exact git commit command
