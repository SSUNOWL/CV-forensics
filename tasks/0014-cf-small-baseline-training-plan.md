# Task 0014: CF-Small Baseline Training Plan

## Task Title

Create a Community Forensics-Small baseline training and evaluation plan.

## Role

Codex is the implementation worker in Codex-only mode. Codex must implement only the files allowed by this task file after the task file is committed and after the user explicitly says to implement.

Codex must not implement this task while creating this task file.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `docs/local_data_readiness.md`
- `docs/cf_small_subset_smoke.md`
- `docs/sid_set_subset_smoke.md`
- `configs/local_data/readiness.example.json`
- `configs/local_data/cf_small_subset_smoke.example.json`
- `configs/local_data/sid_set_subset_smoke.example.json`
- `src/cv_forensics/local_data_gate.py`
- `tasks/0014-cf-small-baseline-training-plan.md`
- `configs/manifests/community_forensics_small.example.json`, only if it exists
- `configs/manifests/combined_smoke_manifest.example.json`, only if it exists
- `tests/test_local_data_gate.py`, only if useful
- `tests/test_cf_small_subset_smoke.py`, only if useful
- `tests/test_sid_set_subset_smoke.py`, only if useful

## Files Codex May Modify

- `configs/training/cf_small_baseline.example.json`
- `scripts/agent/validate_cf_small_baseline_plan.py`
- `tests/test_cf_small_baseline_plan.py`
- `docs/cf_small_baseline_plan.md`

If a parent directory for an allowed file does not exist, Codex may create that parent directory. Codex must not create or modify any other files.

## Files Claude May Modify

This Codex-only task preserves this section name so `scripts/agent/check_agent_changes.py` can enforce the same allowed modification list.

- `configs/training/cf_small_baseline.example.json`
- `scripts/agent/validate_cf_small_baseline_plan.py`
- `tests/test_cf_small_baseline_plan.py`
- `docs/cf_small_baseline_plan.md`

## Forbidden Actions

Codex must not:

- download datasets
- download Community Forensics-Small
- train a model
- run real optimization
- inspect actual dataset directories recursively
- read real images
- read real masks
- validate actual image or mask file existence
- access `.env`, `.env.*`, `secrets`, `data`, `datasets`, `outputs`, or `checkpoints`
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
- import `torch`, `torchvision`, `PIL`, `cv2`, `numpy`, `pandas`, `sklearn`, or `pytest`
- create large files
- implement SNS augmentation
- use danger-full-access
- bypass permissions

## Important Project Facts

- The final project target outputs are class, mask/localization, family/provenance, and reason.
- Community Forensics-Small is intended in this project for shared visual backbone planning and coarse provenance/family planning.
- Community Forensics-Small baseline planning should focus on real-vs-synthetic discrimination and generator-family/provenance planning.
- Community Forensics-Small should not be treated as a tampered mask localization dataset.
- Tampered localization and mask IoU are primarily SID-Set responsibilities.
- The model architecture direction is:
  - shared visual backbone
  - 3-way classification head
  - generator-family provenance head
  - conditional localization head
  - evidence aggregation
  - deterministic template-based explanation
- For this task, the Community Forensics-Small plan should represent:
  - backbone pretraining or initialization planning
  - real-vs-synthetic baseline classification planning
  - generator/model-family holdout planning
  - provenance/family head planning
  - handoff to SID-Set for 3-way classification and tampered localization
- This task must not implement actual training.
- This task must not implement SNS augmentation.
- SNS augmentation begins only after pre-SNS baseline evaluation is committed.

## Implementation Requirements

### 1. Create `configs/training/cf_small_baseline.example.json`

- The config must be symbolic and plan-only.
- It must be dry-run safe.
- It must not contain real local dataset paths.
- It must not contain URLs.
- It must not contain secrets.
- It must not contain `data/`, `datasets/`, `outputs/`, `checkpoints/`, `.env`, or secret-like values.
- It must include:
  - `schema_version`
  - `plan_name`
  - `dataset_name: Community Forensics-Small`
  - `execution_mode: plan_only`
  - `dry_run: true`
  - `no_download: true`
  - `no_training: true`
  - `no_network: true`
  - `no_outputs: true`
  - `no_checkpoints: true`
  - `no_real_image_reading: true`
  - `no_real_mask_reading: true`
  - `requires_user_approval_for_real_training: true`
  - `local_data_gate_ref`
  - `cf_small_subset_smoke_ref`
  - `dataset_role`
  - `training_objective_plan`
  - `backbone_plan`
  - `classification_policy`
  - `family_policy`
  - `generator_holdout_policy`
  - `split_policy`
  - `evaluation_policy`
  - `metric_plan`
  - `output_policy`
  - `checkpoint_policy`
  - `handoff_to_sid_set`
  - `validation_notes`
- `dataset_role` must clearly state:
  - shared backbone planning
  - real-vs-synthetic baseline planning
  - coarse provenance/family planning
- `classification_policy` must represent:
  - `real`
  - `synthetic`
- `classification_policy` must not claim that Community Forensics-Small is the tampered localization dataset.
- `family_policy` must represent coarse family planning, such as:
  - `LatDiff`
  - `PixDiff`
  - `GAN`
  - `Other`
  - `Real-or-N/A`
  - `Unknown`
- `generator_holdout_policy` must represent:
  - model-name or generator-family holdout
  - avoiding train/validation leakage
  - diversity-aware split planning
- `metric_plan` must represent:
  - binary accuracy
  - macro F1
  - generator-family accuracy
  - holdout/generalization evaluation
  - latency or FPS as later evaluation fields
- `metric_plan` must not require real mask IoU for Community Forensics-Small.
- `evaluation_policy` must be plan-only and must not run evaluation.
- `output_policy` must require explicit approval before writing outputs.
- `checkpoint_policy` must require explicit approval before writing checkpoints.
- `handoff_to_sid_set` must explain that SID-Set handles 3-way classification and tampered localization planning.

### 2. Create `scripts/agent/validate_cf_small_baseline_plan.py`

- Use only Python standard library.
- Accept exactly one positional argument: Community Forensics-Small baseline plan config JSON path.
- Add repo `src` path to `sys.path` if needed.
- Reuse `src/cv_forensics/local_data_gate.py` safety checks where appropriate.
- Validate that the config is symbolic, plan-only, and dry-run safe.
- Validate required guardrail flags:
  - `dry_run` is true
  - `no_download` is true
  - `no_training` is true
  - `no_network` is true
  - `no_outputs` is true
  - `no_checkpoints` is true
  - `no_real_image_reading` is true
  - `no_real_mask_reading` is true
  - `requires_user_approval_for_real_training` is true
- Validate `dataset_name` is `Community Forensics-Small`.
- Validate `execution_mode` is `plan_only`.
- Validate `classification_policy` includes `real` and `synthetic`.
- Validate `classification_policy` does not require tampered localization.
- Validate `family_policy` includes coarse family/provenance planning.
- Validate `generator_holdout_policy` includes leakage prevention and model-name or generator-family holdout.
- Validate `split_policy` is explicit.
- Validate `metric_plan` includes:
  - binary accuracy
  - macro F1
  - generator-family accuracy
  - holdout/generalization evaluation
- Validate `metric_plan` does not require real mask IoU for Community Forensics-Small.
- Validate `output_policy` and `checkpoint_policy` are approval-gated.
- Validate `handoff_to_sid_set` is explicit.
- Validate the config recursively rejects protected paths, URLs, absolute paths, Windows drive paths, and secret-like keys/values.
- Print actionable errors and exit non-zero on fail.
- Print a concise success message containing `CF_SMALL_BASELINE_PLAN_OK` on pass.
- Do not write files.
- Do not read images or masks.
- Do not inspect real dataset directories.

### 3. Create `tests/test_cf_small_baseline_plan.py`

- Use only Python standard library.
- Do not import `pytest`.
- It must be runnable with:

```bash
python3 tests/test_cf_small_baseline_plan.py
```

- It may also be pytest-compatible if pytest is installed.
- Tests should cover:
  - valid example config passes
  - `execution_mode` other than `plan_only` is rejected
  - `dry_run: false` is rejected
  - `no_download: false` is rejected
  - `no_training: false` is rejected
  - `no_outputs: false` is rejected
  - `no_checkpoints: false` is rejected
  - `no_real_image_reading: false` is rejected
  - `no_real_mask_reading: false` is rejected
  - `requires_user_approval_for_real_training: false` is rejected
  - wrong `dataset_name` is rejected
  - missing `real` class is rejected
  - missing `synthetic` class is rejected
  - tampered localization requirement in CF-Small plan is rejected
  - missing family policy is rejected
  - missing generator holdout policy is rejected
  - missing leakage prevention policy is rejected
  - missing split policy is rejected
  - missing binary accuracy metric is rejected
  - missing macro F1 metric is rejected
  - missing generator-family accuracy metric is rejected
  - real mask IoU as required CF-Small metric is rejected
  - output writing without approval is rejected
  - checkpoint writing without approval is rejected
  - protected paths are rejected
  - URLs are rejected
  - absolute paths are rejected
  - Windows drive paths are rejected
  - secret-like keys and values are rejected
  - ordinary prose containing words like `authoritative` or `authentication` is not rejected merely because it contains `auth`
  - validator does not write files
- Tests must not access `data`, `datasets`, `outputs`, `checkpoints`, `secrets`, or `.env`.
- Tests must not read real images or masks.

### 4. Create `docs/cf_small_baseline_plan.md`

- Explain the purpose of the Community Forensics-Small baseline training/evaluation plan.
- Explain that this is not dataset download, not training, and not real image/mask reading.
- Explain how Community Forensics-Small maps to project goals:
  - shared backbone planning
  - real-vs-synthetic baseline planning
  - generator-family/provenance planning
  - generator/model-family holdout planning
- Explain why Community Forensics-Small should not be used as the primary tampered localization source.
- Explain how it depends on:
  - local data readiness gate from task 0011
  - CF-Small subset smoke plan from task 0012
  - SID-Set subset smoke plan from task 0013
- Explain what must happen before real Community Forensics-Small baseline training:
  - explicit user approval
  - validated local path policy
  - validated manifest
  - tiny subset smoke confirmation
  - output policy
  - checkpoint policy
  - compute budget policy
- Explain the planned metrics:
  - binary accuracy
  - macro F1
  - generator-family accuracy
  - holdout/generalization evaluation
  - latency
  - FPS
- Explain that mask IoU and localization activation recall are SID-Set/localization-stage metrics, not required CF-Small metrics.
- State that SNS augmentation is not implemented here.
- Include marker string:
  - `CF_SMALL_BASELINE_PLAN_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_cf_small_baseline_plan.py configs/training/cf_small_baseline.example.json
```

```bash
python3 tests/test_cf_small_baseline_plan.py
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0014-cf-small-baseline-training-plan.md
```

```bash
test -f configs/training/cf_small_baseline.example.json
```

```bash
test -f scripts/agent/validate_cf_small_baseline_plan.py
```

```bash
test -f tests/test_cf_small_baseline_plan.py
```

```bash
test -f docs/cf_small_baseline_plan.md
```

```bash
grep -q CF_SMALL_BASELINE_PLAN_OK docs/cf_small_baseline_plan.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- configs/training/cf_small_baseline.example.json scripts/agent/validate_cf_small_baseline_plan.py tests/test_cf_small_baseline_plan.py docs/cf_small_baseline_plan.md
```

Optional validation command:

```bash
pytest -q tests/test_cf_small_baseline_plan.py
```

## Acceptance Criteria

- `python3 scripts/agent/validate_cf_small_baseline_plan.py configs/training/cf_small_baseline.example.json` passes.
- `python3 tests/test_cf_small_baseline_plan.py` passes.
- `python3 scripts/agent/check_agent_changes.py tasks/0014-cf-small-baseline-training-plan.md` passes.
- If pytest exists, `pytest -q tests/test_cf_small_baseline_plan.py` passes.
- `docs/cf_small_baseline_plan.md` contains `CF_SMALL_BASELINE_PLAN_OK`.
- Changed files are limited to:
  - `configs/training/cf_small_baseline.example.json`
  - `scripts/agent/validate_cf_small_baseline_plan.py`
  - `tests/test_cf_small_baseline_plan.py`
  - `docs/cf_small_baseline_plan.md`
- The config is symbolic, plan-only, and dry-run safe.
- Community Forensics-Small real-vs-synthetic baseline planning is represented.
- Shared backbone planning is represented.
- Generator-family/provenance planning is represented.
- Generator/model-family holdout and leakage prevention are represented.
- Output writing and checkpoint writing require explicit approval.
- No dataset download, model training, package installation, network access, real image reading, real mask reading, output writing, checkpoint writing, SNS augmentation, or protected path access occurs.

## Stop Condition

Stop after implementation and validation. Report:

- files changed
- validation commands run and results
- whether optional pytest validation was run or skipped
- confirmation that changed files are limited to the allowed list
- confirmation that no dataset download, model training, package installation, network access, real image reading, real mask reading, output writing, checkpoint writing, SNS augmentation, or protected path access occurred
