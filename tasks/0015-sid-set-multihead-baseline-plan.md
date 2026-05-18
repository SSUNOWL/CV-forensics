# Task 0015: SID-Set Multi-Head Baseline Plan

## Task Title

Create a SID-Set multi-head fine-tuning and localization baseline plan.

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
- `docs/cf_small_baseline_plan.md`
- `docs/model_output_schema.md`
- `docs/inference_stub.md`
- `docs/metrics.md`
- `docs/training_dry_run.md`
- `configs/local_data/readiness.example.json`
- `configs/local_data/cf_small_subset_smoke.example.json`
- `configs/local_data/sid_set_subset_smoke.example.json`
- `configs/training/cf_small_baseline.example.json`
- `configs/training/dry_run_training.example.json`, only if it exists
- `src/cv_forensics/local_data_gate.py`
- `tasks/0015-sid-set-multihead-baseline-plan.md`
- `configs/manifests/sid_set.example.json`, only if it exists
- `configs/manifests/combined_smoke_manifest.example.json`, only if it exists
- `tests/test_local_data_gate.py`, only if useful
- `tests/test_sid_set_subset_smoke.py`, only if useful
- `tests/test_cf_small_baseline_plan.py`, only if useful
- `tests/test_training_dry_run.py`, only if useful
- `tests/test_model_output_schema.py`, only if useful
- `tests/test_metrics.py`, only if useful

## Files Codex May Modify

- `configs/training/sid_set_multihead_baseline.example.json`
- `scripts/agent/validate_sid_set_baseline_plan.py`
- `tests/test_sid_set_baseline_plan.py`
- `docs/sid_set_baseline_plan.md`

If a parent directory for an allowed file does not exist, Codex may create that parent directory. Codex must not create or modify any other files.

## Files Claude May Modify

This Codex-only task preserves this section name so `scripts/agent/check_agent_changes.py` can enforce the same allowed modification list.

- `configs/training/sid_set_multihead_baseline.example.json`
- `scripts/agent/validate_sid_set_baseline_plan.py`
- `tests/test_sid_set_baseline_plan.py`
- `docs/sid_set_baseline_plan.md`

## Forbidden Actions

Codex must not:

- download datasets
- download SID-Set
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
- SID-Set is intended in this project for 3-way classification and tampered localization planning.
- Class labels are `real`, `synthetic`, and `tampered`.
- Localization is relevant mainly for tampered samples.
- Synthetic samples may not have tampered masks.
- Real samples should not have tampered masks.
- Family/provenance labels may be missing or coarse for SID-Set, so the plan must define a policy for missing family/provenance labels.
- Conditional localization activates when `tampered_score >= threshold_tau`.
- Localization activation recall is important because high `threshold_tau` can cause false negatives.
- The model architecture direction is:
  - shared visual backbone
  - 3-way classification head
  - generator-family provenance head
  - conditional localization head
  - evidence aggregation
  - deterministic template-based explanation
- CF-Small baseline planning provides shared backbone and coarse provenance planning.
- SID-Set baseline planning provides 3-way classification, tampered localization, and explanation/evidence alignment planning.
- This task must not implement actual training.
- This task must not implement SNS augmentation.
- SNS augmentation begins only after pre-SNS baseline evaluation is committed.

## Implementation Requirements

### 1. Create `configs/training/sid_set_multihead_baseline.example.json`

- The config must be symbolic and plan-only.
- It must be dry-run safe.
- It must not contain real local dataset paths.
- It must not contain URLs.
- It must not contain secrets.
- It must not contain `data/`, `datasets/`, `outputs/`, `checkpoints/`, `.env`, or secret-like values.
- It must include:
  - `schema_version`
  - `plan_name`
  - `dataset_name: SID-Set`
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
  - `sid_set_subset_smoke_ref`
  - `cf_small_baseline_ref`
  - `dataset_role`
  - `model_plan`
  - `head_plan`
  - `classification_policy`
  - `localization_policy`
  - `mask_policy`
  - `family_policy`
  - `explanation_policy`
  - `threshold_policy`
  - `loss_plan`
  - `freeze_policy`
  - `batch_mixing_policy`
  - `split_policy`
  - `evaluation_policy`
  - `metric_plan`
  - `output_policy`
  - `checkpoint_policy`
  - `handoff_to_pre_sns_baseline`
  - `validation_notes`
- `dataset_role` must clearly state:
  - 3-way classification planning
  - tampered localization planning
  - evidence/reason alignment planning
- `model_plan` must represent:
  - shared visual backbone
  - CF-Small initialized or preplanned backbone handoff
  - 3-way classification head
  - generator-family provenance head
  - conditional localization head
  - evidence aggregation
  - template-based explanation
- `head_plan` must represent each head explicitly:
  - `classification_head`
  - `provenance_head`
  - `localization_head`
  - `evidence_reason_head` or `evidence_aggregation`
- `classification_policy` must represent exactly:
  - `real`
  - `synthetic`
  - `tampered`
- `localization_policy` must represent:
  - conditional localization activation
  - `threshold_tau`
  - `tampered_score >= threshold_tau` activates localization
  - localization activation recall
  - mask IoU
  - false negative risk when `threshold_tau` is too high
- `mask_policy` must represent:
  - tampered samples require mask/localization planning
  - real samples must not require tampered masks
  - synthetic samples may not require tampered masks unless explicitly annotated
  - no real mask files are read in this task
- `family_policy` must represent:
  - missing family/provenance labels are allowed in SID-Set planning
  - missing synthetic/tampered family labels should map to `Unknown` unless confirmed
  - real samples should map to `Real-or-N/A`
  - provenance/family head training remains gated until labels are confirmed
  - CF-Small family/provenance planning can help initialize or regularize the family head later
- `explanation_policy` must represent:
  - deterministic template-based reason generation
  - reason is derived from evidence signals and output schema, not from free-form hallucination
  - no LLM call is required in this task
- `loss_plan` must be plan-only and must include symbolic loss components:
  - `classification_loss`
  - `optional_family_loss`
  - `conditional_localization_loss`
  - `evidence_consistency_loss` or `explanation_alignment_loss`
- `freeze_policy` must describe:
  - option to freeze the shared backbone for initial smoke
  - option to unfreeze selected layers after local smoke passes
  - no actual freezing is executed in this task
- `batch_mixing_policy` must describe:
  - SID-Set-only baseline option
  - optional CF-Small auxiliary batch or regularization policy
  - no actual batch loading is executed in this task
- `split_policy` must represent:
  - train/validation/test separation planning
  - no leakage
  - class balance across `real`/`synthetic`/`tampered`
  - mask availability checks only after explicit local path approval
- `metric_plan` must include:
  - 3-way accuracy
  - Macro-F1
  - mask IoU
  - generator-family accuracy
  - localization activation recall
  - latency
  - FPS
- `output_policy` must require explicit approval before writing outputs.
- `checkpoint_policy` must require explicit approval before writing checkpoints.
- `handoff_to_pre_sns_baseline` must explain that task 0016 will summarize pre-SNS baseline metrics before SNS augmentation.

### 2. Create `scripts/agent/validate_sid_set_baseline_plan.py`

- Use only Python standard library.
- Accept exactly one positional argument: SID-Set multi-head baseline plan config JSON path.
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
- Validate `dataset_name` is `SID-Set`.
- Validate `execution_mode` is `plan_only`.
- Validate `model_plan` contains:
  - shared visual backbone
  - 3-way classification head
  - generator-family provenance head
  - conditional localization head
  - evidence aggregation or reason head
- Validate `classification_policy` includes exactly:
  - `real`
  - `synthetic`
  - `tampered`
- Validate `localization_policy` includes:
  - conditional activation
  - `threshold_tau`
  - mask IoU
  - localization activation recall
  - false negative risk
- Validate `threshold_tau` is between `0.0` and `1.0`.
- Validate `mask_policy` is explicit and does not require masks for real samples.
- Validate `family_policy` includes missing-label handling and `Real-or-N/A` mapping.
- Validate `explanation_policy` is deterministic/template-based and does not require LLM calls.
- Validate `loss_plan` is symbolic and does not execute training.
- Validate `freeze_policy` is explicit.
- Validate `batch_mixing_policy` is explicit.
- Validate `split_policy` includes leakage prevention.
- Validate `metric_plan` includes:
  - 3-way accuracy
  - Macro-F1
  - mask IoU
  - generator-family accuracy
  - localization activation recall
  - latency
  - FPS
- Validate `output_policy` and `checkpoint_policy` are approval-gated.
- Validate `handoff_to_pre_sns_baseline` is explicit.
- Validate the config recursively rejects protected paths, URLs, absolute paths, Windows drive paths, and secret-like keys/values.
- Print actionable errors and exit non-zero on fail.
- Print a concise success message containing `SID_SET_BASELINE_PLAN_OK` on pass.
- Do not write files.
- Do not read images or masks.
- Do not inspect real dataset directories.

### 3. Create `tests/test_sid_set_baseline_plan.py`

- Use only Python standard library.
- Do not import `pytest`.
- It must be runnable with:

```bash
python3 tests/test_sid_set_baseline_plan.py
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
  - missing `tampered` class is rejected
  - extra class label is rejected
  - missing shared visual backbone plan is rejected
  - missing classification head is rejected
  - missing provenance head is rejected
  - missing localization head is rejected
  - missing evidence aggregation or reason head is rejected
  - missing conditional localization activation is rejected
  - missing mask IoU metric is rejected
  - missing localization activation recall metric is rejected
  - `threshold_tau` outside `0.0` to `1.0` is rejected
  - missing false negative risk note is rejected
  - missing family/provenance missing-label policy is rejected
  - missing `Real-or-N/A` policy for real samples is rejected
  - missing deterministic explanation policy is rejected
  - LLM-required explanation policy is rejected
  - missing freeze policy is rejected
  - missing batch mixing policy is rejected
  - missing leakage prevention policy is rejected
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

### 4. Create `docs/sid_set_baseline_plan.md`

- Explain the purpose of the SID-Set multi-head baseline plan.
- Explain that this is not dataset download, not training, and not real image/mask reading.
- Explain how SID-Set maps to project goals:
  - 3-way classification
  - tampered localization
  - mask IoU
  - localization activation recall
  - family/provenance handling
  - reason/template explanation
- Explain the planned architecture:
  - shared visual backbone
  - 3-way classification head
  - generator-family provenance head
  - conditional localization head
  - evidence aggregation
  - deterministic template explanation
- Explain how this depends on:
  - local data readiness gate from task 0011
  - CF-Small subset smoke plan from task 0012
  - SID-Set subset smoke plan from task 0013
  - CF-Small baseline plan from task 0014
- Explain how CF-Small can contribute backbone/provenance initialization while SID-Set handles 3-way classification and tampered localization.
- Explain what must happen before real SID-Set baseline fine-tuning:
  - explicit user approval
  - validated local path policy
  - validated manifest
  - tiny subset smoke confirmation
  - output policy
  - checkpoint policy
  - compute budget policy
- Explain planned metrics:
  - 3-way accuracy
  - Macro-F1
  - mask IoU
  - generator-family accuracy
  - localization activation recall
  - latency
  - FPS
- Explain that SNS augmentation is not implemented here.
- Explain that task 0016 will collect pre-SNS baseline evaluation report scaffolding.
- Include marker string:
  - `SID_SET_BASELINE_PLAN_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_sid_set_baseline_plan.py configs/training/sid_set_multihead_baseline.example.json
```

```bash
python3 tests/test_sid_set_baseline_plan.py
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0015-sid-set-multihead-baseline-plan.md
```

```bash
test -f configs/training/sid_set_multihead_baseline.example.json
```

```bash
test -f scripts/agent/validate_sid_set_baseline_plan.py
```

```bash
test -f tests/test_sid_set_baseline_plan.py
```

```bash
test -f docs/sid_set_baseline_plan.md
```

```bash
grep -q SID_SET_BASELINE_PLAN_OK docs/sid_set_baseline_plan.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- configs/training/sid_set_multihead_baseline.example.json scripts/agent/validate_sid_set_baseline_plan.py tests/test_sid_set_baseline_plan.py docs/sid_set_baseline_plan.md
```

Optional validation command:

```bash
pytest -q tests/test_sid_set_baseline_plan.py
```

## Acceptance Criteria

- `python3 scripts/agent/validate_sid_set_baseline_plan.py configs/training/sid_set_multihead_baseline.example.json` passes.
- `python3 tests/test_sid_set_baseline_plan.py` passes.
- `python3 scripts/agent/check_agent_changes.py tasks/0015-sid-set-multihead-baseline-plan.md` passes.
- If pytest exists, `pytest -q tests/test_sid_set_baseline_plan.py` passes.
- `docs/sid_set_baseline_plan.md` contains `SID_SET_BASELINE_PLAN_OK`.
- Changed files are limited to:
  - `configs/training/sid_set_multihead_baseline.example.json`
  - `scripts/agent/validate_sid_set_baseline_plan.py`
  - `tests/test_sid_set_baseline_plan.py`
  - `docs/sid_set_baseline_plan.md`
- The config is symbolic, plan-only, and dry-run safe.
- SID-Set 3-way classification planning is represented.
- Tampered localization planning is represented.
- Conditional localization threshold and activation recall planning are represented.
- Mask IoU planning is represented.
- Generator-family/provenance policy is represented.
- Missing family/provenance label policy is represented.
- Deterministic template explanation policy is represented.
- Output writing and checkpoint writing require explicit approval.
- No dataset download, model training, package installation, network access, real image reading, real mask reading, output writing, checkpoint writing, SNS augmentation, or protected path access occurs.

## Stop Condition

Stop after implementation and validation. Report:

- files changed
- validation commands run and results
- whether optional pytest validation was run or skipped
- confirmation that changed files are limited to the allowed list
- confirmation that no dataset download, model training, package installation, network access, real image reading, real mask reading, output writing, checkpoint writing, SNS augmentation, or protected path access occurred
