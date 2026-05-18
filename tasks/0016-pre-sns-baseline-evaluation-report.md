# Task 0016: Pre-SNS Baseline Evaluation Report Scaffold

## Task Title

Create a pre-SNS baseline evaluation report scaffold.

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
- `docs/metrics.md`
- `docs/training_dry_run.md`
- `docs/local_data_readiness.md`
- `docs/cf_small_subset_smoke.md`
- `docs/sid_set_subset_smoke.md`
- `docs/cf_small_baseline_plan.md`
- `docs/sid_set_baseline_plan.md`
- `configs/training/cf_small_baseline.example.json`
- `configs/training/sid_set_multihead_baseline.example.json`
- `configs/training/dry_run_training.example.json`, only if it exists
- `configs/local_data/readiness.example.json`
- `configs/local_data/cf_small_subset_smoke.example.json`
- `configs/local_data/sid_set_subset_smoke.example.json`
- `src/cv_forensics/local_data_gate.py`
- `tasks/0016-pre-sns-baseline-evaluation-report.md`
- `configs/manifests/community_forensics_small.example.json`, only if it exists
- `configs/manifests/sid_set.example.json`, only if it exists
- `configs/manifests/combined_smoke_manifest.example.json`, only if it exists
- `tests/test_metrics.py`, only if useful
- `tests/test_training_dry_run.py`, only if useful
- `tests/test_cf_small_baseline_plan.py`, only if useful
- `tests/test_sid_set_baseline_plan.py`, only if useful

## Files Codex May Modify

- `configs/reports/pre_sns_baseline_report.example.json`
- `scripts/agent/validate_pre_sns_baseline_report.py`
- `tests/test_pre_sns_baseline_report.py`
- `docs/pre_sns_baseline_report.md`

If a parent directory for an allowed file does not exist, Codex may create that parent directory. Codex must not create or modify any other files.

## Forbidden Actions

Codex must not:

- download datasets
- download Community Forensics-Small
- download SID-Set
- train a model
- run real optimization
- run real evaluation
- inspect actual dataset directories recursively
- read real images
- read real masks
- validate actual image or mask file existence
- access `.env`, `.env.*`, secrets, data, datasets, outputs, or checkpoints
- create `outputs/`
- create `checkpoints/`
- write generated predictions
- write generated reports to `outputs/`
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
- run SNS perturbation evaluation
- use `danger-full-access`
- bypass permissions

## Important Project Facts

- The final project target outputs are class, mask/localization, family/provenance, and reason.
- Community Forensics-Small is intended for shared backbone planning, real-vs-synthetic baseline planning, and coarse generator-family/provenance planning.
- SID-Set is intended for 3-way classification, tampered localization, and evidence/reason alignment planning.
- The model architecture direction is shared visual backbone, 3-way classification head, generator-family provenance head, conditional localization head, evidence aggregation, and deterministic template-based explanation.
- Pre-SNS baseline metrics must be collected before any SNS augmentation stage.
- The pre-SNS baseline report should include planned fields for 3-way accuracy, Macro-F1, mask IoU, generator-family accuracy, localization activation recall, latency, and FPS.
- SNS robustness drop is a future comparison field and must remain pending in this task.
- This task must not claim real metric values.
- This task must not implement SNS augmentation.
- This task must not run SNS perturbation evaluation.
- SNS augmentation begins only after the pre-SNS baseline report scaffold is committed and the user explicitly approves the next SNS-stage task.

## Implementation Requirements

1. Create `configs/reports/pre_sns_baseline_report.example.json`
   - The config must be symbolic and report-scaffold-only.
   - It must be dry-run safe.
   - It must not contain real local dataset paths.
   - It must not contain URLs.
   - It must not contain secrets.
   - It must not contain `data/`, `datasets/`, `outputs/`, `checkpoints/`, `.env`, or secret-like values.
   - It must not claim real metric values.
   - It must include:
     - `schema_version`
     - `report_name`
     - `report_stage: pre_sns_baseline`
     - `execution_mode: report_scaffold_only`
     - `dry_run: true`
     - `no_download: true`
     - `no_training: true`
     - `no_network: true`
     - `no_outputs: true`
     - `no_checkpoints: true`
     - `no_real_evaluation: true`
     - `no_real_image_reading: true`
     - `no_real_mask_reading: true`
     - `no_sns_augmentation: true`
     - `no_sns_perturbation_eval: true`
     - `requires_user_approval_for_real_evaluation: true`
     - `local_data_gate_ref`
     - `cf_small_baseline_ref`
     - `sid_set_baseline_ref`
     - `report_scope`
     - `dataset_scope`
     - `model_output_scope`
     - `metric_fields`
     - `metric_status_policy`
     - `baseline_collection_policy`
     - `comparison_policy`
     - `output_policy`
     - `checkpoint_policy`
     - `sns_future_stage_policy`
     - `validation_notes`
   - `report_scope` must state that this report is for pre-SNS baseline comparison.
   - `dataset_scope` must include:
     - Community Forensics-Small baseline role
     - SID-Set baseline role
     - no real data access in this task
   - `model_output_scope` must include:
     - class
     - mask/localization
     - family/provenance
     - reason
   - `metric_fields` must include all required metric entries:
     - `three_way_accuracy`
     - `macro_f1`
     - `mask_iou`
     - `generator_family_accuracy`
     - `localization_activation_recall`
     - `latency`
     - `fps`
   - Each metric entry must be a placeholder, not a real value.
   - Each metric entry must include:
     - `display_name`
     - `status`
     - `value`
     - `unit_or_definition`
     - `source_stage`
     - `required_before_sns`
   - For all current metric entries:
     - `status` must be `pending_real_baseline_run`
     - `value` must be `null`
   - `comparison_policy` must include:
     - `pre_sns_baseline_anchor`
     - `future_sns_robustness_drop`
     - `future_sns_augmented_training_comparison`
   - `future_sns_robustness_drop` must be marked `pending_future_sns_stage`.
   - `baseline_collection_policy` must require explicit user approval before real metric collection.
   - `output_policy` must require explicit approval before writing reports or predictions outside tracked docs/configs.
   - `checkpoint_policy` must require explicit approval before writing checkpoints.
   - `sns_future_stage_policy` must state:
     - SNS augmentation is not implemented in this task
     - SNS perturbation evaluation is not run in this task
     - SNS-stage tasks begin only after this pre-SNS baseline report scaffold is committed and explicitly approved

2. Create `scripts/agent/validate_pre_sns_baseline_report.py`
   - Use only Python standard library.
   - Accept exactly one positional argument: pre-SNS baseline report config JSON path.
   - Add repo `src` path to `sys.path` if needed.
   - Reuse `src/cv_forensics/local_data_gate.py` safety checks where appropriate.
   - Validate that the config is symbolic, report-scaffold-only, and dry-run safe.
   - Validate required guardrail flags:
     - `dry_run` is true
     - `no_download` is true
     - `no_training` is true
     - `no_network` is true
     - `no_outputs` is true
     - `no_checkpoints` is true
     - `no_real_evaluation` is true
     - `no_real_image_reading` is true
     - `no_real_mask_reading` is true
     - `no_sns_augmentation` is true
     - `no_sns_perturbation_eval` is true
     - `requires_user_approval_for_real_evaluation` is true
   - Validate `report_stage` is `pre_sns_baseline`.
   - Validate `execution_mode` is `report_scaffold_only`.
   - Validate `model_output_scope` includes class, mask/localization, family/provenance, and reason.
   - Validate `metric_fields` include:
     - `three_way_accuracy`
     - `macro_f1`
     - `mask_iou`
     - `generator_family_accuracy`
     - `localization_activation_recall`
     - `latency`
     - `fps`
   - Validate each required metric entry:
     - `status` is `pending_real_baseline_run`
     - `value` is `null`
     - `required_before_sns` is true
   - Validate that completed real metric values are rejected in this task.
   - Validate `comparison_policy` includes:
     - `pre_sns_baseline_anchor`
     - `future_sns_robustness_drop`
     - `future_sns_augmented_training_comparison`
   - Validate `future_sns_robustness_drop` is `pending_future_sns_stage`.
   - Validate `baseline_collection_policy` requires explicit approval.
   - Validate `output_policy` and `checkpoint_policy` are approval-gated.
   - Validate `sns_future_stage_policy` explicitly says SNS augmentation is not implemented here and SNS perturbation evaluation is not run here.
   - Validate the config recursively rejects protected paths, URLs, absolute paths, Windows drive paths, and secret-like keys/values.
   - Print actionable errors and exit non-zero on fail.
   - Print a concise success message containing `PRE_SNS_BASELINE_REPORT_OK` on pass.
   - Do not write files.
   - Do not read images or masks.
   - Do not inspect real dataset directories.
   - Do not run real evaluation.

3. Create `tests/test_pre_sns_baseline_report.py`
   - Use only Python standard library.
   - Do not import `pytest`.
   - It must be runnable with:
     `python3 tests/test_pre_sns_baseline_report.py`
   - It may be pytest-compatible if pytest is installed.
   - Tests should cover:
     - valid example config passes
     - execution_mode other than `report_scaffold_only` is rejected
     - report_stage other than `pre_sns_baseline` is rejected
     - dry_run false is rejected
     - no_download false is rejected
     - no_training false is rejected
     - no_outputs false is rejected
     - no_checkpoints false is rejected
     - no_real_evaluation false is rejected
     - no_real_image_reading false is rejected
     - no_real_mask_reading false is rejected
     - no_sns_augmentation false is rejected
     - no_sns_perturbation_eval false is rejected
     - requires_user_approval_for_real_evaluation false is rejected
     - missing class output scope is rejected
     - missing mask/localization output scope is rejected
     - missing family/provenance output scope is rejected
     - missing reason output scope is rejected
     - missing three_way_accuracy metric is rejected
     - missing macro_f1 metric is rejected
     - missing mask_iou metric is rejected
     - missing generator_family_accuracy metric is rejected
     - missing localization_activation_recall metric is rejected
     - missing latency metric is rejected
     - missing fps metric is rejected
     - metric value that is not null is rejected
     - metric status other than `pending_real_baseline_run` is rejected
     - metric `required_before_sns` false is rejected
     - missing `future_sns_robustness_drop` comparison field is rejected
     - `future_sns_robustness_drop` not pending is rejected
     - baseline collection without explicit approval gate is rejected
     - output writing without approval is rejected
     - checkpoint writing without approval is rejected
     - SNS augmentation enabled is rejected
     - SNS perturbation evaluation enabled is rejected
     - protected paths are rejected
     - URLs are rejected
     - absolute paths are rejected
     - Windows drive paths are rejected
     - secret-like keys and values are rejected
     - ordinary prose containing words like authoritative or authentication is not rejected merely because it contains auth
     - validator does not write files
   - Tests must not access data, datasets, outputs, checkpoints, secrets, or `.env`.
   - Tests must not read real images or masks.
   - Tests must not run real evaluation.

4. Create `docs/pre_sns_baseline_report.md`
   - Explain the purpose of the pre-SNS baseline evaluation report scaffold.
   - Explain that this is not dataset download, not training, not real evaluation, and not real image/mask reading.
   - Explain that no real metric values are claimed in this task.
   - Explain how this report becomes the anchor for later SNS robustness and SNS augmentation-aware training comparison.
   - Explain how it depends on:
     - model output schema from task 0007
     - fake inference from task 0008
     - metrics from task 0009
     - training dry-run from task 0010
     - local data readiness from task 0011
     - CF-Small subset smoke from task 0012
     - SID-Set subset smoke from task 0013
     - CF-Small baseline plan from task 0014
     - SID-Set multi-head baseline plan from task 0015
   - Explain required pre-SNS metric fields:
     - 3-way accuracy
     - Macro-F1
     - mask IoU
     - generator-family accuracy
     - localization activation recall
     - latency
     - FPS
   - Explain that every metric value remains pending until explicit user approval for real baseline evaluation.
   - Explain that SNS robustness drop is a future comparison field and remains pending until SNS-stage tasks.
   - Explain that SNS augmentation is not implemented here.
   - Explain what must happen before real pre-SNS baseline evaluation:
     - explicit user approval
     - validated local path policy
     - validated manifests
     - local subset smoke confirmation
     - approved output policy
     - approved checkpoint policy
     - compute budget policy
   - Include marker string:
     - `PRE_SNS_BASELINE_REPORT_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_baseline_report.py configs/reports/pre_sns_baseline_report.example.json
```

```bash
python3 tests/test_pre_sns_baseline_report.py
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0016-pre-sns-baseline-evaluation-report.md
```

```bash
test -f configs/reports/pre_sns_baseline_report.example.json
```

```bash
test -f scripts/agent/validate_pre_sns_baseline_report.py
```

```bash
test -f tests/test_pre_sns_baseline_report.py
```

```bash
test -f docs/pre_sns_baseline_report.md
```

```bash
grep -q PRE_SNS_BASELINE_REPORT_OK docs/pre_sns_baseline_report.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- configs/reports/pre_sns_baseline_report.example.json scripts/agent/validate_pre_sns_baseline_report.py tests/test_pre_sns_baseline_report.py docs/pre_sns_baseline_report.md
```

Optional validation command:

```bash
pytest -q tests/test_pre_sns_baseline_report.py
```

## Acceptance Criteria

- `python3 scripts/agent/validate_pre_sns_baseline_report.py configs/reports/pre_sns_baseline_report.example.json` passes.
- `python3 tests/test_pre_sns_baseline_report.py` passes.
- `python3 scripts/agent/check_agent_changes.py tasks/0016-pre-sns-baseline-evaluation-report.md` passes.
- If pytest exists, `pytest -q tests/test_pre_sns_baseline_report.py` passes.
- `docs/pre_sns_baseline_report.md` contains `PRE_SNS_BASELINE_REPORT_OK`.
- Changed files are limited to:
  - `configs/reports/pre_sns_baseline_report.example.json`
  - `scripts/agent/validate_pre_sns_baseline_report.py`
  - `tests/test_pre_sns_baseline_report.py`
  - `docs/pre_sns_baseline_report.md`
- The config is symbolic, report-scaffold-only, and dry-run safe.
- The report includes placeholders for:
  - 3-way accuracy
  - Macro-F1
  - mask IoU
  - generator-family accuracy
  - localization activation recall
  - latency
  - FPS
- No real metric values are claimed.
- All metric values remain pending until explicit real baseline evaluation approval.
- SNS robustness drop remains pending future SNS-stage comparison.
- SNS augmentation is not implemented.
- SNS perturbation evaluation is not run.
- Output writing and checkpoint writing require explicit approval.
- No dataset download, model training, package installation, network access, real evaluation, real image reading, real mask reading, output writing, checkpoint writing, SNS augmentation, SNS perturbation evaluation, or protected path access occurs.

## Stop Condition

Stop after creating the task file. Report:

- task file path
- short summary
- git status
- exact git add command for the task file
- exact git commit command for the task file
