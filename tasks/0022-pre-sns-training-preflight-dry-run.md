# Task 0022: Pre-SNS Training Preflight Dry-Run

## Role

Codex is the implementation worker in Codex-only mode. Codex must implement only the files allowed by this task file after the task file is committed and after the user explicitly says to implement.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `docs/pre_sns_integrated_smoke.md`
- `docs/pre_sns_dataset_manifest.md`
- `src/cv_forensics/pre_sns_integrated_model.py`
- `src/cv_forensics/pre_sns_manifest.py`
- `src/cv_forensics/local_data_gate.py`
- `tasks/0022-pre-sns-training-preflight-dry-run.md`

## Files Codex May Modify

- `src/cv_forensics/pre_sns_preflight.py`
- `configs/training/pre_sns_training_preflight.example.json`
- `scripts/agent/validate_pre_sns_training_preflight_config.py`
- `scripts/training/run_pre_sns_training_preflight.py`
- `tests/test_pre_sns_training_preflight.py`
- `docs/pre_sns_training_preflight.md`

If a parent directory for an allowed file does not exist, Codex may create that parent directory. Codex must not create or modify any other files.

## Forbidden Actions During Implementation

Codex must not:

- download datasets
- run real training
- write outputs
- write checkpoints
- inspect dataset directories recursively
- access protected paths
- access `.env`, `.env.*`, secrets, data, datasets, outputs, or checkpoints
- install packages
- access network resources from shell commands
- use `rg`
- use `rm -rf`
- create large files
- implement SNS augmentation
- run SNS evaluation
- run SNS perturbation evaluation
- run `git push`, `git pull`, or `git fetch`
- run `curl`, `wget`, `ssh`, `scp`, `rsync`, `pip`, `npm`, `yarn`, `pnpm`, `kaggle`, `huggingface-cli`, or `wandb`
- use danger-full-access
- bypass permissions

## Important Project Facts

- The project target outputs are class, mask/localization, family/provenance, and reason.
- Task 0020 introduced the pre-SNS integrated model smoke interface.
- Task 0021 introduced the pre-SNS unified local dataset manifest gate.
- This task is the last dry-run before a guarded training entrypoint.
- The dry-run must prove the integrated model can consume the unified manifest schema, route class/family/localization losses, build a tiny batch, and run a CPU-only forward/backward preflight.
- This task must not perform real training, write outputs, write checkpoints, implement SNS augmentation, or run SNS evaluation.

## Implementation Requirements

1. Create `src/cv_forensics/pre_sns_preflight.py`
   - Use Python standard library plus torch only where needed.
   - Validate preflight policies and summarize readiness.
   - Check class coverage, family coverage, tampered mask coverage, max sample limits, CPU-only policy, and loss routing.
   - Do not write files.

2. Create `configs/training/pre_sns_training_preflight.example.json`
   - The config must be safe and symbolic only.
   - It must not contain real paths or protected paths.
   - It must include guardrails:
     - `no_download: true`
     - `no_network: true`
     - `no_outputs: true`
     - `no_checkpoints: true`
     - `no_real_training: true`
     - `no_sns_augmentation: true`
   - It must include expected unified manifest schema and tiny preflight limits.

3. Create `scripts/agent/validate_pre_sns_training_preflight_config.py`
   - Use only Python standard library.
   - Validate `example_symbolic` and `approved_local_pre_sns_training_preflight`.
   - For approved local mode, require approval text exactly:
     `I_APPROVE_LOCAL_NON_SNS_PRE_SNS_TRAINING_PREFLIGHT`
   - Validate local manifest path under approved roots.
   - Do not read images or masks.
   - Do not write files.
   - Print `PRE_SNS_TRAINING_PREFLIGHT_CONFIG_OK` on success.

4. Create `scripts/training/run_pre_sns_training_preflight.py`
   - Validate config before loading anything.
   - Use CPU by default.
   - Use torch and PIL if available.
   - Read only explicitly listed image/mask paths in an approved manifest.
   - Do not scan directories.
   - Do not write outputs or checkpoints.
   - Run one tiny forward/backward preflight, not full training.
   - Do not run optimizer step unless documented as `dry_run_optimizer_step: false` by default. Prefer no optimizer step.
   - Print JSON summary with marker `PRE_SNS_TRAINING_PREFLIGHT_OK`.

5. Create `tests/test_pre_sns_training_preflight.py`
   - Use a standard library test harness.
   - Use temporary fixtures only.
   - Test example config, approved local config, unsafe paths, missing approval, missing class coverage, missing family coverage, missing tampered masks, `no_outputs` false, `no_checkpoints` false, and preflight summary.
   - If torch or PIL is missing, skip runner execution with a clear message.

6. Create `docs/pre_sns_training_preflight.md`
   - Explain this is the final dry-run gate before a guarded training entrypoint.
   - Explain no outputs, no checkpoints, no SNS augmentation, and no performance claim.
   - Include marker:
     `PRE_SNS_TRAINING_PREFLIGHT_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_pre_sns_training_preflight_config.py configs/training/pre_sns_training_preflight.example.json
```

```bash
python3 tests/test_pre_sns_training_preflight.py
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0022-pre-sns-training-preflight-dry-run.md
```

```bash
test -f src/cv_forensics/pre_sns_preflight.py
```

```bash
test -f configs/training/pre_sns_training_preflight.example.json
```

```bash
test -f scripts/agent/validate_pre_sns_training_preflight_config.py
```

```bash
test -f scripts/training/run_pre_sns_training_preflight.py
```

```bash
test -f tests/test_pre_sns_training_preflight.py
```

```bash
test -f docs/pre_sns_training_preflight.md
```

```bash
grep -q PRE_SNS_TRAINING_PREFLIGHT_OK docs/pre_sns_training_preflight.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- src/cv_forensics/pre_sns_preflight.py configs/training/pre_sns_training_preflight.example.json scripts/agent/validate_pre_sns_training_preflight_config.py scripts/training/run_pre_sns_training_preflight.py tests/test_pre_sns_training_preflight.py docs/pre_sns_training_preflight.md
```

Optional validation command:

```bash
pytest -q tests/test_pre_sns_training_preflight.py
```

## Acceptance Criteria

- Validator passes for example config.
- Standalone tests pass.
- `check_agent_changes` passes.
- Docs contain `PRE_SNS_TRAINING_PREFLIGHT_OK`.
- Preflight can run on temporary fixtures and print `PRE_SNS_TRAINING_PREFLIGHT_OK`.
- No dataset download, real training, network access, output writing, checkpoint writing, protected path access, SNS augmentation, or SNS evaluation occurs.

## Stop Condition

Stop after creating the task file. Report:

- task file path
- short summary
- git status
- exact git add command
- exact git commit command
