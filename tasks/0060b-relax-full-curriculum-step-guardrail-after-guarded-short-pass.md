# Task: 0060b Relax Full Curriculum Step Guardrail After Guarded Short Pass

## Task Title

Relax the 0060b full-curriculum per-phase step guardrail for approved medium/full runs while preserving safety limits.

## Role

Codex-only task writer, implementation worker, reviewer, and limited repair manager for a targeted 0060b config-validator guardrail update.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0060b-snsaug-v2-full-curriculum-finetune.md`
- `tasks/0060b-fix-full-curriculum-actual-training-branch.md`
- `configs/training/snsaug_v2_full_curriculum_finetune.example.json`
- `scripts/agent/validate_snsaug_v2_full_curriculum_finetune_config.py`
- `scripts/training/run_snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
- `tests/test_snsaug_v2_full_curriculum_finetune.py`
- `docs/snsaug_v2_full_curriculum_finetune.md`

## Files Codex May Modify

- `tasks/0060b-relax-full-curriculum-step-guardrail-after-guarded-short-pass.md`
- `configs/training/snsaug_v2_full_curriculum_finetune.example.json`
- `scripts/agent/validate_snsaug_v2_full_curriculum_finetune_config.py`
- `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
- `tests/test_snsaug_v2_full_curriculum_finetune.py`
- `docs/snsaug_v2_full_curriculum_finetune.md`

## Forbidden Actions

- Do not run training in this task.
- Do not run long full training.
- Do not use validation samples for training.
- Do not use validation failures directly for training.
- Do not access network resources from shell commands.
- Do not download datasets, assets, or checkpoints.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or protected directories.
- Do not inspect protected directories recursively.
- Do not write generated artifacts, logs, reports, or checkpoints inside the repository.
- Do not modify existing model checkpoints.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not use `rg`.
- Do not use Claude Code.

## Important Project Facts

- 0060b tiny actual run passed.
- 0060b guarded short actual run with 30 steps per phase passed.
- The next goal is a medium run for more meaningful comparison.
- The current validator only allows:
  - `max_steps_per_phase` in `1..30`
  - `phase_1_max_steps` in `1..30`
  - `phase_2_max_steps` in `1..30`
  - `phase_3_max_steps` in `1..30`
- We need larger but still bounded approved runs.
- Existing guardrails must remain:
  - no network
  - no download
  - output outside repository
  - train-only manifest
  - no evaluation root used as training input
- The documentation marker must remain:
  - `SNSAUG_V2_FULL_CURRICULUM_FINETUNE_OK`

## Implementation Requirements

1. Add a config field:
   - `max_allowed_steps_per_phase`
2. For approved full-curriculum configs with valid approval text, allow:
   - `max_steps_per_phase <= max_allowed_steps_per_phase`
   - `phase_1_max_steps <= max_allowed_steps_per_phase`
   - `phase_2_max_steps <= max_allowed_steps_per_phase`
   - `phase_3_max_steps <= max_allowed_steps_per_phase`
3. Set the default `max_allowed_steps_per_phase` to `500`.
4. Reject `max_allowed_steps_per_phase > 500` unless:
   - `allow_long_run_after_medium_pass == true`
   - and `max_allowed_steps_per_phase <= 2000`
5. Reject any effective phase step value greater than `max_allowed_steps_per_phase`.
6. Reject `5000` steps per phase in all cases.
7. Keep unit tests using tiny values no larger than `30` for actual runner execution.
8. Add validator tests covering:
   - `30` passes
   - `150` passes with `max_allowed_steps_per_phase=500`
   - `500` passes with `max_allowed_steps_per_phase=500`
   - `501` fails unless `max_allowed_steps_per_phase >= 501`
   - `5000` always fails
9. Preserve existing guardrail tests:
   - no network
   - no download
   - output outside repository
   - train-only manifest
   - no evaluation root used as training input
10. Update docs to explain the medium-run guardrail and the long-run override.
11. Do not launch training during validation.

## Validation Commands

```bash
python3 scripts/agent/validate_snsaug_v2_full_curriculum_finetune_config.py configs/training/snsaug_v2_full_curriculum_finetune.example.json
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_full_curriculum_finetune.py
```

```bash
python3 scripts/training/run_snsaug_v2_full_curriculum_finetune.py --help
```

```bash
grep -q SNSAUG_V2_FULL_CURRICULUM_FINETUNE_OK docs/snsaug_v2_full_curriculum_finetune.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0060b-relax-full-curriculum-step-guardrail-after-guarded-short-pass.md
```

## Acceptance Criteria

- Config validation accepts `30` phase steps.
- Config validation accepts `150` phase steps when `max_allowed_steps_per_phase=500`.
- Config validation accepts `500` phase steps when `max_allowed_steps_per_phase=500`.
- Config validation rejects `501` phase steps when `max_allowed_steps_per_phase` is not raised.
- Config validation accepts `501` only when `max_allowed_steps_per_phase >= 501` and the long-run override is valid.
- Config validation always rejects `5000`.
- Existing no-network, no-download, output-root, train-only, and evaluation-root leakage guardrails remain intact.
- Tests still run only tiny training branches and do not launch medium/full training.
- Documentation explains the new guardrail.
- All changed files are within the allowed modify list.

## Stop Condition

Stop immediately if completing this task would require:

- running training,
- running long full training,
- using validation/test/evaluation data as training data,
- using validation failures directly for training,
- downloading datasets/assets/checkpoints,
- installing packages,
- network access,
- protected path access,
- writing outputs/checkpoints inside the repository,
- changing files outside the allowed modify list,
- modifying existing checkpoints,
- or changing the 0058c benchmark or 0058e analysis outputs.
