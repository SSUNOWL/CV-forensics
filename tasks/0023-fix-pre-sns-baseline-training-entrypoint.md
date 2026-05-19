# Task 0023 Fix: Pre-SNS Baseline Training Entrypoint Change-Check Section

## Task title
Fix the task 0023 change-check section heading so `check_agent_changes.py` can parse the allowed file list.

## Role
Codex is the limited repair manager and implementation worker in Codex-only mode. Codex must apply only the narrow repair allowed by this fix task after this fix task is committed and after the user explicitly says to implement.

## Files Codex may read
- AGENTS.md
- CLAUDE.md
- docs/project_brief.md
- docs/project_contract.md
- configs/project_contract.json
- docs/agent_workflow.md
- tasks/0023-pre-sns-baseline-training-entrypoint.md
- tasks/0023-fix-pre-sns-baseline-training-entrypoint.md
- configs/training/pre_sns_baseline_train.example.json
- scripts/agent/validate_pre_sns_baseline_train_config.py
- scripts/training/train_pre_sns_baseline.py
- tests/test_pre_sns_baseline_train_gate.py
- docs/pre_sns_baseline_training.md
- scripts/agent/check_agent_changes.py

## Files Codex May Modify
- tasks/0023-pre-sns-baseline-training-entrypoint.md
- configs/training/pre_sns_baseline_train.example.json
- scripts/agent/validate_pre_sns_baseline_train_config.py
- scripts/training/train_pre_sns_baseline.py
- tests/test_pre_sns_baseline_train_gate.py
- docs/pre_sns_baseline_training.md

The intended repair is limited to `tasks/0023-pre-sns-baseline-training-entrypoint.md`. The implementation files are listed only because they are existing task 0023 worktree changes and `check_agent_changes.py` checks the full worktree. Do not edit implementation files unless a validation command proves a small direct repair is required.

## Forbidden actions
Codex must not:
- download datasets
- run real training on real data
- inspect dataset directories recursively
- access .env, .env.*, secrets, data, datasets, outputs, or checkpoints
- install packages
- access network resources from shell commands
- run git push, git pull, or git fetch
- use rg
- use rm -rf
- create large files
- implement SNS augmentation
- run SNS evaluation
- use danger-full-access
- bypass permissions

## Important project facts
- Task 0023 implementation currently passes its config validator and standalone tests.
- Task 0023 failed review because `python3 scripts/agent/check_agent_changes.py tasks/0023-pre-sns-baseline-training-entrypoint.md` could not parse the task file.
- The exact failure was:
  - `ERROR: Expected section '## Files Claude May Modify' or '## Files Codex May Modify' not found in task file.`
- The root cause is the lowercase heading `## Files Codex may modify` in `tasks/0023-pre-sns-baseline-training-entrypoint.md`.
- `check_agent_changes.py` requires the exact heading `## Files Codex May Modify` or `## Files Claude May Modify`.
- This repair must preserve the original task scope and allowed implementation files.

## Implementation requirements
1. Update `tasks/0023-pre-sns-baseline-training-entrypoint.md`.
   - Change the heading `## Files Codex may modify` to exactly:
     - `## Files Codex May Modify`
   - Do not change the allowed file list under that heading.
   - Do not change task 0023 implementation requirements, validation commands, acceptance criteria, or stop condition unless required only to preserve meaning after the heading repair.

2. Do not modify task 0023 implementation files unless a validation command identifies a separate small direct issue.
   - Existing implementation files are expected to remain as-is.
   - If they are touched, explain why in the review.

3. Run the task 0023 safe validation commands after the repair.
   - The main required fix validation is `check_agent_changes.py` against task 0023.
   - The config validator and standalone tests should remain passing.

## Validation commands

```bash
python3 scripts/agent/validate_pre_sns_baseline_train_config.py configs/training/pre_sns_baseline_train.example.json
```

```bash
python3 tests/test_pre_sns_baseline_train_gate.py
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0023-pre-sns-baseline-training-entrypoint.md
```

```bash
test -f tasks/0023-pre-sns-baseline-training-entrypoint.md
```

```bash
test -f configs/training/pre_sns_baseline_train.example.json
```

```bash
test -f scripts/agent/validate_pre_sns_baseline_train_config.py
```

```bash
test -f scripts/training/train_pre_sns_baseline.py
```

```bash
test -f tests/test_pre_sns_baseline_train_gate.py
```

```bash
test -f docs/pre_sns_baseline_training.md
```

```bash
grep -q PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK docs/pre_sns_baseline_training.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- tasks/0023-pre-sns-baseline-training-entrypoint.md configs/training/pre_sns_baseline_train.example.json scripts/agent/validate_pre_sns_baseline_train_config.py scripts/training/train_pre_sns_baseline.py tests/test_pre_sns_baseline_train_gate.py docs/pre_sns_baseline_training.md
```

Optional validation command:

```bash
pytest -q tests/test_pre_sns_baseline_train_gate.py
```

## Acceptance criteria
- `tasks/0023-pre-sns-baseline-training-entrypoint.md` contains the exact heading `## Files Codex May Modify`.
- The allowed implementation file list in task 0023 is preserved.
- `python3 scripts/agent/validate_pre_sns_baseline_train_config.py configs/training/pre_sns_baseline_train.example.json` passes.
- `python3 tests/test_pre_sns_baseline_train_gate.py` passes.
- `python3 scripts/agent/check_agent_changes.py tasks/0023-pre-sns-baseline-training-entrypoint.md` passes.
- `docs/pre_sns_baseline_training.md` contains `PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK`.
- Changed files remain limited to task 0023 repair plus the existing task 0023 implementation files.
- No dataset download, real training, network access, protected path access, output writing, checkpoint writing, or SNS augmentation occurs.

## Stop condition
Stop after applying the repair, running safe validation commands, and reviewing the result in Korean as PASS or NEEDS_FIX. Do not commit automatically.
