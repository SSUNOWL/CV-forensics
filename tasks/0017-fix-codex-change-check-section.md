# Task 0017: Fix Codex-Only Change Checker Section

## Task Title

Fix `check_agent_changes.py` so Codex-only task files can use `## Files Codex May Modify`.

## Role

Codex is the implementation worker in Codex-only mode. Codex must implement only the files allowed by this task file after the task file is committed and after the user explicitly says to implement.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `scripts/agent/check_agent_changes.py`
- `tests/test_agent_change_check.py`
- `tasks/0016-pre-sns-baseline-evaluation-report.md`
- `tasks/0017-fix-codex-change-check-section.md`

## Files Codex May Modify

- `scripts/agent/check_agent_changes.py`
- `tests/test_agent_change_check.py`

## Files Claude May Modify

- `scripts/agent/check_agent_changes.py`
- `tests/test_agent_change_check.py`

The `## Files Claude May Modify` section is included only for compatibility with the current checker while this fix is being implemented.

## Forbidden Actions

Codex must not:

- download datasets
- train a model
- run real optimization
- run real evaluation
- inspect actual dataset directories recursively
- read real images
- read real masks
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
- use `danger-full-access`
- bypass permissions

## Important Project Facts

- The repository now uses Codex-only implementation mode when Claude Code is unavailable.
- Codex-only task files use `## Files Codex May Modify`.
- Existing agent tooling currently parses only `## Files Claude May Modify`.
- Task 0016 implementation passed its validator and tests, but `python3 scripts/agent/check_agent_changes.py tasks/0016-pre-sns-baseline-evaluation-report.md` failed because the checker could not find `## Files Claude May Modify`.
- The fix must preserve compatibility with older task files that still use `## Files Claude May Modify`.
- This task is tooling-only and must not touch dataset, model, training, report, output, checkpoint, or SNS implementation code.

## Implementation Requirements

1. Update `scripts/agent/check_agent_changes.py`
   - Keep existing behavior for `## Files Claude May Modify`.
   - Add support for `## Files Codex May Modify`.
   - Prefer an exact matching section if one exists.
   - If both sections exist, parse a stable union or otherwise handle them without duplicate false positives.
   - Preserve existing validation command parsing and safety checks.
   - Preserve existing exit codes as much as practical.
   - Improve the missing-section error so it names both accepted section headers.

2. Update `tests/test_agent_change_check.py`
   - Add coverage that a task file with only `## Files Codex May Modify` is accepted.
   - Add coverage that a task file with only `## Files Claude May Modify` remains accepted.
   - Add coverage that a task file with both sections is accepted.
   - Add coverage that missing both sections still fails.
   - Keep tests standard-library-only if the existing file is standard-library-only.

3. Do not modify task 0016 during this fix.
   - The repaired checker should make the existing task 0016 section layout valid.
   - After this fix is committed, rerun the task 0016 change check as part of the 0016 repair review.

## Validation Commands

```bash
python3 tests/test_agent_change_check.py
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0017-fix-codex-change-check-section.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0016-pre-sns-baseline-evaluation-report.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- scripts/agent/check_agent_changes.py tests/test_agent_change_check.py
```

Optional validation command:

```bash
pytest -q tests/test_agent_change_check.py
```

## Acceptance Criteria

- `python3 tests/test_agent_change_check.py` passes.
- `python3 scripts/agent/check_agent_changes.py tasks/0017-fix-codex-change-check-section.md` passes when only fix-task files are changed.
- `python3 scripts/agent/check_agent_changes.py tasks/0016-pre-sns-baseline-evaluation-report.md` recognizes `## Files Codex May Modify`.
- Existing `## Files Claude May Modify` task support remains intact.
- Changed files for this fix are limited to:
  - `scripts/agent/check_agent_changes.py`
  - `tests/test_agent_change_check.py`
- No dataset download, model training, package installation, network access, real evaluation, real image reading, real mask reading, output writing, checkpoint writing, SNS augmentation, or protected path access occurs.

## Stop Condition

Stop after implementing and reviewing this fix task. Report:

- changed files
- validation results
- protected path status
- exact git add command
- exact git commit command
