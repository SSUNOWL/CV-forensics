# Task 0003: Agent Wrapper Smoke Test

## Role of Claude Code

You are Claude Code, the implementation worker. Codex is the supervisor and reviewer.

This task verifies that the safe wrapper can delegate a tiny implementation task to Claude Code.

## Files Claude May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0003-agent-wrapper-smoke-test.md`

## Files Claude May Modify

Claude may create or edit only this file:

- `docs/agent_wrapper_smoke_result.md`

Do not create or modify any other file.

## Forbidden Actions

Do not:

- Download datasets
- Train a model
- Install packages
- Access network resources from shell commands
- Access `.env`, `.env.*`, secrets, data, datasets, outputs, or checkpoints
- Modify unrelated files
- Run `git push`
- Create large files
- Use `rm -rf`
- Use `rg`

## Implementation Requirements

Create `docs/agent_wrapper_smoke_result.md`.

The file must contain:

- The marker string `AGENT_WRAPPER_SMOKE_OK`
- A one-paragraph Korean summary saying that Claude implemented a Codex-created task through the safe wrapper
- A short checklist confirming that no datasets, training code, secrets, outputs, or checkpoints were touched

## Validation Commands

Run:

```bash
test -f docs/agent_wrapper_smoke_result.md
```

Run:

```bash
grep -q AGENT_WRAPPER_SMOKE_OK docs/agent_wrapper_smoke_result.md
```

Check changed files:

```bash
git status --short --untracked-files=all
```

Check the smoke result diff:

```bash
git diff -- docs/agent_wrapper_smoke_result.md
```

## Acceptance Criteria

- `docs/agent_wrapper_smoke_result.md` exists.
- It contains `AGENT_WRAPPER_SMOKE_OK`.
- Changed files are limited to:
  - `docs/agent_wrapper_smoke_result.md`
- No secrets, data, datasets, outputs, checkpoints, or network resources are accessed.
- No dataset download, model training, package installation, large file creation, or unrelated file modification occurs.

## Stop Condition

Stop after implementation and validation. Report:

- Files changed
- Validation commands run and their results
- Confirmation that forbidden paths and actions were not touched
