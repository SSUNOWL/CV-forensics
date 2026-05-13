# Task 0001: Agent Smoke Test

## Role

You are Claude Code acting as the implementation worker.
Read `CLAUDE.md` and `docs/project_brief.md` before making changes.
Follow this task file exactly. Do not expand scope.

## Objective

Verify that Claude Code can implement from a Codex-created task file without touching data, checkpoints, secrets, or training code.

This is a workflow smoke test, not a modeling task.

## Allowed Files

You may create or edit only the following files:

- `docs/project_contract.md`
- `configs/project_contract.json`
- `scripts/agent/validate_project_contract.py`
- `tests/test_project_contract.py`

Do not modify any other file.

## Hard Prohibitions

You must not:

- download datasets
- train a model
- install packages
- access `.env`, secrets, `data`, `datasets`, `outputs`, or `checkpoints`
- modify unrelated files
- run `git push`
- create large files
- use network access

## Required Implementation

Implement the following minimal artifacts.

### 1. Project Contract Document

Create `docs/project_contract.md`.

Purpose:

- summarize the agreed project goal from `docs/project_brief.md`
- keep the scope narrow and concrete

Required sections:

- project goal
- target outputs
- datasets
- implementation stages
- evaluation metrics
- non-goals for this smoke test

The document must match `docs/project_brief.md` and must not introduce new project scope.

### 2. JSON Contract

Create `configs/project_contract.json`.

It must contain, at minimum:

- project title
- final goal
- target outputs
- datasets
- implementation stages
- evaluation metrics

Use a simple, explicit schema that is easy to validate in pure Python.
The contents must align with `docs/project_brief.md`.

### 3. Pure-Python Validator

Create `scripts/agent/validate_project_contract.py`.

Requirements:

- standard library only
- no external dependencies
- accepts the JSON file path as a command-line argument
- validates required top-level fields and basic structure
- prints a short success message and exits with code `0` on success
- prints a clear error and exits non-zero on failure

The validator should check that:

- required keys exist
- lists are non-empty where expected
- the contract reflects the brief at a high level

Do not over-engineer the validator.

### 4. Optional Pytest Test

You may create `tests/test_project_contract.py` if `pytest` already exists in the environment.

If you create it:

- keep it small
- use it only to validate the JSON contract shape/content
- do not add new dependencies

If `pytest` is not available, skip this file.

## Validation Commands

Run:

```bash
python scripts/agent/validate_project_contract.py configs/project_contract.json
```

If `pytest` exists and you created the test file, also run:

```bash
pytest -q tests/test_project_contract.py
```

Also verify:

```bash
git diff -- docs/project_contract.md configs/project_contract.json scripts/agent/validate_project_contract.py tests/test_project_contract.py
git diff --name-only
```

The final diff must contain only allowed files.

## Acceptance Criteria

All of the following must be true:

1. `python scripts/agent/validate_project_contract.py configs/project_contract.json` passes.
2. If `pytest` exists, `pytest -q tests/test_project_contract.py` passes.
3. `git diff` contains only the allowed files.
4. The contract matches `docs/project_brief.md`.
5. No secrets, data, outputs, checkpoints, or network access are touched.

## Execution Notes

- Keep the implementation minimal and deterministic.
- Prefer straightforward JSON and validation logic over abstractions.
- Do not inspect or use protected directories or files.
- Stop after completing this task and reporting the results.
