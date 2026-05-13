# Task 0001: Agent Smoke Test Project Contract

## Role of Claude Code

You are Claude Code, the implementation worker for this repository. Codex is the supervisor and reviewer.

Your job is to verify the Codex -> Claude Code -> Codex review workflow by creating a small project contract artifact set. This task must not touch datasets, checkpoints, secrets, training code, or network resources.

Do not reinterpret the project scope beyond `docs/project_brief.md` and this task file.

## Files Claude May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `tasks/0001-agent-smoke-test.md`
- Any file you are explicitly allowed to create or modify in this task, if it already exists

You may also run basic repository status checks such as:

- `pwd`
- `git status --short`
- `git diff -- docs/project_contract.md configs/project_contract.json scripts/agent/validate_project_contract.py tests/test_project_contract.py`

## Files Claude May Modify

Claude may create or edit only these files:

- `docs/project_contract.md`
- `configs/project_contract.json`
- `scripts/agent/validate_project_contract.py`
- `tests/test_project_contract.py`

If a parent directory for one of those files does not exist, you may create that parent directory. Do not create any other files.

## Forbidden Actions

Do not:

- Download datasets
- Train a model
- Install packages
- Access network resources
- Access `.env`, `.env.*`, secrets, data, datasets, outputs, or checkpoints
- Inspect protected directories or protected files recursively
- Modify unrelated files
- Run `git push`
- Create large files
- Use `--permission-mode bypassPermissions`
- Use `--dangerously-skip-permissions`
- Use `/snap/bin/codex`
- Use `/home/rlatjswo/.npm-global/bin/codex`
- Use `rm -rf`

## Implementation Requirements

Create a small project contract that matches `docs/project_brief.md`.

1. Create `docs/project_contract.md`
   - Summarize the agreed project goal.
   - Include target outputs: class, mask, family, and reason.
   - Include the two main datasets and their roles.
   - Include the high-level architecture.
   - Include the implementation stages.
   - Include evaluation metrics.
   - Include first risks and guardrails.

2. Create `configs/project_contract.json`
   - Use valid JSON.
   - Include at least these top-level keys:
     - `project_title`
     - `final_goal`
     - `target_outputs`
     - `datasets`
     - `architecture`
     - `implementation_stages`
     - `evaluation_metrics`
     - `first_risks`
     - `guardrails`
   - The content must reflect `docs/project_brief.md`.

3. Create `scripts/agent/validate_project_contract.py`
   - Use pure Python standard library only.
   - Do not import third-party packages.
   - Accept the JSON contract path as a command-line argument.
   - Load and validate the JSON contract.
   - Check that required top-level keys exist.
   - Check that target outputs include class, mask, family, and reason.
   - Check that datasets include Community Forensics-Small and SID-Set.
   - Check that metrics include classification, Macro-F1, mask IoU, family accuracy, robustness drop, latency or FPS, and localization activation recall.
   - Exit with status code 0 on success and non-zero on failure.
   - Print a concise success or failure message.

4. Optionally create `tests/test_project_contract.py`
   - Create this only if it can be done without installing packages.
   - The test may use `pytest` conventions, but do not install pytest.
   - It should check the JSON contract and/or validator behavior.

## Validation Commands

Run:

```bash
python3 scripts/agent/validate_project_contract.py configs/project_contract.json
```

If pytest is already available without installing anything, also run:

```bash
pytest -q tests/test_project_contract.py
```

Check the changed files:

```bash
git status --short
git diff -- docs/project_contract.md configs/project_contract.json scripts/agent/validate_project_contract.py tests/test_project_contract.py
```

Do not use `python`; this repository prefers `python3`.

## Acceptance Criteria

- `python3 scripts/agent/validate_project_contract.py configs/project_contract.json` passes.
- If pytest already exists, `pytest -q tests/test_project_contract.py` passes.
- `git status --short` and `git diff` show changes only in the allowed files.
- The contract matches `docs/project_brief.md`.
- No secrets, data, outputs, checkpoints, datasets, or network resources are accessed.
- No dataset download, model training, package installation, or unrelated file modification occurs.

## Stop Condition

Stop after implementation and validation. Report:

- Files changed
- Validation commands run and their results
- Any skipped optional validation, with the reason
- Confirmation that forbidden paths and actions were not touched

