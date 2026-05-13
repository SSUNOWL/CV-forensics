# Task 0002: Safe Claude Runner Wrapper

## Role of Claude Code

You are Claude Code, the implementation worker for this repository. Codex is the supervisor and reviewer.

Your job is to implement a safe local wrapper that lets Codex delegate future task files to Claude Code in a controlled, auditable way.

Do not reinterpret the project scope beyond `docs/project_brief.md`, `docs/project_contract.md`, `configs/project_contract.json`, and this task file.

## Files Claude May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `tasks/0002-claude-runner-wrapper.md`
- Any file you are explicitly allowed to create or modify in this task, if it already exists

You may also run basic repository status and diff checks limited to the allowed files:

- `pwd`
- `git status --short`
- `git diff -- scripts/agent/run_claude_task.sh scripts/agent/check_agent_changes.py docs/agent_workflow.md tests/test_agent_change_check.py`

## Files Claude May Modify

Claude may create or edit only these files:

- `scripts/agent/run_claude_task.sh`
- `scripts/agent/check_agent_changes.py`
- `docs/agent_workflow.md`
- `tests/test_agent_change_check.py`

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
- Run network-capable shell commands such as `curl`, `wget`, `ssh`, `scp`, `rsync`, package managers, or git remote operations
- Use `--permission-mode bypassPermissions`
- Use `--dangerously-skip-permissions`
- Use `/snap/bin/codex`
- Use `/home/rlatjswo/.npm-global/bin/codex`
- Use `rm -rf`

## Implementation Requirements

Implement a controlled Claude Code task runner.

1. Create or update `scripts/agent/run_claude_task.sh`
   - Use POSIX-compatible shell or Bash.
   - Accept a task file path as the first argument.
   - Refuse to run if the task file argument is missing.
   - Refuse to run if the task file does not exist.
   - Resolve the repository root from the script location, not from the caller's current directory.
   - Create a timestamped log directory under `.agent-runs/`.
   - Run Claude Code in non-interactive print mode.
   - Use the Claude binary from `CLAUDE_BIN` when provided, otherwise default to `/home/rlatjswo/.local/bin/claude`.
   - Pass a strict prompt requiring Claude to:
     - read `docs/project_brief.md`;
     - read the provided task file;
     - follow only the task file;
     - obey allowed files and forbidden actions;
     - avoid datasets, checkpoints, secrets, training, package installation, and network access.
   - Restrict Claude tools using `--tools`, `--allowedTools`, and `--disallowedTools`.
   - Do not use `--permission-mode bypassPermissions`.
   - Do not use `--dangerously-skip-permissions`.
   - Capture stdout and stderr into files inside the run log directory.
   - Save the exact task file path and a copy of the task file into the run log directory.
   - After Claude exits, check changed files against the allowed list declared in the task file.
   - Run validation commands listed in the task file when possible.
   - Print a concise summary including:
     - run log directory;
     - Claude exit code;
     - change-check result;
     - validation result;
     - final wrapper exit status.

2. Create or update `scripts/agent/check_agent_changes.py`
   - Use pure Python standard library only.
   - Do not import third-party packages.
   - Accept the task file path as an argument.
   - Optionally accept `--repo-root`.
   - Parse the task file's allowed modification list from a clearly documented section such as `## Files Claude May Modify`.
   - Use `git status --short` to identify changed files.
   - Treat modified, added, deleted, renamed, copied, and untracked files as changes.
   - Fail if any changed file is outside the allowed list declared in the task file.
   - Print a concise success or failure message.
   - Exit with status code 0 on success and non-zero on failure.
   - Do not access `.env`, secrets, data, datasets, outputs, or checkpoints.

3. Create or update `docs/agent_workflow.md`
   - Document the Codex supervisor and Claude implementation-worker workflow.
   - Document the required invocation pattern:

     ```bash
     CLAUDE_BIN=/home/rlatjswo/.local/bin/claude scripts/agent/run_claude_task.sh tasks/<task-file>.md
     ```

   - Document the safety rules:
     - no dataset download;
     - no model training;
     - no package installation;
     - no `.env`, secrets, data, datasets, outputs, or checkpoints access;
     - no git push;
     - no large files;
     - no network access from shell commands.
   - Document how allowed files and validation commands are declared in task files.

4. Create or update `tests/test_agent_change_check.py`
   - Use only Python standard library plus pytest-style assertions.
   - Do not install pytest.
   - Test the allowed-file parser and/or the changed-file checker behavior in a temporary git repository.
   - Keep tests local and lightweight.

## Validation Commands

Run:

```bash
python3 scripts/agent/check_agent_changes.py tasks/0002-claude-runner-wrapper.md
```

If pytest is already available without installing anything, also run:

```bash
pytest -q tests/test_agent_change_check.py
```

Check the changed files:

```bash
git status --short
git diff -- scripts/agent/run_claude_task.sh scripts/agent/check_agent_changes.py docs/agent_workflow.md tests/test_agent_change_check.py
```

Do not use `python`; this repository prefers `python3`.

## Acceptance Criteria

- `scripts/agent/run_claude_task.sh` accepts a task file path and refuses missing or nonexistent task files.
- The wrapper creates timestamped logs under `.agent-runs/`.
- The wrapper runs Claude Code in non-interactive print mode with restricted tools.
- The wrapper captures stdout and stderr.
- The wrapper checks changed files after Claude finishes.
- The wrapper fails when changed files are outside the task file's allowed modification list.
- The wrapper runs validation commands listed in the task file when possible.
- `scripts/agent/check_agent_changes.py` uses only the Python standard library.
- Documentation explains the Codex -> Claude Code -> Codex workflow and safety guardrails.
- Tests are lightweight and do not require package installation.
- Changed files are limited to:
  - `scripts/agent/run_claude_task.sh`
  - `scripts/agent/check_agent_changes.py`
  - `docs/agent_workflow.md`
  - `tests/test_agent_change_check.py`
- No secrets, data, outputs, checkpoints, datasets, or network resources are accessed.
- No dataset download, model training, package installation, large file creation, or unrelated file modification occurs.

## Stop Condition

Stop after implementation and validation. Report:

- Files changed
- Validation commands run and their results
- Any skipped optional validation, with the reason
- Confirmation that forbidden paths and actions were not touched
