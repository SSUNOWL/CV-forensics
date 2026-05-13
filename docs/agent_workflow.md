# Agent Workflow: Codex Supervisor + Claude Code Implementation Worker

This document describes the Codex → Claude Code → Codex review cycle used in
this repository.

---

## Roles

| Role | Tool | Responsibility |
|------|------|----------------|
| Supervisor / Reviewer | Codex | Creates task files; reviews results via `git diff` and test output |
| Implementation Worker | Claude Code | Reads task file; implements only what it specifies; runs validation |

Codex does not implement directly.  Claude Code does not reinterpret project
scope beyond the task file.

---

## Invocation Pattern

Codex delegates a task to Claude Code via the runner wrapper:

```bash
CLAUDE_BIN=/home/rlatjswo/.local/bin/claude \
  scripts/agent/run_claude_task.sh tasks/<task-file>.md
```

The `CLAUDE_BIN` environment variable overrides the default binary path
(`~/.local/bin/claude`).  The task file path is relative to the repository
root.

---

## What the Runner Does

`scripts/agent/run_claude_task.sh` performs these steps in order:

1. **Validates arguments** — refuses to run if the task file argument is
   missing or the file does not exist.
2. **Resolves the repository root** from the script's own location, not the
   caller's working directory.
3. **Creates a timestamped log directory** under `.agent-runs/`:
   ```
   .agent-runs/
     <YYYYMMDDTHHMMSSZ>_<task-name>/
       task_file_path.txt   # absolute path to the task file
       task_file.md         # verbatim copy of the task file
       claude_stdout.log    # Claude Code stdout
       claude_stderr.log    # Claude Code stderr
       summary.txt          # concise run summary
   ```
4. **Runs Claude Code** in non-interactive print mode (`-p`) with:
   - `--permission-mode default` (never `bypassPermissions`)
   - `--tools` listing the available tool families (broad availability list)
   - `--allowedTools` restricting Claude to an explicit narrow subset (see below)
   - `--disallowedTools` blocking network-capable tools and protected paths
   - A strict prompt requiring Claude to follow only the task file
5. **Checks changed files** (`scripts/agent/check_agent_changes.py`) against
   the task file's declared allowed list.
6. **Runs validation commands** from the task file's `## Validation Commands`
   section.  Each command is **allowlist-checked** before execution — unsafe
   commands are rejected, not run.  Commands whose binary is not installed are
   skipped (SKIP), not failed.
7. **Prints a concise summary** (log dir, Claude exit code, change-check
   result, validation result, final wrapper exit status).

---

## Claude Tool Permissions — Three-Flag Policy

All three Claude CLI tool-restriction flags are used together:

| Flag | Variable | Purpose |
|------|----------|---------|
| `--tools` | `AVAILABLE_TOOLS` | Declares all tool families Claude may potentially invoke |
| `--allowedTools` | `ALLOWED_TOOLS` | Restricts Claude to an explicit narrow subset |
| `--disallowedTools` | `DISALLOWED_TOOLS` | Belt-and-suspenders block for dangerous patterns |

### Claude Bash Permissions — Intentionally Limited to Status/Diff

Claude's Bash access via `--allowedTools` is restricted to safe repository
introspection only.  File creation and editing use `Read`, `Write`, `Edit`,
and `MultiEdit` tools — Claude does not need broad Bash patterns to implement
tasks.  Validation commands (`python3`, `pytest`, `grep`, etc.) run **after
Claude exits** through the wrapper's own safe runner, not through Claude's
Bash tool.

Only these Bash patterns are permitted in `--allowedTools`:

| Pattern | Purpose |
|---------|---------|
| `Bash(git status --short*)` | Check working-tree state |
| `Bash(git diff -- *)` | Inspect specific file diffs |

The `--disallowedTools` list explicitly blocks: `curl`, `wget`, `ssh`, `scp`,
`rsync`, all package managers (`pip`, `npm`, `yarn`, `pnpm`, `kaggle`,
`huggingface-cli`, `wandb`), all git remote operations (`push`, `pull`,
`fetch`, `clone`), dangerous deletion (`rm -rf`, `rm -r`, `rm -f`), and
direct access to protected paths.

---

## Validation Command Allowlist

Validation commands listed in a task file's `## Validation Commands` section
are **not executed directly** and are **not run through Claude's Bash tool**.
After Claude exits, the wrapper passes each command to
`check_agent_changes.py --check-cmd` for classification before running it.
No `eval` is used at any point.

### Accepted command families

| Family | Example |
|--------|---------|
| `python3 <script> [args...]` | `python3 scripts/validate.py configs/x.json` |
| `pytest -q <test-file>` | `pytest -q tests/test_foo.py` |
| `python3 -m pytest -q <test-file>` | `python3 -m pytest -q tests/test_foo.py` |
| `test -f <path>` | `test -f scripts/agent/run_claude_task.sh` |
| `grep -q <pattern> <path>` | `grep -q "key" configs/contract.json` |
| `git status --short` | `git status --short` |
| `git status --short --untracked-files=all` | `git status --short --untracked-files=all` |
| `git diff -- <paths...>` | `git diff -- scripts/foo.py docs/bar.md` |

### Rejected commands

The following are always rejected regardless of which section they appear in:

- Commands containing shell control or injection syntax: `;`, `|`, `&`, `` ` ``,
  `$(`, `>`, `<`
- Commands whose first word is a forbidden binary: `curl`, `wget`, `ssh`,
  `scp`, `rsync`, `pip`, `pip3`, `npm`, `yarn`, `pnpm`, `kaggle`,
  `huggingface-cli`, `wandb`, `bash`, `sh`
- Commands starting with forbidden git sub-commands: `git push`, `git pull`,
  `git fetch`, `git clone`, `git remote`
- Commands containing dangerous deletion strings: `rm -rf`, `rm -r `
- Commands referencing protected paths: `.env`, `.env.*`, `secrets/...`,
  `data/...`, `./data/...`, `datasets/...`, `outputs/...`, `checkpoints/...`
  (detection is path-aware — relative paths like `data/foo.py` are rejected
  as well as absolute paths like `/data/foo.py`)
- Commands not matching any accepted family (reject-by-default)

The wrapper prints `ACCEPT`, `REJECT`, or `SKIP` for every validation command
so the run log shows exactly what was and was not executed.

### Task file guidance

Task files should declare only **simple** validation commands — one command
per line, no shell pipelines, no variable expansion, no redirection.
Commands that would be rejected by the allowlist should not appear in task
files.  If a more complex check is needed, wrap it in a `python3` script and
call that script instead.

---

## Safety Rules

All agent tasks must obey these guardrails:

- **No dataset download** — do not run `wget`, `curl`, or any command that
  fetches remote data.
- **No model training** — do not launch training scripts or GPU jobs.
- **No package installation** — do not run `pip install`, `conda install`,
  `npm install`, or similar.
- **No protected path access** — do not read or write `.env`, `.env.*`,
  `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- **No `git push`** — changes stay local until Codex accepts them.
- **No large file creation** — keep artifacts small and reversible.
- **No network access from shell commands** — do not use `curl`, `wget`,
  `ssh`, `scp`, `rsync`, or any package-manager network operation.

---

## Task File Format

Task files live under `tasks/` and are Markdown documents.  Two sections
control the runner's behaviour:

### `## Files Claude May Modify`

Declares the exact set of files Claude Code may create or edit:

```markdown
## Files Claude May Modify

- `scripts/agent/run_claude_task.sh`
- `scripts/agent/check_agent_changes.py`
- `docs/agent_workflow.md`
- `tests/test_agent_change_check.py`
```

After Claude Code exits, `check_agent_changes.py` runs
`git status --short --untracked-files=all` and fails if any changed file is
not in this list.

### `## Validation Commands`

Lists shell commands to run after implementation.  Use only simple commands
from the accepted families listed above:

```markdown
## Validation Commands

```bash
python3 scripts/agent/check_agent_changes.py tasks/<task-file>.md
pytest -q tests/test_something.py
```
```

The runner runs each accepted command in order and reports PASS, FAIL, or SKIP
(when the binary is not installed).

---

## Change Checking

`scripts/agent/check_agent_changes.py` is the standalone checker used both by
the runner and by Codex during manual review:

```bash
python3 scripts/agent/check_agent_changes.py tasks/<task-file>.md
# optionally
python3 scripts/agent/check_agent_changes.py tasks/<task-file>.md --repo-root /path
```

It:
- Parses `## Files Claude May Modify` from the task file
- Runs `git status --short --untracked-files=all` to list modified, added,
  deleted, renamed, copied, and untracked files (including files nested inside
  new untracked directories)
- Exits 0 (PASS) if every changed file is in the allowed list
- Exits 1 (FAIL) with a violation list if any file is outside the allowed list

---

## Review Cycle

```
Codex creates tasks/<task-file>.md
  ↓
run_claude_task.sh tasks/<task-file>.md
  ↓  (inside the wrapper)
  Claude Code reads task file → implements changes
    (narrow --tools, no blanket Bash, no eval)
  check_agent_changes.py → verifies only allowed files changed
  Validation commands → allowlist check → ACCEPT/REJECT/SKIP → PASS/FAIL
  ↓
Codex reviews:
  git diff --stat
  git diff
  test results
  .agent-runs/<run>/summary.txt
  ↓
Codex decision: accept | revise | reject
```

---

## Forbidden Flags

Never pass these flags to the Claude binary:

- `--permission-mode bypassPermissions`
- `--dangerously-skip-permissions`

Never call Codex directly from the runner:

- `/snap/bin/codex`
- `/home/rlatjswo/.npm-global/bin/codex`
