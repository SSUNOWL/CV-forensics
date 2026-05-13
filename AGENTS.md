
## Mandatory project context

Before planning or delegating any project task, read:

- `docs/project_brief.md`

Also inspect the reference PDFs if needed and readable:

- `docs/reference/proposal.pdf`
- `docs/reference/overview_ppt.pdf`

If the PDFs are difficult to parse, trust `docs/project_brief.md` as the canonical summary unless the user says otherwise.

For the first project run, do not train models, download datasets, install packages, or access data/checkpoints.
The first run should only verify the Codex -> Claude Code -> Codex review workflow.

## Local lab execution rules

This repository runs on the lab PC account:

- user: rlatjswo
- project root: /home/rlatjswo/projects/cv-forensics
- Claude Code binary: /home/rlatjswo/.local/bin/claude
- preferred Python command: python3

When delegating to Claude Code, do not call claude directly.
Use exactly this command pattern:

CLAUDE_BIN=/home/rlatjswo/.local/bin/claude scripts/agent/run_claude_task.sh tasks/<task-file>.md

When validating Python scripts, use python3, not python.

Never use:

- --permission-mode bypassPermissions
- --dangerously-skip-permissions
- /snap/bin/codex
- /home/rlatjswo/.npm-global/bin/codex
- git push
- rm -rf

Codex should be started with:

scripts/agent/start_codex.sh

Claude Code is the implementation worker.
Codex is the supervisor and reviewer.
