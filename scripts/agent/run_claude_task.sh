#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/rlatjswo/projects/cv-forensics"
REAL_HOME="/home/rlatjswo"

export HOME="$REAL_HOME"
export CLAUDE_BIN="${CLAUDE_BIN:-$REAL_HOME/.local/bin/claude}"
export PATH="$REAL_HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:/snap/bin:${PATH:-}"

if [[ ! -x "$CLAUDE_BIN" ]]; then
  echo "Claude Code executable not found or not executable: $CLAUDE_BIN" >&2
  echo "Try: command -v claude" >&2
  exit 127
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY RUN OK"
  echo "Project root: $PROJECT_ROOT"
  echo "Using Claude binary: $CLAUDE_BIN"
  "$CLAUDE_BIN" --version
  "$CLAUDE_BIN" auth status --text || true
  exit 0
fi

TASK_FILE="${1:-}"

if [[ -z "$TASK_FILE" ]]; then
  echo "Usage: scripts/agent/run_claude_task.sh tasks/<task-file>.md" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

if [[ ! -f "$TASK_FILE" ]]; then
  echo "Task file not found: $TASK_FILE" >&2
  exit 1
fi

echo "Project root: $PROJECT_ROOT" >&2
echo "Real home: $REAL_HOME" >&2
echo "Using Claude binary: $CLAUDE_BIN" >&2
echo "Using task file: $TASK_FILE" >&2

exec "$CLAUDE_BIN" -p \
  --permission-mode default \
  --max-turns 20 \
  "Read CLAUDE.md, docs/project_brief.md, and ${TASK_FILE}. Implement only this task. Do not access .env, secrets, data, datasets, outputs, or checkpoints. Do not install packages. Do not run git push. End with an IMPLEMENTATION REPORT."
