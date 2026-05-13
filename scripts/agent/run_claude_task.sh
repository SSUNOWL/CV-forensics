#!/usr/bin/env bash
# Safe Claude Code task runner with per-run logging, change checking, and validation.
# Validation commands from task files are allowlisted — eval is never used.
set -euo pipefail

# Resolve the repository root from the script's own location so the wrapper
# works correctly regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REAL_HOME="/home/rlatjswo"
export HOME="$REAL_HOME"
export CLAUDE_BIN="${CLAUDE_BIN:-$REAL_HOME/.local/bin/claude}"
export PATH="$REAL_HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:/snap/bin:${PATH:-}"

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  CLAUDE_BIN=/home/rlatjswo/.local/bin/claude \
    scripts/agent/run_claude_task.sh tasks/<task-file>.md

Environment:
  CLAUDE_BIN   Path to the claude binary (default: ~/.local/bin/claude)

The wrapper:
  1. Validates the task file argument
  2. Creates a timestamped log directory under .agent-runs/
  3. Runs Claude Code in non-interactive print mode (-p) with narrowly
     scoped --tools, --allowedTools, and --disallowedTools
     (Bash restricted to git status/diff introspection; file operations
     use Read/Write/Edit/MultiEdit tools, not broad Bash access)
  4. Captures stdout and stderr
  5. After Claude exits, checks changed files against the task's allowed list
  6. Runs validation commands from the task file through an allowlist check
     (no eval — each command is verified safe before running)
  7. Prints a concise run summary

Do NOT use --permission-mode bypassPermissions or --dangerously-skip-permissions.
Do NOT run Claude recursively or call Codex from inside this wrapper.
EOF
  exit 0
fi

# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
TASK_ARG="${1:-}"
if [[ -z "$TASK_ARG" ]]; then
  echo "ERROR: Task file argument is required." >&2
  echo "Usage: scripts/agent/run_claude_task.sh tasks/<task-file>.md" >&2
  exit 1
fi

# Resolve task file: absolute paths are used as-is; relative paths are
# anchored to the repository root (not the caller's CWD).
if [[ "$TASK_ARG" = /* ]]; then
  TASK_FILE="$TASK_ARG"
else
  TASK_FILE="$PROJECT_ROOT/$TASK_ARG"
fi

if [[ ! -f "$TASK_FILE" ]]; then
  echo "ERROR: Task file not found: $TASK_FILE" >&2
  exit 1
fi

if [[ ! -x "$CLAUDE_BIN" ]]; then
  echo "ERROR: Claude Code binary not found or not executable: $CLAUDE_BIN" >&2
  echo "Set CLAUDE_BIN=/path/to/claude or verify the installation." >&2
  exit 127
fi

# ---------------------------------------------------------------------------
# Timestamped log directory
# ---------------------------------------------------------------------------
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TASK_BASENAME="$(basename "$TASK_FILE" .md)"
LOG_DIR="$PROJECT_ROOT/.agent-runs/${TIMESTAMP}_${TASK_BASENAME}"
mkdir -p "$LOG_DIR"

STDOUT_LOG="$LOG_DIR/claude_stdout.log"
STDERR_LOG="$LOG_DIR/claude_stderr.log"
SUMMARY_LOG="$LOG_DIR/summary.txt"

# Save the task file path and a verbatim copy into the log directory.
echo "$TASK_FILE" > "$LOG_DIR/task_file_path.txt"
cp "$TASK_FILE" "$LOG_DIR/task_file.md"

echo "=== Claude Runner Wrapper ===" >&2
echo "Repository root : $PROJECT_ROOT" >&2
echo "Task file       : $TASK_FILE" >&2
echo "Log directory   : $LOG_DIR" >&2
echo "Claude binary   : $CLAUDE_BIN" >&2
echo "" >&2

# Run everything relative to the repository root.
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------
PROMPT="Read CLAUDE.md, docs/project_brief.md, and ${TASK_ARG}. \
Implement only what is specified in the task file. \
Follow only the task file's allowed file list and forbidden actions. \
Do not access .env, .env.*, secrets, data, datasets, outputs, or checkpoints. \
Do not install packages. \
Do not run git push. \
Do not download datasets or train models. \
Do not access network resources from shell commands. \
End your response with an IMPLEMENTATION REPORT listing files changed and validation results."

# ---------------------------------------------------------------------------
# Narrow tool permissions for Claude
#
# AVAILABLE_TOOLS  : All tool families Claude may potentially invoke.
# ALLOWED_TOOLS    : Explicit narrow allowlist passed via --allowedTools.
#                    Bash is limited to safe repository introspection only;
#                    file creation/editing uses Read/Write/Edit/MultiEdit.
#                    Validation commands run after Claude exits through the
#                    wrapper's own safe runner — Claude does not need broad
#                    Bash access (python3, pytest, grep, etc.) to implement tasks.
# DISALLOWED_TOOLS : Belt-and-suspenders block for network commands, package
#                    managers, git remote operations, dangerous deletion, and
#                    protected-path access.
# ---------------------------------------------------------------------------
AVAILABLE_TOOLS="Read,Write,Edit,MultiEdit,Glob,Grep,Bash"

ALLOWED_TOOLS="\
Read,Write,Edit,MultiEdit,Glob,Grep,\
Bash(git status --short*),\
Bash(git diff -- *)"

DISALLOWED_TOOLS="\
WebSearch,WebFetch,computer,\
Bash(curl*),Bash(wget*),Bash(ssh*),Bash(scp*),Bash(rsync*),\
Bash(pip*),Bash(npm*),Bash(yarn*),Bash(pnpm*),\
Bash(kaggle*),Bash(huggingface-cli*),Bash(wandb*),\
Bash(git push*),Bash(git pull*),Bash(git fetch*),Bash(git clone*),\
Bash(rm -rf*),Bash(rm -r*),Bash(rm -f*),\
Bash(chmod*),Bash(chown*),\
Bash(cat .env*),Bash(cat secrets*),\
Bash(ls data*),Bash(ls datasets*),Bash(ls outputs*),Bash(ls checkpoints*)"

# ---------------------------------------------------------------------------
# Run Claude Code in non-interactive print mode
# ---------------------------------------------------------------------------
echo "Running Claude Code (output -> $STDOUT_LOG) ..." >&2
CLAUDE_EXIT=0
printf '%s\n' "$PROMPT" | "$CLAUDE_BIN" -p \
  --input-format text \
  --permission-mode default \
  --max-turns 30 \
  --tools "$AVAILABLE_TOOLS" \
  --allowedTools "$ALLOWED_TOOLS" \
  --disallowedTools "$DISALLOWED_TOOLS" \
  --output-format text \
  --no-session-persistence \
  > "$STDOUT_LOG" 2> "$STDERR_LOG" || CLAUDE_EXIT=$?

echo "Claude exited with code: $CLAUDE_EXIT" >&2

# ---------------------------------------------------------------------------
# Check changed files against the allowed list
# ---------------------------------------------------------------------------
CHANGE_CHECK_RESULT=0
CHANGE_CHECK_OUTPUT=""
echo "" >&2
echo "--- Checking changed files ---" >&2
CHANGE_CHECK_OUTPUT="$(
  python3 "$SCRIPT_DIR/check_agent_changes.py" \
    --repo-root "$PROJECT_ROOT" \
    "$TASK_ARG" 2>&1
)" || CHANGE_CHECK_RESULT=$?
echo "$CHANGE_CHECK_OUTPUT" >&2

# ---------------------------------------------------------------------------
# Safe validation runner (no eval)
#
# Each command extracted from the task file is first passed through
# check_agent_changes.py --check-cmd for allowlist classification.
# Only commands matching the approved families (python3, pytest, git
# status/diff, test -f, grep -q) are accepted.  Commands with shell
# control characters, forbidden binaries, or protected paths are rejected
# before any execution attempt.
# ---------------------------------------------------------------------------
VALIDATION_RESULT=0
VALIDATION_LINES=""
echo "" >&2
echo "--- Running validation commands ---" >&2

VALIDATION_CMDS="$(
  python3 "$SCRIPT_DIR/check_agent_changes.py" \
    --print-validation-commands \
    --repo-root "$PROJECT_ROOT" \
    "$TASK_ARG" 2>/dev/null
)" || true

if [[ -z "$VALIDATION_CMDS" ]]; then
  echo "  (no validation commands found in task file)" >&2
  VALIDATION_LINES="(no validation commands)"
else
  while IFS= read -r cmd; do
    [[ -z "$cmd" ]] && continue

    # --- Allowlist safety check (delegates to Python module) ---
    if ! python3 "$SCRIPT_DIR/check_agent_changes.py" \
         --repo-root "$PROJECT_ROOT" \
         "$TASK_ARG" \
         --check-cmd "$cmd" 2>/dev/null; then
      echo "  REJECT (not in allowed command family): $cmd" >&2
      VALIDATION_LINES+="REJECT: $cmd"$'\n'
      VALIDATION_RESULT=1
      continue
    fi
    echo "  ACCEPT: $cmd" >&2

    # --- Availability check ---
    cmd_bin="${cmd%% *}"
    if ! command -v "$cmd_bin" >/dev/null 2>&1; then
      echo "  SKIP ($cmd_bin not available): $cmd" >&2
      VALIDATION_LINES+="SKIP ($cmd_bin not available): $cmd"$'\n'
      continue
    fi

    # --- Run safely using array split (no eval) ---
    echo "  \$ $cmd" >&2
    cmd_parts=()
    IFS=' ' read -ra cmd_parts <<< "$cmd"
    CMD_OUT=""
    CMD_EXIT=0
    CMD_OUT="$("${cmd_parts[@]}" 2>&1)" || CMD_EXIT=$?

    if [[ $CMD_EXIT -eq 0 ]]; then
      echo "  PASS" >&2
      VALIDATION_LINES+="PASS: $cmd"$'\n'
    else
      echo "  FAIL (exit $CMD_EXIT)" >&2
      [[ -n "$CMD_OUT" ]] && echo "$CMD_OUT" | head -20 >&2
      VALIDATION_LINES+="FAIL (exit $CMD_EXIT): $cmd"$'\n'
      VALIDATION_RESULT=$CMD_EXIT
    fi
  done <<< "$VALIDATION_CMDS"
fi

# ---------------------------------------------------------------------------
# Determine final wrapper exit status
# ---------------------------------------------------------------------------
WRAPPER_EXIT=0
[[ $CLAUDE_EXIT -ne 0 ]]         && WRAPPER_EXIT=$CLAUDE_EXIT
[[ $CHANGE_CHECK_RESULT -ne 0 ]] && WRAPPER_EXIT=$CHANGE_CHECK_RESULT
[[ $VALIDATION_RESULT -ne 0 ]]   && WRAPPER_EXIT=$VALIDATION_RESULT

# ---------------------------------------------------------------------------
# Print and save summary
# ---------------------------------------------------------------------------
echo "" >&2
echo "=== Run Summary ===" >&2
echo "Log directory      : $LOG_DIR" >&2
echo "Claude exit code   : $CLAUDE_EXIT" >&2
echo "Change check       : $([ $CHANGE_CHECK_RESULT -eq 0 ] && echo PASS || echo FAIL)" >&2
echo "Validation         : $([ $VALIDATION_RESULT -eq 0 ] && echo PASS || echo FAIL)" >&2
echo "Wrapper exit status: $WRAPPER_EXIT" >&2

{
  echo "=== Run Summary ==="
  echo "Log directory      : $LOG_DIR"
  echo "Claude exit code   : $CLAUDE_EXIT"
  echo "Change check       : $([ $CHANGE_CHECK_RESULT -eq 0 ] && echo PASS || echo FAIL)"
  echo ""
  echo "$CHANGE_CHECK_OUTPUT"
  echo ""
  echo "Validation         : $([ $VALIDATION_RESULT -eq 0 ] && echo PASS || echo FAIL)"
  echo ""
  echo "$VALIDATION_LINES"
  echo "Wrapper exit status: $WRAPPER_EXIT"
} > "$SUMMARY_LOG"

exit $WRAPPER_EXIT
