#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/rlatjswo/projects/cv-forensics"
REAL_HOME="/home/rlatjswo"

export HOME="$REAL_HOME"
export NVM_DIR="$HOME/.nvm"

if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
  echo "nvm not found at $NVM_DIR/nvm.sh" >&2
  exit 1
fi

. "$NVM_DIR/nvm.sh"
nvm use --delete-prefix --silent 22 >/dev/null 2>&1 || nvm use --silent 22 >/dev/null 2>&1

hash -r

CODEX_BIN="$(command -v codex || true)"

if [[ -z "$CODEX_BIN" ]]; then
  echo "codex not found after loading nvm Node 22" >&2
  exit 1
fi

case "$CODEX_BIN" in
  "$HOME/.nvm/"*)
    ;;
  *)
    echo "Wrong codex binary selected: $CODEX_BIN" >&2
    echo "Expected nvm Codex under: $HOME/.nvm/" >&2
    echo "All candidates:" >&2
    type -a codex >&2 || true
    exit 1
    ;;
esac

export CLAUDE_BIN="$HOME/.local/bin/claude"
export PYTHON_BIN="/usr/bin/python3"
export PYTHONUNBUFFERED="1"

cd "$PROJECT_ROOT"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY RUN OK"
  echo "Project root: $PROJECT_ROOT"
  echo "Using Codex binary: $CODEX_BIN"
  echo "Using Claude binary: $CLAUDE_BIN"
  "$CODEX_BIN" --version
  exit 0
fi

echo "Project root: $PROJECT_ROOT" >&2
echo "Using Codex binary: $CODEX_BIN" >&2
echo "Using Claude binary: $CLAUDE_BIN" >&2

exec "$CODEX_BIN" \
  --no-alt-screen \
  -C "$PROJECT_ROOT" \
  --sandbox workspace-write \
  --ask-for-approval on-request
