#!/usr/bin/env bash
set -euo pipefail

echo "== identity =="
whoami
pwd
echo "HOME=$HOME"
echo "SHELL=$SHELL"

echo
echo "== node / npm =="
export NVM_DIR="$HOME/.nvm"
if [[ -s "$NVM_DIR/nvm.sh" ]]; then
  . "$NVM_DIR/nvm.sh"
  nvm use --delete-prefix --silent 22 >/dev/null 2>&1 || nvm use --silent 22 >/dev/null 2>&1 || true
fi
node -v
npm -v

echo
echo "== codex =="
type -a codex || true
echo "selected: $(command -v codex || true)"
codex --version || true

echo
echo "== claude =="
type -a claude || true
echo "selected: $(command -v claude || true)"
claude --version || true
claude auth status --text || true

echo
echo "== python =="
command -v python3 || true
python3 --version || true
command -v python || true

echo
echo "== sandbox deps =="
command -v bwrap || true
command -v socat || true

echo
echo "== sensitive env names only =="
env | cut -d= -f1 | grep -Ei 'OPENAI|ANTHROPIC|CLAUDE|GITHUB|GH_|HF_|WANDB|TOKEN|KEY|SECRET|PASSWORD' || true

echo
echo "== project git =="
git status --short || true
