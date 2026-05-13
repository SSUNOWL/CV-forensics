#!/usr/bin/env bash
set -u -o pipefail

PROJECT_DIR="$(pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
TESTDIR="$PROJECT_DIR/.codex-claude-smoke-$STAMP"
LOG="$TESTDIR/smoke.log"

mkdir -p "$TESTDIR"
exec > >(tee -a "$LOG") 2>&1

PASS=1

section() {
  echo
  echo "================================================================"
  echo "$1"
  echo "================================================================"
}

fail() {
  PASS=0
  echo
  echo "[FAIL] $1"
}

pass() {
  echo "[PASS] $1"
}

section "0. BASIC ENVIRONMENT"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "TESTDIR=$TESTDIR"
echo "LOG=$LOG"
echo

echo "whoami: $(whoami)"
echo "shell: ${SHELL:-unknown}"
echo "pwd: $(pwd)"
echo

echo "PATH:"
echo "$PATH" | tr ':' '\n' | nl -ba
echo

echo "codex candidates:"
type -a codex 2>&1 || true
echo

echo "claude candidates:"
type -a claude 2>&1 || true
echo

echo "selected codex:"
command -v codex 2>&1 || true
readlink -f "$(command -v codex 2>/dev/null)" 2>/dev/null || true
codex --version 2>&1 || true
echo

echo "selected claude:"
command -v claude 2>&1 || true
readlink -f "$(command -v claude 2>/dev/null)" 2>/dev/null || true
claude --version 2>&1 || true
echo

if ! command -v codex >/dev/null 2>&1; then
  fail "codex command not found"
fi

if ! command -v claude >/dev/null 2>&1; then
  fail "claude command not found"
fi

cat > "$TESTDIR/brief.md" <<'EOF'
# Smoke test brief

This directory is a disposable test workspace.

Expected behavior:
- Codex can read this file.
- Codex can write files inside this directory.
- Codex can detect the Claude CLI.
- Codex may call Claude once for a tiny non-interactive test.
EOF

section "1. CLAUDE DIRECT NON-INTERACTIVE TEST"

CLAUDE_DIRECT_OUT="$TESTDIR/claude_direct.out"
CLAUDE_DIRECT_ERR="$TESTDIR/claude_direct.err"

timeout 90s claude -p "Return exactly: CLAUDE_DIRECT_OK" \
  > "$CLAUDE_DIRECT_OUT" \
  2> "$CLAUDE_DIRECT_ERR"

CLAUDE_DIRECT_STATUS=$?

echo "status=$CLAUDE_DIRECT_STATUS"
echo "--- stdout ---"
cat "$CLAUDE_DIRECT_OUT" || true
echo
echo "--- stderr ---"
cat "$CLAUDE_DIRECT_ERR" || true
echo

if [ "$CLAUDE_DIRECT_STATUS" -eq 0 ] && grep -q "CLAUDE_DIRECT_OK" "$CLAUDE_DIRECT_OUT"; then
  pass "Claude direct non-interactive call works"
else
  fail "Claude direct non-interactive call failed"
fi

section "2. CODEX SIMPLE WORKSPACE WRITE TEST"

CODEX_SIMPLE_STDOUT="$TESTDIR/codex_simple.stdout"
CODEX_SIMPLE_STDERR="$TESTDIR/codex_simple.stderr"
CODEX_SIMPLE_FINAL="$TESTDIR/codex_simple_final.md"

timeout 240s codex exec \
  --cd "$TESTDIR" \
  --sandbox workspace-write \
  --ephemeral \
  --output-last-message "$CODEX_SIMPLE_FINAL" \
  "You are running a smoke test in this workspace only.

Do these exact tasks:
1. Read brief.md.
2. Create a file named codex_created.txt containing exactly this line:
CODEX_WRITE_OK
3. Create a file named codex_plan.md with exactly two bullet points about how Codex can delegate implementation tasks to Claude and then review Claude's result.
4. Do not access the network.
5. Do not modify anything outside the current workspace." \
  > "$CODEX_SIMPLE_STDOUT" \
  2> "$CODEX_SIMPLE_STDERR"

CODEX_SIMPLE_STATUS=$?

echo "status=$CODEX_SIMPLE_STATUS"
echo "--- stdout tail ---"
tail -n 80 "$CODEX_SIMPLE_STDOUT" || true
echo
echo "--- stderr tail ---"
tail -n 120 "$CODEX_SIMPLE_STDERR" || true
echo
echo "--- final message ---"
cat "$CODEX_SIMPLE_FINAL" 2>/dev/null || true
echo

if [ "$CODEX_SIMPLE_STATUS" -eq 0 ] \
  && [ -f "$TESTDIR/codex_created.txt" ] \
  && grep -q "CODEX_WRITE_OK" "$TESTDIR/codex_created.txt" \
  && [ -f "$TESTDIR/codex_plan.md" ]; then
  pass "Codex can write inside workspace"
else
  fail "Codex workspace-write test failed"
fi

section "3. CODEX CAN SEE CLAUDE COMMAND TEST"

CODEX_PROBE_STDOUT="$TESTDIR/codex_claude_probe.stdout"
CODEX_PROBE_STDERR="$TESTDIR/codex_claude_probe.stderr"
CODEX_PROBE_FINAL="$TESTDIR/codex_claude_probe_final.md"

timeout 240s codex exec \
  --cd "$TESTDIR" \
  --sandbox workspace-write \
  --ephemeral \
  --output-last-message "$CODEX_PROBE_FINAL" \
  "Inside this workspace only:
1. Run 'command -v claude'.
2. Run 'claude --version'.
3. Write both command outputs to codex_claude_probe.txt.
4. Do not run 'claude -p' in this step.
5. Do not modify anything outside the current workspace." \
  > "$CODEX_PROBE_STDOUT" \
  2> "$CODEX_PROBE_STDERR"

CODEX_PROBE_STATUS=$?

echo "status=$CODEX_PROBE_STATUS"
echo "--- stdout tail ---"
tail -n 80 "$CODEX_PROBE_STDOUT" || true
echo
echo "--- stderr tail ---"
tail -n 120 "$CODEX_PROBE_STDERR" || true
echo
echo "--- codex_claude_probe.txt ---"
cat "$TESTDIR/codex_claude_probe.txt" 2>/dev/null || true
echo

if [ "$CODEX_PROBE_STATUS" -eq 0 ] \
  && [ -f "$TESTDIR/codex_claude_probe.txt" ] \
  && grep -q "Claude Code" "$TESTDIR/codex_claude_probe.txt"; then
  pass "Codex can detect Claude CLI"
else
  fail "Codex could not detect Claude CLI cleanly"
fi

section "4. CODEX CALLS CLAUDE NESTED TEST"

CODEX_NESTED_STDOUT="$TESTDIR/codex_nested.stdout"
CODEX_NESTED_STDERR="$TESTDIR/codex_nested.stderr"
CODEX_NESTED_FINAL="$TESTDIR/codex_nested_final.md"

timeout 360s codex exec \
  --cd "$TESTDIR" \
  --sandbox workspace-write \
  -c sandbox_workspace_write.network_access=true \
  --ephemeral \
  --output-last-message "$CODEX_NESTED_FINAL" \
  "Inside this workspace only:
1. Run exactly this command:
claude -p 'Return exactly: CLAUDE_NESTED_OK'
2. Save the raw stdout from that command to nested_claude_result.txt.
3. Save the raw stderr from that command to nested_claude_stderr.txt.
4. Create nested_summary.md explaining in one sentence whether the nested Claude call succeeded.
5. Do not modify anything outside the current workspace." \
  > "$CODEX_NESTED_STDOUT" \
  2> "$CODEX_NESTED_STDERR"

CODEX_NESTED_STATUS=$?

echo "status=$CODEX_NESTED_STATUS"
echo "--- stdout tail ---"
tail -n 80 "$CODEX_NESTED_STDOUT" || true
echo
echo "--- stderr tail ---"
tail -n 160 "$CODEX_NESTED_STDERR" || true
echo
echo "--- nested_claude_result.txt ---"
cat "$TESTDIR/nested_claude_result.txt" 2>/dev/null || true
echo
echo "--- nested_claude_stderr.txt ---"
cat "$TESTDIR/nested_claude_stderr.txt" 2>/dev/null || true
echo
echo "--- nested_summary.md ---"
cat "$TESTDIR/nested_summary.md" 2>/dev/null || true
echo

if [ "$CODEX_NESTED_STATUS" -eq 0 ] \
  && [ -f "$TESTDIR/nested_claude_result.txt" ] \
  && grep -q "CLAUDE_NESTED_OK" "$TESTDIR/nested_claude_result.txt"; then
  pass "Codex can call Claude non-interactively"
else
  fail "Codex nested Claude call failed"
fi

section "5. SUMMARY"

echo "Test directory:"
echo "$TESTDIR"
echo

echo "Files created:"
find "$TESTDIR" -maxdepth 1 -type f -printf "%f\n" | sort
echo

if [ "$PASS" -eq 1 ]; then
  echo "ALL_SMOKE_TESTS_PASSED"
  echo
  echo "해석:"
  echo "- Codex CLI 실행 정상"
  echo "- Claude CLI 실행 정상"
  echo "- Codex workspace-write 정상"
  echo "- Codex -> Claude 하위 호출 정상"
  exit 0
else
  echo "SOME_SMOKE_TESTS_FAILED"
  echo
  echo "아래 진단 정보를 같이 붙여줘."
  echo
  echo "===== DIAGNOSTIC SNAPSHOT ====="
  uname -a || true
  echo
  echo "node/npm:"
  command -v node || true
  node -v 2>&1 || true
  command -v npm || true
  npm -v 2>&1 || true
  echo
  echo "codex:"
  type -a codex 2>&1 || true
  command -v codex 2>&1 || true
  readlink -f "$(command -v codex 2>/dev/null)" 2>/dev/null || true
  codex --version 2>&1 || true
  codex exec --help 2>&1 | sed -n '1,100p' || true
  echo
  echo "claude:"
  type -a claude 2>&1 || true
  command -v claude 2>&1 || true
  readlink -f "$(command -v claude 2>/dev/null)" 2>/dev/null || true
  claude --version 2>&1 || true
  claude --help 2>&1 | sed -n '1,100p' || true
  echo
  echo "npm global:"
  npm prefix -g 2>&1 || true
  npm root -g 2>&1 || true
  npm ls -g --depth=0 @openai/codex 2>&1 || true
  echo
  echo "PATH:"
  echo "$PATH" | tr ':' '\n' | nl -ba
  echo
  echo "testdir listing:"
  ls -la "$TESTDIR" || true
  echo
  echo "log file:"
  echo "$LOG"
  exit 1
fi
