#!/usr/bin/env bash
set -u -o pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="$PWD/.codex-slow-diagnose-$STAMP"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/diagnose.log"

exec > >(tee -a "$LOG") 2>&1

section() {
  echo
  echo "================================================================"
  echo "$1"
  echo "================================================================"
}

run_with_timer() {
  local label="$1"
  shift

  echo
  echo "----- $label -----"
  local start end status
  start="$(date +%s)"
  "$@"
  status=$?
  end="$(date +%s)"
  echo
  echo "[$label] status=$status elapsed=$((end-start))s"
  return "$status"
}

section "0. BASIC STATUS"

echo "OUTDIR=$OUTDIR"
echo "LOG=$LOG"
echo

echo "date:"
date
echo

echo "codex:"
type -a codex 2>/dev/null || true
command -v codex || true
readlink -f "$(command -v codex 2>/dev/null)" 2>/dev/null || true
codex --version 2>&1 || true
echo

echo "claude:"
type -a claude 2>/dev/null || true
command -v claude || true
readlink -f "$(command -v claude 2>/dev/null)" 2>/dev/null || true
claude --version 2>&1 || true
echo

echo "codex login status:"
timeout 20s codex login status 2>&1 || true
echo

section "1. SAFE CODEX AUTH SUMMARY"

python3 - <<'PY'
import json
import os
from pathlib import Path

codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
auth_file = codex_home / "auth.json"
config_file = codex_home / "config.toml"

print(f"CODEX_HOME={codex_home}")
print(f"auth_file={auth_file}")
print(f"auth_exists={auth_file.exists()}")
print(f"config_file={config_file}")
print(f"config_exists={config_file.exists()}")

if auth_file.exists():
    try:
        st = auth_file.stat()
        data = json.loads(auth_file.read_text())
        tokens = data.get("tokens") or {}
        print(json.dumps({
            "auth_file_mode": oct(st.st_mode & 0o777),
            "top_level_keys": sorted(data.keys()),
            "auth_mode": data.get("auth_mode"),
            "has_tokens_object": isinstance(tokens, dict) and bool(tokens),
            "has_access_token": bool(tokens.get("access_token")),
            "has_refresh_token": bool(tokens.get("refresh_token")),
            "has_id_token": bool(tokens.get("id_token")),
            "has_api_key_like_field": any("api" in k.lower() and "key" in k.lower() for k in data.keys()),
            "last_refresh": data.get("last_refresh"),
        }, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"auth_parse_error={type(e).__name__}: {e}")
else:
    print("auth.json not found")
PY

section "2. SAFE CONFIG FILES"

redact_config() {
  local f="$1"
  if [ -f "$f" ]; then
    echo
    echo "### $f"
    sed -E '
      s/(api[_-]?key[[:space:]]*=[[:space:]]*).*/\1<redacted>/I;
      s/(token[[:space:]]*=[[:space:]]*).*/\1<redacted>/I;
      s/(secret[[:space:]]*=[[:space:]]*).*/\1<redacted>/I;
      s/(password[[:space:]]*=[[:space:]]*).*/\1<redacted>/I;
      s/(bearer[[:space:]]*=[[:space:]]*).*/\1<redacted>/I
    ' "$f"
  else
    echo
    echo "### $f"
    echo "missing"
  fi
}

redact_config "$HOME/.codex/config.toml"
redact_config "$PWD/.codex/config.toml"

echo
echo "Other .codex config/rules/hooks under project:"
find "$PWD" -maxdepth 4 \( \
  -path '*/.codex/config.toml' -o \
  -path '*/.codex/hooks.json' -o \
  -path '*/.codex/*.rules' -o \
  -name 'AGENTS.md' \
\) -print 2>/dev/null | sort || true

section "3. NETWORK TIMING, NO SECRETS"

echo "These calls intentionally send no auth token. 401/404 is okay if it returns quickly."
echo

for url in \
  "https://api.openai.com/v1/responses" \
  "https://auth.openai.com/" \
  "https://chatgpt.com/"
do
  echo "URL=$url"
  curl -sS -o /dev/null \
    --connect-timeout 10 \
    --max-time 30 \
    -w 'http=%{http_code} remote_ip=%{remote_ip} namelookup=%{time_namelookup}s connect=%{time_connect}s tls=%{time_appconnect}s starttransfer=%{time_starttransfer}s total=%{time_total}s\n' \
    "$url" || true
  echo
done

echo "TLS check api.openai.com:"
timeout 20s bash -lc 'echo | openssl s_client -connect api.openai.com:443 -servername api.openai.com -brief 2>&1 | sed -n "1,40p"' || true

section "4. CLEAN CODEX PING OUTSIDE PROJECT, IGNORE CONFIG"

CLEAN_DIR="$(mktemp -d /tmp/codex-clean-ping.XXXXXX)"
echo "CLEAN_DIR=$CLEAN_DIR"

timeout 120s codex exec \
  --ignore-user-config \
  --ignore-rules \
  --json \
  --cd "$CLEAN_DIR" \
  --sandbox read-only \
  --skip-git-repo-check \
  --ephemeral \
  -c model_reasoning_effort=minimal \
  -c model_reasoning_summary=none \
  -c model_verbosity=low \
  "Reply exactly: CODEX_CLEAN_PING_OK. Do not run shell commands. Do not inspect files." \
  > "$OUTDIR/clean_ping.stdout" \
  2> "$OUTDIR/clean_ping.stderr"

CLEAN_STATUS=$?

echo "clean_ping status=$CLEAN_STATUS"
echo
echo "--- clean_ping.stdout ---"
cat "$OUTDIR/clean_ping.stdout" || true
echo
echo "--- clean_ping.stderr ---"
cat "$OUTDIR/clean_ping.stderr" || true
echo

section "5. CLEAN CODEX WRITE OUTSIDE PROJECT, IGNORE CONFIG"

WRITE_DIR="$(mktemp -d /tmp/codex-clean-write.XXXXXX)"
echo "WRITE_DIR=$WRITE_DIR"
echo "hello" > "$WRITE_DIR/input.txt"

timeout 180s codex exec \
  --ignore-user-config \
  --ignore-rules \
  --json \
  --cd "$WRITE_DIR" \
  --sandbox workspace-write \
  --skip-git-repo-check \
  --ephemeral \
  -c model_reasoning_effort=minimal \
  -c model_reasoning_summary=none \
  -c model_verbosity=low \
  "Read input.txt. Create output.txt containing exactly CODEX_CLEAN_WRITE_OK. Then finish." \
  > "$OUTDIR/clean_write.stdout" \
  2> "$OUTDIR/clean_write.stderr"

WRITE_STATUS=$?

echo "clean_write status=$WRITE_STATUS"
echo
echo "--- clean_write.stdout ---"
cat "$OUTDIR/clean_write.stdout" || true
echo
echo "--- clean_write.stderr ---"
cat "$OUTDIR/clean_write.stderr" || true
echo
echo "--- WRITE_DIR listing ---"
ls -la "$WRITE_DIR" || true
echo
echo "--- output.txt ---"
cat "$WRITE_DIR/output.txt" 2>/dev/null || true
echo

section "6. PROJECT CODEX PING WITH CURRENT CONFIG"

timeout 120s codex exec \
  --json \
  --cd "$PWD" \
  --sandbox read-only \
  --ephemeral \
  -c model_reasoning_effort=minimal \
  -c model_reasoning_summary=none \
  -c model_verbosity=low \
  "Reply exactly: CODEX_PROJECT_PING_OK. Do not run shell commands." \
  > "$OUTDIR/project_ping.stdout" \
  2> "$OUTDIR/project_ping.stderr"

PROJECT_STATUS=$?

echo "project_ping status=$PROJECT_STATUS"
echo
echo "--- project_ping.stdout ---"
cat "$OUTDIR/project_ping.stdout" || true
echo
echo "--- project_ping.stderr ---"
cat "$OUTDIR/project_ping.stderr" || true
echo

section "7. RECENT CODEX LOG FILES"

CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"

echo "CODEX_HOME=$CODEX_HOME"
echo

echo "Recent possible log files:"
find "$CODEX_HOME" -maxdepth 5 -type f \( \
  -iname '*.log' -o \
  -iname '*.jsonl' -o \
  -path '*/logs/*' \
\) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -30 || true

echo
echo "Tailing recent logs, redacted lightly:"
find "$CODEX_HOME" -maxdepth 5 -type f \( \
  -iname '*.log' -o \
  -path '*/logs/*' \
\) -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr \
  | head -5 \
  | cut -d' ' -f2- \
  | while read -r f; do
      echo
      echo "### $f"
      tail -n 120 "$f" 2>/dev/null \
        | sed -E '
          s/(api[_-]?key|token|secret|password|bearer)[=: ]+[^ ]+/\1=<redacted>/Ig;
          s/(Authorization: Bearer )[A-Za-z0-9._~+\/=-]+/\1<redacted>/Ig
        ' || true
    done

section "8. INTERPRETATION HINTS"

echo "CLEAN_STATUS=$CLEAN_STATUS"
echo "WRITE_STATUS=$WRITE_STATUS"
echo "PROJECT_STATUS=$PROJECT_STATUS"
echo
echo "If clean_ping succeeds but project_ping hangs/fails: project or user config is likely interfering."
echo "If clean_ping also times out with empty stdout/stderr: auth/model/network transport is likely the issue."
echo "If network timings are slow or hang: local network/proxy/TLS/WebSocket path is suspicious."
echo
echo "DIAGNOSE_LOG=$LOG"
echo "DIAGNOSE_OUTDIR=$OUTDIR"

