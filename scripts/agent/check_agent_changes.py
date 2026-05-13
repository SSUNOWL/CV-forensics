#!/usr/bin/env python3
"""
Check that all changed git files are within the allowed modification list
declared in a task file.  Also extracts validation commands for the runner.

Usage:
  # Check changed files against allowed list
  python3 scripts/agent/check_agent_changes.py tasks/<task>.md [--repo-root /path]

  # Print validation commands (one per line) and exit
  python3 scripts/agent/check_agent_changes.py tasks/<task>.md --print-validation-commands

  # Check if a single command is safe to run as a validation command
  python3 scripts/agent/check_agent_changes.py tasks/<task>.md --check-cmd "python3 foo.py"

Exit codes:
  0  PASS (or --print-validation-commands / --check-cmd safe)
  1  FAIL: violation found, task file missing, or --check-cmd unsafe
  2  FAIL: section not found in task file
  3  FAIL: git status command failed
"""

import argparse
import os
import re
import shlex
import stat as _stat
import subprocess
import sys
from pathlib import Path

_ALLOWED_SECTION = "## Files Claude May Modify"

# ---------------------------------------------------------------------------
# Safe validation command classification
# ---------------------------------------------------------------------------

_FORBIDDEN_INJECTION_RE = re.compile(r'[;|&`]|\$\(|[><]')

_FORBIDDEN_FIRST_WORDS = frozenset([
    "curl", "wget", "ssh", "scp", "rsync",
    "pip", "pip3", "npm", "yarn", "pnpm",
    "kaggle", "huggingface-cli", "wandb",
    "bash", "sh", "zsh", "fish",
])

_FORBIDDEN_GIT_SUBCMDS = (
    "git push", "git pull", "git fetch", "git clone", "git remote",
)

_FORBIDDEN_STRINGS = ("rm -rf", "rm -r ")

_PROTECTED_DIRS = frozenset([
    "data", "datasets", "outputs", "checkpoints", "secrets",
])


def _has_protected_path_arg(cmd: str) -> bool:
    """Return True if any token in cmd references a protected path.

    Handles relative paths (data/foo.py), dot-slash prefixes (./data/foo.py),
    and absolute paths (/data/foo.py).  Uses shlex.split for safe tokenization.
    """
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return True  # unparseable command is unsafe
    for token in tokens:
        # Normalize: strip leading ./ repeatedly (./data -> data)
        path = token
        while path.startswith("./"):
            path = path[2:]
        # .env or .env.* (exact file match)
        if path == ".env" or path.startswith(".env."):
            return True
        # Protected top-level directory — filter empty parts to handle /abs/paths
        path_parts = [p for p in path.split("/") if p]
        if path_parts and path_parts[0] in _PROTECTED_DIRS:
            return True
    return False

_ALLOWED_VAL_PATTERNS = [
    re.compile(r"^python3 [a-zA-Z_][-a-zA-Z0-9_./ ]*$"),
    re.compile(r"^pytest -q [a-zA-Z_][-a-zA-Z0-9_./ ]*$"),
    re.compile(r"^python3 -m pytest -q [a-zA-Z_][-a-zA-Z0-9_./ ]*$"),
    re.compile(r"^test -f [a-zA-Z_/][-a-zA-Z0-9_./]*$"),
    re.compile(r"^grep -q \S+ [a-zA-Z_/][-a-zA-Z0-9_./]*$"),
    re.compile(r"^git status --short( --untracked-files=all)?$"),
    re.compile(r"^git diff -- [a-zA-Z_/][-a-zA-Z0-9_./ ]+$"),
]


def is_safe_validation_cmd(cmd: str) -> bool:
    """Return True if cmd is in the allowed validation command families.

    Rejects shell injection syntax, forbidden tools/actions, and protected
    paths.  Only allows a narrow set of simple validation command families.
    """
    cmd = cmd.strip()
    if not cmd:
        return False

    # Reject shell control/injection syntax
    if _FORBIDDEN_INJECTION_RE.search(cmd):
        return False

    # Reject forbidden first words (binaries)
    first_word = cmd.split()[0]
    if first_word in _FORBIDDEN_FIRST_WORDS:
        return False

    # Reject forbidden git sub-commands
    for sub in _FORBIDDEN_GIT_SUBCMDS:
        if cmd.startswith(sub):
            return False

    # Reject dangerous deletion strings
    for s in _FORBIDDEN_STRINGS:
        if s in cmd:
            return False

    # Reject commands referencing protected paths (path-aware, handles relative paths)
    if _has_protected_path_arg(cmd):
        return False

    # Allow only specific command families
    for pattern in _ALLOWED_VAL_PATTERNS:
        if pattern.match(cmd):
            return True

    return False


# ---------------------------------------------------------------------------
# Task file parsers
# ---------------------------------------------------------------------------


def _read_task(task_file: str) -> str:
    with open(task_file, "r", encoding="utf-8") as f:
        return f.read()


def parse_allowed_files(content: str):
    """Return list of allowed file paths from '## Files Claude May Modify' section.

    Returns None if the section is not found.
    """
    match = re.search(
        r"## Files Claude May Modify\s*\n(.*?)(?=\n## |\Z)",
        content,
        re.DOTALL,
    )
    if match is None:
        return None

    allowed = []
    for line in match.group(1).splitlines():
        line = line.strip()
        m = re.match(r"^-\s+[`\"]?([^\s`\"]+)[`\"]?", line)
        if m:
            file_path = m.group(1)
            if file_path:
                allowed.append(file_path)
    return allowed


def parse_validation_commands(content: str) -> list:
    """Return list of validation commands from '## Validation Commands' section."""
    section_match = re.search(
        r"## Validation Commands\s*\n(.*?)(?=\n## |\Z)",
        content,
        re.DOTALL,
    )
    if section_match is None:
        return []

    section = section_match.group(1)
    commands = []
    for block in re.findall(r"```(?:bash)?\n(.*?)```", section, re.DOTALL):
        for line in block.strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                commands.append(line)
    return commands


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _get_repo_root(hint_path: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=hint_path,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return str(Path(".").resolve())


def get_changed_files(repo_root: str) -> list:
    """Return list of changed/untracked file paths from git status.

    Uses --untracked-files=all so that individual files inside new untracked
    directories are listed separately rather than as a collapsed directory
    entry.

    Special filesystem objects (character devices, block devices, sockets,
    named pipes) are excluded because they are pre-existing sandbox artifacts
    and can never be created by an agent implementation worker.
    """
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: git status failed: {result.stderr}", file=sys.stderr)
        sys.exit(3)

    changed = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # git status --short: "XY filename" or "XY old -> new" for renames/copies
        rest = line[3:]
        if " -> " in rest:
            # Rename (R) or copy (C): both source and destination must be checked.
            # If either side is outside the allowed list the checker must fail.
            parts = rest.split(" -> ", 1)
            paths = [
                parts[0].strip().strip('"'),
                parts[1].strip().strip('"'),
            ]
        else:
            paths = [rest.strip().strip('"')]

        for path in paths:
            if not path:
                continue
            # Filter out special filesystem objects (char/block devices, sockets,
            # FIFOs) which are pre-existing sandbox artifacts, not agent output.
            full_path = os.path.join(repo_root, path)
            try:
                st = os.lstat(full_path)
                mode = st.st_mode
                if not (_stat.S_ISREG(mode) or _stat.S_ISDIR(mode) or _stat.S_ISLNK(mode)):
                    continue
            except OSError:
                pass  # path absent (e.g. deleted file or rename source) — include it
            changed.append(path)
    return changed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check git-changed files against task file allowed list."
    )
    parser.add_argument("task_file", help="Path to the task markdown file")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Git repository root (auto-detected if omitted)",
    )
    parser.add_argument(
        "--print-validation-commands",
        action="store_true",
        help="Print validation commands one per line and exit 0",
    )
    parser.add_argument(
        "--check-cmd",
        default=None,
        metavar="CMD",
        help="Check whether CMD is a safe validation command; exit 0 if safe, 1 if not",
    )
    args = parser.parse_args()

    # --check-cmd mode: no repo or task file needed for the safety check itself
    if args.check_cmd is not None:
        if is_safe_validation_cmd(args.check_cmd):
            sys.exit(0)
        else:
            sys.exit(1)

    # Resolve repo root
    if args.repo_root:
        repo_root = str(Path(args.repo_root).resolve())
    else:
        candidate = str(Path(args.task_file).parent.resolve())
        if not Path(candidate).exists():
            candidate = str(Path(".").resolve())
        repo_root = _get_repo_root(candidate)

    # Resolve task file (relative paths are anchored to repo root)
    task_file = args.task_file
    if not Path(task_file).is_absolute():
        task_file = str(Path(repo_root) / task_file)

    if not Path(task_file).exists():
        print(f"ERROR: Task file not found: {task_file}", file=sys.stderr)
        sys.exit(1)

    content = _read_task(task_file)

    # --print-validation-commands mode
    if args.print_validation_commands:
        for cmd in parse_validation_commands(content):
            print(cmd)
        sys.exit(0)

    # --- Changed-file check mode ---
    allowed = parse_allowed_files(content)
    if allowed is None:
        print(
            f"ERROR: Section '{_ALLOWED_SECTION}' not found in task file.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"Allowed files ({len(allowed)}):")
    for f in allowed:
        print(f"  {f}")

    changed = get_changed_files(repo_root)

    if not changed:
        print("\nNo changed files detected.")
        print("PASS: All changes are within the allowed list.")
        sys.exit(0)

    print(f"\nChanged files ({len(changed)}):")
    for f in changed:
        print(f"  {f}")

    violations = [f for f in changed if f not in allowed]

    if violations:
        print(f"\nFAIL: {len(violations)} file(s) changed outside the allowed list:")
        for f in violations:
            print(f"  VIOLATION: {f}")
        sys.exit(1)

    print("\nPASS: All changed files are within the allowed list.")
    sys.exit(0)


if __name__ == "__main__":
    main()
