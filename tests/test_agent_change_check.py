#!/usr/bin/env python3
"""
Tests for scripts/agent/check_agent_changes.py.

Run with pytest:
  pytest -q tests/test_agent_change_check.py

Or standalone (no pytest required):
  python3 tests/test_agent_change_check.py

Uses only the Python standard library plus pytest-style assertions.
"""

import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

# Import the module under test directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "agent"))
import check_agent_changes as cac  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_git_repo(path: str) -> None:
    """Initialise a bare git repo suitable for testing."""
    subprocess.run(["git", "init", path], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path, check=True, capture_output=True,
    )


def _commit_file(repo: str, rel_path: str, content: str = "init\n") -> None:
    """Write a file and commit it to the repo."""
    full_path = os.path.join(repo, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    subprocess.run(["git", "add", rel_path], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", f"add {rel_path}"],
        cwd=repo, check=True, capture_output=True,
    )


def _write_file(repo: str, rel_path: str, content: str = "change\n") -> None:
    """Write (or overwrite) a file without committing."""
    full_path = os.path.join(repo, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)


def _task_content(allowed: list, validation_cmds: list = None) -> str:
    lines = ["# Test Task\n", "## Files Claude May Modify\n\n"]
    for f in allowed:
        lines.append(f"- `{f}`\n")
    lines.append("\n")
    if validation_cmds:
        lines.append("## Validation Commands\n\n```bash\n")
        lines.extend(cmd + "\n" for cmd in validation_cmds)
        lines.append("```\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Unit tests: parse_allowed_files
# ---------------------------------------------------------------------------

def test_parse_allowed_files_basic():
    content = (
        "## Files Claude May Modify\n\n"
        "- `scripts/foo.py`\n"
        "- `docs/bar.md`\n\n"
        "## Other Section\n"
    )
    allowed = cac.parse_allowed_files(content)
    assert allowed is not None
    assert "scripts/foo.py" in allowed
    assert "docs/bar.md" in allowed
    assert len(allowed) == 2


def test_parse_allowed_files_double_quoted():
    content = (
        "## Files Claude May Modify\n\n"
        '- "scripts/foo.py"\n'
        '- "docs/bar.md"\n'
    )
    allowed = cac.parse_allowed_files(content)
    assert allowed is not None
    assert "scripts/foo.py" in allowed
    assert "docs/bar.md" in allowed


def test_parse_allowed_files_no_section():
    content = "# Task\n\nNo allowed section here.\n"
    result = cac.parse_allowed_files(content)
    assert result is None


def test_parse_allowed_files_empty_section():
    content = "## Files Claude May Modify\n\n## Other\n"
    result = cac.parse_allowed_files(content)
    assert result is not None
    assert result == []


def test_parse_allowed_files_ignores_prose():
    content = (
        "## Files Claude May Modify\n\n"
        "Claude may create or edit only these files:\n\n"
        "- `scripts/agent/run_claude_task.sh`\n"
        "- `docs/agent_workflow.md`\n\n"
        "If a parent directory does not exist, create it.\n"
    )
    allowed = cac.parse_allowed_files(content)
    assert allowed is not None
    assert "scripts/agent/run_claude_task.sh" in allowed
    assert "docs/agent_workflow.md" in allowed
    # prose line must not appear
    assert any("If" in f for f in allowed) is False


# ---------------------------------------------------------------------------
# Unit tests: parse_validation_commands
# ---------------------------------------------------------------------------

def test_parse_validation_commands_basic():
    content = (
        "## Validation Commands\n\n"
        "```bash\n"
        "python3 foo.py bar.md\n"
        "pytest -q tests/\n"
        "```\n"
    )
    cmds = cac.parse_validation_commands(content)
    assert "python3 foo.py bar.md" in cmds
    assert "pytest -q tests/" in cmds


def test_parse_validation_commands_multiple_blocks():
    content = (
        "## Validation Commands\n\n"
        "```bash\n"
        "python3 a.py\n"
        "```\n\n"
        "Also:\n\n"
        "```bash\n"
        "pytest -q tests/\n"
        "```\n"
    )
    cmds = cac.parse_validation_commands(content)
    assert "python3 a.py" in cmds
    assert "pytest -q tests/" in cmds


def test_parse_validation_commands_skips_comments():
    content = (
        "## Validation Commands\n\n"
        "```bash\n"
        "# this is a comment\n"
        "python3 foo.py\n"
        "```\n"
    )
    cmds = cac.parse_validation_commands(content)
    assert "python3 foo.py" in cmds
    assert all(not c.startswith("#") for c in cmds)


def test_parse_validation_commands_no_section():
    content = "# Task\n\nNo validation section.\n"
    cmds = cac.parse_validation_commands(content)
    assert cmds == []


# ---------------------------------------------------------------------------
# Unit tests: is_safe_validation_cmd
# ---------------------------------------------------------------------------

def test_safe_cmd_allows_python3_with_args():
    assert cac.is_safe_validation_cmd(
        "python3 scripts/agent/check_agent_changes.py tasks/0002-claude-runner-wrapper.md"
    ) is True


def test_safe_cmd_allows_pytest():
    assert cac.is_safe_validation_cmd("pytest -q tests/test_agent_change_check.py") is True


def test_safe_cmd_allows_python3_m_pytest():
    assert cac.is_safe_validation_cmd(
        "python3 -m pytest -q tests/test_agent_change_check.py"
    ) is True


def test_safe_cmd_allows_test_f():
    assert cac.is_safe_validation_cmd("test -f scripts/agent/run_claude_task.sh") is True


def test_safe_cmd_allows_git_status():
    assert cac.is_safe_validation_cmd("git status --short") is True


def test_safe_cmd_allows_git_status_untracked_all():
    assert cac.is_safe_validation_cmd("git status --short --untracked-files=all") is True


def test_safe_cmd_allows_git_diff():
    assert cac.is_safe_validation_cmd(
        "git diff -- scripts/agent/run_claude_task.sh docs/agent_workflow.md"
    ) is True


def test_safe_cmd_rejects_semicolon():
    assert cac.is_safe_validation_cmd("python3 foo.py; curl evil.com") is False


def test_safe_cmd_rejects_pipe():
    assert cac.is_safe_validation_cmd("python3 foo.py | grep secret") is False


def test_safe_cmd_rejects_ampersand():
    assert cac.is_safe_validation_cmd("python3 foo.py && wget evil.com") is False


def test_safe_cmd_rejects_subshell():
    assert cac.is_safe_validation_cmd("python3 $(cat .env)") is False


def test_safe_cmd_rejects_redirect():
    assert cac.is_safe_validation_cmd("python3 foo.py > /tmp/out") is False


def test_safe_cmd_rejects_curl():
    assert cac.is_safe_validation_cmd("curl https://example.com") is False


def test_safe_cmd_rejects_wget():
    assert cac.is_safe_validation_cmd("wget https://example.com") is False


def test_safe_cmd_rejects_pip():
    assert cac.is_safe_validation_cmd("pip install torch") is False


def test_safe_cmd_rejects_npm():
    assert cac.is_safe_validation_cmd("npm install") is False


def test_safe_cmd_rejects_git_push():
    assert cac.is_safe_validation_cmd("git push origin main") is False


def test_safe_cmd_rejects_git_pull():
    assert cac.is_safe_validation_cmd("git pull") is False


def test_safe_cmd_rejects_git_fetch():
    assert cac.is_safe_validation_cmd("git fetch") is False


def test_safe_cmd_rejects_rm_rf():
    assert cac.is_safe_validation_cmd("rm -rf /") is False


def test_safe_cmd_rejects_protected_dotenv():
    assert cac.is_safe_validation_cmd("python3 .env") is False


def test_safe_cmd_rejects_protected_secrets():
    assert cac.is_safe_validation_cmd("test -f secrets/key.txt") is False


def test_safe_cmd_rejects_unknown_family_bash():
    assert cac.is_safe_validation_cmd("bash run.sh") is False


def test_safe_cmd_rejects_unknown_family_ls():
    assert cac.is_safe_validation_cmd("ls -la") is False


def test_safe_cmd_rejects_empty():
    assert cac.is_safe_validation_cmd("") is False
    assert cac.is_safe_validation_cmd("   ") is False


# ---------------------------------------------------------------------------
# Integration tests: get_changed_files + violation detection
# ---------------------------------------------------------------------------

def test_no_changes_passes(tmp_path):
    repo = str(tmp_path)
    _init_git_repo(repo)
    _commit_file(repo, "README.md")

    changed = cac.get_changed_files(repo)
    assert changed == []


def test_untracked_file_is_detected(tmp_path):
    repo = str(tmp_path)
    _init_git_repo(repo)
    _commit_file(repo, "README.md")

    _write_file(repo, "new_file.py")

    changed = cac.get_changed_files(repo)
    assert "new_file.py" in changed


def test_untracked_nested_files_in_new_dir(tmp_path):
    """--untracked-files=all lists individual files inside new directories."""
    repo = str(tmp_path)
    _init_git_repo(repo)
    _commit_file(repo, "README.md")

    _write_file(repo, "newpkg/module.py")
    _write_file(repo, "newpkg/sub/helper.py")

    changed = cac.get_changed_files(repo)
    # Each file must appear individually, not as a collapsed directory entry
    assert "newpkg/module.py" in changed
    assert "newpkg/sub/helper.py" in changed
    # The directory itself must not appear as a single entry
    assert "newpkg/" not in changed


def test_modified_file_is_detected(tmp_path):
    repo = str(tmp_path)
    _init_git_repo(repo)
    _commit_file(repo, "existing.py", "original\n")

    _write_file(repo, "existing.py", "modified\n")

    changed = cac.get_changed_files(repo)
    assert "existing.py" in changed


def test_allowed_change_passes(tmp_path):
    repo = str(tmp_path)
    _init_git_repo(repo)
    _commit_file(repo, "README.md")

    # Create a file that IS in the allowed list
    _write_file(repo, "allowed.py")

    content = _task_content(["allowed.py"])
    allowed = cac.parse_allowed_files(content)
    changed = cac.get_changed_files(repo)

    assert "allowed.py" in changed
    violations = [f for f in changed if f not in allowed]
    assert violations == []


def test_violation_detected(tmp_path):
    repo = str(tmp_path)
    _init_git_repo(repo)
    _commit_file(repo, "README.md")

    # Create a file that is NOT in the allowed list
    _write_file(repo, "forbidden.py")

    content = _task_content(["allowed_only.py"])
    allowed = cac.parse_allowed_files(content)
    changed = cac.get_changed_files(repo)

    assert "forbidden.py" in changed
    violations = [f for f in changed if f not in allowed]
    assert "forbidden.py" in violations


def test_disallowed_changed_file_is_flagged(tmp_path):
    """A changed file outside the allowed list is correctly flagged as a violation."""
    repo = str(tmp_path)
    _init_git_repo(repo)
    _commit_file(repo, "README.md")

    _write_file(repo, "scripts/allowed.py")
    _write_file(repo, "scripts/sneaky.py")  # not in allowed list

    content = _task_content(["scripts/allowed.py"])
    allowed = cac.parse_allowed_files(content)
    changed = cac.get_changed_files(repo)

    violations = [f for f in changed if f not in allowed]
    assert "scripts/sneaky.py" in violations
    assert "scripts/allowed.py" not in violations


def test_multiple_allowed_files(tmp_path):
    repo = str(tmp_path)
    _init_git_repo(repo)
    _commit_file(repo, "README.md")

    _write_file(repo, "scripts/foo.py")
    _write_file(repo, "docs/bar.md")

    content = _task_content(["scripts/foo.py", "docs/bar.md"])
    allowed = cac.parse_allowed_files(content)
    changed = cac.get_changed_files(repo)

    assert "scripts/foo.py" in changed
    assert "docs/bar.md" in changed
    violations = [f for f in changed if f not in allowed]
    assert violations == []


def test_partial_violation(tmp_path):
    repo = str(tmp_path)
    _init_git_repo(repo)
    _commit_file(repo, "README.md")

    _write_file(repo, "allowed.py")
    _write_file(repo, "forbidden.py")

    content = _task_content(["allowed.py"])
    allowed = cac.parse_allowed_files(content)
    changed = cac.get_changed_files(repo)

    violations = [f for f in changed if f not in allowed]
    assert "forbidden.py" in violations
    assert "allowed.py" not in violations


# ---------------------------------------------------------------------------
# Integration tests: rename/copy path handling
# ---------------------------------------------------------------------------

def test_rename_source_path_violation(tmp_path):
    """Renaming forbidden.py -> allowed.py must fail when only allowed.py is permitted.

    Both sides of the rename must be inspected; the source path (forbidden.py)
    is not in the allowed list, so the checker must fail.
    """
    repo = str(tmp_path)
    _init_git_repo(repo)
    _commit_file(repo, "forbidden.py", "content\n")

    subprocess.run(
        ["git", "mv", "forbidden.py", "allowed.py"],
        cwd=repo, check=True, capture_output=True,
    )

    content = _task_content(["allowed.py"])  # only destination is allowed
    allowed = cac.parse_allowed_files(content)
    changed = cac.get_changed_files(repo)

    # Both source and destination must appear in the changed list
    assert "allowed.py" in changed, f"new path missing from changed: {changed}"
    assert "forbidden.py" in changed, f"old path missing from changed: {changed}"

    violations = [f for f in changed if f not in allowed]
    assert "forbidden.py" in violations, "rename source must be a violation"


def test_rename_both_paths_allowed(tmp_path):
    """Renaming forbidden.py -> allowed.py passes when both paths are in the allowed list."""
    repo = str(tmp_path)
    _init_git_repo(repo)
    _commit_file(repo, "forbidden.py", "content\n")

    subprocess.run(
        ["git", "mv", "forbidden.py", "allowed.py"],
        cwd=repo, check=True, capture_output=True,
    )

    content = _task_content(["forbidden.py", "allowed.py"])  # both sides allowed
    allowed = cac.parse_allowed_files(content)
    changed = cac.get_changed_files(repo)

    violations = [f for f in changed if f not in allowed]
    assert violations == [], f"unexpected violations: {violations}"


# ---------------------------------------------------------------------------
# Unit tests: protected path detection (relative paths)
# ---------------------------------------------------------------------------

def test_safe_cmd_rejects_data_relative():
    assert cac.is_safe_validation_cmd("python3 data/foo.py") is False


def test_safe_cmd_rejects_data_dotslash():
    assert cac.is_safe_validation_cmd("python3 ./data/foo.py") is False


def test_safe_cmd_rejects_datasets_relative():
    assert cac.is_safe_validation_cmd("python3 datasets/foo.py") is False


def test_safe_cmd_rejects_outputs_relative():
    assert cac.is_safe_validation_cmd("python3 outputs/foo.py") is False


def test_safe_cmd_rejects_checkpoints_relative():
    assert cac.is_safe_validation_cmd("python3 checkpoints/foo.py") is False


def test_safe_cmd_rejects_grep_data():
    assert cac.is_safe_validation_cmd("grep -q OK data/foo.txt") is False


def test_safe_cmd_rejects_test_dotenv():
    assert cac.is_safe_validation_cmd("test -f .env") is False


# ---------------------------------------------------------------------------
# Standalone runner (no pytest needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unit_tests = [
        test_parse_allowed_files_basic,
        test_parse_allowed_files_double_quoted,
        test_parse_allowed_files_no_section,
        test_parse_allowed_files_empty_section,
        test_parse_allowed_files_ignores_prose,
        test_parse_validation_commands_basic,
        test_parse_validation_commands_multiple_blocks,
        test_parse_validation_commands_skips_comments,
        test_parse_validation_commands_no_section,
        test_safe_cmd_allows_python3_with_args,
        test_safe_cmd_allows_pytest,
        test_safe_cmd_allows_python3_m_pytest,
        test_safe_cmd_allows_test_f,
        test_safe_cmd_allows_git_status,
        test_safe_cmd_allows_git_status_untracked_all,
        test_safe_cmd_allows_git_diff,
        test_safe_cmd_rejects_semicolon,
        test_safe_cmd_rejects_pipe,
        test_safe_cmd_rejects_ampersand,
        test_safe_cmd_rejects_subshell,
        test_safe_cmd_rejects_redirect,
        test_safe_cmd_rejects_curl,
        test_safe_cmd_rejects_wget,
        test_safe_cmd_rejects_pip,
        test_safe_cmd_rejects_npm,
        test_safe_cmd_rejects_git_push,
        test_safe_cmd_rejects_git_pull,
        test_safe_cmd_rejects_git_fetch,
        test_safe_cmd_rejects_rm_rf,
        test_safe_cmd_rejects_protected_dotenv,
        test_safe_cmd_rejects_protected_secrets,
        test_safe_cmd_rejects_unknown_family_bash,
        test_safe_cmd_rejects_unknown_family_ls,
        test_safe_cmd_rejects_empty,
        test_safe_cmd_rejects_data_relative,
        test_safe_cmd_rejects_data_dotslash,
        test_safe_cmd_rejects_datasets_relative,
        test_safe_cmd_rejects_outputs_relative,
        test_safe_cmd_rejects_checkpoints_relative,
        test_safe_cmd_rejects_grep_data,
        test_safe_cmd_rejects_test_dotenv,
    ]
    tmp_tests = [
        test_no_changes_passes,
        test_untracked_file_is_detected,
        test_untracked_nested_files_in_new_dir,
        test_modified_file_is_detected,
        test_allowed_change_passes,
        test_violation_detected,
        test_disallowed_changed_file_is_flagged,
        test_multiple_allowed_files,
        test_partial_violation,
        test_rename_source_path_violation,
        test_rename_both_paths_allowed,
    ]

    passed = failed = 0

    for fn in unit_tests:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL: {fn.__name__}")
            traceback.print_exc()
            failed += 1

    for fn in tmp_tests:
        with tempfile.TemporaryDirectory() as td:
            try:
                fn(Path(td))
                print(f"PASS: {fn.__name__}")
                passed += 1
            except Exception:
                print(f"FAIL: {fn.__name__}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
