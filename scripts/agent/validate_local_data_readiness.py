#!/usr/bin/env python3
"""Validate a local data readiness config against the readiness gate.

Usage:
    python3 scripts/agent/validate_local_data_readiness.py configs/local_data/readiness.example.json

Checks:
- Config loads and passes safety check
- dry_run, no_download, no_training, no_network, no_outputs, no_checkpoints are all true
- Config validates without errors
- Readiness evaluation passes
- protected_path_exclusions are present
- No protected paths, URLs, secrets, or checkpoint/output paths in manifest_refs

Exits 0 on full pass, non-zero with actionable errors on any failure.
"""
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

_REPO_ROOT = next(
    (p for p in _Path(__file__).resolve().parents if (p / "src" / "cv_forensics").is_dir()),
    None,
)
if _REPO_ROOT is not None:
    _SRC_ROOT = str(_REPO_ROOT / "src")
    if _SRC_ROOT not in _sys.path:
        _sys.path.insert(0, _SRC_ROOT)

import json
import sys


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        print(
            f"Usage: {sys.argv[0]} <readiness_config.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = sys.argv[1]

    # --- Load raw JSON ---
    try:
        with open(config_path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"Cannot load config {config_path!r}: {exc}")

    print(f"Loaded config: {config_path}")

    # --- Import module ---
    try:
        from cv_forensics.local_data_gate import (
            check_local_data_readiness_config_safety,
            validate_local_data_readiness_config,
            evaluate_local_data_readiness,
            summarize_local_data_readiness,
        )
    except ImportError as exc:
        _fail(f"Cannot import cv_forensics.local_data_gate: {exc}")

    # --- Safety check must pass for the example config ---
    try:
        check_local_data_readiness_config_safety(raw)
    except ValueError as exc:
        _fail(f"Config failed safety check: {exc}")
    print("  Safety check: PASS")

    # --- Safety checker must reject unsafe values ---
    unsafe_cases = [
        ({"path": "/home/user/datasets/train"}, "absolute local path"),
        ({"path": "/mnt/storage/images"}, "/mnt path"),
        ({"url": "https://example.com/model"}, "https URL"),
        ({"ref": "s3://bucket/key"}, "s3 URL"),
        ({"ref": "hf://org/repo"}, "hf:// URL"),
        ({"folder": "data/images"}, "protected dir 'data'"),
        ({"folder": "checkpoints/run1"}, "protected dir 'checkpoints'"),
        ({"cfg": ".env"}, ".env reference"),
        ({"cfg": ".env.local"}, ".env.* reference"),
        ({"api_key": "abc123"}, "secret key name 'api_key'"),
        ({"info": "token abc123"}, "secret-looking value with 'token'"),
        ({"drive": "C:\\models\\weights.pt"}, "Windows drive path"),
        ({"auth": "Bearer xyz"}, "secret key 'auth'"),
    ]
    for bad_cfg, label in unsafe_cases:
        try:
            check_local_data_readiness_config_safety(bad_cfg)
            _fail(f"Safety checker did NOT reject {label}: {bad_cfg!r}")
        except ValueError:
            pass
    print("  Safety checker correctly rejects unsafe values: PASS")

    # --- Safety checker must allow ordinary prose ---
    safe_prose_cases = [
        {"note": "authoritative guidance on policy"},
        {"note": "authentication policy document"},
        {"description": "local readiness gate for the project"},
    ]
    for safe_cfg in safe_prose_cases:
        try:
            check_local_data_readiness_config_safety(safe_cfg)
        except ValueError as exc:
            _fail(f"Safety checker incorrectly rejected safe prose {safe_cfg!r}: {exc}")
    print("  Safety checker allows ordinary prose: PASS")

    # --- Validate required boolean flags ---
    for flag in ("dry_run", "no_download", "no_training", "no_network", "no_outputs", "no_checkpoints"):
        if raw.get(flag) is not True:
            _fail(
                f"Config {config_path!r} must have {flag!r}: true "
                f"(got {raw.get(flag)!r})."
            )
    print("  Required guardrail flags: PASS")

    # --- Validate config schema ---
    try:
        config = validate_local_data_readiness_config(raw)
    except ValueError as exc:
        _fail(f"Config validation failed: {exc}")
    print(
        f"  Config validation: PASS "
        f"(schema_version={config.schema_version!r}, "
        f"dry_run={config.dry_run}, "
        f"no_download={config.no_download})"
    )

    # --- Check protected_path_exclusions are present ---
    if not config.protected_path_exclusions:
        _fail("protected_path_exclusions must be present and non-empty.")
    print(f"  protected_path_exclusions present ({len(config.protected_path_exclusions)} entries): PASS")

    # --- Check manifest_refs contain no protected paths ---
    _PROTECTED_PREFIXES = ("checkpoints", "outputs", "data/", "datasets/", "/home", "http", "https", "s3://")
    protected_refs = [
        ref for ref in config.manifest_refs
        if any(p in str(ref).lower() for p in _PROTECTED_PREFIXES)
    ]
    if protected_refs:
        _fail(f"manifest_refs contains protected-looking values: {protected_refs}")
    print("  manifest_refs: no protected paths detected: PASS")

    # --- Run readiness evaluation ---
    try:
        result = evaluate_local_data_readiness(config)
    except Exception as exc:
        _fail(f"evaluate_local_data_readiness raised an exception: {exc}")

    summary = summarize_local_data_readiness(result)
    print(f"  Readiness evaluation: {summary}")

    # The example config is symbolic with local_data_approved=false, so evaluation
    # is expected to produce ready=True (no errors) for the symbolic dry-run config.
    if not result.ready:
        error_issues = [i for i in result.issues if i.severity == "error"]
        _fail(
            f"Readiness evaluation reports not ready. Errors:\n"
            + "\n".join(f"  [{i.field}] {i.message}" for i in error_issues)
        )
    print("  Readiness evaluation: PASS (ready=True)")

    # --- Verify dry_run_safe ---
    if not result.dry_run_safe:
        _fail("Result has dry_run_safe=False.")
    print("  dry_run_safe: PASS")

    # --- Verify has_protected_path_exclusions ---
    if not result.has_protected_path_exclusions:
        _fail("Result has has_protected_path_exclusions=False.")
    print("  has_protected_path_exclusions: PASS")

    # --- Verify summary contains LOCAL_DATA_READINESS_OK ---
    if "LOCAL_DATA_READINESS_OK" not in summary:
        _fail(f"Summary does not contain LOCAL_DATA_READINESS_OK: {summary!r}")
    print("  Summary contains LOCAL_DATA_READINESS_OK: PASS")

    print(f"\nAll validation checks passed for {config_path!r}.")


if __name__ == "__main__":
    main()
