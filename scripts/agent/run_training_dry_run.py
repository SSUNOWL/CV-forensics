#!/usr/bin/env python3
"""Run the dry-run training loop skeleton against a dry-run training config.

Usage:
    python3 scripts/agent/run_training_dry_run.py configs/training/dry_run_training.example.json

Prints a JSON summary to stdout. Does not write files, create outputs/ or checkpoints/,
or access real datasets, images, or masks.
Exits non-zero with an actionable error message on failure.
"""
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

_REPO_ROOT = next(
    (p for p in _Path(__file__).resolve().parents if (p / "src" / "cv_forensics").is_dir()),
    None,
)
if _REPO_ROOT is not None:
    _SRC_ROOT = _REPO_ROOT / "src"
    if str(_SRC_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_SRC_ROOT))

import json
import sys


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python3 scripts/agent/run_training_dry_run.py "
            "<dry_run_training_config.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = sys.argv[1]

    try:
        with open(config_path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        _fail(f"Config file not found: {config_path}")
    except json.JSONDecodeError as exc:
        _fail(f"Invalid JSON in {config_path}: {exc}")

    try:
        from cv_forensics.training_dry_run import (
            check_dry_run_training_config_safety,
            validate_dry_run_training_config,
            run_dry_training,
            dry_run_result_to_dict,
        )
    except ImportError as exc:
        _fail(f"Cannot import cv_forensics.training_dry_run: {exc}")

    try:
        check_dry_run_training_config_safety(raw)
    except ValueError as exc:
        _fail(
            f"Unsafe reference(s) detected in config {config_path!r}. "
            f"No simulation was run.\n  {exc}"
        )

    try:
        config = validate_dry_run_training_config(raw)
    except ValueError as exc:
        _fail(f"Invalid dry-run training config:\n{exc}")

    if not config.dry_run:
        _fail("Config must have dry_run=true. Refusing to run.")
    if not config.no_training:
        _fail("Config must have no_training=true. Refusing to run.")
    if not config.no_outputs:
        _fail("Config must have no_outputs=true. Refusing to run.")
    if not config.no_checkpoints:
        _fail("Config must have no_checkpoints=true. Refusing to run.")

    try:
        result = run_dry_training(config)
    except Exception as exc:
        _fail(f"Dry-run training failed: {exc}")

    output = dry_run_result_to_dict(result)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
