#!/usr/bin/env python3
"""Generate a guarded pre-SNS baseline report from explicit result JSON paths."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_baseline_report import generate_baseline_report, load_report_config, validate_report_config  # noqa: E402


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        return _fail("usage: generate_pre_sns_baseline_report.py <config.json>")
    try:
        raw = load_report_config(argv[0])
    except Exception as exc:
        return _fail(f"failed to read baseline report config: {exc}")
    errors = validate_report_config(raw)
    if errors:
        print("pre-SNS baseline report config validation failed:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    try:
        summary = generate_baseline_report(raw)
    except Exception as exc:
        return _fail(str(exc))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
