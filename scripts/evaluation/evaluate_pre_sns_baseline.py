#!/usr/bin/env python3
"""Run guarded pre-SNS baseline evaluation."""

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

from cv_forensics.pre_sns_evaluation import load_evaluation_config, run_evaluation, validate_evaluation_config  # noqa: E402


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        return _fail("usage: evaluate_pre_sns_baseline.py <config.json>")
    try:
        raw = load_evaluation_config(argv[0])
    except Exception as exc:
        return _fail(f"failed to read evaluation config: {exc}")
    errors = validate_evaluation_config(raw)
    if errors:
        print("pre-SNS evaluation config validation failed:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    try:
        result = run_evaluation(raw)
    except Exception as exc:
        return _fail(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
