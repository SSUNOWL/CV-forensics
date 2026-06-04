#!/usr/bin/env python3
"""Validate basic SNS robustness evaluation configs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_sns_robustness_eval import (  # noqa: E402
    CONFIG_OK_MARKER,
    load_sns_robustness_eval_config,
    validate_sns_robustness_eval_config,
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 scripts/agent/validate_pre_sns_v3_sns_robustness_eval_config.py <config.json>", file=sys.stderr)
        return 2
    try:
        config = load_sns_robustness_eval_config(argv[1])
        errors = validate_sns_robustness_eval_config(config, require_exists=False)
    except Exception as exc:
        print(f"PRE_SNS_V3_SNS_ROBUSTNESS_EVAL_CONFIG_INVALID: {exc}", file=sys.stderr)
        return 1
    if errors:
        print("PRE_SNS_V3_SNS_ROBUSTNESS_EVAL_CONFIG_INVALID", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(CONFIG_OK_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
