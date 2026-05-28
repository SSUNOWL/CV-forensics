#!/usr/bin/env python3
"""Run a guarded pre-SNS v3 dual-scale calibrated report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_dual_scale_report import load_dual_report_config, run_dual_scale_report, validate_dual_report_config  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 scripts/inference/run_pre_sns_v3_dual_scale_report.py <config.json>", file=sys.stderr)
        return 2
    config = load_dual_report_config(argv[1])
    errors = validate_dual_report_config(config, require_exists=True)
    if errors:
        print("dual-scale report config validation failed:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    result = run_dual_scale_report(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
