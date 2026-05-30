#!/usr/bin/env python3
"""Run the pre-SNS clean-long256 + tile-localizer integrated report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_long256_tile_integrated_report import (  # noqa: E402
    IntegratedReportError,
    json_safe,
    load_integrated_report_config,
    run_integrated_report,
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 scripts/inference/run_pre_sns_v3_long256_tile_integrated_report.py <config.json>", file=sys.stderr)
        return 2
    try:
        result = run_integrated_report(load_integrated_report_config(argv[1]))
    except IntegratedReportError as exc:
        print(f"PRE_SNS_V3_LONG256_TILE_INTEGRATED_REPORT_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

