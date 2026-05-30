#!/usr/bin/env python3
"""Run a guarded pre-SNS long256 + tile localization integrated report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_long256_tile_report import (  # noqa: E402
    Long256TileReportError,
    json_safe,
    load_long256_tile_report_config,
    run_long256_tile_report,
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 scripts/inference/run_pre_sns_v3_long256_tile_report.py <config.json>", file=sys.stderr)
        return 2
    try:
        config = load_long256_tile_report_config(argv[1])
        result = run_long256_tile_report(config)
    except Long256TileReportError as exc:
        print(f"PRE_SNS_V3_LONG256_TILE_REPORT_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
