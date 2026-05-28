#!/usr/bin/env python3
"""Run a pre-SNS v3 single-image report from a JSON config."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_report import run_v3_single_image_report  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 scripts/inference/run_pre_sns_v3_report.py <config.json>", file=sys.stderr)
        return 2
    with open(argv[1], "r", encoding="utf-8") as handle:
        config = json.load(handle)
    result = run_v3_single_image_report(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

