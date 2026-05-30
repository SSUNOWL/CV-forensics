#!/usr/bin/env python3
"""Run local red-mask inspection for pre-SNS long256 + tile reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_long256_tile_local_run import LocalRunError, json_safe, load_local_run_config, run_local_run  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 scripts/inference/run_pre_sns_v3_long256_tile_local_run.py <config.json>", file=sys.stderr)
        return 2
    try:
        config = load_local_run_config(argv[1])
        summary = run_local_run(config)
    except LocalRunError as exc:
        print(f"PRE_SNS_V3_LONG256_TILE_LOCAL_RUN_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
