#!/usr/bin/env python3
"""Prepare a dry-run pre-SNS v3 tile localization training plan."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_tile_localization import (  # noqa: E402
    TileLocalizationError,
    json_safe,
    load_tile_localization_config,
    run_tile_training_prepare,
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 scripts/training/prepare_pre_sns_v3_tile_localization.py <config.json>", file=sys.stderr)
        return 2
    try:
        config = load_tile_localization_config(argv[1])
        summary = run_tile_training_prepare(config)
    except TileLocalizationError as exc:
        print(f"PRE_SNS_V3_TILE_LOCALIZATION_PREPARE_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
