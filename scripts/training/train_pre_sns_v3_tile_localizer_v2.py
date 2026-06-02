#!/usr/bin/env python3
"""Run guarded pre-SNS v3 tile-localizer v2 training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_tile_localizer_v2_training import TileLocalizerV2TrainingError, json_safe, load_config, run_training  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 scripts/training/train_pre_sns_v3_tile_localizer_v2.py <config.json>", file=sys.stderr)
        return 2
    try:
        result = run_training(load_config(argv[1]))
    except TileLocalizerV2TrainingError as exc:
        print(f"PRE_SNS_V3_TILE_LOCALIZER_V2_TRAINING_FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(json_safe(result), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
