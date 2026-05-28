#!/usr/bin/env python3
"""Run guarded pre-SNS v3 training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_training import PreSnsV3TrainingError, load_config, run_training  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 scripts/training/train_pre_sns_v3.py <config.json>", file=sys.stderr)
        return 2
    try:
        result = run_training(load_config(argv[1]))
    except (PreSnsV3TrainingError, Exception) as exc:
        print(f"PRE_SNS_V3_TRAINING_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

