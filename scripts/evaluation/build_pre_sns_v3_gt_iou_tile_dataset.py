#!/usr/bin/env python3
"""Build pre-SNS GT-IoU mining records and tile-localization manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_gt_iou_tile_builder import (  # noqa: E402
    GTIoUTileBuilderError,
    json_safe,
    load_gt_iou_tile_builder_config,
    run_gt_iou_tile_builder,
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 scripts/evaluation/build_pre_sns_v3_gt_iou_tile_dataset.py <config.json>", file=sys.stderr)
        return 2
    try:
        config = load_gt_iou_tile_builder_config(argv[1])
        summary = run_gt_iou_tile_builder(config)
    except GTIoUTileBuilderError as exc:
        print(f"PRE_SNS_V3_GT_IOU_TILE_BUILDER_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
