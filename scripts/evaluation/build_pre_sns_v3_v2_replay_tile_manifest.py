#!/usr/bin/env python3
"""Build pre-SNS v2 replay tile manifest and recommended medium training config."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_v2_replay_tile_manifest import (  # noqa: E402
    V2ReplayTileManifestError,
    json_safe,
    load_v2_replay_tile_manifest_config,
    run_v2_replay_tile_manifest,
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 scripts/evaluation/build_pre_sns_v3_v2_replay_tile_manifest.py <config.json>", file=sys.stderr)
        return 2
    try:
        config = load_v2_replay_tile_manifest_config(argv[1])
        summary = run_v2_replay_tile_manifest(config)
    except V2ReplayTileManifestError as exc:
        print(f"PRE_SNS_V3_V2_REPLAY_TILE_MANIFEST_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
