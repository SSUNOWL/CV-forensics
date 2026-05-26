#!/usr/bin/env python3
"""Validate guarded pre-SNS meaningful training v2 configs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_meaningful_training_v2 import (  # noqa: E402
    CONFIG_OK_MARKER,
    load_config,
    validate_config,
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 scripts/agent/validate_pre_sns_meaningful_train_v2_config.py <config.json>", file=sys.stderr)
        return 2
    try:
        raw = load_config(argv[1])
        errors = validate_config(raw, require_manifest_exists=raw.get("config_kind") == "approved_pre_sns_meaningful_training_v2")
    except Exception as exc:
        print(f"PRE_SNS_MEANINGFUL_TRAINING_V2_CONFIG_INVALID: {exc}", file=sys.stderr)
        return 1
    if errors:
        print("PRE_SNS_MEANINGFUL_TRAINING_V2_CONFIG_INVALID", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(CONFIG_OK_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
