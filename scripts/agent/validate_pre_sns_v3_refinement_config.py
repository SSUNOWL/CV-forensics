#!/usr/bin/env python3
"""Validate pre-SNS v3 hard-case refinement configs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_refinement import APPROVED_KIND, CONFIG_OK_MARKER, load_config, validate_config  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 scripts/agent/validate_pre_sns_v3_refinement_config.py <config.json>", file=sys.stderr)
        return 2
    try:
        raw = load_config(argv[1])
        errors = validate_config(raw, require_exists=raw.get("config_kind") == APPROVED_KIND and raw.get("no_write_dry_run") is not True)
    except Exception as exc:
        print(f"PRE_SNS_V3_REFINEMENT_CONFIG_INVALID: {exc}", file=sys.stderr)
        return 1
    if errors:
        print("PRE_SNS_V3_REFINEMENT_CONFIG_INVALID", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(CONFIG_OK_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
