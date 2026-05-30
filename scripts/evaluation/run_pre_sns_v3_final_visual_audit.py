#!/usr/bin/env python3
"""Run the pre-SNS v3 final visual audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_final_visual_audit import (  # noqa: E402
    FinalVisualAuditError,
    json_safe,
    load_final_visual_audit_config,
    run_final_visual_audit,
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 scripts/evaluation/run_pre_sns_v3_final_visual_audit.py <config.json>", file=sys.stderr)
        return 2
    try:
        result = run_final_visual_audit(load_final_visual_audit_config(argv[1]))
    except FinalVisualAuditError as exc:
        print(f"PRE_SNS_V3_FINAL_VISUAL_AUDIT_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

