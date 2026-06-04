#!/usr/bin/env python3
"""Run basic SNS-like robustness evaluation for the frozen pre-SNS best bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_sns_robustness_eval import (  # noqa: E402
    SNSRobustnessEvalError,
    json_safe,
    load_sns_robustness_eval_config,
    run_sns_robustness_eval,
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 scripts/evaluation/run_pre_sns_v3_sns_robustness_eval.py <config.json>", file=sys.stderr)
        return 2
    try:
        config = load_sns_robustness_eval_config(argv[1])
        summary = run_sns_robustness_eval(config)
    except SNSRobustnessEvalError as exc:
        print(f"PRE_SNS_V3_SNS_ROBUSTNESS_EVAL_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
