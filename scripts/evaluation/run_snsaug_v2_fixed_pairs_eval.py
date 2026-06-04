#!/usr/bin/env python3
"""Run snsaug v2 fixed-pairs evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2_fixed_pairs_eval import (  # noqa: E402
    SNSAugV2FixedPairsEvalError,
    load_snsaug_v2_fixed_pairs_eval_config,
    run_snsaug_v2_fixed_pairs_eval,
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"-h", "--help"}:
        print("usage: run_snsaug_v2_fixed_pairs_eval.py <config-path>")
        return 0
    if len(argv) != 1:
        print("usage: run_snsaug_v2_fixed_pairs_eval.py <config-path>")
        return 2
    try:
        summary = run_snsaug_v2_fixed_pairs_eval(load_snsaug_v2_fixed_pairs_eval_config(argv[0]))
    except SNSAugV2FixedPairsEvalError as exc:
        print(str(exc))
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
