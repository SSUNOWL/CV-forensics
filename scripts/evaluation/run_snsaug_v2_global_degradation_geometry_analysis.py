#!/usr/bin/env python3
"""Run SNSAug v2 global degradation and geometry shift analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2_global_degradation_geometry_analysis import (  # noqa: E402
    SNSAugV2GlobalDegradationGeometryAnalysisError,
    load_snsaug_v2_global_degradation_geometry_analysis_config,
    run_snsaug_v2_global_degradation_geometry_analysis,
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"-h", "--help"}:
        print("usage: run_snsaug_v2_global_degradation_geometry_analysis.py [--dry-run] <config-path>")
        return 0
    dry_run = False
    if "--dry-run" in argv:
        dry_run = True
        argv.remove("--dry-run")
    if len(argv) != 1:
        print("usage: run_snsaug_v2_global_degradation_geometry_analysis.py [--dry-run] <config-path>")
        return 2
    try:
        config = load_snsaug_v2_global_degradation_geometry_analysis_config(argv[0])
        summary = run_snsaug_v2_global_degradation_geometry_analysis(config, dry_run=dry_run)
    except SNSAugV2GlobalDegradationGeometryAnalysisError as exc:
        print(str(exc))
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
