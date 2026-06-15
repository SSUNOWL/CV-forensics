#!/usr/bin/env python3
"""Validate 0077a final cross-model ablation metric fill config."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2_fill_final_cross_model_ablation_metrics import (  # noqa: E402
    CONFIG_OK_MARKER,
    load_snsaug_v2_fill_final_cross_model_ablation_metrics_config,
    validate_snsaug_v2_fill_final_cross_model_ablation_metrics_config,
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: validate_snsaug_v2_fill_final_cross_model_ablation_metrics_config.py <config-path>")
        return 2
    config = load_snsaug_v2_fill_final_cross_model_ablation_metrics_config(argv[0])
    errors = validate_snsaug_v2_fill_final_cross_model_ablation_metrics_config(config, require_exists=False)
    if errors:
        print("SNSAUG_V2_0077A_FILL_FINAL_CROSS_MODEL_ABLATION_METRICS_CONFIG_INVALID")
        for error in errors:
            print(error)
        return 1
    print(CONFIG_OK_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
