#!/usr/bin/env python3
"""Build train-only snsaug v2 training manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2_training_manifest import (  # noqa: E402
    SNSAugV2TrainingManifestError,
    build_snsaug_v2_training_manifest,
    load_snsaug_v2_training_manifest_config,
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: build_snsaug_v2_training_manifest.py <config-path>")
        return 2
    try:
        summary = build_snsaug_v2_training_manifest(load_snsaug_v2_training_manifest_config(argv[0]))
    except SNSAugV2TrainingManifestError as exc:
        print(str(exc))
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
