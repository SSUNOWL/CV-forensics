#!/usr/bin/env python3
"""Build a train-only SNSAug V2 curriculum manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2_train_curriculum import (  # noqa: E402
    SNSAugV2TrainCurriculumError,
    build_snsaug_v2_train_curriculum_manifest,
    load_snsaug_v2_train_curriculum_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a train-only SNSAug V2 curriculum manifest without training.")
    parser.add_argument("config_path", nargs="?", help="Path to snsaug_v2_train_curriculum_manifest config JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config_path:
        print("config_path is required unless using --help")
        return 2
    try:
        config = load_snsaug_v2_train_curriculum_config(args.config_path)
        summary = build_snsaug_v2_train_curriculum_manifest(config)
    except SNSAugV2TrainCurriculumError as exc:
        print(str(exc))
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
