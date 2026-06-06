#!/usr/bin/env python3
"""Run guarded SNSAug V2 fine-tuning smoke setup."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2_finetune_runner import (  # noqa: E402
    SNSAugV2FinetuneSmokeError,
    load_snsaug_v2_finetune_smoke_config,
    run_snsaug_v2_finetune_smoke,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a guarded SNSAug V2 fine-tuning smoke job.")
    parser.add_argument("config_path", nargs="?", help="Path to smoke config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the smoke plan without writing outputs or checkpoints.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config_path:
        print("config_path is required unless using --help")
        return 2
    try:
        config = load_snsaug_v2_finetune_smoke_config(args.config_path)
        summary = run_snsaug_v2_finetune_smoke(config, dry_run=bool(args.dry_run))
    except SNSAugV2FinetuneSmokeError as exc:
        print(str(exc))
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
