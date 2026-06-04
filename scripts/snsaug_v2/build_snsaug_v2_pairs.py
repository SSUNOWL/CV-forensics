#!/usr/bin/env python3
"""Build deterministic SNSAug V2 clean/augmented pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2 import SNSAugV2PairGenerator  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic SNSAug V2 pair data.")
    parser.add_argument("--input-manifest", required=False, help="Input manifest path")
    parser.add_argument("--output-root", required=False, help="External output root")
    parser.add_argument("--profiles", nargs="+", default=["combined_sns_realistic"], help="Profiles to generate")
    parser.add_argument("--severity", default="medium", help="Severity level")
    parser.add_argument("--seed", type=int, default=0, help="Base seed")
    parser.add_argument("--split", default="train", help="Split name")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum samples")
    parser.add_argument("--samples-per-class", type=int, default=None, help="Optional per-class limit")
    parser.add_argument("--output-size", type=int, default=None, help="Optional output long side")
    parser.add_argument("--font-path", required=False, help="Optional local font path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input_manifest or not args.output_root:
        parser.print_help()
        return 0
    generator = SNSAugV2PairGenerator(
        input_manifest=args.input_manifest,
        output_root=args.output_root,
        profiles=list(args.profiles),
        severity=args.severity,
        seed=args.seed,
        split=args.split,
        max_samples=args.max_samples,
        samples_per_class=args.samples_per_class,
        output_size=args.output_size,
        font_path=args.font_path,
    )
    result = generator.run()
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
