#!/usr/bin/env python3
"""Validate dataset manifest JSON files against dataset_manifest schema rules.

Usage:
    python3 scripts/agent/validate_dataset_manifest.py \\
        configs/manifests/community_forensics_small.example.json \\
        configs/manifests/sid_set.example.json \\
        configs/manifests/combined_smoke_manifest.example.json

Accepts one or more positional manifest JSON paths.
Validates cross-manifest constraints when multiple manifests are provided.
Does not access actual dataset folders, image files, or network resources.
"""

import json
import os
import sys

# Path setup: allow import of cv_forensics without package installation.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")

for _p in (_REPO_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cv_forensics.dataset_manifest import (  # noqa: E402
    load_manifest,
    validate_manifests,
)


def _load_or_exit(path: str) -> dict:
    if not os.path.isfile(path):
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        return load_manifest(path)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(f"ERROR: Cannot read {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: validate_dataset_manifest.py <manifest.json> [manifest2.json ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    paths = sys.argv[1:]
    manifests = [_load_or_exit(p) for p in paths]

    results = validate_manifests(manifests)

    all_errors = []
    for i, errors in results.items():
        for err in errors:
            all_errors.append(f"[{paths[i]}] {err}")

    if all_errors:
        print("VALIDATION FAILED", file=sys.stderr)
        for err in all_errors:
            print(f"  {err}", file=sys.stderr)
        sys.exit(1)

    label = ", ".join(os.path.basename(p) for p in paths)
    print(f"OK: {label} passed all dataset manifest checks.")


if __name__ == "__main__":
    main()
