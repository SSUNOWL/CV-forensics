#!/usr/bin/env python3
"""Validate dataset and experiment config JSON files against config_schema rules.

Usage:
    python3 scripts/agent/validate_config_schema.py \\
        configs/datasets.example.json \\
        configs/experiments/smoke_baseline.json
"""

import json
import sys
import os

# Path setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")

for _p in (_REPO_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cv_forensics.config_schema import (  # noqa: E402
    validate_datasets_config,
    validate_experiment_config,
    extract_dataset_ids,
)


def _load_json(path: str) -> dict:
    if not os.path.isfile(path):
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"ERROR: Invalid JSON in {path}: {exc}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: validate_config_schema.py <datasets_json> <experiment_json>",
            file=sys.stderr,
        )
        sys.exit(1)

    datasets_path = sys.argv[1]
    experiment_path = sys.argv[2]

    datasets_config = _load_json(datasets_path)
    experiment_config = _load_json(experiment_path)

    all_errors: list = []

    # Validate datasets config
    ds_errors = validate_datasets_config(datasets_config)
    for e in ds_errors:
        all_errors.append(f"[datasets] {e}")

    # Extract known dataset ids for cross-reference
    known_ids = extract_dataset_ids(datasets_config)

    # Validate experiment config
    exp_errors = validate_experiment_config(experiment_config, known_dataset_ids=known_ids)
    for e in exp_errors:
        all_errors.append(f"[experiment] {e}")

    if all_errors:
        print("VALIDATION FAILED", file=sys.stderr)
        for err in all_errors:
            print(f"  {err}", file=sys.stderr)
        sys.exit(1)

    print(
        f"OK: {datasets_path} and {experiment_path} passed all config schema checks."
    )


if __name__ == "__main__":
    main()
