#!/usr/bin/env python3
"""Run the fake inference stub against a config of fake inputs.

Usage:
    python3 scripts/agent/run_fake_inference.py configs/inference/fake_inputs.example.json

Prints a JSON summary of each inference result to stdout.
Exits non-zero with an actionable error message on failure.
Does not write files, create outputs/, or access protected paths.
"""
from __future__ import annotations

# Allow direct execution from the repository root without installing the package.
from pathlib import Path as _Path
import sys as _sys

_REPO_ROOT = next(
    (p for p in _Path(__file__).resolve().parents if (p / "src" / "cv_forensics").is_dir()),
    None,
)
if _REPO_ROOT is not None:
    _SRC_ROOT = _REPO_ROOT / "src"
    if str(_SRC_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_SRC_ROOT))

import json
import sys


def _load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def _check_safe_flags(cfg: dict, path: str) -> None:
    for flag in ("dry_run", "no_download", "no_training", "no_network"):
        if not cfg.get(flag, False):
            print(
                f"ERROR: Config {path} must have {flag!r}: true. "
                "This script only runs in dry-run / no-network mode.",
                file=sys.stderr,
            )
            sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python3 scripts/agent/run_fake_inference.py <fake_input_config.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = _load_config(config_path)
    _check_safe_flags(cfg, config_path)

    raw_inputs = cfg.get("fake_inputs")
    if not isinstance(raw_inputs, list) or len(raw_inputs) == 0:
        print("ERROR: Config must have a non-empty 'fake_inputs' array.", file=sys.stderr)
        sys.exit(1)

    # Import after config validation to give clear errors on config issues first
    try:
        from cv_forensics.inference_stub import (
            FakeInput,
            check_fake_input_config_safety,
            run_fake_inference,
        )
    except ImportError as exc:
        print(f"ERROR: Cannot import inference_stub: {exc}", file=sys.stderr)
        sys.exit(1)

    # Reject unsafe raw config before running any fake inference.
    try:
        check_fake_input_config_safety(cfg, context="fake_input_config")
    except ValueError as exc:
        print(
            "ERROR: Unsafe reference(s) detected in fake input config. "
            "No inference was run.",
            file=sys.stderr,
        )
        print(f"  {exc}", file=sys.stderr)
        sys.exit(1)

    results = []
    errors = []
    for raw in raw_inputs:
        input_id = raw.get("input_id", "<unknown>")
        try:
            fake_input = FakeInput.from_dict(raw)
            output = run_fake_inference(fake_input)
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"input_id": input_id, "error": str(exc)})
            continue

        results.append({
            "input_id": input_id,
            "scenario": raw.get("scenario"),
            "class": output["class"],
            "family": output["family"],
            "localization_head": output["localization_head"],
            "mask_area_pct": output["mask_area_pct"],
            "tampered_score": output["tampered_score"],
            "threshold_tau": output["threshold_tau"],
            "evidence_count": len(output.get("evidence", [])),
            "reason_snippet": output.get("reason", "")[:80],
        })

    summary = {
        "total": len(raw_inputs),
        "succeeded": len(results),
        "failed": len(errors),
        "results": results,
    }
    if errors:
        summary["errors"] = errors

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
