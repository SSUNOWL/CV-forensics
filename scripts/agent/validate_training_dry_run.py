#!/usr/bin/env python3
"""Validate the dry-run training loop skeleton against a dry-run training config.

Usage:
    python3 scripts/agent/validate_training_dry_run.py configs/training/dry_run_training.example.json

Checks:
- Config loads and passes safety check
- dry_run, no_training, no_outputs, no_checkpoints are all true
- Config validates without errors
- Dry-run training simulation runs without error
- Summary includes all expected metric names
- No checkpoint or output paths are requested

Exits 0 on full pass, non-zero with actionable errors on any failure.
"""
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

_REPO_ROOT = next(
    (p for p in _Path(__file__).resolve().parents if (p / "src" / "cv_forensics").is_dir()),
    None,
)
if _REPO_ROOT is not None:
    _SRC_ROOT = _REPO_ROOT / "src"
    if str(_SRC_ROOT) not in _sys.path:
        _SRC_ROOT = str(_SRC_ROOT)
        if _SRC_ROOT not in _sys.path:
            _sys.path.insert(0, _SRC_ROOT)

import json
import math
import sys


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _check_finite(value: object, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{name} is not a numeric value: {value!r}")
    fval = float(value)  # type: ignore[arg-type]
    if math.isnan(fval) or math.isinf(fval):
        _fail(f"{name} is not finite: {value!r}")


def main() -> None:
    if len(sys.argv) != 2:
        print(
            f"Usage: {sys.argv[0]} <dry_run_training_config.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = sys.argv[1]

    # --- Load config ---
    try:
        with open(config_path) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"Cannot load config {config_path!r}: {exc}")

    print(f"Loaded config: {config_path}")

    # --- Import module ---
    try:
        from cv_forensics.training_dry_run import (
            ALLOWED_METRIC_NAMES,
            check_dry_run_training_config_safety,
            validate_dry_run_training_config,
            run_dry_training,
            dry_run_result_to_dict,
        )
    except ImportError as exc:
        _fail(f"Cannot import cv_forensics.training_dry_run: {exc}")

    # --- Safety check must pass for the example config ---
    try:
        check_dry_run_training_config_safety(raw)
    except ValueError as exc:
        _fail(f"Config failed safety check (it should pass): {exc}")
    print("  Safety check: PASS")

    # --- Safety checker must reject unsafe values ---
    unsafe_cases = [
        ({"path": "/home/user/data.json"}, "absolute local path"),
        ({"url": "https://example.com/model"}, "https URL"),
        ({"ref": "s3://bucket/key"}, "s3 URL"),
        ({"folder": "data/images"}, "protected dir 'data'"),
        ({"folder": "checkpoints/run1"}, "protected dir 'checkpoints'"),
        ({"cfg": ".env"}, ".env reference"),
        ({"cfg": ".env.local"}, ".env.* reference"),
        ({"api_key": "abc123"}, "secret key name 'api_key'"),
        ({"info": "token abc"}, "secret-looking value with 'token'"),
        ({"drive": "C:\\models\\weights.pt"}, "Windows drive path"),
        ({"ref": "ftp://host/file"}, "ftp URL"),
        ({"ref": "hf://org/repo"}, "hf:// URL"),
    ]
    for bad_cfg, label in unsafe_cases:
        try:
            check_dry_run_training_config_safety(bad_cfg)
            _fail(f"Safety checker did NOT reject {label}: {bad_cfg!r}")
        except ValueError:
            pass
    print("  Safety checker correctly rejects unsafe values: PASS")

    # --- Validate required boolean flags ---
    for flag in ("dry_run", "no_training", "no_outputs", "no_checkpoints", "no_download", "no_network"):
        if raw.get(flag) is not True:
            _fail(f"Config {config_path!r} must have {flag!r}: true (got {raw.get(flag)!r}).")
    print("  Required safety flags: PASS")

    # --- Validate config schema ---
    try:
        config = validate_dry_run_training_config(raw)
    except ValueError as exc:
        _fail(f"Config validation failed: {exc}")
    print(f"  Config validation: PASS (stage={config.stage!r}, epochs={config.epochs}, seed={config.seed})")

    # --- Verify no checkpoint or output paths in dataset_manifest_refs ---
    protected_in_refs = [
        ref for ref in config.dataset_manifest_refs
        if any(p in str(ref).lower() for p in ("checkpoints", "outputs", "data/", "/home", "http"))
    ]
    if protected_in_refs:
        _fail(f"dataset_manifest_refs contains protected-looking values: {protected_in_refs}")
    print("  dataset_manifest_refs: no protected paths detected: PASS")

    # --- Run dry-run training simulation ---
    try:
        result = run_dry_training(config)
    except Exception as exc:
        _fail(f"run_dry_training raised an exception: {exc}")
    print(f"  Dry-run simulation: PASS ({config.epochs} epoch(s) simulated)")

    # --- Validate result flags ---
    if not result.dry_run:
        _fail("Result has dry_run=False.")
    if not result.no_training:
        _fail("Result has no_training=False.")
    if not result.no_outputs:
        _fail("Result has no_outputs=False.")
    if not result.no_checkpoints:
        _fail("Result has no_checkpoints=False.")
    print("  Result safety flags: PASS")

    # --- Validate epoch summaries ---
    if len(result.epoch_summaries) != config.epochs:
        _fail(
            f"Expected {config.epochs} epoch summaries, "
            f"got {len(result.epoch_summaries)}."
        )
    for summary in result.epoch_summaries:
        if summary.n_samples <= 0:
            _fail(f"Epoch {summary.epoch}: n_samples must be positive.")
        _check_finite(summary.classification_metrics.get("accuracy"), "epoch accuracy")
        _check_finite(summary.classification_metrics.get("macro_f1"), "epoch macro_f1")
    print(f"  Epoch summaries: PASS ({len(result.epoch_summaries)} epochs)")

    # --- Validate final metrics include all expected metric names ---
    final = result.final_metrics
    missing = [m for m in config.metric_names if m not in final]
    if missing:
        _fail(f"final_metrics missing expected metric names: {missing}")
    print(f"  final_metrics covers all metric_names: PASS")

    # Check numeric metrics are finite (perturbation_robustness_drop may be None in dry-run)
    numeric_metrics = [
        "accuracy_3way", "macro_f1", "tampered_mask_iou",
        "generator_family_accuracy", "localization_activation_recall",
        "latency_ms", "fps",
    ]
    for m in numeric_metrics:
        if m in final and final[m] is not None:
            _check_finite(final[m], f"final_metrics[{m!r}]")
    print("  Numeric metrics are finite: PASS")

    # --- Validate serializable to dict ---
    try:
        output_dict = dry_run_result_to_dict(result)
        json.dumps(output_dict)
    except (TypeError, ValueError) as exc:
        _fail(f"Result is not JSON-serializable: {exc}")
    print("  JSON serialization: PASS")

    print(f"\nAll validation checks passed for {config_path!r}.")


if __name__ == "__main__":
    main()
