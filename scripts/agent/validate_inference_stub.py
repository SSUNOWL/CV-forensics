#!/usr/bin/env python3
"""Validate the fake inference stub against a config of fake inputs.

Usage:
    python3 scripts/agent/validate_inference_stub.py configs/inference/fake_inputs.example.json

Checks:
- Config loads and has required safety flags
- All required scenarios are present
- Every fake output validates against the task 0007 schema
- Reasons are non-empty and deterministic
- Conditional localization behavior is correct
- No generated output contains protected paths, URLs, or real file references
Exits 0 on full pass, non-zero with actionable errors on any failure.
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
from typing import Any, Dict, List


_PROTECTED_PATH_COMPONENTS = {"secrets", "data", "datasets", "outputs", "checkpoints"}
_PROTECTED_URL_PREFIXES = ("http://", "https://", "s3://", "gs://", "hf://")
_PROTECTED_LOCAL_PREFIXES = ("/home/", "/mnt/", "/root/", "/Users/")


def _check_no_protected_reference(value: Any, context: str) -> None:
    if not isinstance(value, str) or not value:
        return
    lower = value.lower()
    for prefix in _PROTECTED_URL_PREFIXES:
        if lower.startswith(prefix):
            raise ValueError(f"{context}: contains URL-like reference: {value!r}")
    for prefix in _PROTECTED_LOCAL_PREFIXES:
        if value.startswith(prefix):
            raise ValueError(f"{context}: contains local machine path: {value!r}")
    if "/" in value or "\\" in value:
        import re
        parts = re.split(r"[/\\]", value)
        for part in parts:
            if part in _PROTECTED_PATH_COMPONENTS:
                raise ValueError(
                    f"{context}: contains protected path component {part!r} in {value!r}"
                )


def _scan_output_for_protected(output: Dict[str, Any], input_id: str) -> None:
    for key in ("mask_ref", "visualization_ref", "model_stage", "family_policy"):
        val = output.get(key)
        if val is not None:
            _check_no_protected_reference(val, f"output[{input_id}][{key}]")
    reason = output.get("reason", "")
    _check_no_protected_reference(reason, f"output[{input_id}][reason]")


def _load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"FAIL: Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"FAIL: Invalid JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python3 scripts/agent/validate_inference_stub.py <fake_input_config.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = _load_config(config_path)
    failures: List[str] = []

    # --- Safety flags ---
    for flag in ("dry_run", "no_download", "no_training", "no_network"):
        if not cfg.get(flag, False):
            failures.append(f"Config missing required safety flag: {flag!r}: true")
    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        sys.exit(1)

    raw_inputs: List[Dict] = cfg.get("fake_inputs", [])
    if not isinstance(raw_inputs, list) or len(raw_inputs) == 0:
        print("FAIL: 'fake_inputs' array is missing or empty.", file=sys.stderr)
        sys.exit(1)

    # --- Import modules ---
    try:
        from cv_forensics.inference_stub import (
            FakeInput,
            REQUIRED_SCENARIOS,
            check_fake_input_config_safety,
            run_fake_inference,
        )
        from cv_forensics.model_output_schema import validate_forensic_output
    except ImportError as exc:
        print(f"FAIL: Cannot import modules: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Recursively check the full raw config for unsafe references ---
    try:
        check_fake_input_config_safety(cfg, context="fake_input_config")
    except ValueError as exc:
        failures.append(f"unsafe config: {exc}")
    if failures:
        print("FAIL: Unsafe raw input config detected:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)

    # --- Required scenarios present ---
    present_scenarios = {raw.get("scenario") for raw in raw_inputs}
    for scenario in REQUIRED_SCENARIOS:
        if scenario not in present_scenarios:
            failures.append(f"Required scenario missing from fake_inputs: {scenario!r}")

    # --- Run inference and validate outputs ---
    outputs_by_scenario: Dict[str, Dict] = {}
    for raw in raw_inputs:
        input_id = raw.get("input_id", "<unknown>")
        scenario = raw.get("scenario", "<unknown>")
        try:
            fake_input = FakeInput.from_dict(raw)
            output = run_fake_inference(fake_input)
        except (ValueError, TypeError, KeyError) as exc:
            failures.append(f"[{input_id}] run_fake_inference raised: {exc}")
            continue

        # Schema validation
        try:
            validate_forensic_output(output)
        except (ValueError, TypeError) as exc:
            failures.append(f"[{input_id}] schema validation failed: {exc}")
            continue

        # Non-empty reason
        reason = output.get("reason", "")
        if not isinstance(reason, str) or not reason.strip():
            failures.append(f"[{input_id}] reason is empty or blank")

        # Scan for protected references in output
        try:
            _scan_output_for_protected(output, input_id)
        except ValueError as exc:
            failures.append(str(exc))

        outputs_by_scenario[scenario] = output

        # Determinism check: run a second time
        try:
            output2 = run_fake_inference(fake_input)
        except Exception as exc:
            failures.append(f"[{input_id}] second run raised: {exc}")
            continue
        if output2.get("reason") != output.get("reason"):
            failures.append(f"[{input_id}] reason is not deterministic")
        if output2.get("class") != output.get("class"):
            failures.append(f"[{input_id}] class is not deterministic")

    # --- Conditional localization checks ---
    from cv_forensics.model_output_schema import (
        LOCALIZATION_ACTIVATED,
        LOCALIZATION_NOT_APPLICABLE,
        LOCALIZATION_SKIPPED_BELOW_THRESHOLD,
        CLASS_LABEL_TAMPERED,
        CLASS_LABEL_REAL,
    )

    tl = outputs_by_scenario.get("tampered_localized")
    if tl:
        if tl.get("localization_head") != LOCALIZATION_ACTIVATED:
            failures.append(
                f"tampered_localized: expected localization_head='activated', "
                f"got {tl.get('localization_head')!r}"
            )
        ts = float(tl.get("tampered_score", 0))
        tau = float(tl.get("threshold_tau", 1))
        if ts < tau:
            failures.append(
                f"tampered_localized: tampered_score {ts} < threshold_tau {tau}, "
                "but localization was activated"
            )

    tb = outputs_by_scenario.get("tampered_below_threshold")
    if tb:
        if tb.get("localization_head") != LOCALIZATION_SKIPPED_BELOW_THRESHOLD:
            failures.append(
                f"tampered_below_threshold: expected localization_head='skipped_below_threshold', "
                f"got {tb.get('localization_head')!r}"
            )
        ts = float(tb.get("tampered_score", 1))
        tau = float(tb.get("threshold_tau", 0))
        if ts >= tau:
            failures.append(
                f"tampered_below_threshold: tampered_score {ts} >= threshold_tau {tau}, "
                "but localization should have been skipped"
            )

    rc = outputs_by_scenario.get("real_clean")
    if rc:
        loc = rc.get("localization_head")
        if loc == LOCALIZATION_ACTIVATED:
            failures.append(
                f"real_clean: localization_head should not be 'activated', got {loc!r}"
            )

    # --- Evidence signal ids present in at least one output ---
    all_signal_ids = set()
    for out in outputs_by_scenario.values():
        for ev in out.get("evidence", []):
            all_signal_ids.add(ev.get("signal_id", ""))
    if not all_signal_ids:
        failures.append("No evidence signals produced across any scenario output")

    # --- Report ---
    if failures:
        print("FAIL: Validation found the following errors:", file=sys.stderr)
        for i, f in enumerate(failures, 1):
            print(f"  [{i}] {f}", file=sys.stderr)
        sys.exit(1)

    print(
        f"OK: validate_inference_stub passed "
        f"({len(raw_inputs)} fake inputs, {len(outputs_by_scenario)} scenarios validated)."
    )


if __name__ == "__main__":
    main()
