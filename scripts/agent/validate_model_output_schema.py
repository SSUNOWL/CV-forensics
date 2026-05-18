#!/usr/bin/env python3
"""Validate model output schema example JSON against model_output_schema and explanation_templates.

Usage:
    python3 scripts/agent/validate_model_output_schema.py \\
        configs/model_output_schema.example.json \\
        configs/explanation_templates.example.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root and src to sys.path so imports work without installation.
_HERE = Path(__file__).resolve().parent          # scripts/agent
_REPO_ROOT = _HERE.parent.parent                 # project root
_SRC = _REPO_ROOT / "src"
for _p in [str(_REPO_ROOT), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cv_forensics import model_output_schema as mos
from cv_forensics import explanation_templates as et

# ---------------------------------------------------------------------------
# Required template IDs in the explanation templates JSON
# ---------------------------------------------------------------------------
_REQUIRED_TEMPLATE_IDS = [
    "real",
    "synthetic",
    "tampered_localized",
    "tampered_not_localized",
    "low_confidence",
    "family_estimated",
    "family_unavailable",
    "social_media_perturbation",
]

# Keys that look like secret fields
_SECRET_KEY_PATTERNS = ("password", "secret", "token", "credential", "api_key", "apikey")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_no_protected_values(data: object, context: str, errors: List[str]) -> None:
    """Recursively flag any protected path/URL values and secret-looking keys."""
    if isinstance(data, dict):
        for k, v in data.items():
            lower_k = k.lower()
            if any(p in lower_k for p in _SECRET_KEY_PATTERNS):
                errors.append(f"{context}.{k}: key looks like a secret field")
            _check_no_protected_values(v, f"{context}.{k}", errors)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            _check_no_protected_values(item, f"{context}[{i}]", errors)
    elif isinstance(data, str):
        if mos._is_protected_reference(data):
            errors.append(
                f"{context}: value looks like a protected path or URL: {data!r}"
            )


def _validate_single_output(
    output_dict: dict, label: str, errors: List[str]
) -> None:
    """Validate one output dict and append any error messages."""
    try:
        mos.validate_forensic_output(output_dict)
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(f"{label}: schema validation failed: {exc}")
        return

    # The static 'reason' field in the example JSON must be non-empty.
    if not output_dict.get("reason", ""):
        errors.append(f"{label}: reason field is empty — example JSON must have a non-empty reason")

    # generate_reason must produce a non-empty, deterministic string
    try:
        reason1 = et.generate_reason(output_dict)
        reason2 = et.generate_reason(output_dict)
    except Exception as exc:
        errors.append(f"{label}: generate_reason raised an exception: {exc}")
        return

    if not reason1:
        errors.append(f"{label}: generate_reason returned an empty string")
    if reason1 != reason2:
        errors.append(f"{label}: generate_reason is not deterministic (got different results)")

    # Consistency check: tampered + score >= tau → must be activated
    class_label = output_dict.get("class")
    localization_head = output_dict.get("localization_head")
    tampered_score = float(output_dict.get("tampered_score", 0.0))
    threshold_tau = float(output_dict.get("threshold_tau", 0.5))

    if (
        class_label == mos.CLASS_LABEL_TAMPERED
        and tampered_score >= threshold_tau
        and localization_head != mos.LOCALIZATION_ACTIVATED
    ):
        errors.append(
            f"{label}: tampered example with tampered_score={tampered_score} >= "
            f"threshold_tau={threshold_tau} must have localization_head='activated', "
            f"got {localization_head!r}"
        )

    if (
        class_label == mos.CLASS_LABEL_TAMPERED
        and tampered_score < threshold_tau
        and localization_head == mos.LOCALIZATION_ACTIVATED
    ):
        errors.append(
            f"{label}: tampered example with tampered_score={tampered_score} < "
            f"threshold_tau={threshold_tau} must not have localization_head='activated'"
        )

    # Class and family labels must be from the project contract
    c = output_dict.get("class", "")
    if c not in mos.CLASS_LABELS:
        errors.append(f"{label}: class {c!r} not in CLASS_LABELS {mos.CLASS_LABELS}")
    f = output_dict.get("family", "")
    if f not in mos.FAMILY_LABELS:
        errors.append(f"{label}: family {f!r} not in FAMILY_LABELS {mos.FAMILY_LABELS}")


def _validate_templates_json(
    data: dict, errors: List[str]
) -> None:
    """Validate the explanation_templates.example.json structure."""
    templates = data.get("templates")
    if not isinstance(templates, dict):
        errors.append("explanation_templates: missing or invalid 'templates' object")
        return
    for tid in _REQUIRED_TEMPLATE_IDS:
        if tid not in templates:
            errors.append(
                f"explanation_templates: missing required template id {tid!r}"
            )
        else:
            entry = templates[tid]
            if not isinstance(entry, dict):
                errors.append(f"explanation_templates.templates.{tid}: expected object")
                continue
            if "korean_template" not in entry:
                errors.append(
                    f"explanation_templates.templates.{tid}: missing 'korean_template'"
                )
            if "required_placeholders" not in entry:
                errors.append(
                    f"explanation_templates.templates.{tid}: missing 'required_placeholders'"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: validate_model_output_schema.py "
            "<model_output_schema.example.json> "
            "[<explanation_templates.example.json>]"
        )
        return 1

    schema_path = sys.argv[1]
    templates_path: Optional[str] = sys.argv[2] if len(sys.argv) >= 3 else None

    errors: List[str] = []

    # ---- Load schema example JSON ----------------------------------------
    try:
        schema_data = _load_json(schema_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: Failed to load {schema_path}: {exc}")
        return 1

    if not isinstance(schema_data, dict):
        print(f"ERROR: {schema_path} must be a JSON object")
        return 1

    # Check entire JSON for protected values
    _check_no_protected_values(schema_data, "schema_json", errors)

    # Validate primary example (top-level, skip non-output keys)
    primary: dict = {
        k: v for k, v in schema_data.items() if k not in ("_note", "examples")
    }
    _validate_single_output(primary, "primary_example", errors)

    # Validate each entry in the examples array
    for i, example in enumerate(schema_data.get("examples", [])):
        if not isinstance(example, dict):
            errors.append(f"examples[{i}]: expected object")
            continue
        ex: dict = {k: v for k, v in example.items() if k != "_note"}
        _validate_single_output(ex, f"examples[{i}]", errors)

    # ---- Load explanation templates JSON (optional) -----------------------
    if templates_path is not None:
        try:
            templates_data = _load_json(templates_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: Failed to load {templates_path}: {exc}")
            return 1

        if not isinstance(templates_data, dict):
            errors.append(f"{templates_path}: must be a JSON object")
        else:
            _check_no_protected_values(templates_data, "templates_json", errors)
            _validate_templates_json(templates_data, errors)

    # ---- Report -----------------------------------------------------------
    if errors:
        print("VALIDATION FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("validate_model_output_schema.py: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
