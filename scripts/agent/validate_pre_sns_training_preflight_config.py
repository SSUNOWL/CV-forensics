#!/usr/bin/env python3
"""Validate pre-SNS training preflight configs without loading data files."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_manifest import path_is_under, validate_explicit_local_file, walk_safety  # noqa: E402


OK_MARKER = "PRE_SNS_TRAINING_PREFLIGHT_CONFIG_OK"
APPROVAL_TEXT = "I_APPROVE_LOCAL_NON_SNS_PRE_SNS_TRAINING_PREFLIGHT"
CONFIG_KINDS = {"example_symbolic", "approved_local_pre_sns_training_preflight"}
GUARDRAILS = (
    "no_download",
    "no_network",
    "no_outputs",
    "no_checkpoints",
    "no_real_training",
    "no_sns_augmentation",
)


def _err(message: str) -> str:
    return f"- {message}"


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("config root must be a JSON object")
    return raw


def _require_true(raw: dict[str, Any], errors: list[str]) -> None:
    for flag in GUARDRAILS:
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))


def _validate_limits(raw: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    limits = raw.get("tiny_preflight_limits")
    if not isinstance(limits, dict):
        errors.append(_err("tiny_preflight_limits must be an object"))
        return {}
    caps = {"max_samples": 12, "max_image_size": 64}
    for key, cap in caps.items():
        value = limits.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(_err(f"tiny_preflight_limits.{key} must be a positive integer"))
        elif value > cap:
            errors.append(_err(f"tiny_preflight_limits.{key} must be <= {cap}"))
    if limits.get("cpu_only") is not True:
        errors.append(_err("tiny_preflight_limits.cpu_only must be true"))
    if limits.get("dry_run_optimizer_step") is not False:
        errors.append(_err("tiny_preflight_limits.dry_run_optimizer_step must be false"))
    return limits


def _validate_common(raw: dict[str, Any], errors: list[str]) -> None:
    _require_true(raw, errors)
    _validate_limits(raw, errors)
    expected = raw.get("expected_manifest_schema")
    if not isinstance(expected, dict):
        errors.append(_err("expected_manifest_schema must be an object"))
    elif expected.get("manifest_kind") != "pre_sns_unified_dataset_manifest":
        errors.append(_err("expected_manifest_schema.manifest_kind must be pre_sns_unified_dataset_manifest"))
    loss_policy = raw.get("loss_routing_policy")
    if not isinstance(loss_policy, dict):
        errors.append(_err("loss_routing_policy must be an object"))
    else:
        for key in ("class_loss", "family_loss", "localization_loss"):
            if key not in loss_policy:
                errors.append(_err(f"loss_routing_policy.{key} is required"))


def _validate_example(raw: dict[str, Any], errors: list[str]) -> None:
    if raw.get("execution_mode") != "example_only":
        errors.append(_err("execution_mode must be example_only"))
    _validate_common(raw, errors)
    for issue in walk_safety(raw):
        errors.append(_err(f"{issue.path}: {issue.message}"))


def _validate_approved(raw: dict[str, Any], errors: list[str]) -> None:
    if raw.get("approved_real_data_access") is not True:
        errors.append(_err("approved_real_data_access must be true"))
    if raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(_err("user_approval_text does not match required approval phrase"))
    if raw.get("recursive_scan") is True or raw.get("recursive_directory_scan") is True:
        errors.append(_err("recursive scan flags are rejected"))
    _validate_common(raw, errors)

    roots = raw.get("approved_local_roots")
    if not isinstance(roots, list) or not roots or not all(isinstance(root, str) for root in roots):
        errors.append(_err("approved_local_roots must be a non-empty list of absolute paths"))
        roots = []
    manifest_path = raw.get("manifest_path")
    allowed_abs = set(roots)
    if isinstance(manifest_path, str):
        allowed_abs.add(manifest_path)
    for issue in walk_safety(raw, allowed_abs_values=allowed_abs):
        errors.append(_err(f"{issue.path}: {issue.message}"))
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        errors.append(_err("manifest_path is required"))
    else:
        for issue in validate_explicit_local_file(manifest_path, roots, "manifest_path"):
            errors.append(_err(f"{issue.path}: {issue.message}"))
        if roots and not path_is_under(manifest_path, roots):
            errors.append(_err("manifest_path must be under approved_local_roots"))


def validate_config(raw: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    config_kind = raw.get("config_kind")
    if config_kind not in CONFIG_KINDS:
        errors.append(_err("config_kind must be example_symbolic or approved_local_pre_sns_training_preflight"))
        for issue in walk_safety(raw):
            errors.append(_err(f"{issue.path}: {issue.message}"))
        return errors
    if config_kind == "example_symbolic":
        _validate_example(raw, errors)
    else:
        _validate_approved(raw, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: validate_pre_sns_training_preflight_config.py <config.json>", file=sys.stderr)
        return 2
    try:
        raw = load_config(argv[0])
    except Exception as exc:
        print(f"failed to read config: {exc}", file=sys.stderr)
        return 1
    errors = validate_config(raw)
    if errors:
        print("pre-SNS training preflight config validation failed:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"{OK_MARKER}: config is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
