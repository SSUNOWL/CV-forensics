#!/usr/bin/env python3
"""Validate scaled pre-SNS training plan configs without training."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OK_MARKER = "PRE_SNS_SCALED_TRAINING_PLAN_OK"
APPROVAL_TEXT = "I_APPROVE_PRE_SNS_BASELINE_TRAINING"
CONFIG_KINDS = {"example_symbolic", "approved_pre_sns_baseline_training"}
REMOTE_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
SECRET_KEY_RE = re.compile(r"(^|_)(api[_-]?key|secret|token|password|credential|private[_-]?key)($|_)", re.I)
SECRET_VALUE_RE = re.compile(r"(api[_-]?key|secret|password|token=|bearer |private[_-]?key)", re.I)
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}
REQUIRED_FIELDS = (
    "schema_version",
    "config_kind",
    "execution_mode",
    "scale",
    "required_approval_text",
    "user_approval_text",
    "unified_manifest_path",
    "approved_run_root",
    "approved_checkpoint_root",
    "device",
    "max_samples",
    "epochs",
    "batch_size",
    "max_image_size",
    "no_download",
    "no_network",
    "no_sns_augmentation",
    "result_scope",
)
BOUNDS = {
    "max_samples": 50000,
    "epochs": 10,
    "batch_size": 64,
    "max_image_size": 512,
}


def _err(message: str) -> str:
    return f"- {message}"


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("config root must be a JSON object")
    return raw


def _real(path: str | Path) -> str:
    return os.path.realpath(os.fspath(path))


def _is_under(path: str | Path, root: str | Path) -> bool:
    try:
        return os.path.commonpath([_real(path), _real(root)]) == _real(root)
    except ValueError:
        return False


def _is_under_repo(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT)


def _is_repo_output_or_checkpoint(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT / "outputs") or _is_under(path, REPO_ROOT / "checkpoints")


def _contains_protected_part(value: str, allowed_parts: set[str] | None = None) -> bool:
    allowed_parts = allowed_parts or set()
    return any(
        part in PROTECTED_PARTS and part not in allowed_parts
        for part in value.replace("\\", "/").split("/")
        if part
    )


def _has_path_traversal(value: str) -> bool:
    return any(part == ".." for part in value.replace("\\", "/").split("/"))


def _walk_safety(value: Any, path: str = "", allowed_abs_values: set[str] | None = None) -> list[str]:
    allowed_abs_values = allowed_abs_values or set()
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if SECRET_KEY_RE.search(key_text):
                errors.append(_err(f"{child_path}: secret-like key rejected"))
            errors.extend(_walk_safety(child, child_path, allowed_abs_values))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_walk_safety(child, f"{path}[{index}]", allowed_abs_values))
    elif isinstance(value, str):
        text = value.strip()
        if REMOTE_RE.search(text):
            errors.append(_err(f"{path}: URL or remote scheme rejected"))
        if WINDOWS_DRIVE_RE.search(text):
            errors.append(_err(f"{path}: Windows drive path rejected"))
        if text in allowed_abs_values:
            if _has_path_traversal(text):
                errors.append(_err(f"{path}: path traversal rejected"))
            if SECRET_VALUE_RE.search(text):
                errors.append(_err(f"{path}: secret-like value rejected"))
            return errors
        if text.startswith("/"):
            errors.append(_err(f"{path}: absolute path rejected"))
        if _has_path_traversal(text):
            errors.append(_err(f"{path}: path traversal rejected"))
        if _contains_protected_part(text):
            errors.append(_err(f"{path}: protected path segment rejected"))
        if SECRET_VALUE_RE.search(text):
            errors.append(_err(f"{path}: secret-like value rejected"))
    return errors


def _validate_bounded_int(raw: dict[str, Any], field: str, errors: list[str]) -> None:
    value = raw.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(_err(f"{field} must be a positive integer"))
    elif value > BOUNDS[field]:
        errors.append(_err(f"{field} must be <= {BOUNDS[field]}"))


def _validate_local_root(raw: dict[str, Any], field: str, errors: list[str]) -> None:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(_err(f"{field} must be a non-empty absolute local path"))
        return
    text = value.strip()
    if REMOTE_RE.search(text):
        errors.append(_err(f"{field} must not be a URL or remote scheme"))
    if WINDOWS_DRIVE_RE.search(text):
        errors.append(_err(f"{field} must not be a Windows drive path"))
    if not text.startswith("/"):
        errors.append(_err(f"{field} must be absolute in approved local mode"))
    if _has_path_traversal(text):
        errors.append(_err(f"{field} must not contain path traversal"))
    if _contains_protected_part(text):
        errors.append(_err(f"{field} must not contain protected path segments"))
    if text.startswith("/") and _is_under_repo(text):
        errors.append(_err(f"{field} must be outside the repository"))
    if text.startswith("/") and _is_repo_output_or_checkpoint(text):
        errors.append(_err(f"{field} must not be inside repository outputs/checkpoints"))


def validate_scaled_training_plan(raw: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in raw:
            errors.append(_err(f"{field} is required"))

    kind = raw.get("config_kind")
    if kind not in CONFIG_KINDS:
        errors.append(_err("config_kind must be example_symbolic or approved_pre_sns_baseline_training"))
        errors.extend(_walk_safety(raw))
        return errors

    for field in BOUNDS:
        _validate_bounded_int(raw, field, errors)
    if raw.get("device") not in {"cpu", "cuda"}:
        errors.append(_err("device must be cpu or cuda"))
    for flag in ("no_download", "no_network", "no_sns_augmentation"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if raw.get("recursive_scan") is True or raw.get("recursive_directory_scan") is True:
        errors.append(_err("recursive scan flags are rejected"))
    if raw.get("marker") != OK_MARKER and OK_MARKER not in raw.get("validation_notes", []):
        errors.append(_err(f"marker must include {OK_MARKER}"))

    if kind == "example_symbolic":
        if raw.get("execution_mode") != "example_only":
            errors.append(_err("execution_mode must be example_only"))
        if raw.get("user_approval_text") not in {"", APPROVAL_TEXT}:
            errors.append(_err("example user_approval_text must be empty or documented approval phrase"))
        errors.extend(_walk_safety(raw))
        return errors

    if raw.get("execution_mode") != "approved_local_pre_sns_scaled_training":
        errors.append(_err("execution_mode must be approved_local_pre_sns_scaled_training"))
    if raw.get("approved_training_run") is not True:
        errors.append(_err("approved_training_run must be true"))
    if raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(_err("user_approval_text does not match required approval phrase"))
    if raw.get("no_write_dry_run") is not False:
        errors.append(_err("no_write_dry_run must be false for approved scaled plan"))

    allowed_abs = set()
    for field in ("unified_manifest_path", "approved_run_root", "approved_checkpoint_root"):
        value = raw.get(field)
        if isinstance(value, str):
            allowed_abs.add(value)
    roots = raw.get("approved_local_roots")
    if isinstance(roots, list):
        allowed_abs.update(root for root in roots if isinstance(root, str))
    errors.extend(_walk_safety(raw, allowed_abs_values=allowed_abs))
    _validate_local_root(raw, "approved_run_root", errors)
    _validate_local_root(raw, "approved_checkpoint_root", errors)
    if isinstance(raw.get("approved_run_root"), str) and isinstance(raw.get("approved_checkpoint_root"), str):
        if _real(raw["approved_run_root"]) == _real(raw["approved_checkpoint_root"]):
            errors.append(_err("approved_run_root and approved_checkpoint_root must be distinct"))
    return errors


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: validate_pre_sns_scaled_training_plan.py <config.json>", file=sys.stderr)
        return 2
    try:
        raw = load_config(argv[0])
    except Exception as exc:
        print(f"failed to read config: {exc}", file=sys.stderr)
        return 1
    errors = validate_scaled_training_plan(raw)
    if errors:
        print("pre-SNS scaled training plan validation failed:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"{OK_MARKER}: config is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
