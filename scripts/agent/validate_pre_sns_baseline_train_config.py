#!/usr/bin/env python3
"""Validate guarded pre-SNS baseline training configs without loading data."""

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

from cv_forensics.pre_sns_manifest import (  # noqa: E402
    path_is_under,
    validate_explicit_local_file,
    walk_safety,
)


OK_MARKER = "PRE_SNS_BASELINE_TRAIN_CONFIG_OK"
APPROVAL_TEXT = "I_APPROVE_PRE_SNS_BASELINE_TRAINING"
CONFIG_KINDS = {"example_symbolic", "approved_pre_sns_baseline_training"}
REMOTE_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}
REQUIRED_FIELDS = (
    "unified_manifest_path",
    "approved_run_root",
    "approved_checkpoint_root",
    "device",
    "batch_size",
    "epochs",
    "max_samples",
    "seed",
    "class_loss_weight",
    "family_loss_weight",
    "localization_loss_weight",
)
OPTIONAL_MANIFEST_PATH_FIELDS = (
    "dataset_manifest_path",
    "manifest_path",
    "train_manifest_path",
    "val_manifest_path",
)


def _err(message: str) -> str:
    return f"- {message}"


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("config root must be a JSON object")
    return raw


def _contains_protected_part(value: str) -> bool:
    return any(part in PROTECTED_PARTS for part in value.replace("\\", "/").split("/") if part)


def _has_path_traversal(value: str) -> bool:
    return any(part == ".." for part in value.replace("\\", "/").split("/"))


def _repo_realpath() -> str:
    return os.path.realpath(str(REPO_ROOT))


def _is_under_repo(path_value: str) -> bool:
    real_path = os.path.realpath(path_value)
    try:
        return os.path.commonpath([real_path, _repo_realpath()]) == _repo_realpath()
    except ValueError:
        return False


def _validate_local_directory(path_value: Any, field: str, errors: list[str]) -> None:
    if not isinstance(path_value, str) or not path_value.strip():
        errors.append(_err(f"{field} must be a non-empty absolute local path"))
        return
    text = path_value.strip()
    if REMOTE_RE.search(text):
        errors.append(_err(f"{field} must not be a URL or remote scheme"))
    if WINDOWS_DRIVE_RE.search(text):
        errors.append(_err(f"{field} must not be a Windows drive path"))
    if not text.startswith("/"):
        errors.append(_err(f"{field} must be an absolute local path"))
    if _has_path_traversal(text):
        errors.append(_err(f"{field} must not contain path traversal"))
    if _contains_protected_part(text):
        errors.append(_err(f"{field} must not contain protected path segments"))
    basename = os.path.basename(os.path.normpath(text))
    if basename in {"", ".", "..", "outputs", "checkpoints", "data", "datasets", "secrets", ".env"}:
        errors.append(_err(f"{field} must not be a protected repository directory name"))
    if text.startswith("/") and _is_under_repo(text):
        errors.append(_err(f"{field} must be outside the repository"))


def _validate_optional_local_roots(raw: dict[str, Any], field: str, errors: list[str]) -> list[str]:
    value = raw.get(field)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(root, str) for root in value):
        errors.append(_err(f"{field} must be a list of absolute local paths when present"))
        return []
    for index, root in enumerate(value):
        _validate_local_directory(root, f"{field}[{index}]", errors)
    return list(value)


def _validate_numbers(raw: dict[str, Any], errors: list[str]) -> None:
    int_caps = {"batch_size": 32, "epochs": 100, "max_samples": 1000000}
    for key, cap in int_caps.items():
        value = raw.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(_err(f"{key} must be a positive integer"))
        elif value > cap:
            errors.append(_err(f"{key} is too large for the guarded entrypoint"))
    seed = raw.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        errors.append(_err("seed must be an integer"))
    for key in ("class_loss_weight", "family_loss_weight", "localization_loss_weight"):
        value = raw.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            errors.append(_err(f"{key} must be a non-negative number"))


def _validate_common_shape(raw: dict[str, Any], errors: list[str]) -> None:
    for field in REQUIRED_FIELDS:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    if raw.get("no_download") is not True:
        errors.append(_err("no_download must be true"))
    if raw.get("no_network") is not True:
        errors.append(_err("no_network must be true"))
    if raw.get("no_sns_augmentation") is not True:
        errors.append(_err("no_sns_augmentation must be true"))
    if raw.get("device") not in {"cpu", "cuda"}:
        errors.append(_err("device must be cpu or cuda"))
    _validate_numbers(raw, errors)
    if raw.get("recursive_scan") is True or raw.get("recursive_directory_scan") is True:
        errors.append(_err("recursive scan flags are rejected"))


def _validate_example(raw: dict[str, Any], errors: list[str]) -> None:
    if raw.get("execution_mode") != "example_only":
        errors.append(_err("execution_mode must be example_only"))
    if raw.get("approved_training_run") is not False:
        errors.append(_err("tracked example must set approved_training_run false"))
    if raw.get("required_approval_text") != APPROVAL_TEXT:
        errors.append(_err("required_approval_text must document the exact approval phrase"))
    _validate_common_shape(raw, errors)
    for issue in walk_safety(raw):
        errors.append(_err(f"{issue.path}: {issue.message}"))


def _validate_approved(raw: dict[str, Any], errors: list[str]) -> None:
    if raw.get("approved_training_run") is not True:
        errors.append(_err("approved_training_run must be true"))
    if raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(_err("user_approval_text does not match required approval phrase"))
    _validate_common_shape(raw, errors)

    roots = raw.get("approved_local_roots")
    if not isinstance(roots, list) or not roots or not all(isinstance(root, str) for root in roots):
        errors.append(_err("approved_local_roots must be a non-empty list of absolute paths"))
        roots = []

    manifest_path = raw.get("unified_manifest_path")
    run_root = raw.get("approved_run_root")
    checkpoint_root = raw.get("approved_checkpoint_root")
    allowed_abs = set(roots)
    output_roots = _validate_optional_local_roots(raw, "approved_local_output_roots", errors)
    optional_alias_roots = []
    for alias in ("run_root", "checkpoint_root"):
        value = raw.get(alias)
        if isinstance(value, str):
            _validate_local_directory(value, alias, errors)
            optional_alias_roots.append(value)
    optional_manifest_paths = [
        value
        for field in OPTIONAL_MANIFEST_PATH_FIELDS
        if isinstance((value := raw.get(field)), str)
    ]
    for value in (manifest_path, *optional_manifest_paths, run_root, checkpoint_root, *output_roots, *optional_alias_roots):
        if isinstance(value, str):
            allowed_abs.add(value)
    for issue in walk_safety(raw, allowed_abs_values=allowed_abs):
        errors.append(_err(f"{issue.path}: {issue.message}"))

    if isinstance(manifest_path, str):
        for issue in validate_explicit_local_file(manifest_path, roots, "unified_manifest_path"):
            errors.append(_err(f"{issue.path}: {issue.message}"))
        if roots and not path_is_under(manifest_path, roots):
            errors.append(_err("unified_manifest_path must be under approved_local_roots"))
    else:
        errors.append(_err("unified_manifest_path must be a non-empty path string"))

    for field in OPTIONAL_MANIFEST_PATH_FIELDS:
        value = raw.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            for issue in validate_explicit_local_file(value, roots, field):
                errors.append(_err(f"{issue.path}: {issue.message}"))
            if roots and not path_is_under(value, roots):
                errors.append(_err(f"{field} must be under approved_local_roots"))
        else:
            errors.append(_err(f"{field} must be a path string when present"))

    _validate_local_directory(run_root, "approved_run_root", errors)
    _validate_local_directory(checkpoint_root, "approved_checkpoint_root", errors)

    if isinstance(run_root, str) and isinstance(checkpoint_root, str):
        if os.path.realpath(run_root) == os.path.realpath(checkpoint_root):
            errors.append(_err("approved_run_root and approved_checkpoint_root must be distinct"))


def validate_config(raw: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    config_kind = raw.get("config_kind")
    if config_kind not in CONFIG_KINDS:
        errors.append(_err("config_kind must be example_symbolic or approved_pre_sns_baseline_training"))
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
        print("usage: validate_pre_sns_baseline_train_config.py <config.json>", file=sys.stderr)
        return 2
    try:
        raw = load_config(argv[0])
    except Exception as exc:
        print(f"failed to read config: {exc}", file=sys.stderr)
        return 1
    errors = validate_config(raw)
    if errors:
        print("pre-SNS baseline training config validation failed:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"{OK_MARKER}: config is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
