#!/usr/bin/env python3
"""Validate pre-SNS integrated smoke configs without reading images or masks."""

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

try:
    from cv_forensics.local_data_gate import check_local_data_readiness_config_safety
except Exception:
    check_local_data_readiness_config_safety = None


OK_MARKER = "PRE_SNS_INTEGRATED_SMOKE_CONFIG_OK"
APPROVAL_TEXT = "I_APPROVE_LOCAL_NON_SNS_PRE_SNS_INTEGRATED_SMOKE"
CONFIG_KINDS = {"example_symbolic", "approved_local_pre_sns_integrated_smoke"}
CLASS_LABELS = {"real", "full_synthetic", "tampered"}
FAMILY_LABELS = {"LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A"}
REQUIRED_TRUE_FLAGS = (
    "no_download",
    "no_network",
    "no_outputs",
    "no_checkpoints",
    "no_sns_augmentation",
    "no_sns_perturbation_eval",
)
EXAMPLE_TRUE_FLAGS = REQUIRED_TRUE_FLAGS + ("dry_run", "requires_user_approval_for_real_data")
APPROVED_TRUE_FLAGS = REQUIRED_TRUE_FLAGS + ("approved_real_data_access",)
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}
SECRET_KEY_RE = re.compile(r"(^|_)(api[_-]?key|secret|token|password|credential|private[_-]?key)($|_)", re.I)
REMOTE_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _error(message: str) -> str:
    return f"- {message}"


def _contains_protected_part(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return any(part in PROTECTED_PARTS for part in normalized.split("/") if part)


def _has_path_traversal(value: str) -> bool:
    return any(part == ".." for part in value.replace("\\", "/").split("/"))


def _looks_secret_value(value: str) -> bool:
    lowered = value.lower()
    return (
        "secret" in lowered
        or "password" in lowered
        or "api_key" in lowered
        or "private_key" in lowered
        or "token=" in lowered
        or "bearer " in lowered
    )


def _walk_safety(value: Any, path: str, errors: list[str], allowed_abs_values: set[str] | None = None) -> None:
    allowed_abs_values = allowed_abs_values or set()
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if SECRET_KEY_RE.search(key_text):
                errors.append(_error(f"secret-like key rejected at {child_path}"))
            if _contains_protected_part(key_text):
                errors.append(_error(f"protected path segment rejected in key at {child_path}"))
            _walk_safety(child, child_path, errors, allowed_abs_values)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_safety(child, f"{path}[{index}]", errors, allowed_abs_values)
    elif isinstance(value, str):
        text = value.strip()
        if REMOTE_RE.search(text):
            errors.append(_error(f"URL or remote scheme rejected at {path}"))
        if WINDOWS_DRIVE_RE.search(text):
            errors.append(_error(f"Windows drive path rejected at {path}"))
        if text.startswith("/") and text not in allowed_abs_values:
            errors.append(_error(f"absolute path rejected at {path}"))
        if _contains_protected_part(text):
            errors.append(_error(f"protected path segment rejected at {path}"))
        if _has_path_traversal(text):
            errors.append(_error(f"path traversal rejected at {path}"))
        if _looks_secret_value(text):
            errors.append(_error(f"secret-like value rejected at {path}"))


def _require_true(raw: dict[str, Any], flags: tuple[str, ...], errors: list[str]) -> None:
    for flag in flags:
        if raw.get(flag) is not True:
            errors.append(_error(f"{flag} must be true"))


def _require_dict(raw: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        errors.append(_error(f"{key} must be an object"))
        return {}
    return value


def _labels_from_policy(policy: dict[str, Any]) -> set[str]:
    labels = policy.get("labels")
    if not isinstance(labels, list):
        return set()
    return {label for label in labels if isinstance(label, str)}


def _validate_policies(raw: dict[str, Any], errors: list[str]) -> None:
    class_policy = _require_dict(raw, "class_policy", errors)
    if _labels_from_policy(class_policy) != CLASS_LABELS:
        errors.append(_error("class_policy.labels must be exactly real, full_synthetic, and tampered"))
    family_policy = _require_dict(raw, "family_policy", errors)
    if _labels_from_policy(family_policy) != FAMILY_LABELS:
        errors.append(_error("family_policy.labels must match LatDiff, PixDiff, GAN, Other, Real-or-N/A"))
    if family_policy.get("loss_applies_only_when_family_label_present") is not True:
        errors.append(_error("family loss routing policy must be conditional on available family labels"))
    localization_policy = _require_dict(raw, "localization_policy", errors)
    if localization_policy.get("loss_applies_only_to_tampered_samples_with_masks") is not True:
        errors.append(_error("localization loss must apply only to tampered samples with masks"))
    loss_policy = _require_dict(raw, "loss_routing_policy", errors)
    for key in ("class_loss", "family_loss", "localization_loss", "total_loss"):
        if key not in loss_policy:
            errors.append(_error(f"loss_routing_policy.{key} is required"))


def _validate_tiny_limits(raw: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    limits = _require_dict(raw, "tiny_limits", errors)
    caps = {"max_samples": 12, "max_steps": 5, "max_epochs": 2, "max_image_size": 64}
    for key, cap in caps.items():
        value = limits.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(_error(f"tiny_limits.{key} must be a positive integer"))
        elif value > cap:
            errors.append(_error(f"tiny_limits.{key} must be <= {cap}"))
    if limits.get("cpu_only") is not True:
        errors.append(_error("tiny_limits.cpu_only must be true"))
    return limits


def _run_symbolic_safety(raw: dict[str, Any], errors: list[str]) -> None:
    if check_local_data_readiness_config_safety is not None:
        try:
            check_local_data_readiness_config_safety(raw)
        except ValueError as exc:
            errors.append(_error(f"local_data_gate safety: {exc}"))
    else:
        _walk_safety(raw, "", errors)


def _validate_example(raw: dict[str, Any], errors: list[str]) -> None:
    if raw.get("execution_mode") != "example_only":
        errors.append(_error("execution_mode must be example_only for example_symbolic configs"))
    _require_true(raw, EXAMPLE_TRUE_FLAGS, errors)
    _validate_policies(raw, errors)
    _validate_tiny_limits(raw, errors)
    _run_symbolic_safety(raw, errors)


def _path_is_under(path: str, roots: list[str]) -> bool:
    real_path = os.path.realpath(path)
    for root in roots:
        real_root = os.path.realpath(root)
        try:
            if os.path.commonpath([real_path, real_root]) == real_root:
                return True
        except ValueError:
            continue
    return False


def _validate_file_path(path_value: str, field_path: str, roots: list[str], errors: list[str]) -> None:
    if REMOTE_RE.search(path_value) or WINDOWS_DRIVE_RE.search(path_value):
        errors.append(_error(f"{field_path} must be a local path"))
    if not path_value.startswith("/"):
        errors.append(_error(f"{field_path} must be absolute in approved local mode"))
    if path_value.endswith("/") or os.path.basename(path_value) in {"", ".", ".."}:
        errors.append(_error(f"{field_path} must be an explicit file path"))
    if _has_path_traversal(path_value):
        errors.append(_error(f"{field_path} must not contain path traversal"))
    if roots and not _path_is_under(path_value, roots):
        errors.append(_error(f"{field_path} is outside approved_local_roots"))


def _validate_approved_paths(raw: dict[str, Any], errors: list[str]) -> list[dict[str, Any]]:
    roots = raw.get("approved_local_roots")
    if not isinstance(roots, list) or not roots or not all(isinstance(root, str) for root in roots):
        errors.append(_error("approved_local_roots must be a non-empty list of local absolute paths"))
        roots = []
    for root in roots:
        if not root.startswith("/"):
            errors.append(_error("approved_local_roots entries must be absolute paths"))
        if REMOTE_RE.search(root) or WINDOWS_DRIVE_RE.search(root) or _contains_protected_part(root) or _has_path_traversal(root):
            errors.append(_error("approved_local_roots entries must not be remote, Windows, protected, or traversal paths"))

    manifest = raw.get("sample_manifest")
    if not isinstance(manifest, list) or not manifest:
        errors.append(_error("sample_manifest must be a non-empty list"))
        manifest = []

    allowed_abs_values = set(roots)
    for sample in manifest:
        if not isinstance(sample, dict):
            continue
        for field in ("image_path", "mask_path"):
            value = sample.get(field)
            if isinstance(value, str):
                allowed_abs_values.add(value)
    _walk_safety(raw, "", errors, allowed_abs_values=allowed_abs_values)

    if raw.get("recursive_scan") is True or raw.get("recursive_directory_scan") is True:
        errors.append(_error("recursive scan flags are rejected"))

    for index, sample in enumerate(manifest):
        if not isinstance(sample, dict):
            errors.append(_error(f"sample_manifest[{index}] must be an object"))
            continue
        for field in ("sample_id", "image_path", "class_label"):
            if not isinstance(sample.get(field), str) or not sample.get(field).strip():
                errors.append(_error(f"sample_manifest[{index}].{field} is required"))
        image_path = sample.get("image_path")
        if isinstance(image_path, str):
            _validate_file_path(image_path, f"sample_manifest[{index}].image_path", roots, errors)
        mask_path = sample.get("mask_path")
        if isinstance(mask_path, str):
            _validate_file_path(mask_path, f"sample_manifest[{index}].mask_path", roots, errors)
    return manifest


def _validate_approved_manifest(manifest: list[dict[str, Any]], limits: dict[str, Any], errors: list[str]) -> None:
    if len(manifest) > limits.get("max_samples", 0):
        errors.append(_error("sample_manifest length exceeds tiny_limits.max_samples"))
    seen_classes: set[str] = set()
    has_family = False
    has_tampered_mask = False
    for index, sample in enumerate(manifest):
        if not isinstance(sample, dict):
            continue
        class_label = sample.get("class_label")
        if class_label not in CLASS_LABELS:
            errors.append(_error(f"sample_manifest[{index}].class_label must be real, full_synthetic, or tampered"))
        else:
            seen_classes.add(class_label)
        family_label = sample.get("family_label")
        if family_label is not None:
            if family_label not in FAMILY_LABELS:
                errors.append(_error(f"sample_manifest[{index}].family_label is not an allowed family label"))
            else:
                has_family = True
        has_mask = isinstance(sample.get("mask_path"), str) and bool(sample["mask_path"].strip())
        if class_label == "tampered":
            if not has_mask:
                errors.append(_error(f"sample_manifest[{index}] tampered samples must include mask_path"))
            else:
                has_tampered_mask = True
        elif has_mask:
            errors.append(_error(f"sample_manifest[{index}] non-tampered samples must not include mask_path"))
    for label in sorted(CLASS_LABELS - seen_classes):
        errors.append(_error(f"approved local smoke missing required class {label}"))
    if not has_family:
        errors.append(_error("approved local smoke must include at least one family_label"))
    if not has_tampered_mask:
        errors.append(_error("approved local smoke must include at least one tampered sample with mask_path"))


def _validate_requested_limits(raw: dict[str, Any], limits: dict[str, Any], errors: list[str]) -> None:
    requested = raw.get("requested_run_limits", {})
    if requested is None:
        requested = {}
    if not isinstance(requested, dict):
        errors.append(_error("requested_run_limits must be an object when present"))
        return
    for key in ("max_samples", "max_steps", "max_epochs"):
        if key in requested:
            value = requested[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                errors.append(_error(f"requested_run_limits.{key} must be a positive integer"))
            elif value > limits.get(key, 0):
                errors.append(_error(f"requested_run_limits.{key} exceeds tiny_limits.{key}"))


def _validate_approved(raw: dict[str, Any], errors: list[str]) -> None:
    if raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(_error("user_approval_text does not match required approval phrase"))
    _require_true(raw, APPROVED_TRUE_FLAGS, errors)
    _validate_policies(raw, errors)
    limits = _validate_tiny_limits(raw, errors)
    _validate_requested_limits(raw, limits, errors)
    manifest = _validate_approved_paths(raw, errors)
    _validate_approved_manifest(manifest, limits, errors)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("config root must be a JSON object")
    return raw


def validate_config(raw: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    config_kind = raw.get("config_kind")
    if config_kind not in CONFIG_KINDS:
        errors.append(_error("config_kind must be example_symbolic or approved_local_pre_sns_integrated_smoke"))
        _walk_safety(raw, "", errors)
        return errors
    if config_kind == "example_symbolic":
        _validate_example(raw, errors)
    else:
        _validate_approved(raw, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: validate_pre_sns_integrated_smoke_config.py <config.json>", file=sys.stderr)
        return 2
    try:
        raw = load_config(argv[0])
    except Exception as exc:
        print(f"failed to read config: {exc}", file=sys.stderr)
        return 1
    errors = validate_config(raw)
    if errors:
        print("pre-SNS integrated smoke config validation failed:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"{OK_MARKER}: config is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
