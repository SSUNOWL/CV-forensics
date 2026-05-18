#!/usr/bin/env python3
"""Validate CF-Small tiny training smoke configs.

The validator supports a tracked symbolic example and an explicitly approved
local tiny-smoke config. It never reads images, writes files, downloads data, or
inspects dataset directories recursively.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "src" / "cv_forensics").is_dir()),
    None,
)
if _REPO_ROOT is not None:
    _SRC_ROOT = str(_REPO_ROOT / "src")
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

try:
    from cv_forensics.local_data_gate import check_local_data_readiness_config_safety
except ImportError as exc:  # pragma: no cover
    check_local_data_readiness_config_safety = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


APPROVAL_TEXT = "I_APPROVE_LOCAL_NON_SNS_TINY_TRAINING_SMOKE"
REQUIRED_LABELS = {"real", "synthetic"}
CONFIG_KINDS = {"example_symbolic", "approved_local_smoke"}
REMOTE_SCHEMES = ("http://", "https://", "ftp://", "s3://", "gs://", "hf://")
PROTECTED_PARTS = {"secrets", "data", "datasets", "outputs", "checkpoints"}
SECRET_KEY_TOKENS = {"token", "api", "key", "password", "secret", "credential", "auth", "bearer"}
GUARDRAILS = (
    "no_download",
    "no_network",
    "no_outputs",
    "no_checkpoints",
    "no_sns_augmentation",
    "no_sns_perturbation_eval",
)
EXAMPLE_GUARDRAILS = ("dry_run",) + GUARDRAILS + ("requires_user_approval_for_real_data",)
APPROVED_GUARDRAILS = GUARDRAILS
REQUIRED_EXAMPLE_KEYS = (
    "schema_version",
    "config_kind",
    "smoke_name",
    "dataset_name",
    "stage",
    "execution_mode",
    "dry_run",
    "no_download",
    "no_network",
    "no_outputs",
    "no_checkpoints",
    "no_sns_augmentation",
    "no_sns_perturbation_eval",
    "requires_user_approval_for_real_data",
    "local_config_template_path",
    "approved_real_mode_policy",
    "tiny_limits",
    "class_policy",
    "sample_manifest_policy",
    "training_smoke_policy",
    "metric_policy",
    "output_policy",
    "checkpoint_policy",
    "next_stage_policy",
    "validation_notes",
)


def _tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _secret_key(key: str) -> bool:
    toks = set(_tokens(key))
    if key.lower() in {"token", "api_key", "password", "secret", "credential", "auth", "bearer"}:
        return True
    if "api" in toks and "key" in toks:
        return True
    return bool(toks & {"token", "password", "secret", "credential", "auth", "bearer"})


def _secret_value(value: str) -> bool:
    toks = set(_tokens(value))
    if "api" in toks and "key" in toks:
        return True
    return bool(toks & {"token", "password", "secret", "credential", "auth", "bearer"})


def _is_remote(value: str) -> bool:
    return value.strip().lower().startswith(REMOTE_SCHEMES)


def _path_parts(value: str) -> List[str]:
    return [part for part in re.split(r"[/\\]+", value.strip()) if part]


def _has_protected_path(value: str) -> bool:
    stripped = value.strip()
    if stripped == ".env" or stripped.startswith(".env."):
        return True
    return any(part in PROTECTED_PARTS for part in _path_parts(stripped))


def _is_windows_drive(value: str) -> bool:
    stripped = value.strip()
    return len(stripped) >= 3 and stripped[0].isalpha() and stripped[1] == ":" and stripped[2] in ("\\", "/")


def _contains_text(node: Any, needles: Iterable[str]) -> bool:
    lowered = [n.lower() for n in needles]
    if isinstance(node, dict):
        return any(_contains_text(k, lowered) or _contains_text(v, lowered) for k, v in node.items())
    if isinstance(node, list):
        return any(_contains_text(v, lowered) for v in node)
    if isinstance(node, str):
        value = node.lower()
        return any(n in value for n in lowered)
    return False


def _collect_local_path_fields(raw: Dict[str, Any]) -> Tuple[set, set]:
    roots = set()
    sample_paths = set()
    for root in raw.get("approved_local_roots", []):
        if isinstance(root, str):
            roots.add(root)
    for entry in raw.get("sample_manifest", []):
        if isinstance(entry, dict) and isinstance(entry.get("image_path"), str):
            sample_paths.add(entry["image_path"])
    return roots, sample_paths


def _check_approved_safety(node: Any, path: str, allowed_abs_values: set, errors: List[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}"
            if isinstance(key, str) and _secret_key(key):
                errors.append(f"{child}: secret-like key is not allowed.")
            _check_approved_safety(value, child, allowed_abs_values, errors)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _check_approved_safety(item, f"{path}[{index}]", allowed_abs_values, errors)
    elif isinstance(node, str):
        stripped = node.strip()
        if not stripped:
            return
        if _is_remote(stripped):
            errors.append(f"{path}: URLs and remote schemes are not allowed.")
        if _is_windows_drive(stripped):
            errors.append(f"{path}: Windows drive paths are not allowed.")
        if _has_protected_path(stripped):
            errors.append(f"{path}: protected repository paths are not allowed.")
        if os.path.isabs(stripped) and stripped not in allowed_abs_values:
            errors.append(f"{path}: absolute paths are allowed only for approved roots or explicit sample files.")
        if stripped not in allowed_abs_values and _secret_value(stripped):
            errors.append(f"{path}: secret-like values are not allowed.")


def _as_dict(raw: Dict[str, Any], key: str, errors: List[str]) -> Dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key!r} must be an object.")
        return {}
    return value


def _as_list(raw: Dict[str, Any], key: str, errors: List[str]) -> List[Any]:
    value = raw.get(key)
    if not isinstance(value, list):
        errors.append(f"{key!r} must be a list.")
        return []
    return value


def _validate_class_policy(raw: Dict[str, Any], errors: List[str]) -> None:
    class_policy = _as_dict(raw, "class_policy", errors)
    if not class_policy:
        return
    labels = class_policy.get("labels")
    if not isinstance(labels, list) or set(labels) != REQUIRED_LABELS or len(labels) != 2:
        errors.append("class_policy.labels must contain exactly real and synthetic.")


def _validate_tiny_limits(raw: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
    limits = _as_dict(raw, "tiny_limits", errors)
    for key in ("max_samples", "max_steps", "max_epochs", "max_image_size"):
        value = limits.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(f"tiny_limits.{key} must be a positive integer.")
    if limits.get("cpu_only") is not True:
        errors.append("tiny_limits.cpu_only must be true.")
    return limits


def _validate_example(raw: Dict[str, Any], errors: List[str]) -> None:
    if check_local_data_readiness_config_safety is None:
        errors.append(f"Cannot import local data gate safety checker: {_IMPORT_ERROR}")
    else:
        try:
            check_local_data_readiness_config_safety(raw, context="cf_small_tiny_train_smoke_example")
        except ValueError as exc:
            errors.append(str(exc))
    for key in REQUIRED_EXAMPLE_KEYS:
        if key not in raw:
            errors.append(f"Missing required key: {key!r}.")
    if raw.get("execution_mode") != "example_only":
        errors.append("execution_mode must be example_only for example_symbolic.")
    for flag in EXAMPLE_GUARDRAILS:
        if raw.get(flag) is not True:
            errors.append(f"{flag} must be true for example_symbolic.")
    _validate_class_policy(raw, errors)
    _validate_tiny_limits(raw, errors)
    manifest_policy = _as_dict(raw, "sample_manifest_policy", errors)
    if manifest_policy:
        if manifest_policy.get("explicit_listed_sample_paths_required") is not True:
            errors.append("sample_manifest_policy must require explicit listed sample paths.")
        if manifest_policy.get("recursive_directory_scan_allowed") is not False:
            errors.append("sample_manifest_policy must reject recursive directory scanning.")
    output_policy = _as_dict(raw, "output_policy", errors)
    if output_policy:
        if output_policy.get("stdout_summary_only_by_default") is not True:
            errors.append("output_policy must default to stdout summary only.")
        if output_policy.get("write_outputs") is not False:
            errors.append("output_policy.write_outputs must be false.")
    checkpoint_policy = _as_dict(raw, "checkpoint_policy", errors)
    if checkpoint_policy and checkpoint_policy.get("write_checkpoints") is not False:
        errors.append("checkpoint_policy.write_checkpoints must be false.")
    next_stage = _as_dict(raw, "next_stage_policy", errors)
    if next_stage:
        if next_stage.get("sid_set_tiny_multihead_smoke_later") is not True:
            errors.append("next_stage_policy must defer SID-Set tiny multi-head smoke.")
        if next_stage.get("full_baseline_training_requires_later_explicit_approval") is not True:
            errors.append("next_stage_policy must gate full baseline training.")


def _validate_approved(raw: Dict[str, Any], errors: List[str]) -> None:
    roots, sample_paths = _collect_local_path_fields(raw)
    _check_approved_safety(raw, "cf_small_tiny_train_smoke", roots | sample_paths, errors)
    if raw.get("approved_real_data_access") is not True:
        errors.append("approved_real_data_access must be true for approved_local_smoke.")
    if raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(f"user_approval_text must exactly equal {APPROVAL_TEXT}.")
    for flag in APPROVED_GUARDRAILS:
        if raw.get(flag) is not True:
            errors.append(f"{flag} must be true for approved_local_smoke.")
    if raw.get("dataset_name") != "Community Forensics-Small":
        errors.append("dataset_name must be Community Forensics-Small.")
    _validate_class_policy(raw, errors)
    limits = _validate_tiny_limits(raw, errors)
    if not isinstance(raw.get("approved_local_roots"), list) or not raw.get("approved_local_roots"):
        errors.append("approved_local_roots must be a non-empty list.")
    for root in raw.get("approved_local_roots", []):
        if not isinstance(root, str) or not os.path.isabs(root):
            errors.append("approved_local_roots entries must be absolute local paths.")
        elif _is_remote(root) or _is_windows_drive(root) or _has_protected_path(root):
            errors.append("approved_local_roots entries must not be remote, Windows, or protected paths.")
    if raw.get("recursive_scan") is True or raw.get("recursive_directory_scan_allowed") is True:
        errors.append("recursive scan flags must be false or absent.")
    manifest = _as_list(raw, "sample_manifest", errors)
    labels_seen = set()
    for index, entry in enumerate(manifest):
        if not isinstance(entry, dict):
            errors.append(f"sample_manifest[{index}] must be an object.")
            continue
        for key in ("sample_id", "image_path", "label"):
            if key not in entry:
                errors.append(f"sample_manifest[{index}] missing {key}.")
        label = entry.get("label")
        if label not in REQUIRED_LABELS:
            errors.append(f"sample_manifest[{index}].label must be real or synthetic.")
        else:
            labels_seen.add(label)
        image_path = entry.get("image_path")
        if not isinstance(image_path, str) or not image_path.strip():
            errors.append(f"sample_manifest[{index}].image_path must be a non-empty string.")
            continue
        if _is_remote(image_path):
            errors.append(f"sample_manifest[{index}].image_path must not be a URL.")
        if not os.path.isabs(image_path):
            errors.append(f"sample_manifest[{index}].image_path must be an explicit absolute file path.")
        if image_path.endswith(("/", "\\")):
            errors.append(f"sample_manifest[{index}].image_path must be a file path, not a directory.")
        if image_path in roots:
            errors.append(f"sample_manifest[{index}].image_path must not be an approved root directory.")
        if os.path.basename(image_path) in ("", ".", ".."):
            errors.append(f"sample_manifest[{index}].image_path must include a file name.")
        if roots and not any(os.path.commonpath([os.path.abspath(root), os.path.abspath(image_path)]) == os.path.abspath(root) for root in roots if os.path.isabs(root)):
            errors.append(f"sample_manifest[{index}].image_path must be under approved_local_roots.")
    if manifest and len(manifest) > limits.get("max_samples", 0):
        errors.append("sample_manifest length exceeds tiny_limits.max_samples.")
    for key in ("max_samples", "max_steps", "max_epochs"):
        requested = raw.get(key)
        cap = limits.get(key)
        if requested is not None:
            if not isinstance(requested, int) or isinstance(requested, bool):
                errors.append(f"{key} must be an integer when provided.")
            elif isinstance(cap, int) and requested > cap:
                errors.append(f"{key} exceeds tiny_limits.{key}.")
    if manifest and labels_seen != REQUIRED_LABELS:
        errors.append("approved_local_smoke sample_manifest must include both real and synthetic classes.")


def validate_config(raw: Dict[str, Any]) -> None:
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object.")
    errors: List[str] = []
    kind = raw.get("config_kind")
    if kind not in CONFIG_KINDS:
        errors.append("config_kind must be example_symbolic or approved_local_smoke.")
    if kind == "example_symbolic":
        _validate_example(raw, errors)
    elif kind == "approved_local_smoke":
        _validate_approved(raw, errors)
    if errors:
        raise ValueError(
            "CF-Small tiny train smoke config validation failed:\n"
            + "\n".join(f"  - {msg}" for msg in errors)
        )


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    validate_config(raw)
    return raw


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cf_small_tiny_train_smoke_config.json>", file=sys.stderr)
        sys.exit(1)
    try:
        raw = load_config(sys.argv[1])
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    print(
        "CF_SMALL_TINY_TRAIN_SMOKE_CONFIG_OK: "
        f"{raw.get('config_kind')} config is valid, guarded, and non-SNS."
    )


if __name__ == "__main__":
    main()
