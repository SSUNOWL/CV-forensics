#!/usr/bin/env python3
"""Validate the CF-Small local subset smoke plan config.

This validator is symbolic only. It does not inspect datasets, read images or
masks, train models, write outputs, write checkpoints, or access the network.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "src" / "cv_forensics").is_dir()),
    None,
)
if _REPO_ROOT is not None:
    _SRC_ROOT = str(_REPO_ROOT / "src")
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)


_PROTECTED_DIRS = frozenset({"secrets", "data", "datasets", "outputs", "checkpoints"})
_URL_PREFIXES = ("http://", "https://", "ftp://", "s3://", "gs://", "hf://")
_ABS_PREFIXES = ("/home/", "/mnt/", "/root/", "/users/")
_SECRET_TOKENS = frozenset({"token", "password", "secret", "credential", "auth", "bearer"})
_SECRET_KEY_TOKENS = frozenset({"token", "api_key", "password", "secret", "credential", "auth", "bearer"})
_SAFETY_SKIP_TOP_KEYS = frozenset({"protected_path_exclusions"})

_REQUIRED_KEYS = (
    "schema_version",
    "dry_run",
    "no_download",
    "no_training",
    "no_network",
    "no_outputs",
    "no_checkpoints",
    "dataset",
    "dataset_alias",
    "local_path_gate",
    "manifest_ref",
    "subset_plan",
    "sample_limits",
    "required_metadata",
    "holdout_plan",
    "validation_plan",
    "approval",
    "forbidden_actions",
)

_GUARDRAIL_FLAGS = (
    "dry_run",
    "no_download",
    "no_training",
    "no_network",
    "no_outputs",
    "no_checkpoints",
)

_REQUIRED_METADATA = frozenset({"architecture", "model_name", "subset"})


def _tokens(value: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", value.lower()) if t]


def _has_secret_key(key: str) -> bool:
    lower = key.lower()
    if lower in _SECRET_KEY_TOKENS:
        return True
    toks = _tokens(lower)
    return any(t in _SECRET_TOKENS for t in toks) or ("api" in toks and "key" in toks)


def _unsafe_string_reason(value: str) -> Optional[str]:
    stripped = value.strip()
    if not stripped:
        return None
    lower = stripped.lower()
    for prefix in _URL_PREFIXES:
        if prefix in lower:
            return f"URL-like reference: {value!r}"
    for prefix in _ABS_PREFIXES:
        if lower.startswith(prefix):
            return f"absolute machine path: {value!r}"
    if len(stripped) >= 3 and stripped[0].isalpha() and stripped[1] == ":" and stripped[2] in ("\\", "/"):
        return f"Windows drive path: {value!r}"
    if stripped == ".env" or stripped.startswith(".env."):
        return f".env file reference: {value!r}"
    for part in re.split(r"[/\\]+", stripped):
        if part in _PROTECTED_DIRS:
            return f"protected directory {part!r} in path {value!r}"
    toks = _tokens(lower)
    for token in _SECRET_TOKENS:
        if token in toks:
            return f"secret-looking value with standalone token {token!r}: {value!r}"
    if "api" in toks and "key" in toks:
        return f"secret-looking value with api key tokens: {value!r}"
    return None


def check_config_safety(raw: Any, context: str = "cf_small_subset_smoke_config") -> None:
    """Reject protected paths, URLs, and secret-like keys/values."""
    if isinstance(raw, dict):
        for key, val in raw.items():
            child = f"{context}.{key}"
            if isinstance(key, str) and _has_secret_key(key):
                raise ValueError(f"{child}: secret-looking key name {key!r}")
            if context == "cf_small_subset_smoke_config" and key in _SAFETY_SKIP_TOP_KEYS:
                continue
            check_config_safety(val, child)
    elif isinstance(raw, list):
        for idx, item in enumerate(raw):
            check_config_safety(item, f"{context}[{idx}]")
    elif isinstance(raw, str):
        reason = _unsafe_string_reason(raw)
        if reason:
            raise ValueError(f"{context}: {reason}")


def _as_dict(value: Any, field: str, errors: List[str]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{field!r} must be an object.")
        return {}
    return value


def _contains_any_text(node: Any, needles: Iterable[str]) -> bool:
    lowered = [n.lower() for n in needles]
    if isinstance(node, dict):
        return any(_contains_any_text(k, lowered) or _contains_any_text(v, lowered) for k, v in node.items())
    if isinstance(node, list):
        return any(_contains_any_text(v, lowered) for v in node)
    if isinstance(node, str):
        value = node.lower()
        return any(n in value for n in lowered)
    return False


def _contains_true(node: Any, keys: Iterable[str]) -> bool:
    key_set = set(keys)
    if isinstance(node, dict):
        for key, value in node.items():
            if key in key_set and value is True:
                return True
            if _contains_true(value, key_set):
                return True
    elif isinstance(node, list):
        return any(_contains_true(v, key_set) for v in node)
    return False


def validate_config(raw: Dict[str, Any]) -> None:
    errors: List[str] = []

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object.")

    for key in _REQUIRED_KEYS:
        if key not in raw:
            errors.append(f"Missing required key: {key!r}.")

    for flag in _GUARDRAIL_FLAGS:
        if raw.get(flag) is not True:
            errors.append(f"{flag!r} must be true.")

    if raw.get("dataset") != "Community Forensics-Small":
        errors.append("'dataset' must be 'Community Forensics-Small'.")
    if raw.get("dataset_alias") != "CF-Small":
        errors.append("'dataset_alias' must be 'CF-Small'.")

    local_path_gate = _as_dict(raw.get("local_path_gate"), "local_path_gate", errors)
    if local_path_gate:
        if local_path_gate.get("requires_user_approved_local_paths") is not True:
            errors.append("local_path_gate.requires_user_approved_local_paths must be true.")
        if local_path_gate.get("real_local_paths_approved") is not False:
            errors.append("local_path_gate.real_local_paths_approved must be false in the example config.")
        if str(local_path_gate.get("path_mode", "")) != "symbolic_only":
            errors.append("local_path_gate.path_mode must be 'symbolic_only'.")
        if "approval" not in str(local_path_gate).lower():
            errors.append("local_path_gate must explicitly mention user approval.")

    manifest_ref = raw.get("manifest_ref")
    if not manifest_ref:
        errors.append("manifest_ref must be present and non-empty.")
    elif not _contains_any_text(manifest_ref, ("configs/", "community_forensics_small", "readiness")):
        errors.append("manifest_ref must reference safe symbolic manifest/readiness config information.")

    subset_plan = _as_dict(raw.get("subset_plan"), "subset_plan", errors)
    if subset_plan:
        if not _contains_any_text(subset_plan, ("tiny", "symbolic", "manifest")):
            errors.append("subset_plan must describe a tiny symbolic manifest-based smoke plan.")
        for key in ("image_reading", "mask_reading", "recursive_directory_scan", "file_enumeration"):
            value = subset_plan.get(key)
            if value not in (False, "forbidden", "disabled"):
                errors.append(f"subset_plan.{key} must forbid real access.")

    sample_limits = _as_dict(raw.get("sample_limits"), "sample_limits", errors)
    if sample_limits:
        max_total = sample_limits.get("max_total_manifest_rows")
        if not isinstance(max_total, int) or max_total <= 0 or max_total > 64:
            errors.append("sample_limits.max_total_manifest_rows must be a small positive integer no greater than 64.")

    metadata = raw.get("required_metadata")
    if not isinstance(metadata, list):
        errors.append("required_metadata must be a list.")
    else:
        missing = sorted(_REQUIRED_METADATA - set(metadata))
        if missing:
            errors.append(f"required_metadata is missing: {missing}.")

    holdout_plan = _as_dict(raw.get("holdout_plan"), "holdout_plan", errors)
    if holdout_plan:
        if not _contains_any_text(holdout_plan, ("generator", "model_name", "holdout")):
            errors.append("holdout_plan must represent generator/model-name holdout planning.")
        if holdout_plan.get("random_only_validation_allowed") is not False:
            errors.append("holdout_plan.random_only_validation_allowed must be false.")
        if not _contains_any_text(holdout_plan, ("random-only", "random only", "not sufficient")):
            errors.append("holdout_plan must explicitly state random-only validation is insufficient.")

    validation_plan = _as_dict(raw.get("validation_plan"), "validation_plan", errors)
    if validation_plan:
        if validation_plan.get("scope") != "planning_only":
            errors.append("validation_plan.scope must be 'planning_only'.")
        false_flags = (
            "read_real_images",
            "read_real_masks",
            "run_training",
            "write_prediction_artifacts",
            "write_model_artifacts",
        )
        for flag in false_flags:
            if validation_plan.get(flag) is not False:
                errors.append(f"validation_plan.{flag} must be false.")

    approval = _as_dict(raw.get("approval"), "approval", errors)
    if approval and approval.get("local_data_approved") is not False:
        errors.append("approval.local_data_approved must be false in the example config.")

    forbidden_actions = raw.get("forbidden_actions")
    if not isinstance(forbidden_actions, list) or not forbidden_actions:
        errors.append("forbidden_actions must be a non-empty list.")
    else:
        for required in ("download", "training", "network", "image", "mask", "checkpoint", "SNS"):
            if not _contains_any_text(forbidden_actions, (required,)):
                errors.append(f"forbidden_actions must include a {required!r} guardrail.")

    if _contains_true(raw, ("read_real_images", "read_real_masks", "run_training", "download", "write_outputs", "write_checkpoints")):
        errors.append("Config requests an action that must remain disabled.")

    if errors:
        raise ValueError("CF-Small subset smoke config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    check_config_safety(raw)
    validate_config(raw)
    return raw


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cf_small_subset_smoke_config.json>", file=sys.stderr)
        sys.exit(1)
    try:
        raw = load_config(sys.argv[1])
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    print(
        "CF_SMALL_SUBSET_SMOKE_OK: "
        f"{raw['dataset_alias']} symbolic local subset smoke plan is dry-run safe, "
        "local-path gated, and holdout-aware."
    )


if __name__ == "__main__":
    main()
