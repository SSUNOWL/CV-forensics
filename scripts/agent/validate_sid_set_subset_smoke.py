#!/usr/bin/env python3
"""Validate a symbolic SID-Set local subset smoke config.

This validator checks configuration structure only. It does not inspect dataset
directories, read images or masks, run training, write outputs, or write
checkpoints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


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
except ImportError as exc:  # pragma: no cover - exercised by CLI failure path
    check_local_data_readiness_config_safety = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


_REQUIRED_KEYS = (
    "schema_version",
    "dataset_name",
    "dry_run",
    "no_download",
    "no_training",
    "no_network",
    "no_outputs",
    "no_checkpoints",
    "no_real_image_reading",
    "no_real_mask_reading",
    "local_data_gate_ref",
    "readiness_policy",
    "subset_policy",
    "split_policy",
    "class_policy",
    "mask_policy",
    "family_policy",
    "localization_policy",
    "threshold_tau",
    "expected_task_outputs",
    "validation_notes",
)

_GUARDRAIL_FLAGS = (
    "dry_run",
    "no_download",
    "no_training",
    "no_network",
    "no_outputs",
    "no_checkpoints",
    "no_real_image_reading",
    "no_real_mask_reading",
)

_CLASS_LABELS = {"real", "synthetic", "tampered"}
_EXPECTED_OUTPUTS = {"class", "mask", "family", "reason"}


def check_config_safety(raw: Dict[str, Any]) -> None:
    """Run the repository local-data safety checker on *raw*."""
    if check_local_data_readiness_config_safety is None:
        raise ValueError(f"Cannot import cv_forensics.local_data_gate: {_IMPORT_ERROR}")
    check_local_data_readiness_config_safety(raw, context="sid_set_subset_smoke_config")


def _as_dict(raw: Dict[str, Any], key: str, errors: List[str]) -> Dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key!r} must be an object.")
        return {}
    return value


def _contains_text(node: Any, needles: Iterable[str]) -> bool:
    needle_list = [n.lower() for n in needles]
    if isinstance(node, dict):
        return any(_contains_text(k, needle_list) or _contains_text(v, needle_list) for k, v in node.items())
    if isinstance(node, list):
        return any(_contains_text(v, needle_list) for v in node)
    if isinstance(node, str):
        lower = node.lower()
        return any(n in lower for n in needle_list)
    return False


def validate_config(raw: Dict[str, Any]) -> None:
    """Validate symbolic SID-Set smoke config semantics."""
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object.")

    errors: List[str] = []

    for key in _REQUIRED_KEYS:
        if key not in raw:
            errors.append(f"Missing required key: {key!r}.")

    for flag in _GUARDRAIL_FLAGS:
        if raw.get(flag) is not True:
            errors.append(f"{flag!r} must be true.")

    if raw.get("dataset_name") != "SID-Set":
        errors.append("'dataset_name' must be 'SID-Set'.")

    threshold = raw.get("threshold_tau")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        errors.append("'threshold_tau' must be a number between 0.0 and 1.0.")
    elif not 0.0 <= float(threshold) <= 1.0:
        errors.append("'threshold_tau' must be between 0.0 and 1.0.")

    readiness_policy = _as_dict(raw, "readiness_policy", errors)
    if readiness_policy:
        if readiness_policy.get("local_paths_approved") is not False:
            errors.append("readiness_policy.local_paths_approved must be false in the example config.")
        if readiness_policy.get("requires_user_approval") is not True:
            errors.append("readiness_policy.requires_user_approval must be true.")
        if readiness_policy.get("requires_validated_manifest") is not True:
            errors.append("readiness_policy.requires_validated_manifest must be true.")

    subset_policy = _as_dict(raw, "subset_policy", errors)
    if subset_policy:
        if not _contains_text(subset_policy, ("tiny", "symbolic")):
            errors.append("subset_policy must describe a tiny symbolic subset plan.")
        planned = subset_policy.get("planned_samples")
        if not isinstance(planned, dict) or set(planned) != _CLASS_LABELS:
            errors.append("subset_policy.planned_samples must include real, synthetic, and tampered.")
        elif any(not isinstance(v, int) or v <= 0 or v > 16 for v in planned.values()):
            errors.append("subset_policy planned sample counts must be small positive integers no greater than 16.")
        if subset_policy.get("real_file_access") not in ("forbidden", False):
            errors.append("subset_policy.real_file_access must be forbidden.")
        if subset_policy.get("recursive_directory_scan") not in ("forbidden", False):
            errors.append("subset_policy.recursive_directory_scan must be forbidden.")

    class_policy = _as_dict(raw, "class_policy", errors)
    if class_policy:
        labels = class_policy.get("labels")
        if not isinstance(labels, list):
            errors.append("class_policy.labels must be a list.")
        elif set(labels) != _CLASS_LABELS or len(labels) != 3:
            errors.append("class_policy.labels must contain exactly real, synthetic, and tampered.")

    mask_policy = _as_dict(raw, "mask_policy", errors)
    if mask_policy:
        if not _contains_text(mask_policy, ("tampered", "localization")):
            errors.append("mask_policy must represent tampered localization planning.")
        if not _contains_text(mask_policy, ("real", "must not")):
            errors.append("mask_policy must state real samples do not require tampered masks.")
        if not _contains_text(mask_policy, ("synthetic", "may")):
            errors.append("mask_policy must state synthetic samples may omit tampered masks unless annotated.")
        if mask_policy.get("read_real_mask_files") is not False:
            errors.append("mask_policy.read_real_mask_files must be false.")
        if mask_policy.get("read_real_image_files") is not False:
            errors.append("mask_policy.read_real_image_files must be false.")

    family_policy = _as_dict(raw, "family_policy", errors)
    if family_policy:
        if family_policy.get("sid_set_family_labels_may_be_missing") is not True:
            errors.append("family_policy.sid_set_family_labels_may_be_missing must be true.")
        missing_policy = family_policy.get("missing_label_policy")
        if not isinstance(missing_policy, dict):
            errors.append("family_policy.missing_label_policy must be an object.")
        else:
            if missing_policy.get("real") != "Real-or-N/A":
                errors.append("family_policy missing real labels must map to Real-or-N/A.")
            if missing_policy.get("synthetic") != "Unknown" or missing_policy.get("tampered") != "Unknown":
                errors.append("family_policy missing synthetic/tampered labels must map to Unknown.")
        if family_policy.get("provenance_training_gated_until_labels_confirmed") is not True:
            errors.append("family_policy must gate provenance training until labels are confirmed.")

    localization_policy = _as_dict(raw, "localization_policy", errors)
    if localization_policy:
        if not _contains_text(localization_policy, ("conditional", "tampered_score", "threshold_tau")):
            errors.append("localization_policy must include conditional activation using threshold_tau.")
        if not _contains_text(localization_policy, ("mask iou",)):
            errors.append("localization_policy must include mask IoU planning.")
        if not _contains_text(localization_policy, ("localization activation recall",)):
            errors.append("localization_policy must include localization activation recall planning.")

    outputs = raw.get("expected_task_outputs")
    if not isinstance(outputs, list) or set(outputs) != _EXPECTED_OUTPUTS:
        errors.append("expected_task_outputs must contain class, mask, family, and reason.")

    if errors:
        raise ValueError(
            "SID-Set subset smoke config validation failed:\n"
            + "\n".join(f"  - {msg}" for msg in errors)
        )


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    check_config_safety(raw)
    validate_config(raw)
    return raw


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <sid_set_subset_smoke_config.json>", file=sys.stderr)
        sys.exit(1)

    try:
        raw = load_config(sys.argv[1])
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        "SID_SET_SUBSET_SMOKE_OK: "
        f"{raw['dataset_name']} symbolic subset smoke plan is dry-run safe, "
        "3-way-classification aware, and localization-gated."
    )


if __name__ == "__main__":
    main()
