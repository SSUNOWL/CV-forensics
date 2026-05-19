"""Pre-SNS unified manifest helpers.

The helpers in this module validate and normalize manifest records only. They
do not read image or mask files, scan directories, download data, train models,
or write files.
"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import Any, Dict, Iterable, List, Optional

SOURCE_CF_SMALL = "community_forensics_small"
SOURCE_SID_SET = "sid_set"
SOURCE_DATASETS = (SOURCE_CF_SMALL, SOURCE_SID_SET)

CLASS_LABELS = ("real", "full_synthetic", "tampered")
FAMILY_LABELS = ("LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A")
SID_LABEL_IDS = {"real": 0, "full_synthetic": 1, "tampered": 2}

PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}
REMOTE_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
SECRET_KEY_RE = re.compile(r"(^|_)(api[_-]?key|secret|token|password|credential|private[_-]?key)($|_)", re.I)


@dataclasses.dataclass(frozen=True)
class ManifestIssue:
    path: str
    message: str


def _issue(path: str, message: str) -> ManifestIssue:
    return ManifestIssue(path=path, message=message)


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


def walk_safety(value: Any, path: str = "", allowed_abs_values: Optional[set[str]] = None) -> List[ManifestIssue]:
    """Recursively reject unsafe references in keys and string values."""
    allowed_abs_values = allowed_abs_values or set()
    issues: List[ManifestIssue] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if SECRET_KEY_RE.search(key_text):
                issues.append(_issue(child_path, "secret-like key rejected"))
            _contains_protected_part(key_text) and issues.append(_issue(child_path, "protected path segment rejected in key"))
            issues.extend(walk_safety(child, child_path, allowed_abs_values))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(walk_safety(child, f"{path}[{index}]", allowed_abs_values))
    elif isinstance(value, str):
        text = value.strip()
        if REMOTE_RE.search(text):
            issues.append(_issue(path, "URL or remote scheme rejected"))
        if WINDOWS_DRIVE_RE.search(text):
            issues.append(_issue(path, "Windows drive path rejected"))
        if text.startswith("/") and text not in allowed_abs_values:
            issues.append(_issue(path, "absolute path rejected"))
        if _contains_protected_part(text):
            issues.append(_issue(path, "protected path segment rejected"))
        if _has_path_traversal(text):
            issues.append(_issue(path, "path traversal rejected"))
        if _looks_secret_value(text):
            issues.append(_issue(path, "secret-like value rejected"))
    return issues


def path_is_under(path: str, roots: Iterable[str]) -> bool:
    real_path = os.path.realpath(path)
    for root in roots:
        real_root = os.path.realpath(root)
        try:
            if os.path.commonpath([real_path, real_root]) == real_root:
                return True
        except ValueError:
            continue
    return False


def validate_explicit_local_file(path_value: str, roots: Iterable[str], path_name: str) -> List[ManifestIssue]:
    issues: List[ManifestIssue] = []
    if REMOTE_RE.search(path_value) or WINDOWS_DRIVE_RE.search(path_value):
        issues.append(_issue(path_name, "must be a local path"))
    if not path_value.startswith("/"):
        issues.append(_issue(path_name, "must be absolute in approved local mode"))
    if path_value.endswith("/") or os.path.basename(path_value) in {"", ".", ".."}:
        issues.append(_issue(path_name, "must be an explicit file path"))
    if _has_path_traversal(path_value):
        issues.append(_issue(path_name, "must not contain path traversal"))
    if _contains_protected_part(path_value):
        issues.append(_issue(path_name, "must not contain protected path segments"))
    if roots and not path_is_under(path_value, roots):
        issues.append(_issue(path_name, "must be under approved roots"))
    return issues


def _bool_tasks(class_available: bool, family_available: bool, localization_available: bool) -> Dict[str, bool]:
    return {
        "class": class_available,
        "family": family_available,
        "localization": localization_available,
    }


def _loss_routing(tasks_available: Dict[str, bool]) -> Dict[str, bool]:
    return {
        "class_loss": bool(tasks_available["class"]),
        "family_loss": bool(tasks_available["family"]),
        "localization_loss": bool(tasks_available["localization"]),
    }


def normalize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one CF-Small or SID-Set sample into the unified schema."""
    source = sample.get("source_dataset")
    if source not in SOURCE_DATASETS:
        raise ValueError(f"source_dataset must be one of {SOURCE_DATASETS}")
    class_label = sample.get("class_label")
    if class_label not in CLASS_LABELS:
        raise ValueError("class_label must be real, full_synthetic, or tampered")

    normalized: Dict[str, Any] = {
        "source_dataset": source,
        "sample_id": sample["sample_id"],
        "image_path": sample["image_path"],
        "class_label": class_label,
    }
    if source == SOURCE_CF_SMALL:
        for field in ("family_label", "architecture", "model_name", "subset"):
            normalized[field] = sample[field]
        tasks = _bool_tasks(True, True, False)
    else:
        normalized["label_id"] = sample["label_id"]
        for field in ("split", "img_id"):
            if field in sample:
                normalized[field] = sample[field]
        if class_label == "tampered":
            normalized["mask_path"] = sample["mask_path"]
        tasks = _bool_tasks(True, False, class_label == "tampered" and "mask_path" in sample)

    normalized["tasks_available"] = tasks
    normalized["loss_routing"] = _loss_routing(tasks)
    return normalized


def validate_sample(sample: Dict[str, Any], index: int) -> List[ManifestIssue]:
    issues: List[ManifestIssue] = []
    prefix = f"samples[{index}]"
    source = sample.get("source_dataset")
    if source not in SOURCE_DATASETS:
        issues.append(_issue(f"{prefix}.source_dataset", "unsupported source_dataset"))
    for field in ("sample_id", "image_path", "class_label"):
        if not isinstance(sample.get(field), str) or not sample.get(field).strip():
            issues.append(_issue(f"{prefix}.{field}", "required non-empty string"))
    class_label = sample.get("class_label")
    if class_label not in CLASS_LABELS:
        issues.append(_issue(f"{prefix}.class_label", "invalid class label"))

    if source == SOURCE_CF_SMALL:
        for field in ("family_label", "architecture", "model_name", "subset"):
            if not isinstance(sample.get(field), str) or not sample.get(field).strip():
                issues.append(_issue(f"{prefix}.{field}", "required for CF-Small samples"))
        if sample.get("family_label") not in FAMILY_LABELS:
            issues.append(_issue(f"{prefix}.family_label", "invalid family label"))
    elif source == SOURCE_SID_SET:
        label_id = sample.get("label_id")
        if not isinstance(label_id, int) or isinstance(label_id, bool):
            issues.append(_issue(f"{prefix}.label_id", "required integer for SID-Set samples"))
        elif class_label in SID_LABEL_IDS and label_id != SID_LABEL_IDS[class_label]:
            issues.append(_issue(f"{prefix}.label_id", "label_id does not match class_label"))
        if class_label == "tampered":
            if not isinstance(sample.get("mask_path"), str) or not sample.get("mask_path").strip():
                issues.append(_issue(f"{prefix}.mask_path", "tampered SID-Set sample requires mask_path"))
        elif "mask_path" in sample:
            issues.append(_issue(f"{prefix}.mask_path", "non-tampered SID-Set sample must not include mask_path"))
        for optional in ("split", "img_id"):
            if optional in sample and not isinstance(sample[optional], str):
                issues.append(_issue(f"{prefix}.{optional}", "must be a string when present"))
    return issues


def normalize_manifest(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    normalized_samples = [normalize_sample(sample) for sample in samples]
    class_labels = sorted({sample["class_label"] for sample in normalized_samples})
    source_datasets = sorted({sample["source_dataset"] for sample in normalized_samples})
    return {
        "schema_version": "1.0",
        "manifest_kind": "pre_sns_unified_dataset_manifest",
        "source_datasets": source_datasets,
        "class_labels": class_labels,
        "family_labels": list(FAMILY_LABELS),
        "samples": normalized_samples,
        "summary": {
            "sample_count": len(normalized_samples),
            "class_supervision_count": sum(1 for sample in normalized_samples if sample["tasks_available"]["class"]),
            "family_supervision_count": sum(1 for sample in normalized_samples if sample["tasks_available"]["family"]),
            "localization_supervision_count": sum(1 for sample in normalized_samples if sample["tasks_available"]["localization"]),
        },
    }
