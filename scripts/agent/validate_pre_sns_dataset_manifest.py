#!/usr/bin/env python3
"""Validate pre-SNS dataset manifest gate configs without reading data files."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_manifest import (  # noqa: E402
    CLASS_LABELS,
    FAMILY_LABELS,
    SOURCE_DATASETS,
    normalize_manifest,
    validate_explicit_local_file,
    validate_sample,
    walk_safety,
)

OK_MARKER = "PRE_SNS_DATASET_MANIFEST_OK"
APPROVAL_TEXT = "I_APPROVE_LOCAL_NON_SNS_PRE_SNS_DATASET_MANIFEST"
CONFIG_KINDS = {"example_symbolic", "approved_local_pre_sns_manifest"}
SOURCE_DATASET_ALIASES = {
    "Community Forensics-Small": "community_forensics_small",
    "CF-Small": "community_forensics_small",
    "community_forensics_small": "community_forensics_small",
    "SID-Set": "sid_set",
    "sid_set": "sid_set",
}
GUARDRAIL_FLAGS = (
    "no_download",
    "no_network",
    "no_training",
    "no_outputs",
    "no_checkpoints",
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
    for flag in GUARDRAIL_FLAGS:
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))


def _policy_labels(raw: dict[str, Any], key: str) -> set[str]:
    policy = raw.get(key)
    if not isinstance(policy, dict):
        return set()
    labels = policy.get("labels")
    if not isinstance(labels, list):
        return set()
    return {label for label in labels if isinstance(label, str)}


def _validate_common(raw: dict[str, Any], errors: list[str]) -> None:
    _require_true(raw, errors)
    if _policy_labels(raw, "class_policy") != set(CLASS_LABELS):
        errors.append(_err("class_policy.labels must match real, full_synthetic, tampered"))
    if _policy_labels(raw, "family_policy") != set(FAMILY_LABELS):
        errors.append(_err("family_policy.labels must match the supported family labels"))
    source_policy = raw.get("source_dataset_policy")
    if not isinstance(source_policy, dict):
        errors.append(_err("source_dataset_policy must be an object"))
    else:
        supported = source_policy.get("supported_sources")
        if set(supported or []) != set(SOURCE_DATASETS):
            errors.append(_err("source_dataset_policy.supported_sources must include CF-Small and SID-Set"))


def _validate_example(raw: dict[str, Any], errors: list[str]) -> None:
    if raw.get("execution_mode") != "example_only":
        errors.append(_err("execution_mode must be example_only"))
    _validate_common(raw, errors)
    for issue in walk_safety(raw):
        errors.append(_err(f"{issue.path}: {issue.message}"))


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


def _sample_list(raw: dict[str, Any], errors: list[str]) -> tuple[list[dict[str, Any]], str]:
    samples = raw.get("samples")
    sample_manifest = raw.get("sample_manifest")
    if samples is not None and sample_manifest is not None:
        errors.append(_err("use only one of samples or sample_manifest"))
        return [], "samples"
    key = "sample_manifest" if sample_manifest is not None else "samples"
    selected = sample_manifest if sample_manifest is not None else samples
    if not isinstance(selected, list) or not selected:
        errors.append(_err(f"{key} must be a non-empty list"))
        return [], key
    typed_samples: list[dict[str, Any]] = []
    for index, sample in enumerate(selected):
        if not isinstance(sample, dict):
            errors.append(_err(f"{key}[{index}] must be an object"))
            continue
        typed_samples.append(sample)
    return typed_samples, key


def _validate_existing_sample_file(path_value: str, path_name: str, errors: list[str]) -> None:
    if not os.path.isfile(path_value):
        errors.append(_err(f"{path_name}: must exist as a file"))


def _normalize_sample_for_validation(sample: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(sample)
    source_value = normalized.get("source_dataset", normalized.get("dataset"))
    if isinstance(source_value, str) and source_value in SOURCE_DATASET_ALIASES:
        normalized["source_dataset"] = SOURCE_DATASET_ALIASES[source_value]
    if "class_label" not in normalized and isinstance(normalized.get("label"), str):
        normalized["class_label"] = normalized["label"]
    return normalized


def _validate_roots(raw: dict[str, Any], errors: list[str]) -> tuple[list[str], list[str]]:
    roots = raw.get("approved_local_roots")
    output_roots = raw.get("approved_local_output_roots", [])
    if not isinstance(roots, list) or not roots or not all(isinstance(root, str) for root in roots):
        errors.append(_err("approved_local_roots must be a non-empty list of absolute paths"))
        roots = []
    if not isinstance(output_roots, list) or not output_roots or not all(isinstance(root, str) for root in output_roots):
        errors.append(_err("approved_local_output_roots must be a non-empty list of absolute paths"))
        output_roots = []
    allowed = set(roots) | set(output_roots)
    for root in list(allowed):
        root_issues = walk_safety(root, "approved_root", allowed_abs_values=allowed)
        if root_issues:
            for issue in root_issues:
                errors.append(_err(f"{issue.path}: {issue.message}"))
    return roots, output_roots


def _validate_samples(raw: dict[str, Any], roots: list[str], output_roots: list[str], errors: list[str]) -> list[dict[str, Any]]:
    samples, sample_key = _sample_list(raw, errors)
    if not samples:
        return []
    allowed_abs = set(roots) | set(output_roots)
    if isinstance(raw.get("manifest_output_path"), str):
        allowed_abs.add(raw["manifest_output_path"])
    for sample in samples:
        for field in ("image_path", "mask_path"):
            if isinstance(sample.get(field), str):
                allowed_abs.add(sample[field])
    for issue in walk_safety(raw, allowed_abs_values=allowed_abs):
        errors.append(_err(f"{issue.path}: {issue.message}"))
    seen_classes: set[str] = set()
    seen_sources: set[str] = set()
    normalized_samples: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        normalized_sample = _normalize_sample_for_validation(sample)
        normalized_samples.append(normalized_sample)
        for issue in validate_sample(normalized_sample, index):
            errors.append(_err(f"{issue.path}: {issue.message}"))
        image_path = sample.get("image_path")
        if isinstance(image_path, str):
            path_name = f"{sample_key}[{index}].image_path"
            for issue in validate_explicit_local_file(image_path, roots, path_name):
                errors.append(_err(f"{issue.path}: {issue.message}"))
            _validate_existing_sample_file(image_path, path_name, errors)
        mask_path = sample.get("mask_path")
        if isinstance(mask_path, str):
            path_name = f"{sample_key}[{index}].mask_path"
            for issue in validate_explicit_local_file(mask_path, roots, path_name):
                errors.append(_err(f"{issue.path}: {issue.message}"))
            _validate_existing_sample_file(mask_path, path_name, errors)
        if normalized_sample.get("class_label") in CLASS_LABELS:
            seen_classes.add(normalized_sample["class_label"])
        if normalized_sample.get("source_dataset") in SOURCE_DATASETS:
            seen_sources.add(normalized_sample["source_dataset"])
    for label in sorted(set(CLASS_LABELS) - seen_classes):
        errors.append(_err(f"missing required class coverage: {label}"))
    for source in sorted(set(SOURCE_DATASETS) - seen_sources):
        errors.append(_err(f"missing required source coverage: {source}"))
    if not errors:
        try:
            normalize_manifest(normalized_samples)
        except Exception as exc:
            errors.append(_err(f"manifest normalization failed: {exc}"))
    return samples


def _validate_manifest_output(raw: dict[str, Any], output_roots: list[str], errors: list[str]) -> None:
    output_path = raw.get("manifest_output_path")
    if not isinstance(output_path, str) or not output_path.strip():
        errors.append(_err("manifest_output_path is required in approved mode"))
        return
    for issue in validate_explicit_local_file(output_path, output_roots, "manifest_output_path"):
        errors.append(_err(f"{issue.path}: {issue.message}"))
    if output_roots and not _path_is_under(output_path, output_roots):
        errors.append(_err("manifest_output_path must be under approved_local_output_roots"))


def _validate_approved(raw: dict[str, Any], errors: list[str]) -> None:
    if raw.get("approved_real_data_access") is not True:
        errors.append(_err("approved_real_data_access must be true"))
    if raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(_err("user_approval_text does not match required approval phrase"))
    if raw.get("recursive_scan") is True or raw.get("recursive_directory_scan") is True:
        errors.append(_err("recursive scan flags are rejected"))
    _validate_common(raw, errors)
    roots, output_roots = _validate_roots(raw, errors)
    _validate_samples(raw, roots, output_roots, errors)
    _validate_manifest_output(raw, output_roots, errors)


def validate_config(raw: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    config_kind = raw.get("config_kind")
    if config_kind not in CONFIG_KINDS:
        errors.append(_err("config_kind must be example_symbolic or approved_local_pre_sns_manifest"))
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
        print("usage: validate_pre_sns_dataset_manifest.py <config.json>", file=sys.stderr)
        return 2
    try:
        raw = load_config(argv[0])
    except Exception as exc:
        print(f"failed to read config: {exc}", file=sys.stderr)
        return 1
    errors = validate_config(raw)
    if errors:
        print("pre-SNS dataset manifest validation failed:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"{OK_MARKER}: config is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
