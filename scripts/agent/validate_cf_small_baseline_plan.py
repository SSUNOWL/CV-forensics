#!/usr/bin/env python3
"""Validate a symbolic CF-Small baseline training/evaluation plan.

This validator checks plan structure only. It does not inspect dataset
directories, read images or masks, run training/evaluation, write artifacts, or
write checkpoints.
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
except ImportError as exc:  # pragma: no cover
    check_local_data_readiness_config_safety = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


_REQUIRED_KEYS = (
    "schema_version",
    "plan_name",
    "dataset_name",
    "execution_mode",
    "dry_run",
    "no_download",
    "no_training",
    "no_network",
    "no_outputs",
    "no_checkpoints",
    "no_real_image_reading",
    "no_real_mask_reading",
    "requires_user_approval_for_real_training",
    "local_data_gate_ref",
    "cf_small_subset_smoke_ref",
    "dataset_role",
    "training_objective_plan",
    "backbone_plan",
    "classification_policy",
    "family_policy",
    "generator_holdout_policy",
    "split_policy",
    "evaluation_policy",
    "metric_plan",
    "output_policy",
    "checkpoint_policy",
    "handoff_to_sid_set",
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
    "requires_user_approval_for_real_training",
)

_FAMILY_LABELS = {"LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A", "Unknown"}
_REQUIRED_METRICS = (
    "binary accuracy",
    "macro f1",
    "generator-family accuracy",
    "holdout/generalization evaluation",
)


def check_config_safety(raw: Dict[str, Any]) -> None:
    if check_local_data_readiness_config_safety is None:
        raise ValueError(f"Cannot import cv_forensics.local_data_gate: {_IMPORT_ERROR}")
    check_local_data_readiness_config_safety(raw, context="cf_small_baseline_plan_config")


def _as_dict(raw: Dict[str, Any], key: str, errors: List[str]) -> Dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key!r} must be an object.")
        return {}
    return value


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


def _all_text(node: Any) -> str:
    if isinstance(node, dict):
        return " ".join(_all_text(k) + " " + _all_text(v) for k, v in node.items())
    if isinstance(node, list):
        return " ".join(_all_text(v) for v in node)
    return str(node)


def validate_config(raw: Dict[str, Any]) -> None:
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object.")

    errors: List[str] = []

    for key in _REQUIRED_KEYS:
        if key not in raw:
            errors.append(f"Missing required key: {key!r}.")

    for flag in _GUARDRAIL_FLAGS:
        if raw.get(flag) is not True:
            errors.append(f"{flag!r} must be true.")

    if raw.get("dataset_name") != "Community Forensics-Small":
        errors.append("'dataset_name' must be 'Community Forensics-Small'.")
    if raw.get("execution_mode") != "plan_only":
        errors.append("'execution_mode' must be 'plan_only'.")

    dataset_role = _as_dict(raw, "dataset_role", errors)
    if dataset_role:
        for key in (
            "shared_backbone_planning",
            "real_vs_synthetic_baseline_planning",
            "coarse_provenance_family_planning",
        ):
            if dataset_role.get(key) is not True:
                errors.append(f"dataset_role.{key} must be true.")
        if dataset_role.get("not_primary_tampered_localization_source") is not True:
            errors.append("dataset_role must state CF-Small is not the primary tampered localization source.")

    training_objective = _as_dict(raw, "training_objective_plan", errors)
    if training_objective:
        if training_objective.get("mode") != "planning_only":
            errors.append("training_objective_plan.mode must be 'planning_only'.")
        if training_objective.get("no_real_optimization") is not True:
            errors.append("training_objective_plan.no_real_optimization must be true.")

    classification = _as_dict(raw, "classification_policy", errors)
    if classification:
        labels = classification.get("labels")
        if not isinstance(labels, list):
            errors.append("classification_policy.labels must be a list.")
        else:
            if "real" not in labels:
                errors.append("classification_policy.labels must include real.")
            if "synthetic" not in labels:
                errors.append("classification_policy.labels must include synthetic.")
        if classification.get("tampered_localization_required") is not False:
            errors.append("classification_policy.tampered_localization_required must be false.")
        if _contains_text(classification, ("requires tampered localization", "tampered localization dataset")):
            errors.append("classification_policy must not require tampered localization for CF-Small.")

    family = _as_dict(raw, "family_policy", errors)
    if family:
        labels = family.get("labels")
        if not isinstance(labels, list) or not _FAMILY_LABELS.issubset(set(labels)):
            errors.append("family_policy.labels must include LatDiff, PixDiff, GAN, Other, Real-or-N/A, and Unknown.")
        if not _contains_text(family, ("coarse", "provenance", "family")):
            errors.append("family_policy must describe coarse family/provenance planning.")

    holdout = _as_dict(raw, "generator_holdout_policy", errors)
    if holdout:
        if not _contains_text(holdout, ("model_name", "generator")):
            errors.append("generator_holdout_policy must include model-name or generator-family holdout.")
        if holdout.get("avoid_train_validation_leakage") is not True:
            errors.append("generator_holdout_policy.avoid_train_validation_leakage must be true.")
        if holdout.get("diversity_aware_split_planning") is not True:
            errors.append("generator_holdout_policy.diversity_aware_split_planning must be true.")
        if holdout.get("random_only_split_allowed") is not False:
            errors.append("generator_holdout_policy.random_only_split_allowed must be false.")

    split = _as_dict(raw, "split_policy", errors)
    if split:
        if not _contains_text(split, ("train", "validation")):
            errors.append("split_policy must explicitly mention train and validation planning.")
        if split.get("generator_holdout_required") is not True:
            errors.append("split_policy.generator_holdout_required must be true.")

    evaluation = _as_dict(raw, "evaluation_policy", errors)
    if evaluation:
        if evaluation.get("mode") != "plan_only":
            errors.append("evaluation_policy.mode must be 'plan_only'.")
        if evaluation.get("run_evaluation") is not False:
            errors.append("evaluation_policy.run_evaluation must be false.")

    metric = _as_dict(raw, "metric_plan", errors)
    if metric:
        metric_text = _all_text(metric).lower()
        for required in _REQUIRED_METRICS:
            if required not in metric_text:
                errors.append(f"metric_plan must include {required}.")
        if metric.get("real_mask_iou_required_for_cf_small") is not False:
            errors.append("metric_plan.real_mask_iou_required_for_cf_small must be false.")
        if _contains_text(metric, ("required real mask iou", "require real mask iou", "mask iou required")):
            errors.append("metric_plan must not require real mask IoU for CF-Small.")

    output_policy = _as_dict(raw, "output_policy", errors)
    if output_policy:
        if output_policy.get("write_outputs") is not False:
            errors.append("output_policy.write_outputs must be false.")
        if output_policy.get("requires_explicit_approval_before_writing") is not True:
            errors.append("output_policy must require explicit approval before writing.")

    checkpoint_policy = _as_dict(raw, "checkpoint_policy", errors)
    if checkpoint_policy:
        if checkpoint_policy.get("write_checkpoints") is not False:
            errors.append("checkpoint_policy.write_checkpoints must be false.")
        if checkpoint_policy.get("requires_explicit_approval_before_writing") is not True:
            errors.append("checkpoint_policy must require explicit approval before writing.")

    handoff = _as_dict(raw, "handoff_to_sid_set", errors)
    if handoff:
        if handoff.get("sid_set_handles_3_way_classification") is not True:
            errors.append("handoff_to_sid_set.sid_set_handles_3_way_classification must be true.")
        if handoff.get("sid_set_handles_tampered_localization") is not True:
            errors.append("handoff_to_sid_set.sid_set_handles_tampered_localization must be true.")

    if errors:
        raise ValueError(
            "CF-Small baseline plan validation failed:\n"
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
        print(f"Usage: {sys.argv[0]} <cf_small_baseline_plan.json>", file=sys.stderr)
        sys.exit(1)

    try:
        raw = load_config(sys.argv[1])
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        "CF_SMALL_BASELINE_PLAN_OK: "
        f"{raw['dataset_name']} baseline plan is symbolic, plan-only, "
        "dry-run safe, holdout-aware, and approval-gated."
    )


if __name__ == "__main__":
    main()
