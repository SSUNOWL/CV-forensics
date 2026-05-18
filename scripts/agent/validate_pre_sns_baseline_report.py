#!/usr/bin/env python3
"""Validate a symbolic pre-SNS baseline report scaffold.

This validator checks report scaffold structure only. It does not inspect
datasets, read images or masks, run evaluation, train models, write reports,
write predictions, or write checkpoints.
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
    "report_name",
    "report_stage",
    "execution_mode",
    "dry_run",
    "no_download",
    "no_training",
    "no_network",
    "no_outputs",
    "no_checkpoints",
    "no_real_evaluation",
    "no_real_image_reading",
    "no_real_mask_reading",
    "no_sns_augmentation",
    "no_sns_perturbation_eval",
    "requires_user_approval_for_real_evaluation",
    "local_data_gate_ref",
    "cf_small_baseline_ref",
    "sid_set_baseline_ref",
    "report_scope",
    "dataset_scope",
    "model_output_scope",
    "metric_fields",
    "metric_status_policy",
    "baseline_collection_policy",
    "comparison_policy",
    "output_policy",
    "checkpoint_policy",
    "sns_future_stage_policy",
    "validation_notes",
)

_GUARDRAIL_FLAGS = (
    "dry_run",
    "no_download",
    "no_training",
    "no_network",
    "no_outputs",
    "no_checkpoints",
    "no_real_evaluation",
    "no_real_image_reading",
    "no_real_mask_reading",
    "no_sns_augmentation",
    "no_sns_perturbation_eval",
    "requires_user_approval_for_real_evaluation",
)

_REQUIRED_OUTPUT_SCOPE = (
    ("class", ("class",)),
    ("mask/localization", ("mask", "localization")),
    ("family/provenance", ("family", "provenance")),
    ("reason", ("reason",)),
)

_REQUIRED_METRICS = (
    "three_way_accuracy",
    "macro_f1",
    "mask_iou",
    "generator_family_accuracy",
    "localization_activation_recall",
    "latency",
    "fps",
)

_METRIC_ENTRY_KEYS = (
    "display_name",
    "status",
    "value",
    "unit_or_definition",
    "source_stage",
    "required_before_sns",
)


def check_config_safety(raw: Dict[str, Any]) -> None:
    if check_local_data_readiness_config_safety is None:
        raise ValueError(f"Cannot import cv_forensics.local_data_gate: {_IMPORT_ERROR}")
    check_local_data_readiness_config_safety(raw, context="pre_sns_baseline_report_config")


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

    if raw.get("report_stage") != "pre_sns_baseline":
        errors.append("'report_stage' must be 'pre_sns_baseline'.")
    if raw.get("execution_mode") != "report_scaffold_only":
        errors.append("'execution_mode' must be 'report_scaffold_only'.")

    report_scope = _as_dict(raw, "report_scope", errors)
    if report_scope:
        if not _contains_text(report_scope, ("pre-SNS baseline", "comparison")):
            errors.append("report_scope must state pre-SNS baseline comparison purpose.")
        if report_scope.get("claims_real_metric_values") is not False:
            errors.append("report_scope.claims_real_metric_values must be false.")

    dataset_scope = _as_dict(raw, "dataset_scope", errors)
    if dataset_scope:
        if not _contains_text(dataset_scope, ("Community Forensics-Small", "shared backbone")):
            errors.append("dataset_scope must include the Community Forensics-Small baseline role.")
        if not _contains_text(dataset_scope, ("SID-Set", "3-way")):
            errors.append("dataset_scope must include the SID-Set baseline role.")
        if dataset_scope.get("real_data_access_in_this_task") is not False:
            errors.append("dataset_scope.real_data_access_in_this_task must be false.")

    model_output_scope = _as_dict(raw, "model_output_scope", errors)
    if model_output_scope:
        for label, needles in _REQUIRED_OUTPUT_SCOPE:
            if not any(model_output_scope.get(k) is True for k in model_output_scope if all(n in k.lower() for n in needles)):
                if not _contains_text(model_output_scope, needles):
                    errors.append(f"model_output_scope must include {label}.")

    metric_fields = _as_dict(raw, "metric_fields", errors)
    if metric_fields:
        for metric_name in _REQUIRED_METRICS:
            entry = metric_fields.get(metric_name)
            if not isinstance(entry, dict):
                errors.append(f"metric_fields must include object {metric_name!r}.")
                continue
            for key in _METRIC_ENTRY_KEYS:
                if key not in entry:
                    errors.append(f"metric_fields.{metric_name} missing {key!r}.")
            if entry.get("status") != "pending_real_baseline_run":
                errors.append(f"metric_fields.{metric_name}.status must be pending_real_baseline_run.")
            if entry.get("value") is not None:
                errors.append(f"metric_fields.{metric_name}.value must be null; real metric values are rejected.")
            if entry.get("required_before_sns") is not True:
                errors.append(f"metric_fields.{metric_name}.required_before_sns must be true.")

    metric_status = _as_dict(raw, "metric_status_policy", errors)
    if metric_status:
        if metric_status.get("current_metric_status") != "pending_real_baseline_run":
            errors.append("metric_status_policy.current_metric_status must be pending_real_baseline_run.")
        if metric_status.get("current_metric_values_must_be_null") is not True:
            errors.append("metric_status_policy.current_metric_values_must_be_null must be true.")
        if metric_status.get("completed_real_metric_values_rejected_in_this_task") is not True:
            errors.append("metric_status_policy must reject completed real metric values in this task.")

    baseline_policy = _as_dict(raw, "baseline_collection_policy", errors)
    if baseline_policy:
        if baseline_policy.get("requires_explicit_user_approval_before_real_metric_collection") is not True:
            errors.append("baseline_collection_policy must require explicit user approval.")
        if baseline_policy.get("run_real_evaluation_now") is not False:
            errors.append("baseline_collection_policy.run_real_evaluation_now must be false.")

    comparison = _as_dict(raw, "comparison_policy", errors)
    if comparison:
        for key in (
            "pre_sns_baseline_anchor",
            "future_sns_robustness_drop",
            "future_sns_augmented_training_comparison",
        ):
            if key not in comparison:
                errors.append(f"comparison_policy must include {key}.")
        future_drop = comparison.get("future_sns_robustness_drop")
        if not isinstance(future_drop, dict):
            errors.append("comparison_policy.future_sns_robustness_drop must be an object.")
        elif future_drop.get("status") != "pending_future_sns_stage":
            errors.append("future_sns_robustness_drop.status must be pending_future_sns_stage.")

    output_policy = _as_dict(raw, "output_policy", errors)
    if output_policy:
        if output_policy.get("write_reports") is not False:
            errors.append("output_policy.write_reports must be false.")
        if output_policy.get("write_predictions") is not False:
            errors.append("output_policy.write_predictions must be false.")
        if output_policy.get("requires_explicit_approval_before_external_report_or_prediction_writing") is not True:
            errors.append("output_policy must require explicit approval before report or prediction writing.")

    checkpoint_policy = _as_dict(raw, "checkpoint_policy", errors)
    if checkpoint_policy:
        if checkpoint_policy.get("write_checkpoints") is not False:
            errors.append("checkpoint_policy.write_checkpoints must be false.")
        if checkpoint_policy.get("requires_explicit_approval_before_writing") is not True:
            errors.append("checkpoint_policy must require explicit approval before writing.")

    sns_policy = _as_dict(raw, "sns_future_stage_policy", errors)
    if sns_policy:
        if sns_policy.get("sns_augmentation_implemented_here") is not False:
            errors.append("sns_future_stage_policy.sns_augmentation_implemented_here must be false.")
        if sns_policy.get("sns_perturbation_evaluation_run_here") is not False:
            errors.append("sns_future_stage_policy.sns_perturbation_evaluation_run_here must be false.")
        if sns_policy.get("sns_stage_tasks_begin_only_after_scaffold_committed_and_explicitly_approved") is not True:
            errors.append("sns_future_stage_policy must gate future SNS-stage tasks.")
        if not _contains_text(sns_policy, ("SNS augmentation is not implemented",)):
            errors.append("sns_future_stage_policy must state SNS augmentation is not implemented here.")
        if not _contains_text(sns_policy, ("SNS perturbation evaluation is not run",)):
            errors.append("sns_future_stage_policy must state SNS perturbation evaluation is not run here.")

    if errors:
        raise ValueError(
            "Pre-SNS baseline report validation failed:\n"
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
        print(f"Usage: {sys.argv[0]} <pre_sns_baseline_report.json>", file=sys.stderr)
        sys.exit(1)

    try:
        raw = load_config(sys.argv[1])
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        "PRE_SNS_BASELINE_REPORT_OK: "
        f"{raw['report_name']} is symbolic, report-scaffold-only, dry-run safe, "
        "metric-placeholder-only, and SNS-gated."
    )


if __name__ == "__main__":
    main()
