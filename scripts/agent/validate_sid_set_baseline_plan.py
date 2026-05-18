#!/usr/bin/env python3
"""Validate a symbolic SID-Set multi-head baseline plan.

This validator checks plan structure only. It does not inspect datasets, read
images or masks, train models, write outputs, or write checkpoints.
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
    "sid_set_subset_smoke_ref",
    "cf_small_baseline_ref",
    "dataset_role",
    "model_plan",
    "head_plan",
    "classification_policy",
    "localization_policy",
    "mask_policy",
    "family_policy",
    "explanation_policy",
    "threshold_policy",
    "loss_plan",
    "freeze_policy",
    "batch_mixing_policy",
    "split_policy",
    "evaluation_policy",
    "metric_plan",
    "output_policy",
    "checkpoint_policy",
    "handoff_to_pre_sns_baseline",
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

_CLASS_LABELS = {"real", "synthetic", "tampered"}
_REQUIRED_METRICS = (
    "3-way accuracy",
    "macro-f1",
    "mask iou",
    "generator-family accuracy",
    "localization activation recall",
    "latency",
    "fps",
)


def check_config_safety(raw: Dict[str, Any]) -> None:
    if check_local_data_readiness_config_safety is None:
        raise ValueError(f"Cannot import cv_forensics.local_data_gate: {_IMPORT_ERROR}")
    check_local_data_readiness_config_safety(raw, context="sid_set_baseline_plan_config")


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


def _metric_text(metric_plan: Dict[str, Any]) -> str:
    return _all_text(metric_plan).lower()


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

    if raw.get("dataset_name") != "SID-Set":
        errors.append("'dataset_name' must be 'SID-Set'.")
    if raw.get("execution_mode") != "plan_only":
        errors.append("'execution_mode' must be 'plan_only'.")

    dataset_role = _as_dict(raw, "dataset_role", errors)
    if dataset_role:
        for key in (
            "three_way_classification_planning",
            "tampered_localization_planning",
            "evidence_reason_alignment_planning",
        ):
            if dataset_role.get(key) is not True:
                errors.append(f"dataset_role.{key} must be true.")

    model_plan = _as_dict(raw, "model_plan", errors)
    if model_plan:
        for phrase in (
            "shared visual backbone",
            "3-way classification head",
            "generator-family provenance head",
            "conditional localization head",
        ):
            if not _contains_text(model_plan, (phrase,)):
                errors.append(f"model_plan must include {phrase}.")
        if not _contains_text(model_plan, ("evidence aggregation", "reason head")):
            errors.append("model_plan must include evidence aggregation or reason head.")

    head_plan = _as_dict(raw, "head_plan", errors)
    if head_plan:
        for key in ("classification_head", "provenance_head", "localization_head"):
            if key not in head_plan:
                errors.append(f"head_plan must include {key}.")
        if "evidence_aggregation" not in head_plan and "evidence_reason_head" not in head_plan:
            errors.append("head_plan must include evidence_aggregation or evidence_reason_head.")

    class_policy = _as_dict(raw, "classification_policy", errors)
    if class_policy:
        labels = class_policy.get("labels")
        if not isinstance(labels, list) or set(labels) != _CLASS_LABELS or len(labels) != 3:
            errors.append("classification_policy.labels must contain exactly real, synthetic, and tampered.")

    localization = _as_dict(raw, "localization_policy", errors)
    if localization:
        for phrase in ("conditional", "threshold_tau", "mask iou", "localization activation recall", "false negative"):
            if not _contains_text(localization, (phrase,)):
                errors.append(f"localization_policy must include {phrase}.")

    threshold_policy = _as_dict(raw, "threshold_policy", errors)
    tau = None
    if threshold_policy:
        tau = threshold_policy.get("threshold_tau")
    if tau is None and isinstance(localization, dict):
        tau = localization.get("threshold_tau")
    if not isinstance(tau, (int, float)) or isinstance(tau, bool):
        errors.append("threshold_tau must be a number between 0.0 and 1.0.")
    elif not 0.0 <= float(tau) <= 1.0:
        errors.append("threshold_tau must be between 0.0 and 1.0.")

    mask_policy = _as_dict(raw, "mask_policy", errors)
    if mask_policy:
        if not _contains_text(mask_policy, ("tampered", "mask", "localization")):
            errors.append("mask_policy must state tampered samples require mask/localization planning.")
        if not _contains_text(mask_policy, ("real", "must not")):
            errors.append("mask_policy must state real samples must not require tampered masks.")
        if mask_policy.get("read_real_mask_files") is not False:
            errors.append("mask_policy.read_real_mask_files must be false.")
        if mask_policy.get("read_real_image_files") is not False:
            errors.append("mask_policy.read_real_image_files must be false.")

    family_policy = _as_dict(raw, "family_policy", errors)
    if family_policy:
        if family_policy.get("missing_family_labels_allowed") is not True:
            errors.append("family_policy.missing_family_labels_allowed must be true.")
        mapping = family_policy.get("missing_label_mapping")
        if not isinstance(mapping, dict):
            errors.append("family_policy.missing_label_mapping must be an object.")
        else:
            if mapping.get("real") != "Real-or-N/A":
                errors.append("family_policy must map real samples to Real-or-N/A.")
            if mapping.get("synthetic") != "Unknown" or mapping.get("tampered") != "Unknown":
                errors.append("family_policy must map missing synthetic/tampered labels to Unknown.")
        if family_policy.get("provenance_training_gated_until_labels_confirmed") is not True:
            errors.append("family_policy must gate provenance training until labels are confirmed.")

    explanation = _as_dict(raw, "explanation_policy", errors)
    if explanation:
        if not _contains_text(explanation, ("deterministic", "template")):
            errors.append("explanation_policy must be deterministic and template-based.")
        if explanation.get("llm_call_required") is not False:
            errors.append("explanation_policy.llm_call_required must be false.")
        if explanation.get("free_form_hallucination_allowed") is not False:
            errors.append("explanation_policy.free_form_hallucination_allowed must be false.")

    loss_plan = _as_dict(raw, "loss_plan", errors)
    if loss_plan:
        for key in ("classification_loss", "optional_family_loss", "conditional_localization_loss"):
            if key not in loss_plan:
                errors.append(f"loss_plan must include {key}.")
        if "evidence_consistency_loss" not in loss_plan and "explanation_alignment_loss" not in loss_plan:
            errors.append("loss_plan must include evidence_consistency_loss or explanation_alignment_loss.")
        if loss_plan.get("mode") != "symbolic_plan_only":
            errors.append("loss_plan.mode must be symbolic_plan_only.")
        if loss_plan.get("executes_training") is not False:
            errors.append("loss_plan.executes_training must be false.")

    freeze = _as_dict(raw, "freeze_policy", errors)
    if freeze:
        if freeze.get("freeze_shared_backbone_initial_smoke_option") is not True:
            errors.append("freeze_policy must include shared-backbone freeze option.")
        if freeze.get("unfreeze_selected_layers_after_local_smoke_option") is not True:
            errors.append("freeze_policy must include later unfreeze option.")
        if freeze.get("execute_freezing_now") is not False:
            errors.append("freeze_policy.execute_freezing_now must be false.")

    batch_mixing = _as_dict(raw, "batch_mixing_policy", errors)
    if batch_mixing:
        if batch_mixing.get("sid_set_only_baseline_option") is not True:
            errors.append("batch_mixing_policy must include SID-Set-only baseline option.")
        if batch_mixing.get("optional_cf_small_auxiliary_batch_policy") is not True:
            errors.append("batch_mixing_policy must include optional CF-Small auxiliary policy.")
        if batch_mixing.get("execute_batch_loading_now") is not False:
            errors.append("batch_mixing_policy.execute_batch_loading_now must be false.")

    split = _as_dict(raw, "split_policy", errors)
    if split:
        if split.get("train_validation_test_separation") is not True:
            errors.append("split_policy.train_validation_test_separation must be true.")
        if split.get("avoid_leakage") is not True:
            errors.append("split_policy.avoid_leakage must be true.")
        classes = split.get("class_balance_required")
        if not isinstance(classes, list) or set(classes) != _CLASS_LABELS:
            errors.append("split_policy.class_balance_required must include real, synthetic, and tampered.")

    metric = _as_dict(raw, "metric_plan", errors)
    if metric:
        text = _metric_text(metric)
        for metric_name in _REQUIRED_METRICS:
            if metric_name not in text:
                errors.append(f"metric_plan must include {metric_name}.")

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

    handoff = _as_dict(raw, "handoff_to_pre_sns_baseline", errors)
    if handoff:
        if handoff.get("task_0016_collects_pre_sns_baseline_report") is not True:
            errors.append("handoff_to_pre_sns_baseline must point to task 0016 report scaffolding.")
        if handoff.get("sns_augmentation_waits_for_pre_sns_baseline") is not True:
            errors.append("handoff_to_pre_sns_baseline must gate SNS augmentation after pre-SNS baseline.")

    evaluation = _as_dict(raw, "evaluation_policy", errors)
    if evaluation and evaluation.get("run_evaluation") is not False:
        errors.append("evaluation_policy.run_evaluation must be false.")

    if errors:
        raise ValueError(
            "SID-Set baseline plan validation failed:\n"
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
        print(f"Usage: {sys.argv[0]} <sid_set_baseline_plan.json>", file=sys.stderr)
        sys.exit(1)

    try:
        raw = load_config(sys.argv[1])
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        "SID_SET_BASELINE_PLAN_OK: "
        f"{raw['dataset_name']} multi-head baseline plan is symbolic, plan-only, "
        "dry-run safe, localization-aware, and approval-gated."
    )


if __name__ == "__main__":
    main()
