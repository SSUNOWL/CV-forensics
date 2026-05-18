"""Tests for the pre-SNS baseline report scaffold validator.

Runnable directly:
    python3 tests/test_pre_sns_baseline_report.py

Uses only the Python standard library and does not import pytest.
"""
from __future__ import annotations

from pathlib import Path as _Path
import os
import sys as _sys


_REPO_ROOT = next(
    (p for p in _Path(__file__).resolve().parents if (p / "scripts" / "agent").is_dir()),
    None,
)
if _REPO_ROOT is not None:
    _SCRIPT_ROOT = str(_REPO_ROOT / "scripts" / "agent")
    if _SCRIPT_ROOT not in _sys.path:
        _sys.path.insert(0, _SCRIPT_ROOT)

from validate_pre_sns_baseline_report import check_config_safety, load_config, validate_config


def _metric(display_name: str, source_stage: str) -> dict:
    return {
        "display_name": display_name,
        "status": "pending_real_baseline_run",
        "value": None,
        "unit_or_definition": f"placeholder definition for {display_name}",
        "source_stage": source_stage,
        "required_before_sns": True,
    }


def _valid_raw() -> dict:
    return {
        "schema_version": "0.1.0",
        "report_name": "pre_sns_baseline_evaluation_report_scaffold",
        "report_stage": "pre_sns_baseline",
        "execution_mode": "report_scaffold_only",
        "dry_run": True,
        "no_download": True,
        "no_training": True,
        "no_network": True,
        "no_outputs": True,
        "no_checkpoints": True,
        "no_real_evaluation": True,
        "no_real_image_reading": True,
        "no_real_mask_reading": True,
        "no_sns_augmentation": True,
        "no_sns_perturbation_eval": True,
        "requires_user_approval_for_real_evaluation": True,
        "local_data_gate_ref": "configs/local_data/readiness.example.json",
        "cf_small_baseline_ref": "configs/training/cf_small_baseline.example.json",
        "sid_set_baseline_ref": "configs/training/sid_set_multihead_baseline.example.json",
        "report_scope": {
            "purpose": "pre-SNS baseline comparison anchor",
            "claims_real_metric_values": False,
        },
        "dataset_scope": {
            "community_forensics_small_baseline_role": "Community Forensics-Small shared backbone role",
            "sid_set_baseline_role": "SID-Set 3-way classification role",
            "real_data_access_in_this_task": False,
        },
        "model_output_scope": {
            "class": True,
            "mask_localization": True,
            "family_provenance": True,
            "reason": True,
        },
        "metric_fields": {
            "three_way_accuracy": _metric("3-way accuracy", "SID-Set baseline"),
            "macro_f1": _metric("Macro-F1", "SID-Set baseline"),
            "mask_iou": _metric("mask IoU", "SID-Set localization baseline"),
            "generator_family_accuracy": _metric("generator-family accuracy", "CF-Small baseline"),
            "localization_activation_recall": _metric("localization activation recall", "SID-Set localization baseline"),
            "latency": _metric("latency", "baseline timing"),
            "fps": _metric("FPS", "baseline timing"),
        },
        "metric_status_policy": {
            "current_metric_status": "pending_real_baseline_run",
            "current_metric_values_must_be_null": True,
            "completed_real_metric_values_rejected_in_this_task": True,
        },
        "baseline_collection_policy": {
            "requires_explicit_user_approval_before_real_metric_collection": True,
            "run_real_evaluation_now": False,
            "validated_manifest_required_before_collection": True,
        },
        "comparison_policy": {
            "pre_sns_baseline_anchor": {"status": "scaffold_ready_after_validation"},
            "future_sns_robustness_drop": {"status": "pending_future_sns_stage", "value": None},
            "future_sns_augmented_training_comparison": {"status": "pending_future_sns_stage", "value": None},
        },
        "output_policy": {
            "write_reports": False,
            "write_predictions": False,
            "requires_explicit_approval_before_external_report_or_prediction_writing": True,
        },
        "checkpoint_policy": {
            "write_checkpoints": False,
            "requires_explicit_approval_before_writing": True,
        },
        "sns_future_stage_policy": {
            "sns_augmentation_implemented_here": False,
            "sns_perturbation_evaluation_run_here": False,
            "sns_stage_tasks_begin_only_after_scaffold_committed_and_explicitly_approved": True,
            "notes": [
                "SNS augmentation is not implemented in this task.",
                "SNS perturbation evaluation is not run in this task.",
            ],
        },
        "validation_notes": ["symbolic report scaffold only"],
    }


def _assert_rejected(raw: dict, expected: str = "") -> None:
    try:
        check_config_safety(raw)
        validate_config(raw)
        assert False, f"Expected rejection for {expected or raw!r}"
    except ValueError as exc:
        if expected:
            assert expected in str(exc), f"Expected {expected!r} in {exc!r}"


def test_valid_example_config_passes():
    if _REPO_ROOT is None:
        return
    load_config(str(_REPO_ROOT / "configs" / "reports" / "pre_sns_baseline_report.example.json"))


def test_valid_raw_passes():
    raw = _valid_raw()
    check_config_safety(raw)
    validate_config(raw)


def test_execution_mode_and_report_stage_rejected():
    raw = _valid_raw()
    raw["execution_mode"] = "evaluate"
    _assert_rejected(raw, "execution_mode")
    raw = _valid_raw()
    raw["report_stage"] = "post_sns"
    _assert_rejected(raw, "report_stage")


def test_guardrail_flags_rejected():
    for flag in (
        "dry_run",
        "no_download",
        "no_training",
        "no_outputs",
        "no_checkpoints",
        "no_real_evaluation",
        "no_real_image_reading",
        "no_real_mask_reading",
        "no_sns_augmentation",
        "no_sns_perturbation_eval",
        "requires_user_approval_for_real_evaluation",
    ):
        raw = _valid_raw()
        raw[flag] = False
        _assert_rejected(raw, flag)


def test_model_output_scope_rejected_when_missing():
    for key in ("class", "mask_localization", "family_provenance", "reason"):
        raw = _valid_raw()
        del raw["model_output_scope"][key]
        _assert_rejected(raw, "model_output_scope")


def test_required_metrics_rejected_when_missing():
    for key in (
        "three_way_accuracy",
        "macro_f1",
        "mask_iou",
        "generator_family_accuracy",
        "localization_activation_recall",
        "latency",
        "fps",
    ):
        raw = _valid_raw()
        del raw["metric_fields"][key]
        _assert_rejected(raw, key)


def test_metric_entry_status_value_and_required_flag_rejected():
    raw = _valid_raw()
    raw["metric_fields"]["macro_f1"]["value"] = 0.91
    _assert_rejected(raw, "value must be null")
    raw = _valid_raw()
    raw["metric_fields"]["macro_f1"]["status"] = "complete"
    _assert_rejected(raw, "status")
    raw = _valid_raw()
    raw["metric_fields"]["macro_f1"]["required_before_sns"] = False
    _assert_rejected(raw, "required_before_sns")


def test_comparison_policy_rejected_when_incomplete():
    raw = _valid_raw()
    del raw["comparison_policy"]["future_sns_robustness_drop"]
    _assert_rejected(raw, "future_sns_robustness_drop")
    raw = _valid_raw()
    raw["comparison_policy"]["future_sns_robustness_drop"]["status"] = "complete"
    _assert_rejected(raw, "pending_future_sns_stage")


def test_baseline_collection_without_approval_gate_rejected():
    raw = _valid_raw()
    raw["baseline_collection_policy"]["requires_explicit_user_approval_before_real_metric_collection"] = False
    _assert_rejected(raw, "baseline_collection_policy")
    raw = _valid_raw()
    raw["baseline_collection_policy"]["run_real_evaluation_now"] = True
    _assert_rejected(raw, "run_real_evaluation_now")


def test_output_and_checkpoint_writing_rejected():
    raw = _valid_raw()
    raw["output_policy"]["write_reports"] = True
    _assert_rejected(raw, "write_reports")
    raw = _valid_raw()
    raw["output_policy"]["write_predictions"] = True
    _assert_rejected(raw, "write_predictions")
    raw = _valid_raw()
    raw["output_policy"]["requires_explicit_approval_before_external_report_or_prediction_writing"] = False
    _assert_rejected(raw, "output_policy")
    raw = _valid_raw()
    raw["checkpoint_policy"]["write_checkpoints"] = True
    _assert_rejected(raw, "write_checkpoints")
    raw = _valid_raw()
    raw["checkpoint_policy"]["requires_explicit_approval_before_writing"] = False
    _assert_rejected(raw, "checkpoint_policy")


def test_sns_enabled_or_perturbation_eval_enabled_rejected():
    raw = _valid_raw()
    raw["sns_future_stage_policy"]["sns_augmentation_implemented_here"] = True
    _assert_rejected(raw, "sns_augmentation_implemented_here")
    raw = _valid_raw()
    raw["sns_future_stage_policy"]["sns_perturbation_evaluation_run_here"] = True
    _assert_rejected(raw, "sns_perturbation_evaluation_run_here")


def test_protected_paths_rejected():
    for bad in ("data/raw", "datasets/local", "outputs/report", "checkpoints/model", "secrets/key", ".env"):
        raw = _valid_raw()
        raw["validation_notes"] = [bad]
        _assert_rejected(raw)


def test_urls_and_absolute_paths_rejected():
    for bad in ("https://example.invalid/file", "/home/user/file", "/mnt/storage/file", "C:\\sid\\file"):
        raw = _valid_raw()
        raw["validation_notes"] = [bad]
        _assert_rejected(raw)


def test_secret_like_keys_and_values_rejected():
    raw = _valid_raw()
    raw["api_key"] = "placeholder"
    _assert_rejected(raw)
    raw = _valid_raw()
    raw["validation_notes"] = ["bearer value"]
    _assert_rejected(raw)


def test_auth_substring_prose_allowed():
    raw = _valid_raw()
    raw["validation_notes"] = [
        "authoritative report policy",
        "authentication policy is documented as prose only",
    ]
    check_config_safety(raw)
    validate_config(raw)


def test_validator_does_not_write_files():
    raw = _valid_raw()
    if _REPO_ROOT is not None:
        watched = [
            _REPO_ROOT / "configs" / "reports" / "pre_sns_baseline_report.example.json",
            _REPO_ROOT / "docs" / "pre_sns_baseline_report.md",
        ]
        before = {str(path): os.stat(path).st_mtime_ns for path in watched if path.exists()}
    else:
        before = {}
    check_config_safety(raw)
    validate_config(raw)
    after = {str(path): os.stat(path).st_mtime_ns for path in watched if path.exists()} if _REPO_ROOT is not None else {}
    assert before == after


if __name__ == "__main__":
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for name, test in tests:
        try:
            test()
            print(f"  PASS: {name}")
        except Exception as exc:
            failures.append(name)
            print(f"  FAIL: {name}: {exc}")
            import traceback

            traceback.print_exc()
    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed.")
    if failures:
        print(f"Failed: {failures}")
        raise SystemExit(1)
    print("All tests passed.")
