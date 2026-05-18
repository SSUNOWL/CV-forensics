"""Tests for the CF-Small baseline plan validator.

Runnable directly:
    python3 tests/test_cf_small_baseline_plan.py

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

from validate_cf_small_baseline_plan import check_config_safety, load_config, validate_config


def _valid_raw() -> dict:
    return {
        "schema_version": "0.1.0",
        "plan_name": "cf_small_baseline_training_plan",
        "dataset_name": "Community Forensics-Small",
        "execution_mode": "plan_only",
        "dry_run": True,
        "no_download": True,
        "no_training": True,
        "no_network": True,
        "no_outputs": True,
        "no_checkpoints": True,
        "no_real_image_reading": True,
        "no_real_mask_reading": True,
        "requires_user_approval_for_real_training": True,
        "local_data_gate_ref": "configs/local_data/readiness.example.json",
        "cf_small_subset_smoke_ref": "configs/local_data/cf_small_subset_smoke.example.json",
        "dataset_role": {
            "shared_backbone_planning": True,
            "real_vs_synthetic_baseline_planning": True,
            "coarse_provenance_family_planning": True,
            "not_primary_tampered_localization_source": True,
        },
        "training_objective_plan": {
            "mode": "planning_only",
            "primary_objective": "real-vs-synthetic baseline classification planning",
            "secondary_objective": "coarse generator-family provenance planning",
            "no_real_optimization": True,
        },
        "backbone_plan": {"purpose": "shared visual backbone initialization planning"},
        "classification_policy": {
            "labels": ["real", "synthetic"],
            "task": "binary real-vs-synthetic classification",
            "tampered_localization_required": False,
        },
        "family_policy": {
            "type": "coarse_generator_family_provenance",
            "labels": ["LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A", "Unknown"],
        },
        "generator_holdout_policy": {
            "strategy": "model_name_or_generator_family_holdout",
            "avoid_train_validation_leakage": True,
            "diversity_aware_split_planning": True,
            "random_only_split_allowed": False,
            "holdout_keys": ["model_name", "architecture"],
        },
        "split_policy": {
            "mode": "symbolic_plan_only",
            "train_split": "planned",
            "validation_split": "planned",
            "generator_holdout_required": True,
        },
        "evaluation_policy": {"mode": "plan_only", "run_evaluation": False},
        "metric_plan": {
            "required_metrics": [
                "binary accuracy",
                "macro F1",
                "generator-family accuracy",
                "holdout/generalization evaluation",
            ],
            "later_metrics": ["latency", "FPS"],
            "real_mask_iou_required_for_cf_small": False,
        },
        "output_policy": {
            "write_outputs": False,
            "requires_explicit_approval_before_writing": True,
        },
        "checkpoint_policy": {
            "write_checkpoints": False,
            "requires_explicit_approval_before_writing": True,
        },
        "handoff_to_sid_set": {
            "sid_set_handles_3_way_classification": True,
            "sid_set_handles_tampered_localization": True,
        },
        "validation_notes": ["symbolic plan only"],
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
    load_config(str(_REPO_ROOT / "configs" / "training" / "cf_small_baseline.example.json"))


def test_valid_raw_passes():
    raw = _valid_raw()
    check_config_safety(raw)
    validate_config(raw)


def test_execution_mode_other_than_plan_only_rejected():
    raw = _valid_raw()
    raw["execution_mode"] = "train"
    _assert_rejected(raw, "execution_mode")


def test_guardrail_flags_rejected():
    for flag in (
        "dry_run",
        "no_download",
        "no_training",
        "no_outputs",
        "no_checkpoints",
        "no_real_image_reading",
        "no_real_mask_reading",
        "requires_user_approval_for_real_training",
    ):
        raw = _valid_raw()
        raw[flag] = False
        _assert_rejected(raw, flag)


def test_wrong_dataset_name_rejected():
    raw = _valid_raw()
    raw["dataset_name"] = "SID-Set"
    _assert_rejected(raw, "dataset_name")


def test_missing_real_class_rejected():
    raw = _valid_raw()
    raw["classification_policy"]["labels"] = ["synthetic"]
    _assert_rejected(raw, "real")


def test_missing_synthetic_class_rejected():
    raw = _valid_raw()
    raw["classification_policy"]["labels"] = ["real"]
    _assert_rejected(raw, "synthetic")


def test_tampered_localization_requirement_rejected():
    raw = _valid_raw()
    raw["classification_policy"]["tampered_localization_required"] = True
    _assert_rejected(raw, "tampered_localization_required")


def test_missing_family_policy_rejected():
    raw = _valid_raw()
    del raw["family_policy"]
    _assert_rejected(raw, "family_policy")


def test_missing_generator_holdout_policy_rejected():
    raw = _valid_raw()
    del raw["generator_holdout_policy"]
    _assert_rejected(raw, "generator_holdout_policy")


def test_missing_leakage_prevention_policy_rejected():
    raw = _valid_raw()
    raw["generator_holdout_policy"]["avoid_train_validation_leakage"] = False
    _assert_rejected(raw, "avoid_train_validation_leakage")


def test_missing_split_policy_rejected():
    raw = _valid_raw()
    del raw["split_policy"]
    _assert_rejected(raw, "split_policy")


def test_missing_binary_accuracy_metric_rejected():
    raw = _valid_raw()
    raw["metric_plan"]["required_metrics"].remove("binary accuracy")
    _assert_rejected(raw, "binary accuracy")


def test_missing_macro_f1_metric_rejected():
    raw = _valid_raw()
    raw["metric_plan"]["required_metrics"].remove("macro F1")
    _assert_rejected(raw, "macro f1")


def test_missing_generator_family_accuracy_metric_rejected():
    raw = _valid_raw()
    raw["metric_plan"]["required_metrics"].remove("generator-family accuracy")
    _assert_rejected(raw, "generator-family accuracy")


def test_real_mask_iou_required_rejected():
    raw = _valid_raw()
    raw["metric_plan"]["real_mask_iou_required_for_cf_small"] = True
    _assert_rejected(raw, "real_mask_iou_required_for_cf_small")


def test_output_writing_without_approval_rejected():
    raw = _valid_raw()
    raw["output_policy"]["write_outputs"] = True
    _assert_rejected(raw, "write_outputs")


def test_checkpoint_writing_without_approval_rejected():
    raw = _valid_raw()
    raw["checkpoint_policy"]["write_checkpoints"] = True
    _assert_rejected(raw, "write_checkpoints")


def test_protected_paths_rejected():
    cases = [
        {"path": "data/images"},
        {"path": "datasets/cf"},
        {"path": "outputs/run"},
        {"path": "checkpoints/run"},
        {"path": "secrets/key"},
        {"path": ".env"},
        {"path": ".env.local"},
    ]
    for bad in cases:
        try:
            check_config_safety(bad)
            assert False, f"Expected protected path rejection for {bad}"
        except ValueError:
            pass


def test_urls_rejected():
    for bad in (
        {"url": "http://example.test"},
        {"url": "https://example.test"},
        {"url": "s3://bucket/key"},
        {"url": "gs://bucket/key"},
        {"url": "hf://org/repo"},
    ):
        try:
            check_config_safety(bad)
            assert False, f"Expected URL rejection for {bad}"
        except ValueError:
            pass


def test_absolute_paths_rejected():
    for bad in ({"path": "/home/user/cf"}, {"path": "/mnt/cf"}, {"path": "/root/cf"}, {"path": "/Users/name/cf"}):
        try:
            check_config_safety(bad)
            assert False, f"Expected absolute path rejection for {bad}"
        except ValueError:
            pass


def test_windows_drive_paths_rejected():
    try:
        check_config_safety({"path": "C:\\cf\\images"})
        assert False, "Expected Windows drive path rejection"
    except ValueError:
        pass


def test_secret_like_keys_and_values_rejected():
    cases = [
        {"token": "abc"},
        {"api_key": "abc"},
        {"password": "abc"},
        {"secret": "abc"},
        {"credential": "abc"},
        {"bearer": "abc"},
        {"note": "token abc"},
        {"note": "api key abc"},
        {"auth": "abc"},
        {"note": "auth"},
    ]
    for bad in cases:
        try:
            check_config_safety(bad)
            assert False, f"Expected secret-like rejection for {bad}"
        except ValueError:
            pass


def test_auth_substring_prose_allowed():
    check_config_safety({"note": "authoritative guidance"})
    check_config_safety({"note": "authentication policy"})


def test_validator_does_not_write_files():
    if _REPO_ROOT is None:
        return
    before = set(os.listdir(str(_REPO_ROOT)))
    raw = _valid_raw()
    check_config_safety(raw)
    validate_config(raw)
    after = set(os.listdir(str(_REPO_ROOT)))
    assert before == after, f"Unexpected files written: {after - before}"


if __name__ == "__main__":
    failures = []
    passed = 0
    tests = {name: fn for name, fn in globals().items() if name.startswith("test_")}
    for name in sorted(tests):
        try:
            tests[name]()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as exc:
            import traceback
            print(f"  FAIL: {name}: {exc}")
            traceback.print_exc()
            failures.append(name)
    total = passed + len(failures)
    print(f"\n{passed}/{total} tests passed.")
    if failures:
        print(f"Failed: {failures}")
        _sys.exit(1)
    print("All tests passed.")
