"""Tests for the SID-Set multi-head baseline plan validator.

Runnable directly:
    python3 tests/test_sid_set_baseline_plan.py

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

from validate_sid_set_baseline_plan import check_config_safety, load_config, validate_config


def _valid_raw() -> dict:
    return {
        "schema_version": "0.1.0",
        "plan_name": "sid_set_multihead_baseline_plan",
        "dataset_name": "SID-Set",
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
        "sid_set_subset_smoke_ref": "configs/local_data/sid_set_subset_smoke.example.json",
        "cf_small_baseline_ref": "configs/training/cf_small_baseline.example.json",
        "dataset_role": {
            "three_way_classification_planning": True,
            "tampered_localization_planning": True,
            "evidence_reason_alignment_planning": True,
        },
        "model_plan": {
            "shared_visual_backbone": "CF-Small initialized shared visual backbone",
            "classification_head": "3-way classification head",
            "provenance_head": "generator-family provenance head",
            "localization_head": "conditional localization head",
            "evidence_aggregation": "evidence aggregation reason head",
        },
        "head_plan": {
            "classification_head": "planned",
            "provenance_head": "planned",
            "localization_head": "planned",
            "evidence_aggregation": "planned",
        },
        "classification_policy": {"labels": ["real", "synthetic", "tampered"]},
        "localization_policy": {
            "conditional_activation": "tampered_score >= threshold_tau activates localization",
            "threshold_tau": 0.35,
            "mask_iou_metric": "mask IoU",
            "activation_recall_metric": "localization activation recall",
            "false_negative_risk": "high threshold creates false negative risk by skipping tampered samples",
        },
        "mask_policy": {
            "tampered": "requires mask/localization planning",
            "real": "must not require tampered masks",
            "synthetic": "may omit tampered masks unless explicitly annotated",
            "read_real_mask_files": False,
            "read_real_image_files": False,
        },
        "family_policy": {
            "missing_family_labels_allowed": True,
            "missing_label_mapping": {
                "real": "Real-or-N/A",
                "synthetic": "Unknown",
                "tampered": "Unknown",
            },
            "provenance_training_gated_until_labels_confirmed": True,
        },
        "explanation_policy": {
            "mode": "deterministic_template_based",
            "reason_source": "evidence signals and output schema",
            "free_form_hallucination_allowed": False,
            "llm_call_required": False,
        },
        "threshold_policy": {"threshold_tau": 0.35},
        "loss_plan": {
            "mode": "symbolic_plan_only",
            "classification_loss": "planned",
            "optional_family_loss": "planned",
            "conditional_localization_loss": "planned",
            "evidence_consistency_loss": "planned",
            "executes_training": False,
        },
        "freeze_policy": {
            "freeze_shared_backbone_initial_smoke_option": True,
            "unfreeze_selected_layers_after_local_smoke_option": True,
            "execute_freezing_now": False,
        },
        "batch_mixing_policy": {
            "sid_set_only_baseline_option": True,
            "optional_cf_small_auxiliary_batch_policy": True,
            "execute_batch_loading_now": False,
        },
        "split_policy": {
            "train_validation_test_separation": True,
            "avoid_leakage": True,
            "class_balance_required": ["real", "synthetic", "tampered"],
        },
        "evaluation_policy": {"mode": "plan_only", "run_evaluation": False},
        "metric_plan": {
            "required_metrics": [
                "3-way accuracy",
                "Macro-F1",
                "mask IoU",
                "generator-family accuracy",
                "localization activation recall",
                "latency",
                "FPS",
            ]
        },
        "output_policy": {
            "write_outputs": False,
            "requires_explicit_approval_before_writing": True,
        },
        "checkpoint_policy": {
            "write_checkpoints": False,
            "requires_explicit_approval_before_writing": True,
        },
        "handoff_to_pre_sns_baseline": {
            "task_0016_collects_pre_sns_baseline_report": True,
            "sns_augmentation_waits_for_pre_sns_baseline": True,
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
    load_config(str(_REPO_ROOT / "configs" / "training" / "sid_set_multihead_baseline.example.json"))


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
    raw["dataset_name"] = "Community Forensics-Small"
    _assert_rejected(raw, "dataset_name")


def test_class_labels_rejected_when_missing_or_extra():
    for label in ("real", "synthetic", "tampered"):
        raw = _valid_raw()
        raw["classification_policy"]["labels"] = [x for x in ["real", "synthetic", "tampered"] if x != label]
        _assert_rejected(raw, "classification_policy.labels")
    raw = _valid_raw()
    raw["classification_policy"]["labels"] = ["real", "synthetic", "tampered", "other"]
    _assert_rejected(raw, "classification_policy.labels")


def test_model_plan_required_parts_rejected():
    checks = [
        ("shared_visual_backbone", "shared visual backbone"),
        ("classification_head", "3-way classification head"),
        ("provenance_head", "generator-family provenance head"),
        ("localization_head", "conditional localization head"),
        ("evidence_aggregation", "evidence aggregation"),
    ]
    for key, expected in checks:
        raw = _valid_raw()
        del raw["model_plan"][key]
        _assert_rejected(raw, expected)


def test_head_plan_required_parts_rejected():
    for key in ("classification_head", "provenance_head", "localization_head", "evidence_aggregation"):
        raw = _valid_raw()
        del raw["head_plan"][key]
        expected = key if key != "evidence_aggregation" else "evidence_aggregation"
        _assert_rejected(raw, expected)


def test_localization_policy_required_parts_rejected():
    cases = [
        ("conditional_activation", "conditional"),
        ("mask_iou_metric", "mask iou"),
        ("activation_recall_metric", "localization activation recall"),
        ("false_negative_risk", "false negative"),
    ]
    for key, expected in cases:
        raw = _valid_raw()
        del raw["localization_policy"][key]
        _assert_rejected(raw, expected)


def test_threshold_tau_range_rejected():
    for tau in (-0.1, 1.1):
        raw = _valid_raw()
        raw["threshold_policy"]["threshold_tau"] = tau
        raw["localization_policy"]["threshold_tau"] = tau
        _assert_rejected(raw, "threshold_tau")


def test_family_policy_rejected_when_incomplete():
    raw = _valid_raw()
    raw["family_policy"]["missing_family_labels_allowed"] = False
    _assert_rejected(raw, "missing_family_labels_allowed")
    raw = _valid_raw()
    raw["family_policy"]["missing_label_mapping"]["real"] = "Unknown"
    _assert_rejected(raw, "Real-or-N/A")


def test_explanation_policy_rejected_when_not_deterministic_or_llm_required():
    raw = _valid_raw()
    raw["explanation_policy"]["mode"] = "free_form"
    _assert_rejected(raw, "deterministic")
    raw = _valid_raw()
    raw["explanation_policy"]["llm_call_required"] = True
    _assert_rejected(raw, "llm_call_required")


def test_missing_freeze_batch_mixing_and_leakage_policy_rejected():
    raw = _valid_raw()
    del raw["freeze_policy"]
    _assert_rejected(raw, "freeze_policy")
    raw = _valid_raw()
    del raw["batch_mixing_policy"]
    _assert_rejected(raw, "batch_mixing_policy")
    raw = _valid_raw()
    raw["split_policy"]["avoid_leakage"] = False
    _assert_rejected(raw, "avoid_leakage")


def test_output_and_checkpoint_writing_rejected():
    raw = _valid_raw()
    raw["output_policy"]["write_outputs"] = True
    _assert_rejected(raw, "write_outputs")
    raw = _valid_raw()
    raw["checkpoint_policy"]["write_checkpoints"] = True
    _assert_rejected(raw, "write_checkpoints")


def test_protected_paths_rejected():
    cases = [
        {"path": "data/images"},
        {"path": "datasets/sid"},
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


def test_urls_and_absolute_paths_rejected():
    cases = [
        {"url": "http://example.test"},
        {"url": "https://example.test"},
        {"url": "s3://bucket/key"},
        {"url": "gs://bucket/key"},
        {"url": "hf://org/repo"},
        {"path": "/home/user/sid"},
        {"path": "/mnt/sid"},
        {"path": "/root/sid"},
        {"path": "/Users/name/sid"},
        {"path": "C:\\sid\\images"},
    ]
    for bad in cases:
        try:
            check_config_safety(bad)
            assert False, f"Expected safety rejection for {bad}"
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
