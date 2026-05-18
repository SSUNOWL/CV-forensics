"""Tests for the SID-Set local subset smoke validator.

Runnable directly:
    python3 tests/test_sid_set_subset_smoke.py

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

from validate_sid_set_subset_smoke import check_config_safety, load_config, validate_config


def _valid_raw() -> dict:
    return {
        "schema_version": "0.1.0",
        "dataset_name": "SID-Set",
        "dry_run": True,
        "no_download": True,
        "no_training": True,
        "no_network": True,
        "no_outputs": True,
        "no_checkpoints": True,
        "no_real_image_reading": True,
        "no_real_mask_reading": True,
        "local_data_gate_ref": "configs/local_data/readiness.example.json",
        "readiness_policy": {
            "mode": "symbolic_until_explicit_user_approval",
            "local_paths_approved": False,
            "requires_validated_manifest": True,
            "requires_user_approval": True,
        },
        "subset_policy": {
            "mode": "tiny_symbolic_class_balanced_plan",
            "planned_samples": {"real": 4, "synthetic": 4, "tampered": 4},
            "max_total_manifest_rows": 12,
            "real_file_access": "forbidden",
            "recursive_directory_scan": "forbidden",
        },
        "split_policy": {"mode": "symbolic_manifest_rows_only"},
        "class_policy": {
            "labels": ["real", "synthetic", "tampered"],
            "required_exact_label_set": True,
        },
        "mask_policy": {
            "tampered": "requires localization planning",
            "real": "must not require tampered masks",
            "synthetic": "may omit tampered masks unless explicitly annotated",
            "read_real_mask_files": False,
            "read_real_image_files": False,
        },
        "family_policy": {
            "sid_set_family_labels_may_be_missing": True,
            "missing_label_policy": {
                "real": "Real-or-N/A",
                "synthetic": "Unknown",
                "tampered": "Unknown",
            },
            "provenance_training_gated_until_labels_confirmed": True,
        },
        "localization_policy": {
            "conditional_activation": "activate localization when tampered_score >= threshold_tau",
            "threshold_tau_ref": "threshold_tau",
            "primary_metric": "mask IoU",
            "recall_metric": "localization activation recall",
        },
        "threshold_tau": 0.35,
        "expected_task_outputs": ["class", "mask", "family", "reason"],
        "validation_notes": ["symbolic planning only"],
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
    load_config(str(_REPO_ROOT / "configs" / "local_data" / "sid_set_subset_smoke.example.json"))


def test_valid_raw_passes():
    raw = _valid_raw()
    check_config_safety(raw)
    validate_config(raw)


def test_guardrail_flags_rejected():
    for flag in (
        "dry_run",
        "no_download",
        "no_training",
        "no_outputs",
        "no_checkpoints",
        "no_real_image_reading",
        "no_real_mask_reading",
    ):
        raw = _valid_raw()
        raw[flag] = False
        _assert_rejected(raw, flag)


def test_missing_required_class_label_rejected():
    for missing in ("real", "synthetic", "tampered"):
        raw = _valid_raw()
        raw["class_policy"]["labels"] = [x for x in ["real", "synthetic", "tampered"] if x != missing]
        _assert_rejected(raw, "class_policy.labels")


def test_extra_class_label_rejected():
    raw = _valid_raw()
    raw["class_policy"]["labels"] = ["real", "synthetic", "tampered", "other"]
    _assert_rejected(raw, "class_policy.labels")


def test_missing_mask_policy_rejected():
    raw = _valid_raw()
    del raw["mask_policy"]
    _assert_rejected(raw, "mask_policy")


def test_missing_family_policy_rejected():
    raw = _valid_raw()
    del raw["family_policy"]
    _assert_rejected(raw, "family_policy")


def test_missing_localization_activation_recall_rejected():
    raw = _valid_raw()
    raw["localization_policy"]["recall_metric"] = "tampered recall"
    _assert_rejected(raw, "localization activation recall")


def test_missing_mask_iou_rejected():
    raw = _valid_raw()
    raw["localization_policy"]["primary_metric"] = "overlap score"
    _assert_rejected(raw, "mask IoU")


def test_threshold_tau_range_rejected():
    for bad_tau in (-0.1, 1.1):
        raw = _valid_raw()
        raw["threshold_tau"] = bad_tau
        _assert_rejected(raw, "threshold_tau")


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
    for bad in ({"path": "/home/user/sid"}, {"path": "/mnt/sid"}, {"path": "/root/sid"}, {"path": "/Users/name/sid"}):
        try:
            check_config_safety(bad)
            assert False, f"Expected absolute path rejection for {bad}"
        except ValueError:
            pass


def test_windows_drive_paths_rejected():
    try:
        check_config_safety({"path": "C:\\sid\\images"})
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
