"""Tests for the CF-Small local subset smoke validator.

Runnable directly:
    python3 tests/test_cf_small_subset_smoke.py

Uses only the Python standard library and does not import pytest.
"""
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

_REPO_ROOT = next(
    (p for p in _Path(__file__).resolve().parents if (p / "scripts" / "agent").is_dir()),
    None,
)
if _REPO_ROOT is not None:
    _SCRIPT_ROOT = str(_REPO_ROOT / "scripts" / "agent")
    if _SCRIPT_ROOT not in _sys.path:
        _sys.path.insert(0, _SCRIPT_ROOT)

import copy
import os

from validate_cf_small_subset_smoke import check_config_safety, load_config, validate_config


def _valid_raw() -> dict:
    return {
        "schema_version": "0.1.0",
        "dry_run": True,
        "no_download": True,
        "no_training": True,
        "no_network": True,
        "no_outputs": True,
        "no_checkpoints": True,
        "dataset": "Community Forensics-Small",
        "dataset_alias": "CF-Small",
        "local_path_gate": {
            "policy": "symbolic_until_explicit_user_approval",
            "requires_user_approved_local_paths": True,
            "real_local_paths_approved": False,
            "path_mode": "symbolic_only",
            "note": "Explicit user approval is required before local paths are used.",
        },
        "manifest_ref": {
            "readiness_gate": "configs/local_data/readiness.example.json",
            "cf_small_manifest": "configs/manifests/community_forensics_small.example.json",
        },
        "subset_plan": {
            "mode": "tiny_symbolic_manifest_rows_only",
            "planned_splits": ["train_smoke", "validation_smoke"],
            "file_enumeration": "forbidden",
            "image_reading": "forbidden",
            "mask_reading": "forbidden",
            "recursive_directory_scan": "forbidden",
        },
        "sample_limits": {
            "max_total_manifest_rows": 32,
            "max_train_manifest_rows": 24,
            "max_validation_manifest_rows": 8,
        },
        "required_metadata": ["architecture", "model_name", "subset"],
        "holdout_plan": {
            "strategy": "generator_or_model_name_holdout",
            "primary_holdout_key": "model_name",
            "secondary_holdout_key": "architecture",
            "random_only_validation_allowed": False,
            "reason": "Random-only validation is not sufficient for unseen generator generalization.",
        },
        "validation_plan": {
            "scope": "planning_only",
            "checks": ["validate structure"],
            "read_real_images": False,
            "read_real_masks": False,
            "run_training": False,
            "write_prediction_artifacts": False,
            "write_model_artifacts": False,
        },
        "approval": {
            "local_data_approved": False,
            "approved_by": None,
            "approval_date": None,
            "next_step_policy": "After explicit approval, validate readiness and the CF-Small manifest.",
        },
        "forbidden_actions": [
            "dataset download",
            "model training",
            "network access",
            "real image reading",
            "real mask reading",
            "checkpoint artifact writing",
            "SNS augmentation",
        ],
        "protected_path_exclusions": [".env", ".env.*", "secrets", "data", "datasets", "outputs", "checkpoints"],
    }


def _assert_rejected(raw: dict, expected: str = "") -> None:
    try:
        check_config_safety(raw)
        validate_config(raw)
        assert False, f"Expected rejection for {expected or raw!r}"
    except ValueError as exc:
        if expected:
            assert expected in str(exc), f"Expected {expected!r} in {exc!r}"


def test_valid_symbolic_smoke_config_passes():
    raw = _valid_raw()
    check_config_safety(raw)
    validate_config(raw)


def test_example_file_passes():
    if _REPO_ROOT is None:
        return
    path = str(_REPO_ROOT / "configs" / "local_data" / "cf_small_subset_smoke.example.json")
    load_config(path)


def test_guardrail_flags_rejected():
    for flag in ("dry_run", "no_download", "no_training", "no_outputs", "no_checkpoints"):
        raw = _valid_raw()
        raw[flag] = False
        _assert_rejected(raw, flag)


def test_protected_paths_rejected():
    cases = [
        {"path": "/home/user/cf-small"},
        {"path": "/mnt/local/cf-small"},
        {"path": "/root/cf-small"},
        {"path": "/Users/user/cf-small"},
        {"path": "data/images"},
        {"path": "datasets/cf-small"},
        {"path": "outputs/smoke"},
        {"path": "checkpoints/smoke"},
        {"path": "secrets/key"},
        {"path": ".env"},
        {"path": ".env.local"},
        {"path": "C:\\cf-small\\images"},
    ]
    for bad in cases:
        try:
            check_config_safety(bad)
            assert False, f"Expected safety rejection for {bad}"
        except ValueError:
            pass


def test_urls_rejected():
    for bad in (
        {"url": "http://example.test/file"},
        {"url": "https://example.test/file"},
        {"url": "s3://bucket/key"},
        {"url": "gs://bucket/key"},
        {"url": "hf://org/repo"},
    ):
        try:
            check_config_safety(bad)
            assert False, f"Expected URL rejection for {bad}"
        except ValueError:
            pass


def test_secret_like_keys_or_values_rejected():
    cases = [
        {"token": "abc"},
        {"api_key": "abc"},
        {"password": "abc"},
        {"secret": "abc"},
        {"credential": "abc"},
        {"bearer": "abc"},
        {"note": "token abc"},
        {"note": "api key abc"},
    ]
    for bad in cases:
        try:
            check_config_safety(bad)
            assert False, f"Expected secret rejection for {bad}"
        except ValueError:
            pass


def test_auth_token_rejection_and_safe_prose_allowance():
    for bad in ({"auth": "value"}, {"note": "auth"}):
        try:
            check_config_safety(bad)
            assert False, f"Expected standalone auth rejection for {bad}"
        except ValueError:
            pass
    for safe in ({"note": "authoritative guidance"}, {"note": "authentication policy"}):
        check_config_safety(safe)


def test_missing_local_path_gate_rejected():
    raw = _valid_raw()
    del raw["local_path_gate"]
    _assert_rejected(raw, "local_path_gate")


def test_missing_holdout_plan_rejected():
    raw = _valid_raw()
    del raw["holdout_plan"]
    _assert_rejected(raw, "holdout_plan")


def test_random_only_validation_rejected():
    raw = _valid_raw()
    raw["holdout_plan"] = {
        "strategy": "random_split",
        "random_only_validation_allowed": True,
        "reason": "random split only",
    }
    _assert_rejected(raw, "random_only_validation_allowed")


def test_missing_required_metadata_rejected():
    raw = _valid_raw()
    raw["required_metadata"] = ["architecture", "subset"]
    _assert_rejected(raw, "model_name")


def test_image_reading_or_training_requests_rejected():
    for key in ("read_real_images", "read_real_masks", "run_training"):
        raw = _valid_raw()
        raw["validation_plan"][key] = True
        _assert_rejected(raw, key)


def test_validator_does_not_write_files():
    raw = _valid_raw()
    if _REPO_ROOT is None:
        return
    before = set(os.listdir(str(_REPO_ROOT)))
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
