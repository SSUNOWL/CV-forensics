"""Tests for task 0011 local data readiness gate.

Runnable directly:
    python3 tests/test_local_data_gate.py

Pytest-compatible:
    pytest -q tests/test_local_data_gate.py

Uses only the Python standard library. Does not import pytest or any
third-party package.
"""
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

_REPO_ROOT = next(
    (p for p in _Path(__file__).resolve().parents if (p / "src" / "cv_forensics").is_dir()),
    None,
)
if _REPO_ROOT is not None:
    _SRC_ROOT = str(_REPO_ROOT / "src")
    if _SRC_ROOT not in _sys.path:
        _sys.path.insert(0, _SRC_ROOT)

import json
import os
import sys

from cv_forensics.local_data_gate import (
    LocalDataReadinessConfig,
    LocalDataReadinessResult,
    ReadinessIssue,
    check_local_data_readiness_config_safety,
    evaluate_local_data_readiness,
    load_local_data_readiness_config,
    summarize_local_data_readiness,
    validate_local_data_readiness_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_raw() -> dict:
    return {
        "schema_version": "0.1.0",
        "dry_run": True,
        "no_download": True,
        "no_training": True,
        "no_network": True,
        "no_outputs": True,
        "no_checkpoints": True,
        "dataset_plan": {
            "cf_small": {"role": "backbone_pretraining", "status": "planned_not_downloaded"},
        },
        "manifest_refs": ["community_forensics_small", "sid_set"],
        "local_path_policy": {"policy": "symbolic_only", "note": "symbolic only"},
        "output_policy": {"policy": "no_outputs"},
        "checkpoint_policy": {"policy": "no_checkpoints"},
        "approval": {"local_data_approved": False, "note": "not yet approved"},
        "protected_path_exclusions": [".env", "secrets", "data", "datasets", "outputs", "checkpoints"],
    }


def _make_valid_config() -> LocalDataReadinessConfig:
    return validate_local_data_readiness_config(_make_valid_raw())


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------

def test_valid_symbolic_config_passes():
    config = _make_valid_config()
    assert config.dry_run is True
    assert config.no_download is True
    assert config.no_training is True
    assert config.no_network is True
    assert config.no_outputs is True
    assert config.no_checkpoints is True
    assert config.schema_version == "0.1.0"
    assert config.protected_path_exclusions


def test_dry_run_false_rejected():
    raw = _make_valid_raw()
    raw["dry_run"] = False
    try:
        validate_local_data_readiness_config(raw)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "dry_run" in str(exc)


def test_no_download_false_rejected():
    raw = _make_valid_raw()
    raw["no_download"] = False
    try:
        validate_local_data_readiness_config(raw)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "no_download" in str(exc)


def test_no_training_false_rejected():
    raw = _make_valid_raw()
    raw["no_training"] = False
    try:
        validate_local_data_readiness_config(raw)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "no_training" in str(exc)


def test_no_outputs_false_rejected():
    raw = _make_valid_raw()
    raw["no_outputs"] = False
    try:
        validate_local_data_readiness_config(raw)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "no_outputs" in str(exc)


def test_no_checkpoints_false_rejected():
    raw = _make_valid_raw()
    raw["no_checkpoints"] = False
    try:
        validate_local_data_readiness_config(raw)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "no_checkpoints" in str(exc)


def test_missing_required_key_rejected():
    for key in (
        "schema_version", "dry_run", "no_download", "no_training",
        "no_network", "no_outputs", "no_checkpoints",
        "dataset_plan", "manifest_refs", "local_path_policy",
        "output_policy", "checkpoint_policy", "approval",
        "protected_path_exclusions",
    ):
        raw = _make_valid_raw()
        del raw[key]
        try:
            validate_local_data_readiness_config(raw)
            assert False, f"Should have raised ValueError for missing {key!r}"
        except ValueError as exc:
            assert key in str(exc), f"Error message should mention {key!r}: {exc}"


# ---------------------------------------------------------------------------
# Safety checker tests
# ---------------------------------------------------------------------------

def test_protected_paths_rejected():
    cases = [
        {"path": "/home/user/data.json"},
        {"path": "/mnt/storage/images"},
        {"path": "/root/datasets"},
        {"path": "/Users/user/models"},
        {"folder": "data/images"},
        {"folder": "datasets/train"},
        {"folder": "outputs/run1"},
        {"folder": "checkpoints/epoch1"},
        {"dir": "secrets/keys"},
    ]
    for bad_cfg in cases:
        try:
            check_local_data_readiness_config_safety(bad_cfg)
            assert False, f"Should have rejected: {bad_cfg}"
        except ValueError:
            pass


def test_urls_rejected():
    cases = [
        {"url": "http://example.com/data"},
        {"url": "https://example.com/model.pt"},
        {"url": "ftp://host/file.zip"},
        {"ref": "s3://bucket/key"},
        {"ref": "gs://bucket/path"},
        {"ref": "hf://org/repo"},
    ]
    for bad_cfg in cases:
        try:
            check_local_data_readiness_config_safety(bad_cfg)
            assert False, f"Should have rejected: {bad_cfg}"
        except ValueError:
            pass


def test_secret_like_keys_rejected():
    cases = [
        {"api_key": "abc123"},
        {"password": "hunter2"},
        {"token": "abcdef"},
        {"secret": "mysecret"},
        {"credential": "xyz"},
        {"bearer": "abc"},
    ]
    for bad_cfg in cases:
        try:
            check_local_data_readiness_config_safety(bad_cfg)
            assert False, f"Should have rejected secret key: {bad_cfg}"
        except ValueError:
            pass


def test_secret_like_values_rejected():
    cases = [
        {"info": "token abc123"},
        {"info": "password mysecret"},
        {"info": "secret value here"},
        {"info": "bearer xyz123"},
    ]
    for bad_cfg in cases:
        try:
            check_local_data_readiness_config_safety(bad_cfg)
            assert False, f"Should have rejected secret value: {bad_cfg}"
        except ValueError:
            pass


def test_standalone_auth_key_rejected():
    try:
        check_local_data_readiness_config_safety({"auth": "some_value"})
        assert False, "Should have rejected standalone 'auth' key"
    except ValueError:
        pass


def test_standalone_auth_value_rejected():
    try:
        check_local_data_readiness_config_safety({"info": "auth"})
        assert False, "Should have rejected standalone 'auth' value"
    except ValueError:
        pass


def test_auth_prose_allowed():
    # Ordinary prose containing 'auth' as a substring must NOT be rejected.
    safe_prose = [
        {"note": "authoritative guidance on the policy"},
        {"description": "authentication policy for the project"},
        {"label": "authorization_level"},
    ]
    for cfg in safe_prose:
        try:
            check_local_data_readiness_config_safety(cfg)
        except ValueError as exc:
            assert False, (
                f"Safety checker incorrectly rejected safe prose {cfg!r}: {exc}"
            )


def test_env_file_rejected():
    cases = [
        {"cfg": ".env"},
        {"cfg": ".env.local"},
        {"cfg": ".env.production"},
    ]
    for bad_cfg in cases:
        try:
            check_local_data_readiness_config_safety(bad_cfg)
            assert False, f"Should have rejected: {bad_cfg}"
        except ValueError:
            pass


def test_windows_drive_path_rejected():
    try:
        check_local_data_readiness_config_safety({"path": "C:\\models\\weights.pt"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_nested_unsafe_value_rejected():
    bad_cfg = {"nested": {"deep": {"url": "https://evil.example.com"}}}
    try:
        check_local_data_readiness_config_safety(bad_cfg)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_protected_path_exclusions_field_not_rejected_by_safety():
    # The protected_path_exclusions field is declarative: its values name
    # what is excluded, not real paths. The safety checker must skip it.
    cfg = {
        "schema_version": "0.1.0",
        "dry_run": True,
        "note": "symbolic config",
        "protected_path_exclusions": [
            ".env", ".env.*", "secrets", "data", "datasets", "outputs", "checkpoints"
        ],
    }
    try:
        check_local_data_readiness_config_safety(cfg)
    except ValueError as exc:
        assert False, (
            f"Safety checker incorrectly rejected protected_path_exclusions content: {exc}"
        )


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

def test_valid_config_evaluation_is_ready():
    config = _make_valid_config()
    result = evaluate_local_data_readiness(config)
    assert isinstance(result, LocalDataReadinessResult)
    assert result.ready is True
    assert result.dry_run_safe is True
    assert result.has_approval_field is True
    assert result.has_protected_path_exclusions is True


def test_evaluation_is_deterministic():
    config = _make_valid_config()
    result1 = evaluate_local_data_readiness(config)
    result2 = evaluate_local_data_readiness(config)
    assert result1.ready == result2.ready
    assert result1.dry_run_safe == result2.dry_run_safe
    assert result1.summary == result2.summary
    assert len(result1.issues) == len(result2.issues)
    for i1, i2 in zip(result1.issues, result2.issues):
        assert i1.field == i2.field
        assert i1.message == i2.message
        assert i1.severity == i2.severity


def test_evaluation_summary_ok_on_ready():
    config = _make_valid_config()
    result = evaluate_local_data_readiness(config)
    assert result.ready is True
    assert "LOCAL_DATA_READINESS_OK" in result.summary


def test_evaluation_summary_fail_on_not_ready():
    raw = _make_valid_raw()
    raw["dry_run"] = False
    try:
        validate_local_data_readiness_config(raw)
        assert False, "Should have failed validation"
    except ValueError:
        # Construct config directly to test evaluation path
        pass

    config = LocalDataReadinessConfig(
        schema_version="0.1.0",
        dry_run=False,
        no_download=True,
        no_training=True,
        no_network=True,
        no_outputs=True,
        no_checkpoints=True,
        dataset_plan={},
        manifest_refs=["smoke_ref"],
        local_path_policy={"policy": "symbolic_only"},
        output_policy={"policy": "no_outputs"},
        checkpoint_policy={"policy": "no_checkpoints"},
        approval={"local_data_approved": False},
        protected_path_exclusions=["secrets"],
    )
    result = evaluate_local_data_readiness(config)
    assert result.ready is False
    assert "LOCAL_DATA_READINESS_FAIL" in result.summary


def test_real_path_policy_without_approval_is_not_ready():
    # local_path_policy indicating real paths requires local_data_approved=true.
    config = LocalDataReadinessConfig(
        schema_version="0.1.0",
        dry_run=True,
        no_download=True,
        no_training=True,
        no_network=True,
        no_outputs=True,
        no_checkpoints=True,
        dataset_plan={"cf_small": {"role": "training"}},
        manifest_refs=["smoke"],
        local_path_policy={"policy": "use_real_paths"},
        output_policy={"policy": "no_outputs"},
        checkpoint_policy={"policy": "no_checkpoints"},
        approval={"local_data_approved": False},
        protected_path_exclusions=["secrets", "data"],
    )
    result = evaluate_local_data_readiness(config)
    assert result.ready is False
    assert any("local_data_approved" in i.message for i in result.issues)


def test_real_path_policy_with_approval_is_ready():
    # With local_data_approved=true and real path policy, evaluation should pass.
    config = LocalDataReadinessConfig(
        schema_version="0.1.0",
        dry_run=True,
        no_download=True,
        no_training=True,
        no_network=True,
        no_outputs=True,
        no_checkpoints=True,
        dataset_plan={"cf_small": {"role": "training"}},
        manifest_refs=["smoke"],
        local_path_policy={"policy": "use_real_paths"},
        output_policy={"policy": "no_outputs"},
        checkpoint_policy={"policy": "no_checkpoints"},
        approval={"local_data_approved": True},
        protected_path_exclusions=["secrets", "data"],
    )
    result = evaluate_local_data_readiness(config)
    assert result.ready is True
    assert "LOCAL_DATA_READINESS_OK" in result.summary


def test_missing_protected_path_exclusions_is_not_ready():
    config = LocalDataReadinessConfig(
        schema_version="0.1.0",
        dry_run=True,
        no_download=True,
        no_training=True,
        no_network=True,
        no_outputs=True,
        no_checkpoints=True,
        dataset_plan={},
        manifest_refs=["smoke"],
        local_path_policy={"policy": "symbolic_only"},
        output_policy={"policy": "no_outputs"},
        checkpoint_policy={"policy": "no_checkpoints"},
        approval={"local_data_approved": False},
        protected_path_exclusions=[],
    )
    result = evaluate_local_data_readiness(config)
    assert result.ready is False
    assert result.has_protected_path_exclusions is False
    assert any(i.field == "protected_path_exclusions" for i in result.issues)


def test_summarize_local_data_readiness_returns_string():
    config = _make_valid_config()
    result = evaluate_local_data_readiness(config)
    summary = summarize_local_data_readiness(result)
    assert isinstance(summary, str)
    assert len(summary) > 0


# ---------------------------------------------------------------------------
# No-file-write tests
# ---------------------------------------------------------------------------

def test_no_files_written_by_evaluator():
    config = _make_valid_config()
    before = set(os.listdir("."))
    evaluate_local_data_readiness(config)
    after = set(os.listdir("."))
    new_files = after - before
    assert len(new_files) == 0, f"Unexpected files created by evaluator: {new_files}"


def test_no_files_written_by_validator():
    raw = _make_valid_raw()
    before = set(os.listdir("."))
    validate_local_data_readiness_config(raw)
    after = set(os.listdir("."))
    new_files = after - before
    assert len(new_files) == 0, f"Unexpected files created by validator: {new_files}"


# ---------------------------------------------------------------------------
# Load from file test
# ---------------------------------------------------------------------------

def test_load_from_example_file():
    if _REPO_ROOT is None:
        return
    example_path = str(_REPO_ROOT / "configs" / "local_data" / "readiness.example.json")
    if not os.path.isfile(example_path):
        return
    config = load_local_data_readiness_config(example_path)
    assert config.dry_run is True
    assert config.no_download is True
    assert config.no_training is True
    assert config.no_outputs is True
    assert config.no_checkpoints is True
    assert config.protected_path_exclusions
    result = evaluate_local_data_readiness(config)
    assert result.ready is True
    assert "LOCAL_DATA_READINESS_OK" in result.summary


# ---------------------------------------------------------------------------
# Direct execution runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    failures = []
    passed = 0
    test_fns = {name: fn for name, fn in globals().items() if name.startswith("test_")}
    for name in sorted(test_fns):
        fn = test_fns[name]
        try:
            fn()
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
        sys.exit(1)
    print("All tests passed.")
