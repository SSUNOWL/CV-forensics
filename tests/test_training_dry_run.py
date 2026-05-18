"""Tests for task 0010 dry-run training loop skeleton.

Runnable directly:
    python3 tests/test_training_dry_run.py

Pytest-compatible:
    pytest -q tests/test_training_dry_run.py

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
    _SRC_ROOT = _REPO_ROOT / "src"
    if str(_SRC_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_SRC_ROOT))

import json
import os
import sys

from cv_forensics.training_dry_run import (
    ALLOWED_METRIC_NAMES,
    DryRunTrainingConfig,
    DryRunTrainingResult,
    DryRunEpochSummary,
    build_fake_batches,
    check_dry_run_training_config_safety,
    dry_run_result_to_dict,
    load_dry_run_training_config,
    run_dry_training,
    summarize_dry_run,
    validate_dry_run_training_config,
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
        "stage": "dry_run",
        "epochs": 2,
        "batch_size": 4,
        "seed": 42,
        "threshold_tau": 0.5,
        "family_policy": "mask_out",
        "fake_input_config_ref": "configs/inference/fake_inputs.example.json",
        "toy_metrics_config_ref": "configs/metrics/toy_metrics.example.json",
        "dataset_manifest_refs": ["community_forensics_small_smoke", "sid_set_smoke"],
        "metric_names": list(ALLOWED_METRIC_NAMES),
    }


def _make_valid_config() -> DryRunTrainingConfig:
    return validate_dry_run_training_config(_make_valid_raw())


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------

def test_valid_dry_run_config_passes():
    config = _make_valid_config()
    assert config.dry_run is True
    assert config.no_training is True
    assert config.no_outputs is True
    assert config.no_checkpoints is True
    assert config.stage == "dry_run"
    assert config.epochs == 2
    assert config.batch_size == 4
    assert config.seed == 42


def test_dry_run_false_rejected():
    raw = _make_valid_raw()
    raw["dry_run"] = False
    try:
        validate_dry_run_training_config(raw)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "dry_run" in str(exc)


def test_no_training_false_rejected():
    raw = _make_valid_raw()
    raw["no_training"] = False
    try:
        validate_dry_run_training_config(raw)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "no_training" in str(exc)


def test_no_outputs_false_rejected():
    raw = _make_valid_raw()
    raw["no_outputs"] = False
    try:
        validate_dry_run_training_config(raw)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "no_outputs" in str(exc)


def test_no_checkpoints_false_rejected():
    raw = _make_valid_raw()
    raw["no_checkpoints"] = False
    try:
        validate_dry_run_training_config(raw)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "no_checkpoints" in str(exc)


def test_missing_required_key_rejected():
    for key in ("schema_version", "stage", "epochs", "batch_size", "seed"):
        raw = _make_valid_raw()
        del raw[key]
        try:
            validate_dry_run_training_config(raw)
            assert False, f"Should have raised ValueError for missing {key!r}"
        except ValueError as exc:
            assert key in str(exc)


def test_unknown_stage_rejected():
    raw = _make_valid_raw()
    raw["stage"] = "unknown_stage_xyz"
    try:
        validate_dry_run_training_config(raw)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "unknown_stage_xyz" in str(exc)


def test_unknown_family_policy_rejected():
    raw = _make_valid_raw()
    raw["family_policy"] = "invalid_policy"
    try:
        validate_dry_run_training_config(raw)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "invalid_policy" in str(exc)


def test_unknown_metric_name_rejected():
    raw = _make_valid_raw()
    raw["metric_names"] = ["accuracy_3way", "not_a_real_metric"]
    try:
        validate_dry_run_training_config(raw)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "not_a_real_metric" in str(exc)


# ---------------------------------------------------------------------------
# Safety checker tests
# ---------------------------------------------------------------------------

def test_protected_path_rejected():
    unsafe_cases = [
        {"path": "/home/user/data.json"},
        {"path": "/mnt/storage/images"},
        {"path": "/root/datasets"},
        {"path": "/Users/user/checkpoints"},
        {"folder": "data/images"},
        {"folder": "datasets/train"},
        {"folder": "outputs/run1"},
        {"folder": "checkpoints/epoch1"},
        {"dir": "secrets/api_keys"},
    ]
    for bad_cfg in unsafe_cases:
        try:
            check_dry_run_training_config_safety(bad_cfg)
            assert False, f"Should have rejected: {bad_cfg}"
        except ValueError:
            pass


def test_url_values_rejected():
    url_cases = [
        {"url": "http://example.com/data"},
        {"url": "https://example.com/model.pt"},
        {"url": "ftp://host/file.zip"},
        {"ref": "s3://bucket/key"},
        {"ref": "gs://bucket/path"},
        {"ref": "hf://org/repo"},
    ]
    for bad_cfg in url_cases:
        try:
            check_dry_run_training_config_safety(bad_cfg)
            assert False, f"Should have rejected: {bad_cfg}"
        except ValueError:
            pass


def test_secret_like_keys_rejected():
    secret_key_cases = [
        {"api_key": "abc123"},
        {"password": "hunter2"},
        {"token": "abcdef"},
        {"secret": "mysecret"},
        {"credential": "xyz"},
        {"auth": "Bearer token123"},
        {"bearer": "abc"},
    ]
    for bad_cfg in secret_key_cases:
        try:
            check_dry_run_training_config_safety(bad_cfg)
            assert False, f"Should have rejected secret key: {bad_cfg}"
        except ValueError:
            pass


def test_secret_like_values_rejected():
    secret_value_cases = [
        {"info": "token abc123"},
        {"info": "password mysecret"},
        {"info": "secret value here"},
    ]
    for bad_cfg in secret_value_cases:
        try:
            check_dry_run_training_config_safety(bad_cfg)
            assert False, f"Should have rejected secret value: {bad_cfg}"
        except ValueError:
            pass


def test_env_file_rejected():
    env_cases = [
        {"cfg": ".env"},
        {"cfg": ".env.local"},
        {"cfg": ".env.production"},
    ]
    for bad_cfg in env_cases:
        try:
            check_dry_run_training_config_safety(bad_cfg)
            assert False, f"Should have rejected: {bad_cfg}"
        except ValueError:
            pass


def test_windows_drive_path_rejected():
    try:
        check_dry_run_training_config_safety({"path": "C:\\models\\weights.pt"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_nested_unsafe_value_rejected():
    bad_cfg = {"nested": {"deep": {"url": "https://evil.example.com"}}}
    try:
        check_dry_run_training_config_safety(bad_cfg)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_safe_config_passes_safety_check():
    safe_cfg = {
        "dry_run": True,
        "stage": "dry_run",
        "metric_names": ["accuracy_3way", "macro_f1"],
        "dataset_manifest_refs": ["community_forensics_small_smoke"],
    }
    check_dry_run_training_config_safety(safe_cfg)


# ---------------------------------------------------------------------------
# Fake batch builder tests
# ---------------------------------------------------------------------------

def test_fake_batches_deterministic():
    config = _make_valid_config()
    batches1 = build_fake_batches(config, n_batches=4)
    batches2 = build_fake_batches(config, n_batches=4)
    assert batches1 == batches2


def test_fake_batches_correct_structure():
    config = _make_valid_config()
    n_batches = 3
    batches = build_fake_batches(config, n_batches=n_batches)
    assert len(batches) == n_batches
    for batch in batches:
        assert len(batch) == config.batch_size
        for record in batch:
            assert "input_id" in record
            assert "scenario" in record
            assert "class_hint" in record
            assert "family_hint" in record
            assert "tampered_score" in record
            assert "threshold_tau" in record
            assert record["threshold_tau"] == config.threshold_tau


def test_fake_batches_different_seeds_differ():
    raw1 = _make_valid_raw()
    raw1["seed"] = 0
    config1 = validate_dry_run_training_config(raw1)

    raw2 = _make_valid_raw()
    raw2["seed"] = 99
    config2 = validate_dry_run_training_config(raw2)

    batches1 = build_fake_batches(config1, n_batches=4)
    batches2 = build_fake_batches(config2, n_batches=4)
    # Different seeds may produce different orderings
    ids1 = [r["input_id"] for r in batches1[0]]
    ids2 = [r["input_id"] for r in batches2[0]]
    assert ids1 != ids2


# ---------------------------------------------------------------------------
# Dry-run training tests
# ---------------------------------------------------------------------------

def test_run_dry_training_returns_result():
    config = _make_valid_config()
    result = run_dry_training(config)
    assert isinstance(result, DryRunTrainingResult)


def test_run_dry_training_deterministic():
    config = _make_valid_config()
    result1 = run_dry_training(config)
    result2 = run_dry_training(config)
    d1 = dry_run_result_to_dict(result1)
    d2 = dry_run_result_to_dict(result2)
    assert d1 == d2


def test_run_dry_training_correct_epoch_count():
    raw = _make_valid_raw()
    raw["epochs"] = 3
    config = validate_dry_run_training_config(raw)
    result = run_dry_training(config)
    assert len(result.epoch_summaries) == 3


def test_run_dry_training_flags_are_true():
    config = _make_valid_config()
    result = run_dry_training(config)
    assert result.dry_run is True
    assert result.no_training is True
    assert result.no_outputs is True
    assert result.no_checkpoints is True


def test_output_summary_includes_metric_names():
    config = _make_valid_config()
    result = run_dry_training(config)
    final = result.final_metrics
    for metric_name in config.metric_names:
        assert metric_name in final, f"Missing metric {metric_name!r} in final_metrics"


def test_output_summary_includes_task_0009_metrics():
    # Task 0009 defines: accuracy_3way, macro_f1, tampered_mask_iou,
    # generator_family_accuracy, perturbation_robustness_drop, latency_ms, fps,
    # localization_activation_recall
    raw = _make_valid_raw()
    raw["metric_names"] = list(ALLOWED_METRIC_NAMES)
    config = validate_dry_run_training_config(raw)
    result = run_dry_training(config)
    final = result.final_metrics
    for m in ALLOWED_METRIC_NAMES:
        assert m in final, f"Missing task-0009 metric {m!r} from final_metrics"


def test_final_metrics_contains_dry_run_markers():
    config = _make_valid_config()
    result = run_dry_training(config)
    final = result.final_metrics
    assert final.get("_dry_run") is True
    assert final.get("_no_real_training") is True
    assert final.get("_no_real_data") is True


def test_epoch_summaries_have_positive_samples():
    config = _make_valid_config()
    result = run_dry_training(config)
    for summary in result.epoch_summaries:
        assert summary.n_samples > 0
        assert summary.fake_inference_count > 0


def test_epoch_summaries_accuracy_in_range():
    import math
    config = _make_valid_config()
    result = run_dry_training(config)
    for summary in result.epoch_summaries:
        acc = summary.classification_metrics["accuracy"]
        assert isinstance(acc, float)
        assert not math.isnan(acc)
        assert 0.0 <= acc <= 1.0
        mf1 = summary.classification_metrics["macro_f1"]
        assert 0.0 <= mf1 <= 1.0


def test_no_files_written_by_runner():
    config = _make_valid_config()
    before = set(os.listdir("."))
    run_dry_training(config)
    after = set(os.listdir("."))
    new_files = after - before
    assert len(new_files) == 0, f"Unexpected files created: {new_files}"


def test_result_is_json_serializable():
    config = _make_valid_config()
    result = run_dry_training(config)
    output_dict = dry_run_result_to_dict(result)
    json_str = json.dumps(output_dict)
    assert len(json_str) > 0


def test_summarize_dry_run_includes_all_metrics():
    config = _make_valid_config()
    result = run_dry_training(config)
    summary = summarize_dry_run(result.epoch_summaries, config)
    for m in config.metric_names:
        assert m in summary, f"summarize_dry_run missing {m!r}"


def test_summarize_dry_run_empty_returns_safely():
    config = _make_valid_config()
    result = summarize_dry_run([], config)
    assert result.get("_dry_run") is True


# ---------------------------------------------------------------------------
# Load from file test
# ---------------------------------------------------------------------------

def test_load_dry_run_training_config_from_file():
    repo_root = _REPO_ROOT
    if repo_root is None:
        return
    example_path = str(repo_root / "configs" / "training" / "dry_run_training.example.json")
    if not os.path.isfile(example_path):
        return
    config = load_dry_run_training_config(example_path)
    assert config.dry_run is True
    assert config.no_training is True
    assert config.stage == "dry_run"


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
