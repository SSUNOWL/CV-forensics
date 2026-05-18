"""Tests for task 0009 pure-Python metric calculators.

Runnable directly:
    python3 tests/test_metrics.py

Pytest-compatible:
    pytest -q tests/test_metrics.py

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

import math
import sys

from cv_forensics.metrics import (
    check_toy_metric_config_safety,
    classification_accuracy,
    classification_confusion_matrix,
    classification_metrics,
    family_accuracy,
    fps_summary,
    latency_summary,
    localization_activation_recall,
    macro_f1,
    mask_iou,
    per_class_f1,
    per_class_precision,
    per_class_recall,
    perturbation_robustness_drop,
    robustness_summary,
)


# ---------------------------------------------------------------------------
# Classification metric tests
# ---------------------------------------------------------------------------

def test_accuracy_perfect():
    y = ["real", "synthetic", "tampered"]
    assert classification_accuracy(y, y) == 1.0


def test_accuracy_all_wrong():
    y_true = ["real", "synthetic", "tampered"]
    y_pred = ["synthetic", "tampered", "real"]
    assert classification_accuracy(y_true, y_pred) == 0.0


def test_accuracy_partial():
    y_true = ["real", "real", "synthetic", "tampered"]
    y_pred = ["real", "synthetic", "synthetic", "real"]
    # 2 correct out of 4
    assert abs(classification_accuracy(y_true, y_pred) - 0.5) < 1e-9


def test_macro_f1_perfect():
    y = ["real", "synthetic", "tampered", "real", "synthetic", "tampered"]
    assert abs(macro_f1(y, y) - 1.0) < 1e-9


def test_macro_f1_value():
    y_true = ["real", "synthetic", "tampered", "real", "synthetic", "tampered"]
    y_pred = ["real", "synthetic", "tampered", "synthetic", "tampered", "real"]
    mf1 = macro_f1(y_true, y_pred)
    assert 0.0 <= mf1 <= 1.0
    # 3 of 6 correct, non-trivial F1
    assert mf1 < 1.0


def test_precision_zero_division_safe():
    # Class "tampered" has no predicted instances → precision should be 0.0
    y_true = ["real", "real", "synthetic", "synthetic"]
    y_pred = ["real", "synthetic", "synthetic", "real"]
    prec = per_class_precision(y_true, y_pred)
    assert prec["tampered"] == 0.0


def test_recall_zero_division_safe():
    # Class "tampered" not in y_true → recall should be 0.0
    y_true = ["real", "real", "synthetic", "synthetic"]
    y_pred = ["real", "real", "synthetic", "tampered"]
    rec = per_class_recall(y_true, y_pred)
    assert rec["tampered"] == 0.0


def test_f1_zero_division_safe():
    # Class "tampered" absent entirely → F1 should be 0.0
    y_true = ["real", "synthetic", "real", "synthetic"]
    y_pred = ["real", "synthetic", "real", "synthetic"]
    f1s = per_class_f1(y_true, y_pred)
    assert f1s["tampered"] == 0.0
    # Real and synthetic should be 1.0
    assert abs(f1s["real"] - 1.0) < 1e-9
    assert abs(f1s["synthetic"] - 1.0) < 1e-9


def test_confusion_matrix_shape():
    y_true = ["real", "synthetic", "tampered"]
    y_pred = ["real", "tampered", "tampered"]
    cm = classification_confusion_matrix(y_true, y_pred)
    assert cm["labels"] == ["real", "synthetic", "tampered"]
    assert len(cm["matrix"]) == 3
    assert all(len(row) == 3 for row in cm["matrix"])
    assert cm["n"] == 3


def test_confusion_matrix_values():
    y_true = ["real", "real", "synthetic"]
    y_pred = ["real", "synthetic", "synthetic"]
    cm = classification_confusion_matrix(y_true, y_pred)
    # real→real: 1, real→synthetic: 1, synthetic→synthetic: 1
    assert cm["matrix"][0][0] == 1  # real predicted as real
    assert cm["matrix"][0][1] == 1  # real predicted as synthetic
    assert cm["matrix"][1][1] == 1  # synthetic predicted as synthetic


def test_classification_metrics_bundle_keys():
    y = ["real", "synthetic", "tampered"] * 2
    result = classification_metrics(y, y)
    for key in ("confusion_matrix", "accuracy", "per_class_precision",
                "per_class_recall", "per_class_f1", "macro_f1"):
        assert key in result


def test_invalid_class_label_rejected():
    try:
        classification_accuracy(["real", "UNKNOWN"], ["real", "real"])
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "UNKNOWN" in str(exc)


def test_mismatched_lengths_rejected():
    try:
        classification_accuracy(["real"], ["real", "synthetic"])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_empty_labels_rejected():
    try:
        classification_accuracy([], [])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Family metric tests
# ---------------------------------------------------------------------------

def test_family_accuracy_perfect():
    y = ["LatDiff", "PixDiff", "GAN", "Other"]
    result = family_accuracy(y, y)
    assert result["accuracy"] == 1.0
    assert result["n_evaluated"] == 4
    assert result["n_correct"] == 4
    assert result["n_ignored"] == 0


def test_family_accuracy_ignore_real_or_na():
    y_true = ["Real-or-N/A", "LatDiff", "GAN"]
    y_pred = ["LatDiff",     "LatDiff", "PixDiff"]
    result = family_accuracy(y_true, y_pred, ignore_real_or_na=True)
    assert result["n_ignored"] == 1
    assert result["n_evaluated"] == 2
    assert result["n_correct"] == 1  # only LatDiff matches
    assert abs(result["accuracy"] - 0.5) < 1e-9


def test_family_accuracy_no_ignore():
    y_true = ["Real-or-N/A", "LatDiff"]
    y_pred = ["Real-or-N/A", "GAN"]
    result = family_accuracy(y_true, y_pred, ignore_real_or_na=False)
    assert result["n_ignored"] == 0
    assert result["n_evaluated"] == 2
    assert result["n_correct"] == 1
    assert abs(result["accuracy"] - 0.5) < 1e-9


def test_invalid_family_label_rejected():
    try:
        family_accuracy(["LatDiff", "UNKNOWN"], ["LatDiff", "GAN"])
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "UNKNOWN" in str(exc)


# ---------------------------------------------------------------------------
# Mask IoU tests
# ---------------------------------------------------------------------------

def test_mask_iou_perfect_match():
    pred = [[1, 0], [0, 1]]
    gt   = [[1, 0], [0, 1]]
    assert abs(mask_iou(pred, gt) - 1.0) < 1e-9


def test_mask_iou_no_overlap():
    pred = [[1, 0], [0, 0]]
    gt   = [[0, 0], [0, 1]]
    assert abs(mask_iou(pred, gt) - 0.0) < 1e-9


def test_mask_iou_partial_overlap():
    pred = [[1, 0], [0, 1]]
    gt   = [[1, 1], [0, 0]]
    # intersection=1 (top-left), union=3 (top-left, top-right, bottom-right)
    assert abs(mask_iou(pred, gt) - 1.0 / 3.0) < 1e-9


def test_mask_iou_both_zeros_returns_one():
    pred = [[0, 0], [0, 0]]
    gt   = [[0, 0], [0, 0]]
    assert mask_iou(pred, gt) == 1.0


def test_mask_iou_boolean_values():
    pred = [[True, False], [False, True]]
    gt   = [[True, False], [False, True]]
    assert abs(mask_iou(pred, gt) - 1.0) < 1e-9


def test_mask_iou_ragged_rejected():
    pred = [[1, 0], [1]]  # ragged
    gt   = [[1, 0], [0, 1]]
    try:
        mask_iou(pred, gt)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "ragged" in str(exc).lower()


def test_mask_iou_invalid_value_rejected():
    pred = [[1, 2], [0, 1]]  # 2 is not valid
    gt   = [[1, 0], [0, 1]]
    try:
        mask_iou(pred, gt)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "invalid mask value" in str(exc).lower()


def test_mask_iou_shape_mismatch_rejected():
    pred = [[1, 0, 1], [0, 1, 0]]
    gt   = [[1, 0], [0, 1]]
    try:
        mask_iou(pred, gt)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "shape" in str(exc).lower()


def test_mask_iou_empty_mask_rejected():
    try:
        mask_iou([], [[1, 0]])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Localization activation recall tests
# ---------------------------------------------------------------------------

def test_localization_recall_all_activated():
    gt = ["tampered", "tampered", "real"]
    states = ["activated", "activated", "not_applicable"]
    result = localization_activation_recall(gt, states)
    assert result["n_tampered"] == 2
    assert result["n_activated"] == 2
    assert result["activation_recall"] == 1.0


def test_localization_recall_partial():
    gt = ["tampered", "tampered", "tampered", "real"]
    states = ["activated", "skipped_below_threshold", "activated", "not_applicable"]
    result = localization_activation_recall(gt, states)
    assert result["n_tampered"] == 3
    assert result["n_activated"] == 2
    assert result["n_skipped_below_threshold"] == 1
    assert abs(result["activation_recall"] - 2.0 / 3.0) < 1e-9


def test_localization_recall_no_tampered():
    gt = ["real", "synthetic", "real"]
    states = ["not_applicable", "not_applicable", "not_applicable"]
    result = localization_activation_recall(gt, states)
    assert result["n_tampered"] == 0
    assert result["activation_recall"] == 0.0


def test_localization_recall_skipped_is_missed():
    gt = ["tampered", "tampered"]
    states = ["skipped_below_threshold", "skipped_below_threshold"]
    result = localization_activation_recall(gt, states)
    assert result["n_skipped_below_threshold"] == 2
    assert result["activation_recall"] == 0.0


def test_localization_invalid_state_rejected():
    try:
        localization_activation_recall(["tampered"], ["INVALID_STATE"])
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "INVALID_STATE" in str(exc)


# ---------------------------------------------------------------------------
# Latency and FPS tests
# ---------------------------------------------------------------------------

def test_latency_summary_values():
    values = [10.0, 20.0, 30.0]
    result = latency_summary(values)
    assert result["count"] == 3
    assert abs(result["mean_ms"] - 20.0) < 1e-9
    assert abs(result["min_ms"] - 10.0) < 1e-9
    assert abs(result["max_ms"] - 30.0) < 1e-9


def test_fps_summary_values():
    values = [100.0, 200.0]  # 10 FPS and 5 FPS
    result = fps_summary(values)
    assert result["count"] == 2
    assert abs(result["min_fps"] - 5.0) < 1e-9
    assert abs(result["max_fps"] - 10.0) < 1e-9
    assert abs(result["mean_fps"] - 7.5) < 1e-9


def test_latency_negative_rejected():
    try:
        latency_summary([-1.0])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_latency_zero_rejected():
    try:
        latency_summary([0.0])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_latency_nan_rejected():
    try:
        latency_summary([float("nan")])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_latency_inf_rejected():
    try:
        latency_summary([float("inf")])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_latency_bool_rejected():
    try:
        latency_summary([True])
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_latency_empty_rejected():
    try:
        latency_summary([])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Robustness drop tests
# ---------------------------------------------------------------------------

def test_robustness_drop_positive():
    result = perturbation_robustness_drop(0.85, 0.70, "jpeg")
    assert result["perturbation_tag"] == "jpeg"
    assert abs(result["clean_score"] - 0.85) < 1e-9
    assert abs(result["perturbed_score"] - 0.70) < 1e-9
    assert abs(result["drop"] - 0.15) < 1e-9


def test_robustness_drop_zero():
    result = perturbation_robustness_drop(0.80, 0.80, "resize")
    assert abs(result["drop"]) < 1e-9


def test_robustness_drop_negative():
    # perturbed better than clean (allowed; just a negative drop)
    result = perturbation_robustness_drop(0.70, 0.80, "crop")
    assert result["drop"] < 0.0


def test_robustness_summary_keys():
    records = [
        {"perturbation_tag": "jpeg",   "clean_score": 0.9, "perturbed_score": 0.8},
        {"perturbation_tag": "resize", "clean_score": 0.9, "perturbed_score": 0.85},
    ]
    result = robustness_summary(records)
    assert "per_perturbation" in result
    assert "mean_drop" in result
    assert "max_drop" in result
    assert result["n"] == 2
    assert abs(result["mean_drop"] - 0.075) < 1e-9
    assert abs(result["max_drop"] - 0.10) < 1e-9


def test_robustness_unknown_tag_rejected():
    try:
        perturbation_robustness_drop(0.8, 0.7, "unknown_perturbation")
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "unknown_perturbation" in str(exc)


def test_robustness_non_numeric_score_rejected():
    try:
        perturbation_robustness_drop("high", 0.7, "jpeg")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_robustness_nan_score_rejected():
    try:
        perturbation_robustness_drop(float("nan"), 0.7, "jpeg")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_robustness_bool_score_rejected():
    try:
        perturbation_robustness_drop(True, 0.7, "jpeg")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# ---------------------------------------------------------------------------
# Config safety checker tests
# ---------------------------------------------------------------------------

def test_safety_checker_passes_clean_config():
    cfg = {
        "dry_run": True,
        "labels": ["real", "synthetic"],
        "latency_ms": [12.5, 14.0],
    }
    check_toy_metric_config_safety(cfg)  # must not raise


def test_safety_checker_rejects_absolute_path():
    try:
        check_toy_metric_config_safety({"path": "/home/user/data.csv"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_safety_checker_rejects_https_url():
    try:
        check_toy_metric_config_safety({"url": "https://example.com/model.pt"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_safety_checker_rejects_s3_url():
    try:
        check_toy_metric_config_safety({"ref": "s3://my-bucket/weights.bin"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_safety_checker_rejects_protected_dir_data():
    try:
        check_toy_metric_config_safety({"folder": "data/train"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_safety_checker_rejects_protected_dir_checkpoints():
    try:
        check_toy_metric_config_safety({"folder": "checkpoints/run1"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_safety_checker_rejects_env_file():
    try:
        check_toy_metric_config_safety({"cfg": ".env"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_safety_checker_rejects_env_star_file():
    try:
        check_toy_metric_config_safety({"cfg": ".env.local"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_safety_checker_rejects_secret_key_name():
    try:
        check_toy_metric_config_safety({"api_key": "abc123"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_safety_checker_rejects_token_in_value():
    try:
        check_toy_metric_config_safety({"info": "token abc123"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_safety_checker_rejects_windows_drive_path():
    try:
        check_toy_metric_config_safety({"path": "C:\\models\\weights.pth"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_safety_checker_rejects_hf_url():
    try:
        check_toy_metric_config_safety({"model": "hf://org/repo"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_safety_checker_nested_unsafe_value():
    try:
        check_toy_metric_config_safety({
            "config": {
                "nested": {
                    "deep": "https://malicious.example.com"
                }
            }
        })
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_safety_checker_list_unsafe_value():
    try:
        check_toy_metric_config_safety({
            "paths": ["/home/user/file.json", "safe_value"]
        })
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


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
