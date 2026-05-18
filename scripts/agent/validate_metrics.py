#!/usr/bin/env python3
"""Validate the toy metrics module against a toy metrics config.

Usage:
    python3 scripts/agent/validate_metrics.py configs/metrics/toy_metrics.example.json

Checks:
- Config loads and has required safety flags (dry_run, no_download, no_training, no_network)
- Config passes safety checker (no protected paths, URLs, or secret-like values)
- Safety checker correctly rejects a synthetic unsafe config
- All metric calculators run on toy config data
- All metric results are finite, structured, and semantically plausible
- No files are written

Exits 0 on full pass, non-zero with actionable errors on any failure.
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
import math
import sys
from typing import Any, Dict


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _check_finite(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)):
        _fail(f"{name} is not numeric: {value!r}")
    if math.isnan(float(value)) or math.isinf(float(value)):
        _fail(f"{name} is not finite: {value!r}")


def _validate_safety_flags(cfg: Dict[str, Any]) -> None:
    for flag in ("dry_run", "no_download", "no_training", "no_network"):
        if not cfg.get(flag):
            _fail(f"Config missing or false safety flag: {flag!r}")


def _validate_classification_result(result: Dict[str, Any]) -> None:
    _check_finite(result["accuracy"], "accuracy")
    if not (0.0 <= result["accuracy"] <= 1.0):
        _fail(f"accuracy out of range: {result['accuracy']}")
    _check_finite(result["macro_f1"], "macro_f1")
    if not (0.0 <= result["macro_f1"] <= 1.0):
        _fail(f"macro_f1 out of range: {result['macro_f1']}")
    for cls in ("real", "synthetic", "tampered"):
        for metric_key in ("per_class_precision", "per_class_recall", "per_class_f1"):
            val = result[metric_key][cls]
            _check_finite(val, f"{metric_key}[{cls}]")
            if not (0.0 <= val <= 1.0):
                _fail(f"{metric_key}[{cls}] out of range: {val}")
    cm = result["confusion_matrix"]
    if cm["n"] <= 0:
        _fail(f"confusion_matrix n must be positive, got {cm['n']}")


def _validate_family_result(result: Dict[str, Any]) -> None:
    _check_finite(result["accuracy"], "family accuracy")
    if not (0.0 <= result["accuracy"] <= 1.0):
        _fail(f"family accuracy out of range: {result['accuracy']}")
    if result["n_evaluated"] < 0:
        _fail(f"n_evaluated must be non-negative: {result['n_evaluated']}")
    if result["n_correct"] < 0:
        _fail(f"n_correct must be non-negative: {result['n_correct']}")
    if result["n_correct"] > result["n_evaluated"]:
        _fail("n_correct > n_evaluated")


def _validate_iou(iou: float, idx: int) -> None:
    _check_finite(iou, f"mask_iou[{idx}]")
    if not (0.0 <= iou <= 1.0):
        _fail(f"mask_iou[{idx}] out of range: {iou}")


def _validate_loc_recall(result: Dict[str, Any]) -> None:
    _check_finite(result["activation_recall"], "activation_recall")
    if not (0.0 <= result["activation_recall"] <= 1.0):
        _fail(f"activation_recall out of range: {result['activation_recall']}")
    if result["n_tampered"] < 0:
        _fail(f"n_tampered must be non-negative: {result['n_tampered']}")
    if result["n_activated"] > result["n_tampered"]:
        _fail("n_activated > n_tampered")


def _validate_latency_result(result: Dict[str, Any]) -> None:
    for key in ("mean_ms", "min_ms", "max_ms"):
        _check_finite(result[key], key)
        if result[key] <= 0:
            _fail(f"{key} must be positive: {result[key]}")
    if result["min_ms"] > result["max_ms"]:
        _fail("min_ms > max_ms")
    if result["count"] <= 0:
        _fail(f"latency count must be positive: {result['count']}")


def _validate_fps_result(result: Dict[str, Any]) -> None:
    for key in ("mean_fps", "min_fps", "max_fps"):
        _check_finite(result[key], key)
        if result[key] <= 0:
            _fail(f"{key} must be positive: {result[key]}")
    if result["min_fps"] > result["max_fps"]:
        _fail("min_fps > max_fps")


def _validate_robustness_result(result: Dict[str, Any]) -> None:
    _check_finite(result["mean_drop"], "mean_drop")
    _check_finite(result["max_drop"], "max_drop")
    if result["n"] <= 0:
        _fail(f"robustness n must be positive: {result['n']}")
    for tag, rec in result["per_perturbation"].items():
        _check_finite(rec["drop"], f"robustness drop[{tag}]")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <toy_metrics_config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"Cannot load config {config_path!r}: {exc}")

    print(f"Loaded config: {config_path}")

    # --- Safety flags ---
    _validate_safety_flags(cfg)
    print("  Safety flags: OK")

    # --- Import metrics module ---
    try:
        from cv_forensics.metrics import (
            check_toy_metric_config_safety,
            classification_metrics,
            family_accuracy,
            fps_summary,
            latency_summary,
            mask_iou,
            localization_activation_recall,
            robustness_summary,
        )
    except ImportError as exc:
        _fail(f"Cannot import cv_forensics.metrics: {exc}")

    # --- Config safety check (must pass for toy config) ---
    try:
        check_toy_metric_config_safety(cfg)
    except ValueError as exc:
        _fail(f"Toy config failed safety check (it should pass): {exc}")
    print("  Config safety check: PASS")

    # --- Safety checker must reject unsafe values ---
    unsafe_cases = [
        ({"path": "/home/user/data.json"}, "absolute local path"),
        ({"url": "https://example.com/model"}, "https URL"),
        ({"s3_ref": "s3://bucket/key"}, "s3 URL"),
        ({"folder": "data/images"}, "protected dir 'data'"),
        ({"folder": "checkpoints/run1"}, "protected dir 'checkpoints'"),
        ({"env_file": ".env"}, ".env reference"),
        ({"env_file": ".env.local"}, ".env.* reference"),
        ({"api_key": "abc123"}, "secret key name 'api_key'"),
        ({"info": "token abc"}, "secret-looking value with 'token'"),
        ({"drive": "C:\\models\\weights"}, "Windows drive path"),
    ]
    for bad_cfg, label in unsafe_cases:
        try:
            check_toy_metric_config_safety(bad_cfg)
            _fail(f"Safety checker did NOT reject {label}: {bad_cfg!r}")
        except ValueError:
            pass  # correctly rejected
    print("  Safety checker correctly rejects unsafe values: PASS")

    # --- Classification metrics ---
    cls_cfg = cfg.get("class_labels", {})
    cls_result = classification_metrics(cls_cfg["y_true"], cls_cfg["y_pred"])
    _validate_classification_result(cls_result)
    print(f"  Classification accuracy: {cls_result['accuracy']:.4f}")
    print(f"  Classification Macro-F1: {cls_result['macro_f1']:.4f}")

    # --- Family metrics (with and without ignore_real_or_na) ---
    fam_cfg = cfg.get("family_labels", {})
    fam_result = family_accuracy(
        fam_cfg["y_true"], fam_cfg["y_pred"],
        ignore_real_or_na=fam_cfg.get("ignore_real_or_na", False),
    )
    _validate_family_result(fam_result)
    fam_result_all = family_accuracy(fam_cfg["y_true"], fam_cfg["y_pred"], ignore_real_or_na=False)
    _validate_family_result(fam_result_all)
    print(f"  Family accuracy (ignore Real-or-N/A={fam_cfg.get('ignore_real_or_na', False)}): "
          f"{fam_result['accuracy']:.4f}")

    # --- Mask IoU ---
    masks = cfg.get("masks", [])
    if not masks:
        _fail("Config must contain at least one mask pair.")
    for i, mask_pair in enumerate(masks):
        iou = mask_iou(mask_pair["pred_mask"], mask_pair["gt_mask"])
        _validate_iou(iou, i)
        print(f"  Mask IoU[{i}]: {iou:.4f}")

    # --- Localization activation recall ---
    loc_cfg = cfg.get("localization", {})
    loc_result = localization_activation_recall(
        loc_cfg["gt_class_labels"],
        loc_cfg["localization_states"],
    )
    _validate_loc_recall(loc_result)
    print(f"  Localization activation recall: {loc_result['activation_recall']:.4f} "
          f"({loc_result['n_activated']}/{loc_result['n_tampered']} tampered examples)")

    # --- Latency and FPS ---
    lat_values = cfg.get("latency_ms", [])
    if not lat_values:
        _fail("Config must contain latency_ms values.")
    lat_result = latency_summary(lat_values)
    _validate_latency_result(lat_result)
    fps_result = fps_summary(lat_values)
    _validate_fps_result(fps_result)
    print(f"  Latency mean: {lat_result['mean_ms']:.2f} ms  "
          f"FPS mean: {fps_result['mean_fps']:.2f}")

    # --- Robustness drop ---
    perturb_records = cfg.get("perturbation_records", [])
    if not perturb_records:
        _fail("Config must contain perturbation_records.")
    rob_result = robustness_summary(perturb_records)
    _validate_robustness_result(rob_result)
    print(f"  Robustness mean_drop: {rob_result['mean_drop']:.4f}  "
          f"max_drop: {rob_result['max_drop']:.4f}")

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
