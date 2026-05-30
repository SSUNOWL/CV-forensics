#!/usr/bin/env python3
"""Plain Python tests for pre-SNS v3 GT-IoU localization mining."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TMP_PARENT = Path("/home/rlatjswo/.codex/memories")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_gt_iou_mining import (  # noqa: E402
    MARKER,
    build_buckets,
    classify_model_failure,
    classify_record_failure,
    compute_model_metrics,
    default_thresholds,
    iou_score,
    dice_score,
    summary_schema_ok,
    validate_gt_iou_mining_config,
    write_mining_outputs,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_close(actual: float, expected: float, message: str) -> None:
    if abs(actual - expected) > 1e-9:
        raise AssertionError(f"{message}: expected {expected}, got {actual}")


def safe_config(root: str) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_v3_gt_iou_localization_mining",
        "execution_mode": "approved_local_pre_sns_v3_gt_iou_localization_mining",
        "manifest_path": f"{root}/inputs/manifest.json",
        "long256_checkpoint_path": f"{root}/models/long256.pt",
        "approved_input_roots": [f"{root}/inputs"],
        "approved_checkpoint_roots": [f"{root}/models"],
        "output_root": f"{root}/reports/gt_iou",
        "max_samples": 3,
        "threshold_tau": 0.5,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
    }


def test_iou_and_dice() -> None:
    gt = [1, 1, 0, 0]
    pred = [1, 0, 1, 0]
    assert_close(iou_score(gt, pred), 1.0 / 3.0, "IoU should use intersection over union")
    assert_close(dice_score(gt, pred), 0.5, "Dice should use 2TP/(GT+pred)")


def test_empty_mask_metrics() -> None:
    metrics = compute_model_metrics([1, 0, 0, 0], [0, 0, 0, 0], (2, 2))
    failures = classify_model_failure(metrics, default_thresholds())
    assert_close(metrics["iou"], 0.0, "empty prediction IoU")
    assert_close(metrics["dice"], 0.0, "empty prediction Dice")
    assert_true("empty_prediction" in failures, "empty prediction should be classified")


def test_over_under_segmented_classification() -> None:
    thresholds = default_thresholds()
    under = compute_model_metrics([1, 1, 1, 1], [1, 0, 0, 0], (2, 2))
    over = compute_model_metrics([1, 0, 0, 0], [1, 1, 1, 0], (2, 2))
    assert_true("undersegmented" in classify_model_failure(under, thresholds), "small pred/GT ratio should undersegment")
    assert_true("oversegmented" in classify_model_failure(over, thresholds), "large pred/GT ratio should oversegment")


def test_cross_model_buckets() -> None:
    thresholds = default_thresholds()
    record = {
        "sample_id": "a",
        "models": {
            "long256": {"available": True, "iou": 0.1, "failure_types": ["low_iou_wrong_region"]},
            "long224": {"available": True, "iou": 0.7, "failure_types": ["long256_succeeded"]},
            "refined": {"available": False},
        },
    }
    record["failure_types"] = classify_record_failure(record, thresholds)
    buckets = build_buckets([record], thresholds)
    assert_true("long256_failed_long224_succeeded" in record["failure_types"], "long224 rescue bucket missing")
    assert_true(len(buckets["long256_failed_long224_succeeded"]) == 1, "long224 rescue bucket count")

    failed = {
        "sample_id": "b",
        "models": {
            "long256": {"available": True, "iou": 0.1, "failure_types": ["empty_prediction"]},
            "long224": {"available": True, "iou": 0.2, "failure_types": ["low_iou_wrong_region"]},
            "refined": {"available": True, "iou": 0.0, "failure_types": ["empty_prediction"]},
        },
    }
    failed["failure_types"] = classify_record_failure(failed, thresholds)
    assert_true("all_models_failed" in failed["failure_types"], "all models failed bucket missing")
    assert_true(len(build_buckets([failed], thresholds)["all_models_failed"]) == 1, "all models failed bucket count")


def test_validator_guardrails() -> None:
    TMP_PARENT.mkdir(parents=True, exist_ok=True)
    root = tempfile.mkdtemp(prefix="cvf_gtiou_cfg_", dir=str(TMP_PARENT))
    cfg = safe_config(root)
    assert_true(validate_gt_iou_mining_config(cfg, require_exists=False) == [], "safe config should validate")
    unsafe = dict(cfg)
    unsafe["no_training"] = False
    assert_true(validate_gt_iou_mining_config(unsafe, require_exists=False), "validator must reject training flag")
    unsafe_output = dict(cfg)
    unsafe_output["output_root"] = str(REPO_ROOT / "tmp_gt_iou")
    assert_true(validate_gt_iou_mining_config(unsafe_output, require_exists=False), "validator must reject repo output")
    unsafe_path = dict(cfg)
    unsafe_path["manifest_path"] = f"{root}/inputs/../manifest.json"
    assert_true(validate_gt_iou_mining_config(unsafe_path, require_exists=False), "validator must reject traversal")


def test_no_repo_writes_and_summary_schema() -> None:
    before = set(os.listdir(REPO_ROOT))
    TMP_PARENT.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="cvf_gtiou_out_", dir=str(TMP_PARENT)))
    output_root = root / "reports"
    cfg = safe_config(str(root))
    record = {
        "sample_id": "case0",
        "image_path": f"{root}/inputs/case0.png",
        "mask_path": f"{root}/inputs/case0_mask.png",
        "iou": 0.0,
        "dice": 0.0,
        "gt_area_pct": 25.0,
        "pred_area_pct": 0.0,
        "gt_component_count": 1,
        "pred_component_count": 0,
        "largest_gt_component_area_pct": 25.0,
        "largest_pred_component_area_pct": 0.0,
        "mask_area_ratio_pred_over_gt": 0.0,
        "failure_types": ["empty_prediction", "all_models_failed"],
        "models": {
            "long256": {"available": True, "iou": 0.0, "dice": 0.0, "failure_types": ["empty_prediction"]},
            "long224": {"available": False},
            "refined": {"available": False},
        },
    }
    summary = write_mining_outputs(output_root, [record], cfg)
    after = set(os.listdir(REPO_ROOT))
    assert_true(before == after, "write_mining_outputs must not write repository root files")
    assert_true(summary_schema_ok(summary), "summary schema should be valid")
    assert_true(summary["marker"] == MARKER, "summary marker")
    assert_true((output_root / "localization_iou_records.jsonl").is_file(), "JSONL output")
    with open(output_root / "localization_mining_summary.json", "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert_true(saved["marker"] == MARKER, "saved summary marker")


def main() -> int:
    tests = [
        test_iou_and_dice,
        test_empty_mask_metrics,
        test_over_under_segmented_classification,
        test_cross_model_buckets,
        test_validator_guardrails,
        test_no_repo_writes_and_summary_schema,
    ]
    for test in tests:
        test()
    print("PRE_SNS_V3_GT_IOU_LOCALIZATION_MINING_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
