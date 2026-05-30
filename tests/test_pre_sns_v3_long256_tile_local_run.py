#!/usr/bin/env python3
"""Plain Python tests for pre-SNS long256 tile local run."""

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

from cv_forensics.pre_sns_v3_long256_tile_local_run import (  # noqa: E402
    MARKER,
    bucket_record,
    bucket_records,
    build_sample_plan,
    gallery_schema_ok,
    recommendation_from_buckets,
    run_local_run,
    summary_schema_ok,
    validate_local_run_config,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def temp_root(prefix: str) -> Path:
    TMP_PARENT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(TMP_PARENT)))


def safe_config(root: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_v3_long256_tile_local_run",
        "execution_mode": "approved_local_pre_sns_v3_long256_tile_local_run",
        "manifest_path": str(root / "inputs" / "manifest.json"),
        "gt_iou_mining_records_path": str(root / "inputs" / "records.jsonl"),
        "long256_checkpoint_path": str(root / "models" / "long256.pt"),
        "approved_input_roots": [str(root / "inputs")],
        "approved_checkpoint_roots": [str(root / "models")],
        "output_root": str(root / "reports" / "local_run"),
        "max_samples": 3,
        "tile_size": 2,
        "tile_stride": 1,
        "crop_context_px": 1,
        "high_res_max_size": 4,
        "merge_mode": "union",
        "tampered_suspect_threshold": 0.5,
        "good_iou_threshold": 0.4,
        "good_dice_threshold": 0.5,
        "low_iou_threshold": 0.15,
        "overseg_ratio_threshold": 2.0,
        "underseg_ratio_threshold": 0.5,
        "visual_top_n": 0,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
    }


def sample(root: Path) -> dict[str, object]:
    return {
        "sample_id": "case0",
        "image_path": str(root / "inputs" / "case0.png"),
        "width": 2,
        "height": 2,
        "long256_class": "tampered",
        "tampered_score": 0.9,
        "family": "Other",
        "gt_mask": [1, 0, 0, 0],
        "baseline_mask": [0, 0, 0, 0],
        "tile_mask": [1, 0, 0, 0],
    }


def test_validator_guardrails() -> None:
    root = temp_root("cvf_local_run_cfg_")
    cfg = safe_config(root)
    assert_equal(validate_local_run_config(cfg, require_exists=False), [], "safe config")
    unsafe = dict(cfg)
    unsafe["no_network"] = False
    assert_true(validate_local_run_config(unsafe, require_exists=False), "network flag must fail")
    unsafe_out = dict(cfg)
    unsafe_out["output_root"] = str(REPO_ROOT / "local_run")
    assert_true(validate_local_run_config(unsafe_out, require_exists=False), "repo output must fail")
    unsafe_path = dict(cfg)
    unsafe_path["manifest_path"] = str(root / "inputs" / ".." / "manifest.json")
    assert_true(validate_local_run_config(unsafe_path, require_exists=False), "traversal must fail")


def test_sample_plan_and_buckets() -> None:
    root = temp_root("cvf_local_run_plan_")
    cfg = safe_config(root)
    plan = build_sample_plan(cfg, [sample(root)])
    assert_equal(plan[0]["sample_id"], "case0", "sample id")
    good_record = {
        "class": "tampered",
        "tile_localization_status": "activated_tampered",
        "final_mask_source": "tile_localized",
        "final_mask_stats": {"mask_area_pct": 25.0},
        "baseline_vs_tile_metrics": {"tile_iou": 1.0, "tile_dice": 1.0, "baseline_long256_area_pct": 0.0},
    }
    failed_record = {
        "class": "tampered",
        "tile_localization_status": "activated_tampered",
        "final_mask_source": "baseline_long256_tile_empty",
        "final_mask_stats": {"mask_area_pct": 0.0},
        "baseline_vs_tile_metrics": {"tile_iou": 0.0, "tile_dice": 0.0, "baseline_long256_area_pct": 0.0},
    }
    assert_equal(bucket_record(good_record, cfg), "good_red_mask_cases", "good bucket")
    assert_equal(bucket_record(failed_record, cfg), "failed_red_mask_cases", "failed bucket")
    buckets = bucket_records([good_record, failed_record], cfg)
    assert_equal(len(buckets["good_red_mask_cases"]), 1, "good count")
    assert_equal(recommendation_from_buckets({"good_red_mask_cases": [good_record], "failed_red_mask_cases": [], "needs_manual_review_cases": []}), "ready_for_sns_robustness_evaluation", "ready recommendation")
    assert_equal(recommendation_from_buckets(buckets), "needs_real_tile_localization_training", "training recommendation")


def test_no_repo_writes_and_outputs() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_local_run_exec_")
    (root / "inputs").mkdir(parents=True)
    (root / "models").mkdir(parents=True)
    (root / "models" / "long256.pt").write_text("fixture", encoding="utf-8")
    (root / "inputs" / "manifest.json").write_text(json.dumps({"samples": [sample(root)]}), encoding="utf-8")
    (root / "inputs" / "records.jsonl").write_text("", encoding="utf-8")
    cfg = safe_config(root)
    summary = run_local_run(cfg)
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "no repo root writes")
    assert_true(summary_schema_ok(summary), "summary schema")
    assert_true((root / "reports" / "local_run" / "local_run_records.jsonl").is_file(), "records")
    assert_true((root / "reports" / "local_run" / "good_red_mask_cases.json").is_file(), "good cases")
    with open(root / "reports" / "local_run" / "red_mask_gallery_manifest.json", "r", encoding="utf-8") as handle:
        gallery = json.load(handle)
    assert_true(gallery_schema_ok(gallery), "gallery schema")
    assert_equal(summary["marker"], MARKER, "marker")


def main() -> int:
    tests = [test_validator_guardrails, test_sample_plan_and_buckets, test_no_repo_writes_and_outputs]
    for test in tests:
        test()
    print("PRE_SNS_V3_LONG256_TILE_LOCAL_RUN_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
