#!/usr/bin/env python3
"""Plain Python tests for the pre-SNS long256 + tile integrated report."""

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

from cv_forensics.pre_sns_v3_long256_tile_report import (  # noqa: E402
    MARKER,
    baseline_vs_tile_metrics,
    build_record,
    build_summary,
    reason_text,
    report_schema_ok,
    run_long256_tile_report,
    select_final_mask,
    summary_schema_ok,
    validate_long256_tile_report_config,
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
        "config_kind": "approved_pre_sns_v3_long256_tile_report",
        "execution_mode": "approved_local_pre_sns_v3_long256_tile_report",
        "manifest_path": str(root / "inputs" / "manifest.json"),
        "gt_iou_mining_records_path": str(root / "inputs" / "records.jsonl"),
        "long256_checkpoint_path": str(root / "models" / "long256.pt"),
        "approved_input_roots": [str(root / "inputs")],
        "approved_checkpoint_roots": [str(root / "models")],
        "output_root": str(root / "reports" / "integrated"),
        "max_samples": 5,
        "tile_size": 2,
        "tile_stride": 1,
        "crop_context_px": 1,
        "high_res_max_size": 4,
        "merge_mode": "union",
        "tampered_suspect_threshold": 0.5,
        "improvement_epsilon": 0.0,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
    }


def fixture_sample(root: Path) -> dict[str, object]:
    return {
        "sample_id": "case0",
        "image_path": str(root / "inputs" / "case0.png"),
        "width": 2,
        "height": 2,
        "long256_class": "tampered",
        "tampered_score": 0.91,
        "family": "Other",
        "class_conf": {"tampered": 0.91, "real": 0.04, "full_synthetic": 0.05},
        "family_conf": {"Other": 1.0},
        "gt_mask": [1, 0, 0, 0],
        "baseline_mask": [0, 0, 0, 0],
        "tile_mask": [1, 0, 0, 0],
    }


def test_validator_guardrails() -> None:
    root = temp_root("cvf_l256_tile_cfg_")
    cfg = safe_config(root)
    assert_equal(validate_long256_tile_report_config(cfg, require_exists=False), [], "safe config")
    unsafe = dict(cfg)
    unsafe["no_training"] = False
    assert_true(validate_long256_tile_report_config(unsafe, require_exists=False), "training flag must fail")
    unsafe_output = dict(cfg)
    unsafe_output["output_root"] = str(REPO_ROOT / "integrated")
    assert_true(validate_long256_tile_report_config(unsafe_output, require_exists=False), "repo output must fail")
    unsafe_path = dict(cfg)
    unsafe_path["manifest_path"] = str(root / "inputs" / ".." / "manifest.json")
    assert_true(validate_long256_tile_report_config(unsafe_path, require_exists=False), "path traversal must fail")


def test_mask_selection_and_metrics() -> None:
    baseline = [1, 0, 0, 0]
    tile = [0, 1, 0, 0]
    assert_equal(select_final_mask(baseline, tile, {"activate": False})["source"], "baseline_long256", "skip selects baseline")
    assert_equal(select_final_mask(baseline, tile, {"activate": True, "uncertain": False})["source"], "tile_localized", "active tile selects tile")
    empty = select_final_mask(baseline, [0, 0, 0, 0], {"activate": True, "uncertain": False})
    assert_equal(empty["source"], "baseline_long256_tile_empty", "empty tile falls back")
    assert_true(empty["uncertain"], "empty tile fallback uncertain")
    metrics = baseline_vs_tile_metrics([1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], (2, 2))
    assert_equal(metrics["localization_delta"], "improved", "tile should improve")


def test_record_preserves_primary_class_and_reason() -> None:
    root = temp_root("cvf_l256_tile_record_")
    cfg = safe_config(root)
    record = build_record(fixture_sample(root), cfg)
    assert_true(report_schema_ok(record), "record schema")
    assert_equal(record["class"], "tampered", "primary class")
    assert_true(record["primary_class_unchanged"], "primary class unchanged")
    assert_equal(record["final_mask_source"], "tile_localized", "tile source")
    assert_true("tile localization" in record["reason"], "reason mentions tile localization")
    assert_true("baseline" in reason_text("tampered", 0.9, "baseline_long256_tile_empty", "unknown_no_gt", "Other"), "fallback reason")


def test_summary_schemas() -> None:
    root = temp_root("cvf_l256_tile_summary_")
    cfg = safe_config(root)
    records = [build_record(fixture_sample(root), cfg)]
    summary = build_summary(cfg, records)
    assert_true(summary_schema_ok(summary), "summary schema")
    assert_equal(summary["activation_counts"]["activated"], 1, "activation count")


def test_no_repo_writes_manifest_and_single_image() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_l256_tile_run_")
    (root / "inputs").mkdir(parents=True)
    (root / "models").mkdir(parents=True)
    (root / "models" / "long256.pt").write_text("fixture", encoding="utf-8")
    sample = fixture_sample(root)
    (root / "inputs" / "manifest.json").write_text(json.dumps({"samples": [sample]}), encoding="utf-8")
    (root / "inputs" / "records.jsonl").write_text(json.dumps({"sample_id": "case0"}) + "\n", encoding="utf-8")
    cfg = safe_config(root)
    summary = run_long256_tile_report(cfg)
    assert_true(summary_schema_ok(summary), "manifest summary")
    assert_true((root / "reports" / "integrated" / "long256_tile_integrated_records.jsonl").is_file(), "records output")

    single_root = temp_root("cvf_l256_tile_single_")
    (single_root / "inputs").mkdir(parents=True)
    (single_root / "models").mkdir(parents=True)
    (single_root / "models" / "long256.pt").write_text("fixture", encoding="utf-8")
    single_cfg = safe_config(single_root)
    single_cfg.pop("manifest_path")
    single_cfg.pop("max_samples")
    single_cfg["image_path"] = str(single_root / "inputs" / "single.png")
    single_cfg["fixture_sample"] = fixture_sample(single_root)
    (single_root / "inputs" / "single.png").write_text("fixture", encoding="utf-8")
    (single_root / "inputs" / "records.jsonl").write_text("", encoding="utf-8")
    single_report = run_long256_tile_report(single_cfg)
    assert_true(report_schema_ok(single_report), "single report schema")
    assert_true((single_root / "reports" / "integrated" / "long256_tile_integrated_report.json").is_file(), "single report output")
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "no repo root writes")
    assert_equal(single_report["marker"], MARKER, "marker")


def main() -> int:
    tests = [
        test_validator_guardrails,
        test_mask_selection_and_metrics,
        test_record_preserves_primary_class_and_reason,
        test_summary_schemas,
        test_no_repo_writes_manifest_and_single_image,
    ]
    for test in tests:
        test()
    print("PRE_SNS_V3_LONG256_TILE_REPORT_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
