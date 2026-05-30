#!/usr/bin/env python3
"""Plain Python tests for pre-SNS v3 tile localization."""

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

from cv_forensics.pre_sns_v3_tile_localization import (  # noqa: E402
    MARKER,
    activation_decision,
    build_evaluation_summary,
    build_tile_training_plan,
    compare_masks,
    evaluation_summary_schema_ok,
    expand_box,
    merge_tile_masks,
    run_tile_localization_evaluation,
    run_tile_training_prepare,
    tile_grid,
    training_plan_schema_ok,
    validate_tile_localization_config,
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
        "config_kind": "approved_pre_sns_v3_tile_localization",
        "execution_mode": "approved_local_pre_sns_v3_tile_localization",
        "manifest_path": str(root / "inputs" / "manifest.json"),
        "gt_iou_mining_records_path": str(root / "inputs" / "records.jsonl"),
        "long256_checkpoint_path": str(root / "models" / "long256.pt"),
        "approved_input_roots": [str(root / "inputs")],
        "approved_checkpoint_roots": [str(root / "models")],
        "output_root": str(root / "reports" / "tile"),
        "max_samples": 3,
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
        "dry_run_only": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
    }


def test_validator_guardrails() -> None:
    root = temp_root("cvf_tile_cfg_")
    cfg = safe_config(root)
    assert_equal(validate_tile_localization_config(cfg, require_exists=False), [], "safe config")
    unsafe = dict(cfg)
    unsafe["dry_run_only"] = False
    assert_true(validate_tile_localization_config(unsafe, require_exists=False), "dry_run_only false must fail")
    unsafe_output = dict(cfg)
    unsafe_output["output_root"] = str(REPO_ROOT / "tile_report")
    assert_true(validate_tile_localization_config(unsafe_output, require_exists=False), "repo output must fail")
    unsafe_path = dict(cfg)
    unsafe_path["manifest_path"] = str(root / "inputs" / ".." / "manifest.json")
    assert_true(validate_tile_localization_config(unsafe_path, require_exists=False), "traversal must fail")


def test_tile_grid_and_context() -> None:
    tiles = tile_grid(5, 4, 3, 2)
    assert_equal(tiles[0], {"x0": 0, "y0": 0, "x1": 3, "y1": 3}, "first tile")
    assert_true(any(tile["x1"] == 5 and tile["y1"] == 4 for tile in tiles), "grid must cover bottom-right")
    expanded = expand_box({"x0": 1, "y0": 1, "x1": 3, "y1": 3}, 4, 4, 2)
    assert_equal(expanded, {"x0": 0, "y0": 0, "x1": 4, "y1": 4}, "context clips")


def test_merge_and_activation() -> None:
    merged = merge_tile_masks(
        3,
        3,
        [
            {"tile": {"x0": 0, "y0": 0, "x1": 2, "y1": 2}, "mask": [1, 0, 0, 1]},
            {"tile": {"x0": 1, "y0": 1, "x1": 3, "y1": 3}, "mask": [1, 1, 0, 0]},
        ],
    )
    assert_equal(merged, [1, 0, 0, 0, 1, 1, 0, 0, 0], "union merge")
    assert_true(activation_decision("tampered", 0.1, 0.5)["activate"], "tampered activates")
    suspect = activation_decision("real", 0.8, 0.5)
    assert_true(suspect["activate"] and suspect["uncertain"], "high-score non-tampered activates uncertain")
    assert_true(not activation_decision("real", 0.2, 0.5)["activate"], "low-score real skips")


def test_compare_masks_and_summaries() -> None:
    gt = [1, 0, 0, 0]
    baseline = [0, 0, 0, 0]
    tile = [1, 0, 0, 0]
    comparison = compare_masks(gt, baseline, tile, (2, 2))
    assert_equal(comparison["localization_delta"], "improved", "tile should improve")
    summary = build_evaluation_summary({"tile_size": 2, "tile_stride": 1, "crop_context_px": 1, "high_res_max_size": 4}, [{"activation": {"activate": True}, **comparison}])
    assert_true(evaluation_summary_schema_ok(summary), "evaluation summary schema")


def test_training_plan_schema() -> None:
    root = temp_root("cvf_tile_plan_")
    cfg = safe_config(root)
    manifest = {"samples": [{"sample_id": "a", "image_path": "/x/a.png", "width": 4, "height": 4}]}
    records = [{"sample_id": "a", "failure_types": ["low_iou_wrong_region"]}]
    plan = build_tile_training_plan(cfg, manifest, records)
    assert_true(training_plan_schema_ok(plan), "plan schema")
    assert_equal(plan["planned_sample_count"], 1, "sample count")
    assert_true(plan["total_tile_count"] > 0, "tile count")


def test_no_repo_writes_with_entrypoints() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_tile_run_")
    (root / "inputs").mkdir(parents=True)
    (root / "models").mkdir(parents=True)
    (root / "models" / "long256.pt").write_text("fixture", encoding="utf-8")
    manifest = {
        "samples": [
            {
                "sample_id": "case0",
                "image_path": str(root / "inputs" / "case0.png"),
                "width": 2,
                "height": 2,
                "class_label": "tampered",
                "long256_class": "tampered",
                "tampered_score": 0.9,
                "gt_mask": [1, 0, 0, 0],
                "baseline_mask": [0, 0, 0, 0],
                "tile_mask": [1, 0, 0, 0],
            }
        ]
    }
    (root / "inputs" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "inputs" / "records.jsonl").write_text(json.dumps({"sample_id": "case0", "failure_types": ["empty_prediction"]}) + "\n", encoding="utf-8")
    cfg = safe_config(root)
    prep = run_tile_training_prepare(cfg)
    eval_summary = run_tile_localization_evaluation(cfg)
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "entrypoints must not write repo root files")
    assert_true(training_plan_schema_ok(prep), "prepare schema")
    assert_true(evaluation_summary_schema_ok(eval_summary), "eval schema")
    assert_true((root / "reports" / "tile" / "tile_training_plan_summary.json").is_file(), "plan summary file")
    assert_true((root / "reports" / "tile" / "tile_vs_long256_summary.json").is_file(), "eval summary file")
    assert_equal(eval_summary["marker"], MARKER, "marker")


def main() -> int:
    tests = [
        test_validator_guardrails,
        test_tile_grid_and_context,
        test_merge_and_activation,
        test_compare_masks_and_summaries,
        test_training_plan_schema,
        test_no_repo_writes_with_entrypoints,
    ]
    for test in tests:
        test()
    print("PRE_SNS_V3_TILE_LOCALIZATION_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
