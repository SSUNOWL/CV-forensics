#!/usr/bin/env python3
"""Plain Python tests for pre-SNS v3 GT-IoU tile builder."""

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

from cv_forensics.pre_sns_v3_gt_iou_tile_builder import (  # noqa: E402
    MARKER,
    artifact_manifest_schema_ok,
    build_tile_manifest,
    bucket_records,
    classify_failure_types,
    classify_iou_bucket,
    compute_mask_metrics,
    crop_box_centered,
    hard_negative_tile_records,
    jittered_crop_boxes,
    load_jsonl,
    negative_tile_records,
    positive_tile_records,
    run_gt_iou_tile_builder,
    enrich_loaded_mining_records,
    validate_gt_iou_tile_builder_config,
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
        "config_kind": "approved_pre_sns_v3_gt_iou_tile_builder",
        "execution_mode": "approved_local_pre_sns_v3_gt_iou_tile_builder",
        "approved_real_data_access": True,
        "train_manifest_path": str(root / "inputs" / "train.json"),
        "long256_checkpoint_path": str(root / "models" / "long256.pt"),
        "gt_iou_records_path": None,
        "hard_negative_records_path": str(root / "inputs" / "hard.jsonl"),
        "approved_input_roots": [str(root / "inputs")],
        "approved_checkpoint_roots": [str(root / "models")],
        "output_root": str(root / "reports" / "tile_builder"),
        "source_split": "train",
        "max_samples": 4,
        "tile_size": 4,
        "positive_jitter_count": 1,
        "negative_sample_count": 2,
        "negative_random_count": 1,
        "hard_negative_count": 2,
        "severe_oversample_factor": 2,
        "low_iou_oversample_factor": 2,
        "seed": 7,
        "mask_threshold": 0.5,
        "visual_top_n": 0,
        "run_long256_report": False,
        "write_cropped_images": False,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
    }


def test_metrics_and_empty_mask() -> None:
    gt = [1, 1, 0, 0]
    pred = [1, 0, 1, 0]
    metrics = compute_mask_metrics(gt, pred, (2, 2))
    assert_equal(round(metrics["iou"], 3), 0.333, "IoU")
    assert_equal(round(metrics["dice"], 3), 0.5, "Dice")
    empty = compute_mask_metrics(gt, [0, 0, 0, 0], (2, 2))
    assert_equal(empty["pred_area_pct"], 0.0, "empty pred area")
    assert_true("empty_prediction" in classify_failure_types(empty), "empty type")


def test_bucket_classification() -> None:
    assert_equal(classify_iou_bucket({"iou": 0.01}), "severe_iou_fail", "severe")
    assert_equal(classify_iou_bucket({"iou": 0.10}), "low_iou", "low")
    assert_equal(classify_iou_bucket({"iou": 0.20}), "weak_iou", "weak")
    assert_equal(classify_iou_bucket({"iou": 0.50}), "good_iou", "good")
    under = {"iou": 0.2, "gt_area_pct": 20.0, "pred_area_pct": 5.0, "pred_component_count": 1, "mask_area_ratio_pred_over_gt": 0.25}
    over = {"iou": 0.2, "gt_area_pct": 10.0, "pred_area_pct": 40.0, "pred_component_count": 1, "mask_area_ratio_pred_over_gt": 4.0}
    assert_true("undersegmented" in classify_failure_types(under), "underseg")
    assert_true("oversegmented" in classify_failure_types(over), "overseg")


def test_crop_boxes_and_positive_records() -> None:
    mask = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    boxes = jittered_crop_boxes(mask, (3, 3), 4, 1, __import__("random").Random(1))
    assert_equal(boxes[0], [0, 0, 3, 3], "boundary crop")
    assert_equal(crop_box_centered(0, 0, 10, 10, 4), [0, 0, 4, 4], "top-left crop")
    record = {
        "sample_id": "tampered0",
        "image_path": "/tmp/a.png",
        "mask_path": "/tmp/a_mask.png",
        "mining_bucket": "severe_iou_fail",
        "failure_types": ["severe_iou_fail"],
        "_gt_mask": mask,
        "_gt_shape": (3, 3),
    }
    pos = positive_tile_records(record, {"tile_size": 4, "positive_jitter_count": 1, "severe_oversample_factor": 2}, __import__("random").Random(1))
    assert_true(pos, "positive records")
    assert_equal(pos[0]["tile_class"], "positive_tampered", "positive class")
    assert_equal(pos[0]["tile_size"], 4, "positive tile size")
    assert_equal(pos[0]["expected_mask_type"], "cropped_gt_mask", "positive mask type")


def test_negative_and_hard_negative_records() -> None:
    rng = __import__("random").Random(2)
    samples = [
        {"sample_id": "real0", "class_label": "real", "image_path": "/tmp/r.png", "width": 8, "height": 8},
        {"sample_id": "syn0", "class_label": "full_synthetic", "image_path": "/tmp/s.png", "width": 8, "height": 8},
    ]
    neg = negative_tile_records(samples, {"tile_size": 4, "negative_random_count": 1}, rng)
    assert_equal({item["tile_class"] for item in neg}, {"negative_real", "negative_synthetic"}, "negative classes")
    assert_equal({item["tile_size"] for item in neg}, {4}, "negative tile sizes")
    hard = hard_negative_tile_records([{"sample_id": "h0", "image_path": "/tmp/h.png", "width": 8, "height": 8, "pred_bbox": {"x0": 6, "y0": 6, "x1": 8, "y1": 8}}], {"tile_size": 4, "hard_negative_count": 1}, rng)
    assert_equal(hard[0]["tile_class"], "hard_negative", "hard class")
    assert_equal(hard[0]["tile_size"], 4, "hard tile size")
    assert_equal(hard[0]["expected_mask_type"], "empty_mask", "hard empty")


def test_loaded_mining_record_enrichment() -> None:
    manifest = {"samples": [{"sample_id": "t0", "class_label": "tampered", "image_path": "/tmp/t.png", "gt_mask": [1, 0, 0, 0], "mask_size": [2, 2]}]}
    records = [{"sample_id": "t0", "iou": 0.01, "dice": 0.0}]
    enriched = enrich_loaded_mining_records(records, manifest, {}, [])
    assert_true(enriched[0]["_gt_mask"], "loaded record has gt mask")
    assert_equal(enriched[0]["mining_bucket"], "severe_iou_fail", "loaded bucket")


def test_validator_guardrails() -> None:
    root = temp_root("cvf_gt_tile_cfg_")
    cfg = safe_config(root)
    assert_equal(validate_gt_iou_tile_builder_config(cfg, require_exists=False), [], "safe config")
    for key in ("no_network", "no_download", "no_training", "no_sns_augmentation"):
        unsafe = dict(cfg)
        unsafe[key] = False
        assert_true(validate_gt_iou_tile_builder_config(unsafe, require_exists=False), f"{key} rejected")
    repo_out = dict(cfg)
    repo_out["output_root"] = str(REPO_ROOT / "tile_builder")
    assert_true(validate_gt_iou_tile_builder_config(repo_out, require_exists=False), "repo output rejected")
    val_split = dict(cfg)
    val_split["source_split"] = "validation_hard_cases"
    assert_true(validate_gt_iou_tile_builder_config(val_split, require_exists=False), "validation split rejected")


def test_run_builder_no_repo_writes_and_artifact_schema() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_gt_tile_run_")
    (root / "inputs").mkdir(parents=True)
    (root / "models").mkdir(parents=True)
    (root / "models" / "long256.pt").write_text("fixture", encoding="utf-8")
    hard_path = root / "inputs" / "hard.jsonl"
    hard_path.write_text(json.dumps({"sample_id": "hard0", "image_path": str(root / "inputs" / "real.png"), "width": 4, "height": 4}) + "\n", encoding="utf-8")
    manifest = {
        "samples": [
            {"sample_id": "t0", "class_label": "tampered", "image_path": str(root / "inputs" / "t0.png"), "width": 4, "height": 4, "gt_mask": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "baseline_mask": [0] * 16},
            {"sample_id": "r0", "class_label": "real", "image_path": str(root / "inputs" / "r0.png"), "width": 4, "height": 4},
            {"sample_id": "s0", "class_label": "full_synthetic", "image_path": str(root / "inputs" / "s0.png"), "width": 4, "height": 4},
        ]
    }
    (root / "inputs" / "train.json").write_text(json.dumps(manifest), encoding="utf-8")
    cfg = safe_config(root)
    summary = run_gt_iou_tile_builder(cfg)
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "no repo root writes")
    artifact_path = root / "reports" / "tile_builder" / "artifact_manifest.json"
    assert_true(artifact_path.is_file(), "artifact manifest exists")
    artifact = json.loads(artifact_path.read_text())
    assert_true(artifact_manifest_schema_ok(artifact), "artifact schema")
    assert_equal(artifact["output_paths"]["artifact_manifest"], str(artifact_path), "artifact self path")
    assert_equal(summary["marker"], MARKER, "summary marker")
    tile_manifest = json.loads((root / "reports" / "tile_builder" / "tile_localization_manifest.json").read_text())
    assert_true(tile_manifest["counts_by_tile_class"]["positive_tampered"] > 0, "positive count")
    assert_true(all(record["tile_size"] == cfg["tile_size"] for record in tile_manifest["records"]), "manifest record tile sizes")
    assert_true(load_jsonl(root / "reports" / "tile_builder" / "gt_iou_train_records.jsonl"), "jsonl records")


def main() -> int:
    tests = [
        test_metrics_and_empty_mask,
        test_bucket_classification,
        test_crop_boxes_and_positive_records,
        test_negative_and_hard_negative_records,
        test_loaded_mining_record_enrichment,
        test_validator_guardrails,
        test_run_builder_no_repo_writes_and_artifact_schema,
    ]
    for test in tests:
        test()
    print("PRE_SNS_V3_GT_IOU_TILE_BUILDER_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
