#!/usr/bin/env python3
"""Plain Python tests for pre-SNS v3 final visual audit."""

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

from cv_forensics.pre_sns_v3_final_visual_audit import (  # noqa: E402
    MARKER,
    aggregate_metrics,
    build_gallery_manifest,
    recommendation,
    red_mask_bucket,
    run_final_visual_audit,
    validate_final_visual_audit_config,
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


def write_fixture_inputs(root: Path) -> Path:
    from PIL import Image, ImageDraw

    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, label in enumerate(("tampered", "real", "full_synthetic")):
        image_path = inputs / f"{label}.png"
        image = Image.new("RGB", (10, 8), (40 + index * 30, 60, 80))
        draw = ImageDraw.Draw(image)
        draw.rectangle((2, 2, 5, 5), fill=(220, 40, 40))
        image.save(image_path)
        sample = {"sample_id": label, "class_label": label, "image_path": str(image_path)}
        if label == "tampered":
            mask_path = inputs / "tampered_mask.png"
            mask = Image.new("L", (10, 8), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rectangle((2, 2, 5, 5), fill=255)
            mask.save(mask_path)
            sample["mask_path"] = str(mask_path)
            fixture_mask = [1 if 2 <= (idx % 10) <= 5 and 2 <= (idx // 10) <= 5 else 0 for idx in range(80)]
            sample["fixture_long256_report"] = {"class": "tampered", "family": "Other", "tampered_score": 0.9, "baseline_mask": fixture_mask}
            sample["fixture_tile_mask"] = fixture_mask
        else:
            sample["fixture_long256_report"] = {"class": label, "family": "Real-or-N/A", "tampered_score": 0.01, "baseline_mask": [0] * 80}
            sample["fixture_tile_mask"] = [0] * 80
        samples.append(sample)
    path = inputs / "samples.json"
    path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return path


def safe_config(root: Path, sample_list: Path) -> dict[str, object]:
    ckpt = root / "ckpts"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "long256.pt").write_text("fixture", encoding="utf-8")
    (ckpt / "tile.pt").write_text("fixture", encoding="utf-8")
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_v3_final_visual_audit",
        "execution_mode": "approved_local_pre_sns_v3_final_visual_audit",
        "approved_real_data_access": True,
        "manifest_path": None,
        "sample_list_path": str(sample_list),
        "hard_cases_path": None,
        "long256_checkpoint_path": str(ckpt / "long256.pt"),
        "tile_localizer_checkpoint_path": str(ckpt / "tile.pt"),
        "approved_input_roots": [str(root / "inputs")],
        "approved_checkpoint_roots": [str(ckpt)],
        "output_root": str(root / "audit_out"),
        "max_samples": 3,
        "balance_classes": True,
        "per_class_max": 1,
        "seed": 45,
        "device": "cpu",
        "tile_size": 4,
        "tile_stride": 4,
        "tile_activation_tau": 0.5,
        "mask_threshold": 0.5,
        "max_tiles": 8,
        "aggregation_mode": "average",
        "max_final_mask_area_pct": 35.0,
        "min_final_mask_area_pct": 0.0,
        "max_tile_vs_baseline_area_ratio": 4.0,
        "require_tile_improves_gt_when_gt_available": False,
        "fallback_to_baseline_when_tile_unreliable": True,
        "suppress_mask_for_non_tampered": True,
        "ready_macro_f1_threshold": 0.85,
        "ready_tampered_recall_threshold": 0.85,
        "ready_real_fpr_threshold": 0.1,
        "ready_failed_red_mask_rate_threshold": 0.2,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }


def test_metric_aggregation_and_buckets() -> None:
    good = {"gt_class": "tampered", "pred_class": "tampered", "final_mask_area_pct": 5.0, "tile_localization_activated": True, "localized_evidence_status": "localized_evidence_present", "gt_comparison": {"baseline_long256_iou": 0.3, "baseline_long256_dice": 0.4, "tile_final_iou": 0.5, "tile_final_dice": 0.6, "iou_delta": 0.2}}
    acceptable = {**good, "gt_comparison": {**good["gt_comparison"], "tile_final_iou": 0.3}}
    failed = {**good, "final_mask_area_pct": 0.0, "gt_comparison": {**good["gt_comparison"], "tile_final_iou": 0.1}}
    real = {"gt_class": "real", "pred_class": "real", "final_mask_area_pct": 0.0, "tile_localization_activated": False, "localized_evidence_status": "skipped_non_tampered", "gt_comparison": {}}
    assert_equal(red_mask_bucket(good), "good", "good bucket")
    assert_equal(red_mask_bucket(acceptable), "acceptable", "acceptable bucket")
    assert_equal(red_mask_bucket(failed), "failed", "failed bucket")
    metrics = aggregate_metrics([good, acceptable, failed, real], {"ready_failed_red_mask_rate_threshold": 0.2})
    assert_equal(metrics["good_red_mask_count"], 1, "good count")
    assert_equal(metrics["acceptable_red_mask_count"], 1, "acceptable count")
    assert_equal(metrics["failed_red_mask_count"], 1, "failed count")
    assert_true(metrics["tampered_recall"] == 1.0, "tampered recall")


def test_recommendation_policy_and_gallery() -> None:
    ready = {"macro_f1": 0.9, "tampered_recall": 0.9, "real_false_positive_rate": 0.0, "failed_red_mask_rate": 0.0}
    assert_equal(recommendation(ready, {}), "ready_for_sns_robustness_evaluation", "ready recommendation")
    more = dict(ready)
    more["failed_red_mask_rate"] = 0.5
    assert_equal(recommendation(more, {}), "needs_more_tile_localization_training", "more training recommendation")
    manual = dict(ready)
    manual["macro_f1"] = 0.1
    assert_equal(recommendation(manual, {}), "needs_manual_review", "manual recommendation")
    gallery = build_gallery_manifest([{"sample_id": "a", "gt_class": "real", "pred_class": "real", "red_mask_bucket": "not_tampered", "visual_paths": {"input_image": "/x"}}])
    assert_equal(gallery["marker"], MARKER, "gallery marker")
    assert_equal(gallery["record_count"], 1, "gallery count")


def test_validator_guardrails() -> None:
    example = json.loads((REPO_ROOT / "configs" / "evaluation" / "pre_sns_v3_final_visual_audit.example.json").read_text())
    assert_equal(validate_final_visual_audit_config(example, require_exists=False), [], "example validates")
    root = temp_root("cvf_final_audit_validator_")
    sample_list = write_fixture_inputs(root)
    cfg = safe_config(root, sample_list)
    assert_equal(validate_final_visual_audit_config(cfg, require_exists=False), [], "safe config")
    unsafe = dict(cfg)
    unsafe["output_root"] = str(REPO_ROOT / "audit")
    assert_true(validate_final_visual_audit_config(unsafe, require_exists=False), "repo output rejected")
    for key in ("no_network", "no_download", "no_training", "no_sns_augmentation"):
        unsafe = dict(cfg)
        unsafe[key] = False
        assert_true(validate_final_visual_audit_config(unsafe, require_exists=False), f"{key} rejected")


def test_run_audit_no_repo_writes() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_final_audit_run_")
    sample_list = write_fixture_inputs(root)
    summary = run_final_visual_audit(safe_config(root, sample_list))
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "no repo writes")
    assert_equal(summary["marker"], MARKER, "summary marker")
    out = root / "audit_out"
    for name in (
        "final_audit_records.jsonl",
        "final_audit_summary.json",
        "confusion_matrix.json",
        "good_red_mask_cases.json",
        "acceptable_red_mask_cases.json",
        "failed_red_mask_cases.json",
        "false_positive_real_cases.json",
        "false_negative_tampered_cases.json",
        "needs_manual_review_cases.json",
        "red_mask_gallery_manifest.json",
        "artifact_manifest.json",
    ):
        assert_true((out / name).is_file(), f"{name} exists")
    assert_true((out / "tampered" / "comparison_sheet.jpg").is_file(), "case sheet exists")


def main() -> int:
    tests = [
        test_metric_aggregation_and_buckets,
        test_recommendation_policy_and_gallery,
        test_validator_guardrails,
        test_run_audit_no_repo_writes,
    ]
    for test in tests:
        test()
    print("PRE_SNS_V3_FINAL_VISUAL_AUDIT_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

