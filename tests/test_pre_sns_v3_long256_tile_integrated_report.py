#!/usr/bin/env python3
"""Plain Python tests for pre-SNS long256 + tile-localizer integrated report."""

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

from cv_forensics.pre_sns_v3_long256_tile_integrated_report import (  # noqa: E402
    MARKER,
    aggregate_tile_values,
    build_integrated_record,
    compare_with_gt,
    run_integrated_report,
    threshold_mask,
    tile_reliability_reasons,
    tile_grid,
    validate_integrated_report_config,
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


def write_fixture_image(root: Path) -> tuple[Path, Path]:
    from PIL import Image, ImageDraw

    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    image_path = inputs / "image.png"
    mask_path = inputs / "gt.png"
    image = Image.new("RGB", (10, 7), (40, 60, 80))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 5, 5), fill=(220, 30, 30))
    image.save(image_path)
    mask = Image.new("L", (10, 7), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle((2, 2, 5, 5), fill=255)
    mask.save(mask_path)
    return image_path, mask_path


def safe_config(root: Path, image_path: Path, mask_path: Path | None = None) -> dict[str, object]:
    ckpt_root = root / "ckpts"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    (ckpt_root / "long256.pt").write_text("fixture", encoding="utf-8")
    (ckpt_root / "tile.pt").write_text("fixture", encoding="utf-8")
    cfg: dict[str, object] = {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_v3_long256_tile_integrated_report",
        "execution_mode": "approved_local_pre_sns_v3_long256_tile_integrated_report",
        "long256_checkpoint_path": str(ckpt_root / "long256.pt"),
        "tile_localizer_checkpoint_path": str(ckpt_root / "tile.pt"),
        "image_path": str(image_path),
        "gt_mask_path": str(mask_path) if mask_path else None,
        "sample_list_path": None,
        "approved_input_roots": [str(root / "inputs")],
        "approved_checkpoint_roots": [str(ckpt_root)],
        "output_root": str(root / "reports" / "integrated"),
        "device": "cpu",
        "tile_size": 4,
        "tile_stride": 3,
        "tile_activation_tau": 0.5,
        "mask_threshold": 0.5,
        "max_tiles": 16,
        "aggregation_mode": "average",
        "max_final_mask_area_pct": 35.0,
        "min_final_mask_area_pct": 0.0,
        "max_tile_vs_baseline_area_ratio": 4.0,
        "require_tile_improves_gt_when_gt_available": False,
        "fallback_to_baseline_when_tile_unreliable": True,
        "suppress_mask_for_non_tampered": True,
        "fixture_long256_report": {"class": "tampered", "family": "Other", "tampered_score": 0.9, "baseline_mask": [0] * 70},
        "fixture_tile_mask": [1 if 2 <= (idx % 10) <= 5 and 2 <= (idx // 10) <= 5 else 0 for idx in range(70)],
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }
    return cfg


def test_tile_grid_boundary_and_aggregation() -> None:
    tiles = tile_grid(10, 7, 4, 3)
    assert_true({"x0": 6, "y0": 3, "x1": 10, "y1": 7} in tiles, "boundary tile")
    preds = [{"tile": tile, "values": [1.0] * ((tile["x1"] - tile["x0"]) * (tile["y1"] - tile["y0"]))} for tile in tiles]
    values = aggregate_tile_values(10, 7, preds, "average")
    assert_equal(len(values), 70, "aggregation shape")
    assert_true(all(value == 1.0 for value in values), "aggregation coverage")


def test_threshold_and_gt_metrics() -> None:
    assert_equal(threshold_mask([0.1, 0.5, 0.9], 0.5), [0, 1, 1], "threshold mask")
    metrics = compare_with_gt([1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0])
    assert_equal(metrics["baseline_long256_iou"], 0.5, "baseline iou")
    assert_equal(metrics["tile_final_iou"], 1.0, "tile iou")
    assert_true(metrics["iou_delta"] > 0.0, "iou delta")


def test_decision_logic_activation_and_suppression() -> None:
    root = temp_root("cvf_integrated_logic_")
    image_path, mask_path = write_fixture_image(root)
    cfg = safe_config(root, image_path, mask_path)
    record = build_integrated_record(cfg, {"image_path": str(image_path)}, 0)
    assert_true(record["tile_localization_activated"], "tampered activates tile localization")
    assert_equal(record["final_mask_source"], "tile_localizer", "tile mask selected")
    cfg["fixture_long256_report"] = {"class": "real", "family": "Real-or-N/A", "tampered_score": 0.1, "baseline_mask": [1] * 70}
    record = build_integrated_record(cfg, {"image_path": str(image_path)}, 0)
    assert_true(not record["tile_localization_activated"], "real skips tile localization")
    assert_equal(record["final_mask_area_pct"], 0.0, "non-tampered suppresses mask")


def test_unreliable_tile_mask_falls_back_to_baseline() -> None:
    root = temp_root("cvf_integrated_reliable_")
    image_path, mask_path = write_fixture_image(root)
    gt = [1 if 2 <= (idx % 10) <= 5 and 2 <= (idx // 10) <= 5 else 0 for idx in range(70)]
    cfg = safe_config(root, image_path, mask_path)
    cfg["fixture_long256_report"] = {"class": "tampered", "family": "Other", "tampered_score": 0.9, "baseline_mask": gt}
    cfg["fixture_tile_mask"] = [1] * 70
    record = build_integrated_record(cfg, {"image_path": str(image_path)}, 0)
    assert_equal(record["final_mask_source"], "baseline_long256_tile_unreliable", "overactive tile falls back")
    assert_equal(record["localized_evidence_status"], "tile_unreliable_baseline_retained", "unreliable status")
    assert_true("tile_mask_area_too_large" in record["tile_reliability_reasons"], "area reason")
    assert_equal(record["gt_comparison"]["tile_final_iou"], 1.0, "baseline retained against gt")


def test_gt_regression_reliability_reason() -> None:
    gt = [1, 1, 0, 0]
    baseline = [1, 1, 0, 0]
    tile = [1, 1, 1, 1]
    reasons = tile_reliability_reasons(
        {
            "max_final_mask_area_pct": 100.0,
            "max_tile_vs_baseline_area_ratio": 0.0,
            "require_tile_improves_gt_when_gt_available": True,
        },
        "tampered",
        tile,
        baseline,
        gt,
    )
    assert_true("tile_iou_worse_than_baseline" in reasons, "gt regression reason")


def test_validator_guardrails() -> None:
    example_path = REPO_ROOT / "configs" / "inference" / "pre_sns_v3_long256_tile_integrated_report.example.json"
    example = json.loads(example_path.read_text())
    assert_equal(validate_integrated_report_config(example, require_exists=False), [], "example config")
    root = temp_root("cvf_integrated_validator_")
    image_path, mask_path = write_fixture_image(root)
    cfg = safe_config(root, image_path, mask_path)
    assert_equal(validate_integrated_report_config(cfg, require_exists=False), [], "safe config")
    unsafe = dict(cfg)
    unsafe["output_root"] = str(REPO_ROOT / "integrated_out")
    assert_true(validate_integrated_report_config(unsafe, require_exists=False), "repo output rejected")
    for key in ("no_network", "no_download", "no_training", "no_sns_augmentation"):
        unsafe = dict(cfg)
        unsafe[key] = False
        assert_true(validate_integrated_report_config(unsafe, require_exists=False), f"{key} rejected")


def test_run_integrated_report_no_repo_writes() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_integrated_run_")
    image_path, mask_path = write_fixture_image(root)
    cfg = safe_config(root, image_path, mask_path)
    result = run_integrated_report(cfg)
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "no repo root writes")
    assert_equal(result["marker"], MARKER, "marker")
    out = Path(cfg["output_root"])
    for name in ("integrated_report.json", "integrated_report_summary.json", "final_mask.png", "final_clean_red_overlay.png", "artifact_manifest.json"):
        assert_true((out / name).is_file(), f"{name} exists")
    assert_true((out / "baseline_agreement_overlay.png").is_file(), "baseline agreement")
    assert_true((out / "tile_agreement_overlay.png").is_file(), "tile agreement")
    assert_true((out / "gt_comparison_sheet.jpg").is_file(), "comparison sheet")


def main() -> int:
    tests = [
        test_tile_grid_boundary_and_aggregation,
        test_threshold_and_gt_metrics,
        test_decision_logic_activation_and_suppression,
        test_unreliable_tile_mask_falls_back_to_baseline,
        test_gt_regression_reliability_reason,
        test_validator_guardrails,
        test_run_integrated_report_no_repo_writes,
    ]
    for test in tests:
        test()
    print("PRE_SNS_V3_LONG256_TILE_INTEGRATED_REPORT_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
