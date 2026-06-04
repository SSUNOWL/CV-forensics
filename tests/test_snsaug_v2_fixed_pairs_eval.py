#!/usr/bin/env python3
"""Plain Python tests for snsaug v2 fixed-pairs evaluation."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TMP_PARENT = Path("/home/rlatjswo/.codex/memories")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2_fixed_pairs_eval import (  # noqa: E402
    MARKER,
    aggregate_per_profile,
    compute_mask_metrics,
    group_by_base_id,
    join_clean_and_sns,
    load_meta_rows,
    parse_fixed_pair_rows,
    robustness_drop_metrics,
    run_snsaug_v2_fixed_pairs_eval,
    validate_snsaug_v2_fixed_pairs_eval_config,
    worst_samples,
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


def _mask_box(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


def _write_pair_fixture(root: Path) -> tuple[Path, Path, Path]:
    pair_root = root / "pairs"
    images_dir = pair_root / "images"
    masks_dir = pair_root / "tamper_masks"
    ignore_dir = pair_root / "ignore_masks"
    for path in (images_dir, masks_dir, ignore_dir):
        path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    bundle = {
        "marker": "PRE_SNS_CURRENT_BEST_MODEL_BUNDLE_OK",
        "long256_checkpoint_path": str(root / "bundle" / "dummy_long256.pt"),
        "tile_v2_checkpoint_path": str(root / "bundle" / "dummy_tile_v2.pt"),
        "policy_gated_report": {
            "mask_threshold": 0.4,
            "min_area_pct": 0.01,
            "max_area_pct": 100.0,
            "suppress_non_tampered_mask": True,
            "fallback_to_baseline_on_v2_unreliable": True,
            "tile_size": 8,
            "tile_stride": 4
        }
    }
    (root / "bundle").mkdir(parents=True, exist_ok=True)
    (root / "bundle" / "dummy_long256.pt").write_text("fixture", encoding="utf-8")
    (root / "bundle" / "dummy_tile_v2.pt").write_text("fixture", encoding="utf-8")
    bundle_path = root / "bundle" / "best_bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    for base_id, label in (("real_1", "real"), ("synthetic_1", "synthetic"), ("tampered_1", "tampered")):
        image = Image.new("RGB", (12, 8), (60, 70, 80))
        ImageDraw.Draw(image).rectangle((3, 2, 7, 5), fill=(220, 40, 40))
        clean_path = images_dir / f"{base_id}__clean.png"
        image.save(clean_path)
        gt_path = None
        ignore_clean_path = ignore_dir / f"{base_id}__clean.png"
        Image.new("L", (12, 8), 0).save(ignore_clean_path)
        if label == "tampered":
            gt_path = masks_dir / f"{base_id}__clean.png"
            _mask_box((12, 8), (3, 2, 7, 5)).save(gt_path)
        clean_row = {
            "base_id": base_id,
            "content_label": label,
            "view": "clean",
            "profile": "clean",
            "seed": 1,
            "image_path": str(clean_path),
            "tamper_mask_path": str(gt_path) if gt_path else None,
            "ignore_mask_path": str(ignore_clean_path),
        }
        if label == "tampered":
            clean_row["fixture_long256_report"] = {
                "class": "tampered",
                "class_conf": {"tampered": 0.92, "real": 0.04, "synthetic": 0.04},
                "family": "Other",
                "tampered_score": 0.92,
                "baseline_mask": [1 if 3 <= x <= 7 and 2 <= y <= 5 else 0 for y in range(8) for x in range(12)],
            }
            clean_row["fixture_v2_probability_mask"] = [1.0 if 3 <= x <= 7 and 2 <= y <= 5 else 0.0 for y in range(8) for x in range(12)]
        else:
            clean_row["fixture_long256_report"] = {
                "class": label,
                "class_conf": {label: 0.95, "real": 0.95 if label == "real" else 0.02, "synthetic": 0.95 if label == "synthetic" else 0.02, "tampered": 0.03},
                "family": "Real-or-N/A",
                "tampered_score": 0.03,
                "baseline_mask": [0] * 96,
            }
            clean_row["fixture_v2_probability_mask"] = [0.0] * 96
        rows.append(clean_row)

        sns_path = images_dir / f"{base_id}__tiktok_like.png"
        image.save(sns_path)
        ignore_sns_path = ignore_dir / f"{base_id}__tiktok_like.png"
        _mask_box((12, 8), (0, 0, 2, 2)).save(ignore_sns_path)
        sns_row = {
            "base_id": base_id,
            "content_label": label,
            "view": "sns_aug",
            "profile": "tiktok_like",
            "seed": 2,
            "image_path": str(sns_path),
            "tamper_mask_path": str(gt_path) if gt_path else None,
            "ignore_mask_path": str(ignore_sns_path),
        }
        if label == "tampered":
            sns_row["fixture_long256_report"] = {
                "class": "real",
                "class_conf": {"real": 0.7, "synthetic": 0.05, "tampered": 0.25},
                "family": "Real-or-N/A",
                "tampered_score": 0.25,
                "baseline_mask": [0] * 96,
            }
            sns_row["fixture_v2_probability_mask"] = [0.0] * 96
        else:
            sns_row["fixture_long256_report"] = clean_row["fixture_long256_report"]
            sns_row["fixture_v2_probability_mask"] = [0.0] * 96
        rows.append(sns_row)

        dup_clean = dict(clean_row)
        dup_clean["view"] = "sns_aug"
        rows.append(dup_clean)

    meta_path = pair_root / "meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    (pair_root / "pair_index.json").write_text(json.dumps({"pairs": []}), encoding="utf-8")
    (pair_root / "artifact_manifest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    return pair_root, meta_path, bundle_path


def safe_config(root: Path, pair_root: Path, meta_path: Path, bundle_path: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_fixed_pairs_eval",
        "execution_mode": "approved_local_snsaug_v2_fixed_pairs_eval",
        "user_approval_text": "I_APPROVE_SNSAUG_V2_FIXED_PAIRS_EVAL",
        "best_bundle_path": str(bundle_path),
        "pair_root": str(pair_root),
        "meta_jsonl_path": str(meta_path),
        "approved_input_roots": [str(root / "bundle"), str(pair_root)],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "fixed_eval"),
        "profiles": ["clean", "tiktok_like"],
        "device": "cpu",
        "max_samples": 10,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def test_meta_parsing_and_duplicate_clean_filtering() -> None:
    root = temp_root("cvf_fixed_pairs_parse_")
    pair_root, meta_path, _bundle_path = _write_pair_fixture(root)
    rows = load_meta_rows(meta_path)
    parsed, warnings = parse_fixed_pair_rows(rows, ["clean", "tiktok_like"])
    assert_true(len(parsed) == 6, "filtered rows count")
    assert_true(warnings and "profile=clean" in warnings[0], "duplicate clean warning")
    grouped = group_by_base_id(parsed)
    assert_equal(len(grouped), 3, "group count")
    assert_true(all(grouped[key]["clean"] is not None for key in grouped), "clean baseline required")


def test_mask_metrics_valid_vs_raw() -> None:
    pred = [1, 1, 0, 0]
    gt = [1, 0, 1, 0]
    ignore = [0, 1, 0, 0]
    metrics = compute_mask_metrics(pred, gt, ignore)
    assert_equal(round(float(metrics["raw_iou"]), 4), 0.3333, "raw iou")
    assert_equal(round(float(metrics["valid_iou"]), 4), 0.5, "valid iou")


def test_clean_sns_join_and_fragile_flags() -> None:
    records = [
        {"base_id": "a", "content_label": "tampered", "view": "clean", "profile": "clean", "pred_class": "tampered", "class_correct": True, "p_tampered": 0.9, "valid_iou": 0.8, "raw_iou": 0.8, "localization_activated": True},
        {"base_id": "a", "content_label": "tampered", "view": "sns_aug", "profile": "tiktok_like", "pred_class": "real", "class_correct": False, "p_tampered": 0.3, "valid_iou": 0.1, "raw_iou": 0.1, "localization_activated": False},
    ]
    comparisons = join_clean_and_sns(records)
    assert_equal(len(comparisons), 1, "comparison count")
    comp = comparisons[0]
    assert_true(comp["pred_flip"], "prediction flip")
    assert_true(comp["fragile_class_flip"], "fragile class flip")
    assert_true(comp["fragile_confidence_drop"], "fragile confidence drop")
    assert_true(comp["fragile_mask_drop"], "fragile mask drop")
    assert_true(comp["fragile_activation_flip"], "fragile activation flip")


def test_confusion_macro_f1_and_drop_metrics() -> None:
    records = [
        {"base_id": "r", "content_label": "real", "view": "clean", "profile": "clean", "pred_class": "real", "class_correct": True, "final_mask_area_pct": 0.0, "latency_ms": 10.0},
        {"base_id": "s", "content_label": "synthetic", "view": "clean", "profile": "clean", "pred_class": "synthetic", "class_correct": True, "final_mask_area_pct": 0.0, "latency_ms": 10.0},
        {"base_id": "t", "content_label": "tampered", "view": "clean", "profile": "clean", "pred_class": "tampered", "class_correct": True, "valid_iou": 0.8, "valid_dice": 0.85, "raw_iou": 0.8, "localization_activated": True, "final_mask_area_pct": 10.0, "latency_ms": 10.0},
        {"base_id": "r", "content_label": "real", "view": "sns_aug", "profile": "tiktok_like", "pred_class": "tampered", "class_correct": False, "final_mask_area_pct": 2.0, "latency_ms": 11.0},
        {"base_id": "s", "content_label": "synthetic", "view": "sns_aug", "profile": "tiktok_like", "pred_class": "real", "class_correct": False, "final_mask_area_pct": 0.0, "latency_ms": 11.0},
        {"base_id": "t", "content_label": "tampered", "view": "sns_aug", "profile": "tiktok_like", "pred_class": "real", "class_correct": False, "valid_iou": 0.1, "valid_dice": 0.2, "raw_iou": 0.1, "localization_activated": False, "final_mask_area_pct": 0.0, "latency_ms": 11.0},
    ]
    metrics = aggregate_per_profile(records)
    assert_true("clean" in metrics and "tiktok_like" in metrics, "per-profile metrics")
    comparisons = join_clean_and_sns(records)
    drops = robustness_drop_metrics(comparisons, metrics)
    assert_true("tiktok_like" in drops, "drop metrics")
    worst = worst_samples(comparisons)
    assert_equal(worst[0]["base_id"], "t", "worst tampered sample first")


def test_validator_guardrails() -> None:
    root = temp_root("cvf_fixed_pairs_validator_")
    pair_root, meta_path, bundle_path = _write_pair_fixture(root)
    cfg = safe_config(root, pair_root, meta_path, bundle_path)
    assert_equal(validate_snsaug_v2_fixed_pairs_eval_config(cfg, require_exists=False), [], "valid config")
    unsafe = dict(cfg)
    unsafe["output_root"] = str(REPO_ROOT / "bad_eval_output")
    unsafe["no_network"] = False
    errors = validate_snsaug_v2_fixed_pairs_eval_config(unsafe, require_exists=False)
    assert_true(any("output_root must be outside repository" in error for error in errors), "repo output rejected")
    assert_true(any("no_network must be true" in error for error in errors), "network rejected")


def test_run_eval_output_schema() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_fixed_pairs_run_")
    pair_root, meta_path, bundle_path = _write_pair_fixture(root)
    summary = run_snsaug_v2_fixed_pairs_eval(safe_config(root, pair_root, meta_path, bundle_path))
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "no repo writes")
    assert_equal(summary["marker"], MARKER, "summary marker")
    out = root / "reports" / "fixed_eval"
    for name in (
        "snsaug_v2_eval_records.jsonl",
        "snsaug_v2_eval_comparisons.jsonl",
        "snsaug_v2_per_profile_metrics.json",
        "snsaug_v2_robustness_drop_metrics.json",
        "snsaug_v2_eval_summary.json",
        "snsaug_v2_worst_samples.json",
        "snsaug_v2_fragile_candidates.jsonl",
        "visual_gallery_manifest.json",
        "artifact_manifest.json",
    ):
        assert_true((out / name).is_file(), f"{name} exists")


def main() -> int:
    tests = [
        test_meta_parsing_and_duplicate_clean_filtering,
        test_mask_metrics_valid_vs_raw,
        test_clean_sns_join_and_fragile_flags,
        test_confusion_macro_f1_and_drop_metrics,
        test_validator_guardrails,
        test_run_eval_output_schema,
    ]
    for test in tests:
        test()
    print("SNSAUG_V2_FIXED_PAIRS_EVAL_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
