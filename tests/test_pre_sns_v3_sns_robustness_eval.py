#!/usr/bin/env python3
"""Plain Python tests for basic SNS robustness evaluation."""

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

from cv_forensics.pre_sns_v3_sns_robustness_eval import (  # noqa: E402
    MARKER,
    aggregate_per_perturbation,
    apply_center_crop_resize_back,
    apply_resize,
    join_clean_and_perturbed,
    macro_f1_and_details,
    confusion_matrix,
    normalize_label,
    robustness_drop_metrics,
    run_sns_robustness_eval,
    transform_image_and_mask,
    validate_sns_robustness_eval_config,
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


def write_fixture_inputs(root: Path) -> tuple[Path, Path]:
    from PIL import Image, ImageDraw

    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    sample_list = []
    bundle_path = root / "inputs" / "bundle.json"
    for index, label in enumerate(("tampered", "real", "full_synthetic")):
        image_path = inputs / f"{label}.png"
        image = Image.new("RGB", (12, 8), (40 + index * 20, 60, 80))
        draw = ImageDraw.Draw(image)
        draw.rectangle((3, 2, 7, 5), fill=(220, 30, 30))
        image.save(image_path)
        sample = {"sample_id": label, "class_label": label, "image_path": str(image_path)}
        if label == "tampered":
            mask_path = inputs / "tampered_mask.png"
            mask = Image.new("L", (12, 8), 0)
            ImageDraw.Draw(mask).rectangle((3, 2, 7, 5), fill=255)
            mask.save(mask_path)
            fixture_mask = [1 if 3 <= (idx % 12) <= 7 and 2 <= (idx // 12) <= 5 else 0 for idx in range(96)]
            sample["mask_path"] = str(mask_path)
            sample["fixture_long256_report"] = {"class": "tampered", "family": "Other", "tampered_score": 0.9, "baseline_mask": fixture_mask}
            sample["fixture_v2_probability_mask"] = [1.0 if value else 0.0 for value in fixture_mask]
        else:
            sample["fixture_long256_report"] = {"class": label, "family": "Real-or-N/A", "tampered_score": 0.01, "baseline_mask": [0] * 96}
            sample["fixture_v2_probability_mask"] = [0.0] * 96
        sample_list.append(sample)
    manifest_path = inputs / "samples.json"
    manifest_path.write_text(json.dumps({"samples": sample_list}), encoding="utf-8")
    bundle = {
        "marker": "PRE_SNS_CURRENT_BEST_MODEL_BUNDLE_OK",
        "long256_checkpoint_path": str(inputs / "dummy_long256.pt"),
        "tile_v2_checkpoint_path": str(inputs / "dummy_tile_v2.pt"),
        "policy_gated_report": {
            "mask_threshold": 0.45,
            "min_area_pct": 0.1,
            "max_area_pct": 100.0,
            "suppress_non_tampered_mask": True,
            "fallback_to_baseline_on_v2_unreliable": True,
            "tile_size": 8,
            "tile_stride": 4
        }
    }
    (inputs / "dummy_long256.pt").write_text("fixture", encoding="utf-8")
    (inputs / "dummy_tile_v2.pt").write_text("fixture", encoding="utf-8")
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    return manifest_path, bundle_path


def safe_config(root: Path, manifest_path: Path, bundle_path: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_v3_sns_robustness_eval",
        "execution_mode": "approved_local_pre_sns_v3_sns_robustness_eval",
        "best_bundle_path": str(bundle_path),
        "validation_manifest_path": str(manifest_path),
        "approved_input_roots": [str(root / "inputs")],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "sns_eval"),
        "max_samples": 3,
        "samples_per_class": 1,
        "balanced_sampling": True,
        "perturbations": ["clean", "jpeg_q85", "resize_long_512", "mild_center_crop_95pct_resize_back"],
        "top_n_worst_cases": 4,
        "device": "cpu",
        "no_training": True,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation_training": True,
    }


def test_jpeg_transform_keeps_mask_unchanged() -> None:
    from PIL import Image

    image = Image.new("RGB", (10, 8), (40, 50, 60))
    mask = Image.new("L", (10, 8), 0)
    out_image, out_mask = transform_image_and_mask("jpeg_q85", image, mask)
    assert_equal(out_image.size, image.size, "jpeg image size")
    assert_equal(list(out_mask.getdata()), list(mask.getdata()), "jpeg mask unchanged")


def test_resize_transform_resizes_mask_nearest() -> None:
    from PIL import Image

    image = Image.new("RGB", (2000, 1000), (0, 0, 0))
    mask = Image.new("L", (2000, 1000), 0)
    resized_image, resized_mask = apply_resize(Image, image, mask, 512)
    assert_equal(max(resized_image.size), 512, "resized long side")
    assert_equal(resized_image.size, resized_mask.size, "resized mask size")


def test_center_crop_transform_consistency() -> None:
    from PIL import Image

    image = Image.new("RGB", (20, 10), (0, 0, 0))
    mask = Image.new("L", (20, 10), 0)
    out_image, out_mask = apply_center_crop_resize_back(Image, image, mask, 0.95)
    assert_equal(out_image.size, image.size, "crop resized back image")
    assert_equal(out_mask.size, mask.size, "crop resized back mask")


def test_macro_f1_and_confusion() -> None:
    rows = [
        {"label": "real", "pred_class": "real"},
        {"label": "synthetic", "pred_class": "synthetic"},
        {"label": "tampered", "pred_class": "real"},
    ]
    matrix = confusion_matrix(rows)
    assert_equal(matrix["tampered"]["real"], 1, "tampered misclassified")
    details = macro_f1_and_details(matrix)
    assert_true(details["macro_f1"] < 1.0, "macro f1 reduced")


def test_join_and_fragile_candidate_logic() -> None:
    rows = [
        {"base_id": "a", "perturbation": "clean", "label": "tampered", "pred_class": "tampered", "class_correct": True, "p_tampered": 0.95, "final_iou": 0.7, "final_dice": 0.8, "localization_activated": True},
        {"base_id": "a", "perturbation": "jpeg_q85", "label": "tampered", "pred_class": "real", "class_correct": False, "p_tampered": 0.4, "final_iou": 0.3, "final_dice": 0.4, "localization_activated": False},
    ]
    joined = join_clean_and_perturbed(rows)
    pert = [row for row in joined if row["perturbation"] != "clean"][0]
    assert_true(pert["correct_to_wrong"], "correct to wrong")
    assert_true(pert["activation_flip_off"], "activation flipped off")
    assert_true(pert["fragile_candidate"], "fragile candidate")


def test_per_perturbation_aggregation_and_drop_metrics() -> None:
    rows = [
        {"base_id": "a", "perturbation": "clean", "label": "tampered", "pred_class": "tampered", "class_correct": True, "p_tampered": 0.9, "final_iou": 0.7, "final_dice": 0.8, "final_mask_area_pct": 4.0, "localization_activated": True, "latency_ms": 10.0},
        {"base_id": "a", "perturbation": "jpeg_q85", "label": "tampered", "pred_class": "tampered", "class_correct": True, "p_tampered": 0.5, "final_iou": 0.3, "final_dice": 0.4, "final_mask_area_pct": 3.0, "localization_activated": False, "latency_ms": 12.0},
    ]
    joined = join_clean_and_perturbed(rows)
    per = aggregate_per_perturbation(joined)
    drops = robustness_drop_metrics(joined, per)
    assert_true("clean" in per and "jpeg_q85" in per, "per perturbation metrics")
    assert_true(drops["jpeg_q85"]["mean_iou_drop"] > 0.0, "iou drop positive")


def test_validator_guardrails() -> None:
    root = temp_root("cvf_sns_eval_validator_")
    manifest_path, bundle_path = write_fixture_inputs(root)
    cfg = safe_config(root, manifest_path, bundle_path)
    assert_equal(validate_sns_robustness_eval_config(cfg, require_exists=False), [], "safe config")
    unsafe = dict(cfg)
    unsafe["output_root"] = str(REPO_ROOT / "sns_eval")
    assert_true(validate_sns_robustness_eval_config(unsafe, require_exists=False), "repo output rejected")
    for key in ("no_network", "no_download", "no_training", "no_sns_augmentation_training"):
        unsafe = dict(cfg)
        unsafe[key] = False
        assert_true(validate_sns_robustness_eval_config(unsafe, require_exists=False), f"{key} rejected")


def test_run_eval_no_repo_writes_and_output_schema() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_sns_eval_run_")
    manifest_path, bundle_path = write_fixture_inputs(root)
    summary = run_sns_robustness_eval(safe_config(root, manifest_path, bundle_path))
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "no repo writes")
    assert_equal(summary["marker"], MARKER, "summary marker")
    out = root / "reports" / "sns_eval"
    for name in (
        "sns_robustness_records.jsonl",
        "sns_robustness_summary.json",
        "per_perturbation_metrics.json",
        "robustness_drop_metrics.json",
        "worst_samples.json",
        "worst_perturbations.json",
        "fragile_candidates.jsonl",
        "visual_gallery_manifest.json",
        "artifact_manifest.json",
    ):
        assert_true((out / name).is_file(), f"{name} exists")


def main() -> int:
    tests = [
        test_jpeg_transform_keeps_mask_unchanged,
        test_resize_transform_resizes_mask_nearest,
        test_center_crop_transform_consistency,
        test_macro_f1_and_confusion,
        test_join_and_fragile_candidate_logic,
        test_per_perturbation_aggregation_and_drop_metrics,
        test_validator_guardrails,
        test_run_eval_no_repo_writes_and_output_schema,
    ]
    for test in tests:
        test()
    print("PRE_SNS_V3_SNS_ROBUSTNESS_EVAL_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
