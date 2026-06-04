#!/usr/bin/env python3
"""Plain Python tests for snsaug v2 robustness evaluation."""

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

from cv_forensics.snsaug_v2_robustness_eval import (  # noqa: E402
    MARKER,
    aggregate_per_profile,
    assign_mining_groups,
    build_training_mining_manifest,
    join_clean_and_profiles,
    run_snsaug_v2_robustness_eval,
    validate_snsaug_v2_robustness_eval_config,
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


def _mask_values(width: int, height: int, box: tuple[int, int, int, int]) -> list[int]:
    x1, y1, x2, y2 = box
    values = []
    for y in range(height):
        for x in range(width):
            values.append(1 if x1 <= x <= x2 and y1 <= y <= y2 else 0)
    return values


def write_fixture_inputs(root: Path) -> tuple[Path, Path, Path]:
    from PIL import Image, ImageDraw

    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    mask_values = _mask_values(12, 8, (3, 2, 7, 5))
    half_mask_values = _mask_values(12, 8, (3, 2, 5, 5))
    zero_values = [0] * (12 * 8)
    val_samples = []
    train_samples = []

    for sample_id_value, label, target in (
        ("val_anchor", "tampered", val_samples),
        ("val_clean_fail", "tampered", val_samples),
        ("real_val", "real", val_samples),
        ("train_fragile", "tampered", train_samples),
        ("train_noisy", "tampered", train_samples),
    ):
        image_path = inputs / f"{sample_id_value}.png"
        image = Image.new("RGB", (12, 8), (70, 80, 90))
        ImageDraw.Draw(image).rectangle((3, 2, 7, 5), fill=(220, 20, 20))
        image.save(image_path)
        sample = {
            "sample_id": sample_id_value,
            "class_label": label,
            "image_path": str(image_path),
        }
        if label == "tampered":
            mask_path = inputs / f"{sample_id_value}_mask.png"
            mask = Image.new("L", (12, 8), 0)
            ImageDraw.Draw(mask).rectangle((3, 2, 7, 5), fill=255)
            mask.save(mask_path)
            sample["mask_path"] = str(mask_path)
        if sample_id_value == "val_anchor":
            sample["fixture_long256_report"] = {"class": "tampered", "class_conf": {"tampered": 0.9, "real": 0.05, "synthetic": 0.05}, "family": "Other", "tampered_score": 0.9, "baseline_mask": mask_values}
            sample["fixture_v2_probability_mask"] = [1.0 if value else 0.0 for value in mask_values]
            sample["fixture_by_profile"] = {
                "tiktok_like": {
                    "fixture_long256_report": {"class": "real", "class_conf": {"real": 0.6, "tampered": 0.3, "synthetic": 0.1}, "family": "Real-or-N/A", "tampered_score": 0.3, "baseline_mask": zero_values},
                    "fixture_v2_probability_mask": [0.0] * 96,
                },
                "annotation_sticker": {
                    "fixture_long256_report": {"class": "tampered", "class_conf": {"tampered": 0.8, "real": 0.1, "synthetic": 0.1}, "family": "Other", "tampered_score": 0.8, "baseline_mask": mask_values},
                    "fixture_v2_probability_mask": [1.0 if value else 0.0 for value in half_mask_values],
                },
            }
        elif sample_id_value == "val_clean_fail":
            sample["fixture_long256_report"] = {"class": "real", "class_conf": {"real": 0.7, "tampered": 0.2, "synthetic": 0.1}, "family": "Real-or-N/A", "tampered_score": 0.2, "baseline_mask": zero_values}
            sample["fixture_v2_probability_mask"] = [0.0] * 96
        elif sample_id_value == "train_fragile":
            sample["fixture_long256_report"] = {"class": "tampered", "class_conf": {"tampered": 0.88, "real": 0.06, "synthetic": 0.06}, "family": "Other", "tampered_score": 0.88, "baseline_mask": mask_values}
            sample["fixture_v2_probability_mask"] = [1.0 if value else 0.0 for value in mask_values]
            sample["fixture_by_profile"] = {
                "tiktok_like": {
                    "fixture_long256_report": {"class": "real", "class_conf": {"real": 0.8, "tampered": 0.15, "synthetic": 0.05}, "family": "Real-or-N/A", "tampered_score": 0.15, "baseline_mask": zero_values},
                    "fixture_v2_probability_mask": [0.0] * 96,
                }
            }
        elif sample_id_value == "train_noisy":
            sample["fixture_long256_report"] = {"class": "tampered", "class_conf": {"tampered": 0.8, "real": 0.1, "synthetic": 0.1}, "family": "Other", "tampered_score": 0.8, "baseline_mask": zero_values}
            sample["fixture_v2_probability_mask"] = [0.0] * 96
            sample["fixture_by_profile"] = {
                "tiktok_like": {
                    "fixture_long256_report": {"class": "real", "class_conf": {"real": 0.7, "tampered": 0.2, "synthetic": 0.1}, "family": "Real-or-N/A", "tampered_score": 0.2, "baseline_mask": zero_values},
                    "fixture_v2_probability_mask": [0.0] * 96,
                },
                "instagram_story_like": {
                    "fixture_long256_report": {"class": "synthetic", "class_conf": {"synthetic": 0.6, "tampered": 0.2, "real": 0.2}, "family": "Other", "tampered_score": 0.2, "baseline_mask": zero_values},
                    "fixture_v2_probability_mask": [0.0] * 96,
                },
            }
        else:
            sample["fixture_long256_report"] = {"class": "real", "class_conf": {"real": 0.95, "tampered": 0.02, "synthetic": 0.03}, "family": "Real-or-N/A", "tampered_score": 0.02, "baseline_mask": zero_values}
            sample["fixture_v2_probability_mask"] = [0.0] * 96
        target.append(sample)

    val_manifest = inputs / "val_manifest.json"
    train_manifest = inputs / "train_manifest.json"
    val_manifest.write_text(json.dumps({"samples": val_samples}), encoding="utf-8")
    train_manifest.write_text(json.dumps({"samples": train_samples}), encoding="utf-8")

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
    bundle_path = inputs / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    return val_manifest, train_manifest, bundle_path


def safe_config(root: Path, val_manifest: Path, train_manifest: Path, bundle_path: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_robustness_eval",
        "execution_mode": "approved_local_snsaug_v2_robustness_eval",
        "best_bundle_path": str(bundle_path),
        "validation_manifest_path": str(val_manifest),
        "train_manifest_path": str(train_manifest),
        "approved_input_roots": [str(root / "inputs")],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "snsaug_eval"),
        "profiles": ["clean", "tiktok_like", "annotation_sticker", "instagram_story_like"],
        "severity": "medium",
        "seed": 52,
        "max_samples": 20,
        "samples_per_class": 10,
        "balanced_sampling": False,
        "top_n_worst_cases": 4,
        "device": "cpu",
        "no_training": True,
        "no_download": True,
        "no_network": True,
    }


def test_clean_profile_join() -> None:
    rows = [
        {"base_id": "a", "split": "val", "label": "tampered", "profile": "clean", "seed": 1, "pred_class": "tampered", "class_correct": True, "p_real": 0.0, "p_synthetic": 0.0, "p_tampered": 0.9, "final_iou": 0.8, "final_dice": 0.85, "localization_activated": True, "final_mask_source": "tile_v2", "latency_ms": 10.0, "fps": 100.0, "ignore_mask_area_pct": 0.0, "overlay_area_pct": 0.0},
        {"base_id": "a", "split": "val", "label": "tampered", "profile": "tiktok_like", "seed": 2, "pred_class": "real", "class_correct": False, "p_real": 0.6, "p_synthetic": 0.1, "p_tampered": 0.3, "final_iou": 0.0, "final_dice": 0.0, "localization_activated": False, "final_mask_source": "suppressed_non_tampered", "latency_ms": 12.0, "fps": 80.0, "ignore_mask_area_pct": 5.0, "overlay_area_pct": 5.0},
    ]
    joined = join_clean_and_profiles(rows)
    assert_equal(len(joined), 2, "joined row count")
    profile_row = [row for row in joined if row["profile"] == "tiktok_like"][0]
    assert_true(profile_row["pred_flip"], "prediction flip")
    assert_true(profile_row["correct_to_wrong"], "correct to wrong")


def test_group_assignment_logic() -> None:
    rows = [
        {"base_id": "a", "split": "val", "label": "tampered", "profile": "clean", "seed": 1, "clean_pred": "tampered", "profile_pred": "tampered", "clean_correct": True, "profile_correct": True, "p_tampered_clean": 0.9, "p_tampered_profile": 0.9, "p_tampered_drop": 0.0, "clean_iou": 0.8, "profile_iou": 0.8, "iou_drop": 0.0, "clean_dice": 0.8, "profile_dice": 0.8, "dice_drop": 0.0, "clean_localization_activated": True, "profile_localization_activated": True, "ignore_mask_area_pct": 0.0, "overlay_area_pct": 0.0, "final_mask_source": "tile_v2", "latency_ms": 10.0},
        {"base_id": "a", "split": "val", "label": "tampered", "profile": "tiktok_like", "seed": 2, "clean_pred": "tampered", "profile_pred": "real", "clean_correct": True, "profile_correct": False, "p_tampered_clean": 0.9, "p_tampered_profile": 0.3, "p_tampered_drop": 0.6, "clean_iou": 0.8, "profile_iou": 0.0, "iou_drop": 0.8, "clean_dice": 0.8, "profile_dice": 0.0, "dice_drop": 0.8, "clean_localization_activated": True, "profile_localization_activated": False, "ignore_mask_area_pct": 4.0, "overlay_area_pct": 4.0, "final_mask_source": "suppressed_non_tampered", "latency_ms": 12.0},
    ]
    grouped = assign_mining_groups(rows)
    prof = [row for row in grouped if row["profile"] == "tiktok_like"][0]
    assert_true(prof["stable_correct_anchor"], "stable anchor")
    assert_true(prof["fragile_correct_to_fail"], "fragile correct to fail")
    assert_true(prof["confidence_fragile"], "confidence fragile")
    assert_true(prof["mask_iou_fragile"], "mask fragile")
    assert_true(prof["threshold_flip"], "threshold flip")


def test_metric_aggregation() -> None:
    rows = assign_mining_groups(
        [
            {"base_id": "a", "split": "val", "label": "tampered", "profile": "clean", "seed": 1, "clean_pred": "tampered", "profile_pred": "tampered", "clean_correct": True, "profile_correct": True, "p_tampered_clean": 0.9, "p_tampered_profile": 0.9, "p_tampered_drop": 0.0, "clean_iou": 0.8, "profile_iou": 0.8, "iou_drop": 0.0, "clean_dice": 0.8, "profile_dice": 0.8, "dice_drop": 0.0, "clean_localization_activated": True, "profile_localization_activated": True, "ignore_mask_area_pct": 0.0, "overlay_area_pct": 0.0, "final_mask_source": "tile_v2", "latency_ms": 10.0},
            {"base_id": "a", "split": "val", "label": "tampered", "profile": "tiktok_like", "seed": 2, "clean_pred": "tampered", "profile_pred": "real", "clean_correct": True, "profile_correct": False, "p_tampered_clean": 0.9, "p_tampered_profile": 0.3, "p_tampered_drop": 0.6, "clean_iou": 0.8, "profile_iou": 0.0, "iou_drop": 0.8, "clean_dice": 0.8, "profile_dice": 0.0, "dice_drop": 0.8, "clean_localization_activated": True, "profile_localization_activated": False, "ignore_mask_area_pct": 4.0, "overlay_area_pct": 4.0, "final_mask_source": "suppressed_non_tampered", "latency_ms": 12.0},
        ]
    )
    metrics = aggregate_per_profile(rows)
    assert_true("clean" in metrics and "tiktok_like" in metrics, "metrics created")
    assert_true(metrics["tiktok_like"]["fragile_sample_count"] >= 1, "fragile count")


def test_validator_guardrails() -> None:
    root = temp_root("cvf_snsaug_eval_validator_")
    val_manifest, train_manifest, bundle_path = write_fixture_inputs(root)
    cfg = safe_config(root, val_manifest, train_manifest, bundle_path)
    assert_equal(validate_snsaug_v2_robustness_eval_config(cfg, require_exists=False), [], "safe config")
    unsafe = dict(cfg)
    unsafe["output_root"] = str(REPO_ROOT / "snsaug_eval")
    assert_true(validate_snsaug_v2_robustness_eval_config(unsafe, require_exists=False), "repo output rejected")
    unsafe = dict(cfg)
    unsafe["no_network"] = False
    assert_true(validate_snsaug_v2_robustness_eval_config(unsafe, require_exists=False), "network rejected")


def test_train_only_manifest_generation_excludes_val() -> None:
    rows = [
        {"base_id": "val_anchor", "split": "val", "label": "tampered", "groups": ["fragile_correct_to_fail"], "source_image_path": "/tmp/val.png"},
        {"base_id": "train_anchor", "split": "train", "label": "tampered", "groups": ["fragile_correct_to_fail", "stable_correct_anchor"], "source_image_path": "/tmp/train.png"},
    ]
    manifest = build_training_mining_manifest(rows, {"val_anchor"})
    assert_equal(len(manifest), 1, "train only manifest row count")
    assert_equal(manifest[0]["base_id"], "train_anchor", "val excluded")


def test_run_eval_output_schema() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_snsaug_eval_run_")
    val_manifest, train_manifest, bundle_path = write_fixture_inputs(root)
    summary = run_snsaug_v2_robustness_eval(safe_config(root, val_manifest, train_manifest, bundle_path))
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "no repo writes")
    assert_equal(summary["marker"], MARKER, "summary marker")
    out = root / "reports" / "snsaug_eval"
    for name in (
        "snsaug_v2_robustness_records.jsonl",
        "snsaug_v2_per_profile_metrics.json",
        "snsaug_v2_robustness_summary.json",
        "snsaug_v2_worst_profiles.json",
        "snsaug_v2_worst_samples.json",
        "snsaug_v2_fragile_candidates_val.jsonl",
        "snsaug_v2_fragile_candidates_train.jsonl",
        "snsaug_v2_training_mining_manifest_train_only.jsonl",
        "visual_gallery_manifest.json",
        "artifact_manifest.json",
    ):
        assert_true((out / name).is_file(), f"{name} exists")
    train_rows = [json.loads(line) for line in (out / "snsaug_v2_training_mining_manifest_train_only.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert_true(all(row["split"] == "train" for row in train_rows), "train manifest split")
    assert_true(all(not row["base_id"].startswith("val_") for row in train_rows), "no val ids in train manifest")


def main() -> int:
    tests = [
        test_clean_profile_join,
        test_group_assignment_logic,
        test_metric_aggregation,
        test_validator_guardrails,
        test_train_only_manifest_generation_excludes_val,
        test_run_eval_output_schema,
    ]
    for test in tests:
        test()
    print("SNSAUG_V2_ROBUSTNESS_EVAL_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
