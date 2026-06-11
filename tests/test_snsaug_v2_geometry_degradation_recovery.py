#!/usr/bin/env python3
"""Plain Python tests for SNSAug v2 geometry/degradation recovery."""

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

from cv_forensics.snsaug_v2_geometry_degradation_recovery import (  # noqa: E402
    MARKER,
    aggregate_per_policy_profile,
    apply_recovery_policy,
    border_trim_v2,
    content_box_crop_resize,
    deblock_mild,
    edge_density_content_box_v2,
    estimate_content_box,
    letterbox_unpad_resize_v2,
    load_snsaug_v2_geometry_degradation_recovery_config,
    oracle_gap_closure_summary,
    recovery_delta_summary,
    run_snsaug_v2_geometry_degradation_recovery,
    select_best_geometry_candidate_v2,
    transform_mask_like_image,
    validate_snsaug_v2_geometry_degradation_recovery_config,
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


def _base_mask(size: tuple[int, int], box: tuple[int, int, int, int]) -> list[int]:
    x1, y1, x2, y2 = box
    return [1 if x1 <= x <= x2 and y1 <= y <= y2 else 0 for y in range(size[1]) for x in range(size[0])]


def write_fixture(root: Path) -> tuple[Path, Path, Path]:
    pair_root = root / "pairs_0058c_eval"
    images = pair_root / "images"
    masks = pair_root / "tamper_masks"
    ignores = pair_root / "ignore_masks"
    bundle_dir = root / "bundle"
    for path in (images, masks, ignores, bundle_dir):
        path.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "long.pt").write_text("fixture", encoding="utf-8")
    (bundle_dir / "tile.pt").write_text("fixture", encoding="utf-8")
    bundle = {
        "long256_checkpoint_path": str(bundle_dir / "long.pt"),
        "tile_v2_checkpoint_path": str(bundle_dir / "tile.pt"),
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
    bundle_path = bundle_dir / "best_bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    clean_size = (10, 8)
    tamper_mask = _base_mask(clean_size, (3, 2, 6, 5))
    empty_mask = [0] * (clean_size[0] * clean_size[1])
    rows: list[dict[str, object]] = []
    for base_id, label in (("real_1", "real"), ("synthetic_1", "synthetic"), ("tampered_1", "tampered")):
        clean = Image.new("RGB", clean_size, (40, 80, 120))
        ImageDraw.Draw(clean).rectangle((3, 2, 6, 5), fill=(210, 40, 40))
        clean_path = images / f"{base_id}__clean.png"
        clean.save(clean_path)
        ignore_clean = ignores / f"{base_id}__clean.png"
        Image.new("L", clean_size, 0).save(ignore_clean)
        gt_path = None
        if label == "tampered":
            gt_path = masks / f"{base_id}__clean.png"
            _mask_box(clean_size, (3, 2, 6, 5)).save(gt_path)
        clean_fixture = {
            "class": label,
            "class_conf": {"real": 0.05, "synthetic": 0.05, "tampered": 0.90} if label == "tampered" else {"real": 0.95 if label == "real" else 0.03, "synthetic": 0.95 if label == "synthetic" else 0.03, "tampered": 0.02},
            "family": "Other",
            "tampered_score": 0.90 if label == "tampered" else 0.02,
            "baseline_mask": tamper_mask if label == "tampered" else empty_mask,
        }
        rows.append(
            {
                "base_id": base_id,
                "content_label": label,
                "view": "clean",
                "profile": "clean",
                "image_path": str(clean_path),
                "ignore_mask_path": str(ignore_clean),
                "tamper_mask_path": str(gt_path) if gt_path else None,
                "fixture_long256_report": clean_fixture,
                "fixture_v2_probability_mask": [float(value) for value in (tamper_mask if label == "tampered" else empty_mask)],
            }
        )
        canvas = Image.new("RGB", (16, 16), (8, 8, 8))
        canvas.paste(clean.resize((10, 8)), (3, 4))
        sns_path = images / f"{base_id}__zoom_crop.png"
        canvas.save(sns_path)
        ignore_sns = ignores / f"{base_id}__zoom_crop.png"
        Image.new("L", (16, 16), 0).save(ignore_sns)
        sns_fixture = clean_fixture
        if label == "tampered":
            sns_fixture = {
                "class": "real",
                "class_conf": {"real": 0.70, "synthetic": 0.05, "tampered": 0.25},
                "family": "Other",
                "tampered_score": 0.25,
                "baseline_mask": [0] * (16 * 16),
            }
        rows.append(
            {
                "base_id": base_id,
                "content_label": label,
                "view": "sns_aug",
                "profile": "zoom_crop",
                "image_path": str(sns_path),
                "ignore_mask_path": str(ignore_sns),
                "tamper_mask_path": str(gt_path) if gt_path else None,
                "fixture_long256_report": sns_fixture,
                "fixture_v2_probability_mask": [0.0] * (16 * 16),
                "fixture_by_policy": {
                    "geometry_normalized": {
                        "fixture_long256_report": clean_fixture,
                        "fixture_v2_probability_mask": [float(value) for value in (tamper_mask if label == "tampered" else empty_mask)],
                    },
                    "geometry_plus_deblock": {
                        "fixture_long256_report": clean_fixture,
                        "fixture_v2_probability_mask": [float(value) for value in (tamper_mask if label == "tampered" else empty_mask)],
                    },
                    "border_trim_v2": {
                        "fixture_long256_report": clean_fixture,
                        "fixture_v2_probability_mask": [float(value) for value in (tamper_mask if label == "tampered" else empty_mask)],
                    },
                    "edge_density_content_box_v2": {
                        "fixture_long256_report": clean_fixture,
                        "fixture_v2_probability_mask": [float(value) for value in (tamper_mask if label == "tampered" else empty_mask)],
                    },
                    "letterbox_unpad_resize_v2": {
                        "fixture_long256_report": clean_fixture,
                        "fixture_v2_probability_mask": [float(value) for value in (tamper_mask if label == "tampered" else empty_mask)],
                    },
                    "screenshot_frame_trim_v2": {
                        "fixture_long256_report": clean_fixture,
                        "fixture_v2_probability_mask": [float(value) for value in (tamper_mask if label == "tampered" else empty_mask)],
                    },
                    "multi_candidate_geometry_v2": {
                        "fixture_long256_report": clean_fixture,
                        "fixture_v2_probability_mask": [float(value) for value in (tamper_mask if label == "tampered" else empty_mask)],
                    },
                    "multi_candidate_geometry_plus_deblock_v2": {
                        "fixture_long256_report": clean_fixture,
                        "fixture_v2_probability_mask": [float(value) for value in (tamper_mask if label == "tampered" else empty_mask)],
                    },
                    "oracle_clean_geometry_diagnostic": {
                        "fixture_long256_report": clean_fixture,
                        "fixture_v2_probability_mask": [float(value) for value in (tamper_mask if label == "tampered" else empty_mask)],
                    },
                },
            }
        )
    meta_path = pair_root / "meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return pair_root, meta_path, bundle_path


def safe_config(root: Path, pair_root: Path, meta_path: Path, bundle_path: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_geometry_degradation_recovery",
        "execution_mode": "approved_local_snsaug_v2_geometry_degradation_recovery",
        "user_approval_text": "I_APPROVE_SNSAUG_V2_GEOMETRY_DEGRADATION_RECOVERY",
        "best_bundle_path": str(bundle_path),
        "pair_root": str(pair_root),
        "meta_jsonl_path": str(meta_path),
        "approved_input_roots": [str(root / "bundle"), str(pair_root)],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "recovery"),
        "policies": [
            "original",
            "content_box_crop_resize",
            "deblock_mild",
            "geometry_normalized",
            "geometry_plus_deblock",
            "border_trim_v2",
            "edge_density_content_box_v2",
            "letterbox_unpad_resize_v2",
            "screenshot_frame_trim_v2",
            "multi_candidate_geometry_v2",
            "multi_candidate_geometry_plus_deblock_v2",
            "oracle_clean_geometry_diagnostic",
        ],
        "profiles": ["clean", "zoom_crop"],
        "device": "cpu",
        "max_samples": 10,
        "top_n_gallery": 4,
        "content_background_tolerance": 10,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def test_example_config_validates() -> None:
    cfg = load_snsaug_v2_geometry_degradation_recovery_config(REPO_ROOT / "configs" / "evaluation" / "snsaug_v2_geometry_degradation_recovery.example.json")
    assert_equal(validate_snsaug_v2_geometry_degradation_recovery_config(cfg, require_exists=False), [], "example config validates")


def test_config_validator_rejects_training_flags() -> None:
    root = temp_root("cvf_0069_flags_")
    pair_root, meta_path, bundle_path = write_fixture(root)
    cfg = safe_config(root, pair_root, meta_path, bundle_path)
    cfg["no_training"] = False
    cfg["no_network"] = False
    errors = validate_snsaug_v2_geometry_degradation_recovery_config(cfg, require_exists=False)
    assert_true(any("no_training" in error for error in errors), "no_training required")
    assert_true(any("no_network" in error for error in errors), "no_network required")


def test_content_box_crop_modifies_geometry_profiles() -> None:
    image = Image.new("RGB", (16, 16), (8, 8, 8))
    ImageDraw.Draw(image).rectangle((4, 5, 11, 12), fill=(200, 50, 50))
    box = estimate_content_box(image, tolerance=10)
    assert_true(box != (0, 0, 16, 16), "content box detected")
    cropped = content_box_crop_resize(image, target_size=(10, 8), tolerance=10)
    assert_equal(cropped.size, (10, 8), "content crop resized to target geometry")
    normalized, info = apply_recovery_policy(image, policy="geometry_normalized", clean_image=Image.new("RGB", (10, 8)), tolerance=10)
    assert_equal(normalized.size, (10, 8), "geometry normalized to clean size")
    assert_true(info["diagnostic_oracle"] is False, "geometry policy not oracle")


def test_v2_content_box_estimators_trim_borders() -> None:
    image = Image.new("RGB", (20, 18), (5, 5, 5))
    ImageDraw.Draw(image).rectangle((5, 4, 14, 13), fill=(220, 80, 40))
    bordered, border_info = border_trim_v2(image, target_size=(10, 8), tolerance=10)
    edged, edge_info = edge_density_content_box_v2(image, target_size=(10, 8))
    unpadded, pad_info = letterbox_unpad_resize_v2(image, target_size=(10, 8), tolerance=10)
    assert_equal(bordered.size, (10, 8), "border trim output target size")
    assert_equal(edged.size, (10, 8), "edge density output target size")
    assert_equal(unpadded.size, (10, 8), "letterbox unpad output target size")
    assert_true(border_info["transform"]["box"] != (0, 0, 20, 18), "border trim crops")
    assert_true(edge_info["transform"]["box"] != (0, 0, 20, 18), "edge density crops")
    assert_true(pad_info["transform"]["box"] != (0, 0, 20, 18), "letterbox unpad crops")


def test_mask_transform_follows_image_transform() -> None:
    mask = Image.new("L", (20, 18), 0)
    ImageDraw.Draw(mask).rectangle((5, 4, 14, 13), fill=255)
    transform = {"op": "crop_resize", "box": (5, 4, 15, 14), "target_size": (10, 8)}
    transformed = transform_mask_like_image(mask, transform, source_size=(20, 18))
    assert_equal(transformed.size, (10, 8), "mask resized to policy geometry")
    assert_true(sum(1 for value in transformed.getdata() if int(value) > 0) == 80, "mask crop preserved positive region")


def test_oracle_gap_closure_handles_zero_oracle_gain() -> None:
    per_policy = {
        "original": {"zoom_crop": {"tampered_recall": 0.5, "tampered_valid_mean_iou": 0.2}},
        "oracle_clean_geometry_diagnostic": {"zoom_crop": {"tampered_recall": 0.5, "tampered_valid_mean_iou": 0.2}},
        "border_trim_v2": {"zoom_crop": {"tampered_recall": 0.6, "tampered_valid_mean_iou": 0.3}},
    }
    closure = oracle_gap_closure_summary(per_policy)
    row = [item for item in closure["rows"] if item["policy"] == "border_trim_v2"][0]
    assert_equal(row["tampered_recall_oracle_gap_closure"], 0.0, "zero oracle recall gain closure")
    assert_equal(row["valid_iou_oracle_gap_closure"], 0.0, "zero oracle iou gain closure")


def test_non_oracle_candidate_selection_is_image_only() -> None:
    image = Image.new("RGB", (20, 18), (5, 5, 5))
    ImageDraw.Draw(image).rectangle((5, 4, 14, 13), fill=(220, 80, 40))
    first = select_best_geometry_candidate_v2(image, target_size=(10, 8), tolerance=10)
    second = select_best_geometry_candidate_v2(image, target_size=(10, 8), tolerance=10)
    assert_equal(first["selected"], second["selected"], "selection deterministic from image geometry")
    assert_true("content_label" not in first["selected"], "selection does not expose labels")
    assert_true("tamper_mask" not in first["selected"], "selection does not expose masks")


def test_deblock_mild_keeps_image_size() -> None:
    image = Image.new("RGB", (12, 9), (20, 30, 40))
    filtered = deblock_mild(image)
    assert_equal(filtered.size, image.size, "deblock keeps size")


def test_metrics_handle_missing_tampered_rows_safely() -> None:
    records = [
        {"policy": "original", "profile": "clean", "content_label": "real", "pred_class": "real", "class_correct": True, "p_tampered": 0.01, "localization_activated": False, "final_mask_area_pct": 0.0},
        {"policy": "deblock_mild", "profile": "clean", "content_label": "real", "pred_class": "real", "class_correct": True, "p_tampered": 0.01, "localization_activated": False, "final_mask_area_pct": 0.0},
    ]
    metrics = aggregate_per_policy_profile(records)
    recovery = recovery_delta_summary(metrics)
    assert_true(metrics["original"]["clean"]["tampered_recall"] is None, "tampered recall is NA")
    assert_true(recovery["deblock_mild"]["clean"]["tampered_recall_recovery"] is None, "recovery handles NA")


def test_dry_run_writes_no_records_and_tiny_run_writes_outputs() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_0069_run_")
    pair_root, meta_path, bundle_path = write_fixture(root)
    cfg = safe_config(root, pair_root, meta_path, bundle_path)
    cfg["config_path"] = str(root / "config.json")
    dry = run_snsaug_v2_geometry_degradation_recovery(cfg, dry_run=True)
    assert_true(dry["inference_started"] is False, "dry-run does not infer")
    assert_true(not Path(cfg["output_root"]).exists(), "dry-run writes no records")
    summary = run_snsaug_v2_geometry_degradation_recovery(cfg, dry_run=False)
    assert_equal(before, set(os.listdir(REPO_ROOT)), "no repo writes")
    assert_true(summary["inference_started"] is True, "inference started")
    assert_true(summary["record_count"] > 0, "records produced")
    for key in (
        "geometry_degradation_recovery_records",
        "per_policy_per_profile_metrics",
        "recovery_delta_summary",
        "oracle_gap_closure_summary",
        "content_box_candidate_diagnostics",
        "geometry_degradation_recovery_report",
        "geometry_estimation_report_0069b",
        "visual_gallery_manifest",
        "artifact_manifest",
    ):
        path = Path(summary["output_paths"][key])
        assert_true(path.exists(), f"{key} exists")
        assert_true(not str(path).startswith(str(REPO_ROOT)), f"{key} outside repo")
    rows = [json.loads(line) for line in Path(summary["output_paths"]["geometry_degradation_recovery_records"]).read_text(encoding="utf-8").splitlines()]
    assert_true(any(row["policy"] == "geometry_normalized" for row in rows), "policy rows written")
    artifact = json.loads(Path(summary["output_paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    closure = json.loads(Path(summary["output_paths"]["oracle_gap_closure_summary"]).read_text(encoding="utf-8"))
    diagnostics = [json.loads(line) for line in Path(summary["output_paths"]["content_box_candidate_diagnostics"]).read_text(encoding="utf-8").splitlines()]
    assert_equal(artifact["marker"], MARKER, "artifact marker")
    assert_equal(artifact["record_count"], summary["record_count"], "artifact record count")
    assert_true(artifact["policy_counts"]["geometry_normalized"] > 0, "artifact policy counts")
    assert_true(artifact["no_training"] is True and artifact["no_download"] is True, "artifact guardrails")
    assert_true(closure["by_policy"], "oracle gap closure summary not empty")
    assert_true(any(row["policy"] == "multi_candidate_geometry_v2" for row in diagnostics), "candidate diagnostics written")


def test_docs_marker_present() -> None:
    text = (REPO_ROOT / "docs" / "snsaug_v2_geometry_degradation_recovery.md").read_text(encoding="utf-8")
    assert_true(MARKER in text, "docs marker present")


def main() -> int:
    tests = [
        test_example_config_validates,
        test_config_validator_rejects_training_flags,
        test_content_box_crop_modifies_geometry_profiles,
        test_v2_content_box_estimators_trim_borders,
        test_mask_transform_follows_image_transform,
        test_oracle_gap_closure_handles_zero_oracle_gain,
        test_non_oracle_candidate_selection_is_image_only,
        test_deblock_mild_keeps_image_size,
        test_metrics_handle_missing_tampered_rows_safely,
        test_dry_run_writes_no_records_and_tiny_run_writes_outputs,
        test_docs_marker_present,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
