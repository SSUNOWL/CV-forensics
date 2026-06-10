#!/usr/bin/env python3
"""Plain Python tests for SNSAug v2 global degradation geometry analysis."""

from __future__ import annotations

import json
import math
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

from cv_forensics.snsaug_v2_global_degradation_geometry_analysis import (  # noqa: E402
    MARKER,
    blockiness_score,
    build_shift_records,
    correlation_table,
    extract_image_features,
    highpass_residual_energy,
    join_clean_sns_pairs,
    load_response_records,
    load_snsaug_v2_global_degradation_geometry_analysis_config,
    pearson_correlation,
    run_snsaug_v2_global_degradation_geometry_analysis,
    validate_snsaug_v2_global_degradation_geometry_analysis_config,
)
from cv_forensics.snsaug_v2_fixed_pairs_eval import load_meta_rows, parse_fixed_pair_rows  # noqa: E402


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


def write_fixture(root: Path) -> tuple[Path, Path, Path]:
    pair_root = root / "pairs_0058c_eval"
    images = pair_root / "images"
    masks = pair_root / "tamper_masks"
    ignores = pair_root / "ignore_masks"
    records_dir = root / "records"
    for path in (images, masks, ignores, records_dir):
        path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    responses: list[dict[str, object]] = []
    for base_id, label in (("real_1", "real"), ("tampered_1", "tampered")):
        clean = Image.new("RGB", (12, 8), (70, 80, 90))
        ImageDraw.Draw(clean).rectangle((3, 2, 6, 5), fill=(210, 40, 40))
        clean_path = images / f"{base_id}__clean.png"
        clean.save(clean_path)
        gt_path = None
        if label == "tampered":
            gt_path = masks / f"{base_id}__clean.png"
            _mask_box((12, 8), (3, 2, 6, 5)).save(gt_path)
        clean_row = {
            "base_id": base_id,
            "content_label": label,
            "view": "clean",
            "profile": "clean",
            "image_path": str(clean_path),
            "tamper_mask_path": str(gt_path) if gt_path else None,
            "ignore_mask_path": None,
        }
        rows.append(clean_row)
        responses.append(
            {
                "base_id": base_id,
                "content_label": label,
                "view": "clean",
                "profile": "clean",
                "pred_class": label,
                "p_tampered": 0.9 if label == "tampered" else 0.02,
                "valid_iou": 0.8 if label == "tampered" else None,
                "localization_activated": label == "tampered",
            }
        )

        sns = clean.resize((10, 10))
        ImageDraw.Draw(sns).rectangle((0, 0, 2, 2), fill=(255, 255, 255))
        sns_path = images / f"{base_id}__zoom_crop.png"
        sns.save(sns_path)
        ignore_path = ignores / f"{base_id}__zoom_crop.png"
        _mask_box((10, 10), (0, 0, 2, 2)).save(ignore_path)
        sns_row = {
            "base_id": base_id,
            "content_label": label,
            "view": "sns_aug",
            "profile": "zoom_crop",
            "image_path": str(sns_path),
            "tamper_mask_path": str(gt_path) if gt_path else None,
            "ignore_mask_path": str(ignore_path),
        }
        rows.append(sns_row)
        responses.append(
            {
                "base_id": base_id,
                "content_label": label,
                "view": "sns_aug",
                "profile": "zoom_crop",
                "policy": "original",
                "pred_class": "real",
                "p_tampered": 0.2 if label == "tampered" else 0.04,
                "valid_iou": 0.1 if label == "tampered" else None,
                "localization_activated": False,
            }
        )
        responses.append(
            {
                "base_id": base_id,
                "content_label": label,
                "view": "sns_aug",
                "profile": "zoom_crop",
                "policy": "gray_fill",
                "pred_class": label,
                "p_tampered": 0.99,
                "valid_iou": 0.99,
                "localization_activated": True,
            }
        )
    meta_path = pair_root / "meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    records_path = records_dir / "baseline_records.jsonl"
    with open(records_path, "w", encoding="utf-8") as handle:
        for row in responses:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return pair_root, meta_path, records_path


def safe_config(root: Path, pair_root: Path, meta_path: Path, records_path: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_global_degradation_geometry_analysis",
        "execution_mode": "approved_local_snsaug_v2_global_degradation_geometry_analysis",
        "user_approval_text": "I_APPROVE_SNSAUG_V2_GLOBAL_DEGRADATION_GEOMETRY_ANALYSIS",
        "pair_root": str(pair_root),
        "meta_jsonl_path": str(meta_path),
        "baseline_records_path": str(records_path),
        "approved_input_roots": [str(pair_root), str(records_path.parent)],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "geometry"),
        "profiles": ["clean", "zoom_crop"],
        "max_samples": 10,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def test_example_config_validates() -> None:
    cfg = load_snsaug_v2_global_degradation_geometry_analysis_config(REPO_ROOT / "configs" / "evaluation" / "snsaug_v2_global_degradation_geometry_analysis.example.json")
    assert_equal(validate_snsaug_v2_global_degradation_geometry_analysis_config(cfg, require_exists=False), [], "example config validates")


def test_feature_extraction_handles_missing_masks_and_finite_scores() -> None:
    root = temp_root("cvf_0068_features_")
    pair_root, meta_path, _records_path = write_fixture(root)
    rows = load_meta_rows(meta_path)
    clean = next(row for row in rows if row["base_id"] == "real_1" and row["view"] == "clean")
    features = extract_image_features(clean)
    assert_true(features["ignore_mask_area"] == 0.0, "missing ignore mask safe")
    assert_true(features["tamper_area"] == 0.0, "missing tamper mask safe")
    assert_true(math.isfinite(features["blockiness_score"]), "blockiness finite")
    assert_true(math.isfinite(features["highpass_residual_energy"]), "high-pass finite")
    values = [0, 10, 20, 30] * 16
    assert_true(math.isfinite(blockiness_score(values, 8, 8)), "blockiness helper finite")
    assert_true(math.isfinite(highpass_residual_energy(values, 8, 8)), "highpass helper finite")
    assert_true(str(pair_root) in str(clean["image_path"]), "fixture under pair root")


def test_clean_sns_pair_join_and_response_join() -> None:
    root = temp_root("cvf_0068_join_")
    _pair_root, meta_path, records_path = write_fixture(root)
    parsed, _warnings = parse_fixed_pair_rows(load_meta_rows(meta_path), ["clean", "zoom_crop"])
    pairs = join_clean_sns_pairs(parsed)
    assert_equal(len(pairs), 2, "clean/sns pair count")
    responses = load_response_records(records_path)
    records = build_shift_records(parsed, responses)
    tampered = next(row for row in records if row["content_label"] == "tampered")
    assert_equal(tampered["profile_family"], "geometry", "profile family")
    assert_equal(round(float(tampered["delta_p_tampered"]), 4), 0.7, "p_tampered drop")
    assert_true(tampered["activation_flip_off"] is True, "activation flip off")
    assert_true(tampered["pred_flip"] is True, "prediction flip")


def test_correlation_handles_constant_features() -> None:
    assert_true(pearson_correlation([1.0, 1.0, 1.0], [0.0, 1.0, 2.0]) is None, "constant feature correlation is None")
    rows = [
        {"aspect_ratio_delta_abs": 1.0, "delta_p_tampered": 0.1, "delta_valid_iou": 0.2, "activation_flip_off": False, "pred_flip": False},
        {"aspect_ratio_delta_abs": 1.0, "delta_p_tampered": 0.3, "delta_valid_iou": 0.4, "activation_flip_off": True, "pred_flip": True},
    ]
    table = correlation_table(rows)
    assert_true(table["aspect_ratio_delta_abs"]["delta_p_tampered"]["pearson_r"] is None, "constant table correlation is None")


def test_config_validator_rejects_training_flags() -> None:
    root = temp_root("cvf_0068_flags_")
    pair_root, meta_path, records_path = write_fixture(root)
    cfg = safe_config(root, pair_root, meta_path, records_path)
    cfg["no_training"] = False
    cfg["no_download"] = False
    errors = validate_snsaug_v2_global_degradation_geometry_analysis_config(cfg, require_exists=False)
    assert_true(any("no_training" in error for error in errors), "no_training required")
    assert_true(any("no_download" in error for error in errors), "no_download required")


def test_tiny_run_writes_artifact_manifest() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_0068_run_")
    pair_root, meta_path, records_path = write_fixture(root)
    cfg = safe_config(root, pair_root, meta_path, records_path)
    cfg["config_path"] = str(root / "config.json")
    dry = run_snsaug_v2_global_degradation_geometry_analysis(cfg, dry_run=True)
    assert_true(dry["analysis_started"] is False, "dry-run does not analyze")
    assert_true(not Path(cfg["output_root"]).exists(), "dry-run writes no files")
    summary = run_snsaug_v2_global_degradation_geometry_analysis(cfg, dry_run=False)
    assert_equal(before, set(os.listdir(REPO_ROOT)), "no repo writes")
    assert_true(summary["analysis_started"] is True, "analysis started")
    assert_true(summary["record_count"] > 0, "records produced")
    for key in ("global_degradation_geometry_records", "profile_family_shift_summary", "feature_response_correlation", "global_degradation_geometry_report", "artifact_manifest"):
        path = Path(summary["output_paths"][key])
        assert_true(path.exists(), f"{key} exists")
        assert_true(not str(path).startswith(str(REPO_ROOT)), f"{key} outside repo")
    artifact = json.loads(Path(summary["output_paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert_equal(artifact["marker"], MARKER, "artifact marker")
    assert_equal(artifact["record_count"], summary["record_count"], "artifact record count")
    assert_true(artifact["no_training"] is True and artifact["no_network"] is True, "artifact guardrails")


def test_docs_marker_present() -> None:
    text = (REPO_ROOT / "docs" / "snsaug_v2_global_degradation_geometry_analysis.md").read_text(encoding="utf-8")
    assert_true(MARKER in text, "docs marker present")


def main() -> int:
    tests = [
        test_example_config_validates,
        test_feature_extraction_handles_missing_masks_and_finite_scores,
        test_clean_sns_pair_join_and_response_join,
        test_correlation_handles_constant_features,
        test_config_validator_rejects_training_flags,
        test_tiny_run_writes_artifact_manifest,
        test_docs_marker_present,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
