#!/usr/bin/env python3
"""Plain Python tests for SNSAug v2 residual degradation analysis."""

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

from cv_forensics.snsaug_v2_fixed_pairs_eval import load_meta_rows, parse_fixed_pair_rows  # noqa: E402
from cv_forensics.snsaug_v2_residual_degradation_analysis import (  # noqa: E402
    MARKER,
    blockiness_score,
    build_residual_degradation_records,
    correlation_report,
    decide_next_step,
    dct_energy_summary,
    extract_residual_features,
    feature_group,
    flatten_correlation_schema,
    highpass_residual_energy,
    join_clean_sns_pairs,
    load_response_records,
    load_snsaug_v2_residual_degradation_analysis_config,
    pearson_correlation,
    run_snsaug_v2_residual_degradation_analysis,
    srm_like_features,
    validate_snsaug_v2_residual_degradation_analysis_config,
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
    for base_id, label in (("real_1", "real"), ("synthetic_1", "synthetic"), ("tampered_1", "tampered")):
        clean = Image.new("RGB", (16, 16), (70, 80, 90))
        draw = ImageDraw.Draw(clean)
        draw.rectangle((4, 4, 10, 10), fill=(210, 40, 40))
        draw.line((0, 15, 15, 0), fill=(20, 220, 140), width=1)
        clean_path = images / f"{base_id}__clean.png"
        clean.save(clean_path)
        gt_path = None
        if label == "tampered":
            gt_path = masks / f"{base_id}__clean.png"
            _mask_box((16, 16), (4, 4, 10, 10)).save(gt_path)
        rows.append(
            {
                "base_id": base_id,
                "content_label": label,
                "view": "clean",
                "profile": "clean",
                "image_path": str(clean_path),
                "tamper_mask_path": str(gt_path) if gt_path else None,
                "ignore_mask_path": None,
            }
        )
        responses.append(
            {
                "base_id": base_id,
                "content_label": label,
                "view": "clean",
                "profile": "clean",
                "pred_class": label,
                "p_tampered": 0.88 if label == "tampered" else 0.03,
                "valid_iou": 0.75 if label == "tampered" else None,
                "localization_activated": label == "tampered",
            }
        )
        sns = clean.resize((12, 18))
        sns_draw = ImageDraw.Draw(sns)
        for x in range(0, 12, 8):
            sns_draw.line((x, 0, x, 17), fill=(255, 255, 255), width=1)
        for y in range(0, 18, 8):
            sns_draw.line((0, y, 11, y), fill=(255, 255, 255), width=1)
        sns_path = images / f"{base_id}__zoom_crop.png"
        sns.save(sns_path)
        ignore_path = ignores / f"{base_id}__zoom_crop.png"
        _mask_box((12, 18), (0, 0, 2, 2)).save(ignore_path)
        rows.append(
            {
                "base_id": base_id,
                "content_label": label,
                "view": "sns_aug",
                "profile": "zoom_crop",
                "image_path": str(sns_path),
                "tamper_mask_path": str(gt_path) if gt_path else None,
                "ignore_mask_path": str(ignore_path),
            }
        )
        responses.append(
            {
                "base_id": base_id,
                "content_label": label,
                "view": "sns_aug",
                "profile": "zoom_crop",
                "policy": "original",
                "pred_class": "real",
                "p_tampered": 0.22 if label == "tampered" else 0.05,
                "valid_iou": 0.12 if label == "tampered" else None,
                "localization_activated": False,
            }
        )
        responses.append(
            {
                "base_id": base_id,
                "content_label": label,
                "view": "sns_aug",
                "profile": "zoom_crop",
                "policy": "diagnostic",
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
        "config_kind": "approved_snsaug_v2_residual_degradation_analysis",
        "execution_mode": "approved_local_snsaug_v2_residual_degradation_analysis",
        "user_approval_text": "I_APPROVE_SNSAUG_V2_RESIDUAL_DEGRADATION_ANALYSIS",
        "pair_root": str(pair_root),
        "meta_jsonl_path": str(meta_path),
        "baseline_records_path": str(records_path),
        "approved_input_roots": [str(pair_root), str(records_path.parent)],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "residual"),
        "profiles": ["clean", "zoom_crop"],
        "max_samples": 10,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def test_example_config_validates() -> None:
    cfg = load_snsaug_v2_residual_degradation_analysis_config(REPO_ROOT / "configs" / "evaluation" / "snsaug_v2_residual_degradation_analysis.example.json")
    assert_equal(validate_snsaug_v2_residual_degradation_analysis_config(cfg, require_exists=False), [], "example config validates")


def test_residual_feature_extractors_are_finite() -> None:
    root = temp_root("cvf_0070_features_")
    _pair_root, meta_path, _records_path = write_fixture(root)
    rows = load_meta_rows(meta_path)
    clean = next(row for row in rows if row["base_id"] == "tampered_1" and row["view"] == "clean")
    features = extract_residual_features(clean)
    assert_true(math.isfinite(features["highpass_residual_energy"]), "high-pass finite")
    assert_true(math.isfinite(features["srm_residual_energy"]), "srm energy finite")
    assert_true(math.isfinite(features["blockiness_score"]), "blockiness finite")
    assert_true(math.isfinite(features["dct_high_low_ratio"]), "dct ratio finite")
    values = [float((x * y) % 255) for y in range(16) for x in range(16)]
    assert_true(math.isfinite(highpass_residual_energy(values, 16, 16)), "highpass helper finite")
    assert_true(math.isfinite(srm_like_features(values, 16, 16)["srm_residual_std"]), "srm helper finite")
    assert_true(math.isfinite(blockiness_score(values, 16, 16)), "blockiness helper finite")
    assert_true(math.isfinite(dct_energy_summary(values, 16, 16)["dct_total_energy"]), "dct helper finite")


def test_clean_sns_pair_join_and_response_deltas() -> None:
    root = temp_root("cvf_0070_join_")
    _pair_root, meta_path, records_path = write_fixture(root)
    parsed, _warnings = parse_fixed_pair_rows(load_meta_rows(meta_path), ["clean", "zoom_crop"])
    pairs = join_clean_sns_pairs(parsed)
    assert_equal(len(pairs), 3, "clean/sns pair count")
    responses = load_response_records(records_path)
    records = build_residual_degradation_records(parsed, responses)
    tampered = next(row for row in records if row["content_label"] == "tampered")
    assert_equal(tampered["profile_family"], "geometry", "profile family")
    assert_equal(round(float(tampered["delta_p_tampered"]), 4), 0.66, "p_tampered drop")
    assert_true(tampered["activation_flip_off"] is True, "activation flip off")
    assert_true(tampered["pred_flip"] is True, "prediction flip")
    assert_true(tampered["correct_to_wrong"] is True, "correct-to-wrong")


def test_correlation_handles_constant_values() -> None:
    assert_true(pearson_correlation([1.0, 1.0, 1.0], [0.0, 1.0, 2.0]) is None, "constant feature correlation is None")
    rows = [
        {"content_label": "tampered", "profile_family": "geometry", "highpass_energy_delta_abs": 1.0, "delta_p_tampered": 0.1, "delta_valid_iou": 0.2, "activation_flip_off": False, "pred_flip": False, "correct_to_wrong": False},
        {"content_label": "tampered", "profile_family": "geometry", "highpass_energy_delta_abs": 1.0, "delta_p_tampered": 0.3, "delta_valid_iou": 0.4, "activation_flip_off": True, "pred_flip": True, "correct_to_wrong": True},
    ]
    table = correlation_report(rows)
    assert_true(table["overall"]["highpass_energy_delta_abs"]["delta_p_tampered"]["pearson_r"] is None, "constant table correlation is None")


def test_nested_correlation_schema_flattening_and_decision() -> None:
    nested = {
        "overall": {
            "dct_high_low_ratio_delta_abs": {
                "activation_flip_off": {
                    "pearson_r": -0.72,
                    "abs_pearson_r": 0.72,
                    "spearman_r": -0.68,
                    "abs_spearman_r": 0.68,
                    "pair_count": 120,
                }
            }
        },
        "by_profile_family": {
            "geometry": {
                "crop_scale_proxy": {
                    "delta_valid_iou": {
                        "pearson_r": 0.55,
                        "abs_pearson_r": 0.55,
                        "spearman_r": 0.50,
                        "abs_spearman_r": 0.50,
                        "pair_count": 60,
                    }
                }
            }
        },
    }
    rows = flatten_correlation_schema(nested)
    assert_true(rows, "nested schema produces rows")
    top = rows[0]
    assert_equal(top["feature"], "dct_high_low_ratio_delta_abs", "nested feature inferred")
    assert_equal(top["target"], "activation_flip_off", "nested target inferred")
    assert_equal(top["feature_group"], "residual_dct", "nested feature group")
    decision = decide_next_step(nested)
    assert_true(decision["decision"] != "correlation_parser_failed_or_empty", "valid nested correlations avoid parser failure")
    assert_true(decision["top_correlations"], "decision top correlations not empty")


def test_flat_correlation_schema_flattening() -> None:
    flat = [
        {
            "feature": "crop_scale_proxy",
            "target": "delta_valid_iou",
            "pearson_r": -0.66,
            "abs_pearson_r": 0.66,
            "spearman_r": -0.61,
            "abs_spearman_r": 0.61,
            "pair_count": 20,
        }
    ]
    rows = flatten_correlation_schema(flat)
    assert_equal(len(rows), 1, "flat schema row count")
    assert_equal(rows[0]["feature"], "crop_scale_proxy", "flat feature preserved")
    assert_equal(rows[0]["feature_group"], "geometry", "flat feature group")


def test_feature_group_classification() -> None:
    assert_equal(feature_group("srm_energy_delta_abs"), "residual_dct", "srm group")
    assert_equal(feature_group("blockiness_delta_abs"), "residual_dct", "blockiness group")
    assert_equal(feature_group("aspect_ratio_delta_abs"), "geometry", "geometry group")
    assert_equal(feature_group("ignore_mask_area"), "local_nuisance", "local nuisance group")
    assert_equal(feature_group("histogram_l1"), "color_histogram", "histogram group")
    assert_equal(feature_group("unknown_feature"), "other", "other group")


def test_config_validator_rejects_training_flags() -> None:
    root = temp_root("cvf_0070_flags_")
    pair_root, meta_path, records_path = write_fixture(root)
    cfg = safe_config(root, pair_root, meta_path, records_path)
    cfg["no_training"] = False
    cfg["no_network"] = False
    errors = validate_snsaug_v2_residual_degradation_analysis_config(cfg, require_exists=False)
    assert_true(any("no_training" in error for error in errors), "no_training required")
    assert_true(any("no_network" in error for error in errors), "no_network required")


def test_dry_run_and_tiny_run_write_artifacts() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_0070_run_")
    pair_root, meta_path, records_path = write_fixture(root)
    cfg = safe_config(root, pair_root, meta_path, records_path)
    cfg["config_path"] = str(root / "config.json")
    dry = run_snsaug_v2_residual_degradation_analysis(cfg, dry_run=True)
    assert_true(dry["analysis_started"] is False, "dry-run does not analyze")
    assert_true(not Path(cfg["output_root"]).exists(), "dry-run writes no records")
    summary = run_snsaug_v2_residual_degradation_analysis(cfg, dry_run=False)
    assert_equal(before, set(os.listdir(REPO_ROOT)), "no repo writes")
    assert_true(summary["analysis_started"] is True, "analysis started")
    assert_true(summary["record_count"] > 0, "records produced")
    for key in ("residual_degradation_records", "residual_feature_summary", "residual_feature_response_correlation", "residual_degradation_report", "artifact_manifest"):
        path = Path(summary["output_paths"][key])
        assert_true(path.exists(), f"{key} exists")
        assert_true(not str(path).startswith(str(REPO_ROOT)), f"{key} outside repo")
    artifact = json.loads(Path(summary["output_paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    correlation = json.loads(Path(summary["output_paths"]["residual_feature_response_correlation"]).read_text(encoding="utf-8"))
    report = Path(summary["output_paths"]["residual_degradation_report"]).read_text(encoding="utf-8")
    assert_equal(artifact["marker"], MARKER, "artifact marker")
    assert_equal(artifact["record_count"], summary["record_count"], "artifact record count")
    assert_true(artifact["no_training"] is True and artifact["no_download"] is True, "artifact guardrails")
    assert_true("top_correlations" in correlation, "top correlations written")
    assert_true("Top Correlations" in report, "report has top correlations table")


def test_docs_marker_present() -> None:
    text = (REPO_ROOT / "docs" / "snsaug_v2_residual_degradation_analysis.md").read_text(encoding="utf-8")
    assert_true(MARKER in text, "docs marker present")


def main() -> int:
    tests = [
        test_example_config_validates,
        test_residual_feature_extractors_are_finite,
        test_clean_sns_pair_join_and_response_deltas,
        test_correlation_handles_constant_values,
        test_nested_correlation_schema_flattening_and_decision,
        test_flat_correlation_schema_flattening,
        test_feature_group_classification,
        test_config_validator_rejects_training_flags,
        test_dry_run_and_tiny_run_write_artifacts,
        test_docs_marker_present,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
