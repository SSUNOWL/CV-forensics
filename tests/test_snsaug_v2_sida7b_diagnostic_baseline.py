#!/usr/bin/env python3
"""Plain Python tests for SNSAug v2 SIDA-7B diagnostic baseline."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TMP_PARENT = Path("/home/rlatjswo/.codex/memories")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2_sida7b_diagnostic_baseline import (  # noqa: E402
    CACHED_MODE,
    EXPORT_MODE,
    MARKER,
    build_balanced_subset,
    compare_sida_to_mixed_gate,
    decide_sida_findings,
    export_subset_rows,
    load_mixed_gate_metrics,
    mask_valid_iou,
    normalize_class,
    parse_cached_sida_outputs,
    parse_sida_text_output,
    parse_sida_text_output_detail,
    run_snsaug_v2_sida7b_diagnostic_baseline,
    validate_snsaug_v2_sida7b_diagnostic_baseline_config,
    load_snsaug_v2_sida7b_diagnostic_baseline_config,
)
from cv_forensics.snsaug_v2_fixed_pairs_eval import load_meta_rows  # noqa: E402


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def temp_root(prefix: str) -> Path:
    TMP_PARENT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(TMP_PARENT)))


STRICT_PROFILES = (
    "canvas_9x16_only",
    "combined_sns_realistic",
    "instagram_story_like",
    "news_meme_overlay",
    "platform_ui_same_size",
    "recompression_light",
    "resize_crop_pad",
    "resize_jpeg",
    "screenshot_recapture_light",
    "tiktok_like",
    "youtube_shorts_like",
    "zoom_crop",
)


def strict_metrics(value: float = 0.7) -> dict[str, dict[str, float]]:
    return {profile: {"tampered_recall": value, "synthetic_recall": value, "real_fpr": 0.1, "tampered_valid_mean_iou": value} for profile in STRICT_PROFILES}


def _mask(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).rectangle(box, fill=255)
    return image


def write_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    pair_root = root / "pairs"
    images = pair_root / "images"
    masks = pair_root / "tamper_masks"
    ignores = pair_root / "ignore_masks"
    final_gate = root / "final_gate"
    for path in (images, masks, ignores, final_gate):
        path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for label in ("real", "synthetic", "tampered"):
        for idx in range(2):
            base_id = f"{label}_{idx}"
            clean = Image.new("RGB", (10, 10), (80, 90, 100))
            ImageDraw.Draw(clean).rectangle((3, 3, 6, 6), fill=(210, 30, 30))
            clean_path = images / f"{base_id}__clean.png"
            clean.save(clean_path)
            gt_path = None
            if label == "tampered":
                gt_path = masks / f"{base_id}__clean.png"
                _mask((10, 10), (3, 3, 6, 6)).save(gt_path)
            rows.append({"base_id": base_id, "content_label": label, "view": "clean", "profile": "clean", "image_path": str(clean_path), "tamper_mask_path": str(gt_path) if gt_path else None, "ignore_mask_path": None})
            for profile in ("news_meme_overlay", "zoom_crop"):
                sns_path = images / f"{base_id}__{profile}.png"
                sns = clean.resize((10, 10))
                ImageDraw.Draw(sns).line((0, 0, 9, 9), fill=(255, 255, 255))
                sns.save(sns_path)
                ignore_path = ignores / f"{base_id}__{profile}.png"
                _mask((10, 10), (0, 0, 1, 1)).save(ignore_path)
                rows.append({"base_id": base_id, "content_label": label, "view": "sns_aug", "profile": profile, "image_path": str(sns_path), "tamper_mask_path": str(gt_path) if gt_path else None, "ignore_mask_path": str(ignore_path)})
    meta_path = pair_root / "meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    final_metrics = {
        "marker": "SNSAUG_V2_DEPLOYABLE_POLICY_GATE_REPORT_OK",
        "per_profile": {
            "news_meme_overlay": {"tampered_recall": 0.8, "tampered_valid_mean_iou": 0.6},
            "zoom_crop": {"tampered_recall": 0.7, "tampered_valid_mean_iou": 0.5},
        },
    }
    (final_gate / "final_policy_gate_records.jsonl").write_text("", encoding="utf-8")
    (final_gate / "final_policy_gate_metrics.json").write_text(json.dumps(final_metrics), encoding="utf-8")
    return pair_root, meta_path, final_gate / "final_policy_gate_records.jsonl", final_gate / "final_policy_gate_metrics.json"


def safe_config(root: Path, pair_root: Path, meta_path: Path, final_records: Path, final_metrics: Path, mode: str = EXPORT_MODE, cached: Path | None = None) -> dict[str, object]:
    (root / "reports").mkdir(parents=True, exist_ok=True)
    roots = [str(pair_root), str(final_records.parent)]
    if cached:
        roots.append(str(cached.parent))
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_sida7b_diagnostic_baseline",
        "execution_mode": "approved_local_snsaug_v2_sida7b_diagnostic_baseline",
        "user_approval_text": "I_APPROVE_SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE",
        "mode": mode,
        "pair_root": str(pair_root),
        "meta_jsonl_path": str(meta_path),
        "cached_sida_outputs_path": str(cached) if cached else None,
        "final_policy_gate_records_path": str(final_records),
        "final_policy_gate_metrics_path": str(final_metrics),
        "approved_input_roots": roots,
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "sida"),
        "max_per_label_profile": 2,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def test_example_config_validates() -> None:
    config = load_snsaug_v2_sida7b_diagnostic_baseline_config(REPO_ROOT / "configs" / "evaluation" / "snsaug_v2_sida7b_diagnostic_baseline.example.json")
    assert_equal(validate_snsaug_v2_sida7b_diagnostic_baseline_config(config, require_exists=False), [], "example config validates")


def test_export_manifest_balanced_and_prompt_contains_tags() -> None:
    root = temp_root("cvf_0074_export_")
    _pair_root, meta_path, _final_records, _final_metrics = write_fixture(root)
    subset = build_balanced_subset(load_meta_rows(meta_path), max_per_label_profile=2)
    manifest, prompts = export_subset_rows(subset)
    counts: dict[tuple[str, str], int] = {}
    for row in manifest:
        counts[(row["content_label"], row["profile"])] = counts.get((row["content_label"], row["profile"]), 0) + 1
    assert_true(counts[("real", "clean")] == 2, "real clean balanced")
    assert_true(counts[("tampered", "zoom_crop")] == 2, "tampered zoom balanced")
    assert_true(all("[CLS]" in row["prompt"] and "[SEG]" in row["prompt"] for row in prompts), "prompt tags present")


def test_class_parser_and_cached_missing_masks() -> None:
    manifest = [{"image_id": "a", "base_id": "b", "profile": "zoom_crop", "profile_family": "type_b_global_geometry_degradation", "content_label": "tampered", "image_path": "/tmp/x.png", "tamper_mask_path": None, "ignore_mask_path": None}]
    cached = [{"image_id": "a", "sida_text_output": "[CLS] tampered\n[SEG] unavailable"}]
    records = parse_cached_sida_outputs(cached, manifest)
    assert_equal(parse_sida_text_output("[CLS] synthetic [SEG] none"), "synthetic", "text parser")
    assert_equal(parse_sida_text_output("[CLS] This image is classified as fully synthetic."), "synthetic", "fully synthetic parser")
    assert_equal(parse_sida_text_output("[CLS] This is an ai-generated image, not tampered."), "synthetic", "ai-generated parser")
    assert_equal(parse_sida_text_output("[CLS] manipulated image [SEG] mask"), "tampered", "manipulated parser")
    assert_equal(parse_sida_text_output_detail("[CLS] real but tampered")["pred_class"], "unknown", "ambiguous parser")
    assert_equal(normalize_class("authentic image"), "real", "class normalizer")
    assert_true(records[0]["mask_missing"] is True, "missing mask marked")
    assert_true(records[0]["valid_iou"] is None, "missing mask iou none")
    assert_true(records[0]["mask_extraction_failed"] is True, "seg without mask is extraction failure")
    no_seg = parse_cached_sida_outputs([{"image_id": "a", "sida_text_output": "[CLS] real"}], manifest)
    assert_true(no_seg[0]["mask_not_requested_or_not_generated"] is True, "no seg marked not requested/generated")


def test_mask_iou_and_mixed_gate_comparison_missing_profiles() -> None:
    root = temp_root("cvf_0074_iou_")
    gt = root / "gt.png"
    pred = root / "pred.png"
    _mask((8, 8), (2, 2, 5, 5)).save(gt)
    _mask((8, 8), (2, 2, 5, 5)).save(pred)
    assert_equal(mask_valid_iou(str(pred), str(gt)), 1.0, "perfect iou")
    comparison = compare_sida_to_mixed_gate({"zoom_crop": {"tampered_recall": 0.5}}, {})
    assert_true("zoom_crop" in comparison["missing_mixed_gate_profiles"], "missing mixed profile tracked")


def test_clean_missing_does_not_fail_strict_comparison() -> None:
    sida = strict_metrics(0.5)
    sida["clean"] = {"tampered_recall": 0.9}
    comparison = compare_sida_to_mixed_gate(sida, strict_metrics(0.7))
    assert_equal(comparison["missing_mixed_gate_profiles"], [], "clean excluded from strict missing profiles")
    assert_equal(comparison["missing_reference_only_profiles"], ["clean"], "clean reported as reference-only missing")
    findings = decide_sida_findings(
        {
            "type_a_local_overlay": {"row_count": 1, "tampered_recall": 0.5, "synthetic_recall": 0.0, "mask_missing_rate": 0.0},
            "type_b_global_geometry_degradation": {"row_count": 1, "tampered_recall": 0.5, "synthetic_recall": 0.0, "mask_missing_rate": 0.0},
        },
        comparison,
        cached_available=True,
    )
    assert_true("sida_vs_mixed_gate_strict_comparison_available" in findings, "strict comparison available with only clean missing")
    assert_true("clean_missing_only_expected_for_mixed_gate" in findings, "clean-only missing finding present")


def test_type_a_b_missing_still_fails_strict_comparison() -> None:
    sida = strict_metrics(0.5)
    sida["clean"] = {"tampered_recall": 0.9}
    gate = strict_metrics(0.7)
    gate.pop("news_meme_overlay")
    comparison = compare_sida_to_mixed_gate(sida, gate)
    assert_true("news_meme_overlay" in comparison["missing_mixed_gate_profiles"], "strict Type A missing profile fails")
    findings = decide_sida_findings(
        {
            "type_a_local_overlay": {"row_count": 1, "tampered_recall": 0.5, "synthetic_recall": 0.0, "mask_missing_rate": 0.0},
            "type_b_global_geometry_degradation": {"row_count": 1, "tampered_recall": 0.5, "synthetic_recall": 0.0, "mask_missing_rate": 0.0},
        },
        comparison,
        cached_available=True,
    )
    assert_true("sida_vs_mixed_gate_incomplete" in findings, "strict missing profile remains incomplete")


def test_mixed_gate_metrics_load_selected_schema_by_profile() -> None:
    root = temp_root("cvf_0074_gate_")
    metrics = root / "final_policy_gate_metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "comparison_against_fixed_original": {
                    "zoom_crop": {
                        "selected": {
                            "tampered_recall": 0.7,
                            "synthetic_recall": 0.9,
                            "real_fpr": 0.2,
                            "tampered_valid_mean_iou": 0.4,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    loaded = load_mixed_gate_metrics(str(metrics))
    assert_equal(loaded["zoom_crop"]["tampered_recall"], 0.7, "selected mixed gate profile loaded")
    comparison = compare_sida_to_mixed_gate({"zoom_crop": {"tampered_recall": 0.5, "synthetic_recall": 0.1, "real_fpr": 0.3, "tampered_valid_mean_iou": 0.2}}, loaded)
    assert_true("zoom_crop" not in comparison["missing_mixed_gate_profiles"], "mixed gate profile available")
    assert_equal(comparison["profiles"]["zoom_crop"]["delta_sida_minus_mixed_gate"]["synthetic_recall"], -0.8, "synthetic delta")


def test_mixed_gate_metrics_per_profile_and_records_fallback() -> None:
    root = temp_root("cvf_0074_gate_fallback_")
    metrics = root / "final_policy_gate_metrics.json"
    metrics.write_text(
        json.dumps({"per_profile": {"zoom_crop": {"tampered_recall": 0.6, "tampered_valid_mean_iou": 0.4}}}),
        encoding="utf-8",
    )
    loaded = load_mixed_gate_metrics(str(metrics))
    assert_equal(loaded["zoom_crop"]["tampered_recall"], 0.6, "per_profile fallback loaded")
    empty_metrics = root / "empty_metrics.json"
    records = root / "final_policy_gate_records.jsonl"
    empty_metrics.write_text(json.dumps({"marker": "x"}), encoding="utf-8")
    rows = [
        {"profile": "zoom_crop", "content_label": "tampered", "pred_class": "tampered", "valid_iou": 0.8, "p_tampered": 0.9},
        {"profile": "zoom_crop", "content_label": "tampered", "pred_class": "real", "valid_iou": 0.4, "p_tampered": 0.2},
        {"profile": "zoom_crop", "content_label": "real", "pred_class": "tampered", "p_tampered": 0.7},
    ]
    with open(records, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    fallback = load_mixed_gate_metrics(str(empty_metrics), str(records))
    assert_equal(fallback["zoom_crop"]["tampered_recall"], 0.5, "records fallback tampered recall")
    assert_equal(fallback["zoom_crop"]["tampered_valid_mean_iou"], 0.6000000000000001, "records fallback iou mean")


def test_export_only_run_writes_artifact_and_forbids_conclusion() -> None:
    root = temp_root("cvf_0074_run_")
    pair_root, meta_path, final_records, final_metrics = write_fixture(root)
    config = safe_config(root, pair_root, meta_path, final_records, final_metrics)
    dry = run_snsaug_v2_sida7b_diagnostic_baseline(config, dry_run=True)
    assert_true(dry["dry_run"] is True, "dry-run")
    summary = run_snsaug_v2_sida7b_diagnostic_baseline(config)
    assert_equal(summary["marker"], MARKER, "marker")
    report = Path(summary["output_paths"]["sida7b_diagnostic_report"]).read_text(encoding="utf-8")
    assert_true("no performance conclusion is allowed" in report, "export-only forbids conclusion")
    artifact = json.loads(Path(summary["output_paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert_equal(artifact["decision"], "sida_cached_outputs_missing", "missing cached decision")


def test_cached_run_writes_records() -> None:
    root = temp_root("cvf_0074_cached_")
    pair_root, meta_path, final_records, final_metrics = write_fixture(root)
    cached = root / "cached" / "sida.jsonl"
    cached.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"base_id": "tampered_0", "profile": "zoom_crop", "content_label": "tampered", "sida_text_output": "[CLS] tampered"},
        {"base_id": "real_0", "profile": "news_meme_overlay", "content_label": "real", "sida_text_output": "[CLS] real"},
    ]
    with open(cached, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    config = safe_config(root, pair_root, meta_path, final_records, final_metrics, mode=CACHED_MODE, cached=cached)
    summary = run_snsaug_v2_sida7b_diagnostic_baseline(config)
    records = [json.loads(line) for line in Path(summary["output_paths"]["sida7b_diagnostic_records"]).read_text(encoding="utf-8").splitlines()]
    assert_equal(len(records), 2, "cached records written")
    corrected = Path(summary["output_paths"]["sida7b_corrected_diagnostic_report"]).read_text(encoding="utf-8")
    assert_true("does not claim localization failure" in corrected, "corrected report avoids localization failure claim")
    clean_safe = json.loads(Path(summary["output_paths"]["sida7b_clean_safe_vs_mixed_gate_comparison"]).read_text(encoding="utf-8"))
    assert_true(clean_safe["clean_safe"] is True, "clean-safe comparison json written")
    clean_safe_report = Path(summary["output_paths"]["sida7b_clean_safe_vs_mixed_gate_comparison_report"]).read_text(encoding="utf-8")
    assert_true("Clean-Safe SIDA vs Mixed Gate Comparison" in clean_safe_report, "clean-safe comparison markdown written")


def test_docs_marker_present() -> None:
    text = (REPO_ROOT / "docs" / "snsaug_v2_sida7b_diagnostic_baseline.md").read_text(encoding="utf-8")
    assert_true(MARKER in text, "docs marker present")


def main() -> int:
    tests = [
        test_example_config_validates,
        test_export_manifest_balanced_and_prompt_contains_tags,
        test_class_parser_and_cached_missing_masks,
        test_mask_iou_and_mixed_gate_comparison_missing_profiles,
        test_clean_missing_does_not_fail_strict_comparison,
        test_type_a_b_missing_still_fails_strict_comparison,
        test_mixed_gate_metrics_load_selected_schema_by_profile,
        test_mixed_gate_metrics_per_profile_and_records_fallback,
        test_export_only_run_writes_artifact_and_forbids_conclusion,
        test_cached_run_writes_records,
        test_docs_marker_present,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
