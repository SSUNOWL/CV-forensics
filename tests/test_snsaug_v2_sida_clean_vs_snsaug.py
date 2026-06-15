#!/usr/bin/env python3
"""Plain Python tests for 0079 SIDA clean-vs-SNSAug paired diagnostic."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TMP_PARENT = Path("/home/rlatjswo/.codex/memories")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2_sida_clean_vs_snsaug import (  # noqa: E402
    CACHED_MODE,
    EXPORT_MODE,
    MARKER,
    find_clean_counterparts,
    load_snsaug_v2_sida_clean_vs_snsaug_config,
    pair_cached_outputs,
    run_snsaug_v2_sida_clean_vs_snsaug,
    summarize_pairs,
    validate_snsaug_v2_sida_clean_vs_snsaug_config,
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


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def fixture_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    meta = [
        {"base_id": "b1", "content_label": "synthetic", "profile": "clean", "view": "clean", "image_path": "/tmp/b1_clean.png"},
        {"base_id": "b2", "content_label": "tampered", "profile": "clean", "view": "clean", "image_path": "/tmp/b2_clean.png", "tamper_mask_path": "/tmp/b2_mask.png"},
        {"base_id": "b3", "content_label": "real", "profile": "clean", "view": "clean", "image_path": "/tmp/b3_clean.png"},
    ]
    sns = [
        {"image_id": "b1__tiktok", "base_id": "b1", "content_label": "synthetic", "profile": "tiktok_like", "image_path": "/tmp/b1_tiktok.png"},
        {"image_id": "b2__zoom", "base_id": "b2", "content_label": "tampered", "profile": "zoom_crop", "image_path": "/tmp/b2_zoom.png"},
        {"image_id": "b3__meme", "base_id": "b3", "content_label": "real", "profile": "news_meme_overlay", "image_path": "/tmp/b3_meme.png"},
    ]
    return sns, meta


def safe_config(root: Path, mode: str, manifest: Path, meta: Path, clean: Path | None = None, sns: Path | None = None) -> dict[str, object]:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_sida_clean_vs_snsaug",
        "execution_mode": "approved_local_snsaug_v2_sida_clean_vs_snsaug",
        "user_approval_text": "I_APPROVE_SNSAUG_V2_SIDA_CLEAN_VS_SNSAUG",
        "mode": mode,
        "snsaug_sida_export_manifest_path": str(manifest),
        "meta_jsonl_path": str(meta),
        "cached_clean_sida_outputs_path": str(clean) if clean else "",
        "cached_snsaug_sida_outputs_path": str(sns) if sns else "",
        "approved_input_roots": [str(root)],
        "approved_output_roots": [str(reports)],
        "output_root": str(reports / "out"),
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def test_example_config_validates() -> None:
    config = load_snsaug_v2_sida_clean_vs_snsaug_config(REPO_ROOT / "configs" / "evaluation" / "snsaug_v2_sida_clean_vs_snsaug.example.json")
    assert_equal(validate_snsaug_v2_sida_clean_vs_snsaug_config(config, require_exists=False), [], "example config validates")


def test_clean_counterpart_matching_and_export_mode() -> None:
    root = temp_root("cvf_0079_export_")
    sns_rows, meta_rows = fixture_rows()
    manifest_path = root / "sida_manifest.jsonl"
    meta_path = root / "meta.jsonl"
    write_jsonl(manifest_path, sns_rows)
    write_jsonl(meta_path, meta_rows)
    matched = find_clean_counterparts(sns_rows, meta_rows)
    assert_equal(len(matched), 3, "matched clean counterparts")
    assert_true(matched[0]["clean_image_path"].endswith("b1_clean.png"), "clean image selected")
    config = safe_config(root, EXPORT_MODE, manifest_path, meta_path)
    dry = run_snsaug_v2_sida_clean_vs_snsaug(config, dry_run=True)
    for path in dry["output_paths"].values():
        assert_true(not Path(path).exists(), f"dry-run did not write {path}")
    summary = run_snsaug_v2_sida_clean_vs_snsaug(config)
    assert_equal(summary["manifest_count"], 3, "manifest count")
    prompt_text = Path(summary["output_paths"]["sida_clean_prompt_list"]).read_text(encoding="utf-8")
    assert_true("[CLS]" in prompt_text and "[SEG]" in prompt_text, "prompt tags")
    assert_true(Path(summary["output_paths"]["run_sida_clean_external_template"]).exists(), "external template written")


def test_cached_eval_pairs_synthetic_and_tampered_rules() -> None:
    clean = [
        {"base_id": "b1", "content_label": "synthetic", "profile": "clean", "sida_text_output": "[CLS] synthetic", "sida_mask_path": ""},
        {"base_id": "b2", "content_label": "tampered", "profile": "clean", "sida_text_output": "[CLS] tampered", "valid_iou": 0.8, "sida_mask_path": "/tmp/mask.png"},
        {"base_id": "b3", "content_label": "real", "profile": "clean", "sida_text_output": "[CLS] real"},
    ]
    sns = [
        {"base_id": "b1", "content_label": "synthetic", "profile": "tiktok_like", "sida_text_output": "[CLS] tampered", "sida_mask_path": "/tmp/fp.png"},
        {"base_id": "b2", "content_label": "tampered", "profile": "zoom_crop", "sida_text_output": "[CLS] tampered", "valid_iou": 0.3},
        {"base_id": "b3", "content_label": "real", "profile": "news_meme_overlay", "sida_text_output": "[CLS] synthetic"},
    ]
    pairs = pair_cached_outputs(clean, sns)
    assert_equal(len(pairs), 3, "paired rows")
    synthetic = [row for row in pairs if row["content_label"] == "synthetic"][0]
    assert_true(synthetic["clean_tamper_iou"] is None and synthetic["sns_tamper_iou"] is None, "synthetic excluded from tamper IoU")
    assert_equal(synthetic["synthetic_mask_false_positive_sns"], 1, "synthetic mask fpr tracked")
    tampered = [row for row in pairs if row["content_label"] == "tampered"][0]
    assert_equal(tampered["delta_iou"], -0.5, "tampered delta iou")
    summary = summarize_pairs(pairs)
    assert_equal(summary["type_a_local_overlay"]["row_count"], 2, "type A rows")
    assert_equal(summary["type_b_global_geometry_degradation"]["row_count"], 1, "type B rows")
    assert_equal(summary["strict_type_a_plus_type_b"]["row_count"], 3, "strict rows")


def test_cached_run_writes_artifacts() -> None:
    root = temp_root("cvf_0079_cached_")
    sns_rows, meta_rows = fixture_rows()
    manifest_path = root / "sida_manifest.jsonl"
    meta_path = root / "meta.jsonl"
    clean_path = root / "clean.jsonl"
    sns_path = root / "sns.jsonl"
    write_jsonl(manifest_path, sns_rows)
    write_jsonl(meta_path, meta_rows)
    write_jsonl(clean_path, [{"base_id": "b2", "content_label": "tampered", "profile": "clean", "sida_pred_class": "tampered", "valid_iou": 0.9, "sida_mask_path": "/tmp/mask.png"}])
    write_jsonl(sns_path, [{"base_id": "b2", "content_label": "tampered", "profile": "zoom_crop", "sida_pred_class": "tampered", "valid_iou": 0.4}])
    config = safe_config(root, CACHED_MODE, manifest_path, meta_path, clean_path, sns_path)
    summary = run_snsaug_v2_sida_clean_vs_snsaug(config)
    assert_equal(summary["pair_count"], 1, "pair count")
    report = Path(summary["output_paths"]["sida_clean_vs_snsaug_report"]).read_text(encoding="utf-8")
    assert_true("Synthetic samples" in report or "synthetic samples" in report, "synthetic caveat")
    artifact = json.loads(Path(summary["output_paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert_equal(artifact["marker"], MARKER, "artifact marker")


def test_docs_marker_present() -> None:
    text = (REPO_ROOT / "docs" / "snsaug_v2_sida_clean_vs_snsaug.md").read_text(encoding="utf-8")
    assert_true(MARKER in text, "docs marker present")


def main() -> int:
    tests = [
        test_example_config_validates,
        test_clean_counterpart_matching_and_export_mode,
        test_cached_eval_pairs_synthetic_and_tampered_rules,
        test_cached_run_writes_artifacts,
        test_docs_marker_present,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
