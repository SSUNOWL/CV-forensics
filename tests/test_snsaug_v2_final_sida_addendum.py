#!/usr/bin/env python3
"""Plain Python tests for final SIDA addendum reporting."""

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

from cv_forensics.snsaug_v2_final_sida_addendum import (  # noqa: E402
    MARKER,
    build_comparison_rows,
    load_snsaug_v2_final_sida_addendum_config,
    render_tsv,
    run_snsaug_v2_final_sida_addendum,
    validate_snsaug_v2_final_sida_addendum_config,
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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sida_summary() -> dict[str, object]:
    return {
        "marker": "SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE_OK",
        "summary": {
            "type_a_local_overlay": {
                "row_count": 18,
                "accuracy": 0.4,
                "synthetic_recall": 0.0,
                "tampered_recall": 0.33,
                "tampered_valid_mean_iou": None,
                "mask_missing_rate": 1.0,
            },
            "type_b_global_geometry_degradation": {
                "row_count": 18,
                "accuracy": 0.6,
                "synthetic_recall": 0.0,
                "tampered_recall": 0.67,
                "tampered_valid_mean_iou": None,
                "mask_missing_rate": 1.0,
            },
            "clean": {
                "row_count": 6,
                "accuracy": 0.5,
                "synthetic_recall": 0.0,
                "tampered_recall": 0.5,
                "tampered_valid_mean_iou": None,
                "mask_missing_rate": 1.0,
            },
        },
        "decision_findings": [
            "sida_classification_collapse_synthetic",
            "sida_local_overlay_worse_than_global",
            "sida_localization_unavailable_mask_missing",
            "sida_vs_mixed_gate_strict_comparison_available",
            "clean_missing_only_expected_for_mixed_gate",
        ],
    }


def clean_safe_comparison() -> dict[str, object]:
    return {
        "marker": "SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE_OK",
        "clean_safe": True,
        "strict_comparison_available": True,
        "missing_mixed_gate_profiles": [],
        "missing_reference_only_profiles": ["clean"],
        "reference_only_profiles": ["clean"],
        "clean_reference": {"profile": "clean", "reference_only": True},
        "profiles": {
            "clean": {"sida_tampered_recall": 0.5},
            "news_meme_overlay": {
                "sida_tampered_recall": 0.25,
                "sida_synthetic_recall": 0.0,
                "sida_real_fpr": 0.2,
                "sida_valid_iou": None,
                "mixed_gate_tampered_recall": 0.75,
                "mixed_gate_synthetic_recall": 0.8,
                "mixed_gate_real_fpr": 0.1,
                "mixed_gate_valid_iou": 0.5,
                "tampered_recall_delta_sida_minus_gate": -0.5,
                "valid_iou_delta_sida_minus_gate": None,
            },
            "zoom_crop": {
                "sida_tampered_recall": 0.75,
                "sida_synthetic_recall": 0.0,
                "sida_real_fpr": 0.1,
                "sida_valid_iou": None,
                "mixed_gate_tampered_recall": 0.7,
                "mixed_gate_synthetic_recall": 0.9,
                "mixed_gate_real_fpr": 0.05,
                "mixed_gate_valid_iou": 0.45,
                "tampered_recall_delta_sida_minus_gate": 0.05,
                "valid_iou_delta_sida_minus_gate": None,
            },
        },
    }


def write_fixture(root: Path) -> tuple[Path, Path]:
    sida_root = root / "sida"
    gate_root = root / "gate"
    sida_root.mkdir(parents=True, exist_ok=True)
    gate_root.mkdir(parents=True, exist_ok=True)
    write_json(sida_root / "sida7b_corrected_type_a_type_b_summary.json", sida_summary())
    write_json(sida_root / "sida7b_clean_safe_vs_mixed_gate_comparison.json", clean_safe_comparison())
    write_json(
        sida_root / "sida7b_raw_output_audit.json",
        {
            "marker": "SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE_OK",
            "raw_contains_synthetic_count": 0,
            "raw_contains_generated_count": 0,
            "raw_contains_seg_count": 4,
        },
    )
    write_json(sida_root / "artifact_manifest.json", {"marker": "SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE_OK"})
    write_json(
        gate_root / "final_policy_gate_metrics.json",
        {
            "marker": "SNSAUG_V2_DEPLOYABLE_POLICY_GATE_REPORT_OK",
            "selected_selector": "mixed_feature_gate",
        },
    )
    (gate_root / "final_policy_gate_report.md").write_text("SNSAUG_V2_DEPLOYABLE_POLICY_GATE_REPORT_OK\n", encoding="utf-8")
    write_json(gate_root / "artifact_manifest.json", {"marker": "SNSAUG_V2_DEPLOYABLE_POLICY_GATE_REPORT_OK"})
    return sida_root, gate_root


def safe_config(root: Path, sida_root: Path, gate_root: Path) -> dict[str, object]:
    (root / "reports").mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_final_sida_addendum",
        "execution_mode": "approved_local_snsaug_v2_final_sida_addendum",
        "user_approval_text": "I_APPROVE_SNSAUG_V2_FINAL_SIDA_ADDENDUM",
        "sida_output_root": str(sida_root),
        "mixed_gate_output_root": str(gate_root),
        "approved_input_roots": [str(sida_root), str(gate_root)],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "addendum"),
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def test_example_config_validates() -> None:
    config = load_snsaug_v2_final_sida_addendum_config(REPO_ROOT / "configs" / "evaluation" / "snsaug_v2_final_sida_addendum.example.json")
    assert_equal(validate_snsaug_v2_final_sida_addendum_config(config, require_exists=False), [], "example config validates")


def test_rows_and_tsv_exclude_clean() -> None:
    rows = build_comparison_rows(clean_safe_comparison())
    profiles = [row["profile"] for row in rows]
    assert_true("clean" not in profiles, "clean excluded from strict table")
    assert_true("news_meme_overlay" in profiles, "Type A profile included")
    assert_true("zoom_crop" in profiles, "Type B profile included")
    table = render_tsv(rows)
    assert_true(table.startswith("profile\tprofile_family"), "tsv header present")
    assert_true("\tclean\t" not in table and "\nclean\t" not in table, "tsv excludes clean")


def test_run_writes_required_reports_and_manifest() -> None:
    root = temp_root("cvf_0075_run_")
    sida_root, gate_root = write_fixture(root)
    config = safe_config(root, sida_root, gate_root)
    dry = run_snsaug_v2_final_sida_addendum(config, dry_run=True)
    assert_true(dry["dry_run"] is True, "dry-run returned plan")
    assert_true(not Path(dry["output_paths"]["artifact_manifest"]).exists(), "dry-run writes no manifest")
    summary = run_snsaug_v2_final_sida_addendum(config)
    assert_equal(summary["marker"], MARKER, "marker")
    for key in (
        "final_sida_addendum_report",
        "final_sida_addendum_notion_summary",
        "final_sida_vs_mixed_gate_table",
        "future_work_snsaware_model",
        "artifact_manifest",
    ):
        assert_true(Path(summary["output_paths"][key]).exists(), f"{key} written")
    report = Path(summary["output_paths"]["final_sida_addendum_report"]).read_text(encoding="utf-8")
    assert_true("What SIDA-7B Was Used For" in report, "usage section present")
    assert_true("What Was Actually Measured" in report, "measured section present")
    assert_true("What Was Not Measured" in report, "not measured section present")
    assert_true("SIDA localization performance was not measured" in report, "mask-missing localization caveat present")
    assert_true("Type A local overlay tampered recall" in report, "Type A result present")
    assert_true("Comparison With 0072 mixed_feature_gate" in report, "mixed gate comparison present")
    notion = Path(summary["output_paths"]["final_sida_addendum_notion_summary"]).read_text(encoding="utf-8")
    assert_true("classification diagnostic only" in notion, "notion classification-only caveat")
    future = Path(summary["output_paths"]["future_work_snsaware_model"]).read_text(encoding="utf-8")
    for phrase in ("SIDA-13B", "VLM-guided nuisance preprocessor", "dual-branch model", "Distill SIDA-like VLM"):
        assert_true(phrase in future, f"future work includes {phrase}")
    table = Path(summary["output_paths"]["final_sida_vs_mixed_gate_table"]).read_text(encoding="utf-8")
    assert_true("news_meme_overlay" in table and "zoom_crop" in table, "strict profiles in table")
    assert_true("clean\t" not in table, "clean not a strict table row")
    artifact = json.loads(Path(summary["output_paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert_true(artifact["no_training"] is True, "no_training flag")
    assert_true(artifact["no_download"] is True, "no_download flag")
    assert_true(artifact["network_used"] is False, "network_used false")
    assert_true(artifact["strict_comparison_available"] is True, "strict comparison availability")


def test_docs_marker_present() -> None:
    text = (REPO_ROOT / "docs" / "snsaug_v2_final_sida_addendum.md").read_text(encoding="utf-8")
    assert_true(MARKER in text, "docs marker present")


def main() -> int:
    tests = [
        test_example_config_validates,
        test_rows_and_tsv_exclude_clean,
        test_run_writes_required_reports_and_manifest,
        test_docs_marker_present,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
