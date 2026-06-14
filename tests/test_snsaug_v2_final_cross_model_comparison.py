#!/usr/bin/env python3
"""Plain Python tests for final cross-model comparison matrix."""

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

from cv_forensics.snsaug_v2_final_cross_model_comparison import (  # noqa: E402
    MARKER,
    build_matrix,
    load_snsaug_v2_final_cross_model_comparison_config,
    render_markdown,
    render_tsv,
    run_snsaug_v2_final_cross_model_comparison,
    validate_snsaug_v2_final_cross_model_comparison_config,
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


def unified_summary() -> dict[str, object]:
    return {
        "marker": "SNSAUG_V2_0076_UNIFIED_SIDA_GATE_COMPARISON_OK",
        "families": {
            "type_a_local_overlay": {
                "sida_synthetic_recall": 0.10,
                "sida_real_fpr": 0.20,
                "sida_tampered_recall": 0.30,
                "sida_valid_iou": 0.40,
                "sida_synthetic_tampered_fpr": 0.50,
                "sida_detected_only_iou": 0.60,
                "sida_ignore_capture": 1.0,
                "mixed_gate_synthetic_recall": 0.91,
                "mixed_gate_real_fpr": 0.08,
                "mixed_gate_tampered_recall": 0.72,
                "mixed_gate_valid_iou": 0.44,
            },
            "type_b_global_geometry_degradation": {
                "sida_synthetic_recall": 0.11,
                "sida_real_fpr": 0.21,
                "sida_tampered_recall": 0.31,
                "sida_valid_iou": 0.41,
                "mixed_gate_synthetic_recall": 0.92,
                "mixed_gate_real_fpr": 0.09,
                "mixed_gate_tampered_recall": 0.73,
                "mixed_gate_valid_iou": 0.45,
            },
            "strict_type_a_plus_type_b": {
                "sida_synthetic_recall": 0.12,
                "sida_real_fpr": 0.22,
                "sida_tampered_recall": 0.32,
                "sida_valid_iou": 0.42,
                "mixed_gate_synthetic_recall": 0.93,
                "mixed_gate_real_fpr": 0.10,
                "mixed_gate_tampered_recall": 0.74,
                "mixed_gate_valid_iou": 0.46,
            },
        },
    }


def write_fixture(root: Path, bad_optional: bool = False) -> tuple[Path, Path, Path, Path]:
    unified_root = root / "unified"
    gate_root = root / "gate"
    baseline_root = root / "baseline"
    failed_root = root / "failed"
    unified_root.mkdir(parents=True, exist_ok=True)
    gate_root.mkdir(parents=True, exist_ok=True)
    baseline_root.mkdir(parents=True, exist_ok=True)
    failed_root.mkdir(parents=True, exist_ok=True)
    write_json(unified_root / "unified_sida_gate_comparison_summary.json", unified_summary())
    (unified_root / "unified_sida_gate_profile_table.tsv").write_text("system\tfamily\n", encoding="utf-8")
    (unified_root / "unified_sida_gate_comparison_report.md").write_text("report\n", encoding="utf-8")
    write_json(unified_root / "artifact_manifest.json", {"marker": "SNSAUG_V2_0076_UNIFIED_SIDA_GATE_COMPARISON_OK"})
    write_json(gate_root / "final_policy_gate_metrics.json", {"marker": "SNSAUG_V2_DEPLOYABLE_POLICY_GATE_REPORT_OK"})
    write_json(gate_root / "deployable_policy_gate_spec.json", {"selected_selector": "mixed_feature_gate"})
    write_json(gate_root / "artifact_manifest.json", {"marker": "SNSAUG_V2_DEPLOYABLE_POLICY_GATE_REPORT_OK"})
    if bad_optional:
        (baseline_root / "metrics.json").write_text("{not-json", encoding="utf-8")
    else:
        write_json(
            baseline_root / "metrics.json",
            {
                "families": {
                    "strict_type_a_plus_type_b": {
                        "pre_sns_baseline_synthetic_recall": 0.2,
                        "pre_sns_baseline_real_fpr": 0.3,
                        "pre_sns_baseline_tampered_recall": 0.4,
                        "pre_sns_baseline_valid_iou": 0.5,
                    }
                }
            },
        )
    return unified_root, gate_root, baseline_root, failed_root


def safe_config(root: Path, unified_root: Path, gate_root: Path, baseline_root: Path, failed_root: Path) -> dict[str, object]:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_final_cross_model_comparison",
        "execution_mode": "approved_local_snsaug_v2_final_cross_model_comparison",
        "user_approval_text": "I_APPROVE_SNSAUG_V2_FINAL_CROSS_MODEL_COMPARISON",
        "unified_sida_gate_output_root": str(unified_root),
        "mixed_gate_output_root": str(gate_root),
        "pre_sns_baseline_root": str(baseline_root),
        "failed_single_model_finetune_root": str(failed_root),
        "approved_input_roots": [str(unified_root), str(gate_root), str(baseline_root), str(failed_root)],
        "approved_output_roots": [str(reports)],
        "output_root": str(reports / "final"),
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def test_example_config_validates() -> None:
    config = load_snsaug_v2_final_cross_model_comparison_config(
        REPO_ROOT / "configs" / "evaluation" / "snsaug_v2_final_cross_model_comparison.example.json"
    )
    assert_equal(validate_snsaug_v2_final_cross_model_comparison_config(config, require_exists=False), [], "example config validates")


def test_safety_flags_required() -> None:
    config = load_snsaug_v2_final_cross_model_comparison_config(
        REPO_ROOT / "configs" / "evaluation" / "snsaug_v2_final_cross_model_comparison.example.json"
    )
    config["no_training"] = False
    errors = validate_snsaug_v2_final_cross_model_comparison_config(config, require_exists=False)
    assert_true("no_training must be true" in errors, "no_training false rejected")


def test_matrix_tables_and_na_behavior() -> None:
    root = temp_root("cvf_0077_matrix_")
    unified_root, gate_root, baseline_root, failed_root = write_fixture(root, bad_optional=True)
    config = safe_config(root, unified_root, gate_root, baseline_root, failed_root)
    artifact = run_snsaug_v2_final_cross_model_comparison(config)
    matrix_path = Path(artifact["output_paths"]["final_cross_model_comparison_matrix"])
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert_equal(matrix["main_table_systems"], ["SIDA-7B", "mixed_feature_gate"], "main systems")
    assert_equal([entry["system"] for entry in matrix["model_entries"]], ["SIDA-7B", "mixed_feature_gate"], "main entries only")
    assert_equal(matrix["ablation_table_systems"], ["pre_sns_baseline", "failed_single_model_finetune", "mixed_feature_gate"], "ablation systems")
    assert_true(matrix["ablation_entries"][0]["metrics"]["strict_type_a_plus_type_b"]["synthetic_recall"] is None, "bad baseline metric becomes null")
    markdown = Path(artifact["output_paths"]["final_cross_model_comparison_report"]).read_text(encoding="utf-8")
    assert_true("not a drop-in replacement for SIDA" in markdown, "non-drop-in language present")
    assert_true("SIDA emits 3-way labels and masks" in markdown, "SIDA capability caveat present")
    tsv = Path(artifact["output_paths"]["final_cross_model_comparison_table"]).read_text(encoding="utf-8")
    assert_true("SIDA-7B\ttype_a_local_overlay" in tsv, "SIDA row in TSV")
    assert_true("mixed_feature_gate\tstrict_type_a_plus_type_b" in tsv, "gate strict row in TSV")
    assert_true(Path(artifact["output_paths"]["artifact_manifest"]).exists(), "artifact manifest written")


def test_dry_run_writes_no_output_artifacts() -> None:
    root = temp_root("cvf_0077_dry_")
    unified_root, gate_root, baseline_root, failed_root = write_fixture(root)
    config = safe_config(root, unified_root, gate_root, baseline_root, failed_root)
    plan = run_snsaug_v2_final_cross_model_comparison(config, dry_run=True)
    assert_true(plan["dry_run"] is True, "dry-run plan returned")
    for path in plan["output_paths"].values():
        assert_true(not Path(path).exists(), f"dry-run did not write {path}")


def test_renderers_keep_guardrails() -> None:
    matrix = build_matrix(
        {
            "unified_root": "/tmp/unified",
            "mixed_gate_root": "/tmp/gate",
            "unified_summary": unified_summary(),
            "pre_sns_baseline": {"system": "pre_sns_baseline", "loaded": False, "metrics": {family: {metric: None for metric in ("synthetic_recall", "real_fpr", "tampered_recall", "valid_iou")} for family in ("type_a_local_overlay", "type_b_global_geometry_degradation", "strict_type_a_plus_type_b")}},
            "failed_single_model_finetune": {"system": "failed_single_model_finetune", "loaded": False, "metrics": {family: {metric: None for metric in ("synthetic_recall", "real_fpr", "tampered_recall", "valid_iou")} for family in ("type_a_local_overlay", "type_b_global_geometry_degradation", "strict_type_a_plus_type_b")}},
        },
        {"pre_sns_baseline_root": "", "failed_single_model_finetune_root": ""},
    )
    markdown = render_markdown(matrix)
    tsv = render_tsv(matrix)
    assert_true("SIDA and mixed_feature_gate are not identical systems" in markdown, "systems caveat")
    assert_true(tsv.startswith("system\tfamily\tsynthetic_recall"), "tsv header")


def test_docs_marker_present() -> None:
    text = (REPO_ROOT / "docs" / "snsaug_v2_final_cross_model_comparison.md").read_text(encoding="utf-8")
    assert_true(MARKER in text, "docs marker present")


def main() -> int:
    tests = [
        test_example_config_validates,
        test_safety_flags_required,
        test_matrix_tables_and_na_behavior,
        test_dry_run_writes_no_output_artifacts,
        test_renderers_keep_guardrails,
        test_docs_marker_present,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
