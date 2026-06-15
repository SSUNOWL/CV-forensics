#!/usr/bin/env python3
"""Plain Python tests for 0077a ablation metric filling."""

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

from cv_forensics.snsaug_v2_fill_final_cross_model_ablation_metrics import (  # noqa: E402
    MARKER,
    aggregate_family_metrics,
    load_snsaug_v2_fill_final_cross_model_ablation_metrics_config,
    parse_final_decision_markdown_table,
    run_snsaug_v2_fill_final_cross_model_ablation_metrics,
    select_failed_model,
    validate_snsaug_v2_fill_final_cross_model_ablation_metrics_config,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def assert_close(actual: float | None, expected: float, message: str) -> None:
    if actual is None or abs(actual - expected) > 1e-9:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def temp_root(prefix: str) -> Path:
    TMP_PARENT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(TMP_PARENT)))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def decision_markdown() -> str:
    return """# Report

| model | profile | acc | f1 | real_fpr | syn_rec | tamp_rec | loc_act | valid_iou | high_mask |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pre_sns_baseline | tiktok_like | 0.400 | 0.300 | 0.100 | 0.500 | 0.200 | 0.200 | 0.010 | 0.020 |
| pre_sns_baseline | instagram_story_like | 0.600 | 0.500 | 0.300 | 0.700 | 0.400 | 0.400 | 0.030 | 0.040 |
| pre_sns_baseline | zoom_crop | 0.800 | 0.700 | 0.500 | 0.900 | 0.600 | 0.600 | 0.050 | 0.060 |
| snsaug_0060b_over_tampered_30x3 | tiktok_like | 0.300 | 0.200 | 0.900 | 0.000 | 0.900 | 0.900 | 0.030 | 0.400 |
| snsaug_0064g_hybrid_30x3 | tiktok_like | 0.320 | 0.160 | 1.000 | 0.000 | 0.950 | 1.000 | 0.003 | 0.025 |
| snsaug_0064g_hybrid_30x3 | instagram_story_like | 0.340 | 0.180 | 1.000 | 0.000 | 1.000 | 1.000 | 0.005 | 0.000 |
"""


def matrix_payload() -> dict[str, object]:
    return {
        "marker": "SNSAUG_V2_0077_FINAL_CROSS_MODEL_COMPARISON_OK",
        "source_roots": {"0077": "fixture"},
        "models": {
            "pre_sns_baseline": {
                "type_a_local_overlay": {"synthetic_recall": None},
                "type_b_global_geometry_degradation": {"synthetic_recall": None},
                "strict_type_a_plus_type_b": {"synthetic_recall": None},
            },
            "failed_single_model_finetune": {
                "type_a_local_overlay": {},
                "type_b_global_geometry_degradation": {},
                "strict_type_a_plus_type_b": {},
            },
            "SIDA-7B": {
                "type_a_local_overlay": {"tampered_recall": 0.41},
            },
            "mixed_feature_gate": {
                "type_a_local_overlay": {"synthetic_recall": 1.0, "real_fpr": 0.1, "tampered_recall": 1.0, "valid_iou": 0.5},
                "type_b_global_geometry_degradation": {"synthetic_recall": 0.8, "real_fpr": 0.2, "tampered_recall": 0.6, "valid_iou": 0.3},
                "strict_type_a_plus_type_b": {"synthetic_recall": 0.9, "real_fpr": 0.15, "tampered_recall": 0.8, "valid_iou": 0.4},
            },
        },
        "recommendations": {
            "main_table": ["SIDA-7B", "mixed_feature_gate"],
            "ablation_table": ["pre_sns_baseline", "failed_single_model_finetune", "mixed_feature_gate"],
        },
    }


def write_fixture(root: Path) -> tuple[Path, Path]:
    comparison_root = root / "snsaug_v2_0077_final_cross_model_comparison_20260614_220753"
    decision_root = root / "snsaug_v2_final_decision_report_20260610_112419"
    comparison_root.mkdir(parents=True, exist_ok=True)
    decision_root.mkdir(parents=True, exist_ok=True)
    write_json(comparison_root / "final_cross_model_comparison_matrix.json", matrix_payload())
    write_json(comparison_root / "artifact_manifest.json", {"marker": "SNSAUG_V2_0077_FINAL_CROSS_MODEL_COMPARISON_OK"})
    (decision_root / "final_decision_report.md").write_text(decision_markdown(), encoding="utf-8")
    return comparison_root, decision_root / "final_decision_report.md"


def safe_config(root: Path) -> dict[str, object]:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_fill_final_cross_model_ablation_metrics",
        "execution_mode": "approved_local_snsaug_v2_fill_final_cross_model_ablation_metrics",
        "user_approval_text": "I_APPROVE_SNSAUG_V2_FILL_FINAL_CROSS_MODEL_ABLATION_METRICS",
        "comparison_search_root": str(root),
        "decision_report_search_root": str(root),
        "comparison_output_root": "",
        "decision_report_path": "",
        "approved_input_roots": [str(root)],
        "approved_output_roots": [str(reports)],
        "output_root": str(reports / "filled"),
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def test_example_config_validates() -> None:
    config = load_snsaug_v2_fill_final_cross_model_ablation_metrics_config(
        REPO_ROOT / "configs" / "evaluation" / "snsaug_v2_fill_final_cross_model_ablation_metrics.example.json"
    )
    assert_equal(validate_snsaug_v2_fill_final_cross_model_ablation_metrics_config(config, require_exists=False), [], "example config validates")


def test_table_parser_and_failed_selection() -> None:
    rows = parse_final_decision_markdown_table(decision_markdown())
    assert_equal(len(rows), 6, "row count")
    assert_equal(select_failed_model(rows), "snsaug_0064g_hybrid_30x3", "0064g hybrid preferred")
    type_a = aggregate_family_metrics(rows, "pre_sns_baseline", "type_a_local_overlay")
    assert_close(type_a["synthetic_recall"], 0.6, "type A synthetic mean")
    assert_close(type_a["tampered_recall"], 0.3, "type A tampered mean")
    type_b = aggregate_family_metrics(rows, "pre_sns_baseline", "type_b_global_geometry_degradation")
    assert_close(type_b["synthetic_recall"], 0.9, "type B synthetic mean")


def test_run_writes_filled_outputs_and_preserves_main_metrics() -> None:
    root = temp_root("cvf_0077a_run_")
    write_fixture(root)
    config = safe_config(root)
    dry = run_snsaug_v2_fill_final_cross_model_ablation_metrics(config, dry_run=True)
    for path in dry["output_paths"].values():
        assert_true(not Path(path).exists(), f"dry-run did not write {path}")
    manifest = run_snsaug_v2_fill_final_cross_model_ablation_metrics(config)
    assert_equal(manifest["selected_failed_model"], "snsaug_0064g_hybrid_30x3", "selected failed model")
    matrix = json.loads(Path(manifest["output_paths"]["matrix"]).read_text(encoding="utf-8"))
    assert_equal(matrix["marker"], MARKER, "filled marker")
    assert_close(matrix["models"]["pre_sns_baseline"]["type_a_local_overlay"]["synthetic_recall"], 0.6, "baseline filled")
    assert_close(matrix["models"]["failed_single_model_finetune"]["type_a_local_overlay"]["tampered_recall"], 0.975, "failed filled")
    assert_close(matrix["models"]["mixed_feature_gate"]["type_a_local_overlay"]["synthetic_recall"], 1.0, "gate preserved")
    assert_true(matrix["models"]["pre_sns_baseline"]["type_a_local_overlay"]["source_profile_count"] == 2, "source count")
    table = Path(manifest["output_paths"]["table"]).read_text(encoding="utf-8")
    assert_true("failed_single_model_finetune\ttype_a_local_overlay" in table, "failed row in table")
    report = Path(manifest["output_paths"]["report"]).read_text(encoding="utf-8")
    assert_true("Missing profile groups remain NA" in report, "missing profile guardrail")
    assert_true(Path(manifest["output_paths"]["manifest"]).exists(), "manifest written")


def test_safety_flags_required() -> None:
    root = temp_root("cvf_0077a_flags_")
    config = safe_config(root)
    config["no_network"] = False
    errors = validate_snsaug_v2_fill_final_cross_model_ablation_metrics_config(config, require_exists=False)
    assert_true("no_network must be true" in errors, "network flag required")


def test_docs_marker_present() -> None:
    text = (REPO_ROOT / "docs" / "snsaug_v2_fill_final_cross_model_ablation_metrics.md").read_text(encoding="utf-8")
    assert_true(MARKER in text, "docs marker present")


def main() -> int:
    tests = [
        test_example_config_validates,
        test_table_parser_and_failed_selection,
        test_run_writes_filled_outputs_and_preserves_main_metrics,
        test_safety_flags_required,
        test_docs_marker_present,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
