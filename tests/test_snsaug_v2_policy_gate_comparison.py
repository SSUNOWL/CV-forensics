#!/usr/bin/env python3
"""Plain Python tests for SNSAug v2 policy gate comparison."""

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

from cv_forensics.snsaug_v2_policy_gate_comparison import (  # noqa: E402
    MARKER,
    build_policy_gate_records,
    decide_policy_gate,
    load_snsaug_v2_policy_gate_comparison_config,
    run_snsaug_v2_policy_gate_comparison,
    selector_profile_metrics,
    validate_snsaug_v2_policy_gate_comparison_config,
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


def fixture_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile, geo, residual in (("zoom_crop", 1.5, 0.2), ("resize_jpeg", 0.2, 2.0)):
        for idx, label in enumerate(("tampered", "synthetic", "real")):
            rows.append(
                {
                    "marker": "SNSAUG_V2_RESIDUAL_DEGRADATION_ANALYSIS_OK",
                    "base_id": f"{profile}_{label}_{idx}",
                    "profile": profile,
                    "profile_family": "geometry" if profile == "zoom_crop" else "postprocess",
                    "content_label": label,
                    "p_tampered_clean": 0.9 if label == "tampered" else 0.05,
                    "p_tampered_sns": 0.2 if label == "tampered" else 0.04,
                    "clean_valid_iou": 0.8 if label == "tampered" else None,
                    "sns_valid_iou": 0.1 if label == "tampered" else None,
                    "crop_scale_proxy": geo,
                    "area_ratio_delta_abs": geo,
                    "aspect_ratio_delta_abs": geo / 2,
                    "dct_total_energy_delta_abs": residual,
                    "dct_high_energy_delta_abs": residual,
                    "dct_low_energy_delta_abs": residual,
                    "dct_high_low_ratio_delta_abs": residual,
                    "histogram_l1": residual,
                    "highpass_energy_delta_abs": residual,
                    "srm_energy_delta_abs": residual,
                    "blockiness_delta_abs": residual,
                    "laplacian_variance_delta_abs": residual,
                    "sobel_edge_energy_delta_abs": residual,
                }
            )
    return rows


def test_example_config_validates() -> None:
    cfg = load_snsaug_v2_policy_gate_comparison_config(REPO_ROOT / "configs" / "evaluation" / "snsaug_v2_policy_gate_comparison.example.json")
    assert_equal(validate_snsaug_v2_policy_gate_comparison_config(cfg, require_exists=False), [], "example config validates")


def test_policy_gate_records_include_selectors() -> None:
    records = build_policy_gate_records(fixture_rows())
    selectors = {row["selector"] for row in records}
    assert_true("fixed_original" in selectors, "fixed selector present")
    assert_true("geometry_feature_gate" in selectors, "geometry gate present")
    assert_true("residual_dct_feature_gate" in selectors, "residual gate present")
    assert_true("oracle_best_policy_diagnostic" in selectors, "oracle selector present")
    assert_true(any(row["selected_source"] == "clean" for row in records), "at least one gate selects clean")
    assert_true(all(row.get("chosen_policy") for row in records), "chosen policy populated")
    assert_true(all(row.get("selector_reason") for row in records), "selector reason populated")
    assert_true(all("diagnostic_only_selector" in row for row in records), "diagnostic flag populated")
    fixed = [row for row in records if row["selector"] == "fixed_original"]
    assert_true(all(row["chosen_policy"] == "original" for row in fixed), "fixed original policy provenance")
    oracle = [row for row in records if row["selector"] == "oracle_best_policy_diagnostic"]
    assert_true(all(row["diagnostic_only_selector"] is True for row in oracle), "oracle selector diagnostic")


def test_decision_prefers_deployable_when_gate_improves_profiles() -> None:
    selectors = ["fixed_original", "mixed_feature_gate", "oracle_best_policy_diagnostic"]
    rows = build_policy_gate_records(fixture_rows(), selectors)
    metrics = selector_profile_metrics(rows)
    decision = decide_policy_gate(metrics, ["zoom_crop", "resize_jpeg"], selectors)
    assert_equal(decision["decision"], "deployable_policy_gate_promising", "mixed gate is deployable")


def test_run_writes_outputs() -> None:
    root = temp_root("cvf_0071_run_")
    records_path = root / "input" / "residual_degradation_records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with open(records_path, "w", encoding="utf-8") as handle:
        for row in fixture_rows():
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    cfg = {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_policy_gate_comparison",
        "execution_mode": "approved_local_snsaug_v2_policy_gate_comparison",
        "user_approval_text": "I_APPROVE_SNSAUG_V2_POLICY_GATE_COMPARISON",
        "residual_records_path": str(records_path),
        "approved_input_roots": [str(records_path.parent)],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "policy_gate"),
        "focus_profiles": ["zoom_crop", "resize_jpeg"],
        "selectors": ["fixed_original", "mixed_feature_gate", "oracle_best_policy_diagnostic"],
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    summary = run_snsaug_v2_policy_gate_comparison(cfg)
    assert_equal(summary["marker"], MARKER, "run marker")
    for path in summary["output_paths"].values():
        assert_true(Path(path).exists(), f"output exists: {path}")
    assert_true(Path(summary["output_paths"]["policy_gate_report"]).exists(), "policy gate report exists")
    assert_true(Path(summary["output_paths"]["policy_gate_comparison_report"]).exists(), "policy gate comparison report exists")
    artifact = json.loads(Path(summary["output_paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert_true("policy_gate_report" in artifact["output_paths"], "artifact lists report")
    assert_true("policy_gate_comparison_report" in artifact["output_paths"], "artifact lists comparison report")
    rows = [json.loads(line) for line in Path(summary["output_paths"]["policy_gate_records"]).read_text(encoding="utf-8").splitlines()]
    assert_true(rows and all(row.get("chosen_policy") for row in rows), "run records have chosen policy")
    assert_true(any(row["selector"] == "fixed_original" and row["chosen_policy"] == "original" for row in rows), "run fixed original provenance")


def test_dry_run_writes_no_records() -> None:
    root = temp_root("cvf_0071_dry_")
    records_path = root / "input" / "residual_degradation_records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_policy_gate_comparison",
        "execution_mode": "approved_local_snsaug_v2_policy_gate_comparison",
        "user_approval_text": "I_APPROVE_SNSAUG_V2_POLICY_GATE_COMPARISON",
        "residual_records_path": str(records_path),
        "approved_input_roots": [str(records_path.parent)],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "policy_gate"),
        "focus_profiles": ["zoom_crop", "resize_jpeg"],
        "selectors": ["fixed_original", "mixed_feature_gate", "oracle_best_policy_diagnostic"],
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    summary = run_snsaug_v2_policy_gate_comparison(cfg, dry_run=True)
    assert_true(summary["dry_run"] is True, "dry-run summary")
    assert_true(not Path(summary["output_paths"]["policy_gate_records"]).exists(), "dry-run writes no records")


def main() -> None:
    test_example_config_validates()
    test_policy_gate_records_include_selectors()
    test_decision_prefers_deployable_when_gate_improves_profiles()
    test_run_writes_outputs()
    test_dry_run_writes_no_records()
    print("SNSAUG_V2_POLICY_GATE_COMPARISON_TESTS_OK")


if __name__ == "__main__":
    main()
