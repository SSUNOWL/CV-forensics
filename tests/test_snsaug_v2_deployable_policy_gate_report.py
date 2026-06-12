#!/usr/bin/env python3
"""Plain Python tests for SNSAug v2 deployable policy gate final report."""

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

from cv_forensics.snsaug_v2_deployable_policy_gate_report import (  # noqa: E402
    MARKER,
    SNSAugV2DeployablePolicyGateReportError,
    build_final_records,
    load_snsaug_v2_deployable_policy_gate_report_config,
    run_snsaug_v2_deployable_policy_gate_report,
    select_best_deployable_candidate,
    validate_snsaug_v2_deployable_policy_gate_report_config,
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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def candidate(selector: str, score: float, diagnostic: bool = False) -> dict[str, object]:
    return {
        "selector": selector,
        "diagnostic_only": diagnostic,
        "score": score,
        "good_profile_count": 2,
        "good_profiles": ["zoom_crop", "resize_jpeg"],
        "mean_tampered_recall_gain": 0.4,
        "mean_valid_iou_gain": 0.25,
        "mean_tampered_gap_closure": 0.6,
        "mean_valid_iou_gap_closure": 0.7,
        "mean_synthetic_recall_change": 0.1,
        "mean_real_fpr_change": -0.02,
    }


def fixture_records(selector: str = "mixed_feature_gate", chosen_policy: str = "geometry_feature_clean_policy") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile in ("zoom_crop", "resize_jpeg"):
        for label in ("tampered", "synthetic", "real"):
            p = 0.8 if label == "tampered" else 0.1
            rows.append(
                {
                    "marker": "SNSAUG_V2_POLICY_GATE_COMPARISON_OK",
                    "selector": selector,
                    "chosen_policy": chosen_policy,
                    "selector_reason": "fixture_reason",
                    "diagnostic_only_selector": False,
                    "base_id": f"{profile}_{label}",
                    "profile": profile,
                    "profile_family": "geometry" if profile == "zoom_crop" else "postprocess",
                    "content_label": label,
                    "pred_tampered": label == "tampered",
                    "p_tampered": p,
                    "valid_iou": 0.7 if label == "tampered" else None,
                    "geometry_score": 1.2,
                    "residual_score": 0.9,
                    "selected_by_profile_family": "geometry",
                }
            )
    rows.append(
        {
            "marker": "SNSAUG_V2_POLICY_GATE_COMPARISON_OK",
            "selector": "fixed_original",
            "chosen_policy": "original",
            "selector_reason": "fixed",
            "diagnostic_only_selector": False,
            "base_id": "fixed",
            "profile": "zoom_crop",
            "profile_family": "geometry",
            "content_label": "tampered",
            "pred_tampered": False,
            "p_tampered": 0.2,
            "valid_iou": 0.1,
            "geometry_score": 1.2,
            "residual_score": 0.9,
            "selected_by_profile_family": "geometry",
        }
    )
    rows.append(
        {
            "marker": "SNSAUG_V2_POLICY_GATE_COMPARISON_OK",
            "selector": "oracle_best_policy_diagnostic",
            "chosen_policy": "oracle_clean_policy",
            "selector_reason": "oracle",
            "diagnostic_only_selector": True,
            "base_id": "oracle",
            "profile": "zoom_crop",
            "profile_family": "geometry",
            "content_label": "tampered",
            "pred_tampered": True,
            "p_tampered": 0.9,
            "valid_iou": 0.9,
            "geometry_score": 1.2,
            "residual_score": 0.9,
            "selected_by_profile_family": "geometry",
        }
    )
    return rows


def fixture_metrics() -> dict[str, object]:
    per_profile = {
        "zoom_crop": {
            "row_count": 3,
            "tampered_recall": 1.0,
            "synthetic_recall": 1.0,
            "real_fpr": 0.0,
            "tampered_valid_mean_iou": 0.7,
            "mean_p_tampered": 0.33,
        },
        "resize_jpeg": {
            "row_count": 3,
            "tampered_recall": 1.0,
            "synthetic_recall": 1.0,
            "real_fpr": 0.0,
            "tampered_valid_mean_iou": 0.7,
            "mean_p_tampered": 0.33,
        },
    }
    fixed = {
        "zoom_crop": {"tampered_recall": 0.0, "synthetic_recall": 1.0, "real_fpr": 0.0, "tampered_valid_mean_iou": 0.1, "mean_p_tampered": 0.2},
        "resize_jpeg": {"tampered_recall": 0.0, "synthetic_recall": 1.0, "real_fpr": 0.0, "tampered_valid_mean_iou": 0.1, "mean_p_tampered": 0.2},
    }
    oracle = {
        "zoom_crop": {"tampered_recall": 1.0, "synthetic_recall": 1.0, "real_fpr": 0.0, "tampered_valid_mean_iou": 0.9, "mean_p_tampered": 0.4},
        "resize_jpeg": {"tampered_recall": 1.0, "synthetic_recall": 1.0, "real_fpr": 0.0, "tampered_valid_mean_iou": 0.9, "mean_p_tampered": 0.4},
    }
    return {
        "marker": "SNSAUG_V2_POLICY_GATE_COMPARISON_OK",
        "metrics": {
            "fixed_original": fixed,
            "mixed_feature_gate": per_profile,
            "geometry_feature_gate": per_profile,
            "oracle_best_policy_diagnostic": oracle,
        },
        "decision": {
            "decision": "deployable_policy_gate_promising",
            "next_recommendation": "freeze",
            "deployable_candidates": [
                candidate("geometry_feature_gate", 0.4),
                candidate("mixed_feature_gate", 0.9),
            ],
            "diagnostic_candidates": [candidate("oracle_best_policy_diagnostic", 1.0, diagnostic=True)],
        },
    }


def write_policy_gate_root(root: Path, rows: list[dict[str, object]] | None = None, metrics: dict[str, object] | None = None) -> Path:
    gate = root / "policy_gate"
    gate.mkdir(parents=True, exist_ok=True)
    _write_jsonl(gate / "policy_gate_records.jsonl", rows or fixture_records())
    _write_json(gate / "policy_gate_metrics.json", metrics or fixture_metrics())
    _write_json(gate / "policy_gate_oracle_gap_summary.json", {"marker": "SNSAUG_V2_POLICY_GATE_COMPARISON_OK"})
    (gate / "policy_gate_report.md").write_text("SNSAUG_V2_POLICY_GATE_COMPARISON_OK\n", encoding="utf-8")
    (gate / "policy_gate_comparison_report.md").write_text("SNSAUG_V2_POLICY_GATE_COMPARISON_OK\n", encoding="utf-8")
    _write_json(gate / "artifact_manifest.json", {"marker": "SNSAUG_V2_POLICY_GATE_COMPARISON_OK", "record_count": len(rows or fixture_records())})
    return gate


def safe_config(root: Path, gate_root: Path) -> dict[str, object]:
    (root / "reports").mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_deployable_policy_gate_report",
        "execution_mode": "approved_local_snsaug_v2_deployable_policy_gate_report",
        "user_approval_text": "I_APPROVE_SNSAUG_V2_DEPLOYABLE_POLICY_GATE_REPORT",
        "policy_gate_output_root": str(gate_root),
        "approved_input_roots": [str(gate_root)],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "final_gate"),
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def test_example_config_validates() -> None:
    config = load_snsaug_v2_deployable_policy_gate_report_config(REPO_ROOT / "configs" / "evaluation" / "snsaug_v2_deployable_policy_gate_report.example.json")
    assert_equal(validate_snsaug_v2_deployable_policy_gate_report_config(config, require_exists=False), [], "example config validates")


def test_best_deployable_candidate_selection() -> None:
    selected = select_best_deployable_candidate(fixture_metrics())
    assert_equal(selected["selector"], "mixed_feature_gate", "highest score deployable candidate selected")


def test_diagnostic_candidate_rejected() -> None:
    metrics = fixture_metrics()
    metrics["decision"]["deployable_candidates"] = [candidate("oracle_best_policy_diagnostic", 2.0, diagnostic=True)]
    try:
        select_best_deployable_candidate(metrics)
    except SNSAugV2DeployablePolicyGateReportError:
        return
    raise AssertionError("diagnostic deployable candidate rejected")


def test_oracle_chosen_policy_rejected_for_deployable_export() -> None:
    rows = fixture_records(chosen_policy="oracle_clean_policy")
    try:
        build_final_records(rows, "mixed_feature_gate")
    except SNSAugV2DeployablePolicyGateReportError:
        return
    raise AssertionError("oracle chosen policy rejected")


def test_run_writes_final_artifacts_and_records() -> None:
    root = temp_root("cvf_0072_run_")
    gate = write_policy_gate_root(root)
    config = safe_config(root, gate)
    dry = run_snsaug_v2_deployable_policy_gate_report(config, dry_run=True)
    assert_true(dry["dry_run"] is True, "dry-run summary")
    assert_true(not Path(dry["output_paths"]["final_policy_gate_records"]).exists(), "dry-run writes no final records")
    summary = run_snsaug_v2_deployable_policy_gate_report(config, dry_run=False)
    assert_equal(summary["marker"], MARKER, "summary marker")
    for key in ("deployable_policy_gate_spec", "final_policy_gate_records", "final_policy_gate_metrics", "final_policy_gate_report", "final_policy_gate_notion_summary", "final_policy_gate_tables", "artifact_manifest"):
        path = Path(summary["output_paths"][key])
        assert_true(path.exists(), f"{key} exists")
        assert_true(not str(path).startswith(str(REPO_ROOT)), f"{key} outside repo")
    rows = [json.loads(line) for line in Path(summary["output_paths"]["final_policy_gate_records"]).read_text(encoding="utf-8").splitlines()]
    assert_true(rows, "final records non-empty")
    assert_true(all(row["selector"] == "mixed_feature_gate" for row in rows), "final records selected selector only")
    assert_true(all(row.get("chosen_policy") for row in rows), "final records chosen policy non-null")
    assert_true(all(row.get("diagnostic_only_selector") is False for row in rows), "final records not diagnostic")
    artifact = json.loads(Path(summary["output_paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert_equal(artifact["record_count"], len(rows), "artifact record count")


def test_docs_marker_present() -> None:
    text = (REPO_ROOT / "docs" / "snsaug_v2_deployable_policy_gate_report.md").read_text(encoding="utf-8")
    assert_true(MARKER in text, "docs marker present")


def main() -> int:
    tests = [
        test_example_config_validates,
        test_best_deployable_candidate_selection,
        test_diagnostic_candidate_rejected,
        test_oracle_chosen_policy_rejected_for_deployable_export,
        test_run_writes_final_artifacts_and_records,
        test_docs_marker_present,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
