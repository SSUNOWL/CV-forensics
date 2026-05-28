#!/usr/bin/env python3
"""Plain Python tests for pre-SNS v3 dual-scale report and mining."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_dual_scale_report import (  # noqa: E402
    class_mask_consistency,
    explanation_reason,
    mask_stats,
    postprocess_mask,
    resize_binary_mask_nearest,
    select_final_mask,
    validate_dual_report_config,
)
from cv_forensics.pre_sns_v3_hard_mining import (  # noqa: E402
    _records_from_checkpoint_runs,
    mining_summary_schema_ok,
    mine_cases_from_records,
    validate_hard_mining_config,
    write_mining_outputs,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def temp_parent() -> str | None:
    preferred = Path("/home/rlatjswo/.codex/memories")
    if preferred.is_dir() and os.access(preferred, os.W_OK):
        return str(preferred)
    return None


def external_root(tmp: Path, name: str) -> str:
    return str(tmp / name)


def dual_config(tmp: Path) -> dict:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_v3_dual_report",
        "execution_mode": "approved_local_pre_sns_v3_dual_report",
        "image_path": str(tmp / "inputs" / "image.png"),
        "long224_checkpoint_path": str(tmp / "models" / "long224.pt"),
        "long256_checkpoint_path": str(tmp / "models" / "long256.pt"),
        "approved_input_roots": [str(tmp / "inputs")],
        "approved_checkpoint_roots": [str(tmp / "models")],
        "approved_output_root": str(tmp / "reports" / "dual"),
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
    }


def mining_config(tmp: Path) -> dict:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_v3_hard_mining",
        "execution_mode": "approved_local_pre_sns_v3_hard_mining",
        "manifest_path": str(tmp / "inputs" / "manifest.json"),
        "approved_input_roots": [str(tmp / "inputs")],
        "approved_checkpoint_roots": [str(tmp / "models")],
        "approved_output_root": str(tmp / "reports" / "mining"),
        "max_samples": 10,
        "tampered_score_threshold": 0.5,
        "low_iou_threshold": 0.3,
        "use_manifest_predictions": True,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
    }


def test_config_validation() -> None:
    with tempfile.TemporaryDirectory(dir=temp_parent()) as raw:
        tmp = Path(raw)
        cfg = dual_config(tmp)
        assert_true(validate_dual_report_config(cfg) == [], "dual config should pass")
        bad = copy.deepcopy(cfg)
        bad["no_network"] = False
        assert_true(validate_dual_report_config(bad), "missing no_network guardrail should fail")
        bad = copy.deepcopy(cfg)
        bad["approved_output_root"] = str(REPO_ROOT / "reports" / "bad")
        assert_true(validate_dual_report_config(bad), "repo output root should fail")
        bad = copy.deepcopy(cfg)
        bad["no_sns_augmentation"] = False
        assert_true(validate_dual_report_config(bad), "SNS augmentation should fail")

        mcfg = mining_config(tmp)
        assert_true(validate_hard_mining_config(mcfg) == [], "mining config should pass")
        badm = copy.deepcopy(mcfg)
        badm["max_samples"] = 0
        assert_true(validate_hard_mining_config(badm), "max_samples=0 must be rejected")
        badm = copy.deepcopy(mcfg)
        badm["manifest_path"] = str(tmp / "unapproved" / "manifest.json")
        assert_true(validate_hard_mining_config(badm), "manifest outside approved roots should fail")


def test_mask_component_cleanup() -> None:
    shape = (5, 5)
    active = [0] * 25
    active[0] = 1
    for y in range(2, 5):
        for x in range(2, 5):
            active[y * 5 + x] = 1
    cleaned = postprocess_mask(active, shape, min_component_area_px=4, keep_top_k_components=1)
    stats = mask_stats(cleaned, shape)
    assert_true(stats["component_count"] == 1, "tiny component should be removed")
    assert_true(stats["mask_area_px"] == 9, "large component should remain")


def test_mask_stats_edge_cases() -> None:
    empty = mask_stats([], (0, 0))
    assert_true(empty["component_count"] == 0, "empty mask should have no components")
    assert_true(empty["mask_stats_malformed"] is True, "empty invalid shape should be marked malformed")

    single = mask_stats([0, 0, 0, 0, 1, 0, 0, 0, 0], (3, 3))
    assert_true(single["component_count"] == 1, "single active pixel should be one component")
    assert_true(single["largest_component_area_px"] == 1, "single active pixel largest component mismatch")

    all_active = mask_stats([1] * 9, (3, 3))
    assert_true(all_active["component_count"] == 1, "all-active mask should be one component")
    assert_true(all_active["mask_area_pct"] == 100.0, "all-active area should be 100%")

    isolated = mask_stats([1, 0, 1, 0, 0, 0, 1, 0, 1], (3, 3))
    assert_true(isolated["component_count"] == 4, "diagonal isolated pixels should be separate 4-neighbor components")

    one_by_n = mask_stats([1, 0, 1], (3, 1))
    assert_true(one_by_n["component_count"] == 2, "1xN separated pixels should be separate components")
    n_by_one = mask_stats([1, 0, 1], (1, 3))
    assert_true(n_by_one["component_count"] == 2, "Nx1 separated pixels should be separate components")

    malformed = mask_stats([1, 0], (3, 3))
    assert_true(malformed["mask_stats_malformed"] is True, "shape mismatch should be marked malformed")
    assert_true(malformed["mask_width"] == 3 and malformed["mask_height"] == 3, "malformed stats should preserve requested shape")
    assert_true(malformed["mask_area_px"] == 1, "malformed short mask should be padded safely")


def test_consistency_logic() -> None:
    empty_selection = {
        "stats": {"mask_area_pct": 0.0},
        "uncertain": False,
    }
    gate = class_mask_consistency("tampered", 0.91, empty_selection)
    assert_true(gate["localized_evidence_status"] == "not_found", "empty tampered mask must be not_found")
    assert_true(gate["final_decision"] == "tampered_suspect_no_localized_evidence", "empty tampered decision should be suspect")
    active_selection = {
        "stats": {"mask_area_pct": 4.0},
        "uncertain": False,
    }
    gate = class_mask_consistency("real", 0.30, active_selection)
    assert_true(gate["localized_evidence_status"] == "suppressed_non_tampered_mask", "non-tampered active mask should suppress")


def test_disputed_dual_tampered_masks_are_not_high_confidence() -> None:
    m224 = [0] * 16
    m256 = [0] * 16
    for idx in (0, 1, 4, 5):
        m224[idx] = 1
    for idx in (10, 11, 14, 15):
        m256[idx] = 1
    cfg = {"agreement_iou_threshold": 0.15, "cross_scale_disagreement_iou_threshold": 0.05}
    selected = select_final_mask(model(m224, "tampered", 0.9), model(m256, "tampered", 0.92), cfg)
    assert_true(selected["agreement_iou"] == 0.0, "fixture should have zero mask agreement")
    gate = class_mask_consistency("tampered", 0.92, selected, cfg)
    assert_true(gate["final_decision"] == "tampered_suspect_disputed_localization", "low agreement must be disputed")
    assert_true(gate["final_decision_confidence"] != "high", "disputed localization must not be high confidence")
    assert_true(gate["localized_evidence_status"] in {"disputed", "weak_disputed"}, "localized evidence should be disputed")
    assert_true(gate["class_mask_consistency"] == "cross_scale_mask_disagreement", "consistency should mention cross-scale disagreement")
    reason = explanation_reason("tampered", "GAN", 0.92, selected, gate)
    assert_true("localize different regions" in reason, "explanation should mention disagreement")


def test_cross_scale_disagreement_with_different_mask_shapes() -> None:
    m224 = [0] * 16
    for idx in (0, 1, 4, 5):
        m224[idx] = 1
    m256 = [0] * 64
    for idx in (54, 55, 62, 63):
        m256[idx] = 1
    resized = resize_binary_mask_nearest(m224, (4, 4), (8, 8))
    assert_true(len(resized) == 64, "resized mask should match long256 grid")
    selected = select_final_mask(
        {"class": "tampered", "tampered_score": 0.91, "processed_mask": m224, "mask_shape": (4, 4)},
        {"class": "tampered", "tampered_score": 0.94, "processed_mask": m256, "mask_shape": (8, 8)},
        {"agreement_iou_threshold": 0.15, "cross_scale_disagreement_iou_threshold": 0.05},
    )
    gate = class_mask_consistency("tampered", 0.94, selected, {"cross_scale_disagreement_iou_threshold": 0.05})
    assert_true(selected["agreement_iou"] == 0.0, "different-shape masks should still compare on common grid")
    assert_true(gate["class_mask_consistency"] == "cross_scale_mask_disagreement", "shape mismatch should not hide disagreement")


def test_tiny_mask_low_agreement_downgrades_localization() -> None:
    m224 = [0] * 1024
    m256 = [0] * 1024
    m224[0] = 1
    m256[-1] = 1
    cfg = {
        "agreement_iou_threshold": 0.15,
        "cross_scale_disagreement_iou_threshold": 0.05,
        "tiny_mask_area_pct_threshold": 0.15,
    }
    selected = select_final_mask(
        {"class": "tampered", "tampered_score": 0.91, "processed_mask": m224, "mask_shape": (32, 32)},
        {"class": "tampered", "tampered_score": 0.93, "processed_mask": m256, "mask_shape": (32, 32)},
        cfg,
    )
    gate = class_mask_consistency("tampered", 0.93, selected, cfg)
    assert_true(gate["localized_evidence_status"] == "weak_disputed", "tiny low-agreement mask should be weak_disputed")
    assert_true(gate["final_decision_confidence"] != "high", "tiny low-agreement mask must not be high confidence")
    assert_true(gate["localization_confidence"] == "low", "tiny low-agreement localization should be low confidence")


def test_single_scale_mask_downgrades_without_strong_component() -> None:
    m224 = [0] * 100
    m256 = [0] * 100
    m256[10] = 1
    selected = select_final_mask(
        {"class": "tampered", "tampered_score": 0.88, "processed_mask": m224, "mask_shape": (10, 10)},
        {"class": "tampered", "tampered_score": 0.91, "processed_mask": m256, "mask_shape": (10, 10)},
        {"strong_component_area_pct_threshold": 5.0},
    )
    gate = class_mask_consistency("tampered", 0.91, selected, {"strong_component_area_pct_threshold": 5.0})
    assert_true(gate["localized_evidence_status"] == "weak_single_scale", "single weak mask should downgrade localization")
    assert_true(gate["final_decision_confidence"] != "high", "single weak mask must not be high confidence")


def model(mask: list[int], cls: str = "tampered", score: float = 0.7) -> dict:
    return {
        "class": cls,
        "tampered_score": score,
        "processed_mask": mask,
        "mask_shape": (4, 4),
    }


def test_dual_scale_mask_selection_rules() -> None:
    m224 = [0] * 16
    m256 = [0] * 16
    for idx in (5, 6, 9, 10):
        m224[idx] = 1
        m256[idx] = 1
    selected = select_final_mask(model(m224), model(m256), {"agreement_iou_threshold": 0.2})
    assert_true(selected["source"] == "dual_scale_union_agreement", "overlapping masks should use dual-scale union")
    assert_true(sum(selected["mask"]) == 4, "union mask area mismatch")

    selected = select_final_mask(model(m224), model([0] * 16, "tampered", 0.8), {"uncertain_tampered_score": 0.35})
    assert_true(selected["source"] == "long224_fallback_primary_tampered_or_uncertain", "long224 fallback should activate")

    selected = select_final_mask(model(m224), model([0] * 16, "real", 0.2), {"uncertain_tampered_score": 0.35})
    assert_true(sum(selected["mask"]) == 0, "non-tampered low-score long224 mask should be suppressed")

    selected = select_final_mask(model([0] * 16), model(m256, "real", 0.4), {"very_high_tampered_score": 0.85})
    assert_true(selected["source"] == "none", "non-tampered long256 mask should not be selected before class gate")


def test_hard_mining_summary_schema_and_no_repo_writes() -> None:
    with tempfile.TemporaryDirectory(dir=temp_parent()) as raw:
        tmp = Path(raw)
        cfg = mining_config(tmp)
        records = [
            {"sample_id": "r1", "image_path": str(tmp / "inputs" / "r1.png"), "class_label": "real", "final_decision": "tampered_suspect_no_localized_evidence", "tampered_score": 0.8, "final_mask_area_pct": 0.0},
            {"sample_id": "s1", "image_path": str(tmp / "inputs" / "s1.png"), "class_label": "full_synthetic", "final_decision": "tampered_with_localized_evidence", "tampered_score": 0.6, "final_mask_area_pct": 3.0},
            {"sample_id": "t1", "image_path": str(tmp / "inputs" / "t1.png"), "class_label": "tampered", "final_decision": "tampered_with_localized_evidence", "tampered_score": 0.9, "final_mask_area_pct": 2.0, "localization_iou": 0.1},
        ]
        mined = mine_cases_from_records(records, cfg)
        assert_true(len(mined["hard_negative_real"]) == 1, "real hard negative missing")
        assert_true(len(mined["hard_negative_non_tampered"]) == 2, "non-tampered hard negatives missing")
        assert_true(len(mined["hard_positive_tampered_low_iou"]) == 1, "low-IoU tampered case missing")
        summary = write_mining_outputs(cfg["approved_output_root"], mined, cfg)
        assert_true(mining_summary_schema_ok(summary), "mining summary schema invalid")
        for path in summary["output_paths"].values():
            assert_true(Path(path).exists(), f"missing mining output: {path}")
            assert_true(not str(Path(path).resolve()).startswith(str(REPO_ROOT.resolve())), "mining output wrote inside repo")


def test_hard_mining_records_failed_cases_and_continues() -> None:
    with tempfile.TemporaryDirectory(dir=temp_parent()) as raw:
        tmp = Path(raw)
        cfg = mining_config(tmp)
        cfg.update(
            {
                "use_manifest_predictions": False,
                "long224_checkpoint_path": str(tmp / "models" / "long224.pt"),
                "long256_checkpoint_path": str(tmp / "models" / "long256.pt"),
                "fail_fast": False,
            }
        )
        manifest = {
            "samples": [
                {"sample_id": "ok", "class_label": "tampered", "image_path": str(tmp / "inputs" / "ok.png")},
                {"sample_id": "bad", "class_label": "tampered", "image_path": str(tmp / "inputs" / "bad.png")},
            ]
        }

        def report_runner(report_config: dict) -> dict:
            if report_config["image_path"].endswith("bad.png"):
                raise RuntimeError("synthetic report failure")
            return {
                "long256": {"tampered_score": 0.9},
                "final_decision": "tampered_with_localized_evidence",
                "final_mask_area_pct": 2.0,
                "class_mask_consistency": "consistent",
                "localized_evidence_status": "found",
                "localization_confidence": "high",
                "localization_disagreement_reason": "none",
            }

        records, failed = _records_from_checkpoint_runs(manifest, cfg, report_runner=report_runner)
        assert_true(len(records) == 1, "one successful record should be retained")
        assert_true(len(failed) == 1, "one failed sample should be recorded")
        assert_true(failed[0]["sample_id"] == "bad", "failed sample id mismatch")
        mined = mine_cases_from_records(records, cfg)
        mined["failed_cases"] = failed
        summary = write_mining_outputs(cfg["approved_output_root"], mined, cfg)
        failed_path = Path(summary["output_paths"]["failed_cases"])
        assert_true(summary["failed_case_count"] == 1, "summary failed_case_count mismatch")
        assert_true(failed_path.exists(), "failed_cases.json should be written")
        failed_payload = json.loads(failed_path.read_text(encoding="utf-8"))
        assert_true(failed_payload[0]["exception"].startswith("RuntimeError"), "failed exception should be captured")


def main() -> int:
    for test in (
        test_config_validation,
        test_mask_component_cleanup,
        test_mask_stats_edge_cases,
        test_consistency_logic,
        test_disputed_dual_tampered_masks_are_not_high_confidence,
        test_cross_scale_disagreement_with_different_mask_shapes,
        test_tiny_mask_low_agreement_downgrades_localization,
        test_single_scale_mask_downgrades_without_strong_component,
        test_dual_scale_mask_selection_rules,
        test_hard_mining_summary_schema_and_no_repo_writes,
        test_hard_mining_records_failed_cases_and_continues,
    ):
        test()
    print("PRE_SNS_V3_DUAL_SCALE_REPORT_AND_MINING_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
