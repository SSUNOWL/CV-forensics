#!/usr/bin/env python3
"""Standalone tests for pre-SNS baseline report generation."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_baseline_report import (  # noqa: E402
    APPROVAL_TEXT,
    MARKER,
    build_summary,
    generate_baseline_report,
    load_report_config,
    parse_json_with_logs,
    render_markdown,
    validate_report_config,
)
from scripts.agent.validate_pre_sns_baseline_report import validate_report_result  # noqa: E402

EXAMPLE_CONFIG = REPO_ROOT / "configs" / "reporting" / "pre_sns_baseline_report.example.json"
TEMP_ROOT = Path.home() / ".codex" / "memories"


def temporary_root():
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=str(TEMP_ROOT))


def assert_pass(raw: dict) -> None:
    errors = validate_report_config(raw)
    assert not errors, "\n".join(errors)


def assert_fail(raw: dict, expected: str) -> None:
    errors = validate_report_config(raw)
    assert errors, "expected validation failure"
    joined = "\n".join(errors)
    assert expected in joined, joined


def mutate(raw: dict, key: str, value) -> dict:
    changed = copy.deepcopy(raw)
    changed[key] = value
    return changed


def fake_training() -> dict:
    return {
        "samples_seen": 12,
        "steps_completed": 3,
        "epochs": 1,
        "initial_total_loss": 1.2,
        "final_total_loss": 0.8,
        "checkpoint_path": str(Path.home() / "cvf_checkpoints" / "run" / "model.pt"),
    }


def fake_inference() -> dict:
    return {
        "class": "tampered",
        "family": "GAN",
        "localization_head": "activated",
        "mask_area_pct": 12.5,
        "reason": "조작 의심 영역이 활성화됨.",
        "tampered_score": 0.9,
    }


def fake_evaluation() -> dict:
    return {
        "accuracy_3way": 0.7,
        "macro_f1_3way": 0.66,
        "family_accuracy": 0.5,
        "mask_iou_mean": 0.4,
        "localization_activation_recall": 0.8,
        "latency_ms_mean": 3.0,
        "fps_estimate": 333.3,
    }


def write_json(path: Path, payload: dict, prefix: str = "") -> None:
    path.write_text(prefix + json.dumps(payload), encoding="utf-8")


def make_config(tmp: str, write_report: bool = False, report_root: str | None = None) -> dict:
    root = Path(tmp)
    training = root / "training.json"
    inference = root / "inference.json"
    evaluation = root / "evaluation.json"
    scaled = root / "scaled.json"
    write_json(training, fake_training())
    write_json(inference, fake_inference())
    write_json(evaluation, fake_evaluation())
    write_json(scaled, {**fake_training(), "samples_seen": 8192, "epochs": 2})
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_baseline_report",
        "execution_mode": "approved_local_pre_sns_baseline_report",
        "required_approval_text": APPROVAL_TEXT,
        "user_approval_text": APPROVAL_TEXT,
        "training_result_path": str(training),
        "inference_report_path": str(inference),
        "evaluation_result_path": str(evaluation),
        "scaled_training_result_path": str(scaled),
        "approved_report_root": report_root or str(root / "report"),
        "write_report": write_report,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_inference": True,
        "no_evaluation": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
        "result_scope": "pre-SNS baseline report test fixture",
    }


def test_example_config_validates() -> None:
    assert_pass(load_report_config(EXAMPLE_CONFIG))


def test_result_parser_handles_leading_logs() -> None:
    with temporary_root() as tmp:
        path = Path(tmp) / "logged.json"
        write_json(path, {"ok": True}, prefix="leading log\n")
        assert parse_json_with_logs(path) == {"ok": True}


def test_markdown_contains_required_sections() -> None:
    summary = build_summary(fake_training(), fake_inference(), fake_evaluation(), fake_training(), "test")
    markdown = render_markdown(summary)
    for section in summary["required_sections"]:
        assert f"## {section}" in markdown
    assert "SNS augmentation has not been applied" in markdown


def test_summary_json_contains_key_metrics() -> None:
    summary = build_summary(fake_training(), fake_inference(), fake_evaluation(), fake_training(), "test")
    assert summary["marker"] == MARKER
    assert summary["evaluation_metrics"]["accuracy_3way"] == 0.7
    assert summary["single_image_report_summary"]["class"] == "tampered"


def test_missing_evaluation_metrics_rejected() -> None:
    bad = fake_evaluation()
    del bad["macro_f1_3way"]
    try:
        build_summary(fake_training(), fake_inference(), bad, fake_training(), "test")
    except Exception as exc:
        assert "evaluation result missing" in str(exc)
    else:
        raise AssertionError("expected missing metric failure")


def test_missing_inference_report_rejected() -> None:
    bad = fake_inference()
    del bad["reason"]
    try:
        build_summary(fake_training(), bad, fake_evaluation(), fake_training(), "test")
    except Exception as exc:
        assert "inference report missing" in str(exc)
    else:
        raise AssertionError("expected missing inference failure")


def test_report_root_inside_repo_outputs_checkpoints_rejected() -> None:
    with temporary_root() as tmp:
        assert_fail(make_config(tmp, report_root=str(REPO_ROOT / "outputs" / "report")), "approved_report_root")
        assert_fail(make_config(tmp, report_root=str(REPO_ROOT / "checkpoints" / "report")), "approved_report_root")


def test_no_sns_augmentation_false_rejected() -> None:
    with temporary_root() as tmp:
        assert_fail(mutate(make_config(tmp), "no_sns_augmentation", False), "no_sns_augmentation")


def test_valid_report_artifacts_accepted() -> None:
    with temporary_root() as tmp:
        config = make_config(tmp, write_report=True)
        result = generate_baseline_report(config)
        errors = validate_report_result(config, result)
        assert not errors, "\n".join(errors)
        assert os.path.isfile(result["report_markdown_path"])
        assert os.path.isfile(result["report_summary_path"])


def main() -> int:
    tests = [
        test_example_config_validates,
        test_result_parser_handles_leading_logs,
        test_markdown_contains_required_sections,
        test_summary_json_contains_key_metrics,
        test_missing_evaluation_metrics_rejected,
        test_missing_inference_report_rejected,
        test_report_root_inside_repo_outputs_checkpoints_rejected,
        test_no_sns_augmentation_false_rejected,
        test_valid_report_artifacts_accepted,
    ]
    for test in tests:
        test()
    print("PRE_SNS_BASELINE_REPORT_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
