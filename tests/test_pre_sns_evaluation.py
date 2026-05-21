#!/usr/bin/env python3
"""Standalone tests for pre-SNS evaluation helpers."""

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

from cv_forensics.pre_sns_evaluation import (  # noqa: E402
    APPROVAL_TEXT,
    MARKER,
    accuracy_score,
    binary_mask_iou,
    family_accuracy_score,
    load_evaluation_config,
    localization_activation_recall_score,
    macro_f1_score,
    validate_evaluation_config,
)
from scripts.agent.validate_pre_sns_evaluation import parse_result_json, validate_evaluation_result  # noqa: E402


EXAMPLE_CONFIG = REPO_ROOT / "configs" / "evaluation" / "pre_sns_evaluation.example.json"
TEMP_ROOT = Path.home() / ".codex" / "memories"


def temporary_root():
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=str(TEMP_ROOT))


def assert_pass(raw: dict) -> None:
    errors = validate_evaluation_config(raw)
    assert not errors, "\n".join(errors)


def assert_fail(raw: dict, expected: str) -> None:
    errors = validate_evaluation_config(raw)
    assert errors, "expected validation failure"
    joined = "\n".join(errors)
    assert expected in joined, joined


def mutate(raw: dict, key: str, value) -> dict:
    changed = copy.deepcopy(raw)
    changed[key] = value
    return changed


def make_approved(tmp: str, eval_root: str | None = None) -> dict:
    manifest_path = os.path.join(tmp, "manifest.json")
    model_path = os.path.join(tmp, "model.pt")
    Path(manifest_path).write_text('{"samples":[]}\n', encoding="utf-8")
    Path(model_path).write_bytes(b"fake")
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_evaluation",
        "execution_mode": "approved_local_pre_sns_evaluation",
        "required_approval_text": APPROVAL_TEXT,
        "user_approval_text": APPROVAL_TEXT,
        "manifest_path": manifest_path,
        "checkpoint_path": model_path,
        "approved_eval_root": eval_root or os.path.join(tmp, "eval"),
        "write_eval_artifact": False,
        "device": "cpu",
        "cuda_device_index": 0,
        "max_samples": 4,
        "batch_size": 1,
        "max_image_size": 8,
        "threshold_tau": 0.5,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
        "result_scope": "pre-SNS evaluation test fixture",
    }


def make_valid_result() -> dict:
    return {
        "marker": MARKER,
        "accuracy_3way": 0.75,
        "macro_f1_3way": 0.7,
        "family_accuracy": None,
        "mask_iou_mean": 0.5,
        "localization_activation_recall": 1.0,
        "latency_ms_mean": 2.0,
        "fps_estimate": 500.0,
        "samples_evaluated": 4,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }


def test_macro_f1_calculation() -> None:
    score = macro_f1_score(
        ["real", "full_synthetic", "tampered"],
        ["real", "tampered", "tampered"],
    )
    assert score is not None
    assert abs(score - ((1.0 + 0.0 + (2.0 / 3.0)) / 3.0)) < 1e-9


def test_accuracy_calculation() -> None:
    assert accuracy_score(["real", "tampered", "real"], ["real", "real", "real"]) == 2.0 / 3.0


def test_family_accuracy_missing_labels_ignored() -> None:
    assert family_accuracy_score(["GAN", None, "LatDiff"], ["GAN", "Other", "PixDiff"]) == 0.5


def test_mask_iou_calculation() -> None:
    pred = [[1, 0], [1, 0]]
    true = [[1, 1], [0, 0]]
    assert binary_mask_iou(pred, true) == 1.0 / 3.0


def test_localization_activation_recall() -> None:
    labels = ["real", "tampered", "tampered"]
    states = ["not_applicable", "activated", "skipped_below_threshold"]
    assert localization_activation_recall_score(labels, states) == 0.5


def test_json_result_parser_with_leading_logs() -> None:
    with temporary_root() as tmp:
        path = Path(tmp) / "result.log"
        payload = make_valid_result()
        path.write_text("leading log\n" + json.dumps(payload), encoding="utf-8")
        assert parse_result_json(path) == payload


def test_example_config_validation() -> None:
    assert_pass(load_evaluation_config(EXAMPLE_CONFIG))


def test_approved_local_alias_accepts_explicit_absolute_paths() -> None:
    with temporary_root() as tmp:
        raw = make_approved(tmp)
        raw["config_kind"] = "approved_local_pre_sns_evaluation"
        assert_pass(raw)


def test_unsafe_paths_rejected() -> None:
    raw = load_evaluation_config(EXAMPLE_CONFIG)
    assert_fail(mutate(raw, "manifest_path", "https://example.invalid/manifest.json"), "URL")


def test_eval_root_inside_repo_outputs_checkpoints_rejected() -> None:
    with temporary_root() as tmp:
        raw = make_approved(tmp, str(REPO_ROOT / "outputs" / "eval"))
        assert_fail(raw, "approved_eval_root")
        raw = make_approved(tmp, str(REPO_ROOT / "checkpoints" / "eval"))
        assert_fail(raw, "approved_eval_root")


def test_result_validator_rejects_missing_metrics() -> None:
    result = make_valid_result()
    del result["macro_f1_3way"]
    assert "macro_f1_3way" in "\n".join(validate_evaluation_result({}, result))


def test_result_validator_accepts_valid_result() -> None:
    errors = validate_evaluation_result({}, make_valid_result())
    assert not errors, "\n".join(errors)


def main() -> int:
    tests = [
        test_macro_f1_calculation,
        test_accuracy_calculation,
        test_family_accuracy_missing_labels_ignored,
        test_mask_iou_calculation,
        test_localization_activation_recall,
        test_json_result_parser_with_leading_logs,
        test_example_config_validation,
        test_approved_local_alias_accepts_explicit_absolute_paths,
        test_unsafe_paths_rejected,
        test_eval_root_inside_repo_outputs_checkpoints_rejected,
        test_result_validator_rejects_missing_metrics,
        test_result_validator_accepts_valid_result,
    ]
    for test in tests:
        test()
    print("PRE_SNS_EVALUATION_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
