#!/usr/bin/env python3
"""Standalone tests for pre-SNS meaningful training v2."""

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

from cv_forensics.pre_sns_meaningful_training_v2 import (  # noqa: E402
    APPROVAL_TEXT,
    CLASS_LABELS,
    compute_metrics,
    load_config,
    run_training,
    select_tau,
    validate_config,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def temp_parent() -> str | None:
    preferred = Path("/home/rlatjswo/.codex/memories")
    if preferred.is_dir() and os.access(preferred, os.W_OK):
        return str(preferred)
    return None


def example_config() -> dict:
    return load_config(REPO_ROOT / "configs/training/pre_sns_meaningful_train_v2.example.json")


def approved_config(tmp: Path) -> dict:
    train_manifest = tmp / "manifests" / "train.json"
    val_manifest = tmp / "manifests" / "val.json"
    samples = [
        {"sample_id": "r0", "class_label": "real", "family_label": "Real-or-N/A", "image_path": "unused-r.jpg"},
        {"sample_id": "s0", "class_label": "full_synthetic", "family_label": "LatDiff", "image_path": "unused-s.jpg"},
        {"sample_id": "t0", "class_label": "tampered", "family_label": "GAN", "image_path": "unused-t.jpg", "mask_path": "unused-t.png"},
        {"sample_id": "t1", "class_label": "tampered", "family_label": "Other", "image_path": "unused-t1.jpg", "mask_path": "unused-t1.png"},
    ]
    write_json(train_manifest, {"samples": samples})
    write_json(val_manifest, {"samples": samples})
    cfg = example_config()
    cfg.update(
        {
            "config_kind": "approved_pre_sns_meaningful_training_v2",
            "execution_mode": "approved_local_pre_sns_meaningful_training_v2",
            "user_approval_text": APPROVAL_TEXT,
            "train_manifest_path": str(train_manifest),
            "val_manifest_path": str(val_manifest),
            "approved_input_roots": [str(tmp / "manifests")],
            "approved_run_root": str(tmp / "cvf_runs" / "run-0033"),
            "approved_checkpoint_root": str(tmp / "cvf_ckpts" / "run-0033"),
            "max_samples_train": 4,
            "max_samples_val": 4,
            "max_image_size": 16,
            "batch_size": 2,
            "epochs": 2,
            "no_write_dry_run": True,
        }
    )
    return cfg


def test_example_config_validation() -> None:
    errors = validate_config(example_config())
    assert_true(errors == [], f"example config should pass: {errors}")


def test_guardrail_rejections() -> None:
    for key in ("no_download", "no_network", "no_sns_augmentation"):
        cfg = example_config()
        cfg[key] = False
        assert_true(validate_config(cfg), f"{key}=false must be rejected")

    cfg = example_config()
    cfg["approved_run_root"] = "outputs/pre_sns"
    assert_true(validate_config(cfg), "repository outputs root must be rejected")

    cfg = example_config()
    cfg["approved_checkpoint_root"] = "checkpoints/pre_sns"
    assert_true(validate_config(cfg), "repository checkpoints root must be rejected")

    cfg = example_config()
    cfg["train_manifest_path"] = "data/train.json"
    assert_true(validate_config(cfg), "protected data path must be rejected")

    cfg = example_config()
    cfg["val_manifest_path"] = ".env.local"
    assert_true(validate_config(cfg), ".env path must be rejected")


def test_approved_manifest_validation_and_dry_run_no_writes() -> None:
    with tempfile.TemporaryDirectory(dir=temp_parent()) as raw_tmp:
        tmp = Path(raw_tmp)
        cfg = approved_config(tmp)
        errors = validate_config(cfg, require_manifest_exists=True)
        assert_true(errors == [], f"approved temp config should pass: {errors}")
        result = run_training(cfg)
        assert_true(result["marker"] == "PRE_SNS_MEANINGFUL_TRAINING_V2_DRY_RUN_OK", "dry-run marker missing")
        assert_true(result["artifact_writes"] is False, "dry-run must not write artifacts")
        assert_true(result["checkpoint_writes"] is False, "dry-run must not write checkpoints")
        assert_true(not (tmp / "cvf_runs").exists(), "dry-run created run root")
        assert_true(not (tmp / "cvf_ckpts").exists(), "dry-run created checkpoint root")
        assert_true(result["samples_train"] == 4, "dry-run should read train manifest rows")
        assert_true(result["samples_val"] == 4, "dry-run should read val manifest rows")


def test_metrics_and_tau_selection() -> None:
    predictions = [
        {
            "gt_class": "real",
            "pred_class": "real",
            "tampered_score": 0.20,
            "gt_family": "Real-or-N/A",
            "pred_family": "Real-or-N/A",
            "localization_iou": None,
        },
        {
            "gt_class": "full_synthetic",
            "pred_class": "full_synthetic",
            "tampered_score": 0.35,
            "gt_family": "LatDiff",
            "pred_family": "LatDiff",
            "localization_iou": None,
        },
        {
            "gt_class": "tampered",
            "pred_class": "tampered",
            "tampered_score": 0.70,
            "gt_family": "GAN",
            "pred_family": "Other",
            "localization_iou": 0.50,
        },
        {
            "gt_class": "tampered",
            "pred_class": "real",
            "tampered_score": 0.42,
            "gt_family": "Other",
            "pred_family": "Other",
            "localization_iou": 0.25,
        },
    ]
    metrics = compute_metrics(predictions)
    assert_true(abs(metrics["class_accuracy"] - 0.75) < 1e-9, "class accuracy mismatch")
    assert_true(0.0 <= metrics["class_macro_f1"] <= 1.0, "macro-F1 out of range")
    assert_true(set(metrics["per_class"]) == set(CLASS_LABELS), "per-class metrics missing labels")
    assert_true(metrics["per_class"]["tampered"]["recall"] == 0.5, "tampered recall mismatch")
    assert_true(metrics["confusion_matrix"]["matrix"][2][0] == 1, "confusion matrix mismatch")
    assert_true(abs(metrics["family_accuracy"] - (2 / 3)) < 1e-9, "family accuracy should ignore Real-or-N/A")
    assert_true(metrics["tampered_score_summary_by_gt_class"]["real"]["count"] == 1, "score summary missing real count")
    assert_true(len(metrics["threshold_sweep"]) == 8, "tau sweep should have 8 rows")
    assert_true(metrics["selected_tau"] == select_tau(metrics["threshold_sweep"]), "selected tau not deterministic")
    assert_true(metrics["localization_activation_recall"] >= 0.5, "activation recall too low for fixture")
    assert_true(0.0 <= metrics["false_activation_rate"] <= 1.0, "false activation rate out of range")
    assert_true(abs(metrics["localization_mean_iou"] - 0.375) < 1e-9, "mean IoU mismatch")
    assert_true(abs(metrics["localization_median_iou"] - 0.375) < 1e-9, "median IoU mismatch")


def test_reject_bad_approved_paths() -> None:
    with tempfile.TemporaryDirectory(dir=temp_parent()) as raw_tmp:
        tmp = Path(raw_tmp)
        cfg = approved_config(tmp)
        bad = copy.deepcopy(cfg)
        bad["approved_run_root"] = str(REPO_ROOT / "outputs" / "bad")
        assert_true(validate_config(bad, require_manifest_exists=True), "repo outputs root should fail")
        bad = copy.deepcopy(cfg)
        bad["approved_checkpoint_root"] = str(REPO_ROOT / "checkpoints" / "bad")
        assert_true(validate_config(bad, require_manifest_exists=True), "repo checkpoints root should fail")
        bad = copy.deepcopy(cfg)
        bad["train_manifest_path"] = str(tmp / "data" / "train.json")
        assert_true(validate_config(bad, require_manifest_exists=True), "protected manifest path should fail")


def main() -> int:
    tests = [
        test_example_config_validation,
        test_guardrail_rejections,
        test_approved_manifest_validation_and_dry_run_no_writes,
        test_metrics_and_tau_selection,
        test_reject_bad_approved_paths,
    ]
    for test in tests:
        test()
    print("PRE_SNS_MEANINGFUL_TRAINING_V2_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
