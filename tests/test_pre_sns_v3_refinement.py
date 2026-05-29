#!/usr/bin/env python3
"""Plain Python tests for pre-SNS v3 hard-case refinement."""

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

from cv_forensics.pre_sns_v3_refinement import (  # noqa: E402
    CONFIG_OK_MARKER,
    _write_refinement_artifacts,
    build_oversampled_train_samples,
    load_config,
    map_hard_cases_to_train,
    run_refinement,
    validate_config,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def temp_parent() -> str | None:
    preferred = Path("/home/rlatjswo/.codex/memories")
    if preferred.is_dir() and os.access(preferred, os.W_OK):
        return str(preferred)
    return None


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def samples() -> list[dict]:
    return [
        {"sample_id": "real-1", "image_path": "/tmp/refine-inputs/real-1.jpg", "class_label": "real", "family_label": "Real-or-N/A", "source_dataset": "SID-Set"},
        {"sample_id": "real-2", "image_path": "/tmp/refine-inputs/real-2.jpg", "class_label": "real", "family_label": "Real-or-N/A", "source_dataset": "CF-Small"},
        {"sample_id": "syn-1", "image_path": "/tmp/refine-inputs/syn-1.jpg", "class_label": "full_synthetic", "family_label": "GAN", "source_dataset": "CF-Small"},
        {"sample_id": "tam-1", "image_path": "/tmp/refine-inputs/tam-1.jpg", "class_label": "tampered", "family_label": "Other", "source_dataset": "SID-Set"},
    ]


def make_fixture(tmp: Path) -> dict:
    input_root = tmp / "inputs"
    hard_root = input_root / "hard_mining_train"
    run_root = tmp / "runs" / "refine"
    ckpt_root = tmp / "ckpts" / "refine"
    train_manifest = input_root / "train.json"
    val_manifest = input_root / "val.json"
    base_checkpoint = input_root / "base_best_checkpoint.pt"
    write_json(train_manifest, {"samples": samples()})
    write_json(val_manifest, {"samples": samples()})
    base_checkpoint.write_text("fixture checkpoint\n", encoding="utf-8")
    hard_payloads = {
        "hard_negative_real": [{"sample_id": "real-1", "image_path": "/tmp/refine-inputs/real-1.jpg"}],
        "hard_negative_non_tampered": [{"sample_id": "syn-1", "image_path": "/tmp/refine-inputs/syn-1.jpg"}],
        "hard_positive_tampered_low_iou": [{"sample_id": "tam-1", "image_path": "/tmp/refine-inputs/tam-1.jpg"}],
        "class_mask_inconsistent_cases": [
            {"sample_id": "real-2", "image_path": "/tmp/refine-inputs/real-2.jpg"},
            {"sample_id": "not-train", "image_path": "/tmp/refine-inputs/not-train.jpg"},
        ],
    }
    paths = {}
    for key, payload in hard_payloads.items():
        path = hard_root / f"{key}.json"
        write_json(path, payload)
        paths[key] = path
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_v3_refinement",
        "execution_mode": "approved_local_pre_sns_v3_refinement",
        "required_approval_text": "I_APPROVE_PRE_SNS_V3_REFINEMENT",
        "user_approval_text": "I_APPROVE_PRE_SNS_V3_REFINEMENT",
        "hard_cases_split": "train",
        "train_manifest_path": str(train_manifest),
        "val_manifest_path": str(val_manifest),
        "base_checkpoint_path": str(base_checkpoint),
        "hard_negative_real_path": str(paths["hard_negative_real"]),
        "hard_negative_non_tampered_path": str(paths["hard_negative_non_tampered"]),
        "hard_positive_tampered_low_iou_path": str(paths["hard_positive_tampered_low_iou"]),
        "class_mask_inconsistent_cases_path": str(paths["class_mask_inconsistent_cases"]),
        "approved_input_roots": [str(input_root)],
        "approved_checkpoint_roots": [str(input_root)],
        "approved_run_root": str(run_root),
        "approved_checkpoint_root": str(ckpt_root),
        "device": "cpu",
        "seed": 37,
        "max_samples_train": 4,
        "max_samples_val": 4,
        "max_image_size": 16,
        "batch_size": 2,
        "epochs": 1,
        "learning_rate": 0.00003,
        "hard_negative_real_oversample_weight": 5,
        "hard_negative_non_tampered_oversample_weight": 4,
        "hard_inconsistent_oversample_weight": 3,
        "hard_positive_low_iou_oversample_weight": 3,
        "class_loss_weight": 1.0,
        "tamper_binary_loss_weight": 1.5,
        "family_loss_weight": 0.2,
        "localization_loss_weight": 12.0,
        "non_tampered_empty_mask_loss_weight": 0.8,
        "dice_loss_weight": 1.0,
        "threshold_tau_values": [0.05, 0.2, 0.5, 0.8],
        "max_false_activation_rate": 0.5,
        "no_write_dry_run": True,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation": True,
        "class_labels": ["real", "full_synthetic", "tampered"],
        "family_labels": ["LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A"],
    }


def test_example_validator() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "training" / "pre_sns_v3_refinement.example.json")
    assert_true(validate_config(cfg) == [], "example config should validate")


def test_validator_rejections() -> None:
    with tempfile.TemporaryDirectory(dir=temp_parent()) as raw:
        cfg = make_fixture(Path(raw))
        assert_true(validate_config(cfg, require_exists=False) == [], "approved dry-run fixture should validate")
        bad = copy.deepcopy(cfg)
        bad["hard_cases_split"] = "val"
        assert_true(validate_config(bad), "hard_cases_split=val must be rejected")
        bad = copy.deepcopy(cfg)
        bad["hard_negative_real_path"] = str(Path(raw) / "inputs" / "hard_mining_full_repaired" / "val" / "hard_negative_real.json")
        assert_true(validate_config(bad), "validation hard-mining path must be rejected")
        bad = copy.deepcopy(cfg)
        bad["no_network"] = False
        assert_true(validate_config(bad), "no_network=false must be rejected")
        bad = copy.deepcopy(cfg)
        bad["no_download"] = False
        assert_true(validate_config(bad), "no_download=false must be rejected")
        bad = copy.deepcopy(cfg)
        bad["no_sns_augmentation"] = False
        assert_true(validate_config(bad), "no_sns_augmentation=false must be rejected")
        bad = copy.deepcopy(cfg)
        bad["approved_run_root"] = str(REPO_ROOT / "tmp_refinement_run")
        assert_true(validate_config(bad), "repo-local run root must be rejected")
        bad = copy.deepcopy(cfg)
        bad["approved_checkpoint_root"] = str(REPO_ROOT / "tmp_refinement_ckpt")
        assert_true(validate_config(bad), "repo-local checkpoint root must be rejected")


def test_oversampling_logic() -> None:
    train = samples()
    hard = {
        "hard_negative_real": [{"sample_id": "real-1"}],
        "hard_negative_non_tampered": [{"image_path": "/tmp/refine-inputs/syn-1.jpg"}],
        "hard_positive_tampered_low_iou": [{"sample_id": "tam-1"}],
        "class_mask_inconsistent_cases": [{"sample_id": "real-2"}, {"sample_id": "missing"}],
    }
    matched, dropped = map_hard_cases_to_train(train, hard)
    expanded, summary = build_oversampled_train_samples(train, matched, {})
    assert_true(dropped["class_mask_inconsistent_cases"] == 1, "non-train hard case should be dropped")
    assert_true(summary["hard_negative_real_count_used"] == 1, "real hard count mismatch")
    assert_true(summary["hard_negative_non_tampered_count_used"] == 1, "non-tampered hard count mismatch")
    assert_true(summary["hard_inconsistent_count_used"] == 1, "inconsistent hard count mismatch")
    assert_true(summary["hard_positive_low_iou_count_used"] == 1, "low-IoU hard count mismatch")
    assert_true(len(expanded) == 4 + 4 + 3 + 2 + 2, "oversampled length should reflect default weights")


def test_dry_run_no_write_and_schema() -> None:
    with tempfile.TemporaryDirectory(dir=temp_parent()) as raw:
        cfg = make_fixture(Path(raw))
        result = run_refinement(cfg)
        assert_true(result["marker"] == "PRE_SNS_V3_REFINEMENT_DRY_RUN_OK", "dry-run marker mismatch")
        assert_true(result["artifact_writes"] is False, "dry-run must not write artifacts")
        assert_true(result["checkpoint_writes"] is False, "dry-run must not write checkpoints")
        assert_true(result["train_samples_seen"] > len(samples()), "dry-run should account for oversampled train records")
        for key in (
            "class_accuracy",
            "class_macro_f1",
            "real_recall",
            "tampered_precision",
            "tampered_recall",
            "tampered_f1",
            "family_accuracy",
            "selected_tau",
            "false_activation_rate",
            "localization_activation_recall",
            "localization_mean_iou",
            "localization_median_iou",
            "hard_negative_real_count_used",
            "hard_negative_non_tampered_count_used",
            "hard_inconsistent_count_used",
            "hard_positive_low_iou_count_used",
            "class_mask_inconsistency_proxy",
            "non_tampered_mask_activation_rate",
        ):
            assert_true(key in result, f"missing result key {key}")
        assert_true(result["non_tampered_mask_activation_rate"] is not None, "non-tampered activation rate should be computed")
        assert_true(not (Path(cfg["approved_run_root"]) / "run_summary.json").exists(), "dry-run should not create run summary")


def test_fixture_artifact_writing_outside_repo() -> None:
    with tempfile.TemporaryDirectory(dir=temp_parent()) as raw:
        cfg = make_fixture(Path(raw))
        cfg["no_write_dry_run"] = False
        dry = run_refinement({**cfg, "no_write_dry_run": True})
        summary = {
            **dry,
            "marker": "PRE_SNS_V3_REFINEMENT_RUN_OK",
            "no_write_dry_run": False,
            "artifact_writes": True,
            "checkpoint_writes": True,
        }
        result = _write_refinement_artifacts(cfg, summary, None)
        assert_true(result["marker"] == "PRE_SNS_V3_REFINEMENT_RUN_OK", "fixture run marker mismatch")
        required_files = [
            "run_summary.json",
            "val_metrics.json",
            "threshold_calibration.json",
            "confusion_matrix.json",
            "per_source_confusion_matrix.json",
            "refinement_hard_case_summary.json",
            "train_metrics.jsonl",
            "artifact_manifest.json",
            "config_snapshot.json",
        ]
        for name in required_files:
            path = Path(cfg["approved_run_root"]) / name
            assert_true(path.exists(), f"missing artifact {name}")
            assert_true(not str(path.resolve()).startswith(str(REPO_ROOT.resolve())), "artifact must be outside repo")
        for name in ("best_checkpoint.pt", "latest_checkpoint.pt"):
            path = Path(cfg["approved_checkpoint_root"]) / name
            assert_true(path.exists(), f"missing checkpoint {name}")
            assert_true(not str(path.resolve()).startswith(str(REPO_ROOT.resolve())), "checkpoint must be outside repo")


def main() -> None:
    test_example_validator()
    test_validator_rejections()
    test_oversampling_logic()
    test_dry_run_no_write_and_schema()
    test_fixture_artifact_writing_outside_repo()
    print("PRE_SNS_V3_REFINEMENT_TESTS_OK")


if __name__ == "__main__":
    main()
