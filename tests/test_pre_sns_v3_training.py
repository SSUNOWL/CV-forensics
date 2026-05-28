#!/usr/bin/env python3
"""Standalone tests for pre-SNS v3 training."""

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

from cv_forensics.pre_sns_v3_metrics import compute_metrics, select_tau, threshold_sweep  # noqa: E402
from cv_forensics.pre_sns_v3_model import CLASS_LABELS  # noqa: E402
from cv_forensics.pre_sns_v3_training import (  # noqa: E402
    APPROVAL_TEXT,
    compute_v3_losses,
    default_loss_weights,
    dice_loss,
    load_config,
    run_training,
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
    return load_config(REPO_ROOT / "configs/training/pre_sns_v3_train.example.json")


def make_fixture(tmp: Path) -> dict:
    from PIL import Image

    img_root = tmp / "fixture_inputs" / "images"
    img_root.mkdir(parents=True)
    samples = []
    labels = ["real", "full_synthetic", "tampered", "tampered", "real", "full_synthetic"]
    families = ["Real-or-N/A", "LatDiff", "GAN", "Other", "Real-or-N/A", "PixDiff"]
    for i, label in enumerate(labels):
        image_path = img_root / f"img_{i}.png"
        mask_path = img_root / f"mask_{i}.png"
        color = (20 + i * 30, 80 + i * 10, 140 - i * 10)
        Image.new("RGB", (18, 18), color).save(image_path)
        sample = {
            "sample_id": f"s{i}",
            "class_label": label,
            "family_label": families[i],
            "image_path": str(image_path),
            "source_dataset": "sid_set" if label == "tampered" else "cf_small",
        }
        if label == "tampered":
            mask = Image.new("L", (18, 18), 0)
            for x in range(5, 13):
                for y in range(5, 13):
                    mask.putpixel((x, y), 255)
            mask.save(mask_path)
            sample["mask_path"] = str(mask_path)
        samples.append(sample)
    train_manifest = tmp / "fixture_inputs" / "manifests" / "train.json"
    val_manifest = tmp / "fixture_inputs" / "manifests" / "val.json"
    write_json(train_manifest, {"samples": samples})
    write_json(val_manifest, {"samples": samples})
    cfg = example_config()
    cfg.update(
        {
            "config_kind": "approved_pre_sns_v3_training",
            "execution_mode": "approved_local_pre_sns_v3_training",
            "user_approval_text": APPROVAL_TEXT,
            "train_manifest_path": str(train_manifest),
            "val_manifest_path": str(val_manifest),
            "approved_input_roots": [str(tmp / "fixture_inputs")],
            "approved_run_root": str(tmp / "cvf_runs" / "v3_run"),
            "approved_checkpoint_root": str(tmp / "cvf_ckpts" / "v3_run"),
            "max_samples_train": 6,
            "max_samples_val": 6,
            "max_image_size": 16,
            "batch_size": 2,
            "epochs": 1,
            "base_channels": 4,
            "no_write_dry_run": True,
        }
    )
    return cfg


def test_config_validation() -> None:
    assert_true(validate_config(example_config()) == [], "example config should pass")
    cfg = example_config()
    cfg["no_sns_augmentation"] = False
    assert_true(validate_config(cfg), "SNS augmentation must be rejected")
    cfg = example_config()
    cfg["approved_run_root"] = "outputs/bad"
    assert_true(validate_config(cfg), "repo outputs path must be rejected")


def test_no_write_dry_run() -> None:
    with tempfile.TemporaryDirectory(dir=temp_parent()) as raw:
        tmp = Path(raw)
        cfg = make_fixture(tmp)
        errors = validate_config(cfg, require_manifest_exists=True)
        assert_true(errors == [], f"approved fixture config should pass: {errors}")
        result = run_training(cfg)
        assert_true(result["marker"] == "PRE_SNS_V3_TRAINING_DRY_RUN_OK", "dry-run marker mismatch")
        assert_true(result["no_write_dry_run"] is True, "dry-run flag missing")
        assert_true(not (tmp / "cvf_runs").exists(), "dry-run wrote run root")
        assert_true(not (tmp / "cvf_ckpts").exists(), "dry-run wrote checkpoint root")
        for key in ("class_accuracy", "class_macro_f1", "real_recall", "tampered_precision", "tampered_recall", "tampered_f1", "selected_tau", "false_activation_rate", "localization_activation_recall"):
            assert_true(key in result, f"result missing {key}")
        metrics = result["val_metrics"]
        for key in ("per_class", "confusion_matrix", "tampered_score_summary_by_gt_class", "threshold_sweep", "mask_area_pct_summary", "per_source_dataset_confusion_matrix", "per_family_confusion_matrix"):
            assert_true(key in metrics, f"metrics missing {key}")


def test_selected_tau_fpr_constraint() -> None:
    predictions = [
        {"gt_class": "real", "pred_class": "real", "tampered_score": 0.10, "gt_family": "Real-or-N/A", "pred_family": "Real-or-N/A"},
        {"gt_class": "full_synthetic", "pred_class": "full_synthetic", "tampered_score": 0.30, "gt_family": "LatDiff", "pred_family": "LatDiff"},
        {"gt_class": "tampered", "pred_class": "tampered", "tampered_score": 0.40, "gt_family": "GAN", "pred_family": "GAN", "localization_iou": 0.2},
        {"gt_class": "tampered", "pred_class": "tampered", "tampered_score": 0.80, "gt_family": "Other", "pred_family": "Other", "localization_iou": 0.7},
    ]
    sweep = threshold_sweep(predictions, [0.2, 0.35, 0.5])
    chosen = select_tau(sweep, 0.20)
    assert_true(chosen["selected_tau"] == 0.35, f"unexpected constrained tau: {chosen}")
    metrics = compute_metrics(predictions, 0.20, [0.2, 0.35, 0.5])
    assert_true(metrics["fallback_selected_tau"] is False, "should not fallback")
    assert_true(metrics["selected_tau"] == 0.35, "metric selected_tau mismatch")


def test_losses() -> None:
    import torch

    logits = torch.tensor([[[[8.0, -8.0], [-8.0, 8.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    assert_true(float(dice_loss(torch, logits, targets)) < 0.01, "dice loss should be low for matching mask")
    outputs = {
        "class_logits": torch.randn(3, 3, requires_grad=True),
        "tamper_binary_logits": torch.randn(3, 2, requires_grad=True),
        "family_logits": torch.randn(3, 5, requires_grad=True),
        "localization_logits": torch.randn(3, 1, 4, 4, requires_grad=True),
    }
    batch = {
        "class_targets": torch.tensor([0, 1, 2]),
        "tamper_binary_targets": torch.tensor([0, 0, 1]),
        "family_targets": torch.tensor([4, 0, 2]),
        "family_supervised_indices": [1, 2],
        "mask_targets": torch.zeros(3, 1, 4, 4),
        "localization_indices": [2],
        "non_tampered_indices": [0, 1],
        "class_weight_tensor": None,
    }
    batch["mask_targets"][2, :, 1:3, 1:3] = 1.0
    losses = compute_v3_losses(torch, outputs, batch, default_loss_weights())
    assert_true(losses["total_loss_finite"], "total loss should be finite")
    assert_true(float(losses["non_tampered_empty_mask_loss"].detach().item()) > 0.0, "empty-mask loss should be active")


def test_tiny_actual_training_writes_outside_repo() -> None:
    with tempfile.TemporaryDirectory(dir=temp_parent()) as raw:
        tmp = Path(raw)
        cfg = make_fixture(tmp)
        cfg["no_write_dry_run"] = False
        errors = validate_config(cfg, require_manifest_exists=True)
        assert_true(errors == [], f"actual fixture config should pass: {errors}")
        result = run_training(cfg)
        assert_true(result["marker"] == "PRE_SNS_V3_TRAINING_RUN_OK", "run marker mismatch")
        for key in ("artifact_manifest_path", "run_summary_path", "val_metrics_path", "threshold_calibration_path", "confusion_matrix_path", "best_checkpoint_path", "latest_checkpoint_path"):
            path = Path(result[key])
            assert_true(path.exists(), f"{key} missing on disk")
            assert_true(str(path).startswith(str(tmp)), f"{key} not under temp external root")
            assert_true(not str(path).startswith(str(REPO_ROOT)), f"{key} wrote inside repo")


def test_reject_bad_approved_paths() -> None:
    with tempfile.TemporaryDirectory(dir=temp_parent()) as raw:
        tmp = Path(raw)
        cfg = make_fixture(tmp)
        bad = copy.deepcopy(cfg)
        bad["approved_checkpoint_root"] = str(REPO_ROOT / "checkpoints" / "bad")
        assert_true(validate_config(bad, require_manifest_exists=True), "repo checkpoint root should fail")
        bad = copy.deepcopy(cfg)
        bad["train_manifest_path"] = str(tmp / "data" / "train.json")
        assert_true(validate_config(bad, require_manifest_exists=True), "protected data path should fail")


def main() -> int:
    for test in (
        test_config_validation,
        test_no_write_dry_run,
        test_selected_tau_fpr_constraint,
        test_losses,
        test_tiny_actual_training_writes_outside_repo,
        test_reject_bad_approved_paths,
    ):
        test()
    print("PRE_SNS_V3_TRAINING_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

