#!/usr/bin/env python3
"""Plain Python tests for pre-SNS v3 tile-localizer training."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TMP_PARENT = Path("/home/rlatjswo/.codex/memories")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_v3_tile_localizer_model import build_pre_sns_v3_tile_localizer  # noqa: E402
from cv_forensics.pre_sns_v3_tile_localizer_training import (  # noqa: E402
    APPROVAL_TEXT,
    MARKER,
    TileLocalizationDataset,
    compute_losses,
    evaluate_model,
    load_config,
    make_batch,
    run_training,
    validate_config,
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


def runtime_deps():
    import torch
    from PIL import Image, ImageDraw

    return torch, Image, ImageDraw


def write_fixture_manifest(root: Path, tile_size: int = 8) -> Path:
    _torch, Image, ImageDraw = runtime_deps()
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(5):
        image_path = inputs / f"image_{index}.png"
        mask_path = inputs / f"mask_{index}.png"
        image = Image.new("RGB", (16, 16), (20 + index * 20, 40, 80))
        draw = ImageDraw.Draw(image)
        draw.rectangle((4, 4, 11, 11), fill=(220, 40, 40))
        image.save(image_path)
        mask = Image.new("L", (16, 16), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle((4, 4, 11, 11), fill=255)
        mask.save(mask_path)
        if index < 3:
            records.append(
                {
                    "source_sample_id": f"pos_{index}",
                    "source_image_path": str(image_path),
                    "source_mask_path": str(mask_path),
                    "crop_box": [2, 2, 10, 10],
                    "tile_size": tile_size,
                    "tile_class": "positive_tampered",
                    "mining_bucket": "severe_iou_fail",
                    "expected_mask_type": "cropped_gt_mask",
                }
            )
        else:
            records.append(
                {
                    "source_sample_id": f"neg_{index}",
                    "source_image_path": str(image_path),
                    "crop_box": [0, 0, 8, 8],
                    "tile_size": tile_size,
                    "tile_class": "negative_real",
                    "expected_mask_type": "empty_mask",
                }
            )
    manifest = {"marker": "PRE_SNS_V3_GT_IOU_TILE_BUILDER_OK", "tile_size": tile_size, "records": records}
    path = inputs / "tile_localization_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def safe_config(root: Path, no_write: bool = True) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_v3_tile_localizer_training",
        "execution_mode": "approved_local_pre_sns_v3_tile_localizer_training",
        "required_approval_text": APPROVAL_TEXT,
        "user_approval_text": APPROVAL_TEXT,
        "tile_manifest_path": str(root / "inputs" / "tile_localization_manifest.json"),
        "approved_input_roots": [str(root / "inputs")],
        "approved_run_root": str(root / "runs" / "tile_localizer"),
        "approved_checkpoint_root": str(root / "ckpts" / "tile_localizer"),
        "device": "cpu",
        "seed": 43,
        "tile_size": 8,
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "base_channels": 2,
        "max_tiles_train": 4,
        "max_tiles_val": 1,
        "gradient_clip_norm": 1.0,
        "bce_loss_weight": 1.0,
        "dice_loss_weight": 1.0,
        "focal_loss_weight": 0.0,
        "empty_mask_loss_weight": 1.0,
        "threshold_values": [0.25, 0.5],
        "negative_activation_area_pct": 0.1,
        "no_write_dry_run": no_write,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation": True,
    }


def test_model_forward_loss_and_metrics() -> None:
    torch, _Image, _ImageDraw = runtime_deps()
    model = build_pre_sns_v3_tile_localizer(torch, base_channels=2)
    images = torch.rand(2, 3, 8, 8)
    outputs = model(images)
    assert_equal(tuple(outputs["mask_logits"].shape), (2, 1, 8, 8), "mask logits shape")
    batch = {"masks": torch.zeros(2, 1, 8, 8), "is_positive": torch.zeros(2, 1)}
    losses = compute_losses(torch, outputs, batch, safe_config(temp_root("cvf_tile_loss_")))
    assert_true(torch.isfinite(losses["total_loss"]).item(), "finite total loss")


def test_dataset_positive_and_negative_crops() -> None:
    torch, Image, _ImageDraw = runtime_deps()
    root = temp_root("cvf_tile_dataset_")
    manifest_path = write_fixture_manifest(root)
    records = json.loads(manifest_path.read_text())["records"]
    dataset = TileLocalizationDataset(records, 8, torch, Image)
    pos = dataset[0]
    neg = dataset[-1]
    assert_equal(tuple(pos["image"].shape), (3, 8, 8), "image tensor shape")
    assert_true(float(pos["mask"].sum().item()) > 0.0, "positive crop has mask")
    assert_equal(float(neg["mask"].sum().item()), 0.0, "negative crop empty mask")


def test_evaluate_metrics_finite() -> None:
    torch, Image, _ImageDraw = runtime_deps()
    root = temp_root("cvf_tile_metrics_")
    manifest_path = write_fixture_manifest(root)
    records = json.loads(manifest_path.read_text())["records"][:2]
    dataset = TileLocalizationDataset(records, 8, torch, Image)
    model = build_pre_sns_v3_tile_localizer(torch, base_channels=2)
    metrics, rows, calibration = evaluate_model(torch, model, dataset, safe_config(root), "cpu")
    assert_true(rows, "tile metric rows")
    assert_true(calibration["selected_mask_threshold"] in {0.25, 0.5}, "selected threshold")
    for key in ("tile_mean_iou", "tile_median_iou", "tile_mean_dice", "negative_tile_false_activation_rate"):
        assert_true(isinstance(metrics[key], float), f"{key} finite")


def test_validator_accepts_example_and_rejects_repo_outputs() -> None:
    example = load_config(REPO_ROOT / "configs" / "training" / "pre_sns_v3_tile_localizer_train.example.json")
    assert_equal(validate_config(example, require_exists=False), [], "example config")
    root = temp_root("cvf_tile_validator_")
    write_fixture_manifest(root)
    cfg = safe_config(root)
    assert_equal(validate_config(cfg, require_exists=False), [], "safe fixture config")
    repo_run = dict(cfg)
    repo_run["approved_run_root"] = str(REPO_ROOT / "tile_localizer_run")
    assert_true(validate_config(repo_run, require_exists=False), "repo run root rejected")
    unsafe = dict(cfg)
    unsafe["no_sns_augmentation"] = False
    assert_true(validate_config(unsafe, require_exists=False), "SNS rejected")
    unsafe = dict(cfg)
    unsafe["no_network"] = False
    assert_true(validate_config(unsafe, require_exists=False), "network rejected")
    unsafe = dict(cfg)
    unsafe["no_download"] = False
    assert_true(validate_config(unsafe, require_exists=False), "download rejected")


def test_no_write_dry_run_writes_no_artifacts() -> None:
    root = temp_root("cvf_tile_dry_")
    write_fixture_manifest(root)
    before = set(os.listdir(root))
    summary = run_training(safe_config(root, no_write=True))
    after = set(os.listdir(root))
    assert_equal(before, after, "dry run wrote no new root entries")
    assert_true(summary["artifact_writes"] is False, "no artifacts")
    assert_true(summary["checkpoint_writes"] is False, "no checkpoints")


def test_tiny_actual_training_writes_expected_artifacts() -> None:
    root = temp_root("cvf_tile_train_")
    write_fixture_manifest(root)
    cfg = safe_config(root, no_write=False)
    summary = run_training(cfg)
    assert_equal(summary["marker"], MARKER, "run marker")
    run_root = Path(cfg["approved_run_root"])
    ckpt_root = Path(cfg["approved_checkpoint_root"])
    for name in ("run_summary.json", "val_metrics.json", "threshold_calibration.json", "tile_metrics.jsonl", "artifact_manifest.json", "config_snapshot.json"):
        assert_true((run_root / name).is_file(), f"{name} exists")
    assert_true((ckpt_root / "best_tile_localizer.pt").is_file(), "best checkpoint exists")
    assert_true((ckpt_root / "latest_tile_localizer.pt").is_file(), "latest checkpoint exists")


def main() -> int:
    tests = [
        test_model_forward_loss_and_metrics,
        test_dataset_positive_and_negative_crops,
        test_evaluate_metrics_finite,
        test_validator_accepts_example_and_rejects_repo_outputs,
        test_no_write_dry_run_writes_no_artifacts,
        test_tiny_actual_training_writes_expected_artifacts,
    ]
    for test in tests:
        test()
    print("PRE_SNS_V3_TILE_LOCALIZER_TRAINING_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

