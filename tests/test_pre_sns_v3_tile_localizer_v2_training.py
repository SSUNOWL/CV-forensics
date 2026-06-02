#!/usr/bin/env python3
"""Plain python tests for the pre-SNS tile localizer v2 training path."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cv_forensics.pre_sns_v3_tile_localizer_v2_model import (  # noqa: E402
    build_pre_sns_v3_tile_localizer_v2,
    feature_channel_count,
    fixed_forensic_features,
)
from src.cv_forensics.pre_sns_v3_tile_localizer_v2_training import (  # noqa: E402
    APPROVAL_TEXT,
    APPROVED_KIND,
    APPROVED_MODE,
    DRY_RUN_MARKER,
    MARKER,
    PROGRESS_MARKER,
    TileLocalizerV2Dataset,
    compute_losses,
    evaluate,
    expanded_crop_box,
    load_config,
    oversample_records,
    run_training,
    validate_config,
)


EXTERNAL_ROOT = Path("/home/rlatjswo/.codex/memories/pre_sns_v3_tile_localizer_v2_test")


def reset_root() -> Path:
    if EXTERNAL_ROOT.exists():
        shutil.rmtree(EXTERNAL_ROOT)
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    return EXTERNAL_ROOT


def write_fixture(root: Path) -> Path:
    from PIL import Image, ImageDraw

    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    records = []
    for idx in range(5):
        img = Image.new("RGB", (20, 20), (35 + idx * 20, 60, 95))
        draw = ImageDraw.Draw(img)
        draw.rectangle([5, 5, 12, 12], fill=(180, 40 + idx * 20, 30))
        image_path = inputs / f"image_{idx}.png"
        img.save(image_path)

        mask_path = inputs / f"mask_{idx}.png"
        mask = Image.new("L", (20, 20), 0)
        ImageDraw.Draw(mask).rectangle([6, 6, 11, 11], fill=255)
        mask.save(mask_path)

        if idx == 0:
            tile_class, bucket, expected = "positive_tampered", "severe_iou_fail", "cropped_gt_mask"
            source_mask_path = str(mask_path)
        elif idx == 1:
            tile_class, bucket, expected = "positive_tampered", "low_iou", "cropped_gt_mask"
            source_mask_path = str(mask_path)
        elif idx == 2:
            tile_class, bucket, expected = "positive_tampered", "weak_iou", "cropped_gt_mask"
            source_mask_path = str(mask_path)
        elif idx == 3:
            tile_class, bucket, expected = "negative_real", "negative", "empty_mask"
            source_mask_path = None
        else:
            tile_class, bucket, expected = "hard_negative", "hard_negative", "empty_mask"
            source_mask_path = None
        rec = {
            "source_image_path": str(image_path),
            "crop_box": [4, 4, 12, 12],
            "tile_size": 8,
            "tile_class": tile_class,
            "mining_bucket": bucket,
            "expected_mask_type": expected,
        }
        if source_mask_path:
            rec["source_mask_path"] = source_mask_path
        records.append(rec)

    manifest = inputs / "tile_manifest.json"
    manifest.write_text(json.dumps({"tile_records": records}, indent=2), encoding="utf-8")
    return manifest


def base_config(root: Path, manifest: Path, *, no_write: bool) -> dict:
    return {
        "schema_version": 1,
        "config_kind": APPROVED_KIND,
        "execution_mode": APPROVED_MODE,
        "required_approval_text": APPROVAL_TEXT,
        "user_approval_text": APPROVAL_TEXT,
        "tile_manifest_path": str(manifest),
        "approved_input_roots": [str(root / "inputs")],
        "approved_run_root": str(root / "run"),
        "approved_checkpoint_root": str(root / "ckpt"),
        "device": "cpu",
        "seed": 46,
        "tile_size": 16,
        "input_feature_mode": "rgb_edge_residual",
        "base_channels": 2,
        "boundary_head": True,
        "confidence_head": True,
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "max_tiles_train": 10,
        "max_tiles_val": 2,
        "gradient_accumulation_steps": 1,
        "gradient_clip_norm": 1.0,
        "mixed_precision": False,
        "progress_log_interval_steps": 1,
        "progress_write_json": True,
        "progress_write_jsonl": True,
        "stdout_progress_interval_steps": 1,
        "severe_iou_oversample_factor": 2,
        "low_iou_oversample_factor": 2,
        "weak_iou_oversample_factor": 2,
        "hard_negative_oversample_factor": 2,
        "negative_oversample_factor": 2,
        "bce_loss_weight": 1.0,
        "dice_loss_weight": 2.0,
        "tversky_loss_weight": 1.0,
        "boundary_loss_weight": 0.5,
        "empty_mask_loss_weight": 2.0,
        "false_activation_area_loss_weight": 1.0,
        "threshold_values": [0.25, 0.5, 0.75],
        "negative_activation_area_pct": 0.1,
        "no_write_dry_run": no_write,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation": True,
    }


def read_records(manifest: Path) -> list[dict]:
    return json.loads(manifest.read_text(encoding="utf-8"))["tile_records"]


def test_fixed_features_and_model_forward() -> None:
    import torch

    x = torch.rand(2, 3, 8, 8)
    assert fixed_forensic_features(torch, x, "rgb_only").shape == (2, 3, 8, 8)
    assert fixed_forensic_features(torch, x, "rgb_residual").shape == (2, 7, 8, 8)
    assert fixed_forensic_features(torch, x, "rgb_edge_residual").shape == (2, 11, 8, 8)
    assert feature_channel_count("rgb_edge_residual") == 11

    model = build_pre_sns_v3_tile_localizer_v2(
        torch, tile_size=16, input_feature_mode="rgb_only", base_channels=2, boundary_head=True, confidence_head=True
    )
    out = model(torch.rand(2, 3, 16, 16))
    assert out["mask_logits"].shape == (2, 1, 16, 16)
    assert out["boundary_logits"].shape == (2, 1, 16, 16)
    assert out["tile_confidence"].shape == (2, 1)

    model = build_pre_sns_v3_tile_localizer_v2(
        torch, tile_size=16, input_feature_mode="rgb_edge_residual", base_channels=2, boundary_head=True, confidence_head=False
    )
    assert model(torch.rand(1, 3, 16, 16))["mask_logits"].shape == (1, 1, 16, 16)


def test_crop_dataset_loss_metrics_and_oversampling() -> None:
    import torch
    from PIL import Image

    root = reset_root()
    manifest = write_fixture(root)
    records = read_records(manifest)

    assert expanded_crop_box([4, 4, 12, 12], (20, 20), 16) == [0, 0, 16, 16]

    ds = TileLocalizerV2Dataset(records, 16, torch, Image)
    assert ds[0]["mask"].sum().item() > 0
    assert ds[3]["mask"].sum().item() == 0

    cfg = base_config(root, manifest, no_write=True)
    batch = {
        "images": torch.stack([ds[0]["image"], ds[3]["image"]]),
        "masks": torch.stack([ds[0]["mask"], ds[3]["mask"]]),
        "is_positive": torch.stack([ds[0]["is_positive"], ds[3]["is_positive"]]),
        "records": [records[0], records[3]],
    }
    model = build_pre_sns_v3_tile_localizer_v2(torch, tile_size=16, input_feature_mode="rgb_edge_residual", base_channels=2)
    losses = compute_losses(torch, model(batch["images"]), batch, cfg)
    assert torch.isfinite(losses["total_loss"]).item()

    metrics, rows, calibration = evaluate(torch, model, ds, cfg, "cpu")
    assert len(rows) == len(ds)
    assert metrics["tile_mean_iou"] >= 0.0
    assert metrics["tile_mean_dice"] >= 0.0
    assert calibration["selected_mask_threshold"] in cfg["threshold_values"]

    expanded = oversample_records(records, cfg)
    assert len(expanded) > len(records)
    assert sum(1 for r in expanded if r["tile_class"] == "hard_negative") >= 2


def test_validator_and_training_entrypoint_guards() -> None:
    root = reset_root()
    manifest = write_fixture(root)
    example = load_config(REPO_ROOT / "configs/training/pre_sns_v3_tile_localizer_v2_train.example.json")
    assert validate_config(example, require_exists=False) == []

    cfg = base_config(root, manifest, no_write=True)
    assert validate_config(cfg, require_exists=False) == []
    bad_interval = dict(cfg)
    bad_interval["progress_log_interval_steps"] = -1
    assert any("progress_log_interval_steps must be a positive integer" in err for err in validate_config(bad_interval, require_exists=False))
    bad = dict(cfg)
    bad["approved_run_root"] = str(REPO_ROOT / "tmp_v2_run")
    assert any("outside the repository" in err for err in validate_config(bad, require_exists=False))

    before = sorted(p.relative_to(root) for p in root.rglob("*"))
    result = run_training(cfg)
    after = sorted(p.relative_to(root) for p in root.rglob("*"))
    assert result["marker"] == DRY_RUN_MARKER
    assert before == after
    assert not (root / "run" / "progress.json").exists()
    assert not (root / "run" / "progress.jsonl").exists()


def test_tiny_actual_training_writes_expected_external_artifacts() -> None:
    root = reset_root()
    manifest = write_fixture(root)
    cfg = base_config(root, manifest, no_write=False)

    result = run_training(cfg)
    assert result["marker"] == MARKER
    assert result["training_completed"] is True

    run_root = Path(cfg["approved_run_root"])
    ckpt_root = Path(cfg["approved_checkpoint_root"])
    for name in (
        "run_summary.json",
        "val_metrics.json",
        "threshold_calibration.json",
        "tile_metrics.jsonl",
        "failure_bucket_metrics.json",
        "artifact_manifest.json",
        "config_snapshot.json",
        "progress.json",
        "progress.jsonl",
    ):
        assert (run_root / name).exists(), name
    progress = json.loads((run_root / "progress.json").read_text(encoding="utf-8"))
    assert progress["marker"] == PROGRESS_MARKER
    assert isinstance(progress["percent_complete"], (int, float))
    lines = [line for line in (run_root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 1
    first_line = json.loads(lines[0])
    assert first_line["marker"] == PROGRESS_MARKER
    assert (run_root / "visual_samples").is_dir()
    assert (ckpt_root / "best_tile_localizer_v2.pt").exists()
    assert (ckpt_root / "latest_tile_localizer_v2.pt").exists()
    assert str(run_root).startswith(str(EXTERNAL_ROOT))
    assert str(ckpt_root).startswith(str(EXTERNAL_ROOT))


def main() -> None:
    test_fixed_features_and_model_forward()
    test_crop_dataset_loss_metrics_and_oversampling()
    test_validator_and_training_entrypoint_guards()
    test_tiny_actual_training_writes_expected_external_artifacts()
    print("PRE_SNS_V3_TILE_LOCALIZER_V2_TRAINING_TESTS_OK")


if __name__ == "__main__":
    main()
