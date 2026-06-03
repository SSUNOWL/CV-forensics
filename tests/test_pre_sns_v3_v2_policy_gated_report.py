#!/usr/bin/env python3
"""Plain Python tests for the pre-SNS v2 policy-gated integrated report."""

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

from cv_forensics.pre_sns_v3_v2_policy_gated_report import (  # noqa: E402
    CONFIG_OK_MARKER,
    MARKER,
    _load_v2_model,
    _model_forward_rgb,
    _tile_boxes,
    build_policy_gated_record,
    run_policy_gated_report,
    threshold_mask,
    validate_policy_gated_report_config,
)
from cv_forensics.pre_sns_v3_tile_localizer_v2_model import build_pre_sns_v3_tile_localizer_v2  # noqa: E402


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def temp_root(prefix: str) -> Path:
    TMP_PARENT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(TMP_PARENT)))


def write_fixture_image(root: Path) -> tuple[Path, Path]:
    from PIL import Image, ImageDraw

    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    image_path = inputs / "image.png"
    mask_path = inputs / "gt.png"
    image = Image.new("RGB", (10, 7), (40, 60, 80))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 5, 5), fill=(220, 30, 30))
    image.save(image_path)
    mask = Image.new("L", (10, 7), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle((2, 2, 5, 5), fill=255)
    mask.save(mask_path)
    return image_path, mask_path


def gt_mask_values() -> list[int]:
    return [1 if 2 <= (idx % 10) <= 5 and 2 <= (idx // 10) <= 5 else 0 for idx in range(70)]


def safe_config(root: Path, image_path: Path, mask_path: Path | None = None) -> dict[str, object]:
    ckpt_root = root / "ckpts"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    (ckpt_root / "long256.pt").write_text("fixture", encoding="utf-8")
    (ckpt_root / "tile_v2.pt").write_text("fixture", encoding="utf-8")
    cfg: dict[str, object] = {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_v3_v2_policy_gated_report",
        "execution_mode": "approved_local_pre_sns_v3_v2_policy_gated_report",
        "long256_checkpoint_path": str(ckpt_root / "long256.pt"),
        "tile_localizer_v2_checkpoint_path": str(ckpt_root / "tile_v2.pt"),
        "image_path": str(image_path),
        "gt_mask_path": str(mask_path) if mask_path else None,
        "sample_list_path": None,
        "approved_input_roots": [str(root / "inputs")],
        "approved_checkpoint_roots": [str(ckpt_root)],
        "output_root": str(root / "reports" / "policy"),
        "device": "cpu",
        "mask_threshold": 0.4,
        "min_area_pct": 0.01,
        "max_area_pct": 35.0,
        "suppress_non_tampered_mask": True,
        "fallback_to_baseline_on_v2_unreliable": True,
        "component_filtering": False,
        "min_largest_component_area_pct": None,
        "max_component_count": None,
        "max_samples": 1,
        "fixture_long256_report": {"class": "tampered", "family": "Other", "tampered_score": 0.9, "baseline_mask": [0] * 70},
        "fixture_v2_probability_mask": [1.0 if value else 0.0 for value in gt_mask_values()],
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }
    return cfg


def write_fixture_v2_checkpoint(path: Path, *, rough_mask_prior: bool) -> None:
    import torch

    model = build_pre_sns_v3_tile_localizer_v2(
        torch,
        tile_size=8,
        input_feature_mode="rgb_edge_residual",
        base_channels=8,
        boundary_head=True,
        confidence_head=True,
        rough_mask_prior=rough_mask_prior,
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "tile_size": 8,
        "input_feature_mode": "rgb_edge_residual",
    }
    torch.save(checkpoint, path)


def standalone_helper_probability_mask(config: dict[str, object], image_path: Path, baseline_mask: list[int]) -> list[float]:
    import numpy as np
    import torch
    from PIL import Image

    device = "cpu"
    model, checkpoint, model_info = _load_v2_model(torch, str(config["tile_localizer_v2_checkpoint_path"]), device, config)
    tile_size = int(model_info["tile_size"])
    tile_stride = int(config.get("tile_stride", max(1, tile_size // 2)))
    resample_bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    resample_nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        rough_full = Image.new("L", (width, height))
        rough_full.putdata([255 if value else 0 for value in baseline_mask])
        accum = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)
        for x1, y1, x2, y2 in _tile_boxes(width, height, tile_size, tile_stride):
            crop = image.crop((x1, y1, x2, y2))
            original_size = crop.size
            rough_crop = rough_full.crop((x1, y1, x2, y2))
            if crop.size != (tile_size, tile_size):
                crop_for_model = crop.resize((tile_size, tile_size), resample_bilinear)
                rough_for_model = rough_crop.resize((tile_size, tile_size), resample_nearest)
            else:
                crop_for_model = crop
                rough_for_model = rough_crop
            prob = _model_forward_rgb(torch, model, model_info, crop_for_model, device, rough_tile_img=rough_for_model)
            if prob.shape != (tile_size, tile_size):
                prob_img = Image.fromarray((prob * 255.0).clip(0, 255).astype("uint8"))
                prob_img = prob_img.resize((tile_size, tile_size), resample_bilinear)
                prob = np.asarray(prob_img).astype("float32") / 255.0
            prob_img = Image.fromarray((prob * 255.0).clip(0, 255).astype("uint8"))
            prob_img = prob_img.resize(original_size, resample_bilinear)
            prob_arr = np.asarray(prob_img).astype("float32") / 255.0
            accum[y1:y2, x1:x2] += prob_arr
            counts[y1:y2, x1:x2] += 1.0
    counts[counts == 0] = 1.0
    return [float(value) for value in (accum / counts).reshape(-1).tolist()]


def test_threshold_mask() -> None:
    assert_equal(threshold_mask([0.1, 0.4, 0.9], 0.4), [0, 1, 1], "threshold mask")


def test_v2_checkpoint_loader_selects_matching_stem_shape() -> None:
    import torch

    root = temp_root("cvf_v2_gate_loader_")
    image_path, mask_path = write_fixture_image(root)
    cfg = safe_config(root, image_path, mask_path)
    ckpt_path = root / "ckpts" / "tile_v2.pt"
    write_fixture_v2_checkpoint(ckpt_path, rough_mask_prior=False)
    cfg["tile_localizer_v2_checkpoint_path"] = str(ckpt_path)
    model, checkpoint, info = _load_v2_model(torch, str(ckpt_path), "cpu", cfg)
    assert_equal(info["checkpoint_stem_shape"], [8, 11, 3, 3], "checkpoint stem shape")
    assert_equal(info["model_stem_shape"], [8, 11, 3, 3], "model stem shape")
    assert_true(info["rough_mask_prior"] is False, "rough prior should be false for 11 channels")


def test_v2_checkpoint_loader_supports_rough_mask_prior_true() -> None:
    import torch

    root = temp_root("cvf_v2_gate_loader_rough_")
    image_path, mask_path = write_fixture_image(root)
    cfg = safe_config(root, image_path, mask_path)
    ckpt_path = root / "ckpts" / "tile_v2.pt"
    write_fixture_v2_checkpoint(ckpt_path, rough_mask_prior=True)
    cfg["tile_localizer_v2_checkpoint_path"] = str(ckpt_path)
    model, checkpoint, info = _load_v2_model(torch, str(ckpt_path), "cpu", cfg)
    assert_equal(info["checkpoint_stem_shape"], [8, 12, 3, 3], "checkpoint stem shape")
    assert_equal(info["model_stem_shape"], [8, 12, 3, 3], "model stem shape")
    assert_true(info["rough_mask_prior"] is True, "rough prior should be true for 12 channels")


def test_v2_rgb_forward_and_helper_consistency() -> None:
    root = temp_root("cvf_v2_gate_consistency_")
    image_path, mask_path = write_fixture_image(root)
    cfg = safe_config(root, image_path, mask_path)
    ckpt_path = root / "ckpts" / "tile_v2.pt"
    write_fixture_v2_checkpoint(ckpt_path, rough_mask_prior=False)
    cfg["tile_localizer_v2_checkpoint_path"] = str(ckpt_path)
    cfg["tile_size"] = 8
    cfg["tile_stride"] = 4
    cfg.pop("fixture_v2_probability_mask", None)
    baseline_mask = list(cfg["fixture_long256_report"]["baseline_mask"])
    ours = build_policy_gated_record(cfg, {"image_path": str(image_path)}, 0)
    helper = standalone_helper_probability_mask(cfg, image_path, baseline_mask)
    ours_values = [float(v) for v in ours["_v2_probability_values"]]
    assert_equal(len(ours_values), len(helper), "probability mask length")
    max_delta = max(abs(a - b) for a, b in zip(ours_values, helper))
    assert_true(max_delta <= 1e-6, f"helper consistency delta={max_delta}")


def test_non_tampered_suppression() -> None:
    root = temp_root("cvf_v2_gate_suppress_")
    image_path, mask_path = write_fixture_image(root)
    cfg = safe_config(root, image_path, mask_path)
    cfg["fixture_long256_report"] = {"class": "real", "family": "Real-or-N/A", "tampered_score": 0.1, "baseline_mask": [1] * 70}
    record = build_policy_gated_record(cfg, {"image_path": str(image_path)}, 0)
    assert_true(not record["tile_localization_activated"], "non-tampered should suppress")
    assert_equal(record["localized_evidence_status"], "suppressed_non_tampered", "suppressed status")
    assert_equal(record["final_mask_source"], "suppressed_non_tampered", "suppressed source")


def test_tampered_v2_mask_use() -> None:
    root = temp_root("cvf_v2_gate_v2_use_")
    image_path, mask_path = write_fixture_image(root)
    cfg = safe_config(root, image_path, mask_path)
    record = build_policy_gated_record(cfg, {"image_path": str(image_path)}, 0)
    assert_true(record["tile_localization_activated"], "tampered activates v2")
    assert_equal(record["final_mask_source"], "tile_v2", "v2 should be selected")
    assert_true(record["final_iou"] is not None and float(record["final_iou"]) > 0.0, "final iou should exist")


def test_area_too_small_fallback() -> None:
    root = temp_root("cvf_v2_gate_small_")
    image_path, mask_path = write_fixture_image(root)
    cfg = safe_config(root, image_path, mask_path)
    cfg["min_area_pct"] = 40.0
    cfg["fixture_long256_report"] = {"class": "tampered", "family": "Other", "tampered_score": 0.9, "baseline_mask": gt_mask_values()}
    record = build_policy_gated_record(cfg, {"image_path": str(image_path)}, 0)
    assert_equal(record["final_mask_source"], "baseline_fallback", "small area fallback")
    assert_true("mask_area_too_small" in record["reliability_reasons"], "small area reason")


def test_area_too_large_fallback() -> None:
    root = temp_root("cvf_v2_gate_large_")
    image_path, mask_path = write_fixture_image(root)
    cfg = safe_config(root, image_path, mask_path)
    cfg["fixture_long256_report"] = {"class": "tampered", "family": "Other", "tampered_score": 0.9, "baseline_mask": gt_mask_values()}
    cfg["fixture_v2_probability_mask"] = [1.0] * 70
    record = build_policy_gated_record(cfg, {"image_path": str(image_path)}, 0)
    assert_equal(record["final_mask_source"], "baseline_fallback", "large area fallback")
    assert_true("mask_area_too_large" in record["reliability_reasons"], "large area reason")


def test_final_iou_computation() -> None:
    root = temp_root("cvf_v2_gate_iou_")
    image_path, mask_path = write_fixture_image(root)
    cfg = safe_config(root, image_path, mask_path)
    cfg["fixture_long256_report"] = {"class": "tampered", "family": "Other", "tampered_score": 0.9, "baseline_mask": [0] * 70}
    record = build_policy_gated_record(cfg, {"image_path": str(image_path)}, 0)
    assert_equal(record["baseline_iou"], 0.0, "baseline iou")
    assert_true(float(record["final_iou"]) > float(record["baseline_iou"]), "final iou improvement")
    assert_true(float(record["final_iou_delta_vs_baseline"]) > 0.0, "iou delta positive")


def test_validator_guardrails() -> None:
    example_path = REPO_ROOT / "configs" / "inference" / "pre_sns_v3_v2_policy_gated_report.example.json"
    example = json.loads(example_path.read_text(encoding="utf-8"))
    assert_equal(validate_policy_gated_report_config(example, require_exists=False), [], "example config")
    root = temp_root("cvf_v2_gate_validator_")
    image_path, mask_path = write_fixture_image(root)
    cfg = safe_config(root, image_path, mask_path)
    assert_equal(validate_policy_gated_report_config(cfg, require_exists=False), [], "safe config")
    unsafe = dict(cfg)
    unsafe["output_root"] = str(REPO_ROOT / "policy_out")
    assert_true(validate_policy_gated_report_config(unsafe, require_exists=False), "repo output rejected")
    for key in ("no_network", "no_download", "no_training", "no_sns_augmentation"):
        unsafe = dict(cfg)
        unsafe[key] = False
        assert_true(validate_policy_gated_report_config(unsafe, require_exists=False), f"{key} rejected")


def test_run_policy_gated_report_no_repo_writes() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_v2_gate_run_")
    image_path, mask_path = write_fixture_image(root)
    cfg = safe_config(root, image_path, mask_path)
    result = run_policy_gated_report(cfg)
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "no repo root writes")
    assert_equal(result["marker"], MARKER, "marker")
    out = Path(cfg["output_root"])
    for name in (
        "policy_gated_report.json",
        "final_mask.png",
        "final_clean_red_overlay.png",
        "tile_v2_clean_red_overlay.png",
        "tile_v2_probability_mask.png",
        "comparison_sheet.jpg",
        "artifact_manifest.json",
    ):
        assert_true((out / name).is_file(), f"{name} exists")
    assert_true((out / "baseline_clean_red_overlay.png").is_file(), "baseline overlay exists")


def main() -> int:
    tests = [
        test_threshold_mask,
        test_v2_checkpoint_loader_selects_matching_stem_shape,
        test_v2_checkpoint_loader_supports_rough_mask_prior_true,
        test_v2_rgb_forward_and_helper_consistency,
        test_non_tampered_suppression,
        test_tampered_v2_mask_use,
        test_area_too_small_fallback,
        test_area_too_large_fallback,
        test_final_iou_computation,
        test_validator_guardrails,
        test_run_policy_gated_report_no_repo_writes,
    ]
    for test in tests:
        test()
    print(CONFIG_OK_MARKER)
    print("PRE_SNS_V3_V2_POLICY_GATED_REPORT_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
