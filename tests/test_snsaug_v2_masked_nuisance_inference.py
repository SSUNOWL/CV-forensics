#!/usr/bin/env python3
"""Plain Python tests for SNSAug v2 masked nuisance inference."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TMP_PARENT = Path("/home/rlatjswo/.codex/memories")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2_masked_nuisance_inference import (  # noqa: E402
    MARKER,
    aggregate_per_policy_profile,
    apply_masking_policy,
    compute_recovery_metrics,
    dilate_mask,
    load_snsaug_v2_masked_nuisance_inference_config,
    mask_area_ratio,
    resolve_pair_mask_path,
    run_snsaug_v2_masked_nuisance_inference,
    validate_snsaug_v2_masked_nuisance_inference_config,
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


def _mask_box(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


def _base_mask(size: tuple[int, int], box: tuple[int, int, int, int]) -> list[int]:
    x1, y1, x2, y2 = box
    return [1 if x1 <= x <= x2 and y1 <= y <= y2 else 0 for y in range(size[1]) for x in range(size[0])]


def write_fixture(root: Path) -> tuple[Path, Path, Path]:
    pair_root = root / "pairs_0058c_eval"
    images_dir = pair_root / "images"
    ignore_dir = pair_root / "ignore_masks"
    tamper_dir = pair_root / "tamper_masks"
    bundle_dir = root / "bundle"
    for path in (images_dir, ignore_dir, tamper_dir, bundle_dir):
        path.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "long.pt").write_text("fixture", encoding="utf-8")
    (bundle_dir / "tile.pt").write_text("fixture", encoding="utf-8")
    bundle = {
        "long256_checkpoint_path": str(bundle_dir / "long.pt"),
        "tile_v2_checkpoint_path": str(bundle_dir / "tile.pt"),
        "policy_gated_report": {
            "mask_threshold": 0.4,
            "min_area_pct": 0.01,
            "max_area_pct": 100.0,
            "suppress_non_tampered_mask": True,
            "fallback_to_baseline_on_v2_unreliable": True,
            "tile_size": 8,
            "tile_stride": 4
        }
    }
    bundle_path = bundle_dir / "best_bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    size = (10, 8)
    tamper_mask = _base_mask(size, (3, 2, 6, 5))
    empty_mask = [0] * (size[0] * size[1])
    rows: list[dict[str, object]] = []
    for base_id, label in (("real_1", "real"), ("synthetic_1", "synthetic"), ("tampered_1", "tampered")):
        clean_img = Image.new("RGB", size, (40, 80, 120))
        ImageDraw.Draw(clean_img).rectangle((3, 2, 6, 5), fill=(210, 40, 40))
        clean_path = images_dir / f"{base_id}__clean.png"
        clean_img.save(clean_path)
        clean_ignore = ignore_dir / f"{base_id}__clean.png"
        Image.new("L", size, 0).save(clean_ignore)
        gt_path = None
        if label == "tampered":
            gt_path = tamper_dir / f"{base_id}__clean.png"
            _mask_box(size, (3, 2, 6, 5)).save(gt_path)
        clean_fixture = {
            "class": label,
            "class_conf": {"real": 0.05, "synthetic": 0.05, "tampered": 0.90} if label == "tampered" else {"real": 0.95 if label == "real" else 0.03, "synthetic": 0.95 if label == "synthetic" else 0.03, "tampered": 0.02},
            "family": "Other",
            "tampered_score": 0.90 if label == "tampered" else 0.02,
            "baseline_mask": tamper_mask if label == "tampered" else empty_mask,
        }
        rows.append(
            {
                "base_id": base_id,
                "content_label": label,
                "view": "clean",
                "profile": "clean",
                "image_path": str(clean_path),
                "ignore_mask_path": str(clean_ignore),
                "tamper_mask_path": str(gt_path) if gt_path else None,
                "fixture_long256_report": clean_fixture,
                "fixture_v2_probability_mask": [float(value) for value in (tamper_mask if label == "tampered" else empty_mask)],
            }
        )
        sns_path = images_dir / f"{base_id}__tiktok_like.png"
        clean_img.save(sns_path)
        sns_ignore = ignore_dir / f"{base_id}__tiktok_like.png"
        _mask_box(size, (0, 0, 2, 2)).save(sns_ignore)
        sns_fixture = clean_fixture
        if label == "tampered":
            sns_fixture = {
                "class": "real",
                "class_conf": {"real": 0.7, "synthetic": 0.05, "tampered": 0.25},
                "family": "Other",
                "tampered_score": 0.25,
                "baseline_mask": empty_mask,
            }
        rows.append(
            {
                "base_id": base_id,
                "content_label": label,
                "view": "sns_aug",
                "profile": "tiktok_like",
                "image_path": str(sns_path),
                "ignore_mask_path": str(sns_ignore),
                "tamper_mask_path": str(gt_path) if gt_path else None,
                "fixture_long256_report": sns_fixture,
                "fixture_v2_probability_mask": [0.0 for _ in empty_mask],
                "fixture_by_policy": {
                    "gray_fill": {
                        "fixture_long256_report": clean_fixture,
                        "fixture_v2_probability_mask": [float(value) for value in (tamper_mask if label == "tampered" else empty_mask)],
                    }
                },
            }
        )
    meta_path = pair_root / "meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return pair_root, meta_path, bundle_path


def safe_config(root: Path, pair_root: Path, meta_path: Path, bundle_path: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_masked_nuisance_inference",
        "execution_mode": "approved_local_snsaug_v2_masked_nuisance_inference",
        "user_approval_text": "I_APPROVE_SNSAUG_V2_MASKED_NUISANCE_INFERENCE",
        "best_bundle_path": str(bundle_path),
        "pair_root": str(pair_root),
        "meta_jsonl_path": str(meta_path),
        "approved_input_roots": [str(root / "bundle"), str(pair_root)],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "masked"),
        "policies": ["original", "gray_fill", "blur_fill"],
        "profiles": ["clean", "tiktok_like"],
        "device": "cpu",
        "max_samples": 10,
        "dilation_radius": 1,
        "top_n_gallery": 4,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def test_example_config_validates() -> None:
    cfg = load_snsaug_v2_masked_nuisance_inference_config(REPO_ROOT / "configs" / "evaluation" / "snsaug_v2_masked_nuisance_inference.example.json")
    assert_equal(validate_snsaug_v2_masked_nuisance_inference_config(cfg, require_exists=False), [], "example config validates")


def test_ignore_mask_path_resolution() -> None:
    root = temp_root("cvf_0067_resolve_")
    pair_root, _meta_path, _bundle = write_fixture(root)
    row = {"base_id": "real_1", "profile": "tiktok_like", "image_path": str(pair_root / "images" / "real_1__tiktok_like.png")}
    resolved = resolve_pair_mask_path(row, pair_root, "ignore_mask_path")
    assert_true(resolved is not None and resolved.endswith("real_1__tiktok_like.png"), "ignore mask resolved from pair root")


def test_mask_policies_modify_only_ignore_region_and_dilate() -> None:
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    ImageDraw.Draw(image).point((1, 1), fill=(200, 10, 10))
    mask = Image.new("L", (4, 4), 0)
    ImageDraw.Draw(mask).point((1, 1), fill=255)
    gray, used = apply_masking_policy(image, mask, "gray_fill")
    blur, _used_blur = apply_masking_policy(image, mask, "blur_fill")
    assert_equal(gray.getpixel((0, 0)), image.getpixel((0, 0)), "gray outside unchanged")
    assert_equal(gray.getpixel((1, 1)), (128, 128, 128), "gray inside changed")
    assert_equal(blur.getpixel((0, 0)), image.getpixel((0, 0)), "blur outside unchanged")
    assert_true(blur.getpixel((1, 1)) != image.getpixel((1, 1)), "blur inside changed")
    dilated = dilate_mask(used, radius=1)
    assert_true(mask_area_ratio(dilated) > mask_area_ratio(used), "dilation increases mask area")


def test_metrics_handle_no_tampered_rows_safely() -> None:
    records = [
        {"policy": "original", "profile": "clean", "content_label": "real", "pred_class": "real", "class_correct": True, "p_tampered": 0.01, "localization_activated": False, "final_mask_area_pct": 0.0, "mask_area_ratio": 0.0},
        {"policy": "gray_fill", "profile": "clean", "content_label": "real", "pred_class": "real", "class_correct": True, "p_tampered": 0.01, "localization_activated": False, "final_mask_area_pct": 0.0, "mask_area_ratio": 0.0},
    ]
    metrics = aggregate_per_policy_profile(records)
    recovery = compute_recovery_metrics(metrics)
    assert_true(metrics["original"]["clean"]["tampered_recall"] is None, "no tampered recall is NA")
    assert_true(recovery["gray_fill"]["clean"]["tampered_recall_recovery"] is None, "recovery handles NA")


def test_config_validator_rejects_training_flags() -> None:
    root = temp_root("cvf_0067_flags_")
    pair_root, meta_path, bundle_path = write_fixture(root)
    cfg = safe_config(root, pair_root, meta_path, bundle_path)
    cfg["no_training"] = False
    cfg["no_finetune"] = False
    errors = validate_snsaug_v2_masked_nuisance_inference_config(cfg, require_exists=False)
    assert_true(any("no_training" in error for error in errors), "no_training required")
    assert_true(any("no_finetune" in error for error in errors), "no_finetune required")


def test_dry_run_writes_no_records() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_0067_dry_")
    pair_root, meta_path, bundle_path = write_fixture(root)
    cfg = safe_config(root, pair_root, meta_path, bundle_path)
    summary = run_snsaug_v2_masked_nuisance_inference(cfg, dry_run=True)
    assert_equal(before, set(os.listdir(REPO_ROOT)), "no repo writes")
    assert_true(summary["inference_started"] is False, "dry-run does not infer")
    assert_true(not Path(cfg["output_root"]).exists(), "dry-run writes no records")


def test_tiny_run_writes_summary_and_records() -> None:
    root = temp_root("cvf_0067_run_")
    pair_root, meta_path, bundle_path = write_fixture(root)
    cfg = safe_config(root, pair_root, meta_path, bundle_path)
    summary = run_snsaug_v2_masked_nuisance_inference(cfg, dry_run=False)
    assert_true(summary["inference_started"] is True, "inference started")
    assert_true(summary["record_count"] > 0, "records produced")
    for key in ("model_eval_records_masked", "per_policy_per_profile_metrics", "clean_vs_sns_masked_delta", "masked_nuisance_inference_summary", "masked_nuisance_inference_report", "visual_gallery_manifest"):
        path = Path(summary["output_paths"][key])
        assert_true(path.exists(), f"{key} exists")
        assert_true(not str(path).startswith(str(REPO_ROOT)), f"{key} outside repo")
    rows = [json.loads(line) for line in Path(summary["output_paths"]["model_eval_records_masked"]).read_text(encoding="utf-8").splitlines()]
    assert_true(any(row["policy"] == "gray_fill" for row in rows), "gray_fill rows written")
    metrics = json.loads(Path(summary["output_paths"]["per_policy_per_profile_metrics"]).read_text(encoding="utf-8"))
    assert_true("original" in metrics and "gray_fill" in metrics, "per-policy metrics written")


def test_docs_marker_present() -> None:
    text = (REPO_ROOT / "docs" / "snsaug_v2_masked_nuisance_inference.md").read_text(encoding="utf-8")
    assert_true(MARKER in text, "docs marker present")


def main() -> int:
    tests = [
        test_example_config_validates,
        test_ignore_mask_path_resolution,
        test_mask_policies_modify_only_ignore_region_and_dilate,
        test_metrics_handle_no_tampered_rows_safely,
        test_config_validator_rejects_training_flags,
        test_dry_run_writes_no_records,
        test_tiny_run_writes_summary_and_records,
        test_docs_marker_present,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
