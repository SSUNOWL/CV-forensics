#!/usr/bin/env python3
"""Plain Python tests for SNSAug V2 forced-localization oracle-gate diagnostic."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TMP_PARENT = Path("/home/rlatjswo/.codex/memories")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2_fixed_pairs_eval import compute_mask_metrics  # noqa: E402
from cv_forensics.snsaug_v2_forced_localization_oracle_gate import (  # noqa: E402
    MARKER,
    run_snsaug_v2_forced_oracle_gate,
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


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    pair_root = root / "pairs"
    images_dir = pair_root / "images"
    masks_dir = pair_root / "tamper_masks"
    ignore_dir = pair_root / "ignore_masks"
    for path in (images_dir, masks_dir, ignore_dir):
        path.mkdir(parents=True, exist_ok=True)
    bundle = {
        "marker": "PRE_SNS_CURRENT_BEST_MODEL_BUNDLE_OK",
        "long256_checkpoint_path": str(root / "bundle" / "dummy_long256.pt"),
        "tile_v2_checkpoint_path": str(root / "bundle" / "dummy_tile_v2.pt"),
        "policy_gated_report": {
            "mask_threshold": 0.4,
            "min_area_pct": 0.01,
            "max_area_pct": 100.0,
            "suppress_non_tampered_mask": True,
            "fallback_to_baseline_on_v2_unreliable": True,
            "tile_size": 8,
            "tile_stride": 4,
        },
    }
    (root / "bundle").mkdir(parents=True, exist_ok=True)
    (root / "bundle" / "dummy_long256.pt").write_text("fixture", encoding="utf-8")
    (root / "bundle" / "dummy_tile_v2.pt").write_text("fixture", encoding="utf-8")
    bundle_path = root / "bundle" / "best_bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    rows: list[dict[str, object]] = []
    for base_id, label in (("real_1", "real"), ("synthetic_1", "synthetic"), ("tampered_1", "tampered")):
        image = Image.new("RGB", (12, 8), (60, 70, 80))
        ImageDraw.Draw(image).rectangle((3, 2, 7, 5), fill=(220, 40, 40))
        clean_path = images_dir / f"{base_id}__clean.png"
        sns_path = images_dir / f"{base_id}__tiktok_like.png"
        image.save(clean_path)
        image.save(sns_path)
        ignore_clean = ignore_dir / f"{base_id}__clean.png"
        ignore_sns = ignore_dir / f"{base_id}__tiktok_like.png"
        Image.new("L", (12, 8), 0).save(ignore_clean)
        _mask_box((12, 8), (0, 0, 2, 2)).save(ignore_sns)
        gt_path = None
        if label == "tampered":
            gt_path = masks_dir / f"{base_id}.png"
            _mask_box((12, 8), (3, 2, 7, 5)).save(gt_path)
        pred_values = [1 if label == "tampered" and 3 <= x <= 7 and 2 <= y <= 5 else 0 for y in range(8) for x in range(12)]
        class_conf = {"real": 0.92, "synthetic": 0.04, "tampered": 0.04} if label == "tampered" else {label: 0.94, "tampered": 0.03}
        for view, profile, path, ignore in (("clean", "clean", clean_path, ignore_clean), ("sns_aug", "tiktok_like", sns_path, ignore_sns)):
            rows.append(
                {
                    "base_id": base_id,
                    "content_label": label,
                    "view": view,
                    "profile": profile,
                    "seed": 1,
                    "image_path": str(path),
                    "tamper_mask_path": str(gt_path) if gt_path else None,
                    "ignore_mask_path": str(ignore),
                    "fixture_long256_report": {
                        "class": "real" if label == "tampered" else label,
                        "class_conf": class_conf,
                        "family": "Real-or-N/A",
                        "tampered_score": 0.04 if label == "tampered" else 0.03,
                        "baseline_mask": [0] * 96,
                    },
                    "fixture_v2_probability_mask": [float(value) for value in pred_values],
                }
            )
    meta_path = pair_root / "meta.jsonl"
    meta_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")
    (pair_root / "pair_index.json").write_text(json.dumps({"pairs": []}), encoding="utf-8")
    return pair_root, meta_path, bundle_path


def _config(root: Path, pair_root: Path, meta_path: Path, bundle_path: Path) -> dict[str, object]:
    return {
        "best_bundle_path": str(bundle_path),
        "pair_root": str(pair_root),
        "meta_jsonl_path": str(meta_path),
        "approved_input_roots": [str(root / "bundle"), str(pair_root)],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "oracle"),
        "profiles": ["clean", "tiktok_like"],
        "device": "cpu",
        "max_samples": 10,
        "thresholds": [0.0, 0.01, 0.05, 0.10, 0.25, 0.50],
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def test_valid_iou_excludes_ignore_mask() -> None:
    metrics = compute_mask_metrics([1, 0], [1, 1], [1, 0])
    assert_equal(float(metrics["raw_iou"]), 0.5, "raw iou includes ignored pixel")
    assert_equal(float(metrics["valid_iou"]), 0.0, "valid iou excludes ignored pixel")


def test_oracle_gate_diagnostic_outputs() -> None:
    root = temp_root("cvf_oracle_gate_")
    pair_root, meta_path, bundle_path = _write_fixture(root)
    long_checkpoint = root / "bundle" / "dummy_long256.pt"
    tile_checkpoint = root / "bundle" / "dummy_tile_v2.pt"
    long_before = long_checkpoint.read_text(encoding="utf-8")
    tile_before = tile_checkpoint.read_text(encoding="utf-8")
    summary = run_snsaug_v2_forced_oracle_gate(_config(root, pair_root, meta_path, bundle_path))
    assert_equal(summary["marker"], MARKER, "summary marker")
    assert_true(summary["no_training"], "no training")
    assert_true(summary["no_finetune"], "no finetune")
    assert_equal(long_checkpoint.read_text(encoding="utf-8"), long_before, "long checkpoint unchanged")
    assert_equal(tile_checkpoint.read_text(encoding="utf-8"), tile_before, "tile checkpoint unchanged")
    out = root / "reports" / "oracle"
    for name in (
        "oracle_gate_eval_records.jsonl",
        "oracle_gate_eval_comparisons.jsonl",
        "oracle_gate_per_profile_metrics.json",
        "oracle_gate_drop_metrics.json",
        "threshold_sweep_metrics.json",
        "activation_bottleneck_summary.json",
        "forced_localization_worst_samples.json",
        "visual_gallery_manifest.json",
        "artifact_manifest.json",
    ):
        assert_true((out / name).is_file(), f"{name} exists")
    records = [json.loads(line) for line in (out / "oracle_gate_eval_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    normal_t = next(row for row in records if row["mode"] == "normal_gate" and row["base_id"] == "tampered_1" and row["profile"] == "clean")
    oracle_t = next(row for row in records if row["mode"] == "oracle_tampered_gate" and row["base_id"] == "tampered_1" and row["profile"] == "clean")
    oracle_real = next(row for row in records if row["mode"] == "oracle_tampered_gate" and row["base_id"] == "real_1" and row["profile"] == "clean")
    assert_true(not normal_t["localization_activated"], "normal gate unchanged")
    assert_true(oracle_t["localization_activated"], "oracle gate forces tampered localization")
    assert_true(oracle_t["tile_localization_forced"], "oracle forced flag")
    assert_true(not oracle_real["localization_activated"], "oracle keeps non-tampered normal")
    assert_true(Path(oracle_t["pred_mask_path"]).is_file(), "visual pred mask exists")
    assert_true(Path(oracle_t["pred_red_overlay_path"]).is_file(), "visual red overlay exists")
    with Image.open(oracle_t["image_path"]) as image, Image.open(oracle_t["pred_red_overlay_path"]) as overlay:
        assert_equal(overlay.size, image.size, "overlay size matches")
    metrics = json.loads((out / "oracle_gate_per_profile_metrics.json").read_text(encoding="utf-8"))
    assert_equal(metrics["normal_gate"]["clean"]["localization_activation_recall"], 0.0, "normal activation recall")
    assert_equal(metrics["oracle_tampered_gate"]["clean"]["localization_activation_recall"], 1.0, "oracle activation recall")
    sweep = json.loads((out / "threshold_sweep_metrics.json").read_text(encoding="utf-8"))
    assert_true("0.05" in sweep, "threshold sweep key")
    assert_true(sweep["0.05"]["clean"]["localization_activation_recall"] is not None, "sweep activation recall")
    bottleneck = json.loads((out / "activation_bottleneck_summary.json").read_text(encoding="utf-8"))
    assert_true(bottleneck["failure_type"] in {"class_activation_bottleneck", "mask_decoder_geometry_bottleneck", "mixed_or_unclear"}, "failure type")


def main() -> int:
    tests = [
        test_valid_iou_excludes_ignore_mask,
        test_oracle_gate_diagnostic_outputs,
    ]
    for test in tests:
        test()
    print("SNSAUG_V2_FORCED_LOCALIZATION_ORACLE_GATE_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
