#!/usr/bin/env python3
"""Plain Python tests for SNSAug V2 small benchmark geometry/postprocess evaluation."""

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

from cv_forensics.snsaug_v2 import SNSAugV2Augmentor, SNSAugV2Config, SNSAugV2PairGenerator  # noqa: E402
from cv_forensics.snsaug_v2.source_manifest_audit import validate_snsaug_v2_generation_config  # noqa: E402
from cv_forensics.snsaug_v2_fixed_pairs_eval import run_snsaug_v2_fixed_pairs_eval, validate_snsaug_v2_fixed_pairs_eval_config  # noqa: E402


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def temp_root(prefix: str) -> Path:
    TMP_PARENT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(TMP_PARENT)))


def fixture_image_and_mask() -> tuple[Image.Image, Image.Image]:
    image = Image.new("RGB", (180, 120), (48, 60, 72))
    draw = ImageDraw.Draw(image)
    draw.rectangle((58, 30, 128, 96), fill=(230, 50, 50))
    draw.ellipse((14, 12, 42, 40), fill=(240, 210, 40))
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((58, 30, 128, 96), fill=255)
    return image, mask


def fixture_manifest(root: Path) -> Path:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    image, mask = fixture_image_and_mask()
    real_path = inputs / "real.png"
    synth_path = inputs / "synthetic.png"
    tamp_path = inputs / "tampered.png"
    mask_path = inputs / "tampered_mask.png"
    image.save(real_path)
    image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(synth_path)
    image.save(tamp_path)
    mask.save(mask_path)
    manifest_path = inputs / "manifest.jsonl"
    rows = [
        {"base_id": "real_1", "image_path": str(real_path), "content_label": "real", "split": "validation", "source_dataset": "fixture"},
        {"base_id": "synthetic_1", "image_path": str(synth_path), "content_label": "synthetic", "split": "validation", "source_dataset": "fixture"},
        {"base_id": "tampered_1", "image_path": str(tamp_path), "tamper_mask_path": str(mask_path), "content_label": "tampered", "split": "validation", "source_dataset": "fixture"},
    ]
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return manifest_path


def build_fixture_bundle(root: Path) -> Path:
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
    return bundle_path


def add_fixture_predictions(meta_path: Path) -> None:
    rows = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_rows = []
    for row in rows:
        label = row["content_label"]
        profile = row["profile"]
        if row["view"] == "clean":
            pred = label
            p_t = 0.94 if label == "tampered" else 0.03
            mask_values = [1 if label == "tampered" and 58 <= (i % 180) <= 128 and 30 <= (i // 180) <= 96 else 0 for i in range(180 * 120)]
        else:
            if profile in {"canvas_9x16_only", "tiktok_like", "instagram_story_like", "youtube_shorts_like", "combined_sns_realistic"} and label == "tampered":
                pred = "real"
                p_t = 0.22
                mask_values = [0] * (Image.open(row["image_path"]).size[0] * Image.open(row["image_path"]).size[1])
            else:
                pred = label
                p_t = 0.82 if label == "tampered" else 0.04
                size = Image.open(row["image_path"]).size
                mask_values = [0] * (size[0] * size[1])
                if label == "tampered":
                    mask_values = [1 if 0.25 * size[0] <= (i % size[0]) <= 0.7 * size[0] and 0.25 * size[1] <= (i // size[0]) <= 0.8 * size[1] else 0 for i in range(size[0] * size[1])]
        row["fixture_long256_report"] = {
            "class": pred,
            "class_conf": {
                "real": 0.9 if pred == "real" else 0.04,
                "synthetic": 0.9 if pred == "synthetic" else 0.04,
                "tampered": p_t,
            },
            "family": "Other",
            "tampered_score": p_t,
            "baseline_mask": mask_values,
        }
        row["fixture_v2_probability_mask"] = [float(value) for value in mask_values]
        out_rows.append(row)
    meta_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in out_rows) + "\n", encoding="utf-8")


def test_generation_profiles_and_mask_policy() -> None:
    image, mask = fixture_image_and_mask()
    resize_crop = SNSAugV2Augmentor(SNSAugV2Config(profile="resize_crop_pad", severity="medium", seed=3407))(image, mask, label="tampered", base_id="a")
    assert_equal(resize_crop.image.size, resize_crop.tamper_mask.size, "resize_crop_pad mask sync")
    zoom = SNSAugV2Augmentor(SNSAugV2Config(profile="zoom_crop", severity="medium", seed=3407))(image, mask, label="tampered", base_id="a")
    assert_equal(zoom.image.size, zoom.tamper_mask.size, "zoom_crop mask sync")
    recomp = SNSAugV2Augmentor(SNSAugV2Config(profile="recompression_light", severity="medium", seed=3407))(image, mask, label="tampered", base_id="a")
    assert_equal(recomp.tamper_mask.tobytes(), mask.tobytes(), "recompression does not alter mask")
    shot = SNSAugV2Augmentor(SNSAugV2Config(profile="screenshot_recapture_light", severity="medium", seed=3407))(image, mask, label="tampered", base_id="a")
    assert_true(shot.ignore_mask.getbbox() is not None, "screenshot ui goes to ignore mask")
    assert_equal(shot.image.size, shot.tamper_mask.size, "screenshot mask sync")


def test_no_duplicate_clean_rows_and_validators() -> None:
    root = temp_root("cvf_small_benchmark_pairs_")
    manifest_path = fixture_manifest(root)
    out_root = root / "pairs"
    profiles = ["clean", "resize_crop_pad", "zoom_crop", "recompression_light", "resize_jpeg", "screenshot_recapture_light", "canvas_9x16_only", "platform_ui_same_size", "news_meme_overlay", "tiktok_like", "instagram_story_like", "youtube_shorts_like", "combined_sns_realistic"]
    SNSAugV2PairGenerator(
        source_manifest_path=manifest_path,
        output_root=out_root,
        profiles=profiles,
        severity="medium",
        seed=3407,
        max_samples=6,
        max_samples_per_class=1,
    ).run()
    rows = [json.loads(line) for line in (out_root / "meta.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    clean_rows = [row for row in rows if row["view"] == "clean"]
    assert_equal(len(clean_rows), 3, "one clean row per base id")
    assert_true(all(row["profile"] != "clean" for row in rows if row["view"] == "sns_aug"), "no duplicate clean sns rows")
    gen_cfg = {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_tiny_fixed_pairs",
        "execution_mode": "approved_local_snsaug_v2_tiny_fixed_pairs",
        "source_manifest_path": str(manifest_path),
        "approved_input_roots": [str(root / "inputs")],
        "approved_output_roots": [str(root / "runs")],
        "output_root": str(root / "runs" / "pairs"),
        "profiles": profiles,
        "severity": "medium",
        "seed": 3407,
        "max_samples_per_class": 1,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    assert_equal(validate_snsaug_v2_generation_config(gen_cfg, require_exists=False), [], "generation validator accepts benchmark profiles")


def test_report_generation_and_eval_support() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_small_benchmark_eval_")
    manifest_path = fixture_manifest(root)
    pair_root = root / "pairs"
    profiles = ["clean", "resize_crop_pad", "zoom_crop", "recompression_light", "resize_jpeg", "screenshot_recapture_light", "canvas_9x16_only", "platform_ui_same_size", "news_meme_overlay", "tiktok_like", "instagram_story_like", "youtube_shorts_like", "combined_sns_realistic"]
    SNSAugV2PairGenerator(
        source_manifest_path=manifest_path,
        output_root=pair_root,
        profiles=profiles,
        severity="medium",
        seed=3407,
        max_samples=6,
        max_samples_per_class=1,
    ).run()
    add_fixture_predictions(pair_root / "meta.jsonl")
    bundle_path = build_fixture_bundle(root)
    eval_cfg = {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_fixed_pairs_eval",
        "execution_mode": "approved_local_snsaug_v2_fixed_pairs_eval",
        "user_approval_text": "I_APPROVE_SNSAUG_V2_FIXED_PAIRS_EVAL",
        "best_bundle_path": str(bundle_path),
        "pair_root": str(pair_root),
        "meta_jsonl_path": str(pair_root / "meta.jsonl"),
        "approved_input_roots": [str(root / "bundle"), str(pair_root)],
        "approved_output_roots": [str(root / "eval_outputs")],
        "output_root": str(root / "eval_outputs" / "run"),
        "profiles": profiles,
        "device": "cpu",
        "max_samples": 50,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    assert_equal(validate_snsaug_v2_fixed_pairs_eval_config(eval_cfg, require_exists=False), [], "eval validator accepts benchmark profiles")
    summary = run_snsaug_v2_fixed_pairs_eval(eval_cfg)
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "no repo writes")
    assert_true(summary["record_count"] > 0, "records generated")
    out = root / "eval_outputs" / "run"
    for name in (
        "small_benchmark_summary.md",
        "small_benchmark_metrics_table.csv",
        "small_benchmark_drop_table.csv",
        "small_benchmark_interpretation.json",
        "artifact_manifest.json",
    ):
        assert_true((out / name).is_file(), f"{name} exists")
    interpretation = json.loads((out / "small_benchmark_interpretation.json").read_text(encoding="utf-8"))
    assert_true("main_collapse_cause" in interpretation, "interpretation has cause")


def main() -> int:
    tests = [
        test_generation_profiles_and_mask_policy,
        test_no_duplicate_clean_rows_and_validators,
        test_report_generation_and_eval_support,
    ]
    for test in tests:
        test()
    print("SNSAUG_V2_SMALL_BENCHMARK_EVAL_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
