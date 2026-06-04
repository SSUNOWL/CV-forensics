#!/usr/bin/env python3
"""Plain Python tests for SNSAug V2 layout-cause ablation profiles."""

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

from cv_forensics.snsaug_v2 import SNSAugV2Augmentor, SNSAugV2Config, SNSAugV2PairGenerator  # noqa: E402
from cv_forensics.snsaug_v2.source_manifest_audit import validate_snsaug_v2_generation_config  # noqa: E402
from cv_forensics.snsaug_v2_fixed_pairs_eval import aggregate_per_profile, validate_snsaug_v2_fixed_pairs_eval_config  # noqa: E402


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
    image = Image.new("RGB", (160, 96), (42, 53, 64))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 18, 112, 74), fill=(230, 60, 60))
    draw.ellipse((10, 12, 34, 36), fill=(240, 220, 40))
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((50, 18, 112, 74), fill=255)
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


def test_ablation_profiles_generate_outputs() -> None:
    image, mask = fixture_image_and_mask()
    profiles = [
        "canvas_9x16_only",
        "canvas_9x16_full_content",
        "platform_ui_same_size",
        "tiktok_like_no_actionbar",
        "instagram_story_no_text_sticker",
        "youtube_shorts_no_actionbar",
    ]
    for profile in profiles:
        result = SNSAugV2Augmentor(SNSAugV2Config(profile=profile, severity="medium", seed=3407))(image, mask, label="tampered", base_id="sample")
        assert_equal(result.image.size, result.ignore_mask.size, f"{profile} output size")
        assert_true(result.meta["label_preserved"], f"{profile} preserves label")


def test_canvas_only_profiles_have_no_ui_or_degradation() -> None:
    image, mask = fixture_image_and_mask()
    for profile in ("canvas_9x16_only", "canvas_9x16_full_content"):
        result = SNSAugV2Augmentor(SNSAugV2Config(profile=profile, severity="medium", seed=3407))(image, mask, label="tampered", base_id="sample")
        assert_true(result.image.size == (540, 960), f"{profile} uses portrait canvas")
        assert_equal(result.ignore_mask.getbbox(), None, f"{profile} ignore mask empty")
        assert_true(not result.meta["ui_applied"], f"{profile} no ui")
        assert_true(not result.meta["text_sticker_applied"], f"{profile} no text")
        assert_equal(result.meta["postprocess_applied"], [], f"{profile} no degradation")
        assert_true(not result.meta["degradation_applied"], f"{profile} no degradation flag")


def test_platform_ui_same_size_preserves_size_and_aspect() -> None:
    image, mask = fixture_image_and_mask()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="platform_ui_same_size", severity="medium", seed=3407))(image, mask, label="tampered", base_id="sample")
    assert_equal(result.image.size, image.size, "same-size ui keeps image size")
    assert_true(result.ignore_mask.getbbox() is not None, "same-size ui adds ignore mask")
    assert_true(result.meta["ui_applied"], "same-size ui applied")
    assert_true(not result.meta["portrait_canvas_applied"], "same-size ui avoids portrait expansion")


def test_fixed_pair_generator_and_eval_accept_ablation_profiles() -> None:
    root = temp_root("cvf_snsaug_ablation_pairs_")
    manifest_path = fixture_manifest(root)
    out_root = root / "pairs"
    SNSAugV2PairGenerator(
        source_manifest_path=manifest_path,
        output_root=out_root,
        profiles=["canvas_9x16_only", "platform_ui_same_size", "tiktok_like_no_actionbar"],
        severity="medium",
        seed=3407,
        max_samples=3,
        max_samples_per_class=1,
    ).run()
    rows = [json.loads(line) for line in (out_root / "meta.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    profiles = sorted({row["profile"] for row in rows if row["view"] == "sns_aug"})
    assert_equal(profiles, ["canvas_9x16_only", "platform_ui_same_size", "tiktok_like_no_actionbar"], "pair generator ablation profiles")
    eval_cfg = {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_fixed_pairs_eval",
        "execution_mode": "approved_local_snsaug_v2_fixed_pairs_eval",
        "user_approval_text": "I_APPROVE_SNSAUG_V2_FIXED_PAIRS_EVAL",
        "best_bundle_path": str(root / "bundle" / "fixture.json"),
        "pair_root": str(out_root),
        "meta_jsonl_path": str(out_root / "meta.jsonl"),
        "approved_input_roots": [str(root / "bundle"), str(out_root)],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "eval"),
        "profiles": ["clean", "canvas_9x16_only", "platform_ui_same_size", "tiktok_like_no_actionbar"],
        "device": "cpu",
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    (root / "bundle").mkdir(parents=True, exist_ok=True)
    (root / "bundle" / "fixture.json").write_text("{}", encoding="utf-8")
    assert_equal(validate_snsaug_v2_fixed_pairs_eval_config(eval_cfg, require_exists=False), [], "eval config accepts ablation profiles")


def test_per_profile_aggregation_accepts_ablation_profiles() -> None:
    records = [
        {"base_id": "a", "content_label": "tampered", "view": "clean", "profile": "clean", "pred_class": "tampered", "class_correct": True, "valid_iou": 0.8, "valid_dice": 0.9, "raw_iou": 0.8, "localization_activated": True, "final_mask_area_pct": 12.0, "latency_ms": 12.0},
        {"base_id": "a", "content_label": "tampered", "view": "sns_aug", "profile": "canvas_9x16_only", "pred_class": "real", "class_correct": False, "valid_iou": 0.1, "valid_dice": 0.2, "raw_iou": 0.1, "localization_activated": False, "final_mask_area_pct": 0.0, "latency_ms": 13.0},
        {"base_id": "b", "content_label": "real", "view": "clean", "profile": "clean", "pred_class": "real", "class_correct": True, "final_mask_area_pct": 0.0, "latency_ms": 12.0},
        {"base_id": "b", "content_label": "real", "view": "sns_aug", "profile": "platform_ui_same_size", "pred_class": "real", "class_correct": True, "final_mask_area_pct": 0.0, "latency_ms": 13.0},
    ]
    metrics = aggregate_per_profile(records)
    assert_true("canvas_9x16_only" in metrics, "canvas ablation metric present")
    assert_true("platform_ui_same_size" in metrics, "same-size ui metric present")


def test_generation_validator_accepts_ablation_profiles() -> None:
    root = temp_root("cvf_snsaug_ablation_validator_")
    manifest_path = fixture_manifest(root)
    config = {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_tiny_fixed_pairs",
        "execution_mode": "approved_local_snsaug_v2_tiny_fixed_pairs",
        "source_manifest_path": str(manifest_path),
        "approved_input_roots": [str(root / "inputs")],
        "approved_output_roots": [str(root / "generated")],
        "output_root": str(root / "generated" / "pairs"),
        "profiles": ["canvas_9x16_only", "platform_ui_same_size", "instagram_story_no_text_sticker"],
        "severity": "medium",
        "seed": 3407,
        "max_samples_per_class": 1,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    assert_equal(validate_snsaug_v2_generation_config(config, require_exists=False), [], "generation config accepts ablation profiles")


def main() -> int:
    tests = [
        test_ablation_profiles_generate_outputs,
        test_canvas_only_profiles_have_no_ui_or_degradation,
        test_platform_ui_same_size_preserves_size_and_aspect,
        test_fixed_pair_generator_and_eval_accept_ablation_profiles,
        test_per_profile_aggregation_accepts_ablation_profiles,
        test_generation_validator_accepts_ablation_profiles,
    ]
    for test in tests:
        test()
    print("SNSAUG_V2_LAYOUT_CAUSE_ABLATION_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
