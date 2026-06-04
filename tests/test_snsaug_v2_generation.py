#!/usr/bin/env python3
"""Plain Python tests for SNSAug V2 generation and tiny fixed pairs."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TMP_PARENT = Path("/home/rlatjswo/.codex/memories")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2 import SNSAugV2Augmentor, SNSAugV2Config, SNSAugV2PairGenerator  # noqa: E402
from cv_forensics.snsaug_v2.source_manifest_audit import (  # noqa: E402
    audit_source_manifest,
    validate_snsaug_v2_generation_config,
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


def fixture_image_and_mask() -> tuple[Image.Image, Image.Image]:
    image = Image.new("RGB", (120, 80), (50, 60, 70))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 20, 80, 60), fill=(220, 40, 40))
    mask = Image.new("L", (120, 80), 0)
    ImageDraw.Draw(mask).rectangle((40, 20, 80, 60), fill=255)
    return image, mask


def image_bytes(image: Image.Image) -> bytes:
    return image.tobytes()


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
    bad_path = inputs / "missing.png"
    rows = [
        {"base_id": "real_1", "image_path": str(real_path), "content_label": "real", "split": "train", "source_dataset": "fixture"},
        {"base_id": "synth_1", "image_path": str(synth_path), "content_label": "full_synthetic", "split": "validation", "source_dataset": "fixture"},
        {"base_id": "tamp_1", "image_path": str(tamp_path), "tamper_mask_path": str(mask_path), "content_label": "tampered", "split": "train", "source_dataset": "fixture"},
        {"base_id": "bad_1", "image_path": str(bad_path), "content_label": "real", "split": "train", "source_dataset": "fixture"},
    ]
    manifest_path = inputs / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return manifest_path


def test_source_manifest_audit_valid_invalid_records() -> None:
    root = temp_root("cvf_snsaug_audit_")
    manifest_path = fixture_manifest(root)
    audit_root = root / "audit"
    result = audit_source_manifest(manifest_path, output_root=audit_root, fail_fast=False)
    assert_equal(result["summary"]["valid_record_count"], 3, "valid count")
    assert_equal(result["summary"]["invalid_record_count"], 1, "invalid count")
    assert_true((audit_root / "source_manifest_valid_records.jsonl").is_file(), "valid audit file")
    assert_true((audit_root / "source_manifest_invalid_records.jsonl").is_file(), "invalid audit file")


def test_same_seed_reproducibility() -> None:
    image, mask = fixture_image_and_mask()
    config = SNSAugV2Config(profile="combined_sns_realistic", severity="medium", seed=7)
    first = SNSAugV2Augmentor(config)(image, mask, label="tampered", base_id="a")
    second = SNSAugV2Augmentor(config)(image, mask, label="tampered", base_id="a")
    assert_equal(image_bytes(first.image), image_bytes(second.image), "same seed image")
    assert_equal(image_bytes(first.ignore_mask), image_bytes(second.ignore_mask), "same seed ignore mask")


def test_different_seed_can_change_output() -> None:
    image, mask = fixture_image_and_mask()
    first = SNSAugV2Augmentor(SNSAugV2Config(profile="combined_sns_realistic", severity="medium", seed=7))(image, mask, label="tampered", base_id="a")
    second = SNSAugV2Augmentor(SNSAugV2Config(profile="combined_sns_realistic", severity="medium", seed=9))(image, mask, label="tampered", base_id="a")
    assert_true(ImageChops.difference(first.image, second.image).getbbox() is not None, "different seed changes output")


def test_output_sizes_and_clean_profile_ignore_mask() -> None:
    image, mask = fixture_image_and_mask()
    clean = SNSAugV2Augmentor(SNSAugV2Config(profile="clean", severity="medium", seed=1))(image, mask, label="tampered", base_id="g")
    assert_equal(clean.image.size, image.size, "clean image size")
    assert_equal(clean.tamper_mask.size, mask.size, "clean mask size")
    assert_equal(clean.ignore_mask.getbbox(), None, "clean ignore mask empty")


def test_overlay_regions_go_to_ignore_mask_not_tamper_mask() -> None:
    image, mask = fixture_image_and_mask()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="news_meme_overlay", severity="medium", seed=11))(image, mask, label="tampered", base_id="c")
    assert_true(result.ignore_mask.getbbox() is not None, "ignore mask has overlay")
    assert_equal(image_bytes(result.tamper_mask), image_bytes(mask), "overlay does not alter tamper mask")
    assert_true(result.meta["label_preserved"], "label preserved")


def test_geometric_transforms_and_mask_nearest_neighbor() -> None:
    image, mask = fixture_image_and_mask()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="tiktok_like", severity="medium", seed=5))(image, mask, label="tampered", base_id="d")
    assert_equal(result.image.size, result.tamper_mask.size, "mask aligned with transformed image")
    assert_true(all(value in {0, 255} for value in result.tamper_mask.tobytes()), "mask stays binary")


def test_recompression_does_not_alter_mask() -> None:
    image, mask = fixture_image_and_mask()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="jpeg_resize", severity="medium", seed=13))(image, mask, label="tampered", base_id="e")
    assert_true(all(value in {0, 255} for value in result.tamper_mask.tobytes()), "mask stays binary")


def test_tiny_fixed_pair_generator_writes_outputs() -> None:
    root = temp_root("cvf_snsaug_pairs_")
    manifest_path = fixture_manifest(root)
    out_root = root / "out"
    artifact = SNSAugV2PairGenerator(
        source_manifest_path=manifest_path,
        output_root=out_root,
        profiles=["annotation_sticker"],
        severity="medium",
        seed=99,
        max_samples=3,
        max_samples_per_class=1,
    ).run()
    assert_true((out_root / "meta.jsonl").is_file(), "meta jsonl exists")
    assert_true((out_root / "pair_index.json").is_file(), "pair index exists")
    assert_true((out_root / "source_manifest_audit_summary.json").is_file(), "audit summary exists")
    rows = [json.loads(line) for line in (out_root / "meta.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert_true(all(str(row["image_path"]).startswith(str(out_root)) for row in rows), "output paths under output root")
    view_lookup: dict[str, set[str]] = {}
    for row in rows:
        view_lookup.setdefault(row["base_id"], set()).add(row["view"])
    assert_true(all(views == {"clean", "sns_aug"} for views in view_lookup.values()), "clean and sns views share base id")
    assert_true("debug_overlays_dir" in artifact, "artifact includes debug overlays")


def test_validator_guardrails() -> None:
    root = temp_root("cvf_snsaug_validator_")
    manifest_path = fixture_manifest(root)
    valid_config = {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_tiny_fixed_pairs",
        "execution_mode": "approved_local_snsaug_v2_tiny_fixed_pairs",
        "source_manifest_path": str(manifest_path),
        "approved_input_roots": [str(root / "inputs")],
        "approved_output_roots": [str(root / "generated_artifacts")],
        "output_root": str(root / "generated_artifacts" / "run"),
        "profiles": ["clean", "combined_sns_realistic"],
        "severity": "medium",
        "seed": 3,
        "max_samples_per_class": 2,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    assert_equal(validate_snsaug_v2_generation_config(valid_config, require_exists=False), [], "valid config")
    invalid = dict(valid_config)
    invalid["output_root"] = str(REPO_ROOT / "outputs" / "bad_run")
    invalid["no_network"] = False
    errors = validate_snsaug_v2_generation_config(invalid, require_exists=False)
    assert_true(any("output_root must be outside repository" in error for error in errors), "reject repo-local output")
    assert_true(any("no_network must be true" in error for error in errors), "reject network")


def main() -> int:
    tests = [
        test_source_manifest_audit_valid_invalid_records,
        test_same_seed_reproducibility,
        test_different_seed_can_change_output,
        test_output_sizes_and_clean_profile_ignore_mask,
        test_overlay_regions_go_to_ignore_mask_not_tamper_mask,
        test_geometric_transforms_and_mask_nearest_neighbor,
        test_recompression_does_not_alter_mask,
        test_tiny_fixed_pair_generator_writes_outputs,
        test_validator_guardrails,
    ]
    for test in tests:
        test()
    print("SNSAUG_V2_GENERATION_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
