#!/usr/bin/env python3
"""Plain Python tests for SNSAug V2."""

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


def test_same_seed_reproducibility() -> None:
    image, mask = fixture_image_and_mask()
    config = SNSAugV2Config(profile="combined_sns_realistic", severity="medium", seed=7)
    first = SNSAugV2Augmentor(config)(image, mask, label="tampered", base_id="a")
    second = SNSAugV2Augmentor(config)(image, mask, label="tampered", base_id="a")
    assert_equal(list(first.image.getdata()), list(second.image.getdata()), "same seed image")
    assert_equal(list(first.ignore_mask.getdata()), list(second.ignore_mask.getdata()), "same seed ignore mask")


def test_different_seed_can_change_output() -> None:
    image, mask = fixture_image_and_mask()
    first = SNSAugV2Augmentor(SNSAugV2Config(profile="combined_sns_realistic", severity="medium", seed=7))(image, mask, label="tampered", base_id="a")
    second = SNSAugV2Augmentor(SNSAugV2Config(profile="combined_sns_realistic", severity="medium", seed=9))(image, mask, label="tampered", base_id="a")
    assert_true(ImageChops.difference(first.image, second.image).getbbox() is not None, "different seed changes output")


def test_output_sizes_and_label_preservation() -> None:
    image, mask = fixture_image_and_mask()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="instagram_story_like", severity="medium", seed=3))(image, mask, label="tampered", base_id="b")
    assert_equal(result.image.size, result.ignore_mask.size, "image ignore same size")
    assert_equal(result.image.size, result.tamper_mask.size, "image mask same size")
    assert_true(result.meta["label_preserved"], "label preserved")


def test_overlay_regions_go_to_ignore_mask_not_tamper_mask() -> None:
    image, mask = fixture_image_and_mask()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="ai_badge_overlay", severity="medium", seed=11))(image, mask, label="tampered", base_id="c")
    assert_true(max(result.ignore_mask.getdata()) > 0, "ignore mask has overlay")
    assert_equal(list(result.tamper_mask.getdata()), list(mask.getdata()), "overlay does not alter tamper mask")


def test_geometric_transforms_apply_to_image_and_mask() -> None:
    image, mask = fixture_image_and_mask()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="tiktok_like", severity="medium", seed=5))(image, mask, label="tampered", base_id="d")
    bbox = result.tamper_mask.getbbox()
    assert_true(bbox is not None, "tamper bbox exists")
    assert_true(bbox[2] <= result.image.size[0] and bbox[3] <= result.image.size[1], "mask aligned with image canvas")


def test_recompression_does_not_alter_mask() -> None:
    image, mask = fixture_image_and_mask()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="jpeg_resize", severity="medium", seed=13))(image, mask, label="tampered", base_id="e")
    assert_true(set(result.tamper_mask.getdata()).issubset({0, 255}), "mask stays binary")


def test_mask_resizing_uses_nearest_neighbor() -> None:
    image = Image.new("RGB", (7, 5), (0, 0, 0))
    mask = Image.new("L", (7, 5), 0)
    ImageDraw.Draw(mask).rectangle((2, 1, 4, 3), fill=255)
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="jpeg_resize", severity="strong", output_size=20, seed=17))(image, mask, label="tampered", base_id="f")
    assert_true(set(result.tamper_mask.getdata()).issubset({0, 255}), "nearest neighbor mask stays binary")


def test_clean_profile_preserves_shape_and_zero_ignore_mask() -> None:
    image, mask = fixture_image_and_mask()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="clean", severity="medium", seed=1))(image, mask, label="tampered", base_id="g")
    assert_equal(result.image.size, image.size, "clean image size")
    assert_equal(result.tamper_mask.size, mask.size, "clean mask size")
    assert_true(max(result.ignore_mask.getdata()) == 0, "clean ignore mask empty")


def test_pair_generator_writes_meta_and_shared_base_id() -> None:
    root = temp_root("cvf_snsaug_pairs_")
    image, mask = fixture_image_and_mask()
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    image_path = inputs / "sample.png"
    mask_path = inputs / "sample_mask.png"
    image.save(image_path)
    mask.save(mask_path)
    manifest_path = inputs / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": [{"sample_id": "sample1", "image_path": str(image_path), "mask_path": str(mask_path), "content_label": "tampered", "source_dataset": "fixture"}]}), encoding="utf-8")
    out_root = root / "out"
    artifact = SNSAugV2PairGenerator(
        input_manifest=manifest_path,
        output_root=out_root,
        profiles=["clean", "annotation_sticker"],
        severity="medium",
        seed=99,
        split="train",
        max_samples=1,
    ).run()
    assert_true((out_root / "meta.jsonl").is_file(), "meta jsonl exists")
    rows = [json.loads(line) for line in (out_root / "meta.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    base_ids = {row["base_id"] for row in rows}
    assert_equal(base_ids, {"sample1"}, "same base id")
    assert_true("images_dir" in artifact, "artifact manifest content")


def main() -> int:
    tests = [
        test_same_seed_reproducibility,
        test_different_seed_can_change_output,
        test_output_sizes_and_label_preservation,
        test_overlay_regions_go_to_ignore_mask_not_tamper_mask,
        test_geometric_transforms_apply_to_image_and_mask,
        test_recompression_does_not_alter_mask,
        test_mask_resizing_uses_nearest_neighbor,
        test_clean_profile_preserves_shape_and_zero_ignore_mask,
        test_pair_generator_writes_meta_and_shared_base_id,
    ]
    for test in tests:
        test()
    print("SNSAUG_V2_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
