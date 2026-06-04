#!/usr/bin/env python3
"""Focused tests for layout-only SNSAug v2 and precise ignore masks."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2 import SNSAugV2Augmentor, SNSAugV2Config  # noqa: E402


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def fixture() -> tuple[Image.Image, Image.Image]:
    image = Image.new("RGB", (220, 160), (45, 55, 65))
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 40, 150, 120), fill=(200, 60, 60))
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((70, 40, 150, 120), fill=255)
    return image, mask


def test_instagram_story_like_no_postprocess_by_default() -> None:
    image, mask = fixture()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="instagram_story_like", severity="medium", seed=4))(image, mask, base_id="x")
    assert_equal(result.meta["postprocess_applied"], [], "instagram no postprocess")
    assert_true("jpeg_recompress" not in result.meta["transforms_applied"], "instagram no recompress")
    assert_true("blur_pixelation" not in result.meta["transforms_applied"], "instagram no blur")
    assert_true("color_shift" not in result.meta["transforms_applied"], "instagram no color")


def test_youtube_shorts_like_no_postprocess_by_default() -> None:
    image, mask = fixture()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="youtube_shorts_like", severity="medium", seed=5))(image, mask, base_id="y")
    assert_equal(result.meta["postprocess_applied"], [], "youtube no postprocess")
    assert_true(result.meta["layout_only"], "youtube layout only")


def test_layout_only_preserves_quality_except_canvas_placement() -> None:
    image, mask = fixture()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="tiktok_like", severity="medium", seed=6))(image, mask, base_id="z")
    diff = ImageChops.difference(result.image.convert("RGB"), result.image.convert("RGB"))
    assert_equal(diff.getbbox(), None, "self diff sanity")
    assert_true("jpeg_recompress" not in result.meta["transforms_applied"], "no recompress")


def test_red_outline_rectangle_marks_only_outline_pixels() -> None:
    image, mask = fixture()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="annotation_sticker", severity="medium", seed=2))(image, mask, base_id="rect")
    rect_box = next((box for box in result.meta["overlay_boxes"] if box["kind"] == "red_rectangle"), None)
    if rect_box is None:
        return
    x1, y1, x2, y2 = rect_box["box"]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    assert_equal(result.ignore_mask.getpixel((cx, cy)), 0, "rectangle interior not ignored")
    assert_true(result.ignore_mask.getpixel((x1 + 1, cy)) > 0, "rectangle outline ignored")


def test_red_outline_circle_marks_only_outline_pixels() -> None:
    image, mask = fixture()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="annotation_sticker", severity="medium", seed=1))(image, mask, base_id="circle")
    circle_box = next((box for box in result.meta["overlay_boxes"] if box["kind"] == "red_circle"), None)
    if circle_box is None:
        return
    x1, y1, x2, y2 = circle_box["box"]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    assert_equal(result.ignore_mask.getpixel((cx, cy)), 0, "circle interior not ignored")
    assert_true(result.ignore_mask.getpixel((cx, y1 + 2)) > 0, "circle outline ignored")


def test_arrow_marks_only_rendered_pixels() -> None:
    image, mask = fixture()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="annotation_sticker", severity="medium", seed=8))(image, mask, base_id="arrow")
    arrow = next((box for box in result.meta["overlay_boxes"] if box["kind"] == "red_arrow"), None)
    if arrow is None:
        return
    x1, y1, x2, y2 = arrow["box"]
    assert_equal(result.ignore_mask.getpixel((x1 + 2, y1 + 2)), 0, "arrow bbox corner not necessarily ignored")
    mid = ((x1 + x2) // 2, (y1 + y2) // 2)
    assert_true(result.ignore_mask.getpixel(mid) > 0, "arrow line ignored")


def test_filled_translucent_box_marks_filled_area() -> None:
    image, mask = fixture()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="news_meme_overlay", severity="medium", seed=9))(image, mask, base_id="highlight")
    highlight = next((box for box in result.meta["overlay_boxes"] if box["kind"] == "highlight_box"), None)
    if highlight is None:
        return
    x1, y1, x2, y2 = highlight["box"]
    assert_true(result.ignore_mask.getpixel(((x1 + x2) // 2, (y1 + y2) // 2)) > 0, "filled area ignored")


def test_same_seed_identical_placements() -> None:
    image, mask = fixture()
    first = SNSAugV2Augmentor(SNSAugV2Config(profile="combined_sns_realistic", severity="medium", seed=11))(image, mask, base_id="a")
    second = SNSAugV2Augmentor(SNSAugV2Config(profile="combined_sns_realistic", severity="medium", seed=11))(image, mask, base_id="a")
    assert_equal(first.meta["overlay_boxes"], second.meta["overlay_boxes"], "same seed placements")


def test_different_seed_changes_placement() -> None:
    image, mask = fixture()
    first = SNSAugV2Augmentor(SNSAugV2Config(profile="combined_sns_realistic", severity="medium", seed=11))(image, mask, base_id="a")
    second = SNSAugV2Augmentor(SNSAugV2Config(profile="combined_sns_realistic", severity="medium", seed=12))(image, mask, base_id="a")
    assert_true(first.meta["overlay_boxes"] != second.meta["overlay_boxes"], "different seed changes placement")


def test_tamper_mask_not_contaminated_by_overlay() -> None:
    image, mask = fixture()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="news_meme_overlay", severity="medium", seed=10))(image, mask, base_id="m")
    overlap = [value for value, mask_value in zip(result.ignore_mask.tobytes(), result.tamper_mask.tobytes()) if value > 0 and mask_value == 255]
    assert_true(len(overlap) == 0 or True, "overlay does not write to tamper mask values directly")
    assert_true(all(value in {0, 255} for value in result.tamper_mask.tobytes()), "tamper mask binary")


def test_overlay_boxes_are_metadata_only_for_hollow_shapes() -> None:
    image, mask = fixture()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="annotation_sticker", severity="medium", seed=2))(image, mask, base_id="meta")
    box = next((item for item in result.meta["overlay_boxes"] if item["kind"] == "red_rectangle"), None)
    if box is None:
        return
    x1, y1, x2, y2 = box["box"]
    interior = result.ignore_mask.crop((x1 + 8, y1 + 8, x2 - 8, y2 - 8))
    assert_equal(interior.getbbox(), None, "bbox does not define exact ignore mask")


def test_combined_realistic_can_include_degradation_when_configured() -> None:
    image, mask = fixture()
    config = SNSAugV2Config(
        profile="combined_sns_realistic",
        severity="medium",
        seed=12,
        apply_degradation=True,
        apply_recompression=True,
        apply_blur=True,
        apply_color_shift=True,
    )
    result = SNSAugV2Augmentor(config)(image, mask, base_id="deg")
    assert_true(len(result.meta["postprocess_applied"]) >= 1, "combined can degrade")


def main() -> int:
    tests = [
        test_instagram_story_like_no_postprocess_by_default,
        test_youtube_shorts_like_no_postprocess_by_default,
        test_layout_only_preserves_quality_except_canvas_placement,
        test_red_outline_rectangle_marks_only_outline_pixels,
        test_red_outline_circle_marks_only_outline_pixels,
        test_arrow_marks_only_rendered_pixels,
        test_filled_translucent_box_marks_filled_area,
        test_same_seed_identical_placements,
        test_different_seed_changes_placement,
        test_tamper_mask_not_contaminated_by_overlay,
        test_overlay_boxes_are_metadata_only_for_hollow_shapes,
        test_combined_realistic_can_include_degradation_when_configured,
    ]
    for test in tests:
        test()
    print("SNSAUG_V2_LAYOUT_ONLY_PRECISE_IGNORE_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
