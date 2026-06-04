#!/usr/bin/env python3
"""Tests for SNSAug V2 overlay placement collision guard."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2 import SNSAugV2Augmentor, SNSAugV2Config, PlacementManager  # noqa: E402


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


def _intersects(a: list[int], b: list[int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _image_hash(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def test_text_block_and_badge_do_not_overlap() -> None:
    image, mask = fixture()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="youtube_shorts_like", severity="medium", seed=5))(image, mask, base_id="yt")
    text_boxes = [item["box"] for item in result.meta["overlay_boxes"] if item.get("kind") == "text_block" and item.get("accepted", True)]
    badge_boxes = [item["box"] for item in result.meta["overlay_boxes"] if item.get("kind") == "chip" and item.get("element_id") == "youtube_ai_badge" and item.get("accepted", True)]
    if text_boxes and badge_boxes:
        assert_true(not _intersects(text_boxes[0], badge_boxes[0]), "text and badge should not overlap")


def test_sticker_and_rectangle_overlap_below_threshold() -> None:
    image, mask = fixture()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="news_meme_overlay", severity="medium", seed=14))(image, mask, base_id="news")
    stickers = [item for item in result.meta["overlay_boxes"] if item.get("element_type") == "sticker" and item.get("accepted", True)]
    for item in stickers:
        assert_true(float(item.get("max_overlap_ratio", 0.0)) <= 0.05, "sticker overlap should stay low")


def test_same_seed_reproducible_metadata_and_image() -> None:
    image, mask = fixture()
    config = SNSAugV2Config(profile="combined_sns_realistic", severity="medium", seed=3407)
    first = SNSAugV2Augmentor(config)(image, mask, base_id="same")
    second = SNSAugV2Augmentor(config)(image, mask, base_id="same")
    assert_equal(first.meta["overlay_boxes"], second.meta["overlay_boxes"], "metadata reproducible")
    assert_equal(_image_hash(first.image), _image_hash(second.image), "image hash reproducible")


def test_different_seed_can_change_metadata() -> None:
    image, mask = fixture()
    first = SNSAugV2Augmentor(SNSAugV2Config(profile="combined_sns_realistic", severity="medium", seed=3407))(image, mask, base_id="diff")
    second = SNSAugV2Augmentor(SNSAugV2Config(profile="combined_sns_realistic", severity="medium", seed=9999))(image, mask, base_id="diff")
    assert_true(first.meta["overlay_boxes"] != second.meta["overlay_boxes"], "different seed can change placement")


def test_optional_overlay_skip_metadata() -> None:
    image, mask = fixture()
    manager = PlacementManager(canvas_size=image.size, config=SNSAugV2Config(profile="combined_sns_realistic", max_placement_attempts=1, placement_margin_px=20))
    full = Image.new("L", image.size, 255)
    manager.register_existing(element_id="fixed_full", element_type="fixed_ui_block", bbox=(0, 0, image.size[0], image.size[1]), alpha_mask=full, metadata={})
    decision = manager.place_variable(
        element_id="blocked_text",
        element_type="text_block",
        candidate_regions=[(0, 0, image.size[0], image.size[1])],
        base_size=(100, 40),
        rng=__import__("random").Random(1),
        render_preview=lambda rect: (Image.new("L", image.size, 255), {}),
        optional=True,
    )
    assert_true(not decision.accepted, "blocked overlay should skip")
    assert_equal(decision.rejected_reason, "overlap_threshold_exceeded", "skip reason")


def test_ignore_mask_and_tamper_mask_policy_still_hold() -> None:
    image, mask = fixture()
    result = SNSAugV2Augmentor(SNSAugV2Config(profile="instagram_story_like", severity="medium", seed=7))(image, mask, base_id="mask")
    assert_true(result.ignore_mask.getbbox() is not None, "ignore mask exists")
    assert_true(all(value in {0, 255} for value in result.tamper_mask.tobytes()), "tamper mask binary")


def test_preview_and_tiny_pair_scripts_still_run() -> None:
    preview = subprocess.run([sys.executable, "scripts/snsaug_v2/preview_snsaug_v2_overlays.py", "--help"], cwd=str(REPO_ROOT), capture_output=True, text=True)
    tiny = subprocess.run([sys.executable, "scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py", "--help"], cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert_equal(preview.returncode, 0, "preview help")
    assert_equal(tiny.returncode, 0, "tiny pair help")


def main() -> int:
    tests = [
        test_text_block_and_badge_do_not_overlap,
        test_sticker_and_rectangle_overlap_below_threshold,
        test_same_seed_reproducible_metadata_and_image,
        test_different_seed_can_change_metadata,
        test_optional_overlay_skip_metadata,
        test_ignore_mask_and_tamper_mask_policy_still_hold,
        test_preview_and_tiny_pair_scripts_still_run,
    ]
    for test in tests:
        test()
    print("SNSAUG_V2_OVERLAY_COLLISION_GUARD_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
