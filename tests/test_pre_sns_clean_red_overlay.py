#!/usr/bin/env python3
"""Standalone tests for clean pre-SNS red mask overlays."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_visualization import (  # noqa: E402
    CLEAN_OVERLAY_MARKER,
    VisualArtifactError,
    focused_mask_from_values,
    resolve_artifact_path,
    write_clean_red_overlay,
)

TEMP_ROOT = Path.home() / ".codex" / "memories"


def temporary_root():
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=str(TEMP_ROOT))


def make_image(path: Path) -> None:
    from PIL import Image

    pixels = []
    for y in range(4):
        for x in range(4):
            pixels.append((40 + x * 10, 70 + y * 10, 100 + x + y))
    image = Image.new("RGB", (4, 4))
    image.putdata(pixels)
    image.save(path)


def load_pixels(path: str | Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").size, list(image.convert("RGB").getdata())


def load_mask(path: str | Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("L").size, list(image.convert("L").getdata())


def test_clean_overlay_writes_expected_artifacts() -> None:
    with temporary_root() as tmp:
        root = Path(tmp)
        image_path = root / "image.png"
        make_image(image_path)
        result = write_clean_red_overlay(image_path, [0.0, 0.1, 0.2, 1.0], root / "report", mask_size=2)
        for key in ("clean_focused_mask_path", "clean_red_overlay_path", "clean_red_overlay_manifest_path"):
            assert os.path.isfile(result[key]), key
            assert str(result[key]).startswith(str(root / "report"))
        assert result["clean_red_overlay_written"] is True


def test_pixels_outside_mask_unchanged_and_inside_red_blended() -> None:
    with temporary_root() as tmp:
        root = Path(tmp)
        image_path = root / "image.png"
        make_image(image_path)
        result = write_clean_red_overlay(
            image_path,
            [0.0, 0.0, 0.0, 1.0],
            root / "report",
            mask_size=2,
            keep_ratio=0.25,
            component_mode="largest_component",
            alpha=0.5,
        )
        _size, original = load_pixels(image_path)
        _overlay_size, overlay = load_pixels(result["clean_red_overlay_path"])
        _mask_size, mask = load_mask(result["clean_focused_mask_path"])
        changed_inside = 0
        for index, mask_value in enumerate(mask):
            if mask_value <= 0:
                assert overlay[index] == original[index]
            else:
                assert overlay[index][0] > original[index][0]
                changed_inside += 1
        assert changed_inside > 0


def test_no_box_like_changes_outside_mask() -> None:
    with temporary_root() as tmp:
        root = Path(tmp)
        image_path = root / "image.png"
        make_image(image_path)
        result = write_clean_red_overlay(image_path, [0.0, 0.0, 0.0, 1.0], root / "report", mask_size=2)
        _size, original = load_pixels(image_path)
        _overlay_size, overlay = load_pixels(result["clean_red_overlay_path"])
        _mask_size, mask = load_mask(result["clean_focused_mask_path"])
        outside_changes = [index for index, value in enumerate(mask) if value <= 0 and overlay[index] != original[index]]
        assert outside_changes == []


def test_top_percentile_thresholding_is_deterministic() -> None:
    mask = [0.1, 0.3, 0.2, 0.9, 0.8, 0.0, 0.0, 0.0, 0.0]
    first, size = focused_mask_from_values(mask, mask_size=(3, 3), keep_ratio=0.25, component_mode="all_components")
    second, second_size = focused_mask_from_values(mask, mask_size=(3, 3), keep_ratio=0.25, component_mode="all_components")
    assert size == (3, 3)
    assert second_size == (3, 3)
    assert first == second
    assert sum(first) == 3


def test_largest_component_selection_is_deterministic() -> None:
    mask = [
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.9,
        0.0,
        0.0,
        0.8,
        0.0,
        0.0,
        0.0,
        0.7,
    ]
    focused, _size = focused_mask_from_values(mask, mask_size=(4, 4), keep_ratio=0.5, component_mode="largest_component")
    assert focused == [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]


def test_zero_mask_safe() -> None:
    with temporary_root() as tmp:
        root = Path(tmp)
        image_path = root / "image.png"
        make_image(image_path)
        result = write_clean_red_overlay(image_path, [0.0, 0.0, 0.0, 0.0], root / "report", mask_size=2)
        assert result["clean_overlay_area_pct"] == 0.0
        _size, original = load_pixels(image_path)
        _overlay_size, overlay = load_pixels(result["clean_red_overlay_path"])
        assert overlay == original


def test_invalid_alpha_rejected() -> None:
    with temporary_root() as tmp:
        root = Path(tmp)
        image_path = root / "image.png"
        make_image(image_path)
        try:
            write_clean_red_overlay(image_path, [0.0, 1.0, 0.0, 1.0], root / "report", mask_size=2, alpha=1.5)
        except VisualArtifactError:
            return
        raise AssertionError("expected invalid alpha rejection")


def test_path_traversal_rejected() -> None:
    with temporary_root() as tmp:
        try:
            resolve_artifact_path(tmp, "../clean_red_overlay.png")
        except VisualArtifactError:
            return
        raise AssertionError("expected traversal rejection")


def test_output_outside_report_root_rejected() -> None:
    with temporary_root() as tmp:
        try:
            resolve_artifact_path(tmp, "/tmp/clean_red_overlay.png")
        except VisualArtifactError:
            return
        raise AssertionError("expected outside-root rejection")


def test_manifest_contains_expected_keys() -> None:
    with temporary_root() as tmp:
        root = Path(tmp)
        image_path = root / "image.png"
        make_image(image_path)
        result = write_clean_red_overlay(image_path, [0.0, 1.0, 0.0, 1.0], root / "report", mask_size=2)
        manifest = json.loads(Path(result["clean_red_overlay_manifest_path"]).read_text(encoding="utf-8"))
        assert manifest["marker"] == CLEAN_OVERLAY_MARKER
        for key in (
            "clean_focused_mask_path",
            "clean_red_overlay_path",
            "clean_overlay_area_pct",
            "clean_overlay_keep_ratio",
            "clean_overlay_alpha",
            "clean_overlay_component_mode",
            "clean_overlay_threshold_mode",
            "uses_ground_truth_mask",
        ):
            assert key in manifest
        assert manifest["uses_ground_truth_mask"] is False


def main() -> int:
    tests = [
        test_clean_overlay_writes_expected_artifacts,
        test_pixels_outside_mask_unchanged_and_inside_red_blended,
        test_no_box_like_changes_outside_mask,
        test_top_percentile_thresholding_is_deterministic,
        test_largest_component_selection_is_deterministic,
        test_zero_mask_safe,
        test_invalid_alpha_rejected,
        test_path_traversal_rejected,
        test_output_outside_report_root_rejected,
        test_manifest_contains_expected_keys,
    ]
    for test in tests:
        test()
    print("PRE_SNS_CLEAN_RED_OVERLAY_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
