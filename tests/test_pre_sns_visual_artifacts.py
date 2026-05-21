#!/usr/bin/env python3
"""Standalone tests for pre-SNS visual artifact writing."""

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
    MARKER,
    VisualArtifactError,
    normalize_mask_values,
    resolve_artifact_path,
    write_visual_artifacts,
)

TEMP_ROOT = Path.home() / ".codex" / "memories"


def temporary_root():
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=str(TEMP_ROOT))


def make_image(path: Path) -> None:
    from PIL import Image

    image = Image.new("RGB", (6, 4), color=(80, 100, 120))
    image.save(path)


def test_writes_visual_artifacts() -> None:
    with temporary_root() as tmp:
        root = Path(tmp)
        image_path = root / "toy.png"
        report_root = root / "report"
        make_image(image_path)
        result = write_visual_artifacts(image_path, [0.0, 0.2, 0.8, 1.0], report_root, mask_size=2)
        for key in ("predicted_mask_path", "heatmap_path", "overlay_path", "visual_artifacts_manifest_path"):
            assert os.path.isfile(result[key]), key
            assert str(result[key]).startswith(str(report_root))
        assert result["visual_artifacts_written"] is True


def test_overlay_non_empty() -> None:
    with temporary_root() as tmp:
        root = Path(tmp)
        image_path = root / "toy.png"
        make_image(image_path)
        result = write_visual_artifacts(image_path, [0.0, 1.0, 1.0, 0.0], root / "report", mask_size=2)
        assert os.path.getsize(result["overlay_path"]) > 0


def test_path_traversal_rejected() -> None:
    with temporary_root() as tmp:
        try:
            resolve_artifact_path(tmp, "../overlay.png")
        except VisualArtifactError:
            return
        raise AssertionError("expected traversal rejection")


def test_outside_report_root_rejected() -> None:
    with temporary_root() as tmp:
        try:
            resolve_artifact_path(tmp, "/tmp/outside.png")
        except VisualArtifactError:
            return
        raise AssertionError("expected outside-root rejection")


def test_zero_mask_safe() -> None:
    with temporary_root() as tmp:
        root = Path(tmp)
        image_path = root / "toy.png"
        make_image(image_path)
        result = write_visual_artifacts(image_path, [0.0, 0.0, 0.0, 0.0], root / "report", mask_size=2)
        assert os.path.isfile(result["predicted_mask_path"])


def test_mask_normalization_deterministic() -> None:
    first = normalize_mask_values([-1.0, 0.0, 1.0])
    second = normalize_mask_values([-1.0, 0.0, 1.0])
    assert first == second
    assert first == [0.0, 0.5, 1.0]


def test_manifest_expected_keys() -> None:
    with temporary_root() as tmp:
        root = Path(tmp)
        image_path = root / "toy.png"
        make_image(image_path)
        result = write_visual_artifacts(image_path, [0.0, 1.0, 0.0, 1.0], root / "report", mask_size=2)
        manifest = json.loads(Path(result["visual_artifacts_manifest_path"]).read_text(encoding="utf-8"))
        assert manifest["marker"] == MARKER
        for key in ("predicted_mask_path", "heatmap_path", "overlay_path"):
            assert key in manifest


def main() -> int:
    tests = [
        test_writes_visual_artifacts,
        test_overlay_non_empty,
        test_path_traversal_rejected,
        test_outside_report_root_rejected,
        test_zero_mask_safe,
        test_mask_normalization_deterministic,
        test_manifest_expected_keys,
    ]
    for test in tests:
        test()
    print("PRE_SNS_VISUAL_ARTIFACTS_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
