"""Visual artifacts for pre-SNS single-image localization reports."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER = "PRE_SNS_VISUAL_ARTIFACTS_OK"
CLEAN_OVERLAY_MARKER = "PRE_SNS_CLEAN_RED_OVERLAY_OK"
ARTIFACT_FILENAMES = {
    "predicted_mask_path": "predicted_mask.png",
    "heatmap_path": "heatmap.png",
    "overlay_path": "overlay.png",
    "visual_artifacts_manifest_path": "visual_artifacts_manifest.json",
}
CLEAN_OVERLAY_FILENAMES = {
    "clean_focused_mask_path": "clean_focused_mask.png",
    "clean_red_overlay_path": "clean_red_overlay.png",
    "clean_red_overlay_manifest_path": "clean_red_overlay_manifest.json",
}


class VisualArtifactError(ValueError):
    """Raised when a visual artifact path or mask is unsafe."""


def _real(path: str | Path) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def _is_under(path: str | Path, root: str | Path) -> bool:
    try:
        return os.path.commonpath([str(_real(path)), str(_real(root))]) == str(_real(root))
    except ValueError:
        return False


def _repo_outputs_or_checkpoints(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT / "outputs") or _is_under(path, REPO_ROOT / "checkpoints")


def resolve_artifact_path(report_root: str | Path, relative_name: str) -> Path:
    """Resolve one artifact filename under report_root and reject traversal."""
    if not isinstance(relative_name, str) or not relative_name:
        raise VisualArtifactError("artifact name must be non-empty")
    normalized = relative_name.replace("\\", "/")
    if normalized.startswith("/") or any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise VisualArtifactError("artifact path traversal rejected")
    root = _real(report_root)
    path = _real(root / normalized)
    if not _is_under(path, root):
        raise VisualArtifactError("artifact path must stay under report_root")
    if _repo_outputs_or_checkpoints(path):
        raise VisualArtifactError("artifact path must not be under repository outputs/checkpoints")
    return path


def _flatten_values(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    elif hasattr(values, "reshape") and hasattr(values, "tolist"):
        values = values.reshape(-1).tolist()
    flat: list[float] = []
    if isinstance(values, list):
        for item in values:
            if isinstance(item, list):
                flat.extend(_flatten_values(item))
            else:
                try:
                    flat.append(float(item))
                except (TypeError, ValueError):
                    flat.append(0.0)
    else:
        try:
            flat.append(float(values))
        except (TypeError, ValueError):
            pass
    return flat


def normalize_mask_values(values: Any) -> list[float]:
    """Return deterministic [0, 1] probabilities, handling constant masks."""
    flat = _flatten_values(values)
    if not flat:
        raise VisualArtifactError("mask values must not be empty")
    finite = [value for value in flat if math.isfinite(value)]
    if not finite:
        return [0.0 for _ in flat]
    min_value = min(finite)
    max_value = max(finite)
    normalized: list[float] = []
    if 0.0 <= min_value and max_value <= 1.0:
        for value in flat:
            normalized.append(float(value) if math.isfinite(value) else 0.0)
        return normalized
    if max_value == min_value:
        return [0.0 for _ in flat]
    scale = max_value - min_value
    for value in flat:
        normalized.append(((float(value) - min_value) / scale) if math.isfinite(value) else 0.0)
    return [min(1.0, max(0.0, value)) for value in normalized]


def _mask_dimensions(count: int, mask_size: int | tuple[int, int] | None) -> tuple[int, int]:
    if isinstance(mask_size, tuple):
        width, height = int(mask_size[0]), int(mask_size[1])
    elif isinstance(mask_size, int):
        width = height = int(mask_size)
    else:
        side = int(math.sqrt(count))
        if side * side != count:
            raise VisualArtifactError("flat mask length is not square; provide mask_size")
        width = height = side
    if width <= 0 or height <= 0 or width * height != count:
        raise VisualArtifactError("mask_size does not match mask value count")
    return width, height


def _load_image_module():
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("PIL is required for pre-SNS visual artifacts") from exc
    return Image


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool):
        raise VisualArtifactError("clean overlay alpha must be numeric")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value <= 1.0:
        raise VisualArtifactError("clean overlay alpha must be in (0, 1]")
    return alpha_value


def _validate_keep_ratio(keep_ratio: float) -> float:
    if isinstance(keep_ratio, bool):
        raise VisualArtifactError("clean overlay keep_ratio must be numeric")
    ratio = float(keep_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise VisualArtifactError("clean overlay keep_ratio must be in (0, 1]")
    return ratio


def _validate_red_rgb(red_rgb: tuple[int, int, int] | list[int] | None) -> tuple[int, int, int]:
    if red_rgb is None:
        return (255, 0, 0)
    if not isinstance(red_rgb, (list, tuple)) or len(red_rgb) != 3:
        raise VisualArtifactError("clean overlay red_rgb must contain three integers")
    values: list[int] = []
    for item in red_rgb:
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255:
            raise VisualArtifactError("clean overlay red_rgb values must be integers in [0, 255]")
        values.append(item)
    return values[0], values[1], values[2]


def focused_mask_from_values(
    mask_values: Any,
    mask_size: int | tuple[int, int] | None = None,
    keep_ratio: float = 0.1,
    threshold_mode: str = "top_percentile",
    component_mode: str = "largest_component",
) -> tuple[list[int], tuple[int, int]]:
    """Build a deterministic binary mask from predicted localization values."""
    normalized = normalize_mask_values(mask_values)
    width, height = _mask_dimensions(len(normalized), mask_size)
    ratio = _validate_keep_ratio(keep_ratio)
    if threshold_mode not in {"top_percentile", "fixed_0_5"}:
        raise VisualArtifactError("clean overlay threshold_mode is unsupported")
    if component_mode not in {"largest_component", "all_components"}:
        raise VisualArtifactError("clean overlay component_mode is unsupported")

    if not normalized or max(normalized) <= 0.0:
        return [0 for _ in normalized], (width, height)
    if threshold_mode == "fixed_0_5":
        active = [1 if value >= 0.5 else 0 for value in normalized]
    else:
        keep_count = max(1, int(math.ceil(len(normalized) * ratio)))
        threshold = sorted(normalized, reverse=True)[keep_count - 1]
        active = [1 if value >= threshold and value > 0.0 else 0 for value in normalized]
    if component_mode == "largest_component":
        active = _largest_component(active, width, height)
    return active, (width, height)


def _largest_component(active: list[int], width: int, height: int) -> list[int]:
    visited = [False for _ in active]
    best: list[int] = []
    for index, value in enumerate(active):
        if not value or visited[index]:
            continue
        stack = [index]
        visited[index] = True
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            x = current % width
            y = current // width
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                neighbor = ny * width + nx
                if active[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if len(component) > len(best):
            best = component
    focused = [0 for _ in active]
    for index in best:
        focused[index] = 1
    return focused


def write_clean_red_overlay(
    image_path: str | Path,
    mask_values: Any,
    report_root: str | Path,
    mask_size: int | tuple[int, int] | None = None,
    keep_ratio: float = 0.1,
    component_mode: str = "largest_component",
    alpha: float = 0.45,
    threshold_mode: str = "top_percentile",
    red_rgb: tuple[int, int, int] | list[int] | None = None,
) -> dict[str, str | bool | float]:
    """Write a presentation-clean red mask overlay under report_root."""
    Image = _load_image_module()
    root = _real(report_root)
    if _repo_outputs_or_checkpoints(root):
        raise VisualArtifactError("report_root must not be under repository outputs/checkpoints")
    root.mkdir(parents=True, exist_ok=True)
    paths = {key: resolve_artifact_path(root, filename) for key, filename in CLEAN_OVERLAY_FILENAMES.items()}

    alpha_value = _validate_alpha(alpha)
    rgb = _validate_red_rgb(red_rgb)
    active, (mask_width, mask_height) = focused_mask_from_values(
        mask_values,
        mask_size=mask_size,
        keep_ratio=keep_ratio,
        threshold_mode=threshold_mode,
        component_mode=component_mode,
    )
    with Image.open(image_path) as original:
        original_rgb = original.convert("RGB")
    image_size = original_rgb.size

    mask_image = Image.new("L", (mask_width, mask_height))
    mask_image.putdata([255 if value else 0 for value in active])
    resample_nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    mask_resized = mask_image.resize(image_size, resample_nearest)
    mask_resized.save(paths["clean_focused_mask_path"])

    original_pixels = list(original_rgb.getdata())
    mask_pixels = list(mask_resized.getdata())
    blended_pixels: list[tuple[int, int, int]] = []
    active_count = 0
    for original_pixel, mask_pixel in zip(original_pixels, mask_pixels):
        if mask_pixel <= 0:
            blended_pixels.append(original_pixel)
            continue
        active_count += 1
        blended_pixels.append(
            tuple(
                int(round((1.0 - alpha_value) * original_pixel[channel] + alpha_value * rgb[channel]))
                for channel in range(3)
            )
        )
    overlay = Image.new("RGB", image_size)
    overlay.putdata(blended_pixels)
    overlay.save(paths["clean_red_overlay_path"])

    area_pct = float(active_count / max(len(mask_pixels), 1) * 100.0)
    artifact_paths: dict[str, str | bool | float] = {
        "clean_red_overlay_written": True,
        "clean_focused_mask_path": str(paths["clean_focused_mask_path"]),
        "clean_red_overlay_path": str(paths["clean_red_overlay_path"]),
        "clean_overlay_area_pct": area_pct,
        "clean_overlay_keep_ratio": float(keep_ratio),
        "clean_overlay_alpha": alpha_value,
        "clean_overlay_component_mode": component_mode,
        "clean_overlay_threshold_mode": threshold_mode,
    }
    manifest: dict[str, str | bool | float | list[int]] = {
        "marker": CLEAN_OVERLAY_MARKER,
        **artifact_paths,
        "clean_overlay_red_rgb": list(rgb),
        "uses_ground_truth_mask": False,
    }
    with open(paths["clean_red_overlay_manifest_path"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    artifact_paths["clean_red_overlay_manifest_path"] = str(paths["clean_red_overlay_manifest_path"])
    return artifact_paths


def write_visual_artifacts(
    image_path: str | Path,
    mask_values: Any,
    report_root: str | Path,
    mask_size: int | tuple[int, int] | None = None,
) -> dict[str, str | bool]:
    """Write predicted mask, heatmap, overlay, and manifest under report_root."""
    Image = _load_image_module()
    root = _real(report_root)
    if _repo_outputs_or_checkpoints(root):
        raise VisualArtifactError("report_root must not be under repository outputs/checkpoints")
    root.mkdir(parents=True, exist_ok=True)
    paths = {key: resolve_artifact_path(root, filename) for key, filename in ARTIFACT_FILENAMES.items()}

    normalized = normalize_mask_values(mask_values)
    mask_width, mask_height = _mask_dimensions(len(normalized), mask_size)
    with Image.open(image_path) as original:
        original_rgb = original.convert("RGB")
    image_size = original_rgb.size

    grayscale = [int(round(value * 255.0)) for value in normalized]
    mask_image = Image.new("L", (mask_width, mask_height))
    mask_image.putdata(grayscale)
    mask_resized = mask_image.resize(image_size)
    mask_resized.save(paths["predicted_mask_path"])

    heatmap = Image.new("RGB", (mask_width, mask_height))
    heatmap.putdata([(value, 0, 255 - value) for value in grayscale])
    heatmap_resized = heatmap.resize(image_size)
    heatmap_resized.save(paths["heatmap_path"])

    red_layer = Image.new("RGB", image_size, (255, 0, 0))
    alpha = mask_resized.point(lambda value: int(value * 0.45))
    overlay = Image.composite(red_layer, original_rgb, alpha)
    overlay = Image.blend(original_rgb, overlay, 0.65)
    overlay.save(paths["overlay_path"])

    artifact_paths: dict[str, str | bool] = {
        "visual_artifacts_written": True,
        "predicted_mask_path": str(paths["predicted_mask_path"]),
        "heatmap_path": str(paths["heatmap_path"]),
        "overlay_path": str(paths["overlay_path"]),
    }
    manifest: dict[str, str | bool] = {
        "marker": MARKER,
        **artifact_paths,
    }
    with open(paths["visual_artifacts_manifest_path"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    artifact_paths["visual_artifacts_manifest_path"] = str(paths["visual_artifacts_manifest_path"])
    return artifact_paths
