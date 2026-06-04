"""Deterministic image and mask transforms for SNSAug V2."""

from __future__ import annotations

import io
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter


def severity_values(severity: str) -> dict[str, Any]:
    table = {
        "light": {"jpeg_quality": 90, "resize_long": 1024, "blur_radius": 0.5, "pixel_step": 2, "color_factor": 1.05},
        "medium": {"jpeg_quality": 78, "resize_long": 900, "blur_radius": 1.2, "pixel_step": 3, "color_factor": 1.1},
        "strong": {"jpeg_quality": 62, "resize_long": 720, "blur_radius": 2.0, "pixel_step": 4, "color_factor": 1.18},
    }
    if severity not in table:
        raise ValueError(f"unsupported severity: {severity}")
    return table[severity]


def ensure_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def ensure_mask(mask: Image.Image | None, size: tuple[int, int]) -> Image.Image | None:
    if mask is None:
        return None
    return mask.convert("L").resize(size, Image.Resampling.NEAREST) if mask.size != size else mask.convert("L")


def blank_mask(size: tuple[int, int]) -> Image.Image:
    return Image.new("L", size, 0)


def resize_long_side(
    image: Image.Image,
    mask: Image.Image | None,
    long_side: int,
) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    width, height = image.size
    current = max(width, height)
    if current == long_side:
        return image.copy(), None if mask is None else mask.copy(), {"type": "resize_long_side", "long_side": long_side, "scale": 1.0}
    scale = float(long_side) / float(current)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    out_image = image.resize(new_size, Image.Resampling.BILINEAR)
    out_mask = mask.resize(new_size, Image.Resampling.NEAREST) if mask is not None else None
    return out_image, out_mask, {"type": "resize_long_side", "long_side": long_side, "scale": scale, "output_size": list(new_size)}


def fit_content_to_canvas(
    image: Image.Image,
    mask: Image.Image | None,
    canvas_size: tuple[int, int],
    background: tuple[int, int, int] = (18, 18, 18),
) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    canvas_w, canvas_h = canvas_size
    src_w, src_h = image.size
    scale = min(canvas_w / float(src_w), canvas_h / float(src_h))
    scaled_size = (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale))))
    resized = image.resize(scaled_size, Image.Resampling.BILINEAR)
    resized_mask = mask.resize(scaled_size, Image.Resampling.NEAREST) if mask is not None else None
    x = max(0, (canvas_w - scaled_size[0]) // 2)
    y = max(0, (canvas_h - scaled_size[1]) // 2)
    canvas = Image.new("RGB", canvas_size, background)
    canvas.paste(resized, (x, y))
    canvas_mask = Image.new("L", canvas_size, 0) if mask is not None else None
    if canvas_mask is not None and resized_mask is not None:
        canvas_mask.paste(resized_mask, (x, y))
    return canvas, canvas_mask, {
        "type": "fit_content_to_canvas",
        "canvas_size": list(canvas_size),
        "scaled_size": list(scaled_size),
        "offset": [x, y],
        "scale": scale,
    }


def center_crop_resize_back(
    image: Image.Image,
    mask: Image.Image | None,
    keep_ratio: float,
) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    width, height = image.size
    crop_w = max(1, int(round(width * keep_ratio)))
    crop_h = max(1, int(round(height * keep_ratio)))
    x1 = max(0, (width - crop_w) // 2)
    y1 = max(0, (height - crop_h) // 2)
    box = (x1, y1, x1 + crop_w, y1 + crop_h)
    out_image = image.crop(box).resize((width, height), Image.Resampling.BILINEAR)
    out_mask = mask.crop(box).resize((width, height), Image.Resampling.NEAREST) if mask is not None else None
    return out_image, out_mask, {"type": "center_crop_resize_back", "keep_ratio": keep_ratio, "crop_box": list(box)}


def recompress_jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_color_shift(image: Image.Image, rng, factor: float) -> Image.Image:
    sat = 1.0 + (rng.random() - 0.5) * 2.0 * (factor - 1.0)
    bright = 1.0 + (rng.random() - 0.5) * 0.12
    out = ImageEnhance.Color(image).enhance(max(0.6, sat))
    out = ImageEnhance.Brightness(out).enhance(max(0.7, bright))
    return out


def apply_blur_pixelation(image: Image.Image, blur_radius: float, pixel_step: int) -> Image.Image:
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if pixel_step <= 1:
        return blurred
    small = blurred.resize((max(1, image.size[0] // pixel_step), max(1, image.size[1] // pixel_step)), Image.Resampling.BILINEAR)
    return small.resize(image.size, Image.Resampling.NEAREST)
