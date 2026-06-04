"""PIL drawing helpers for SNSAug V2 overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def load_font(font_path: str | None, size: int) -> Any:
    if font_path:
        try:
            return ImageFont.truetype(str(Path(font_path)), size=size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def rect_to_box(rect: tuple[int, int, int, int]) -> list[int]:
    return [int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])]


def _overlay_area(alpha: Image.Image) -> int:
    return int(sum(1 for value in alpha.getdata() if value > 0))


def _apply_overlay(image: Image.Image, ignore_mask: Image.Image, overlay: Image.Image) -> tuple[int, float]:
    composited = Image.alpha_composite(image.convert("RGBA"), overlay)
    image.paste(composited.convert(image.mode))
    alpha = overlay.getchannel("A")
    ignore_mask.paste(Image.new("L", ignore_mask.size, 255), (0, 0), alpha)
    area = _overlay_area(alpha)
    pct = (float(area) * 100.0 / float(max(1, ignore_mask.size[0] * ignore_mask.size[1])))
    return area, pct


def _new_overlay(image: Image.Image) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    return overlay, ImageDraw.Draw(overlay, "RGBA")


def _meta(kind: str, rect: tuple[int, int, int, int], area: int, pct: float, **extra: Any) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "element_type": kind,
        "box": rect_to_box(rect),
        "final_bbox": rect_to_box(rect),
        "alpha_mask_area_px": int(area),
        "ignore_mask_area_pct": float(pct),
    }
    payload.update(extra)
    return payload


def _clean_extra_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    extra = dict(meta or {})
    for key in ("kind", "box", "final_bbox", "alpha_mask_area_px", "ignore_mask_area_pct", "element_type"):
        extra.pop(key, None)
    return extra


def clamp_rect(rect: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x1 = max(0, min(width - 1, int(rect[0])))
    y1 = max(0, min(height - 1, int(rect[1])))
    x2 = max(x1 + 1, min(width, int(rect[2])))
    y2 = max(y1 + 1, min(height, int(rect[3])))
    return (x1, y1, x2, y2)


def place_in_region(
    region: tuple[int, int, int, int],
    base_size: tuple[int, int],
    image_size: tuple[int, int],
    rng: Any,
    *,
    size_jitter: float = 0.2,
    pos_jitter: float = 0.12,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    rx1, ry1, rx2, ry2 = region
    rw = max(1, rx2 - rx1)
    rh = max(1, ry2 - ry1)
    base_w, base_h = base_size
    scale = max(0.6, 1.0 + ((rng.random() - 0.5) * 2.0 * size_jitter))
    width = max(1, min(rw, int(round(base_w * scale))))
    height = max(1, min(rh, int(round(base_h * scale))))
    center_x = rx1 + rw // 2
    center_y = ry1 + rh // 2
    jitter_x = int(round((rng.random() - 0.5) * 2.0 * rw * pos_jitter))
    jitter_y = int(round((rng.random() - 0.5) * 2.0 * rh * pos_jitter))
    x1 = center_x - width // 2 + jitter_x
    y1 = center_y - height // 2 + jitter_y
    rect = clamp_rect((x1, y1, x1 + width, y1 + height), image_size)
    return rect, {
        "chosen_candidate_region": rect_to_box(region),
        "size_scale": float(scale),
        "jitter": [int(jitter_x), int(jitter_y)],
        "rotation_deg": 0.0,
    }


def draw_labeled_chip(
    image: Image.Image,
    ignore_mask: Image.Image,
    rect: tuple[int, int, int, int],
    text: str,
    *,
    fill: tuple[int, int, int, int] = (25, 25, 25, 220),
    text_fill: tuple[int, int, int] = (255, 255, 255),
    outline: tuple[int, int, int] | None = None,
    font_path: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overlay, draw = _new_overlay(image)
    draw.rounded_rectangle(rect, radius=10, fill=fill, outline=outline)
    font = load_font(font_path, size=max(10, (rect[3] - rect[1]) // 3))
    draw.text((rect[0] + 8, rect[1] + 8), text, fill=text_fill, font=font)
    area, pct = _apply_overlay(image, ignore_mask, overlay)
    extra = _clean_extra_meta(meta)
    extra.pop("text", None)
    return _meta("chip", rect, area, pct, text=text, **extra)


def draw_text_block(
    image: Image.Image,
    ignore_mask: Image.Image,
    rect: tuple[int, int, int, int],
    lines: list[str],
    *,
    fill: tuple[int, int, int, int] = (10, 10, 10, 150),
    text_fill: tuple[int, int, int] = (255, 255, 255),
    font_path: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overlay, draw = _new_overlay(image)
    draw.rounded_rectangle(rect, radius=10, fill=fill)
    font = load_font(font_path, size=max(10, (rect[3] - rect[1]) // max(3, len(lines) + 1)))
    y = rect[1] + 8
    for line in lines:
        draw.text((rect[0] + 8, y), line, fill=text_fill, font=font)
        y += 14
    area, pct = _apply_overlay(image, ignore_mask, overlay)
    extra = _clean_extra_meta(meta)
    extra.pop("lines", None)
    return _meta("text_block", rect, area, pct, lines=list(lines), **extra)


def draw_progress_bars(image: Image.Image, ignore_mask: Image.Image, count: int = 5) -> list[dict[str, Any]]:
    width, _height = image.size
    bar_gap = 4
    total_gap = bar_gap * (count - 1)
    bar_w = max(16, (width - 32 - total_gap) // count)
    boxes = []
    for index in range(count):
        x1 = 16 + index * (bar_w + bar_gap)
        rect = (x1, 16, x1 + bar_w, 22)
        overlay, draw = _new_overlay(image)
        draw.rounded_rectangle(rect, radius=3, fill=(255, 255, 255, 210 if index == 0 else 120))
        area, pct = _apply_overlay(image, ignore_mask, overlay)
        boxes.append(_meta("progress_bar", rect, area, pct))
    return boxes


def draw_simple_icon(
    image: Image.Image,
    ignore_mask: Image.Image,
    center: tuple[int, int],
    kind: str,
    radius: int = 18,
    *,
    color: tuple[int, int, int] = (255, 255, 255),
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cx, cy = center
    rect = (cx - radius, cy - radius, cx + radius, cy + radius)
    overlay, draw = _new_overlay(image)
    if kind == "heart":
        draw.ellipse((cx - radius, cy - radius + 6, cx, cy), fill=color)
        draw.ellipse((cx, cy - radius + 6, cx + radius, cy), fill=color)
        draw.polygon([(cx - radius, cy - 2), (cx + radius, cy - 2), (cx, cy + radius)], fill=color)
    elif kind == "comment":
        draw.rounded_rectangle(rect, radius=10, outline=color, width=3)
        draw.polygon([(cx - 6, cy + radius - 2), (cx - 1, cy + radius + 7), (cx + 5, cy + radius - 2)], fill=color)
    elif kind == "bookmark":
        draw.rectangle(rect, outline=color, width=3)
        draw.polygon([(cx - radius + 2, cy + radius - 2), (cx, cy + radius - 10), (cx + radius - 2, cy + radius - 2)], fill=color)
    elif kind == "share":
        draw.line((cx - radius + 3, cy + radius - 6, cx + radius - 2, cy - radius + 4), fill=color, width=4)
        draw.polygon([(cx + radius - 2, cy - radius + 4), (cx + radius - 10, cy - radius + 14), (cx + radius - 14, cy - radius + 2)], fill=color)
    elif kind == "play":
        draw.polygon([(cx - 6, cy - 10), (cx - 6, cy + 10), (cx + 10, cy)], fill=color)
    elif kind == "circle":
        draw.ellipse(rect, outline=color, width=3)
    else:
        draw.ellipse(rect, outline=color, width=3)
    area, pct = _apply_overlay(image, ignore_mask, overlay)
    return _meta(f"icon_{kind}", rect, area, pct, **(meta or {}))


def draw_bottom_nav(image: Image.Image, ignore_mask: Image.Image, labels: list[str], font_path: str | None = None) -> list[dict[str, Any]]:
    width, height = image.size
    rect = (0, height - 56, width, height)
    overlay, draw = _new_overlay(image)
    draw.rectangle(rect, fill=(0, 0, 0, 210))
    font = load_font(font_path, size=12)
    items = []
    for index, label in enumerate(labels):
        x = int((index + 0.5) * width / max(1, len(labels)))
        draw.text((x - 16, height - 34), label, fill=(255, 255, 255), font=font)
        items.append({"kind": "nav_label", "text": label, "box": [max(0, x - 22), height - 40, min(width, x + 22), height - 18]})
    area, pct = _apply_overlay(image, ignore_mask, overlay)
    items.insert(0, _meta("bottom_nav", rect, area, pct, labels=list(labels)))
    return items


def draw_overlay_debug(image: Image.Image, overlay_boxes: list[dict[str, Any]]) -> Image.Image:
    debug = image.convert("RGB").copy()
    draw = ImageDraw.Draw(debug, "RGBA")
    font = load_font(None, 12)
    for index, box_info in enumerate(overlay_boxes):
        box = box_info.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        rect = tuple(int(value) for value in box)
        label = str(box_info.get("kind") or f"overlay_{index}")
        draw.rectangle(rect, outline=(255, 64, 64, 255), width=2)
        tag = (rect[0], max(0, rect[1] - 16), min(debug.size[0], rect[0] + 110), rect[1])
        draw.rectangle(tag, fill=(0, 0, 0, 180))
        draw.text((tag[0] + 2, tag[1] + 2), label[:18], fill=(255, 255, 255), font=font)
    return debug
