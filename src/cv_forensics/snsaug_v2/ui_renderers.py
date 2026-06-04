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


def add_ignore_rectangle(ignore_mask: Image.Image, rect: tuple[int, int, int, int], radius: int = 0) -> None:
    draw = ImageDraw.Draw(ignore_mask)
    if radius > 0:
        draw.rounded_rectangle(rect, radius=radius, fill=255)
    else:
        draw.rectangle(rect, fill=255)


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
) -> dict[str, Any]:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(rect, radius=10, fill=fill, outline=outline)
    font = load_font(font_path, size=max(10, (rect[3] - rect[1]) // 3))
    draw.text((rect[0] + 8, rect[1] + 8), text, fill=text_fill, font=font)
    add_ignore_rectangle(ignore_mask, rect, radius=10)
    return {"kind": "chip", "text": text, "box": rect_to_box(rect)}


def draw_text_block(
    image: Image.Image,
    ignore_mask: Image.Image,
    rect: tuple[int, int, int, int],
    lines: list[str],
    *,
    fill: tuple[int, int, int, int] = (10, 10, 10, 150),
    text_fill: tuple[int, int, int] = (255, 255, 255),
    font_path: str | None = None,
) -> dict[str, Any]:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(rect, radius=10, fill=fill)
    font = load_font(font_path, size=max(10, (rect[3] - rect[1]) // max(3, len(lines) + 1)))
    y = rect[1] + 8
    for line in lines:
        draw.text((rect[0] + 8, y), line, fill=text_fill, font=font)
        y += 14
    add_ignore_rectangle(ignore_mask, rect, radius=10)
    return {"kind": "text_block", "lines": list(lines), "box": rect_to_box(rect)}


def draw_progress_bars(image: Image.Image, ignore_mask: Image.Image, count: int = 5) -> list[dict[str, Any]]:
    width, _height = image.size
    bar_gap = 4
    total_gap = bar_gap * (count - 1)
    bar_w = max(16, (width - 32 - total_gap) // count)
    boxes = []
    for index in range(count):
        x1 = 16 + index * (bar_w + bar_gap)
        rect = (x1, 16, x1 + bar_w, 22)
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rounded_rectangle(rect, radius=3, fill=(255, 255, 255, 210 if index == 0 else 120))
        add_ignore_rectangle(ignore_mask, rect, radius=3)
        boxes.append({"kind": "progress_bar", "box": rect_to_box(rect)})
    return boxes


def draw_simple_icon(
    image: Image.Image,
    ignore_mask: Image.Image,
    center: tuple[int, int],
    kind: str,
    radius: int = 18,
    *,
    color: tuple[int, int, int] = (255, 255, 255),
) -> dict[str, Any]:
    cx, cy = center
    rect = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw = ImageDraw.Draw(image, "RGBA")
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
    add_ignore_rectangle(ignore_mask, rect, radius=radius)
    return {"kind": f"icon_{kind}", "box": rect_to_box(rect)}


def draw_bottom_nav(image: Image.Image, ignore_mask: Image.Image, labels: list[str], font_path: str | None = None) -> list[dict[str, Any]]:
    width, height = image.size
    rect = (0, height - 56, width, height)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle(rect, fill=(0, 0, 0, 210))
    add_ignore_rectangle(ignore_mask, rect)
    font = load_font(font_path, size=12)
    items = []
    for index, label in enumerate(labels):
        x = int((index + 0.5) * width / max(1, len(labels)))
        draw.text((x - 16, height - 34), label, fill=(255, 255, 255), font=font)
        items.append({"kind": "nav_label", "text": label, "box": [max(0, x - 22), height - 40, min(width, x + 22), height - 18]})
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
