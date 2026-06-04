"""Overlay sticker drawers built from PIL primitives only."""

from __future__ import annotations

import random
from typing import Any

from PIL import Image, ImageDraw

from .ui_renderers import add_ignore_rectangle, draw_labeled_chip, rect_to_box

AI_BADGES = ("Made with AI", "AI generated", "AI-edited", "Synthetic image", "Generated with AI")
WATERMARKS = ("short-video", "@user_sample", "@clipview")
NEWS_BANNERS = ("속보", "BREAKING", "NEWS", "LIVE", "단독")


def draw_ai_badge(image: Image.Image, ignore_mask: Image.Image, rect: tuple[int, int, int, int], rng: random.Random, font_path: str | None = None) -> dict[str, Any]:
    text = AI_BADGES[rng.randrange(len(AI_BADGES))]
    return draw_labeled_chip(image, ignore_mask, rect, text, fill=(24, 44, 64, 230), outline=(120, 200, 255), font_path=font_path)


def draw_watermark(image: Image.Image, ignore_mask: Image.Image, rect: tuple[int, int, int, int], rng: random.Random, font_path: str | None = None) -> dict[str, Any]:
    text = WATERMARKS[rng.randrange(len(WATERMARKS))]
    return draw_labeled_chip(image, ignore_mask, rect, text, fill=(255, 255, 255, 180), text_fill=(0, 0, 0), font_path=font_path)


def draw_news_banner(image: Image.Image, ignore_mask: Image.Image, rect: tuple[int, int, int, int], rng: random.Random, font_path: str | None = None) -> dict[str, Any]:
    text = NEWS_BANNERS[rng.randrange(len(NEWS_BANNERS))]
    return draw_labeled_chip(image, ignore_mask, rect, text, fill=(180, 20, 20, 235), font_path=font_path)


def draw_red_circle(image: Image.Image, ignore_mask: Image.Image, rect: tuple[int, int, int, int]) -> dict[str, Any]:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.ellipse(rect, outline=(255, 0, 0, 255), width=6)
    add_ignore_rectangle(ignore_mask, rect, radius=min(rect[2] - rect[0], rect[3] - rect[1]) // 2)
    return {"kind": "red_circle", "box": rect_to_box(rect)}


def draw_red_arrow(image: Image.Image, ignore_mask: Image.Image, rect: tuple[int, int, int, int]) -> dict[str, Any]:
    draw = ImageDraw.Draw(image, "RGBA")
    x1, y1, x2, y2 = rect
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    draw.line((left, bottom, right - 10, top + 10), fill=(255, 0, 0, 255), width=6)
    draw.polygon([(right - 10, top + 10), (right - 30, top + 6), (right - 14, top + 28)], fill=(255, 0, 0, 255))
    normalized = (left, top, right, bottom)
    add_ignore_rectangle(ignore_mask, normalized)
    return {"kind": "red_arrow", "box": rect_to_box(normalized)}


def draw_red_rectangle(image: Image.Image, ignore_mask: Image.Image, rect: tuple[int, int, int, int]) -> dict[str, Any]:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle(rect, outline=(255, 30, 30, 255), width=5)
    add_ignore_rectangle(ignore_mask, rect)
    return {"kind": "red_rectangle", "box": rect_to_box(rect)}


def draw_highlight_box(image: Image.Image, ignore_mask: Image.Image, rect: tuple[int, int, int, int]) -> dict[str, Any]:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(rect, radius=8, fill=(255, 235, 59, 120), outline=(255, 180, 0, 255), width=3)
    add_ignore_rectangle(ignore_mask, rect, radius=8)
    return {"kind": "highlight_box", "box": rect_to_box(rect)}


def draw_speech_bubble(image: Image.Image, ignore_mask: Image.Image, rect: tuple[int, int, int, int], font_path: str | None = None) -> dict[str, Any]:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(rect, radius=12, fill=(255, 255, 255, 220), outline=(0, 0, 0, 255), width=2)
    tail = [(rect[0] + 18, rect[3] - 2), (rect[0] + 36, rect[3] - 2), (rect[0] + 26, rect[3] + 12)]
    draw.polygon(tail, fill=(255, 255, 255, 220), outline=(0, 0, 0, 255))
    add_ignore_rectangle(ignore_mask, rect, radius=12)
    return {"kind": "speech_bubble", "box": rect_to_box((rect[0], rect[1], rect[2], rect[3] + 12))}


def draw_emoji_face(image: Image.Image, ignore_mask: Image.Image, rect: tuple[int, int, int, int]) -> dict[str, Any]:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.ellipse(rect, fill=(255, 211, 67, 235), outline=(0, 0, 0, 180), width=2)
    x1, y1, x2, y2 = rect
    eye_y = y1 + (y2 - y1) // 3
    draw.ellipse((x1 + 10, eye_y, x1 + 16, eye_y + 6), fill=(0, 0, 0, 255))
    draw.ellipse((x2 - 16, eye_y, x2 - 10, eye_y + 6), fill=(0, 0, 0, 255))
    draw.arc((x1 + 10, y1 + 12, x2 - 10, y2 - 8), start=10, end=170, fill=(0, 0, 0, 255), width=2)
    add_ignore_rectangle(ignore_mask, rect, radius=(x2 - x1) // 2)
    return {"kind": "emoji_face", "box": rect_to_box(rect)}


def draw_heart(image: Image.Image, ignore_mask: Image.Image, rect: tuple[int, int, int, int]) -> dict[str, Any]:
    draw = ImageDraw.Draw(image, "RGBA")
    x1, y1, x2, y2 = rect
    mid = (x1 + x2) // 2
    draw.ellipse((x1, y1, mid, y1 + (y2 - y1) // 2), fill=(255, 0, 80, 235))
    draw.ellipse((mid, y1, x2, y1 + (y2 - y1) // 2), fill=(255, 0, 80, 235))
    draw.polygon([(x1 + 3, y1 + (y2 - y1) // 3), (x2 - 3, y1 + (y2 - y1) // 3), (mid, y2)], fill=(255, 0, 80, 235))
    add_ignore_rectangle(ignore_mask, rect)
    return {"kind": "heart", "box": rect_to_box(rect)}


def draw_check_or_cross(image: Image.Image, ignore_mask: Image.Image, rect: tuple[int, int, int, int], positive: bool = True) -> dict[str, Any]:
    draw = ImageDraw.Draw(image, "RGBA")
    x1, y1, x2, y2 = rect
    color = (10, 180, 80, 255) if positive else (220, 60, 60, 255)
    if positive:
        draw.line((x1 + 8, y1 + (y2 - y1) // 2, x1 + (x2 - x1) // 2, y2 - 8), fill=color, width=6)
        draw.line((x1 + (x2 - x1) // 2, y2 - 8, x2 - 8, y1 + 8), fill=color, width=6)
    else:
        draw.line((x1 + 8, y1 + 8, x2 - 8, y2 - 8), fill=color, width=6)
        draw.line((x2 - 8, y1 + 8, x1 + 8, y2 - 8), fill=color, width=6)
    add_ignore_rectangle(ignore_mask, rect)
    return {"kind": "check" if positive else "cross", "box": rect_to_box(rect)}
