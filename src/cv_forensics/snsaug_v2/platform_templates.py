"""Generic platform-like UI templates for SNSAug V2."""

from __future__ import annotations

import random
from typing import Any

from PIL import Image

from .sticker_packs import (
    draw_ai_badge,
    draw_emoji_face,
    draw_heart,
    draw_highlight_box,
    draw_news_banner,
    draw_red_arrow,
    draw_red_circle,
    draw_red_rectangle,
    draw_speech_bubble,
    draw_watermark,
)
from .transforms import fit_content_to_canvas
from .ui_renderers import clamp_rect, draw_bottom_nav, draw_labeled_chip, draw_progress_bars, draw_simple_icon, draw_text_block, place_in_region


def _portrait_canvas(image: Image.Image, mask: Image.Image | None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    return fit_content_to_canvas(image, mask, (540, 960), background=(14, 14, 14))


def _choose_region(
    regions: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int, int, int]:
    roll = rng.random()
    if roll < 0.7:
        index = 0
    elif roll < 0.9:
        index = min(1, len(regions) - 1)
    else:
        index = min(2, len(regions) - 1)
    return regions[index]


def _placed_rect(
    image_size: tuple[int, int],
    rng: random.Random,
    regions: list[tuple[int, int, int, int]],
    base_size: tuple[int, int],
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    region = _choose_region(regions, rng)
    rect, meta = place_in_region(region, base_size, image_size, rng)
    return clamp_rect(rect, image_size), meta


def render_tiktok_like(image: Image.Image, mask: Image.Image | None, ignore_mask: Image.Image, rng: random.Random, font_path: str | None = None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    canvas, canvas_mask, geom = _portrait_canvas(image, mask)
    boxes = []
    boxes.append(draw_labeled_chip(canvas, ignore_mask, (26, 34, 138, 68), "Following", font_path=font_path))
    boxes.append(draw_labeled_chip(canvas, ignore_mask, (146, 34, 248, 68), "For You", fill=(255, 255, 255, 180), text_fill=(0, 0, 0), font_path=font_path))
    y = 260
    for kind in ("circle", "heart", "comment", "bookmark", "share"):
        boxes.append(draw_simple_icon(canvas, ignore_mask, (492, y), kind, radius=19))
        y += 92
    text_rect, text_meta = _placed_rect(canvas.size, rng, [(16, 720, 360, 900), (30, 640, 360, 860), (80, 460, 380, 700)], (290, 110))
    boxes.append(draw_text_block(canvas, ignore_mask, text_rect, ["@viewer_lab", "Shared clip", "audio: muted remix"], font_path=font_path, meta=text_meta))
    wm_rect, wm_meta = _placed_rect(canvas.size, rng, [(18, 680, 200, 760), (24, 600, 220, 760), (120, 500, 300, 700)], (146, 34))
    boxes.append(draw_watermark(canvas, ignore_mask, wm_rect, rng, font_path=font_path, meta=wm_meta))
    if rng.random() < 0.5:
        boxes.extend(draw_bottom_nav(canvas, ignore_mask, ["Home", "Discover", "Upload", "Inbox", "Me"], font_path=font_path))
    if rng.random() < 0.5:
        badge_rect, badge_meta = _placed_rect(canvas.size, rng, [(320, 660, 522, 800), (250, 700, 500, 860), (170, 560, 420, 760)], (150, 36))
        boxes.append(draw_ai_badge(canvas, ignore_mask, badge_rect, rng, font_path=font_path, meta=badge_meta))
    return canvas, canvas_mask, {"template": "tiktok_like", "overlay_boxes": boxes, "geometric_transform_meta": geom}


def render_instagram_story_like(image: Image.Image, mask: Image.Image | None, ignore_mask: Image.Image, rng: random.Random, font_path: str | None = None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    canvas, canvas_mask, geom = _portrait_canvas(image, mask)
    boxes = draw_progress_bars(canvas, ignore_mask, count=5)
    boxes.append(draw_labeled_chip(canvas, ignore_mask, (20, 34, 170, 72), "@daily.story  2h", font_path=font_path))
    if rng.random() < 0.7:
        rect, meta = _placed_rect(canvas.size, rng, [(36, 120, 350, 300), (70, 200, 380, 370), (140, 360, 420, 520)], (260, 90))
        boxes.append(draw_text_block(canvas, ignore_mask, rect, ["Weekend check-in", "location sticker"], fill=(255, 255, 255, 150), text_fill=(0, 0, 0), font_path=font_path, meta=meta))
    if rng.random() < 0.6:
        rect, meta = _placed_rect(canvas.size, rng, [(70, 520, 320, 640), (120, 420, 360, 580), (140, 640, 420, 760)], (210, 50))
        boxes.append(draw_labeled_chip(canvas, ignore_mask, rect, "Ask a question", fill=(255, 255, 255, 230), text_fill=(32, 32, 32), font_path=font_path, meta=meta))
    boxes.append(draw_labeled_chip(canvas, ignore_mask, (28, 884, 360, 934), "Send message", fill=(255, 255, 255, 120), text_fill=(255, 255, 255), font_path=font_path))
    boxes.append(draw_simple_icon(canvas, ignore_mask, (430, 908), "heart", radius=18))
    boxes.append(draw_simple_icon(canvas, ignore_mask, (484, 908), "share", radius=18))
    return canvas, canvas_mask, {"template": "instagram_story_like", "overlay_boxes": boxes, "geometric_transform_meta": geom}


def render_youtube_shorts_like(image: Image.Image, mask: Image.Image | None, ignore_mask: Image.Image, rng: random.Random, font_path: str | None = None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    canvas, canvas_mask, geom = _portrait_canvas(image, mask)
    boxes = []
    y = 300
    for kind in ("heart", "cross", "comment", "share", "play"):
        actual = "circle" if kind == "cross" else kind
        boxes.append(draw_simple_icon(canvas, ignore_mask, (492, y), actual, radius=18))
        y += 94
    rect, meta = _placed_rect(canvas.size, rng, [(14, 720, 380, 900), (24, 640, 380, 860), (120, 520, 430, 760)], (300, 110))
    boxes.append(draw_text_block(canvas, ignore_mask, rect, ["Channel Lab", "Shorts headline", "Subscribe"], font_path=font_path, meta=meta))
    if rng.random() < 0.5:
        badge_rect, badge_meta = _placed_rect(canvas.size, rng, [(320, 690, 520, 820), (260, 740, 510, 860), (150, 580, 400, 760)], (164, 36))
        boxes.append(draw_ai_badge(canvas, ignore_mask, badge_rect, rng, font_path=font_path, meta=badge_meta))
    if rng.random() < 0.6:
        boxes.extend(draw_bottom_nav(canvas, ignore_mask, ["Home", "Shorts", "Create", "Subs", "You"], font_path=font_path))
    return canvas, canvas_mask, {"template": "youtube_shorts_like", "overlay_boxes": boxes, "geometric_transform_meta": geom}


def render_news_meme_overlay(image: Image.Image, mask: Image.Image | None, ignore_mask: Image.Image, rng: random.Random, font_path: str | None = None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    canvas = image.copy()
    canvas_mask = None if mask is None else mask.copy()
    boxes = []
    width, height = canvas.size
    banner_rect, banner_meta = _placed_rect(canvas.size, rng, [(0, 0, min(width, 220), 80), (0, 0, min(width, 280), 120), (0, 0, min(width, width), 120)], (170, 40))
    boxes.append(draw_news_banner(canvas, ignore_mask, banner_rect, rng, font_path=font_path, meta=banner_meta))
    caption_rect, caption_meta = _placed_rect(canvas.size, rng, [(0, max(0, height - 120), width, height), (0, max(0, height - 180), width, height), (0, max(0, height // 2), width, height)], (width, 90))
    boxes.append(draw_text_block(canvas, ignore_mask, caption_rect, ["Headline overlay", "caption follows"], fill=(0, 0, 0, 210), font_path=font_path, meta=caption_meta))
    choice = rng.randrange(5)
    regions = [(width // 8, height // 8, width * 3 // 4, height * 3 // 4), (width // 5, height // 5, width - 20, height * 4 // 5), (width // 3, 20, width - 20, height // 2)]
    if choice == 0:
        rect, meta = _placed_rect(canvas.size, rng, regions, (width // 3, height // 3))
        boxes.append(draw_red_circle(canvas, ignore_mask, rect, meta=meta))
    elif choice == 1:
        rect, meta = _placed_rect(canvas.size, rng, regions, (width // 2, height // 4))
        boxes.append(draw_red_arrow(canvas, ignore_mask, rect, meta=meta))
    elif choice == 2:
        rect, meta = _placed_rect(canvas.size, rng, regions, (width // 3, height // 3))
        boxes.append(draw_red_rectangle(canvas, ignore_mask, rect, meta=meta))
    elif choice == 3:
        rect, meta = _placed_rect(canvas.size, rng, regions, (width // 3, height // 4))
        boxes.append(draw_speech_bubble(canvas, ignore_mask, rect, font_path=font_path, meta=meta))
    else:
        rect, meta = _placed_rect(canvas.size, rng, regions, (width // 3, height // 4))
        boxes.append(draw_highlight_box(canvas, ignore_mask, rect, meta=meta))
    return canvas, canvas_mask, {"template": "news_meme_overlay", "overlay_boxes": boxes, "geometric_transform_meta": {"type": "identity"}}


def render_annotation_sticker(image: Image.Image, mask: Image.Image | None, ignore_mask: Image.Image, rng: random.Random, font_path: str | None = None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    canvas = image.copy()
    canvas_mask = None if mask is None else mask.copy()
    width, height = canvas.size
    boxes = []
    regions = [(0, 0, width, height), (0, 0, width * 3 // 4, height * 3 // 4), (width // 4, height // 4, width, height)]
    choice = rng.randrange(4)
    if choice == 0:
        rect, meta = _placed_rect(canvas.size, rng, regions, (width // 3, height // 3))
        boxes.append(draw_red_circle(canvas, ignore_mask, rect, meta=meta))
    elif choice == 1:
        rect, meta = _placed_rect(canvas.size, rng, regions, (width // 2, height // 4))
        boxes.append(draw_red_arrow(canvas, ignore_mask, rect, meta=meta))
    elif choice == 2:
        rect, meta = _placed_rect(canvas.size, rng, regions, (width // 3, height // 3))
        boxes.append(draw_red_rectangle(canvas, ignore_mask, rect, meta=meta))
    else:
        rect1, meta1 = _placed_rect(canvas.size, rng, regions, (70, 70))
        rect2, meta2 = _placed_rect(canvas.size, rng, regions, (60, 60))
        boxes.append(draw_emoji_face(canvas, ignore_mask, rect1, meta=meta1))
        boxes.append(draw_heart(canvas, ignore_mask, rect2, meta=meta2))
    return canvas, canvas_mask, {"template": "annotation_sticker", "overlay_boxes": boxes, "geometric_transform_meta": {"type": "identity"}}
