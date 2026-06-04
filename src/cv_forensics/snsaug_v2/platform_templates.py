"""Generic platform-like UI templates for SNSAug V2."""

from __future__ import annotations

import random
from typing import Any

from PIL import Image, ImageDraw

from .sticker_packs import (
    draw_ai_badge,
    draw_emoji_face,
    draw_heart,
    draw_news_banner,
    draw_red_arrow,
    draw_red_circle,
    draw_red_rectangle,
    draw_speech_bubble,
    draw_watermark,
)
from .transforms import fit_content_to_canvas
from .ui_renderers import draw_bottom_nav, draw_labeled_chip, draw_progress_bars, draw_simple_icon, draw_text_block


def _portrait_canvas(image: Image.Image, mask: Image.Image | None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    return fit_content_to_canvas(image, mask, (540, 960), background=(14, 14, 14))


def render_tiktok_like(image: Image.Image, mask: Image.Image | None, ignore_mask: Image.Image, rng: random.Random, font_path: str | None = None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    canvas, canvas_mask, geom = _portrait_canvas(image, mask)
    boxes = []
    boxes.append(draw_labeled_chip(canvas, ignore_mask, (26, 34, 138, 68), "Following", font_path=font_path))
    boxes.append(draw_labeled_chip(canvas, ignore_mask, (146, 34, 248, 68), "For You", fill=(255, 255, 255, 180), text_fill=(0, 0, 0), font_path=font_path))
    y = 260
    for kind in ("circle", "heart", "comment", "bookmark", "share"):
        boxes.append(draw_simple_icon(canvas, ignore_mask, (492, y), kind, radius=19))
        y += 92
    boxes.append(draw_text_block(canvas, ignore_mask, (24, 760, 330, 884), ["@viewer_lab", "Shared clip", "audio: muted remix"], font_path=font_path))
    boxes.append(draw_watermark(canvas, ignore_mask, (24, 708, 170, 742), rng, font_path=font_path))
    if rng.random() < 0.5:
        boxes.extend(draw_bottom_nav(canvas, ignore_mask, ["Home", "Discover", "Upload", "Inbox", "Me"], font_path=font_path))
    if rng.random() < 0.5:
        boxes.append(draw_ai_badge(canvas, ignore_mask, (364, 700, 510, 736), rng, font_path=font_path))
    return canvas, canvas_mask, {"template": "tiktok_like", "overlay_boxes": boxes, "geometric_transform_meta": geom}


def render_instagram_story_like(image: Image.Image, mask: Image.Image | None, ignore_mask: Image.Image, rng: random.Random, font_path: str | None = None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    canvas, canvas_mask, geom = _portrait_canvas(image, mask)
    boxes = draw_progress_bars(canvas, ignore_mask, count=5)
    boxes.append(draw_labeled_chip(canvas, ignore_mask, (20, 34, 170, 72), "@daily.story  2h", font_path=font_path))
    if rng.random() < 0.7:
        boxes.append(draw_text_block(canvas, ignore_mask, (44, 150, 330, 250), ["Weekend check-in", "location sticker"], fill=(255, 255, 255, 150), text_fill=(0, 0, 0), font_path=font_path))
    if rng.random() < 0.6:
        boxes.append(draw_labeled_chip(canvas, ignore_mask, (80, 560, 290, 610), "Ask a question", fill=(255, 255, 255, 230), text_fill=(32, 32, 32), font_path=font_path))
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
    boxes.append(draw_text_block(canvas, ignore_mask, (20, 748, 356, 878), ["Channel Lab", "Shorts headline", "Subscribe"], font_path=font_path))
    if rng.random() < 0.5:
        boxes.append(draw_ai_badge(canvas, ignore_mask, (350, 740, 514, 776), rng, font_path=font_path))
    if rng.random() < 0.6:
        boxes.extend(draw_bottom_nav(canvas, ignore_mask, ["Home", "Shorts", "Create", "Subs", "You"], font_path=font_path))
    return canvas, canvas_mask, {"template": "youtube_shorts_like", "overlay_boxes": boxes, "geometric_transform_meta": geom}


def render_news_meme_overlay(image: Image.Image, mask: Image.Image | None, ignore_mask: Image.Image, rng: random.Random, font_path: str | None = None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    canvas = image.copy()
    canvas_mask = None if mask is None else mask.copy()
    boxes = []
    width, height = canvas.size
    boxes.append(draw_news_banner(canvas, ignore_mask, (0, 0, min(width, 200), 44), rng, font_path=font_path))
    boxes.append(draw_text_block(canvas, ignore_mask, (0, height - 90, width, height), ["Headline overlay", "caption follows"], fill=(0, 0, 0, 210), font_path=font_path))
    choice = rng.randrange(4)
    if choice == 0:
        boxes.append(draw_red_circle(canvas, ignore_mask, (width // 4, height // 4, width // 2, height // 2)))
    elif choice == 1:
        boxes.append(draw_red_arrow(canvas, ignore_mask, (width // 3, height // 3, width - 20, height // 2)))
    elif choice == 2:
        boxes.append(draw_red_rectangle(canvas, ignore_mask, (width // 4, height // 4, width // 2, height // 2)))
    else:
        boxes.append(draw_speech_bubble(canvas, ignore_mask, (width // 2, 40, min(width - 20, width // 2 + 160), 110), font_path=font_path))
    return canvas, canvas_mask, {"template": "news_meme_overlay", "overlay_boxes": boxes, "geometric_transform_meta": {"type": "identity"}}


def render_annotation_sticker(image: Image.Image, mask: Image.Image | None, ignore_mask: Image.Image, rng: random.Random, font_path: str | None = None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    canvas = image.copy()
    canvas_mask = None if mask is None else mask.copy()
    width, height = canvas.size
    boxes = []
    choice = rng.randrange(4)
    if choice == 0:
        boxes.append(draw_red_circle(canvas, ignore_mask, (width // 4, height // 4, width // 2, height // 2)))
    elif choice == 1:
        boxes.append(draw_red_arrow(canvas, ignore_mask, (width // 5, height // 2, width - width // 6, height // 4)))
    elif choice == 2:
        boxes.append(draw_red_rectangle(canvas, ignore_mask, (width // 3, height // 3, width - width // 4, height - height // 4)))
    else:
        boxes.append(draw_emoji_face(canvas, ignore_mask, (width - 90, 20, width - 20, 90)))
        boxes.append(draw_heart(canvas, ignore_mask, (20, 20, 80, 80)))
    return canvas, canvas_mask, {"template": "annotation_sticker", "overlay_boxes": boxes, "geometric_transform_meta": {"type": "identity"}}
