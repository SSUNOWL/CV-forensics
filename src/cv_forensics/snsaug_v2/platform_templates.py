"""Generic platform-like UI templates for SNSAug V2."""

from __future__ import annotations

import random
from typing import Any

from PIL import Image

from .placement import PLACEMENT_POLICY_VERSION, PlacementManager
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
from .ui_renderers import clamp_rect, draw_bottom_nav, draw_labeled_chip, draw_progress_bars, draw_simple_icon, draw_text_block, place_in_region, rect_to_box


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


def _mask_delta(before: Image.Image, after: Image.Image) -> Image.Image:
    out = Image.new("L", after.size, 0)
    before_bytes = before.tobytes()
    after_bytes = after.tobytes()
    out.putdata([255 if after_bytes[index] > before_bytes[index] else 0 for index in range(len(after_bytes))])
    return out


def _register_fixed(manager: PlacementManager, image: Image.Image, ignore_mask: Image.Image, draw_fn, element_id: str, element_type: str) -> dict[str, Any]:
    before = ignore_mask.copy()
    meta = draw_fn()
    alpha = _mask_delta(before, ignore_mask)
    manager.register_existing(
        element_id=element_id,
        element_type=element_type,
        bbox=tuple(meta.get("box", [0, 0, image.size[0], image.size[1]])),
        alpha_mask=alpha,
        metadata=meta,
    )
    return meta


def _preview_variable(draw_fn, image_size: tuple[int, int]) -> tuple[Image.Image, dict[str, Any]]:
    preview_image = Image.new("RGB", image_size, (0, 0, 0))
    preview_ignore = Image.new("L", image_size, 0)
    meta = draw_fn(preview_image, preview_ignore)
    return preview_ignore, meta


def _place_variable(
    *,
    manager: PlacementManager,
    image: Image.Image,
    ignore_mask: Image.Image,
    rng: random.Random,
    element_id: str,
    element_type: str,
    candidate_regions: list[tuple[int, int, int, int]],
    base_size: tuple[int, int],
    preview_builder,
    commit_builder,
    optional: bool = True,
) -> dict[str, Any] | None:
    decision = manager.place_variable(
        element_id=element_id,
        element_type=element_type,
        candidate_regions=candidate_regions,
        base_size=base_size,
        rng=rng,
        render_preview=lambda rect: _preview_variable(preview_builder(rect), image.size),
        optional=optional,
    )
    if not decision.accepted or decision.bbox is None:
        return {
            "kind": element_type,
            "element_id": element_id,
            "element_type": element_type,
            "accepted": False,
            "skipped_due_to_overlap": True,
            "attempt_count": decision.attempt_count,
            "rejected_candidates": decision.metadata.get("rejected_candidates", decision.attempt_count),
            "max_overlap_ratio": decision.max_overlap_ratio,
            "overlapped_with": decision.metadata.get("overlapped_with", []),
            "placement_margin_px": int(manager.config.placement_margin_px),
            "placement_policy_version": PLACEMENT_POLICY_VERSION,
        }
    before = ignore_mask.copy()
    meta = commit_builder(decision.bbox, decision.metadata)
    alpha = _mask_delta(before, ignore_mask)
    manager.register_existing(
        element_id=element_id,
        element_type=element_type,
        bbox=tuple(meta.get("box", rect_to_box(decision.bbox))),
        alpha_mask=alpha,
        candidate_region=tuple(decision.metadata.get("candidate_region", rect_to_box(decision.bbox))),
        metadata=meta,
    )
    meta["element_id"] = element_id
    meta["accepted"] = True
    meta["attempt_count"] = decision.attempt_count
    meta["skipped_due_to_overlap"] = False
    meta["rejected_candidates"] = max(0, decision.attempt_count - 1)
    meta["max_overlap_ratio"] = decision.max_overlap_ratio
    meta["overlapped_with"] = decision.metadata.get("overlapped_with", [])
    meta["placement_margin_px"] = int(manager.config.placement_margin_px)
    meta["placement_policy_version"] = PLACEMENT_POLICY_VERSION
    meta["candidate_region"] = decision.metadata.get("candidate_region")
    return meta


def render_tiktok_like(image: Image.Image, mask: Image.Image | None, ignore_mask: Image.Image, rng: random.Random, font_path: str | None = None, config: Any | None = None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    canvas, canvas_mask, geom = _portrait_canvas(image, mask)
    manager = PlacementManager(canvas_size=canvas.size, config=config)
    boxes = []
    boxes.append(_register_fixed(manager, canvas, ignore_mask, lambda: draw_labeled_chip(canvas, ignore_mask, (26, 34, 138, 68), "Following", font_path=font_path), "fixed_tiktok_following", "fixed_ui_chip"))
    boxes.append(_register_fixed(manager, canvas, ignore_mask, lambda: draw_labeled_chip(canvas, ignore_mask, (146, 34, 248, 68), "For You", fill=(255, 255, 255, 180), text_fill=(0, 0, 0), font_path=font_path), "fixed_tiktok_for_you", "fixed_ui_chip"))
    y = 260
    for kind in ("circle", "heart", "comment", "bookmark", "share"):
        boxes.append(_register_fixed(manager, canvas, ignore_mask, lambda kind=kind, y=y: draw_simple_icon(canvas, ignore_mask, (492, y), kind, radius=19), f"fixed_tiktok_{kind}_{y}", f"fixed_ui_icon_{kind}"))
        y += 92
    text_item = _place_variable(
        manager=manager,
        image=canvas,
        ignore_mask=ignore_mask,
        rng=rng,
        element_id="tiktok_text_block",
        element_type="text_block",
        candidate_regions=[(16, 720, 360, 900), (30, 640, 360, 860), (80, 460, 380, 700)],
        base_size=(290, 110),
        preview_builder=lambda rect: (lambda img, mask_img: draw_text_block(img, mask_img, rect, ["@viewer_lab", "Shared clip", "audio: muted remix"], font_path=font_path)),
        commit_builder=lambda rect, meta: draw_text_block(canvas, ignore_mask, rect, ["@viewer_lab", "Shared clip", "audio: muted remix"], font_path=font_path, meta=meta),
    )
    if text_item is not None:
        boxes.append(text_item)
    wm_item = _place_variable(
        manager=manager,
        image=canvas,
        ignore_mask=ignore_mask,
        rng=rng,
        element_id="tiktok_watermark",
        element_type="sticker",
        candidate_regions=[(18, 680, 200, 760), (24, 600, 220, 760), (120, 500, 300, 700)],
        base_size=(146, 34),
        preview_builder=lambda rect: (lambda img, mask_img: draw_watermark(img, mask_img, rect, random.Random(1), font_path=font_path)),
        commit_builder=lambda rect, meta: draw_watermark(canvas, ignore_mask, rect, random.Random(1), font_path=font_path, meta=meta),
    )
    if wm_item is not None:
        boxes.append(wm_item)
    if rng.random() < 0.5:
        nav_items = draw_bottom_nav(canvas, ignore_mask, ["Home", "Discover", "Upload", "Inbox", "Me"], font_path=font_path)
        boxes.extend(nav_items)
        manager.register_existing(element_id="fixed_tiktok_bottom_nav", element_type="fixed_ui_bottom_nav", bbox=(0, canvas.size[1] - 56, canvas.size[0], canvas.size[1]), alpha_mask=_mask_delta(Image.new("L", ignore_mask.size, 0), ignore_mask), metadata={"kind": "bottom_nav"})
    if rng.random() < 0.5:
        badge_item = _place_variable(
            manager=manager,
            image=canvas,
            ignore_mask=ignore_mask,
            rng=rng,
            element_id="tiktok_ai_badge",
            element_type="badge",
            candidate_regions=[(320, 660, 522, 800), (250, 700, 500, 860), (170, 560, 420, 760)],
            base_size=(150, 36),
            preview_builder=lambda rect: (lambda img, mask_img: draw_ai_badge(img, mask_img, rect, random.Random(2), font_path=font_path)),
            commit_builder=lambda rect, meta: draw_ai_badge(canvas, ignore_mask, rect, random.Random(2), font_path=font_path, meta=meta),
        )
        if badge_item is not None:
            boxes.append(badge_item)
    return canvas, canvas_mask, {"template": "tiktok_like", "overlay_boxes": boxes, "geometric_transform_meta": geom, "placement_manager": manager}


def render_instagram_story_like(image: Image.Image, mask: Image.Image | None, ignore_mask: Image.Image, rng: random.Random, font_path: str | None = None, config: Any | None = None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    canvas, canvas_mask, geom = _portrait_canvas(image, mask)
    manager = PlacementManager(canvas_size=canvas.size, config=config)
    boxes = draw_progress_bars(canvas, ignore_mask, count=5)
    for index, item in enumerate(boxes):
        manager.register_existing(element_id=f"fixed_instagram_progress_{index}", element_type="fixed_ui_progress_bar", bbox=tuple(item["box"]), alpha_mask=Image.new("L", canvas.size, 0), metadata=item)
    boxes.append(_register_fixed(manager, canvas, ignore_mask, lambda: draw_labeled_chip(canvas, ignore_mask, (20, 34, 170, 72), "@daily.story  2h", font_path=font_path), "fixed_instagram_account_row", "fixed_ui_chip"))
    if rng.random() < 0.7:
        item = _place_variable(
            manager=manager,
            image=canvas,
            ignore_mask=ignore_mask,
            rng=rng,
            element_id="instagram_text_block",
            element_type="text_block",
            candidate_regions=[(36, 120, 350, 300), (70, 200, 380, 370), (140, 360, 420, 520)],
            base_size=(260, 90),
            preview_builder=lambda rect: (lambda img, mask_img: draw_text_block(img, mask_img, rect, ["Weekend check-in", "location sticker"], fill=(255, 255, 255, 150), text_fill=(0, 0, 0), font_path=font_path)),
            commit_builder=lambda rect, meta: draw_text_block(canvas, ignore_mask, rect, ["Weekend check-in", "location sticker"], fill=(255, 255, 255, 150), text_fill=(0, 0, 0), font_path=font_path, meta=meta),
        )
        if item is not None:
            boxes.append(item)
    if rng.random() < 0.6:
        item = _place_variable(
            manager=manager,
            image=canvas,
            ignore_mask=ignore_mask,
            rng=rng,
            element_id="instagram_question_sticker",
            element_type="text_block",
            candidate_regions=[(70, 520, 320, 640), (120, 420, 360, 580), (140, 640, 420, 760)],
            base_size=(210, 50),
            preview_builder=lambda rect: (lambda img, mask_img: draw_labeled_chip(img, mask_img, rect, "Ask a question", fill=(255, 255, 255, 230), text_fill=(32, 32, 32), font_path=font_path)),
            commit_builder=lambda rect, meta: draw_labeled_chip(canvas, ignore_mask, rect, "Ask a question", fill=(255, 255, 255, 230), text_fill=(32, 32, 32), font_path=font_path, meta=meta),
        )
        if item is not None:
            boxes.append(item)
    boxes.append(_register_fixed(manager, canvas, ignore_mask, lambda: draw_labeled_chip(canvas, ignore_mask, (28, 884, 360, 934), "Send message", fill=(255, 255, 255, 120), text_fill=(255, 255, 255), font_path=font_path), "fixed_instagram_message_bar", "fixed_ui_chip"))
    boxes.append(_register_fixed(manager, canvas, ignore_mask, lambda: draw_simple_icon(canvas, ignore_mask, (430, 908), "heart", radius=18), "fixed_instagram_heart", "fixed_ui_icon_heart"))
    boxes.append(_register_fixed(manager, canvas, ignore_mask, lambda: draw_simple_icon(canvas, ignore_mask, (484, 908), "share", radius=18), "fixed_instagram_share", "fixed_ui_icon_share"))
    return canvas, canvas_mask, {"template": "instagram_story_like", "overlay_boxes": boxes, "geometric_transform_meta": geom, "placement_manager": manager}


def render_youtube_shorts_like(image: Image.Image, mask: Image.Image | None, ignore_mask: Image.Image, rng: random.Random, font_path: str | None = None, config: Any | None = None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    canvas, canvas_mask, geom = _portrait_canvas(image, mask)
    manager = PlacementManager(canvas_size=canvas.size, config=config)
    boxes = []
    y = 300
    for kind in ("heart", "cross", "comment", "share", "play"):
        actual = "circle" if kind == "cross" else kind
        boxes.append(_register_fixed(manager, canvas, ignore_mask, lambda actual=actual, y=y: draw_simple_icon(canvas, ignore_mask, (492, y), actual, radius=18), f"fixed_youtube_{actual}_{y}", f"fixed_ui_icon_{actual}"))
        y += 94
    item = _place_variable(
        manager=manager,
        image=canvas,
        ignore_mask=ignore_mask,
        rng=rng,
        element_id="youtube_text_block",
        element_type="text_block",
        candidate_regions=[(14, 720, 380, 900), (24, 640, 380, 860), (120, 520, 430, 760)],
        base_size=(300, 110),
        preview_builder=lambda rect: (lambda img, mask_img: draw_text_block(img, mask_img, rect, ["Channel Lab", "Shorts headline", "Subscribe"], font_path=font_path)),
        commit_builder=lambda rect, meta: draw_text_block(canvas, ignore_mask, rect, ["Channel Lab", "Shorts headline", "Subscribe"], font_path=font_path, meta=meta),
    )
    if item is not None:
        boxes.append(item)
    if rng.random() < 0.5:
        badge = _place_variable(
            manager=manager,
            image=canvas,
            ignore_mask=ignore_mask,
            rng=rng,
            element_id="youtube_ai_badge",
            element_type="badge",
            candidate_regions=[(320, 690, 520, 820), (260, 740, 510, 860), (150, 580, 400, 760)],
            base_size=(164, 36),
            preview_builder=lambda rect: (lambda img, mask_img: draw_ai_badge(img, mask_img, rect, random.Random(3), font_path=font_path)),
            commit_builder=lambda rect, meta: draw_ai_badge(canvas, ignore_mask, rect, random.Random(3), font_path=font_path, meta=meta),
        )
        if badge is not None:
            boxes.append(badge)
    if rng.random() < 0.6:
        boxes.extend(draw_bottom_nav(canvas, ignore_mask, ["Home", "Shorts", "Create", "Subs", "You"], font_path=font_path))
    return canvas, canvas_mask, {"template": "youtube_shorts_like", "overlay_boxes": boxes, "geometric_transform_meta": geom, "placement_manager": manager}


def render_news_meme_overlay(image: Image.Image, mask: Image.Image | None, ignore_mask: Image.Image, rng: random.Random, font_path: str | None = None, config: Any | None = None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    canvas = image.copy()
    canvas_mask = None if mask is None else mask.copy()
    manager = PlacementManager(canvas_size=canvas.size, config=config)
    boxes = []
    width, height = canvas.size
    banner = _place_variable(
        manager=manager,
        image=canvas,
        ignore_mask=ignore_mask,
        rng=rng,
        element_id="news_banner",
        element_type="text_block",
        candidate_regions=[(0, 0, min(width, 220), 80), (0, 0, min(width, 280), 120), (0, 0, min(width, width), 120)],
        base_size=(170, 40),
        preview_builder=lambda rect: (lambda img, mask_img: draw_news_banner(img, mask_img, rect, random.Random(4), font_path=font_path)),
        commit_builder=lambda rect, meta: draw_news_banner(canvas, ignore_mask, rect, random.Random(4), font_path=font_path, meta=meta),
        optional=False,
    )
    if banner is not None:
        boxes.append(banner)
    caption = _place_variable(
        manager=manager,
        image=canvas,
        ignore_mask=ignore_mask,
        rng=rng,
        element_id="news_caption",
        element_type="text_block",
        candidate_regions=[(0, max(0, height - 120), width, height), (0, max(0, height - 180), width, height), (0, max(0, height // 2), width, height)],
        base_size=(width, 90),
        preview_builder=lambda rect: (lambda img, mask_img: draw_text_block(img, mask_img, rect, ["Headline overlay", "caption follows"], fill=(0, 0, 0, 210), font_path=font_path)),
        commit_builder=lambda rect, meta: draw_text_block(canvas, ignore_mask, rect, ["Headline overlay", "caption follows"], fill=(0, 0, 0, 210), font_path=font_path, meta=meta),
        optional=False,
    )
    if caption is not None:
        boxes.append(caption)
    choice = rng.randrange(5)
    regions = [(width // 8, height // 8, width * 3 // 4, height * 3 // 4), (width // 5, height // 5, width - 20, height * 4 // 5), (width // 3, 20, width - 20, height // 2)]
    if choice == 0:
        item = _place_variable(manager=manager, image=canvas, ignore_mask=ignore_mask, rng=rng, element_id="news_circle", element_type="sticker", candidate_regions=regions, base_size=(width // 3, height // 3), preview_builder=lambda rect: (lambda img, mask_img: draw_red_circle(img, mask_img, rect)), commit_builder=lambda rect, meta: draw_red_circle(canvas, ignore_mask, rect, meta=meta))
        if item is not None:
            boxes.append(item)
    elif choice == 1:
        item = _place_variable(manager=manager, image=canvas, ignore_mask=ignore_mask, rng=rng, element_id="news_arrow", element_type="sticker", candidate_regions=regions, base_size=(width // 2, height // 4), preview_builder=lambda rect: (lambda img, mask_img: draw_red_arrow(img, mask_img, rect)), commit_builder=lambda rect, meta: draw_red_arrow(canvas, ignore_mask, rect, meta=meta))
        if item is not None:
            boxes.append(item)
    elif choice == 2:
        item = _place_variable(manager=manager, image=canvas, ignore_mask=ignore_mask, rng=rng, element_id="news_rectangle", element_type="sticker", candidate_regions=regions, base_size=(width // 3, height // 3), preview_builder=lambda rect: (lambda img, mask_img: draw_red_rectangle(img, mask_img, rect)), commit_builder=lambda rect, meta: draw_red_rectangle(canvas, ignore_mask, rect, meta=meta))
        if item is not None:
            boxes.append(item)
    elif choice == 3:
        item = _place_variable(manager=manager, image=canvas, ignore_mask=ignore_mask, rng=rng, element_id="news_speech_bubble", element_type="sticker", candidate_regions=regions, base_size=(width // 3, height // 4), preview_builder=lambda rect: (lambda img, mask_img: draw_speech_bubble(img, mask_img, rect, font_path=font_path)), commit_builder=lambda rect, meta: draw_speech_bubble(canvas, ignore_mask, rect, font_path=font_path, meta=meta))
        if item is not None:
            boxes.append(item)
    else:
        item = _place_variable(manager=manager, image=canvas, ignore_mask=ignore_mask, rng=rng, element_id="news_highlight", element_type="sticker", candidate_regions=regions, base_size=(width // 3, height // 4), preview_builder=lambda rect: (lambda img, mask_img: draw_highlight_box(img, mask_img, rect)), commit_builder=lambda rect, meta: draw_highlight_box(canvas, ignore_mask, rect, meta=meta))
        if item is not None:
            boxes.append(item)
    return canvas, canvas_mask, {"template": "news_meme_overlay", "overlay_boxes": boxes, "geometric_transform_meta": {"type": "identity"}, "placement_manager": manager}


def render_annotation_sticker(image: Image.Image, mask: Image.Image | None, ignore_mask: Image.Image, rng: random.Random, font_path: str | None = None, config: Any | None = None) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    canvas = image.copy()
    canvas_mask = None if mask is None else mask.copy()
    manager = PlacementManager(canvas_size=canvas.size, config=config)
    width, height = canvas.size
    boxes = []
    regions = [(0, 0, width, height), (0, 0, width * 3 // 4, height * 3 // 4), (width // 4, height // 4, width, height)]
    choice = rng.randrange(4)
    if choice == 0:
        item = _place_variable(manager=manager, image=canvas, ignore_mask=ignore_mask, rng=rng, element_id="annotation_circle", element_type="sticker", candidate_regions=regions, base_size=(width // 3, height // 3), preview_builder=lambda rect: (lambda img, mask_img: draw_red_circle(img, mask_img, rect)), commit_builder=lambda rect, meta: draw_red_circle(canvas, ignore_mask, rect, meta=meta))
        if item is not None:
            boxes.append(item)
    elif choice == 1:
        item = _place_variable(manager=manager, image=canvas, ignore_mask=ignore_mask, rng=rng, element_id="annotation_arrow", element_type="sticker", candidate_regions=regions, base_size=(width // 2, height // 4), preview_builder=lambda rect: (lambda img, mask_img: draw_red_arrow(img, mask_img, rect)), commit_builder=lambda rect, meta: draw_red_arrow(canvas, ignore_mask, rect, meta=meta))
        if item is not None:
            boxes.append(item)
    elif choice == 2:
        item = _place_variable(manager=manager, image=canvas, ignore_mask=ignore_mask, rng=rng, element_id="annotation_rectangle", element_type="sticker", candidate_regions=regions, base_size=(width // 3, height // 3), preview_builder=lambda rect: (lambda img, mask_img: draw_red_rectangle(img, mask_img, rect)), commit_builder=lambda rect, meta: draw_red_rectangle(canvas, ignore_mask, rect, meta=meta))
        if item is not None:
            boxes.append(item)
    else:
        item1 = _place_variable(manager=manager, image=canvas, ignore_mask=ignore_mask, rng=rng, element_id="annotation_emoji", element_type="sticker", candidate_regions=regions, base_size=(70, 70), preview_builder=lambda rect: (lambda img, mask_img: draw_emoji_face(img, mask_img, rect)), commit_builder=lambda rect, meta: draw_emoji_face(canvas, ignore_mask, rect, meta=meta))
        item2 = _place_variable(manager=manager, image=canvas, ignore_mask=ignore_mask, rng=rng, element_id="annotation_heart", element_type="sticker", candidate_regions=regions, base_size=(60, 60), preview_builder=lambda rect: (lambda img, mask_img: draw_heart(img, mask_img, rect)), commit_builder=lambda rect, meta: draw_heart(canvas, ignore_mask, rect, meta=meta))
        if item1 is not None:
            boxes.append(item1)
        if item2 is not None:
            boxes.append(item2)
    return canvas, canvas_mask, {"template": "annotation_sticker", "overlay_boxes": boxes, "geometric_transform_meta": {"type": "identity"}, "placement_manager": manager}
