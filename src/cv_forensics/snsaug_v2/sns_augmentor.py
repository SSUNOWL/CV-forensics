"""SNSAug V2 deterministic augmentor."""

from __future__ import annotations

import random
from typing import Any

from PIL import Image

from .configs import SNSAugV2Config, SNSAugV2Result, validate_config
from .platform_templates import (
    render_annotation_sticker,
    render_instagram_story_like,
    render_news_meme_overlay,
    render_tiktok_like,
    render_youtube_shorts_like,
)
from .sticker_packs import draw_ai_badge, draw_emoji_face, draw_heart
from .transforms import (
    apply_blur_pixelation,
    apply_color_shift,
    blank_mask,
    ensure_mask,
    ensure_rgb,
    fit_content_to_canvas,
    recompress_jpeg,
    resize_long_side,
    severity_values,
)


def _float_seed(seed: Any) -> int:
    if seed is None:
        return 0
    if isinstance(seed, int):
        return seed
    text = str(seed)
    return sum((index + 1) * ord(ch) for index, ch in enumerate(text))


class SNSAugV2Augmentor:
    def __init__(self, config: SNSAugV2Config):
        validate_config(config)
        self.config = config

    def _profile_seed(self, base_id: Any, seed: Any | None) -> int:
        if seed is not None:
            return _float_seed(seed)
        if self.config.seed is not None:
            return _float_seed(self.config.seed) + _float_seed(base_id)
        return _float_seed(base_id)

    def __call__(self, image, tamper_mask=None, label=None, base_id=None, seed=None) -> SNSAugV2Result:
        image = ensure_rgb(image)
        tamper_mask = ensure_mask(tamper_mask, image.size)
        ignore_mask = blank_mask(image.size)
        rng = random.Random(self._profile_seed(base_id, seed))
        params = severity_values(self.config.severity)
        overlay_boxes: list[dict[str, Any]] = []
        geom_meta: dict[str, Any] = {"type": "identity"}
        transforms_applied: list[str] = []

        working_image = image.copy()
        working_mask = None if tamper_mask is None else tamper_mask.copy()

        if self.config.profile == "clean":
            pass
        elif self.config.profile == "jpeg_resize":
            working_image, working_mask, geom_meta = resize_long_side(working_image, working_mask, self.config.output_size or params["resize_long"])
            working_image = recompress_jpeg(working_image, params["jpeg_quality"])
            transforms_applied.extend(["resize_long_side", "jpeg_recompress"])
        elif self.config.profile == "screenshot_basic":
            working_image, working_mask, geom_meta = fit_content_to_canvas(working_image, working_mask, (working_image.size[0], int(round(working_image.size[1] * 1.08))), background=(32, 32, 32))
            ignore_mask = blank_mask(working_image.size)
            from .ui_renderers import draw_labeled_chip

            overlay_boxes.append(draw_labeled_chip(working_image, ignore_mask, (8, 6, 110, 30), "12:41", fill=(0, 0, 0, 220)))
            overlay_boxes.append(draw_labeled_chip(working_image, ignore_mask, (8, working_image.size[1] - 34, 180, working_image.size[1] - 8), "screenshot bar", fill=(0, 0, 0, 220)))
            transforms_applied.append("screenshot_frame")
        elif self.config.profile in {"tiktok_like", "instagram_story_like", "youtube_shorts_like"}:
            ignore_mask = blank_mask((540, 960))
            renderer = {
                "tiktok_like": render_tiktok_like,
                "instagram_story_like": render_instagram_story_like,
                "youtube_shorts_like": render_youtube_shorts_like,
            }[self.config.profile]
            working_image, working_mask, meta = renderer(image, tamper_mask, ignore_mask, rng, self.config.font_path)
            overlay_boxes.extend(meta["overlay_boxes"])
            geom_meta = meta["geometric_transform_meta"]
            transforms_applied.append(meta["template"])
        elif self.config.profile == "news_meme_overlay":
            ignore_mask = blank_mask(image.size)
            working_image, working_mask, meta = render_news_meme_overlay(image, tamper_mask, ignore_mask, rng, self.config.font_path)
            overlay_boxes.extend(meta["overlay_boxes"])
            geom_meta = meta["geometric_transform_meta"]
            transforms_applied.append(meta["template"])
        elif self.config.profile == "ai_badge_overlay":
            ignore_mask = blank_mask(image.size)
            width, _height = image.size
            working_image = image.copy()
            working_mask = None if tamper_mask is None else tamper_mask.copy()
            overlay_boxes.append(draw_ai_badge(working_image, ignore_mask, (max(0, width - 190), 16, width - 16, 54), rng, self.config.font_path))
            transforms_applied.append("ai_badge")
        elif self.config.profile == "annotation_sticker":
            ignore_mask = blank_mask(image.size)
            working_image, working_mask, meta = render_annotation_sticker(image, tamper_mask, ignore_mask, rng, self.config.font_path)
            overlay_boxes.extend(meta["overlay_boxes"])
            geom_meta = meta["geometric_transform_meta"]
            transforms_applied.append(meta["template"])
        elif self.config.profile == "combined_sns_realistic":
            ignore_mask = blank_mask((540, 960))
            template_name = self.config.platform_template or ("tiktok_like" if rng.random() < 0.34 else "instagram_story_like" if rng.random() < 0.67 else "youtube_shorts_like")
            renderer = {
                "tiktok_like": render_tiktok_like,
                "instagram_story_like": render_instagram_story_like,
                "youtube_shorts_like": render_youtube_shorts_like,
            }[template_name]
            working_image, working_mask, meta = renderer(image, tamper_mask, ignore_mask, rng, self.config.font_path)
            overlay_boxes.extend(meta["overlay_boxes"])
            geom_meta = meta["geometric_transform_meta"]
            transforms_applied.append(template_name)
            if rng.random() < self.config.p_ai_badge:
                overlay_boxes.append(draw_ai_badge(working_image, ignore_mask, (16, 710, 190, 748), rng, self.config.font_path))
            if rng.random() < self.config.p_emoji_sticker:
                overlay_boxes.append(draw_emoji_face(working_image, ignore_mask, (24, 92, 88, 156)))
            if rng.random() < self.config.p_annotation_sticker:
                overlay_boxes.append(draw_heart(working_image, ignore_mask, (428, 120, 500, 192)))
            if rng.random() < self.config.p_news_banner:
                from .sticker_packs import draw_news_banner

                overlay_boxes.append(draw_news_banner(working_image, ignore_mask, (0, 0, 170, 40), rng, self.config.font_path))
        else:
            raise ValueError(f"unsupported profile: {self.config.profile}")
        if self.config.output_size and working_image.size[0] != self.config.output_size and working_image.size[1] != self.config.output_size:
            target = self.config.output_size
            working_image, working_mask, extra_geom = resize_long_side(working_image, working_mask, target)
            ignore_mask = ignore_mask.resize(working_image.size, Image.Resampling.NEAREST)
            geom_meta = {"base": geom_meta, "post_resize": extra_geom}
            transforms_applied.append("output_resize")
        if self.config.profile != "clean":
            if rng.random() < self.config.p_recompression:
                working_image = recompress_jpeg(working_image, params["jpeg_quality"])
                transforms_applied.append("jpeg_recompress")
            if rng.random() < self.config.p_color_shift:
                working_image = apply_color_shift(working_image, rng, params["color_factor"])
                transforms_applied.append("color_shift")
            if rng.random() < self.config.p_blur_pixelation:
                working_image = apply_blur_pixelation(working_image, params["blur_radius"], params["pixel_step"])
                transforms_applied.append("blur_pixelation")
        meta = {
            "profile": self.config.profile,
            "severity": self.config.severity,
            "seed": self._profile_seed(base_id, seed),
            "label": label,
            "base_id": base_id,
            "label_preserved": True,
            "overlay_boxes": overlay_boxes,
            "geometric_transform_meta": geom_meta,
            "transforms_applied": transforms_applied,
            "platform_template": self.config.platform_template,
        }
        return SNSAugV2Result(image=working_image, tamper_mask=working_mask, ignore_mask=ignore_mask, meta=meta)
