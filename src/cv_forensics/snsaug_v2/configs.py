"""Configuration objects for SNSAug V2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

PROFILES = (
    "clean",
    "jpeg_resize",
    "screenshot_basic",
    "recompression_light",
    "resize_jpeg",
    "screenshot_recapture",
    "blur_color_shift",
    "canvas_9x16_only",
    "canvas_9x16_full_content",
    "platform_ui_same_size",
    "tiktok_like_no_actionbar",
    "instagram_story_no_text_sticker",
    "youtube_shorts_no_actionbar",
    "tiktok_like",
    "instagram_story_like",
    "youtube_shorts_like",
    "news_meme_overlay",
    "ai_badge_overlay",
    "annotation_sticker",
    "combined_sns_realistic",
)
SEVERITIES = ("light", "medium", "strong")


@dataclass
class SNSAugV2Config:
    profile: str
    severity: str = "medium"
    output_size: Optional[int] = None
    seed: Optional[int] = None
    platform_template: Optional[str] = None
    p_recompression: float = 0.7
    p_platform_ui: float = 0.8
    p_ai_badge: float = 0.4
    p_news_banner: float = 0.3
    p_annotation_sticker: float = 0.4
    p_emoji_sticker: float = 0.3
    p_color_shift: float = 0.2
    p_blur_pixelation: float = 0.2
    apply_degradation: bool = False
    apply_recompression: bool = False
    apply_blur: bool = False
    apply_color_shift: bool = False
    apply_screenshot_recapture: bool = False
    text_vs_sticker_max_overlap: float = 0.0
    text_vs_text_max_overlap: float = 0.02
    sticker_vs_sticker_max_overlap: float = 0.05
    badge_vs_text_max_overlap: float = 0.0
    variable_vs_fixed_ui_max_overlap: float = 0.0
    max_placement_attempts: int = 30
    placement_margin_px: int = 8
    max_ignore_mask_area_pct_medium: float = 0.30
    font_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SNSAugV2Result:
    image: Any
    tamper_mask: Any | None
    ignore_mask: Any
    meta: dict[str, Any]


def validate_config(config: SNSAugV2Config) -> None:
    if config.profile not in PROFILES:
        raise ValueError(f"unsupported profile: {config.profile}")
    if config.severity not in SEVERITIES:
        raise ValueError(f"unsupported severity: {config.severity}")
    for name in (
        "p_recompression",
        "p_platform_ui",
        "p_ai_badge",
        "p_news_banner",
        "p_annotation_sticker",
        "p_emoji_sticker",
        "p_color_shift",
        "p_blur_pixelation",
        "text_vs_sticker_max_overlap",
        "text_vs_text_max_overlap",
        "sticker_vs_sticker_max_overlap",
        "badge_vs_text_max_overlap",
        "variable_vs_fixed_ui_max_overlap",
        "max_ignore_mask_area_pct_medium",
    ):
        value = getattr(config, name)
        if not isinstance(value, (int, float)) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if config.output_size is not None and (not isinstance(config.output_size, int) or config.output_size <= 0):
        raise ValueError("output_size must be a positive integer")
    if config.font_path is not None and not isinstance(config.font_path, str):
        raise ValueError("font_path must be a string when provided")
    for name in ("max_placement_attempts", "placement_margin_px"):
        value = getattr(config, name)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name in (
        "apply_degradation",
        "apply_recompression",
        "apply_blur",
        "apply_color_shift",
        "apply_screenshot_recapture",
    ):
        if not isinstance(getattr(config, name), bool):
            raise ValueError(f"{name} must be a boolean")


def load_config(path: str | Path) -> SNSAugV2Config:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("snsaug_v2 config root must be a JSON object")
    config = SNSAugV2Config(**raw)
    validate_config(config)
    return config
