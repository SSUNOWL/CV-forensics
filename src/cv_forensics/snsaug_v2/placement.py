"""Deterministic alpha-mask-based placement collision guard for SNSAug V2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image, ImageFilter

from .ui_renderers import clamp_rect, place_in_region, rect_to_box

PLACEMENT_POLICY_VERSION = "snsaug_v2_collision_guard_v1"


@dataclass
class PlacedOverlay:
    element_id: str
    element_type: str
    bbox: tuple[int, int, int, int]
    alpha_mask: Image.Image
    candidate_region: tuple[int, int, int, int] | None
    metadata: dict[str, Any]


@dataclass
class PlacementDecision:
    accepted: bool
    bbox: tuple[int, int, int, int] | None
    attempt_count: int
    rejected_reason: str | None
    max_overlap_ratio: float
    metadata: dict[str, Any]


def _overlay_area(alpha_mask: Image.Image) -> int:
    return int(sum(1 for value in alpha_mask.getdata() if value > 0))


def _expanded_mask(mask: Image.Image, margin_px: int) -> Image.Image:
    if margin_px <= 0:
        return mask
    size = max(3, margin_px * 2 + 1)
    if size % 2 == 0:
        size += 1
    return mask.filter(ImageFilter.MaxFilter(size=size))


def _intersection_area(left: Image.Image, right: Image.Image) -> int:
    overlap = Image.new("L", left.size, 0)
    overlap.paste(left)
    overlap = Image.eval(ImageChops.multiply(overlap, right), lambda value: 255 if value > 0 else 0)
    return _overlay_area(overlap)


def _image_chops_multiply(left: Image.Image, right: Image.Image) -> Image.Image:
    # local import to avoid expanding module-level surface for tests
    from PIL import ImageChops

    return ImageChops.multiply(left, right)


class PlacementManager:
    def __init__(self, *, canvas_size: tuple[int, int], config: Any):
        self.canvas_size = canvas_size
        self.config = config
        self.placed: list[PlacedOverlay] = []
        self.retries_total = 0
        self.skipped_count = 0

    def _group(self, element_type: str) -> str:
        text_like = {"text_block", "chip"}
        badge_like = {"badge", "ai_badge"}
        fixed_ui = {"progress_bar", "bottom_nav", "icon_comment", "icon_share", "icon_circle", "icon_heart", "icon_bookmark", "icon_play"}
        if element_type in text_like:
            return "text"
        if element_type in badge_like:
            return "badge"
        if element_type in fixed_ui or element_type.startswith("fixed_ui"):
            return "fixed_ui"
        return "sticker"

    def _threshold(self, left_type: str, right_type: str) -> float:
        left = self._group(left_type)
        right = self._group(right_type)
        groups = {left, right}
        if "fixed_ui" in groups:
            return float(self.config.variable_vs_fixed_ui_max_overlap)
        if groups == {"text"}:
            return float(self.config.text_vs_text_max_overlap)
        if groups == {"sticker"}:
            return float(self.config.sticker_vs_sticker_max_overlap)
        if groups == {"text", "badge"}:
            return float(self.config.badge_vs_text_max_overlap)
        if groups == {"text", "sticker"}:
            return float(self.config.text_vs_sticker_max_overlap)
        return float(self.config.sticker_vs_sticker_max_overlap)

    def register_existing(
        self,
        *,
        element_id: str,
        element_type: str,
        bbox: tuple[int, int, int, int],
        alpha_mask: Image.Image,
        candidate_region: tuple[int, int, int, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.placed.append(
            PlacedOverlay(
                element_id=element_id,
                element_type=element_type,
                bbox=bbox,
                alpha_mask=_expanded_mask(alpha_mask, int(self.config.placement_margin_px)),
                candidate_region=candidate_region,
                metadata=metadata or {},
            )
        )

    def _evaluate(self, element_type: str, alpha_mask: Image.Image) -> tuple[bool, float, list[str]]:
        candidate_area = max(1, _overlay_area(alpha_mask))
        expanded = _expanded_mask(alpha_mask, int(self.config.placement_margin_px))
        max_ratio = 0.0
        overlapped_with: list[str] = []
        for placed in self.placed:
            overlap = _overlay_area(_image_chops_multiply(expanded, placed.alpha_mask))
            if overlap <= 0:
                continue
            existing_area = max(1, _overlay_area(placed.alpha_mask))
            candidate_ratio = float(overlap) / float(candidate_area)
            existing_ratio = float(overlap) / float(existing_area)
            ratio = max(candidate_ratio, existing_ratio)
            max_ratio = max(max_ratio, ratio)
            if ratio > self._threshold(element_type, placed.element_type):
                overlapped_with.append(placed.element_id)
        return len(overlapped_with) == 0, max_ratio, overlapped_with

    def place_variable(
        self,
        *,
        element_id: str,
        element_type: str,
        candidate_regions: list[tuple[int, int, int, int]],
        base_size: tuple[int, int],
        rng: Any,
        render_preview: Callable[[tuple[int, int, int, int]], tuple[Image.Image, dict[str, Any]]],
        optional: bool = True,
    ) -> PlacementDecision:
        attempts = max(1, int(self.config.max_placement_attempts))
        rejected = 0
        best_ratio = 1.0
        best_meta: dict[str, Any] = {}
        overlapped_ids: list[str] = []
        for attempt in range(1, attempts + 1):
            region = candidate_regions[(attempt - 1) % max(1, len(candidate_regions))]
            rect, placement_meta = place_in_region(region, base_size, self.canvas_size, rng)
            rect = clamp_rect(rect, self.canvas_size)
            alpha_mask, preview_meta = render_preview(rect)
            accepted, ratio, overlap_ids = self._evaluate(element_type, alpha_mask)
            best_ratio = min(best_ratio, ratio)
            combined_meta = dict(placement_meta)
            combined_meta.update(preview_meta)
            combined_meta["candidate_region"] = rect_to_box(region)
            combined_meta["attempt_count"] = attempt
            combined_meta["overlapped_with"] = list(overlap_ids)
            if accepted:
                self.retries_total += max(0, attempt - 1)
                return PlacementDecision(
                    accepted=True,
                    bbox=rect,
                    attempt_count=attempt,
                    rejected_reason=None,
                    max_overlap_ratio=ratio,
                    metadata=combined_meta,
                )
            rejected += 1
            overlapped_ids = list(overlap_ids)
            best_meta = combined_meta
        self.retries_total += max(0, attempts - 1)
        if optional:
            self.skipped_count += 1
            best_meta["skipped_due_to_overlap"] = True
            best_meta["rejected_candidates"] = rejected
            return PlacementDecision(
                accepted=False,
                bbox=None,
                attempt_count=attempts,
                rejected_reason="overlap_threshold_exceeded",
                max_overlap_ratio=best_ratio,
                metadata=best_meta | {"overlapped_with": overlapped_ids},
            )
        fallback_region = candidate_regions[0]
        rect, placement_meta = place_in_region(fallback_region, base_size, self.canvas_size, rng, size_jitter=0.0, pos_jitter=0.0)
        return PlacementDecision(
            accepted=True,
            bbox=rect,
            attempt_count=attempts,
            rejected_reason="fallback_after_overlap",
            max_overlap_ratio=best_ratio,
            metadata=placement_meta | {
                "candidate_region": rect_to_box(fallback_region),
                "overlapped_with": overlapped_ids,
                "rejected_candidates": rejected,
            },
        )
