"""Dual-scale pre-SNS v3 reporting and class-mask calibration."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

from .pre_sns_v3_model import CLASS_LABELS, FAMILY_LABELS
from .pre_sns_v3_report import _image_tensor, confidence_map, load_v3_model

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER = "PRE_SNS_V3_DUAL_SCALE_REPORT_OK"
CONFIG_OK_MARKER = "PRE_SNS_V3_DUAL_SCALE_REPORT_CONFIG_OK"
APPROVED_KIND = "approved_pre_sns_v3_dual_report"
EXAMPLE_KIND = "example_symbolic"
REMOTE_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}


class DualScaleReportError(ValueError):
    """Raised when dual-scale report inputs are invalid."""


def _err(message: str) -> str:
    return f"- {message}"


def _real(path: str | Path) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def _is_under(path: str | Path, root: str | Path) -> bool:
    try:
        return os.path.commonpath([str(_real(path)), str(_real(root))]) == str(_real(root))
    except ValueError:
        return False


def _inside_repo(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT)


def _has_traversal(value: str) -> bool:
    return any(part == ".." for part in value.replace("\\", "/").split("/"))


def _has_protected_part(value: str, allowed_parts: set[str] | None = None) -> bool:
    allowed_parts = allowed_parts or set()
    return any(part in PROTECTED_PARTS and part not in allowed_parts for part in value.replace("\\", "/").split("/") if part)


def _validate_absolute_path(value: Any, field: str, *, require_file: bool = False, require_dir_parent: bool = False, allow_parts: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, str) or not value.strip():
        return [_err(f"{field} must be a non-empty absolute local path")]
    text = value.strip()
    if REMOTE_RE.search(text):
        errors.append(_err(f"{field} must not be a URL or remote scheme"))
    if WINDOWS_DRIVE_RE.search(text):
        errors.append(_err(f"{field} must not be a Windows drive path"))
    if not text.startswith("/"):
        errors.append(_err(f"{field} must be absolute"))
    if _has_traversal(text):
        errors.append(_err(f"{field} must not contain path traversal"))
    if _has_protected_part(text, allow_parts):
        errors.append(_err(f"{field} must not contain protected path segments"))
    if text.startswith("/") and _inside_repo(text):
        errors.append(_err(f"{field} must be outside the repository"))
    if require_file and not os.path.isfile(text):
        errors.append(_err(f"{field} must exist as a file"))
    if require_dir_parent:
        parent = Path(text).parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not parent.is_dir() or not os.access(parent, os.W_OK):
            errors.append(_err(f"{field} parent must exist and be writable"))
    return errors


def _under_any(path_value: str, roots: list[str]) -> bool:
    return any(_is_under(path_value, root) for root in roots)


def _validate_under_roots(path_value: Any, field: str, roots: list[str], *, require_file: bool = False, allow_parts: set[str] | None = None) -> list[str]:
    errors = _validate_absolute_path(path_value, field, require_file=require_file, allow_parts=allow_parts)
    if isinstance(path_value, str) and path_value.startswith("/") and not _under_any(path_value, roots):
        errors.append(_err(f"{field} must be under an approved input/checkpoint root"))
    return errors


def load_dual_report_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise DualScaleReportError("dual report config root must be a JSON object")
    return raw


def validate_dual_report_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "image_path",
        "long224_checkpoint_path",
        "long256_checkpoint_path",
        "approved_input_roots",
        "approved_output_root",
        "no_download",
        "no_network",
        "no_training",
        "no_checkpoint_writes",
        "no_sns_augmentation",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    kind = raw.get("config_kind")
    if kind not in {APPROVED_KIND, EXAMPLE_KIND}:
        errors.append(_err("config_kind must be approved_pre_sns_v3_dual_report or example_symbolic"))
    for flag in ("no_download", "no_network", "no_training", "no_checkpoint_writes", "no_sns_augmentation"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    roots = raw.get("approved_input_roots")
    if not isinstance(roots, list) or not roots or not all(isinstance(root, str) for root in roots):
        errors.append(_err("approved_input_roots must be a non-empty list of absolute paths"))
        roots = []
    for index, root in enumerate(roots):
        errors.extend(_validate_absolute_path(root, f"approved_input_roots[{index}]", require_dir_parent=False))
    checkpoint_roots = raw.get("approved_checkpoint_roots", [])
    if checkpoint_roots is None:
        checkpoint_roots = []
    if not isinstance(checkpoint_roots, list) or not all(isinstance(root, str) for root in checkpoint_roots):
        errors.append(_err("approved_checkpoint_roots must be a list of absolute paths when present"))
        checkpoint_roots = []
    for index, root in enumerate(checkpoint_roots):
        errors.extend(_validate_absolute_path(root, f"approved_checkpoint_roots[{index}]", require_dir_parent=False, allow_parts={"checkpoints"}))
    approved_roots = list(roots) + list(checkpoint_roots)
    errors.extend(_validate_under_roots(raw.get("image_path"), "image_path", roots, require_file=require_exists))
    errors.extend(_validate_under_roots(raw.get("long224_checkpoint_path"), "long224_checkpoint_path", approved_roots, require_file=require_exists, allow_parts={"checkpoints"}))
    errors.extend(_validate_under_roots(raw.get("long256_checkpoint_path"), "long256_checkpoint_path", approved_roots, require_file=require_exists, allow_parts={"checkpoints"}))
    errors.extend(_validate_absolute_path(raw.get("approved_output_root"), "approved_output_root", require_dir_parent=kind == APPROVED_KIND and require_exists))
    if kind == APPROVED_KIND and raw.get("execution_mode") != "approved_local_pre_sns_v3_dual_report":
        errors.append(_err("execution_mode must be approved_local_pre_sns_v3_dual_report"))
    if kind == EXAMPLE_KIND and raw.get("execution_mode") != "example_only":
        errors.append(_err("example execution_mode must be example_only"))
    for field in (
        "mask_threshold",
        "agreement_iou_threshold",
        "cross_scale_disagreement_iou_threshold",
        "tiny_mask_area_pct_threshold",
        "strong_component_area_pct_threshold",
        "uncertain_tampered_score",
        "very_high_tampered_score",
    ):
        if field in raw:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                errors.append(_err(f"{field} must be a number in [0, 1]"))
    for field in ("min_component_area_px", "keep_top_k_components"):
        if field in raw and raw[field] is not None:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(_err(f"{field} must be a non-negative integer"))
    return errors


def flatten_mask(mask: Any) -> list[float]:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().reshape(-1).tolist()
    elif hasattr(mask, "reshape") and hasattr(mask, "tolist"):
        mask = mask.reshape(-1).tolist()
    flat: list[float] = []
    if isinstance(mask, list):
        for item in mask:
            if isinstance(item, list):
                flat.extend(flatten_mask(item))
            else:
                try:
                    flat.append(float(item))
                except (TypeError, ValueError):
                    flat.append(0.0)
    else:
        try:
            flat.append(float(mask))
        except (TypeError, ValueError):
            pass
    return [value if math.isfinite(value) else 0.0 for value in flat]


def infer_mask_shape(mask: Any, mask_size: tuple[int, int] | int | None = None) -> tuple[int, int]:
    count = len(flatten_mask(mask))
    if isinstance(mask_size, tuple):
        width, height = int(mask_size[0]), int(mask_size[1])
    elif isinstance(mask_size, int):
        width = height = int(mask_size)
    elif isinstance(mask, list) and mask and all(isinstance(row, list) for row in mask):
        height = len(mask)
        width = len(mask[0])
    else:
        side = int(math.sqrt(count))
        if side * side != count:
            raise DualScaleReportError("mask shape is not square; provide mask_size")
        width = height = side
    if width <= 0 or height <= 0 or width * height != count:
        raise DualScaleReportError("mask size does not match mask values")
    return width, height


def binary_mask(mask: Any, threshold: float = 0.5, mask_size: tuple[int, int] | int | None = None) -> tuple[list[int], tuple[int, int]]:
    flat = flatten_mask(mask)
    shape = infer_mask_shape(flat, mask_size)
    return [1 if value >= threshold else 0 for value in flat], shape


def _empty_mask_stats(width: int = 0, height: int = 0, malformed: bool = False, warning: str = "") -> dict[str, Any]:
    return {
        "mask_area_px": 0,
        "mask_area_pct": 0.0,
        "component_count": 0,
        "largest_component_area_px": 0,
        "largest_component_area_pct": 0.0,
        "mask_width": int(max(width, 0)),
        "mask_height": int(max(height, 0)),
        "mask_stats_malformed": bool(malformed),
        "mask_stats_warning": warning,
    }


def _normalize_binary_grid(active: Any, shape: tuple[int, int] | list[int] | None) -> tuple[list[list[int]], int, int, bool, str]:
    malformed = False
    warnings: list[str] = []
    try:
        width = int(shape[0]) if shape and len(shape) >= 1 else 0
        height = int(shape[1]) if shape and len(shape) >= 2 else 0
    except (TypeError, ValueError):
        width, height = 0, 0
        malformed = True
        warnings.append("invalid shape")
    flat = [1 if value else 0 for value in flatten_mask(active)]
    if width <= 0 or height <= 0:
        malformed = True
        warnings.append("non-positive shape")
        return [], max(width, 0), max(height, 0), malformed, "; ".join(warnings)
    expected = width * height
    if expected <= 0:
        malformed = True
        warnings.append("empty shape")
        return [], width, height, malformed, "; ".join(warnings)
    if len(flat) != expected:
        malformed = True
        warnings.append(f"mask length {len(flat)} does not match shape area {expected}")
        if len(flat) < expected:
            flat = flat + [0] * (expected - len(flat))
        else:
            flat = flat[:expected]
    grid = [flat[row * width : (row + 1) * width] for row in range(height)]
    return grid, width, height, malformed, "; ".join(warnings)


def mask_components(active: Any, shape: tuple[int, int]) -> list[list[int]]:
    grid, width, height, _malformed, _warning = _normalize_binary_grid(active, shape)
    if width <= 0 or height <= 0 or not grid:
        return []
    visited = [[False for _ in range(width)] for _ in range(height)]
    components: list[list[int]] = []
    for y in range(height):
        for x in range(width):
            if not grid[y][x] or visited[y][x]:
                continue
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited[y][x] = True
            component: list[int] = []
            while queue:
                current_x, current_y = queue.popleft()
                if current_x < 0 or current_y < 0 or current_x >= width or current_y >= height:
                    continue
                component.append(current_y * width + current_x)
                for nx, ny in ((current_x - 1, current_y), (current_x + 1, current_y), (current_x, current_y - 1), (current_x, current_y + 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if grid[ny][nx] and not visited[ny][nx]:
                        visited[ny][nx] = True
                        queue.append((nx, ny))
            components.append(component)
    return components


def mask_stats(active: Any, shape: tuple[int, int]) -> dict[str, Any]:
    grid, width, height, malformed, warning = _normalize_binary_grid(active, shape)
    if width <= 0 or height <= 0 or not grid:
        return _empty_mask_stats(width, height, malformed=True, warning=warning or "empty or invalid mask")
    total = width * height
    active_count = sum(1 for row in grid for value in row if value)
    if active_count == 0:
        stats = _empty_mask_stats(width, height, malformed, warning)
        stats["mask_stats_malformed"] = bool(malformed)
        stats["mask_stats_warning"] = warning
        return stats
    comps = mask_components([value for row in grid for value in row], (width, height))
    largest = max((len(component) for component in comps), default=0)
    return {
        "mask_area_px": int(active_count),
        "mask_area_pct": float(active_count / total * 100.0),
        "component_count": int(len(comps)),
        "largest_component_area_px": int(largest),
        "largest_component_area_pct": float(largest / total * 100.0),
        "mask_width": int(width),
        "mask_height": int(height),
        "mask_stats_malformed": bool(malformed),
        "mask_stats_warning": warning,
    }


def _neighbor_count(active: list[int], shape: tuple[int, int], x: int, y: int) -> int:
    width, height = shape
    count = 0
    for ny in range(max(0, y - 1), min(height, y + 2)):
        for nx in range(max(0, x - 1), min(width, x + 2)):
            index = ny * width + nx
            if 0 <= index < len(active) and active[index]:
                count += 1
    return count


def _dilate(active: list[int], shape: tuple[int, int]) -> list[int]:
    width, height = shape
    return [1 if _neighbor_count(active, shape, x, y) > 0 else 0 for y in range(height) for x in range(width)]


def _erode(active: list[int], shape: tuple[int, int]) -> list[int]:
    width, height = shape
    result: list[int] = []
    for y in range(height):
        for x in range(width):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                result.append(0)
            else:
                result.append(1 if _neighbor_count(active, shape, x, y) == 9 else 0)
    return result


def postprocess_mask(active: list[int], shape: tuple[int, int], min_component_area_px: int = 0, keep_top_k_components: int | None = None, morphology: str = "none") -> list[int]:
    grid, width, height, _malformed, _warning = _normalize_binary_grid(active, shape)
    processed = [value for row in grid for value in row] if grid else []
    normalized_shape = (width, height)
    if morphology == "open":
        processed = _dilate(_erode(processed, normalized_shape), normalized_shape)
    elif morphology == "close":
        processed = _erode(_dilate(processed, normalized_shape), normalized_shape)
    elif morphology not in {"none", "", None}:
        raise DualScaleReportError("morphology must be none, open, or close")
    comps = sorted(mask_components(processed, normalized_shape), key=len, reverse=True)
    if min_component_area_px > 0:
        comps = [component for component in comps if len(component) >= min_component_area_px]
    if keep_top_k_components is not None and keep_top_k_components > 0:
        comps = comps[:keep_top_k_components]
    kept = [0] * len(processed)
    for component in comps:
        for index in component:
            kept[index] = 1
    return kept


def mask_iou(a: list[int], b: list[int]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    intersection = sum(1 for av, bv in zip(a, b) if av and bv)
    union = sum(1 for av, bv in zip(a, b) if av or bv)
    return 1.0 if union == 0 else intersection / union


def resize_binary_mask_nearest(active: list[int], src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> list[int]:
    if src_shape == dst_shape:
        return [1 if value else 0 for value in active]
    src_w, src_h = int(src_shape[0]), int(src_shape[1])
    dst_w, dst_h = int(dst_shape[0]), int(dst_shape[1])
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0 or len(active) != src_w * src_h:
        return [0] * max(dst_w * dst_h, 0)
    resized: list[int] = []
    for y in range(dst_h):
        src_y = min(src_h - 1, int(y * src_h / dst_h))
        for x in range(dst_w):
            src_x = min(src_w - 1, int(x * src_w / dst_w))
            resized.append(1 if active[src_y * src_w + src_x] else 0)
    return resized


def select_final_mask(long224: dict[str, Any], long256: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or {}
    agreement_threshold = float(cfg.get("agreement_iou_threshold", 0.15))
    disagreement_threshold = float(cfg.get("cross_scale_disagreement_iou_threshold", 0.05))
    uncertain_threshold = float(cfg.get("uncertain_tampered_score", 0.35))
    very_high_threshold = float(cfg.get("very_high_tampered_score", 0.85))
    primary_class = str(long256.get("class", "tampered"))
    primary_score = float(long256.get("tampered_score", 0.0))
    class224 = str(long224.get("class", ""))
    class256 = str(long256.get("class", primary_class))
    mask224 = list(long224.get("processed_mask", []))
    mask256 = list(long256.get("processed_mask", []))
    shape224 = tuple(long224.get("mask_shape") or (0, 0))
    shape256 = tuple(long256.get("mask_shape") or (0, 0))
    shape = tuple(long256.get("mask_shape") or long224.get("mask_shape") or (0, 0))
    comparable224 = resize_binary_mask_nearest(mask224, shape224, shape) if mask224 and shape224 != (0, 0) and shape != (0, 0) else mask224
    comparable256 = resize_binary_mask_nearest(mask256, shape256, shape) if mask256 and shape256 != (0, 0) and shape != (0, 0) else mask256
    active224 = sum(mask224) > 0
    active256 = sum(mask256) > 0
    selected = [0] * len(comparable256 or comparable224)
    source = "none"
    reason = "no active mask selected"
    agreement = 0.0
    uncertain = False
    cross_scale_disagreement = False
    empty_peer_downgrade = False
    if active224 and active256 and len(comparable224) == len(comparable256):
        agreement = mask_iou(comparable224, comparable256)
        if agreement >= agreement_threshold:
            selected = [1 if a or b else 0 for a, b in zip(comparable224, comparable256)]
            source = "dual_scale_union_agreement"
            reason = f"both masks active with agreement IoU {agreement:.4f}"
        elif primary_class == "tampered":
            selected = comparable256
            source = "long256_primary_disagreement"
            uncertain = True
            reason = f"both masks active but agreement IoU {agreement:.4f} is below union threshold {agreement_threshold:.4f}; long256 mask retained as primary"
        if class224 == "tampered" and class256 == "tampered" and agreement < disagreement_threshold:
            cross_scale_disagreement = True
    elif active256 and primary_class == "tampered":
        selected = comparable256
        source = "long256_primary"
        empty_peer_downgrade = True
        reason = "long256 mask selected while long224 mask is empty or absent"
    elif active224 and not active256 and (primary_class == "tampered" or primary_score >= uncertain_threshold):
        selected = comparable224
        source = "long224_fallback_primary_tampered_or_uncertain"
        uncertain = primary_class != "tampered"
        empty_peer_downgrade = True
        reason = "long224 mask selected because long256 mask is empty and long256 class is tampered or uncertain"
    if primary_class != "tampered" and sum(selected) > 0:
        if primary_score >= very_high_threshold:
            uncertain = True
            source = f"{source}_kept_high_tamper_score"
            reason = f"{reason}; non-tampered class retained mask only because tampered score is very high"
        else:
            selected = [0] * len(selected)
            source = "suppressed_non_tampered_mask"
            uncertain = True
            reason = "mask suppressed because long256 class is non-tampered and tampered score is not very high"
    stats = mask_stats(selected, shape) if selected and shape != (0, 0) else {
        "mask_area_px": 0,
        "mask_area_pct": 0.0,
        "component_count": 0,
        "largest_component_area_px": 0,
        "largest_component_area_pct": 0.0,
        "mask_width": int(shape[0]) if shape else 0,
        "mask_height": int(shape[1]) if shape else 0,
    }
    return {
        "mask": selected,
        "mask_shape": shape,
        "source": source,
        "reason": reason,
        "agreement_iou": float(agreement),
        "uncertain": bool(uncertain),
        "active_long224": bool(active224),
        "active_long256": bool(active256),
        "long224_class": class224,
        "long256_class": class256,
        "cross_scale_disagreement": bool(cross_scale_disagreement),
        "empty_peer_downgrade": bool(empty_peer_downgrade),
        "stats": stats,
    }


def class_mask_consistency(primary_class: str, tampered_score: float, selected: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, str | float]:
    cfg = config or {}
    high_threshold = float(cfg.get("very_high_tampered_score", 0.85))
    disagreement_threshold = float(cfg.get("cross_scale_disagreement_iou_threshold", 0.05))
    tiny_area_threshold = float(cfg.get("tiny_mask_area_pct_threshold", 0.15))
    strong_component_threshold = float(cfg.get("strong_component_area_pct_threshold", 0.5))
    area = float(selected.get("stats", {}).get("mask_area_pct", 0.0))
    largest = float(selected.get("stats", {}).get("largest_component_area_pct", 0.0))
    agreement = float(selected.get("agreement_iou", 0.0))
    uncertain = bool(selected.get("uncertain"))
    disputed = bool(selected.get("cross_scale_disagreement"))
    empty_peer = bool(selected.get("empty_peer_downgrade"))
    tiny_low_agreement = area > 0.0 and area < tiny_area_threshold and agreement < disagreement_threshold
    strong_single_scale = largest >= strong_component_threshold and area >= tiny_area_threshold
    disagreement_reason = "none"
    localization_confidence = "none"
    if primary_class == "tampered":
        if area <= 0.0:
            return {
                "class_mask_consistency": "inconsistent",
                "localized_evidence_status": "not_found",
                "final_decision": "tampered_suspect_no_localized_evidence",
                "final_decision_confidence": "medium",
                "localization_confidence": "none",
                "localization_disagreement_reason": "tampered class but no retained localization mask",
            }
        if disputed:
            status = "weak_disputed" if tiny_low_agreement else "disputed"
            return {
                "class_mask_consistency": "cross_scale_mask_disagreement",
                "localized_evidence_status": status,
                "final_decision": "tampered_suspect_disputed_localization",
                "final_decision_confidence": "medium",
                "localization_confidence": "low" if tiny_low_agreement else "medium",
                "localization_disagreement_reason": f"both models predict tampered but processed mask agreement IoU {agreement:.4f} is below {disagreement_threshold:.4f}",
            }
        if empty_peer and not strong_single_scale:
            return {
                "class_mask_consistency": "single_scale_localization_only",
                "localized_evidence_status": "weak_single_scale",
                "final_decision": "tampered_suspect_weak_localization",
                "final_decision_confidence": "medium",
                "localization_confidence": "low",
                "localization_disagreement_reason": "only one scale produced a mask and component quality is not strong enough for high localization confidence",
            }
        if tiny_low_agreement:
            return {
                "class_mask_consistency": "weak_tiny_low_agreement_mask",
                "localized_evidence_status": "weak_disputed",
                "final_decision": "tampered_suspect_weak_localization",
                "final_decision_confidence": "medium",
                "localization_confidence": "low",
                "localization_disagreement_reason": f"final mask area {area:.4f}% is below {tiny_area_threshold:.4f}% with low cross-scale agreement",
            }
        if uncertain:
            return {
                "class_mask_consistency": "partial",
                "localized_evidence_status": "found_uncertain",
                "final_decision": "tampered_suspect_localized_uncertain",
                "final_decision_confidence": "medium",
                "localization_confidence": "medium",
                "localization_disagreement_reason": "mask selection was marked uncertain",
            }
        return {
            "class_mask_consistency": "consistent",
            "localized_evidence_status": "found",
            "final_decision": "tampered_with_localized_evidence",
            "final_decision_confidence": "high",
            "localization_confidence": "high",
            "localization_disagreement_reason": disagreement_reason,
        }
    if area > 0.0 and tampered_score >= high_threshold:
        return {
            "class_mask_consistency": "partial",
            "localized_evidence_status": "suppressed_non_tampered_high_score",
            "final_decision": "non_tampered_class_with_tamper_mask_uncertainty",
            "final_decision_confidence": "low",
            "localization_confidence": "low",
            "localization_disagreement_reason": "non-tampered class retained uncertainty because tampered score is very high",
        }
    if area > 0.0:
        return {
            "class_mask_consistency": "inconsistent",
            "localized_evidence_status": "suppressed_non_tampered_mask",
            "final_decision": "non_tampered_mask_suppressed",
            "final_decision_confidence": "medium",
            "localization_confidence": "none",
            "localization_disagreement_reason": "non-tampered class with active mask suppressed by consistency gate",
        }
    return {
        "class_mask_consistency": "consistent",
        "localized_evidence_status": "not_applicable",
        "final_decision": primary_class,
        "final_decision_confidence": "high" if tampered_score < high_threshold else "medium",
        "localization_confidence": localization_confidence,
        "localization_disagreement_reason": disagreement_reason,
    }


def explanation_reason(primary_class: str, family: str, tampered_score: float, selection: dict[str, Any], gate: dict[str, Any]) -> str:
    area = float(selection.get("stats", {}).get("mask_area_pct", 0.0))
    if gate.get("class_mask_consistency") == "cross_scale_mask_disagreement":
        return "Both models predict tampered, but long224 and long256 localize different regions; localized evidence is disputed."
    if gate.get("localized_evidence_status") == "weak_single_scale":
        return "long256 predicts tampered, but only one scale produced a usable mask; localized evidence is weak."
    if gate.get("localized_evidence_status") == "weak_disputed":
        return "long256 predicts tampered, but the retained mask is tiny and cross-scale agreement is low; localized evidence is weak."
    if gate["final_decision"] == "tampered_with_localized_evidence":
        return f"long256 predicts tampered with localized mask evidence covering {area:.2f}% of the image; family estimate is {family}."
    if gate["final_decision"] == "tampered_suspect_no_localized_evidence":
        return f"long256 predicts tampered, but no reliable localized mask evidence was found; family estimate is {family}."
    if "uncertain" in str(gate["final_decision"]):
        return f"class and mask evidence disagree at tampered score {tampered_score:.3f}; report is marked uncertain."
    return f"long256 predicts {primary_class} and no active tamper mask is retained; family estimate is {family}."


def _runtime_deps():
    try:
        import torch
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("torch and PIL are required for dual-scale v3 reporting") from exc
    return torch, Image, ImageDraw


def _run_one_model(torch: Any, Image: Any, config: dict[str, Any], checkpoint_field: str, label: str) -> dict[str, Any]:
    device = "cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    model, checkpoint, image_size = load_v3_model(torch, config[checkpoint_field], device)
    image = _image_tensor(torch, Image, config["image_path"], image_size, device)
    started = time.perf_counter()
    with torch.no_grad():
        outputs = model(image)
        class_probs = torch.softmax(outputs["class_logits"][0], dim=0).detach().cpu().tolist()
        tamper_probs = torch.softmax(outputs["tamper_binary_logits"][0], dim=0).detach().cpu().tolist()
        family_probs = torch.softmax(outputs["family_logits"][0], dim=0).detach().cpu().tolist()
        mask_probs = torch.sigmoid(outputs["localization_logits"][0]).detach().cpu()
    latency_ms = max((time.perf_counter() - started) * 1000.0, 0.000001)
    raw_mask, shape = binary_mask(mask_probs, float(config.get("mask_threshold", 0.5)), (image_size, image_size))
    processed = postprocess_mask(
        raw_mask,
        shape,
        int(config.get("min_component_area_px", 0)),
        config.get("keep_top_k_components"),
        str(config.get("morphology", "none")),
    )
    class_label = CLASS_LABELS[int(max(range(len(class_probs)), key=lambda idx: class_probs[idx]))]
    family_label = FAMILY_LABELS[int(max(range(len(family_probs)), key=lambda idx: family_probs[idx]))]
    return {
        "scale_label": label,
        "checkpoint_path": config[checkpoint_field],
        "image_size": image_size,
        "class": class_label,
        "class_conf": confidence_map(CLASS_LABELS, class_probs),
        "family": family_label,
        "family_conf": confidence_map(FAMILY_LABELS, family_probs),
        "tampered_score": float(tamper_probs[1]),
        "raw_mask_stats": mask_stats(raw_mask, shape),
        "processed_mask": processed,
        "processed_mask_stats": mask_stats(processed, shape),
        "mask_shape": shape,
        "latency_ms": float(latency_ms),
        "device": device,
        "checkpoint_selected_tau": checkpoint.get("selected_tau"),
    }


def _write_mask_png(Image: Any, mask: list[int], shape: tuple[int, int], path: Path) -> None:
    image = Image.new("L", shape)
    image.putdata([255 if value else 0 for value in mask])
    image.save(path)


def _write_clean_overlay(Image: Any, image_path: str, mask: list[int], shape: tuple[int, int], path: Path, alpha: float = 0.45) -> bool:
    if not mask or sum(mask) == 0:
        return False
    with Image.open(image_path) as original:
        original_rgb = original.convert("RGB")
    mask_image = Image.new("L", shape)
    mask_image.putdata([255 if value else 0 for value in mask])
    resample_nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    mask_resized = mask_image.resize(original_rgb.size, resample_nearest)
    original_pixels = list(original_rgb.getdata())
    mask_pixels = list(mask_resized.getdata())
    blended = []
    for pixel, active in zip(original_pixels, mask_pixels):
        if active:
            blended.append(tuple(int(round((1.0 - alpha) * pixel[i] + alpha * (255 if i == 0 else 0))) for i in range(3)))
        else:
            blended.append(pixel)
    overlay = Image.new("RGB", original_rgb.size)
    overlay.putdata(blended)
    overlay.save(path)
    return True


def _write_comparison_sheet(Image: Any, ImageDraw: Any, image_path: str, artifacts: dict[str, str], path: Path) -> None:
    with Image.open(image_path) as original:
        original_rgb = original.convert("RGB").resize((256, 256))
    panels = [("input", original_rgb)]
    for key, label in (("long224_clean_red_overlay_path", "long224"), ("long256_clean_red_overlay_path", "long256"), ("final_clean_red_overlay_path", "final")):
        if key in artifacts:
            with Image.open(artifacts[key]) as panel:
                panels.append((label, panel.convert("RGB").resize((256, 256))))
    sheet = Image.new("RGB", (256 * len(panels), 286), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(panels):
        x = index * 256
        sheet.paste(panel, (x, 30))
        draw.text((x + 8, 8), label, fill=(0, 0, 0))
    sheet.save(path, quality=92)


def write_visual_artifacts(config: dict[str, Any], long224: dict[str, Any], long256: dict[str, Any], selection: dict[str, Any]) -> dict[str, str | bool]:
    _torch, Image, ImageDraw = _runtime_deps()
    root = _real(config["approved_output_root"])
    if _inside_repo(root):
        raise DualScaleReportError("approved_output_root must be outside repository")
    root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str | bool] = {}
    final_mask_path = root / "final_mask.png"
    _write_mask_png(Image, selection["mask"], selection["mask_shape"], final_mask_path)
    artifacts["final_mask_path"] = str(final_mask_path)
    if _write_clean_overlay(Image, config["image_path"], selection["mask"], selection["mask_shape"], root / "final_clean_red_overlay.png", float(config.get("overlay_alpha", 0.45))):
        artifacts["final_clean_red_overlay_path"] = str(root / "final_clean_red_overlay.png")
    for model_key, model in (("long224", long224), ("long256", long256)):
        overlay_path = root / f"{model_key}_clean_red_overlay.png"
        if _write_clean_overlay(Image, config["image_path"], model["processed_mask"], model["mask_shape"], overlay_path, float(config.get("overlay_alpha", 0.45))):
            artifacts[f"{model_key}_clean_red_overlay_path"] = str(overlay_path)
    sheet_path = root / "dual_scale_comparison_sheet.jpg"
    _write_comparison_sheet(Image, ImageDraw, config["image_path"], {k: str(v) for k, v in artifacts.items()}, sheet_path)
    artifacts["dual_scale_comparison_sheet_path"] = str(sheet_path)
    artifacts["visual_artifacts_written"] = True
    return artifacts


def run_dual_scale_report(config: dict[str, Any]) -> dict[str, Any]:
    errors = validate_dual_report_config(config, require_exists=True)
    if errors:
        raise DualScaleReportError("dual report config validation failed:\n" + "\n".join(errors))
    if config.get("config_kind") != APPROVED_KIND:
        raise DualScaleReportError("dual report execution requires approved_pre_sns_v3_dual_report")
    torch, Image, _ImageDraw = _runtime_deps()
    long224 = _run_one_model(torch, Image, config, "long224_checkpoint_path", "long224")
    long256 = _run_one_model(torch, Image, config, "long256_checkpoint_path", "long256")
    selection = select_final_mask(long224, long256, config)
    gate = class_mask_consistency(long256["class"], long256["tampered_score"], selection, config)
    reason = explanation_reason(long256["class"], long256["family"], long256["tampered_score"], selection, gate)
    primary_class_conf = float(long256.get("class_conf", {}).get(long256["class"], 0.0))
    artifacts: dict[str, Any] = {}
    if config.get("write_visual_artifacts", True) is True:
        artifacts = write_visual_artifacts(config, long224, long256, selection)
    report = {
        "marker": MARKER,
        "schema_version": config.get("schema_version", "1.0"),
        "image_path": config["image_path"],
        "primary_class_model": "long256",
        "long224": {key: value for key, value in long224.items() if key != "processed_mask"},
        "long256": {key: value for key, value in long256.items() if key != "processed_mask"},
        **gate,
        "classification_confidence": primary_class_conf,
        "localization_confidence": gate.get("localization_confidence", "none"),
        "localization_disagreement_reason": gate.get("localization_disagreement_reason", "none"),
        "mask_selection_reason": selection["reason"],
        "explanation_reason": reason,
        "final_mask_area_pct": float(selection["stats"]["mask_area_pct"]),
        "final_mask_stats": selection["stats"],
        "final_mask_source": selection["source"],
        "dual_scale_mask_agreement_iou": selection["agreement_iou"],
        "visual_artifacts": artifacts,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
    }
    if config.get("write_report", True) is True:
        root = _real(config["approved_output_root"])
        root.mkdir(parents=True, exist_ok=True)
        path = root / "pre_sns_v3_dual_scale_report.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        report["report_path"] = str(path)
    return report


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value
