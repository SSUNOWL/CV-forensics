"""High-resolution crop/tile localization planning and evaluation for pre-SNS v3."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from .pre_sns_v3_dual_scale_report import (
    _err,
    _inside_repo,
    _is_under,
    _real,
    _validate_absolute_path,
    _validate_under_roots,
    mask_stats,
)
from .pre_sns_v3_gt_iou_mining import dice_score, iou_score

MARKER = "PRE_SNS_V3_TILE_LOCALIZATION_OK"
CONFIG_OK_MARKER = "PRE_SNS_V3_TILE_LOCALIZATION_CONFIG_OK"
APPROVED_KIND = "approved_pre_sns_v3_tile_localization"
EXAMPLE_KIND = "example_symbolic"


class TileLocalizationError(ValueError):
    """Raised when tile-localization inputs or configs are invalid."""


def load_tile_localization_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TileLocalizationError("tile localization config root must be a JSON object")
    return raw


def validate_tile_localization_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "manifest_path",
        "long256_checkpoint_path",
        "approved_input_roots",
        "output_root",
        "max_samples",
        "tile_size",
        "tile_stride",
        "crop_context_px",
        "high_res_max_size",
        "tampered_suspect_threshold",
        "no_download",
        "no_network",
        "no_training",
        "dry_run_only",
        "no_checkpoint_writes",
        "no_sns_augmentation",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    kind = raw.get("config_kind")
    if kind not in {APPROVED_KIND, EXAMPLE_KIND}:
        errors.append(_err(f"config_kind must be {APPROVED_KIND} or {EXAMPLE_KIND}"))
    for flag in ("no_download", "no_network", "no_training", "dry_run_only", "no_checkpoint_writes", "no_sns_augmentation"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))

    roots = raw.get("approved_input_roots")
    if not isinstance(roots, list) or not roots or not all(isinstance(root, str) for root in roots):
        errors.append(_err("approved_input_roots must be a non-empty list of absolute paths"))
        roots = []
    for index, root in enumerate(roots):
        errors.extend(_validate_absolute_path(root, f"approved_input_roots[{index}]"))

    checkpoint_roots = raw.get("approved_checkpoint_roots", [])
    if checkpoint_roots is None:
        checkpoint_roots = []
    if not isinstance(checkpoint_roots, list) or not all(isinstance(root, str) for root in checkpoint_roots):
        errors.append(_err("approved_checkpoint_roots must be a list when present"))
        checkpoint_roots = []
    for index, root in enumerate(checkpoint_roots):
        errors.extend(_validate_absolute_path(root, f"approved_checkpoint_roots[{index}]", allow_parts={"checkpoints"}))
    approved_model_roots = list(roots) + list(checkpoint_roots)
    errors.extend(_validate_under_roots(raw.get("manifest_path"), "manifest_path", roots, require_file=require_exists))
    if raw.get("gt_iou_mining_records_path"):
        errors.extend(_validate_under_roots(raw.get("gt_iou_mining_records_path"), "gt_iou_mining_records_path", roots, require_file=require_exists))
    errors.extend(_validate_under_roots(raw.get("long256_checkpoint_path"), "long256_checkpoint_path", approved_model_roots, require_file=require_exists, allow_parts={"checkpoints"}))

    output_root = raw.get("output_root")
    errors.extend(_validate_absolute_path(output_root, "output_root", require_dir_parent=kind == APPROVED_KIND and require_exists))
    if isinstance(output_root, str) and output_root.startswith("/") and any(_is_under(output_root, root) for root in roots):
        errors.append(_err("output_root must not be under an approved input root"))

    if isinstance(raw.get("max_samples"), bool) or not isinstance(raw.get("max_samples"), int) or raw.get("max_samples", 0) <= 0:
        errors.append(_err("max_samples must be a positive integer"))
    for field in ("tile_size", "tile_stride", "crop_context_px", "high_res_max_size"):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(_err(f"{field} must be a positive integer"))
    for field in ("visual_top_n_per_bucket", "preview_tile_count"):
        if field in raw and raw[field] is not None:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(_err(f"{field} must be a non-negative integer"))
    for field in ("tampered_suspect_threshold", "improvement_epsilon"):
        if field in raw:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
                errors.append(_err(f"{field} must be a non-negative number"))
    if raw.get("merge_mode", "union") not in {"union", "max", "weighted_average"}:
        errors.append(_err("merge_mode must be union, max, or weighted_average"))
    if kind == APPROVED_KIND and raw.get("execution_mode") != "approved_local_pre_sns_v3_tile_localization":
        errors.append(_err("execution_mode must be approved_local_pre_sns_v3_tile_localization"))
    if kind == EXAMPLE_KIND and raw.get("execution_mode") != "example_only":
        errors.append(_err("example execution_mode must be example_only"))
    return errors


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise TileLocalizationError("manifest must be a JSON object with a samples list")
    return manifest


def load_gt_iou_records(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
    return records


def tile_grid(width: int, height: int, tile_size: int, tile_stride: int) -> list[dict[str, int]]:
    if width <= 0 or height <= 0 or tile_size <= 0 or tile_stride <= 0:
        raise TileLocalizationError("width, height, tile_size, and tile_stride must be positive")
    xs = list(range(0, max(width - tile_size + 1, 1), tile_stride))
    ys = list(range(0, max(height - tile_size + 1, 1), tile_stride))
    if not xs or xs[-1] != max(width - tile_size, 0):
        xs.append(max(width - tile_size, 0))
    if not ys or ys[-1] != max(height - tile_size, 0):
        ys.append(max(height - tile_size, 0))
    tiles: list[dict[str, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for y in ys:
        for x in xs:
            tile = {
                "x0": int(x),
                "y0": int(y),
                "x1": int(min(x + tile_size, width)),
                "y1": int(min(y + tile_size, height)),
            }
            key = (tile["x0"], tile["y0"], tile["x1"], tile["y1"])
            if key not in seen:
                tiles.append(tile)
                seen.add(key)
    return tiles


def expand_box(box: dict[str, int], width: int, height: int, context_px: int) -> dict[str, int]:
    return {
        "x0": max(0, int(box["x0"]) - context_px),
        "y0": max(0, int(box["y0"]) - context_px),
        "x1": min(width, int(box["x1"]) + context_px),
        "y1": min(height, int(box["y1"]) + context_px),
    }


def mask_bounding_box(mask: list[int], shape: tuple[int, int]) -> dict[str, int] | None:
    width, height = shape
    points = [(index % width, index // width) for index, value in enumerate(mask) if value]
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {"x0": min(xs), "y0": min(ys), "x1": max(xs) + 1, "y1": max(ys) + 1}


def hint_boxes_from_record(record: dict[str, Any] | None, width: int, height: int, context_px: int) -> list[dict[str, int]]:
    if not record:
        return []
    boxes: list[dict[str, int]] = []
    for key in ("bbox", "gt_bbox", "mask_bbox", "failure_bbox"):
        value = record.get(key)
        if isinstance(value, dict) and all(name in value for name in ("x0", "y0", "x1", "y1")):
            boxes.append(expand_box({name: int(value[name]) for name in ("x0", "y0", "x1", "y1")}, width, height, context_px))
    mask = record.get("gt_mask")
    shape_value = record.get("gt_shape")
    if isinstance(mask, list) and isinstance(shape_value, dict):
        shape = (int(shape_value.get("width", width)), int(shape_value.get("height", height)))
        box = mask_bounding_box([1 if value else 0 for value in mask], shape)
        if box:
            boxes.append(expand_box(box, width, height, context_px))
    return boxes


def activation_decision(primary_class: str, tampered_score: float, suspect_threshold: float) -> dict[str, Any]:
    if primary_class == "tampered":
        return {"activate": True, "status": "activated_tampered", "uncertain": False}
    if tampered_score >= suspect_threshold:
        return {"activate": True, "status": "activated_tampered_suspect", "uncertain": True}
    return {"activate": False, "status": "skipped_non_tampered_below_threshold", "uncertain": False}


def merge_tile_masks(width: int, height: int, tile_predictions: list[dict[str, Any]], merge_mode: str = "union", threshold: float = 0.5) -> list[int]:
    total = width * height
    accum = [0.0] * total
    counts = [0] * total
    for pred in tile_predictions:
        tile = pred["tile"]
        mask = pred.get("mask", [])
        tw = int(tile["x1"]) - int(tile["x0"])
        th = int(tile["y1"]) - int(tile["y0"])
        if tw <= 0 or th <= 0 or len(mask) != tw * th:
            continue
        for ty in range(th):
            for tx in range(tw):
                full_index = (int(tile["y0"]) + ty) * width + int(tile["x0"]) + tx
                tile_index = ty * tw + tx
                value = 1.0 if mask[tile_index] else 0.0
                if merge_mode in {"union", "max"}:
                    accum[full_index] = max(accum[full_index], value)
                elif merge_mode == "weighted_average":
                    accum[full_index] += value
                    counts[full_index] += 1
                else:
                    raise TileLocalizationError("unsupported merge mode")
    if merge_mode == "weighted_average":
        return [1 if counts[index] and (accum[index] / counts[index]) >= threshold else 0 for index in range(total)]
    return [1 if value >= threshold else 0 for value in accum]


def compare_masks(gt_mask: list[int] | None, baseline_mask: list[int], tile_mask: list[int], shape: tuple[int, int], epsilon: float = 0.0) -> dict[str, Any]:
    baseline_stats = mask_stats(baseline_mask, shape)
    tile_stats = mask_stats(tile_mask, shape)
    result: dict[str, Any] = {
        "baseline_area_pct": float(baseline_stats["mask_area_pct"]),
        "tile_area_pct": float(tile_stats["mask_area_pct"]),
        "baseline_component_count": int(baseline_stats["component_count"]),
        "tile_component_count": int(tile_stats["component_count"]),
        "baseline_largest_component_area_pct": float(baseline_stats["largest_component_area_pct"]),
        "tile_largest_component_area_pct": float(tile_stats["largest_component_area_pct"]),
    }
    if gt_mask is None:
        result.update({"baseline_iou": None, "tile_iou": None, "baseline_dice": None, "tile_dice": None, "localization_delta": "unknown_no_gt"})
        return result
    baseline_iou = iou_score(gt_mask, baseline_mask)
    tile_iou = iou_score(gt_mask, tile_mask)
    result.update({
        "baseline_iou": baseline_iou,
        "tile_iou": tile_iou,
        "baseline_dice": dice_score(gt_mask, baseline_mask),
        "tile_dice": dice_score(gt_mask, tile_mask),
    })
    if tile_iou > baseline_iou + epsilon:
        result["localization_delta"] = "improved"
    elif tile_iou + epsilon < baseline_iou:
        result["localization_delta"] = "regressed"
    else:
        result["localization_delta"] = "unchanged"
    return result


def _sample_id(sample: dict[str, Any], index: int) -> str:
    raw = str(sample.get("sample_id") or sample.get("id") or f"sample_{index:06d}")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)[:80]


def _sample_size(sample: dict[str, Any], config: dict[str, Any]) -> tuple[int, int]:
    width = int(sample.get("width") or sample.get("image_width") or config.get("high_res_max_size", 512))
    height = int(sample.get("height") or sample.get("image_height") or config.get("high_res_max_size", 512))
    limit = int(config.get("high_res_max_size", max(width, height)))
    if max(width, height) > limit:
        scale = limit / max(width, height)
        width = max(1, int(round(width * scale)))
        height = max(1, int(round(height * scale)))
    return width, height


def _records_by_sample_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get("sample_id")): record for record in records if record.get("sample_id") is not None}


def _selected_failure_buckets(record: dict[str, Any] | None) -> list[str]:
    if not record:
        return []
    return [str(item) for item in record.get("failure_types", []) if item]


def build_tile_training_plan(config: dict[str, Any], manifest: dict[str, Any], gt_iou_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    records_by_id = _records_by_sample_id(gt_iou_records or [])
    sample_plans: list[dict[str, Any]] = []
    max_samples = int(config.get("max_samples", 0))
    for index, sample in enumerate(manifest.get("samples", [])):
        if len(sample_plans) >= max_samples:
            break
        if not isinstance(sample, dict):
            continue
        width, height = _sample_size(sample, config)
        sample_id = _sample_id(sample, index)
        mining_record = records_by_id.get(sample_id)
        tiles = tile_grid(width, height, int(config["tile_size"]), int(config["tile_stride"]))
        hint_boxes = hint_boxes_from_record(mining_record, width, height, int(config["crop_context_px"]))
        sample_plans.append({
            "sample_id": sample_id,
            "image_path": sample.get("image_path"),
            "width": width,
            "height": height,
            "tile_count": len(tiles),
            "hint_box_count": len(hint_boxes),
            "estimated_pixel_coverage": int(sum((tile["x1"] - tile["x0"]) * (tile["y1"] - tile["y0"]) for tile in tiles)),
            "failure_buckets": _selected_failure_buckets(mining_record),
            "hint_boxes_preview": hint_boxes[: int(config.get("preview_tile_count", 4))],
            "tiles_preview": tiles[: int(config.get("preview_tile_count", 4))],
        })
    bucket_counts: dict[str, int] = {}
    for plan in sample_plans:
        for bucket in plan["failure_buckets"]:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    return {
        "marker": MARKER,
        "planned_sample_count": len(sample_plans),
        "total_tile_count": sum(plan["tile_count"] for plan in sample_plans),
        "failure_bucket_counts": bucket_counts,
        "tile_config": tile_config(config),
        "sample_plans": sample_plans,
        "no_training": True,
        "dry_run_only": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
    }


def tile_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "tile_size": int(config.get("tile_size", 0)),
        "tile_stride": int(config.get("tile_stride", 0)),
        "crop_context_px": int(config.get("crop_context_px", 0)),
        "high_res_max_size": int(config.get("high_res_max_size", 0)),
        "merge_mode": str(config.get("merge_mode", "union")),
        "tampered_suspect_threshold": float(config.get("tampered_suspect_threshold", 0.5)),
    }


def _write_json(path: Path, value: Any) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True))
            handle.write("\n")
    return str(path)


def _safe_output_root(config: dict[str, Any]) -> Path:
    root = _real(config["output_root"])
    if _inside_repo(root):
        raise TileLocalizationError("output_root must be outside repository")
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_tile_training_prepare(config: dict[str, Any]) -> dict[str, Any]:
    errors = validate_tile_localization_config(config, require_exists=True)
    if errors:
        raise TileLocalizationError("tile localization config validation failed:\n" + "\n".join(errors))
    if config.get("config_kind") != APPROVED_KIND:
        raise TileLocalizationError(f"execution requires {APPROVED_KIND}")
    manifest = load_manifest(config["manifest_path"])
    gt_records = load_gt_iou_records(config.get("gt_iou_mining_records_path"))
    plan = build_tile_training_plan(config, manifest, gt_records)
    root = _safe_output_root(config)
    output_paths = {
        "tile_training_plan_summary": _write_json(root / "tile_training_plan_summary.json", {key: value for key, value in plan.items() if key != "sample_plans"}),
        "tile_training_sample_plan": _write_json(root / "tile_training_sample_plan.json", plan["sample_plans"]),
    }
    summary = {key: value for key, value in plan.items() if key != "sample_plans"}
    summary.update({
        "no_download": True,
        "no_network": True,
        "output_paths": output_paths,
    })
    _write_json(root / "tile_training_plan_summary.json", summary)
    return summary


def _mask_from_sample(sample: dict[str, Any], key: str, width: int, height: int) -> list[int]:
    value = sample.get(key)
    if isinstance(value, list):
        flat: list[int] = []
        for item in value:
            if isinstance(item, list):
                flat.extend(1 if child else 0 for child in item)
            else:
                flat.append(1 if item else 0)
        if len(flat) == width * height:
            return flat
    return [0] * (width * height)


def _simulated_tile_predictions(width: int, height: int, tiles: list[dict[str, int]], sample: dict[str, Any]) -> list[dict[str, Any]]:
    source = _mask_from_sample(sample, "tile_mask", width, height)
    predictions: list[dict[str, Any]] = []
    for tile in tiles:
        tw = tile["x1"] - tile["x0"]
        th = tile["y1"] - tile["y0"]
        mask: list[int] = []
        for y in range(tile["y0"], tile["y1"]):
            for x in range(tile["x0"], tile["x1"]):
                mask.append(source[y * width + x])
        if len(mask) == tw * th:
            predictions.append({"tile": tile, "mask": mask})
    return predictions


def _overlay_mask(Image: Any, base: Any, mask: list[int], shape: tuple[int, int], color: tuple[int, int, int]) -> Any:
    if not mask or sum(mask) == 0:
        return base.copy()
    mask_image = Image.new("L", shape)
    mask_image.putdata([255 if value else 0 for value in mask])
    resample_nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    mask_image = mask_image.resize(base.size, resample_nearest)
    pixels = list(base.getdata())
    mask_pixels = list(mask_image.getdata())
    blended = []
    alpha = 0.45
    for pixel, active in zip(pixels, mask_pixels):
        if active:
            blended.append(tuple(int(round((1.0 - alpha) * pixel[index] + alpha * color[index])) for index in range(3)))
        else:
            blended.append(pixel)
    panel = Image.new("RGB", base.size)
    panel.putdata(blended)
    return panel


def _write_visual_sheet_for_record(output_root: Path, record: dict[str, Any], sample: dict[str, Any], width: int, height: int, baseline_mask: list[int], tile_mask: list[int], gt_mask: list[int] | None) -> str | None:
    image_path = sample.get("image_path")
    if not image_path:
        return None
    try:
        from PIL import Image, ImageDraw
        with Image.open(image_path) as image:
            base = image.convert("RGB").resize((192, 192))
    except Exception:
        return None
    panels = [("input", base)]
    if gt_mask is not None:
        panels.append(("GT", _overlay_mask(Image, base, gt_mask, (width, height), (255, 0, 0))))
    panels.append(("long256", _overlay_mask(Image, base, baseline_mask, (width, height), (255, 0, 0))))
    panels.append(("tile", _overlay_mask(Image, base, tile_mask, (width, height), (255, 0, 0))))
    diff = [1 if b != t else 0 for b, t in zip(baseline_mask, tile_mask)]
    panels.append(("diff", _overlay_mask(Image, base, diff, (width, height), (0, 80, 255))))
    sheet = Image.new("RGB", (192 * len(panels), 216), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(panels):
        x = index * 192
        sheet.paste(panel, (x, 24))
        draw.text((x + 6, 6), label, fill=(0, 0, 0))
    visual_root = output_root / "tile_visual_sheets"
    visual_root.mkdir(parents=True, exist_ok=True)
    path = visual_root / f"{record['sample_id']}.jpg"
    sheet.save(path, quality=90)
    return str(path)


def evaluate_fixture_records(config: dict[str, Any], manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    max_samples = int(config.get("max_samples", 0))
    visual_paths: list[str] = []
    output_root = _real(config["output_root"])
    visual_limit = int(config.get("visual_top_n_per_bucket", 0))
    for index, sample in enumerate(manifest.get("samples", [])[:max_samples]):
        if not isinstance(sample, dict):
            continue
        width, height = _sample_size(sample, config)
        primary_class = str(sample.get("long256_class", sample.get("class_label", "real")))
        score = float(sample.get("tampered_score", 0.0))
        decision = activation_decision(primary_class, score, float(config.get("tampered_suspect_threshold", 0.5)))
        baseline_mask = _mask_from_sample(sample, "baseline_mask", width, height)
        gt_mask = _mask_from_sample(sample, "gt_mask", width, height) if sample.get("gt_mask") is not None else None
        tiles = tile_grid(width, height, int(config["tile_size"]), int(config["tile_stride"])) if decision["activate"] else []
        tile_predictions = _simulated_tile_predictions(width, height, tiles, sample) if decision["activate"] else []
        tile_mask = merge_tile_masks(width, height, tile_predictions, str(config.get("merge_mode", "union"))) if decision["activate"] else baseline_mask
        comparison = compare_masks(gt_mask, baseline_mask, tile_mask, (width, height), float(config.get("improvement_epsilon", 0.0)))
        record = {
            "sample_id": _sample_id(sample, index),
            "primary_class": primary_class,
            "primary_class_unchanged": True,
            "tampered_score": score,
            "activation": decision,
            "tile_count": len(tiles),
            **comparison,
        }
        if visual_limit > 0 and len(visual_paths) < visual_limit:
            visual_path = _write_visual_sheet_for_record(output_root, record, sample, width, height, baseline_mask, tile_mask, gt_mask)
            if visual_path:
                visual_paths.append(visual_path)
                record["visual_sheet_path"] = visual_path
        records.append(record)
    summary = build_evaluation_summary(config, records)
    summary["visual_sheets"] = visual_paths
    return records, summary


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def build_evaluation_summary(config: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_ious = [float(record["baseline_iou"]) for record in records if record.get("baseline_iou") is not None]
    tile_ious = [float(record["tile_iou"]) for record in records if record.get("tile_iou") is not None]
    baseline_dices = [float(record["baseline_dice"]) for record in records if record.get("baseline_dice") is not None]
    tile_dices = [float(record["tile_dice"]) for record in records if record.get("tile_dice") is not None]
    return {
        "marker": MARKER,
        "tile_config": tile_config(config),
        "record_count": len(records),
        "activation_counts": {
            "activated": sum(1 for record in records if record.get("activation", {}).get("activate") is True),
            "skipped": sum(1 for record in records if record.get("activation", {}).get("activate") is not True),
            "uncertain": sum(1 for record in records if record.get("activation", {}).get("uncertain") is True),
        },
        "baseline_long256": {
            "mean_iou": _mean(baseline_ious),
            "median_iou": _median(baseline_ious),
            "mean_dice": _mean(baseline_dices),
            "median_dice": _median(baseline_dices),
        },
        "tile_localized": {
            "mean_iou": _mean(tile_ious),
            "median_iou": _median(tile_ious),
            "mean_dice": _mean(tile_dices),
            "median_dice": _median(tile_dices),
        },
        "improvement_counts": {
            "improved": sum(1 for record in records if record.get("localization_delta") == "improved"),
            "regressed": sum(1 for record in records if record.get("localization_delta") == "regressed"),
            "unchanged": sum(1 for record in records if record.get("localization_delta") == "unchanged"),
            "unknown_no_gt": sum(1 for record in records if record.get("localization_delta") == "unknown_no_gt"),
        },
        "component_statistics": {
            "baseline_component_count_total": sum(int(record.get("baseline_component_count", 0)) for record in records),
            "tile_component_count_total": sum(int(record.get("tile_component_count", 0)) for record in records),
        },
        "no_training": True,
        "dry_run_only": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
        "no_download": True,
        "no_network": True,
    }


def run_tile_localization_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    errors = validate_tile_localization_config(config, require_exists=True)
    if errors:
        raise TileLocalizationError("tile localization config validation failed:\n" + "\n".join(errors))
    if config.get("config_kind") != APPROVED_KIND:
        raise TileLocalizationError(f"execution requires {APPROVED_KIND}")
    root = _safe_output_root(config)
    manifest = load_manifest(config["manifest_path"])
    records, summary = evaluate_fixture_records(config, manifest)
    output_paths = {
        "tile_localization_records": _write_jsonl(root / "tile_localization_records.jsonl", records),
        "tile_vs_long256_summary": str(root / "tile_vs_long256_summary.json"),
        "tile_failure_bucket_summary": _write_json(root / "tile_failure_bucket_summary.json", summary["improvement_counts"]),
        "tile_localization_artifact_manifest": str(root / "tile_localization_artifact_manifest.json"),
    }
    summary["output_paths"] = output_paths
    _write_json(root / "tile_vs_long256_summary.json", summary)
    _write_json(root / "tile_localization_artifact_manifest.json", {"marker": MARKER, "output_paths": output_paths, "visual_sheets": summary.get("visual_sheets", [])})
    return summary


def training_plan_schema_ok(summary: dict[str, Any]) -> bool:
    return (
        summary.get("marker") == MARKER
        and isinstance(summary.get("tile_config"), dict)
        and isinstance(summary.get("planned_sample_count"), int)
        and isinstance(summary.get("total_tile_count"), int)
        and summary.get("no_training") is True
        and summary.get("dry_run_only") is True
        and summary.get("no_checkpoint_writes") is True
        and summary.get("no_sns_augmentation") is True
    )


def evaluation_summary_schema_ok(summary: dict[str, Any]) -> bool:
    return (
        summary.get("marker") == MARKER
        and isinstance(summary.get("activation_counts"), dict)
        and isinstance(summary.get("baseline_long256"), dict)
        and isinstance(summary.get("tile_localized"), dict)
        and isinstance(summary.get("improvement_counts"), dict)
        and isinstance(summary.get("component_statistics"), dict)
        and summary.get("no_training") is True
        and summary.get("no_checkpoint_writes") is True
        and summary.get("no_sns_augmentation") is True
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value
