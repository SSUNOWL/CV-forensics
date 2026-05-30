"""Local run orchestration for pre-SNS long256 + tile red-mask inspection."""

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
)
from .pre_sns_v3_long256_tile_report import (
    APPROVED_KIND as REPORT_APPROVED_KIND,
    MARKER as REPORT_MARKER,
    build_record,
    flatten_binary,
    load_manifest,
    sample_size,
)

MARKER = "PRE_SNS_V3_LONG256_TILE_LOCAL_RUN_OK"
CONFIG_OK_MARKER = "PRE_SNS_V3_LONG256_TILE_LOCAL_RUN_CONFIG_OK"
APPROVED_KIND = "approved_pre_sns_v3_long256_tile_local_run"
EXAMPLE_KIND = "example_symbolic"


class LocalRunError(ValueError):
    """Raised when local run inputs or configs are invalid."""


def load_local_run_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise LocalRunError("local run config root must be a JSON object")
    return raw


def validate_local_run_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "long256_checkpoint_path",
        "approved_input_roots",
        "output_root",
        "max_samples",
        "tile_size",
        "tile_stride",
        "crop_context_px",
        "high_res_max_size",
        "tampered_suspect_threshold",
        "visual_top_n",
        "no_download",
        "no_network",
        "no_training",
        "no_checkpoint_writes",
        "no_sns_augmentation",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    if not raw.get("manifest_path") and not raw.get("sample_list_path"):
        errors.append(_err("manifest_path or sample_list_path is required"))
    kind = raw.get("config_kind")
    if kind not in {APPROVED_KIND, EXAMPLE_KIND}:
        errors.append(_err(f"config_kind must be {APPROVED_KIND} or {EXAMPLE_KIND}"))
    for flag in ("no_download", "no_network", "no_training", "no_checkpoint_writes", "no_sns_augmentation"):
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
    model_roots = list(roots) + list(checkpoint_roots)
    for field in ("manifest_path", "sample_list_path", "gt_iou_mining_records_path"):
        if raw.get(field):
            errors.extend(_validate_under_roots(raw.get(field), field, roots, require_file=require_exists))
    errors.extend(_validate_under_roots(raw.get("long256_checkpoint_path"), "long256_checkpoint_path", model_roots, require_file=require_exists, allow_parts={"checkpoints"}))
    output_root = raw.get("output_root")
    errors.extend(_validate_absolute_path(output_root, "output_root", require_dir_parent=kind == APPROVED_KIND and require_exists))
    if isinstance(output_root, str) and output_root.startswith("/") and any(_is_under(output_root, root) for root in roots):
        errors.append(_err("output_root must not be under an approved input root"))

    for field in ("max_samples", "tile_size", "tile_stride", "crop_context_px", "high_res_max_size"):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(_err(f"{field} must be a positive integer"))
    if isinstance(raw.get("visual_top_n"), bool) or not isinstance(raw.get("visual_top_n"), int) or raw.get("visual_top_n", -1) < 0:
        errors.append(_err("visual_top_n must be a non-negative integer"))
    for field in ("tampered_suspect_threshold", "good_iou_threshold", "good_dice_threshold", "low_iou_threshold", "overseg_ratio_threshold", "underseg_ratio_threshold"):
        if field in raw:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
                errors.append(_err(f"{field} must be a non-negative number"))
    if raw.get("merge_mode", "union") not in {"union", "max", "weighted_average"}:
        errors.append(_err("merge_mode must be union, max, or weighted_average"))
    if kind == APPROVED_KIND and raw.get("execution_mode") != "approved_local_pre_sns_v3_long256_tile_local_run":
        errors.append(_err("execution_mode must be approved_local_pre_sns_v3_long256_tile_local_run"))
    if kind == EXAMPLE_KIND and raw.get("execution_mode") != "example_only":
        errors.append(_err("example execution_mode must be example_only"))
    return errors


def load_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("manifest_path"):
        manifest = load_manifest(config["manifest_path"])
        samples = manifest.get("samples", [])
    else:
        with open(config["sample_list_path"], "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        samples = raw.get("samples", raw) if isinstance(raw, dict) else raw
    return [sample for sample in samples[: int(config["max_samples"])] if isinstance(sample, dict)]


def build_sample_plan(config: dict[str, Any], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        width, height = sample_size(sample, config)
        plan.append({
            "sample_id": sample.get("sample_id") or sample.get("id") or f"sample_{index:06d}",
            "image_path": sample.get("image_path"),
            "width": width,
            "height": height,
            "run_index": index,
        })
    return plan


def _thresholds(config: dict[str, Any]) -> dict[str, float]:
    return {
        "good_iou_threshold": float(config.get("good_iou_threshold", 0.4)),
        "good_dice_threshold": float(config.get("good_dice_threshold", 0.5)),
        "low_iou_threshold": float(config.get("low_iou_threshold", 0.15)),
        "overseg_ratio_threshold": float(config.get("overseg_ratio_threshold", 2.0)),
        "underseg_ratio_threshold": float(config.get("underseg_ratio_threshold", 0.5)),
    }


def bucket_record(record: dict[str, Any], config: dict[str, Any]) -> str:
    thresholds = _thresholds(config)
    metrics = record.get("baseline_vs_tile_metrics", {})
    final_stats = record.get("final_mask_stats", {})
    final_area = float(final_stats.get("mask_area_pct", 0.0))
    tile_status = str(record.get("tile_localization_status", ""))
    suspicious = str(record.get("class")) == "tampered" or tile_status.startswith("activated")
    has_gt = metrics.get("tile_iou") is not None
    if has_gt:
        iou = float(metrics.get("tile_iou", 0.0))
        dice = float(metrics.get("tile_dice", 0.0))
        baseline_area = float(metrics.get("baseline_long256_area_pct", 0.0))
        ratio = None if baseline_area <= 0.0 else final_area / baseline_area
        if iou >= thresholds["good_iou_threshold"] and dice >= thresholds["good_dice_threshold"]:
            return "good_red_mask_cases"
        if suspicious and final_area <= 0.0:
            return "failed_red_mask_cases"
        if iou < thresholds["low_iou_threshold"]:
            return "failed_red_mask_cases"
        if ratio is not None and ratio >= thresholds["overseg_ratio_threshold"]:
            return "failed_red_mask_cases"
        if ratio is not None and ratio <= thresholds["underseg_ratio_threshold"]:
            return "failed_red_mask_cases"
        return "needs_manual_review_cases"
    if suspicious and final_area > 0.0 and str(record.get("final_mask_source")) in {"tile_localized", "baseline_long256"}:
        return "good_red_mask_cases"
    if suspicious and final_area <= 0.0:
        return "failed_red_mask_cases"
    return "needs_manual_review_cases"


def bucket_records(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    buckets = {"good_red_mask_cases": [], "failed_red_mask_cases": [], "needs_manual_review_cases": []}
    for record in records:
        buckets[bucket_record(record, config)].append(record)
    return buckets


def recommendation_from_buckets(buckets: dict[str, list[dict[str, Any]]]) -> str:
    total = sum(len(items) for items in buckets.values())
    if total == 0:
        return "needs_manual_review"
    failed = len(buckets.get("failed_red_mask_cases", []))
    manual = len(buckets.get("needs_manual_review_cases", []))
    good = len(buckets.get("good_red_mask_cases", []))
    if failed == 0 and manual == 0 and good > 0:
        return "ready_for_sns_robustness_evaluation"
    if failed / total >= 0.25:
        return "needs_real_tile_localization_training"
    return "needs_manual_review"


def summarize_local_run(records: list[dict[str, Any]], buckets: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    metrics = [record.get("baseline_vs_tile_metrics", {}) for record in records]
    tile_ious = [float(m["tile_iou"]) for m in metrics if m.get("tile_iou") is not None]
    tile_dices = [float(m["tile_dice"]) for m in metrics if m.get("tile_dice") is not None]
    final_areas = [float(record.get("final_mask_stats", {}).get("mask_area_pct", 0.0)) for record in records]
    return {
        "marker": MARKER,
        "record_count": len(records),
        "bucket_counts": {key: len(value) for key, value in buckets.items()},
        "recommendation": recommendation_from_buckets(buckets),
        "thresholds": _thresholds(config),
        "activation_counts": {
            "activated": sum(1 for record in records if str(record.get("tile_localization_status", "")).startswith("activated")),
            "skipped": sum(1 for record in records if str(record.get("tile_localization_status", "")).startswith("skipped")),
        },
        "localization_metrics": {
            "tile_mean_iou": sum(tile_ious) / len(tile_ious) if tile_ious else 0.0,
            "tile_mean_dice": sum(tile_dices) / len(tile_dices) if tile_dices else 0.0,
            "final_mean_area_pct": sum(final_areas) / len(final_areas) if final_areas else 0.0,
            "final_max_area_pct": max(final_areas, default=0.0),
        },
        "improvement_counts": {
            "improved": sum(1 for record in records if record.get("localization_delta") == "improved"),
            "regressed": sum(1 for record in records if record.get("localization_delta") == "regressed"),
            "unchanged": sum(1 for record in records if record.get("localization_delta") == "unchanged"),
            "unknown_no_gt": sum(1 for record in records if record.get("localization_delta") == "unknown_no_gt"),
        },
        "component_totals": {
            "final_component_count": sum(int(record.get("final_mask_stats", {}).get("component_count", 0)) for record in records),
            "final_largest_component_area_pct": max((float(record.get("final_mask_stats", {}).get("largest_component_area_pct", 0.0)) for record in records), default=0.0),
        },
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
    }


def _safe_output_root(config: dict[str, Any]) -> Path:
    root = _real(config["output_root"])
    if _inside_repo(root):
        raise LocalRunError("output_root must be outside repository")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_json(path: Path, value: Any) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
    return str(path)


def _overlay(Image: Any, base: Any, mask: list[int], shape: tuple[int, int]) -> Any:
    if not mask or sum(mask) == 0:
        return base.copy()
    mask_image = Image.new("L", shape)
    mask_image.putdata([255 if value else 0 for value in mask])
    resample_nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    mask_image = mask_image.resize(base.size, resample_nearest)
    pixels = list(base.getdata())
    mask_pixels = list(mask_image.getdata())
    alpha = 0.45
    blended = []
    for pixel, active in zip(pixels, mask_pixels):
        blended.append(tuple(int(round((1 - alpha) * pixel[i] + alpha * (255 if i == 0 else 0))) for i in range(3)) if active else pixel)
    panel = Image.new("RGB", base.size)
    panel.putdata(blended)
    return panel


def _agreement_overlay(Image: Any, base: Any, gt: list[int], pred: list[int], shape: tuple[int, int]) -> Any:
    gt_image = Image.new("L", shape)
    pred_image = Image.new("L", shape)
    gt_image.putdata([255 if value else 0 for value in gt])
    pred_image.putdata([255 if value else 0 for value in pred])
    resample_nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    gt_image = gt_image.resize(base.size, resample_nearest)
    pred_image = pred_image.resize(base.size, resample_nearest)
    alpha = 0.5
    colors = {
        "tp": (0, 180, 0),
        "fp": (255, 0, 0),
        "fn": (0, 80, 255),
    }
    blended = []
    for pixel, gt_active, pred_active in zip(base.getdata(), gt_image.getdata(), pred_image.getdata()):
        color = None
        if gt_active and pred_active:
            color = colors["tp"]
        elif pred_active:
            color = colors["fp"]
        elif gt_active:
            color = colors["fn"]
        blended.append(tuple(int(round((1 - alpha) * pixel[i] + alpha * color[i])) for i in range(3)) if color else pixel)
    panel = Image.new("RGB", base.size)
    panel.putdata(blended)
    return panel


def _save_clean_overlay(path: Path, image: Any) -> str:
    image.save(path)
    return str(path)


def write_gallery(output_root: Path, samples: list[dict[str, Any]], records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    top_n = int(config.get("visual_top_n", 0))
    manifest = {"marker": MARKER, "items": []}
    if top_n <= 0:
        return manifest
    try:
        from PIL import Image, ImageDraw
    except Exception:
        manifest["warning"] = "PIL not available"
        return manifest
    visual_root = output_root / "red_mask_gallery"
    visual_root.mkdir(parents=True, exist_ok=True)
    for sample, record in zip(samples, records):
        if len(manifest["items"]) >= top_n:
            break
        image_path = sample.get("image_path")
        if not image_path:
            continue
        try:
            with Image.open(image_path) as image:
                base = image.convert("RGB").resize((192, 192))
        except Exception:
            continue
        width, height = sample_size(sample, config)
        baseline = flatten_binary(sample.get("baseline_mask", sample.get("long256_mask")), width, height)
        tile = flatten_binary(sample.get("tile_mask"), width, height)
        final_mask = tile if record.get("final_mask_source") == "tile_localized" else baseline
        gt = flatten_binary(sample.get("gt_mask"), width, height) if sample.get("gt_mask") is not None else None
        sample_id = str(record.get("sample_id", len(manifest["items"])))
        long256_overlay = _overlay(Image, base, baseline, (width, height))
        final_overlay = _overlay(Image, base, final_mask, (width, height))
        clean_overlay_paths = {
            "long256_red_overlay_path": _save_clean_overlay(visual_root / f"{sample_id}_long256_red_overlay.png", long256_overlay),
            "final_red_overlay_path": _save_clean_overlay(visual_root / f"{sample_id}_final_red_overlay.png", final_overlay),
        }
        panels = [("input", base), ("long256", long256_overlay), ("final", final_overlay)]
        if str(record.get("tile_localization_status", "")).startswith("activated"):
            tile_overlay = _overlay(Image, base, tile, (width, height))
            clean_overlay_paths["tile_red_overlay_path"] = _save_clean_overlay(visual_root / f"{sample_id}_tile_red_overlay.png", tile_overlay)
            panels.append(("tile", tile_overlay))
        if gt is not None:
            gt_overlay = _overlay(Image, base, gt, (width, height))
            clean_overlay_paths["gt_red_overlay_path"] = _save_clean_overlay(visual_root / f"{sample_id}_gt_red_overlay.png", gt_overlay)
            agreement_overlay = _agreement_overlay(Image, base, gt, final_mask, (width, height))
            clean_overlay_paths["agreement_overlay_path"] = _save_clean_overlay(visual_root / f"{sample_id}_agreement_overlay.png", agreement_overlay)
            panels.append(("GT", gt_overlay))
            panels.append(("agree", agreement_overlay))
        sheet = Image.new("RGB", (192 * len(panels), 216), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for index, (label, panel) in enumerate(panels):
            x = index * 192
            sheet.paste(panel, (x, 24))
            draw.text((x + 6, 6), label, fill=(0, 0, 0))
        path = visual_root / f"{sample_id}_comparison_sheet.jpg"
        sheet.save(path, quality=90)
        manifest["items"].append({
            "sample_id": record.get("sample_id"),
            "image_path": image_path,
            "comparison_sheet_path": str(path),
            **clean_overlay_paths,
        })
    return manifest


def run_local_run(config: dict[str, Any]) -> dict[str, Any]:
    errors = validate_local_run_config(config, require_exists=True)
    if errors:
        raise LocalRunError("local run config validation failed:\n" + "\n".join(errors))
    if config.get("config_kind") != APPROVED_KIND:
        raise LocalRunError(f"execution requires {APPROVED_KIND}")
    samples = load_samples(config)
    records = [build_record(sample, {**config, "config_kind": REPORT_APPROVED_KIND}, index) for index, sample in enumerate(samples)]
    buckets = bucket_records(records, config)
    summary = summarize_local_run(records, buckets, config)
    output_root = _safe_output_root(config)
    gallery = write_gallery(output_root, samples, records, config)
    paths = {
        "local_run_records": _write_jsonl(output_root / "local_run_records.jsonl", records),
        "good_red_mask_cases": _write_json(output_root / "good_red_mask_cases.json", buckets["good_red_mask_cases"]),
        "failed_red_mask_cases": _write_json(output_root / "failed_red_mask_cases.json", buckets["failed_red_mask_cases"]),
        "needs_manual_review_cases": _write_json(output_root / "needs_manual_review_cases.json", buckets["needs_manual_review_cases"]),
        "red_mask_gallery_manifest": _write_json(output_root / "red_mask_gallery_manifest.json", gallery),
    }
    paths["local_run_summary"] = str(output_root / "local_run_summary.json")
    paths["artifact_manifest"] = str(output_root / "artifact_manifest.json")
    summary["output_paths"] = paths
    _write_json(output_root / "local_run_summary.json", summary)
    _write_json(output_root / "artifact_manifest.json", {"marker": MARKER, "output_paths": paths, "gallery_items": gallery.get("items", [])})
    return summary


def summary_schema_ok(summary: dict[str, Any]) -> bool:
    return (
        summary.get("marker") == MARKER
        and isinstance(summary.get("bucket_counts"), dict)
        and summary.get("recommendation") in {"ready_for_sns_robustness_evaluation", "needs_real_tile_localization_training", "needs_manual_review"}
        and isinstance(summary.get("activation_counts"), dict)
        and isinstance(summary.get("localization_metrics"), dict)
        and isinstance(summary.get("improvement_counts"), dict)
        and summary.get("no_training") is True
        and summary.get("no_checkpoint_writes") is True
        and summary.get("no_sns_augmentation") is True
    )


def gallery_schema_ok(gallery: dict[str, Any]) -> bool:
    return gallery.get("marker") == MARKER and isinstance(gallery.get("items"), list)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value
