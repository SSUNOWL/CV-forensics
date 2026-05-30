"""Integrated clean-long256 detector + tile localization report path."""

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
from .pre_sns_v3_tile_localization import (
    activation_decision,
    merge_tile_masks,
    tile_config,
    tile_grid,
)

MARKER = "PRE_SNS_V3_LONG256_TILE_REPORT_OK"
CONFIG_OK_MARKER = "PRE_SNS_V3_LONG256_TILE_REPORT_CONFIG_OK"
APPROVED_KIND = "approved_pre_sns_v3_long256_tile_report"
EXAMPLE_KIND = "example_symbolic"


class Long256TileReportError(ValueError):
    """Raised when integrated long256+tile report inputs are invalid."""


def load_long256_tile_report_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise Long256TileReportError("long256 tile report config root must be a JSON object")
    return raw


def validate_long256_tile_report_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "long256_checkpoint_path",
        "approved_input_roots",
        "output_root",
        "tile_size",
        "tile_stride",
        "crop_context_px",
        "high_res_max_size",
        "tampered_suspect_threshold",
        "no_download",
        "no_network",
        "no_training",
        "no_checkpoint_writes",
        "no_sns_augmentation",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    if not raw.get("manifest_path") and not raw.get("image_path"):
        errors.append(_err("manifest_path or image_path is required"))
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
    if raw.get("manifest_path"):
        errors.extend(_validate_under_roots(raw.get("manifest_path"), "manifest_path", roots, require_file=require_exists))
    if raw.get("image_path"):
        errors.extend(_validate_under_roots(raw.get("image_path"), "image_path", roots, require_file=require_exists))
    if raw.get("gt_iou_mining_records_path"):
        errors.extend(_validate_under_roots(raw.get("gt_iou_mining_records_path"), "gt_iou_mining_records_path", roots, require_file=require_exists))
    errors.extend(_validate_under_roots(raw.get("long256_checkpoint_path"), "long256_checkpoint_path", model_roots, require_file=require_exists, allow_parts={"checkpoints"}))

    output_root = raw.get("output_root")
    errors.extend(_validate_absolute_path(output_root, "output_root", require_dir_parent=kind == APPROVED_KIND and require_exists))
    if isinstance(output_root, str) and output_root.startswith("/") and any(_is_under(output_root, root) for root in roots):
        errors.append(_err("output_root must not be under an approved input root"))

    if raw.get("manifest_path"):
        if isinstance(raw.get("max_samples"), bool) or not isinstance(raw.get("max_samples"), int) or raw.get("max_samples", 0) <= 0:
            errors.append(_err("max_samples must be a positive integer when manifest_path is used"))
    for field in ("tile_size", "tile_stride", "crop_context_px", "high_res_max_size"):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(_err(f"{field} must be a positive integer"))
    for field in ("visual_top_n",):
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
    if kind == APPROVED_KIND and raw.get("execution_mode") != "approved_local_pre_sns_v3_long256_tile_report":
        errors.append(_err("execution_mode must be approved_local_pre_sns_v3_long256_tile_report"))
    if kind == EXAMPLE_KIND and raw.get("execution_mode") != "example_only":
        errors.append(_err("example execution_mode must be example_only"))
    return errors


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise Long256TileReportError("manifest must be a JSON object with samples list")
    return manifest


def flatten_binary(value: Any, width: int, height: int) -> list[int]:
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


def sample_size(sample: dict[str, Any], config: dict[str, Any]) -> tuple[int, int]:
    width = int(sample.get("width") or sample.get("image_width") or config.get("high_res_max_size", 512))
    height = int(sample.get("height") or sample.get("image_height") or config.get("high_res_max_size", 512))
    limit = int(config.get("high_res_max_size", max(width, height)))
    if max(width, height) > limit:
        scale = limit / max(width, height)
        width = max(1, int(round(width * scale)))
        height = max(1, int(round(height * scale)))
    return width, height


def sample_id(sample: dict[str, Any], index: int) -> str:
    raw = str(sample.get("sample_id") or sample.get("id") or f"sample_{index:06d}")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)[:80]


def long256_report_from_sample(sample: dict[str, Any], config: dict[str, Any], index: int = 0) -> dict[str, Any]:
    primary_class = str(sample.get("long256_class", sample.get("class", sample.get("class_label", "real"))))
    tampered_score = float(sample.get("tampered_score", sample.get("long256_tampered_score", 0.0)))
    family = str(sample.get("family", sample.get("long256_family", "Real-or-N/A")))
    return {
        "sample_id": sample_id(sample, index),
        "image_path": sample.get("image_path", config.get("image_path")),
        "class": primary_class,
        "class_conf": sample.get("class_conf", {primary_class: 1.0}),
        "family": family,
        "family_conf": sample.get("family_conf", {family: 1.0}),
        "tampered_score": tampered_score,
    }


def simulated_tile_predictions(width: int, height: int, tiles: list[dict[str, int]], sample: dict[str, Any]) -> list[dict[str, Any]]:
    source = flatten_binary(sample.get("tile_mask"), width, height)
    predictions: list[dict[str, Any]] = []
    for tile in tiles:
        mask: list[int] = []
        for y in range(tile["y0"], tile["y1"]):
            for x in range(tile["x0"], tile["x1"]):
                mask.append(source[y * width + x])
        predictions.append({"tile": tile, "mask": mask})
    return predictions


def select_final_mask(baseline_mask: list[int], tile_mask: list[int], activation: dict[str, Any]) -> dict[str, Any]:
    if not activation.get("activate"):
        return {"mask": baseline_mask, "source": "baseline_long256", "uncertain": False}
    if sum(tile_mask) > 0:
        return {"mask": tile_mask, "source": "tile_localized", "uncertain": bool(activation.get("uncertain"))}
    return {"mask": baseline_mask, "source": "baseline_long256_tile_empty", "uncertain": True}


def baseline_vs_tile_metrics(gt_mask: list[int] | None, baseline_mask: list[int], tile_mask: list[int], shape: tuple[int, int], epsilon: float = 0.0) -> dict[str, Any]:
    baseline_stats = mask_stats(baseline_mask, shape)
    tile_stats = mask_stats(tile_mask, shape)
    result: dict[str, Any] = {
        "baseline_long256_area_pct": float(baseline_stats["mask_area_pct"]),
        "tile_area_pct": float(tile_stats["mask_area_pct"]),
        "baseline_long256_component_count": int(baseline_stats["component_count"]),
        "tile_component_count": int(tile_stats["component_count"]),
        "baseline_long256_largest_component_area_pct": float(baseline_stats["largest_component_area_pct"]),
        "tile_largest_component_area_pct": float(tile_stats["largest_component_area_pct"]),
    }
    if gt_mask is None:
        result.update({"baseline_long256_iou": None, "tile_iou": None, "baseline_long256_dice": None, "tile_dice": None, "localization_delta": "unknown_no_gt"})
        return result
    baseline_iou = iou_score(gt_mask, baseline_mask)
    tile_iou = iou_score(gt_mask, tile_mask)
    result.update({
        "baseline_long256_iou": baseline_iou,
        "tile_iou": tile_iou,
        "baseline_long256_dice": dice_score(gt_mask, baseline_mask),
        "tile_dice": dice_score(gt_mask, tile_mask),
    })
    if tile_iou > baseline_iou + epsilon:
        result["localization_delta"] = "improved"
    elif tile_iou + epsilon < baseline_iou:
        result["localization_delta"] = "regressed"
    else:
        result["localization_delta"] = "unchanged"
    return result


def reason_text(primary_class: str, tampered_score: float, final_source: str, localization_delta: str, family: str) -> str:
    if final_source == "tile_localized":
        return f"clean long256 predicts {primary_class}; tile localization is active and {localization_delta}; family estimate is {family}."
    if final_source == "baseline_long256_tile_empty":
        return f"clean long256 predicts {primary_class}, but tile localization returned an empty mask; baseline long256 mask is retained with uncertainty."
    if primary_class == "tampered" or tampered_score >= 0.5:
        return f"clean long256 predicts tampered evidence, but tile localization is skipped; baseline long256 mask is retained."
    return f"clean long256 predicts {primary_class}; tile localization is skipped and no class change is applied."


def build_record(sample: dict[str, Any], config: dict[str, Any], index: int = 0) -> dict[str, Any]:
    width, height = sample_size(sample, config)
    report = long256_report_from_sample(sample, config, index)
    activation = activation_decision(report["class"], report["tampered_score"], float(config.get("tampered_suspect_threshold", 0.5)))
    baseline_mask = flatten_binary(sample.get("baseline_mask", sample.get("long256_mask")), width, height)
    gt_mask = flatten_binary(sample.get("gt_mask"), width, height) if sample.get("gt_mask") is not None else None
    tiles = tile_grid(width, height, int(config["tile_size"]), int(config["tile_stride"])) if activation["activate"] else []
    tile_mask = merge_tile_masks(width, height, simulated_tile_predictions(width, height, tiles, sample), str(config.get("merge_mode", "union"))) if activation["activate"] else baseline_mask
    selected = select_final_mask(baseline_mask, tile_mask, activation)
    metrics = baseline_vs_tile_metrics(gt_mask, baseline_mask, tile_mask, (width, height), float(config.get("improvement_epsilon", 0.0)))
    final_stats = mask_stats(selected["mask"], (width, height))
    baseline_stats = mask_stats(baseline_mask, (width, height))
    tile_stats = mask_stats(tile_mask, (width, height))
    return {
        "marker": MARKER,
        "sample_id": report["sample_id"],
        "image_path": report["image_path"],
        "primary_detector": "clean_long256",
        "class": report["class"],
        "class_conf": report["class_conf"],
        "family": report["family"],
        "family_conf": report["family_conf"],
        "tampered_score": report["tampered_score"],
        "tile_localization_status": activation["status"],
        "primary_class_unchanged": True,
        "baseline_long256_mask_stats": baseline_stats,
        "tile_mask_stats": tile_stats,
        "final_mask_source": selected["source"],
        "final_mask_stats": final_stats,
        "baseline_vs_tile_metrics": metrics,
        "localization_delta": metrics["localization_delta"],
        "reason": reason_text(report["class"], report["tampered_score"], selected["source"], metrics["localization_delta"], report["family"]),
        "visual_artifacts": {},
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
    }


def _overlay_mask(Image: Any, base: Any, mask: list[int], shape: tuple[int, int], color: tuple[int, int, int]) -> Any:
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
        if active:
            blended.append(tuple(int(round((1.0 - alpha) * pixel[index] + alpha * color[index])) for index in range(3)))
        else:
            blended.append(pixel)
    panel = Image.new("RGB", base.size)
    panel.putdata(blended)
    return panel


def _record_masks(sample: dict[str, Any], config: dict[str, Any], record: dict[str, Any]) -> tuple[list[int], list[int], list[int], list[int] | None, tuple[int, int]]:
    width, height = sample_size(sample, config)
    baseline = flatten_binary(sample.get("baseline_mask", sample.get("long256_mask")), width, height)
    tile = flatten_binary(sample.get("tile_mask"), width, height)
    final_mask = tile if record.get("final_mask_source") == "tile_localized" else baseline
    gt = flatten_binary(sample.get("gt_mask"), width, height) if sample.get("gt_mask") is not None else None
    return baseline, tile, final_mask, gt, (width, height)


def write_visual_sheets(output_root: Path, samples: list[dict[str, Any]], records: list[dict[str, Any]], config: dict[str, Any]) -> list[str]:
    top_n = int(config.get("visual_top_n", 0))
    if top_n <= 0:
        return []
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return []
    visual_root = output_root / "long256_tile_visual_sheets"
    visual_root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for sample, record in zip(samples, records):
        if len(paths) >= top_n:
            break
        image_path = sample.get("image_path") or record.get("image_path")
        if not image_path:
            continue
        try:
            with Image.open(image_path) as image:
                base = image.convert("RGB").resize((192, 192))
        except Exception:
            continue
        baseline, tile, final_mask, gt, shape = _record_masks(sample, config, record)
        panels = [("input", base)]
        panels.append(("long256", _overlay_mask(Image, base, baseline, shape, (255, 0, 0))))
        if record.get("tile_localization_status", "").startswith("activated"):
            panels.append(("tile", _overlay_mask(Image, base, tile, shape, (255, 0, 0))))
        panels.append(("final", _overlay_mask(Image, base, final_mask, shape, (255, 0, 0))))
        if gt is not None:
            panels.append(("GT", _overlay_mask(Image, base, gt, shape, (255, 0, 0))))
        diff = [1 if b != f else 0 for b, f in zip(baseline, final_mask)]
        panels.append(("diff", _overlay_mask(Image, base, diff, shape, (0, 80, 255))))
        sheet = Image.new("RGB", (192 * len(panels), 216), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for index, (label, panel) in enumerate(panels):
            x = index * 192
            sheet.paste(panel, (x, 24))
            draw.text((x + 6, 6), label, fill=(0, 0, 0))
        path = visual_root / f"{record.get('sample_id', len(paths))}.jpg"
        sheet.save(path, quality=90)
        paths.append(str(path))
    return paths


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def build_summary(config: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_ious = [float(record["baseline_vs_tile_metrics"]["baseline_long256_iou"]) for record in records if record["baseline_vs_tile_metrics"].get("baseline_long256_iou") is not None]
    tile_ious = [float(record["baseline_vs_tile_metrics"]["tile_iou"]) for record in records if record["baseline_vs_tile_metrics"].get("tile_iou") is not None]
    return {
        "marker": MARKER,
        "record_count": len(records),
        "tile_config": tile_config(config),
        "activation_counts": {
            "activated": sum(1 for record in records if record["tile_localization_status"].startswith("activated")),
            "skipped": sum(1 for record in records if record["tile_localization_status"].startswith("skipped")),
            "uncertain": sum(1 for record in records if record["final_mask_source"] == "baseline_long256_tile_empty" or record["tile_localization_status"] == "activated_tampered_suspect"),
        },
        "baseline_vs_tile_metrics": {
            "baseline_long256_mean_iou": _mean(baseline_ious),
            "baseline_long256_median_iou": _median(baseline_ious),
            "tile_mean_iou": _mean(tile_ious),
            "tile_median_iou": _median(tile_ious),
        },
        "improvement_counts": {
            "improved": sum(1 for record in records if record["localization_delta"] == "improved"),
            "regressed": sum(1 for record in records if record["localization_delta"] == "regressed"),
            "unchanged": sum(1 for record in records if record["localization_delta"] == "unchanged"),
            "unknown_no_gt": sum(1 for record in records if record["localization_delta"] == "unknown_no_gt"),
        },
        "component_stats": {
            "baseline_component_count_total": sum(int(record["baseline_long256_mask_stats"]["component_count"]) for record in records),
            "tile_component_count_total": sum(int(record["tile_mask_stats"]["component_count"]) for record in records),
            "final_component_count_total": sum(int(record["final_mask_stats"]["component_count"]) for record in records),
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
        raise Long256TileReportError("output_root must be outside repository")
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


def run_long256_tile_report(config: dict[str, Any]) -> dict[str, Any]:
    errors = validate_long256_tile_report_config(config, require_exists=True)
    if errors:
        raise Long256TileReportError("long256 tile report config validation failed:\n" + "\n".join(errors))
    if config.get("config_kind") != APPROVED_KIND:
        raise Long256TileReportError(f"execution requires {APPROVED_KIND}")
    if config.get("manifest_path"):
        manifest = load_manifest(config["manifest_path"])
        samples = [sample for sample in manifest.get("samples", [])[: int(config.get("max_samples", 1))] if isinstance(sample, dict)]
    else:
        samples = [dict(config.get("fixture_sample", {}), image_path=config.get("image_path"))]
    records = [build_record(sample, config, index) for index, sample in enumerate(samples)]
    summary = build_summary(config, records)
    root = _safe_output_root(config)
    visual_sheets = write_visual_sheets(root, samples, records, config)
    output_paths: dict[str, str] = {}
    if config.get("manifest_path"):
        output_paths["long256_tile_integrated_records"] = _write_jsonl(root / "long256_tile_integrated_records.jsonl", records)
    else:
        output_paths["long256_tile_integrated_report"] = _write_json(root / "long256_tile_integrated_report.json", records[0] if records else {})
    output_paths["long256_tile_integrated_summary"] = str(root / "long256_tile_integrated_summary.json")
    output_paths["long256_tile_artifact_manifest"] = str(root / "long256_tile_artifact_manifest.json")
    summary["output_paths"] = output_paths
    _write_json(root / "long256_tile_integrated_summary.json", summary)
    _write_json(root / "long256_tile_artifact_manifest.json", {"marker": MARKER, "output_paths": output_paths, "visual_sheets": visual_sheets})
    return summary if config.get("manifest_path") else {**records[0], "summary": summary, "output_paths": output_paths}


def report_schema_ok(record: dict[str, Any]) -> bool:
    required = {
        "marker",
        "image_path",
        "primary_detector",
        "class",
        "class_conf",
        "family",
        "family_conf",
        "tampered_score",
        "tile_localization_status",
        "primary_class_unchanged",
        "baseline_long256_mask_stats",
        "tile_mask_stats",
        "final_mask_source",
        "final_mask_stats",
        "baseline_vs_tile_metrics",
        "localization_delta",
        "reason",
        "visual_artifacts",
    }
    return required.issubset(record) and record.get("marker") == MARKER and record.get("primary_class_unchanged") is True


def summary_schema_ok(summary: dict[str, Any]) -> bool:
    return (
        summary.get("marker") == MARKER
        and isinstance(summary.get("activation_counts"), dict)
        and isinstance(summary.get("baseline_vs_tile_metrics"), dict)
        and isinstance(summary.get("improvement_counts"), dict)
        and isinstance(summary.get("component_stats"), dict)
        and summary.get("no_download") is True
        and summary.get("no_network") is True
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
