"""Pre-SNS real GT-IoU mining and tile-localization manifest building."""

from __future__ import annotations

import json
import math
import random
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
    resize_binary_mask_nearest,
)
from .pre_sns_v3_gt_iou_mining import (
    dice_score,
    iou_score,
    load_gt_mask_for_sample,
    run_models_for_sample,
    to_binary_mask,
)

MARKER = "PRE_SNS_V3_GT_IOU_TILE_BUILDER_OK"
CONFIG_OK_MARKER = "PRE_SNS_V3_GT_IOU_TILE_BUILDER_CONFIG_OK"
APPROVED_KIND = "approved_pre_sns_v3_gt_iou_tile_builder"
EXAMPLE_KIND = "example_symbolic"


class GTIoUTileBuilderError(ValueError):
    """Raised when GT-IoU tile-builder inputs or configs are invalid."""


def load_gt_iou_tile_builder_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise GTIoUTileBuilderError("GT-IoU tile-builder config root must be a JSON object")
    return raw


def _as_roots(value: Any) -> list[str]:
    return value if isinstance(value, list) and all(isinstance(item, str) for item in value) else []


def validate_gt_iou_tile_builder_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "approved_real_data_access",
        "train_manifest_path",
        "long256_checkpoint_path",
        "approved_input_roots",
        "output_root",
        "max_samples",
        "tile_size",
        "no_download",
        "no_network",
        "no_training",
        "no_sns_augmentation",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))

    kind = raw.get("config_kind")
    if kind not in {APPROVED_KIND, EXAMPLE_KIND}:
        errors.append(_err(f"config_kind must be {APPROVED_KIND} or {EXAMPLE_KIND}"))
    if kind == APPROVED_KIND and raw.get("execution_mode") != "approved_local_pre_sns_v3_gt_iou_tile_builder":
        errors.append(_err("execution_mode must be approved_local_pre_sns_v3_gt_iou_tile_builder"))
    if kind == EXAMPLE_KIND and raw.get("execution_mode") != "example_only":
        errors.append(_err("example execution_mode must be example_only"))

    for flag in ("approved_real_data_access", "no_download", "no_network", "no_training", "no_sns_augmentation"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if raw.get("no_checkpoint_writes", True) is not True:
        errors.append(_err("no_checkpoint_writes must be true when present"))
    if raw.get("write_cropped_images", False) is True:
        errors.append(_err("write_cropped_images must be false for this manifest-only task"))

    roots = _as_roots(raw.get("approved_input_roots"))
    if not roots:
        errors.append(_err("approved_input_roots must be a non-empty list of absolute paths"))
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
    errors.extend(_validate_under_roots(raw.get("train_manifest_path"), "train_manifest_path", roots, require_file=require_exists))
    if raw.get("gt_iou_records_path"):
        errors.extend(_validate_under_roots(raw.get("gt_iou_records_path"), "gt_iou_records_path", roots, require_file=require_exists))
    if raw.get("hard_negative_records_path"):
        errors.extend(_validate_under_roots(raw.get("hard_negative_records_path"), "hard_negative_records_path", roots, require_file=require_exists))
    for field in ("long256_checkpoint_path", "long224_checkpoint_path", "refined_checkpoint_path"):
        if field == "long256_checkpoint_path" or raw.get(field):
            errors.extend(_validate_under_roots(raw.get(field), field, approved_model_roots, require_file=require_exists, allow_parts={"checkpoints"}))

    output_root = raw.get("output_root")
    errors.extend(_validate_absolute_path(output_root, "output_root", require_dir_parent=kind == APPROVED_KIND and require_exists))
    if isinstance(output_root, str) and output_root.startswith("/"):
        if _inside_repo(_real(output_root)):
            errors.append(_err("output_root must be outside repository"))
        if any(_is_under(output_root, root) for root in roots):
            errors.append(_err("output_root must not be under an approved input root"))

    if isinstance(raw.get("max_samples"), bool) or not isinstance(raw.get("max_samples"), int) or raw.get("max_samples", 0) <= 0:
        errors.append(_err("max_samples must be a positive integer"))
    for field in (
        "tile_size",
        "positive_jitter_count",
        "negative_random_count",
        "hard_negative_count",
        "severe_oversample_factor",
        "low_iou_oversample_factor",
        "visual_top_n",
        "seed",
    ):
        if field in raw:
            value = raw[field]
            lower = 1 if field == "tile_size" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                errors.append(_err(f"{field} must be an integer >= {lower}"))
    for field in (
        "mask_threshold",
        "threshold_tau",
        "tiny_gt_area_pct_threshold",
        "underseg_ratio_threshold",
        "overseg_ratio_threshold",
        "fragmented_component_threshold",
    ):
        if field in raw:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                errors.append(_err(f"{field} must be a non-negative number"))

    split_value = str(raw.get("source_split", "train")).lower()
    if "val" in split_value or "valid" in split_value:
        errors.append(_err("validation hard cases must not be used as training tile records"))
    return errors


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    samples = manifest.get("samples") or manifest.get("sample_manifest") if isinstance(manifest, dict) else None
    if not isinstance(samples, list):
        raise GTIoUTileBuilderError("manifest must contain samples or sample_manifest list")
    return {"samples": samples, "source": str(path)}


def load_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
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


def sample_id(sample: dict[str, Any], index: int = 0) -> str:
    raw = str(sample.get("sample_id") or sample.get("id") or f"sample_{index:06d}")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)[:96]


def sample_class(sample: dict[str, Any]) -> str:
    for key in ("class_label", "gt_class", "label", "class"):
        if sample.get(key) is not None:
            return str(sample[key]).lower()
    if sample.get("is_tampered") is True:
        return "tampered"
    return ""


def mask_path_for_sample(sample: dict[str, Any]) -> str | None:
    for key in ("mask_path", "gt_mask_path", "tamper_mask_path", "tampered_mask_path", "localization_mask_path"):
        value = sample.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def has_gt_mask(sample: dict[str, Any]) -> bool:
    return mask_path_for_sample(sample) is not None or sample.get("gt_mask") is not None or sample.get("mask") is not None


def select_tampered_train_samples(manifest: dict[str, Any], max_samples: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for sample in manifest.get("samples", []):
        if isinstance(sample, dict) and sample_class(sample) == "tampered" and has_gt_mask(sample):
            selected.append(sample)
            if len(selected) >= max_samples:
                break
    return selected


def select_negative_samples(manifest: dict[str, Any], max_samples: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for sample in manifest.get("samples", []):
        if not isinstance(sample, dict):
            continue
        label = sample_class(sample)
        if label in {"real", "full_synthetic", "synthetic"}:
            selected.append(sample)
            if len(selected) >= max_samples:
                break
    return selected


def compute_mask_metrics(gt_mask: list[int], pred_mask: list[int], shape: tuple[int, int]) -> dict[str, Any]:
    if len(pred_mask) != len(gt_mask):
        pred_mask = resize_binary_mask_nearest(pred_mask, shape, shape) if pred_mask else [0] * len(gt_mask)
    gt_stats = mask_stats(gt_mask, shape)
    pred_stats = mask_stats(pred_mask, shape)
    gt_area = float(gt_stats["mask_area_pct"])
    pred_area = float(pred_stats["mask_area_pct"])
    return {
        "iou": iou_score(gt_mask, pred_mask),
        "dice": dice_score(gt_mask, pred_mask),
        "gt_area_pct": gt_area,
        "pred_area_pct": pred_area,
        "gt_component_count": int(gt_stats["component_count"]),
        "pred_component_count": int(pred_stats["component_count"]),
        "largest_gt_component_area_pct": float(gt_stats["largest_component_area_pct"]),
        "largest_pred_component_area_pct": float(pred_stats["largest_component_area_pct"]),
        "mask_area_ratio_pred_over_gt": None if gt_area <= 0.0 else float(pred_area / gt_area),
    }


def classify_iou_bucket(metrics: dict[str, Any]) -> str:
    iou = float(metrics.get("iou", 0.0))
    if iou < 0.05:
        return "severe_iou_fail"
    if iou < 0.15:
        return "low_iou"
    if iou < 0.25:
        return "weak_iou"
    if iou >= 0.40:
        return "good_iou"
    return "mid_iou"


def classify_failure_types(metrics: dict[str, Any], config: dict[str, Any] | None = None) -> list[str]:
    cfg = config or {}
    types = [classify_iou_bucket(metrics)]
    iou = float(metrics.get("iou", 0.0))
    gt_area = float(metrics.get("gt_area_pct", 0.0))
    pred_area = float(metrics.get("pred_area_pct", 0.0))
    pred_components = int(metrics.get("pred_component_count", 0))
    ratio = metrics.get("mask_area_ratio_pred_over_gt")
    ratio_value = float(ratio) if ratio is not None else 0.0
    if pred_area <= 0.0:
        types.append("empty_prediction")
    if gt_area > 0.0 and gt_area <= float(cfg.get("tiny_gt_area_pct_threshold", 0.25)):
        types.append("tiny_gt_mask")
    if iou < 0.15 and pred_area > 0.0:
        types.append("wrong_region")
    if gt_area > 0.0 and pred_area > 0.0 and ratio_value <= float(cfg.get("underseg_ratio_threshold", 0.5)):
        types.append("undersegmented")
    if gt_area > 0.0 and ratio_value >= float(cfg.get("overseg_ratio_threshold", 2.0)):
        types.append("oversegmented")
    if pred_components >= int(cfg.get("fragmented_component_threshold", 4)):
        types.append("fragmented_prediction")
    ordered = [
        "severe_iou_fail",
        "low_iou",
        "weak_iou",
        "good_iou",
        "mid_iou",
        "empty_prediction",
        "undersegmented",
        "oversegmented",
        "wrong_region",
        "tiny_gt_mask",
        "fragmented_prediction",
    ]
    return [name for name in ordered if name in set(types)]


def _inline_prediction(sample: dict[str, Any], shape: tuple[int, int], threshold: float) -> list[int] | None:
    for key in ("long256_mask", "baseline_mask", "pred_mask", "predicted_mask", "mask_prediction"):
        if sample.get(key) is not None:
            mask, mask_shape = to_binary_mask(sample[key], threshold, sample.get("pred_mask_size") or sample.get("mask_size") or shape)
            return resize_binary_mask_nearest(mask, mask_shape, shape) if mask_shape != shape else mask
    return None


def mining_record_for_sample(sample: dict[str, Any], index: int, config: dict[str, Any], approved_roots: list[str]) -> dict[str, Any]:
    gt_mask, shape, mask_path = load_gt_mask_for_sample(sample, approved_roots)
    pred_mask = _inline_prediction(sample, shape, float(config.get("mask_threshold", 0.5)))
    pred_source = "manifest_predicted_mask"
    if pred_mask is None and config.get("run_long256_report") is True:
        predictions = run_models_for_sample(config, sample, shape)
        pred_mask = list(predictions.get("long256", {}).get("pred_mask", []))
        pred_source = "long256_report"
    if pred_mask is None:
        pred_mask = [0] * len(gt_mask)
        pred_source = "missing_prediction_empty_mask"
    metrics = compute_mask_metrics(gt_mask, pred_mask, shape)
    failure_types = classify_failure_types(metrics, config)
    return {
        "sample_id": sample_id(sample, index),
        "image_path": sample.get("image_path"),
        "mask_path": mask_path or mask_path_for_sample(sample),
        "gt_shape": {"width": int(shape[0]), "height": int(shape[1])},
        "prediction_source": pred_source,
        **metrics,
        "mining_bucket": classify_iou_bucket(metrics),
        "failure_types": failure_types,
        "_gt_mask": gt_mask,
        "_pred_mask": pred_mask,
        "_gt_shape": shape,
    }


def mask_bounding_box(mask: list[int], shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    width, height = shape
    points = [(idx % width, idx // width) for idx, value in enumerate(mask) if value]
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def crop_box_centered(cx: int, cy: int, width: int, height: int, tile_size: int) -> list[int]:
    crop_w = min(tile_size, width)
    crop_h = min(tile_size, height)
    x1 = max(0, min(width - crop_w, int(round(cx - crop_w / 2))))
    y1 = max(0, min(height - crop_h, int(round(cy - crop_h / 2))))
    return [int(x1), int(y1), int(x1 + crop_w), int(y1 + crop_h)]


def jittered_crop_boxes(mask: list[int], shape: tuple[int, int], tile_size: int, jitter_count: int, rng: random.Random) -> list[list[int]]:
    width, height = shape
    bbox = mask_bounding_box(mask, shape)
    if bbox is None:
        return []
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    boxes = [crop_box_centered(cx, cy, width, height, tile_size)]
    max_jitter = max(1, tile_size // 8)
    for _ in range(jitter_count):
        boxes.append(crop_box_centered(cx + rng.randint(-max_jitter, max_jitter), cy + rng.randint(-max_jitter, max_jitter), width, height, tile_size))
    unique: list[list[int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for box in boxes:
        key = tuple(box)
        if key not in seen:
            unique.append(box)
            seen.add(key)
    return unique


def mask_area_pct_in_crop(mask: list[int], shape: tuple[int, int], crop_box: list[int]) -> float:
    width, _height = shape
    x1, y1, x2, y2 = crop_box
    area = max(1, (x2 - x1) * (y2 - y1))
    active = 0
    for y in range(y1, y2):
        for x in range(x1, x2):
            if mask[y * width + x]:
                active += 1
    return float((active / area) * 100.0)


def positive_tile_records(record: dict[str, Any], config: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    shape = tuple(record.get("_gt_shape") or (record.get("gt_shape", {}).get("width"), record.get("gt_shape", {}).get("height")))
    gt_mask = list(record.get("_gt_mask", []))
    if not gt_mask or not shape[0] or not shape[1]:
        return []
    bucket = str(record.get("mining_bucket", "unknown"))
    jitter_count = int(config.get("positive_jitter_count", 2))
    boxes = jittered_crop_boxes(gt_mask, (int(shape[0]), int(shape[1])), int(config.get("tile_size", 512)), jitter_count, rng)
    factor = 1
    if bucket == "severe_iou_fail":
        factor = max(factor, int(config.get("severe_oversample_factor", 3)))
    elif bucket == "low_iou":
        factor = max(factor, int(config.get("low_iou_oversample_factor", 2)))
    repeated = boxes * factor
    records: list[dict[str, Any]] = []
    tile_size = int(config.get("tile_size", 512))
    for index, box in enumerate(repeated):
        records.append({
            "source_sample_id": record.get("sample_id"),
            "source_image_path": record.get("image_path"),
            "source_mask_path": record.get("mask_path"),
            "crop_box": box,
            "tile_size": tile_size,
            "tile_class": "positive_tampered",
            "mining_bucket": bucket,
            "failure_types": record.get("failure_types", []),
            "gt_area_pct_in_crop": mask_area_pct_in_crop(gt_mask, (int(shape[0]), int(shape[1])), box),
            "expected_mask_type": "cropped_gt_mask",
            "record_index": index,
        })
    return records


def _sample_size(sample: dict[str, Any], default_size: int) -> tuple[int, int]:
    width = sample.get("width")
    height = sample.get("height")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return width, height
    if sample.get("image_size"):
        size = sample["image_size"]
        if isinstance(size, list) and len(size) >= 2:
            return int(size[0]), int(size[1])
    return default_size, default_size


def negative_tile_records(samples: list[dict[str, Any]], config: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    per_sample = int(config.get("negative_random_count", 1))
    tile_size = int(config.get("tile_size", 512))
    for sample_index, sample in enumerate(samples):
        width, height = _sample_size(sample, tile_size)
        crop_w = min(tile_size, width)
        crop_h = min(tile_size, height)
        tile_class = "negative_real" if sample_class(sample) == "real" else "negative_synthetic"
        for crop_index in range(per_sample):
            x1 = rng.randint(0, max(0, width - crop_w))
            y1 = rng.randint(0, max(0, height - crop_h))
            records.append({
                "source_sample_id": sample_id(sample, sample_index),
                "source_image_path": sample.get("image_path"),
                "crop_box": [x1, y1, x1 + crop_w, y1 + crop_h],
                "tile_size": tile_size,
                "tile_class": tile_class,
                "expected_mask_type": "empty_mask",
                "record_index": crop_index,
            })
    return records


def hard_negative_tile_records(hard_records: list[dict[str, Any]], config: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    tile_size = int(config.get("tile_size", 512))
    limit = int(config.get("hard_negative_count", len(hard_records)))
    for index, item in enumerate(hard_records[:limit]):
        width = int(item.get("width") or item.get("image_width") or tile_size)
        height = int(item.get("height") or item.get("image_height") or tile_size)
        bbox = item.get("pred_bbox") or item.get("bbox") or item.get("false_positive_bbox")
        if isinstance(bbox, dict):
            cx = int((int(bbox.get("x0", 0)) + int(bbox.get("x1", 0))) / 2)
            cy = int((int(bbox.get("y0", 0)) + int(bbox.get("y1", 0))) / 2)
        else:
            cx = rng.randint(0, max(0, width - 1))
            cy = rng.randint(0, max(0, height - 1))
        records.append({
            "source_sample_id": item.get("sample_id") or f"hard_negative_{index:06d}",
            "source_image_path": item.get("image_path"),
            "crop_box": crop_box_centered(cx, cy, width, height, tile_size),
            "tile_size": tile_size,
            "tile_class": "hard_negative",
            "expected_mask_type": "empty_mask",
            "mining_bucket": item.get("mining_bucket") or item.get("failure_type") or "hard_negative_false_positive",
            "record_index": index,
        })
    return records


def build_tile_manifest(config: dict[str, Any], manifest: dict[str, Any], mining_records: list[dict[str, Any]], hard_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rng = random.Random(int(config.get("seed", 23)))
    tile_records: list[dict[str, Any]] = []
    for record in mining_records:
        tile_records.extend(positive_tile_records(record, config, rng))
    negative_count = int(config.get("negative_sample_count", max(1, len(mining_records))))
    negative_samples = select_negative_samples(manifest, negative_count)
    tile_records.extend(negative_tile_records(negative_samples, config, rng))
    tile_records.extend(hard_negative_tile_records(hard_records or [], config, rng))
    counts: dict[str, int] = {}
    for record in tile_records:
        counts[record["tile_class"]] = counts.get(record["tile_class"], 0) + 1
    return {
        "marker": MARKER,
        "schema_version": config.get("schema_version", "1.0"),
        "source_train_manifest_path": config.get("train_manifest_path"),
        "source_gt_iou_records_path": config.get("gt_iou_records_path"),
        "source_hard_negative_records_path": config.get("hard_negative_records_path"),
        "tile_size": int(config.get("tile_size", 512)),
        "crop_policy": "gt_centered_with_jitter_manifest_only",
        "write_cropped_images": False,
        "record_count": len(tile_records),
        "counts_by_tile_class": counts,
        "records": tile_records,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }


def enrich_loaded_mining_records(records: list[dict[str, Any]], manifest: dict[str, Any], config: dict[str, Any], approved_roots: list[str]) -> list[dict[str, Any]]:
    by_id = {sample_id(sample, index): sample for index, sample in enumerate(manifest.get("samples", [])) if isinstance(sample, dict)}
    enriched: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        item = dict(record)
        sid = str(item.get("sample_id") or f"record_{index:06d}")
        sample = by_id.get(sid)
        if sample and "_gt_mask" not in item and has_gt_mask(sample):
            gt_mask, shape, loaded_mask_path = load_gt_mask_for_sample(sample, approved_roots)
            item.setdefault("image_path", sample.get("image_path"))
            item.setdefault("mask_path", loaded_mask_path or mask_path_for_sample(sample))
            item.setdefault("gt_shape", {"width": int(shape[0]), "height": int(shape[1])})
            item["_gt_mask"] = gt_mask
            item["_gt_shape"] = shape
        if "mining_bucket" not in item and item.get("iou") is not None:
            item["mining_bucket"] = classify_iou_bucket(item)
        if "failure_types" not in item and item.get("iou") is not None:
            item["failure_types"] = classify_failure_types(item, config)
        enriched.append(item)
    return enriched


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def bucket_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    names = [
        "severe_iou_fail_cases",
        "low_iou_cases",
        "weak_iou_cases",
        "empty_prediction_cases",
        "wrong_region_cases",
        "undersegmented_cases",
        "oversegmented_cases",
        "tiny_gt_mask_cases",
        "fragmented_prediction_cases",
    ]
    buckets = {name: [] for name in names}
    for record in records:
        types = set(record.get("failure_types", []))
        bucket = record.get("mining_bucket")
        if bucket == "severe_iou_fail":
            buckets["severe_iou_fail_cases"].append(record)
        if bucket == "low_iou":
            buckets["low_iou_cases"].append(record)
        if bucket == "weak_iou":
            buckets["weak_iou_cases"].append(record)
        mapping = {
            "empty_prediction": "empty_prediction_cases",
            "wrong_region": "wrong_region_cases",
            "undersegmented": "undersegmented_cases",
            "oversegmented": "oversegmented_cases",
            "tiny_gt_mask": "tiny_gt_mask_cases",
            "fragmented_prediction": "fragmented_prediction_cases",
        }
        for failure_type, bucket_name in mapping.items():
            if failure_type in types:
                buckets[bucket_name].append(record)
    return buckets


def summarize_mining(records: list[dict[str, Any]], buckets: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    ious = [float(record.get("iou", 0.0)) for record in records]
    dices = [float(record.get("dice", 0.0)) for record in records]
    return {
        "marker": MARKER,
        "record_count": len(records),
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "median_iou": sorted(ious)[len(ious) // 2] if ious else 0.0,
        "mean_dice": sum(dices) / len(dices) if dices else 0.0,
        "bucket_counts": {key: len(value) for key, value in buckets.items()},
        "max_samples": int(config.get("max_samples", 0)),
        "primary_model": "clean_long256",
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }


def _write_json(path: Path, value: Any) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_public_record(record), sort_keys=True))
            handle.write("\n")
    return str(path)


def write_visual_audit(output_root: Path, records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    top_n = int(config.get("visual_top_n", 0))
    if top_n <= 0:
        return {"visual_audit_written": False, "paths": []}
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return {"visual_audit_written": False, "paths": [], "warning": "PIL not available"}
    root = output_root / "visual_audit"
    root.mkdir(parents=True, exist_ok=True)
    selected = [record for record in records if record.get("mining_bucket") in {"severe_iou_fail", "low_iou"}][:top_n]
    paths: list[str] = []
    for record in selected:
        try:
            with Image.open(record["image_path"]) as image:
                base = image.convert("RGB").resize((192, 192))
            shape = tuple(record["_gt_shape"])
            gt = list(record["_gt_mask"])
            pred = list(record["_pred_mask"])
            panels = [
                ("input", base),
                ("GT", _red_overlay(Image, base, gt, shape)),
                ("long256", _red_overlay(Image, base, pred, shape)),
                ("agreement", _agreement_overlay(Image, base, gt, pred, shape)),
            ]
            sheet = Image.new("RGB", (192 * len(panels), 216), (255, 255, 255))
            draw = ImageDraw.Draw(sheet)
            for idx, (label, panel) in enumerate(panels):
                x = idx * 192
                draw.text((x + 6, 6), label, fill=(0, 0, 0))
                sheet.paste(panel, (x, 24))
            path = root / f"{record['mining_bucket']}_{record['sample_id']}.jpg"
            sheet.save(path, quality=90)
            paths.append(str(path))
        except Exception:
            continue
    return {"visual_audit_written": bool(paths), "paths": paths}


def _red_overlay(Image: Any, base: Any, mask: list[int], shape: tuple[int, int]) -> Any:
    mask_image = Image.new("L", shape)
    mask_image.putdata([255 if value else 0 for value in mask])
    resample_nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    mask_image = mask_image.resize(base.size, resample_nearest)
    pixels = list(base.getdata())
    blended = []
    for pixel, active in zip(pixels, mask_image.getdata()):
        blended.append(tuple(int(round(0.55 * pixel[i] + 0.45 * (255 if i == 0 else 0))) for i in range(3)) if active else pixel)
    out = Image.new("RGB", base.size)
    out.putdata(blended)
    return out


def _agreement_overlay(Image: Any, base: Any, gt: list[int], pred: list[int], shape: tuple[int, int]) -> Any:
    color = []
    for g, p in zip(gt, pred):
        if g and p:
            color.append((0, 180, 0))
        elif p and not g:
            color.append((255, 0, 0))
        elif g and not p:
            color.append((0, 80, 255))
        else:
            color.append((0, 0, 0))
    color_image = Image.new("RGB", shape)
    color_image.putdata(color)
    active = Image.new("L", shape)
    active.putdata([255 if g or p else 0 for g, p in zip(gt, pred)])
    resample_nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    color_image = color_image.resize(base.size, resample_nearest)
    active = active.resize(base.size, resample_nearest)
    blended = []
    for pixel, overlay, enabled in zip(base.getdata(), color_image.getdata(), active.getdata()):
        blended.append(tuple(int(round(0.45 * pixel[i] + 0.55 * overlay[i])) for i in range(3)) if enabled else pixel)
    out = Image.new("RGB", base.size)
    out.putdata(blended)
    return out


def run_gt_iou_tile_builder(config: dict[str, Any]) -> dict[str, Any]:
    errors = validate_gt_iou_tile_builder_config(config, require_exists=True)
    if errors:
        raise GTIoUTileBuilderError("GT-IoU tile-builder config validation failed:\n" + "\n".join(errors))
    if config.get("config_kind") != APPROVED_KIND:
        raise GTIoUTileBuilderError(f"execution requires {APPROVED_KIND}")
    output_root = _real(config["output_root"])
    if _inside_repo(output_root):
        raise GTIoUTileBuilderError("output_root must be outside repository")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(config["train_manifest_path"])
    approved_roots = list(config.get("approved_input_roots", []))
    if config.get("gt_iou_records_path"):
        loaded_records = load_jsonl(config["gt_iou_records_path"])[: int(config["max_samples"])]
        mining_records = enrich_loaded_mining_records(loaded_records, manifest, config, approved_roots)
    else:
        samples = select_tampered_train_samples(manifest, int(config["max_samples"]))
        mining_records = [mining_record_for_sample(sample, index, config, approved_roots) for index, sample in enumerate(samples)]
    hard_records = load_jsonl(config.get("hard_negative_records_path"))
    buckets = bucket_records(mining_records)
    tile_manifest = build_tile_manifest(config, manifest, mining_records, hard_records)
    mining_summary = summarize_mining(mining_records, buckets, config)
    tile_summary = {
        "marker": MARKER,
        "tile_manifest_path": str(output_root / "tile_localization_manifest.json"),
        "record_count": tile_manifest["record_count"],
        "counts_by_tile_class": tile_manifest["counts_by_tile_class"],
        "write_cropped_images": False,
        "no_training": True,
        "no_sns_augmentation": True,
    }
    visual_summary = write_visual_audit(output_root, mining_records, config)

    output_paths: dict[str, str] = {}
    output_paths["gt_iou_train_records"] = _write_jsonl(output_root / "gt_iou_train_records.jsonl", mining_records)
    output_paths["gt_iou_train_summary"] = _write_json(output_root / "gt_iou_train_summary.json", mining_summary)
    for bucket_name, items in buckets.items():
        output_paths[bucket_name] = _write_json(output_root / f"{bucket_name}.json", [_public_record(item) for item in items])
    output_paths["tile_localization_manifest"] = _write_json(output_root / "tile_localization_manifest.json", tile_manifest)
    output_paths["tile_manifest_summary"] = _write_json(output_root / "tile_manifest_summary.json", tile_summary)
    output_paths["artifact_manifest"] = str(output_root / "artifact_manifest.json")
    artifact_manifest = {
        "marker": MARKER,
        "output_paths": output_paths,
        "visual_audit": visual_summary,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }
    _write_json(output_root / "artifact_manifest.json", artifact_manifest)
    return {
        "marker": MARKER,
        "output_root": str(output_root),
        "mining_summary": mining_summary,
        "tile_manifest_summary": tile_summary,
        "visual_audit": visual_summary,
        "output_paths": output_paths,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }


def artifact_manifest_schema_ok(value: dict[str, Any]) -> bool:
    return (
        value.get("marker") == MARKER
        and isinstance(value.get("output_paths"), dict)
        and value.get("no_download") is True
        and value.get("no_network") is True
        and value.get("no_training") is True
        and value.get("no_sns_augmentation") is True
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value
