"""GT-IoU localization mining for pre-SNS v3 checkpoints."""

from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

from .pre_sns_v3_dual_scale_report import (
    _err,
    _inside_repo,
    _is_under,
    _real,
    _validate_absolute_path,
    _validate_under_roots,
    binary_mask,
    mask_components,
    mask_stats,
    postprocess_mask,
    resize_binary_mask_nearest,
)
from .pre_sns_v3_model import CLASS_LABELS, FAMILY_LABELS
from .pre_sns_v3_report import _image_tensor, confidence_map, load_v3_model

MARKER = "PRE_SNS_V3_GT_IOU_LOCALIZATION_MINING_OK"
CONFIG_OK_MARKER = "PRE_SNS_V3_GT_IOU_LOCALIZATION_MINING_CONFIG_OK"
APPROVED_KIND = "approved_pre_sns_v3_gt_iou_localization_mining"
EXAMPLE_KIND = "example_symbolic"


class GTIoUMiningError(ValueError):
    """Raised when GT-IoU mining inputs or configs are invalid."""


def load_gt_iou_mining_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise GTIoUMiningError("GT-IoU localization mining config root must be a JSON object")
    return raw


def _config_output_root(raw: dict[str, Any]) -> Any:
    return raw.get("output_root", raw.get("approved_output_root"))


def validate_gt_iou_mining_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
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
        "threshold_tau",
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
    for flag in ("no_download", "no_network", "no_training", "no_sns_augmentation"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if raw.get("no_checkpoint_writes", True) is not True:
        errors.append(_err("no_checkpoint_writes must be true when present"))

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

    approved_roots = list(roots) + list(checkpoint_roots)
    errors.extend(_validate_under_roots(raw.get("manifest_path"), "manifest_path", roots, require_file=require_exists))
    for field in ("long256_checkpoint_path", "long224_checkpoint_path", "refined_checkpoint_path"):
        if field == "long256_checkpoint_path" or raw.get(field):
            errors.extend(_validate_under_roots(raw.get(field), field, approved_roots, require_file=require_exists, allow_parts={"checkpoints"}))

    output_root = _config_output_root(raw)
    errors.extend(_validate_absolute_path(output_root, "output_root", require_dir_parent=kind == APPROVED_KIND and require_exists))
    if isinstance(output_root, str) and output_root.startswith("/") and any(_is_under(output_root, root) for root in roots):
        errors.append(_err("output_root must not be under an approved input root"))

    if isinstance(raw.get("max_samples"), bool) or not isinstance(raw.get("max_samples"), int) or raw.get("max_samples", 0) <= 0:
        errors.append(_err("max_samples must be a positive integer"))
    for field in (
        "threshold_tau",
        "mask_threshold",
        "low_iou_threshold",
        "success_iou_threshold",
        "tiny_gt_area_pct_threshold",
        "underseg_ratio_threshold",
        "overseg_ratio_threshold",
        "fragmented_component_threshold",
    ):
        if field in raw:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
                errors.append(_err(f"{field} must be a non-negative number"))
    for field in ("long224_max_image_size", "long256_max_image_size", "refined_max_image_size", "visual_top_n_per_bucket"):
        if field in raw and raw[field] is not None:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(_err(f"{field} must be a non-negative integer"))
    if kind == APPROVED_KIND and raw.get("execution_mode") != "approved_local_pre_sns_v3_gt_iou_localization_mining":
        errors.append(_err("execution_mode must be approved_local_pre_sns_v3_gt_iou_localization_mining"))
    if kind == EXAMPLE_KIND and raw.get("execution_mode") != "example_only":
        errors.append(_err("example execution_mode must be example_only"))
    return errors


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise GTIoUMiningError("manifest must be a JSON object with a samples list")
    return manifest


def _sample_class(sample: dict[str, Any]) -> str:
    for key in ("class_label", "gt_class", "label", "class"):
        if sample.get(key) is not None:
            return str(sample[key]).lower()
    if sample.get("is_tampered") is True:
        return "tampered"
    return ""


def _sample_mask_path(sample: dict[str, Any]) -> str | None:
    for key in ("mask_path", "gt_mask_path", "tamper_mask_path", "localization_mask_path"):
        value = sample.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def sample_has_gt_mask(sample: dict[str, Any]) -> bool:
    return _sample_mask_path(sample) is not None or sample.get("gt_mask") is not None or sample.get("mask") is not None


def iter_tampered_samples_with_gt_mask(manifest: dict[str, Any], max_samples: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for sample in manifest.get("samples", []):
        if not isinstance(sample, dict):
            continue
        if _sample_class(sample) == "tampered" and sample_has_gt_mask(sample):
            selected.append(sample)
            if len(selected) >= max_samples:
                break
    return selected


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


def infer_shape(mask: Any, mask_size: tuple[int, int] | int | None = None) -> tuple[int, int]:
    flat = flatten_mask(mask)
    if isinstance(mask_size, tuple):
        width, height = int(mask_size[0]), int(mask_size[1])
    elif isinstance(mask_size, int):
        width = height = int(mask_size)
    elif isinstance(mask, list) and mask and all(isinstance(row, list) for row in mask):
        height = len(mask)
        width = len(mask[0])
    else:
        side = int(math.sqrt(len(flat)))
        if side * side != len(flat):
            raise GTIoUMiningError("mask length is not square; provide mask size")
        width = height = side
    if width <= 0 or height <= 0 or width * height != len(flat):
        raise GTIoUMiningError("mask shape does not match mask length")
    return width, height


def to_binary_mask(mask: Any, threshold: float = 0.5, mask_size: tuple[int, int] | int | None = None) -> tuple[list[int], tuple[int, int]]:
    flat = flatten_mask(mask)
    shape = infer_shape(flat, mask_size)
    return [1 if value >= threshold else 0 for value in flat], shape


def iou_score(gt: list[int], pred: list[int]) -> float:
    if len(gt) != len(pred) or not gt:
        return 0.0
    intersection = sum(1 for g, p in zip(gt, pred) if g and p)
    union = sum(1 for g, p in zip(gt, pred) if g or p)
    return 1.0 if union == 0 else float(intersection / union)


def dice_score(gt: list[int], pred: list[int]) -> float:
    if len(gt) != len(pred) or not gt:
        return 0.0
    intersection = sum(1 for g, p in zip(gt, pred) if g and p)
    denom = sum(1 for value in gt if value) + sum(1 for value in pred if value)
    return 1.0 if denom == 0 else float((2.0 * intersection) / denom)


def compute_model_metrics(gt_mask: list[int], pred_mask: list[int], shape: tuple[int, int]) -> dict[str, Any]:
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


def classify_model_failure(metrics: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    failure_types: list[str] = []
    iou = float(metrics.get("iou", 0.0))
    gt_area = float(metrics.get("gt_area_pct", 0.0))
    pred_area = float(metrics.get("pred_area_pct", 0.0))
    pred_components = int(metrics.get("pred_component_count", 0))
    ratio = metrics.get("mask_area_ratio_pred_over_gt")
    ratio_value = float(ratio) if ratio is not None else 0.0
    if pred_area <= 0.0:
        failure_types.append("empty_prediction")
    if gt_area > 0.0 and gt_area <= float(thresholds.get("tiny_gt_area_pct_threshold", 0.25)):
        failure_types.append("tiny_gt_mask")
    if iou < float(thresholds.get("low_iou_threshold", 0.3)) and pred_area > 0.0:
        failure_types.append("low_iou_wrong_region")
    if gt_area > 0.0 and pred_area > 0.0 and ratio_value <= float(thresholds.get("underseg_ratio_threshold", 0.5)):
        failure_types.append("undersegmented")
    if gt_area > 0.0 and ratio_value >= float(thresholds.get("overseg_ratio_threshold", 2.0)):
        failure_types.append("oversegmented")
    if pred_components >= int(float(thresholds.get("fragmented_component_threshold", 4.0))):
        failure_types.append("fragmented_prediction")
    if iou >= float(thresholds.get("success_iou_threshold", 0.5)):
        failure_types.append("long256_succeeded")
    return failure_types


def classify_record_failure(record: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    types = set(record.get("models", {}).get("long256", {}).get("failure_types", []))
    success_threshold = float(thresholds.get("success_iou_threshold", 0.5))
    model_metrics = record.get("models", {})
    available = [value for value in model_metrics.values() if isinstance(value, dict) and value.get("available") is True]
    if available and all(float(value.get("iou", 0.0)) < success_threshold for value in available):
        types.add("all_models_failed")
    long224 = model_metrics.get("long224", {})
    long256 = model_metrics.get("long256", {})
    if long224.get("available") is True and float(long256.get("iou", 0.0)) < success_threshold <= float(long224.get("iou", 0.0)):
        types.add("long256_failed_long224_succeeded")
    if float(long256.get("iou", 0.0)) >= success_threshold:
        types.add("long256_succeeded")
    ordered = [
        "empty_prediction",
        "low_iou_wrong_region",
        "undersegmented",
        "oversegmented",
        "tiny_gt_mask",
        "fragmented_prediction",
        "all_models_failed",
        "long256_failed_long224_succeeded",
        "long256_succeeded",
    ]
    return [name for name in ordered if name in types]


def default_thresholds(config: dict[str, Any] | None = None) -> dict[str, float]:
    cfg = config or {}
    return {
        "low_iou_threshold": float(cfg.get("low_iou_threshold", 0.3)),
        "success_iou_threshold": float(cfg.get("success_iou_threshold", 0.5)),
        "tiny_gt_area_pct_threshold": float(cfg.get("tiny_gt_area_pct_threshold", 0.25)),
        "underseg_ratio_threshold": float(cfg.get("underseg_ratio_threshold", 0.5)),
        "overseg_ratio_threshold": float(cfg.get("overseg_ratio_threshold", 2.0)),
        "fragmented_component_threshold": float(cfg.get("fragmented_component_threshold", 4.0)),
    }


def build_buckets(records: list[dict[str, Any]], thresholds: dict[str, float] | None = None) -> dict[str, list[dict[str, Any]]]:
    thresholds = thresholds or default_thresholds()
    buckets = {
        "low_iou_long256": [],
        "empty_prediction_tampered": [],
        "all_models_failed": [],
        "long256_failed_long224_succeeded": [],
        "tiny_gt_mask_cases": [],
        "oversegmented_cases": [],
        "undersegmented_cases": [],
    }
    low_iou = float(thresholds.get("low_iou_threshold", 0.3))
    for record in records:
        types = set(record.get("failure_types", []))
        long256 = record.get("models", {}).get("long256", {})
        if float(long256.get("iou", 0.0)) < low_iou:
            buckets["low_iou_long256"].append(record)
        if "empty_prediction" in types:
            buckets["empty_prediction_tampered"].append(record)
        if "all_models_failed" in types:
            buckets["all_models_failed"].append(record)
        if "long256_failed_long224_succeeded" in types:
            buckets["long256_failed_long224_succeeded"].append(record)
        if "tiny_gt_mask" in types:
            buckets["tiny_gt_mask_cases"].append(record)
        if "oversegmented" in types:
            buckets["oversegmented_cases"].append(record)
        if "undersegmented" in types:
            buckets["undersegmented_cases"].append(record)
    return buckets


def summarize_models(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in ("long256", "long224", "refined"):
        values = [record.get("models", {}).get(name, {}) for record in records]
        available = [value for value in values if value.get("available") is True]
        ious = [float(value.get("iou", 0.0)) for value in available]
        dices = [float(value.get("dice", 0.0)) for value in available]
        summary[name] = {
            "available_count": len(available),
            "mean_iou": float(sum(ious) / len(ious)) if ious else 0.0,
            "median_iou": float(sorted(ious)[len(ious) // 2]) if ious else 0.0,
            "mean_dice": float(sum(dices) / len(dices)) if dices else 0.0,
            "empty_prediction_count": sum(1 for value in available if "empty_prediction" in value.get("failure_types", [])),
            "low_iou_count": sum(1 for value in available if "low_iou_wrong_region" in value.get("failure_types", [])),
        }
    return summary


def _sample_id(sample: dict[str, Any], index: int) -> str:
    raw = str(sample.get("sample_id") or sample.get("id") or f"sample_{index:06d}")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)[:80]


def _load_mask_from_path(mask_path: str, approved_roots: list[str]) -> tuple[list[int], tuple[int, int]]:
    if not any(_is_under(mask_path, root) for root in approved_roots):
        raise GTIoUMiningError("ground-truth mask path is outside approved input roots")
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("PIL is required to load ground-truth mask image files") from exc
    with Image.open(mask_path) as image:
        mask_image = image.convert("L")
        width, height = mask_image.size
        values = list(mask_image.getdata())
    return [1 if int(value) > 0 else 0 for value in values], (width, height)


def load_gt_mask_for_sample(sample: dict[str, Any], approved_roots: list[str]) -> tuple[list[int], tuple[int, int], str | None]:
    mask_path = _sample_mask_path(sample)
    if mask_path:
        mask, shape = _load_mask_from_path(mask_path, approved_roots)
        return mask, shape, mask_path
    value = sample.get("gt_mask", sample.get("mask"))
    mask, shape = to_binary_mask(value, 0.5, sample.get("mask_size"))
    return mask, shape, None


def _prediction_from_report(report: dict[str, Any], gt_shape: tuple[int, int], threshold_tau: float) -> dict[str, Any]:
    if float(report.get("tampered_score", 0.0)) < threshold_tau:
        pred = [0] * (gt_shape[0] * gt_shape[1])
        source_shape = gt_shape
    else:
        pred = [1 if value else 0 for value in report.get("processed_mask", [])]
        source_shape = tuple(report.get("mask_shape") or gt_shape)
        if source_shape != gt_shape:
            pred = resize_binary_mask_nearest(pred, source_shape, gt_shape)
    return {
        "available": True,
        "class": report.get("class"),
        "tampered_score": float(report.get("tampered_score", 0.0)),
        "localization_head": "activated" if float(report.get("tampered_score", 0.0)) >= threshold_tau else "skipped_below_threshold",
        "pred_mask": pred,
        "pred_mask_shape": gt_shape,
    }


def _run_one_gt_iou_model(torch: Any, Image: Any, config: dict[str, Any], sample: dict[str, Any], checkpoint_field: str, max_size_field: str, label: str) -> dict[str, Any]:
    device = "cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    model, checkpoint, checkpoint_image_size = load_v3_model(torch, config[checkpoint_field], device)
    image_size = int(config.get(max_size_field) or config.get("max_image_size") or checkpoint_image_size)
    image = _image_tensor(torch, Image, sample["image_path"], image_size, device)
    with torch.no_grad():
        outputs = model(image)
        class_probs = torch.softmax(outputs["class_logits"][0], dim=0).detach().cpu().tolist()
        tamper_probs = torch.softmax(outputs["tamper_binary_logits"][0], dim=0).detach().cpu().tolist()
        family_probs = torch.softmax(outputs["family_logits"][0], dim=0).detach().cpu().tolist()
        mask_probs = torch.sigmoid(outputs["localization_logits"][0]).detach().cpu()
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
        "checkpoint_image_size": int(checkpoint_image_size),
        "image_size": int(image_size),
        "class": class_label,
        "class_conf": confidence_map(CLASS_LABELS, class_probs),
        "family": family_label,
        "family_conf": confidence_map(FAMILY_LABELS, family_probs),
        "tampered_score": float(tamper_probs[1]),
        "processed_mask": processed,
        "processed_mask_stats": mask_stats(processed, shape),
        "mask_shape": shape,
        "device": device,
        "checkpoint_selected_tau": checkpoint.get("selected_tau"),
    }


def run_models_for_sample(config: dict[str, Any], sample: dict[str, Any], gt_shape: tuple[int, int]) -> dict[str, dict[str, Any]]:
    from .pre_sns_v3_dual_scale_report import _runtime_deps

    torch, Image, _ImageDraw = _runtime_deps()
    model_specs = [
        ("long256", "long256_checkpoint_path", "long256_max_image_size"),
        ("long224", "long224_checkpoint_path", "long224_max_image_size"),
        ("refined", "refined_checkpoint_path", "refined_max_image_size"),
    ]
    predictions: dict[str, dict[str, Any]] = {}
    for label, checkpoint_field, max_size_field in model_specs:
        if not config.get(checkpoint_field):
            predictions[label] = {"available": False}
            continue
        report = _run_one_gt_iou_model(torch, Image, config, sample, checkpoint_field, max_size_field, label)
        predictions[label] = _prediction_from_report(report, gt_shape, float(config.get("threshold_tau", 0.5)))
    return predictions


def record_for_sample(
    sample: dict[str, Any],
    index: int,
    gt_mask: list[int],
    gt_shape: tuple[int, int],
    mask_path: str | None,
    predictions: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = default_thresholds(config)
    models: dict[str, dict[str, Any]] = {}
    for name in ("long256", "long224", "refined"):
        pred = predictions.get(name, {"available": False})
        if pred.get("available") is not True:
            models[name] = {"available": False}
            continue
        metrics = compute_model_metrics(gt_mask, pred.get("pred_mask", []), gt_shape)
        failure_types = classify_model_failure(metrics, thresholds)
        models[name] = {
            "available": True,
            "class": pred.get("class"),
            "tampered_score": pred.get("tampered_score"),
            "localization_head": pred.get("localization_head"),
            **metrics,
            "failure_types": failure_types,
        }
    record = {
        "sample_id": _sample_id(sample, index),
        "image_path": sample.get("image_path"),
        "mask_path": mask_path,
        "gt_shape": {"width": int(gt_shape[0]), "height": int(gt_shape[1])},
        "models": models,
    }
    long256 = models.get("long256", {})
    for key in (
        "iou",
        "dice",
        "gt_area_pct",
        "pred_area_pct",
        "gt_component_count",
        "pred_component_count",
        "largest_gt_component_area_pct",
        "largest_pred_component_area_pct",
        "mask_area_ratio_pred_over_gt",
    ):
        record[key] = long256.get(key)
    record["failure_types"] = classify_record_failure(record, thresholds)
    if int(config.get("visual_top_n_per_bucket", 0)) > 0:
        record["_visual_masks"] = {
            "gt_mask": gt_mask,
            "gt_shape": gt_shape,
            "pred_masks": {
                name: predictions.get(name, {}).get("pred_mask")
                for name in ("long256", "long224", "refined")
                if predictions.get(name, {}).get("available") is True
            },
        }
    return record


def _write_json(path: Path, value: Any) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _public_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_public_record(record) for record in records]


def _overlay_mask(Image: Any, base: Any, mask: list[int], shape: tuple[int, int], alpha: float = 0.45) -> Any:
    if not mask or sum(1 for value in mask if value) == 0:
        return base.copy()
    mask_image = Image.new("L", shape)
    mask_image.putdata([255 if value else 0 for value in mask])
    resample_nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    mask_resized = mask_image.resize(base.size, resample_nearest)
    pixels = list(base.getdata())
    mask_pixels = list(mask_resized.getdata())
    blended = []
    for pixel, active in zip(pixels, mask_pixels):
        if active:
            blended.append(tuple(int(round((1.0 - alpha) * pixel[i] + alpha * (255 if i == 0 else 0))) for i in range(3)))
        else:
            blended.append(pixel)
    panel = Image.new("RGB", base.size)
    panel.putdata(blended)
    return panel


def _agreement_overlay(Image: Any, base: Any, gt_mask: list[int], pred_mask: list[int], shape: tuple[int, int], alpha: float = 0.55) -> Any:
    if len(pred_mask) != len(gt_mask):
        pred_mask = [0] * len(gt_mask)
    color_pixels = []
    for gt_value, pred_value in zip(gt_mask, pred_mask):
        if gt_value and pred_value:
            color_pixels.append((0, 180, 0))
        elif gt_value and not pred_value:
            color_pixels.append((255, 0, 0))
        elif pred_value and not gt_value:
            color_pixels.append((0, 80, 255))
        else:
            color_pixels.append((0, 0, 0))
    color = Image.new("RGB", shape)
    color.putdata(color_pixels)
    mask = Image.new("L", shape)
    mask.putdata([255 if gt_value or pred_value else 0 for gt_value, pred_value in zip(gt_mask, pred_mask)])
    resample_nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    color = color.resize(base.size, resample_nearest)
    mask = mask.resize(base.size, resample_nearest)
    pixels = list(base.getdata())
    color_data = list(color.getdata())
    mask_data = list(mask.getdata())
    blended = []
    for pixel, overlay, active in zip(pixels, color_data, mask_data):
        if active:
            blended.append(tuple(int(round((1.0 - alpha) * pixel[i] + alpha * overlay[i])) for i in range(3)))
        else:
            blended.append(pixel)
    panel = Image.new("RGB", base.size)
    panel.putdata(blended)
    return panel


def write_visual_sheets(output_root: Path, records: list[dict[str, Any]], buckets: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    top_n = int(config.get("visual_top_n_per_bucket", 0))
    if top_n <= 0:
        return {"visual_sheets_written": False, "paths": []}
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return {"visual_sheets_written": False, "paths": [], "warning": "PIL not available"}
    by_id = {record["sample_id"]: record for record in records}
    root = output_root / "visual_sheets"
    root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for bucket_name, bucket_records in buckets.items():
        for record in bucket_records[:top_n]:
            source = by_id.get(record["sample_id"], record)
            image_path = source.get("image_path")
            if not image_path:
                continue
            try:
                with Image.open(image_path) as image:
                    base = image.convert("RGB").resize((192, 192))
                visual = source.get("_visual_masks", {})
                gt_mask = list(visual.get("gt_mask", []))
                shape = tuple(visual.get("gt_shape") or (0, 0))
                pred_masks = visual.get("pred_masks", {}) if isinstance(visual.get("pred_masks"), dict) else {}
                panels = [("input", base)]
                if gt_mask and shape != (0, 0):
                    panels.append(("GT", _overlay_mask(Image, base, gt_mask, shape)))
                for model_name in ("long256", "long224", "refined"):
                    pred_mask = pred_masks.get(model_name)
                    if pred_mask and shape != (0, 0):
                        panels.append((model_name, _overlay_mask(Image, base, list(pred_mask), shape)))
                long256_mask = list(pred_masks.get("long256", []))
                if gt_mask and long256_mask and shape != (0, 0):
                    panels.append(("agreement", _agreement_overlay(Image, base, gt_mask, long256_mask, shape)))
                sheet = Image.new("RGB", (192 * len(panels), 216), (255, 255, 255))
                draw = ImageDraw.Draw(sheet)
                for idx, (label, panel) in enumerate(panels):
                    x = idx * 192
                    sheet.paste(panel, (x, 24))
                    draw.text((x + 6, 6), label, fill=(0, 0, 0))
                path = root / f"{bucket_name}_{source['sample_id']}.jpg"
                sheet.save(path, quality=90)
                paths.append(str(path))
            except Exception:
                continue
    return {"visual_sheets_written": bool(paths), "paths": paths}


def write_mining_outputs(output_root: str | Path, records: list[dict[str, Any]], config: dict[str, Any], failed_cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    root = _real(output_root)
    if _inside_repo(root):
        raise GTIoUMiningError("output_root must be outside repository")
    root.mkdir(parents=True, exist_ok=True)
    thresholds = default_thresholds(config)
    buckets = build_buckets(records, thresholds)
    output_paths: dict[str, str] = {}
    jsonl_path = root / "localization_iou_records.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_public_record(record), sort_keys=True))
            handle.write("\n")
    output_paths["localization_iou_records"] = str(jsonl_path)
    for key, items in buckets.items():
        output_paths[key] = _write_json(root / f"{key}.json", _public_records(items))
    model_summary = summarize_models(records)
    output_paths["model_iou_summary"] = _write_json(root / "model_iou_summary.json", model_summary)
    visual_summary = write_visual_sheets(root, records, buckets, config)
    failed_cases = failed_cases or []
    if failed_cases:
        output_paths["failed_cases"] = _write_json(root / "failed_cases.json", failed_cases)
    summary = {
        "marker": MARKER,
        "schema_version": config.get("schema_version", "1.0"),
        "processed_tampered_with_gt_count": len(records),
        "failed_case_count": len(failed_cases),
        "bucket_counts": {key: len(value) for key, value in buckets.items()},
        "model_availability": {
            "long256": True,
            "long224": bool(config.get("long224_checkpoint_path")),
            "refined": bool(config.get("refined_checkpoint_path")),
        },
        "thresholds": {
            **thresholds,
            "threshold_tau": float(config.get("threshold_tau", 0.5)),
            "mask_threshold": float(config.get("mask_threshold", 0.5)),
        },
        "max_samples": int(config.get("max_samples", 0)),
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
        "visual_outputs": visual_summary,
        "output_paths": output_paths,
    }
    output_paths["localization_mining_summary"] = _write_json(root / "localization_mining_summary.json", summary)
    summary["output_paths"] = output_paths
    return summary


def _failure_record(sample: dict[str, Any], index: int, exc: BaseException) -> dict[str, Any]:
    return {
        "sample_id": sample.get("sample_id") or sample.get("id") or f"sample_{index:06d}",
        "image_path": sample.get("image_path"),
        "exception": f"{type(exc).__name__}: {exc}",
        "traceback_tail": traceback.format_exc().strip().splitlines()[-8:],
    }


def run_gt_iou_localization_mining(config: dict[str, Any]) -> dict[str, Any]:
    errors = validate_gt_iou_mining_config(config, require_exists=True)
    if errors:
        raise GTIoUMiningError("GT-IoU localization mining config validation failed:\n" + "\n".join(errors))
    if config.get("config_kind") != APPROVED_KIND:
        raise GTIoUMiningError(f"execution requires {APPROVED_KIND}")
    manifest = load_manifest(config["manifest_path"])
    samples = iter_tampered_samples_with_gt_mask(manifest, int(config["max_samples"]))
    records: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []
    approved_roots = list(config.get("approved_input_roots", []))
    for index, sample in enumerate(samples):
        try:
            gt_mask, gt_shape, mask_path = load_gt_mask_for_sample(sample, approved_roots)
            predictions = run_models_for_sample(config, sample, gt_shape)
            records.append(record_for_sample(sample, index, gt_mask, gt_shape, mask_path, predictions, config))
        except Exception as exc:
            failed_cases.append(_failure_record(sample, index, exc))
            if config.get("fail_fast") is True:
                raise
    return write_mining_outputs(_config_output_root(config), records, config, failed_cases)


def summary_schema_ok(summary: dict[str, Any]) -> bool:
    return (
        summary.get("marker") == MARKER
        and isinstance(summary.get("bucket_counts"), dict)
        and isinstance(summary.get("model_availability"), dict)
        and isinstance(summary.get("thresholds"), dict)
        and isinstance(summary.get("output_paths"), dict)
        and summary.get("no_download") is True
        and summary.get("no_network") is True
        and summary.get("no_training") is True
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
