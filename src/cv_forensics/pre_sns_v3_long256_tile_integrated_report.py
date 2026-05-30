"""Pre-SNS clean-long256 + 512 tile-localizer integrated report."""

from __future__ import annotations

import json
import math
import os
import time
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
from .pre_sns_v3_gt_iou_mining import dice_score, iou_score, to_binary_mask
from .pre_sns_v3_model import CLASS_LABELS, FAMILY_LABELS
from .pre_sns_v3_report import _image_tensor, confidence_map, load_v3_model
from .pre_sns_v3_tile_localizer_model import build_pre_sns_v3_tile_localizer

MARKER = "PRE_SNS_V3_LONG256_TILE_INTEGRATED_REPORT_OK"
CONFIG_OK_MARKER = "PRE_SNS_V3_LONG256_TILE_INTEGRATED_REPORT_CONFIG_OK"
APPROVED_KIND = "approved_pre_sns_v3_long256_tile_integrated_report"
APPROVED_MODE = "approved_local_pre_sns_v3_long256_tile_integrated_report"


class IntegratedReportError(ValueError):
    """Raised when integrated report inputs or guardrails fail."""


def load_integrated_report_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise IntegratedReportError("integrated report config root must be a JSON object")
    return raw


def validate_integrated_report_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "long256_checkpoint_path",
        "tile_localizer_checkpoint_path",
        "approved_input_roots",
        "output_root",
        "tile_size",
        "tile_stride",
        "tile_activation_tau",
        "mask_threshold",
        "max_tiles",
        "suppress_mask_for_non_tampered",
        "no_download",
        "no_network",
        "no_training",
        "no_sns_augmentation",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    if not raw.get("image_path") and not raw.get("sample_list_path"):
        errors.append(_err("image_path or sample_list_path is required"))
    if raw.get("config_kind") != APPROVED_KIND:
        errors.append(_err(f"config_kind must be {APPROVED_KIND}"))
    if raw.get("execution_mode") != APPROVED_MODE:
        errors.append(_err(f"execution_mode must be {APPROVED_MODE}"))
    for flag in ("no_download", "no_network", "no_training", "no_sns_augmentation"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if raw.get("suppress_mask_for_non_tampered") not in {True, False}:
        errors.append(_err("suppress_mask_for_non_tampered must be boolean"))
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
    for field in ("long256_checkpoint_path", "tile_localizer_checkpoint_path"):
        errors.extend(_validate_under_roots(raw.get(field), field, model_roots, require_file=require_exists, allow_parts={"checkpoints"}))
    for field in ("image_path", "gt_mask_path", "sample_list_path"):
        if raw.get(field):
            errors.extend(_validate_under_roots(raw.get(field), field, roots, require_file=require_exists))
    output_root = raw.get("output_root")
    errors.extend(_validate_absolute_path(output_root, "output_root", require_dir_parent=require_exists))
    if isinstance(output_root, str) and output_root.startswith("/") and any(_is_under(output_root, root) for root in roots):
        errors.append(_err("output_root must not be under approved input roots"))
    for field in ("tile_size", "tile_stride", "max_tiles"):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(_err(f"{field} must be a positive integer"))
    for field in ("tile_activation_tau", "mask_threshold", "tile_reliability_min_area_pct"):
        if field in raw:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                errors.append(_err(f"{field} must be a non-negative finite number"))
    if raw.get("aggregation_mode", "average") not in {"average", "max"}:
        errors.append(_err("aggregation_mode must be average or max"))
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_integrated_report_config(config, require_exists)
    if errors:
        raise IntegratedReportError("integrated report config validation failed:\n" + "\n".join(errors))


def tile_grid(width: int, height: int, tile_size: int, tile_stride: int, max_tiles: int | None = None) -> list[dict[str, int]]:
    if width <= 0 or height <= 0 or tile_size <= 0 or tile_stride <= 0:
        raise IntegratedReportError("width, height, tile_size, and tile_stride must be positive")
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
            tile = {"x0": int(x), "y0": int(y), "x1": int(min(x + tile_size, width)), "y1": int(min(y + tile_size, height))}
            key = (tile["x0"], tile["y0"], tile["x1"], tile["y1"])
            if key not in seen:
                tiles.append(tile)
                seen.add(key)
            if max_tiles is not None and len(tiles) >= max_tiles:
                return tiles
    return tiles


def aggregate_tile_values(width: int, height: int, tile_predictions: list[dict[str, Any]], mode: str = "average") -> list[float]:
    total = width * height
    accum = [0.0] * total
    counts = [0] * total
    for pred in tile_predictions:
        tile = pred["tile"]
        values = [float(value) for value in pred.get("values", [])]
        tw = int(tile["x1"]) - int(tile["x0"])
        th = int(tile["y1"]) - int(tile["y0"])
        if tw <= 0 or th <= 0 or len(values) != tw * th:
            continue
        for ty in range(th):
            for tx in range(tw):
                idx = (int(tile["y0"]) + ty) * width + int(tile["x0"]) + tx
                value = values[ty * tw + tx]
                if mode == "max":
                    accum[idx] = max(accum[idx], value)
                else:
                    accum[idx] += value
                    counts[idx] += 1
    if mode == "max":
        return accum
    return [accum[index] / counts[index] if counts[index] else 0.0 for index in range(total)]


def threshold_mask(values: list[float], threshold: float) -> list[int]:
    return [1 if float(value) >= threshold else 0 for value in values]


def mask_area_pct(mask: list[int]) -> float:
    return float((sum(1 for value in mask if value) / max(len(mask), 1)) * 100.0)


def final_vs_baseline_iou(final_mask: list[int], baseline_mask: list[int]) -> float:
    return iou_score(baseline_mask, final_mask)


def activation_decision(primary_class: str, tampered_score: float, activation_tau: float) -> bool:
    return str(primary_class) == "tampered" or float(tampered_score) >= float(activation_tau)


def _runtime_deps():
    try:
        import torch
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise IntegratedReportError("torch and PIL are required for integrated reporting") from exc
    return torch, Image, ImageDraw


def _load_image_size(Image: Any, path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.size[0]), int(image.size[1])


def _flatten_mask(value: Any, width: int, height: int, threshold: float = 0.5) -> list[int]:
    if value is None:
        return [0] * (width * height)
    try:
        mask, shape = to_binary_mask(value, threshold, (width, height))
    except Exception:
        return [0] * (width * height)
    if shape != (width, height):
        return resize_binary_mask_nearest(mask, shape, (width, height))
    return mask


def _load_mask_from_path(Image: Any, path: str, width: int, height: int, threshold: float = 0.5) -> list[int]:
    with Image.open(path) as image:
        image = image.convert("L").resize((width, height))
        values = list(image.getdata())
    return [1 if value >= int(threshold * 255) else 0 for value in values]


def _long256_report(config: dict[str, Any], sample: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    fixture = sample.get("fixture_long256_report") or config.get("fixture_long256_report")
    if isinstance(fixture, dict):
        primary_class = str(fixture.get("class", "real"))
        family = str(fixture.get("family", "Real-or-N/A"))
        return {
            "class": primary_class,
            "class_conf": fixture.get("class_conf", {primary_class: 1.0}),
            "family": family,
            "family_conf": fixture.get("family_conf", {family: 1.0}),
            "tampered_score": float(fixture.get("tampered_score", 0.0)),
            "baseline_mask": _flatten_mask(fixture.get("baseline_mask"), width, height, float(config.get("mask_threshold", 0.5))),
            "source": "fixture_long256_report",
        }
    torch, Image, _ImageDraw = _runtime_deps()
    device = "cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    model, checkpoint, image_size = load_v3_model(torch, config["long256_checkpoint_path"], device)
    image = _image_tensor(torch, Image, sample["image_path"], image_size, device)
    with torch.no_grad():
        outputs = model(image)
        class_probs = torch.softmax(outputs["class_logits"][0], dim=0).detach().cpu().tolist()
        tamper_probs = torch.softmax(outputs["tamper_binary_logits"][0], dim=0).detach().cpu().tolist()
        family_probs = torch.softmax(outputs["family_logits"][0], dim=0).detach().cpu().tolist()
        mask_probs = torch.sigmoid(outputs["localization_logits"][0]).detach().cpu().reshape(-1).tolist()
    primary_class = CLASS_LABELS[int(max(range(len(class_probs)), key=lambda idx: class_probs[idx]))]
    family = FAMILY_LABELS[int(max(range(len(family_probs)), key=lambda idx: family_probs[idx]))]
    baseline_small = threshold_mask(mask_probs, float(config.get("mask_threshold", 0.5)))
    baseline_mask = resize_binary_mask_nearest(baseline_small, (image_size, image_size), (width, height))
    return {
        "class": primary_class,
        "class_conf": confidence_map(CLASS_LABELS, class_probs),
        "family": family,
        "family_conf": confidence_map(FAMILY_LABELS, family_probs),
        "tampered_score": float(tamper_probs[1]),
        "baseline_mask": baseline_mask,
        "source": f"long256_checkpoint:{checkpoint.get('model_name', 'unknown')}",
    }


def _load_tile_model(torch: Any, checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise IntegratedReportError("tile localizer checkpoint must be a dict")
    model = build_pre_sns_v3_tile_localizer(torch, int(checkpoint.get("base_channels", 8))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _tile_values_from_fixture(sample: dict[str, Any], config: dict[str, Any], width: int, height: int, tiles: list[dict[str, int]]) -> list[dict[str, Any]] | None:
    raw = sample.get("fixture_tile_values", config.get("fixture_tile_values"))
    if raw is None:
        mask = sample.get("fixture_tile_mask", config.get("fixture_tile_mask"))
        raw = [float(value) for value in _flatten_mask(mask, width, height, float(config.get("mask_threshold", 0.5)))] if mask is not None else None
    if raw is None:
        return None
    values = [float(value) for value in raw]
    predictions: list[dict[str, Any]] = []
    for tile in tiles:
        tile_values = []
        for y in range(tile["y0"], tile["y1"]):
            for x in range(tile["x0"], tile["x1"]):
                tile_values.append(values[y * width + x])
        predictions.append({"tile": tile, "values": tile_values})
    return predictions


def run_tile_localizer(config: dict[str, Any], sample: dict[str, Any], width: int, height: int, tiles: list[dict[str, int]]) -> list[dict[str, Any]]:
    fixture = _tile_values_from_fixture(sample, config, width, height, tiles)
    if fixture is not None:
        return fixture
    torch, Image, _ImageDraw = _runtime_deps()
    device = "cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    model, checkpoint = _load_tile_model(torch, config["tile_localizer_checkpoint_path"], device)
    tile_size = int(config.get("tile_size", checkpoint.get("tile_size", 512)))
    predictions: list[dict[str, Any]] = []
    resample_bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    with Image.open(sample["image_path"]) as image:
        image = image.convert("RGB")
        with torch.no_grad():
            for tile in tiles:
                crop = image.crop((tile["x0"], tile["y0"], tile["x1"], tile["y1"]))
                raw_size = crop.size
                if crop.size != (tile_size, tile_size):
                    crop = crop.resize((tile_size, tile_size), resample_bilinear)
                raw = torch.ByteTensor(torch.ByteStorage.from_buffer(crop.tobytes()))
                tensor = raw.reshape(1, tile_size, tile_size, 3).permute(0, 3, 1, 2).float().div(255.0).to(device)
                probs = torch.sigmoid(model(tensor)["mask_logits"][0, 0]).detach().cpu()
                if raw_size != (tile_size, tile_size):
                    probs = torch.nn.functional.interpolate(probs.reshape(1, 1, tile_size, tile_size), size=(raw_size[1], raw_size[0]), mode="bilinear", align_corners=False)[0, 0]
                predictions.append({"tile": tile, "values": probs.reshape(-1).tolist()})
    return predictions


def _localized_status(activated: bool, final_decision: str, final_area: float) -> str:
    if not activated:
        return "skipped_non_tampered"
    if final_decision == "tampered_suspect_no_localized_evidence":
        return "no_localized_evidence"
    if final_area > 0.0:
        return "localized_evidence_present"
    return "uncertain_empty_mask"


def _localization_confidence(final_area: float, tile_count: int, activated: bool) -> float:
    if not activated:
        return 0.0
    area_score = min(final_area / 12.5, 1.0)
    tile_score = min(tile_count / 4.0, 1.0)
    return float(max(0.0, min(1.0, 0.7 * area_score + 0.3 * tile_score)))


def _reason(primary_class: str, family: str, activated: bool, evidence_status: str) -> str:
    if activated and evidence_status == "localized_evidence_present":
        return f"clean long256 predicts {primary_class}; the conditional tile localizer provides high-resolution mask evidence. Family estimate remains {family}."
    if activated:
        return f"clean long256 predicts {primary_class} or tampered-suspect, but the tile localizer did not find a reliable localized mask. Family estimate remains {family}."
    return f"clean long256 predicts {primary_class}; tile localization is skipped and the primary detector decision is retained. Family estimate remains {family}."


def build_integrated_record(config: dict[str, Any], sample: dict[str, Any], index: int = 0) -> dict[str, Any]:
    torch, Image, _ImageDraw = _runtime_deps()
    image_path = str(sample.get("image_path") or config.get("image_path"))
    width, height = _load_image_size(Image, image_path)
    sample = {**sample, "image_path": image_path}
    long_report = _long256_report(config, sample, width, height)
    baseline_mask = list(long_report["baseline_mask"])
    activated = activation_decision(long_report["class"], float(long_report["tampered_score"]), float(config.get("tile_activation_tau", 0.5)))
    tiles = tile_grid(width, height, int(config.get("tile_size", 512)), int(config.get("tile_stride", 256)), int(config.get("max_tiles", 256))) if activated else []
    tile_values = [0.0] * (width * height)
    tile_mask = [0] * (width * height)
    if activated:
        predictions = run_tile_localizer(config, sample, width, height, tiles)
        tile_values = aggregate_tile_values(width, height, predictions, str(config.get("aggregation_mode", "average")))
        tile_mask = threshold_mask(tile_values, float(config.get("mask_threshold", 0.5)))
    if not activated and bool(config.get("suppress_mask_for_non_tampered", True)):
        final_mask = [0] * (width * height)
        final_mask_source = "suppressed_non_tampered"
    elif activated and sum(tile_mask) > 0:
        final_mask = tile_mask
        final_mask_source = "tile_localizer"
    elif activated and str(long_report["class"]) == "tampered":
        final_mask = [0] * (width * height)
        final_mask_source = "tile_empty"
    else:
        final_mask = baseline_mask if not bool(config.get("suppress_mask_for_non_tampered", True)) else [0] * (width * height)
        final_mask_source = "baseline_long256"
    final_area = mask_area_pct(final_mask)
    baseline_area = mask_area_pct(baseline_mask)
    final_decision = "tampered_suspect_no_localized_evidence" if activated and str(long_report["class"]) == "tampered" and final_area <= 0.0 else str(long_report["class"])
    gt_mask = None
    gt_path = sample.get("gt_mask_path") or config.get("gt_mask_path")
    if gt_path:
        gt_mask = _load_mask_from_path(Image, str(gt_path), width, height, float(config.get("mask_threshold", 0.5)))
    elif sample.get("gt_mask") is not None or config.get("gt_mask") is not None:
        gt_mask = _flatten_mask(sample.get("gt_mask", config.get("gt_mask")), width, height, float(config.get("mask_threshold", 0.5)))
    comparison = compare_with_gt(gt_mask, baseline_mask, final_mask) if gt_mask is not None else {}
    evidence_status = _localized_status(activated, final_decision, final_area)
    return {
        "marker": MARKER,
        "sample_id": str(sample.get("sample_id") or f"sample_{index:06d}"),
        "image_path": image_path,
        "primary_detector": "clean_long256",
        "class": long_report["class"],
        "class_conf": long_report["class_conf"],
        "tampered_score": long_report["tampered_score"],
        "family": long_report["family"],
        "family_conf": long_report["family_conf"],
        "final_class": long_report["class"],
        "final_family": long_report["family"],
        "final_decision": final_decision,
        "tile_localization_activated": activated,
        "tile_count": len(tiles),
        "final_mask_area_pct": final_area,
        "baseline_mask_area_pct": baseline_area,
        "final_vs_baseline_mask_iou": final_vs_baseline_iou(final_mask, baseline_mask),
        "localized_evidence_status": evidence_status,
        "localization_confidence": _localization_confidence(final_area, len(tiles), activated),
        "final_mask_source": final_mask_source,
        "baseline_mask_stats": mask_stats(baseline_mask, (width, height)),
        "final_mask_stats": mask_stats(final_mask, (width, height)),
        "gt_comparison": comparison,
        "reason": _reason(str(long_report["class"]), str(long_report["family"]), activated, evidence_status),
        "_shape": (width, height),
        "_baseline_mask": baseline_mask,
        "_tile_mask": tile_mask,
        "_final_mask": final_mask,
        "_gt_mask": gt_mask,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }


def compare_with_gt(gt_mask: list[int] | None, baseline_mask: list[int], final_mask: list[int]) -> dict[str, Any]:
    if gt_mask is None:
        return {}
    baseline_iou = iou_score(gt_mask, baseline_mask)
    final_iou = iou_score(gt_mask, final_mask)
    baseline_dice = dice_score(gt_mask, baseline_mask)
    final_dice = dice_score(gt_mask, final_mask)
    return {
        "baseline_long256_iou": baseline_iou,
        "baseline_long256_dice": baseline_dice,
        "tile_final_iou": final_iou,
        "tile_final_dice": final_dice,
        "iou_delta": float(final_iou - baseline_iou),
        "dice_delta": float(final_dice - baseline_dice),
    }


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _write_json(path: Path, value: Any) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def _write_mask_png(Image: Any, path: Path, mask: list[int], shape: tuple[int, int]) -> str:
    image = Image.new("L", shape)
    image.putdata([255 if value else 0 for value in mask])
    image.save(path)
    return str(path)


def _red_overlay(Image: Any, base: Any, mask: list[int], shape: tuple[int, int]) -> Any:
    mask_image = Image.new("L", shape)
    mask_image.putdata([255 if value else 0 for value in mask])
    if mask_image.size != base.size:
        resample = getattr(getattr(Image, "Resampling", Image), "NEAREST")
        mask_image = mask_image.resize(base.size, resample)
    pixels = []
    for pixel, active in zip(base.getdata(), mask_image.getdata()):
        pixels.append(tuple(int(round(0.55 * pixel[i] + 0.45 * (255 if i == 0 else 0))) for i in range(3)) if active else pixel)
    out = Image.new("RGB", base.size)
    out.putdata(pixels)
    return out


def _agreement_overlay(Image: Any, base: Any, gt: list[int], pred: list[int], shape: tuple[int, int]) -> Any:
    colors = []
    active = []
    for g, p in zip(gt, pred):
        if g and p:
            colors.append((0, 180, 0))
            active.append(255)
        elif p and not g:
            colors.append((255, 0, 0))
            active.append(255)
        elif g and not p:
            colors.append((0, 80, 255))
            active.append(255)
        else:
            colors.append((0, 0, 0))
            active.append(0)
    color_image = Image.new("RGB", shape)
    color_image.putdata(colors)
    active_image = Image.new("L", shape)
    active_image.putdata(active)
    if color_image.size != base.size:
        resample = getattr(getattr(Image, "Resampling", Image), "NEAREST")
        color_image = color_image.resize(base.size, resample)
        active_image = active_image.resize(base.size, resample)
    pixels = []
    for pixel, overlay, enabled in zip(base.getdata(), color_image.getdata(), active_image.getdata()):
        pixels.append(tuple(int(round(0.45 * pixel[i] + 0.55 * overlay[i])) for i in range(3)) if enabled else pixel)
    out = Image.new("RGB", base.size)
    out.putdata(pixels)
    return out


def write_artifacts(output_root: Path, record: dict[str, Any]) -> dict[str, str]:
    torch, Image, ImageDraw = _runtime_deps()
    output_root.mkdir(parents=True, exist_ok=True)
    shape = tuple(record["_shape"])
    paths: dict[str, str] = {}
    paths["report_json"] = _write_json(output_root / "integrated_report.json", _public_record(record))
    paths["final_mask"] = _write_mask_png(Image, output_root / "final_mask.png", record["_final_mask"], shape)
    with Image.open(record["image_path"]) as image:
        base = image.convert("RGB")
    _red_overlay(Image, base, record["_final_mask"], shape).save(output_root / "final_clean_red_overlay.png")
    paths["final_clean_red_overlay"] = str(output_root / "final_clean_red_overlay.png")
    if record.get("_gt_mask") is not None:
        _agreement_overlay(Image, base, record["_gt_mask"], record["_baseline_mask"], shape).save(output_root / "baseline_agreement_overlay.png")
        _agreement_overlay(Image, base, record["_gt_mask"], record["_final_mask"], shape).save(output_root / "tile_agreement_overlay.png")
        paths["baseline_agreement_overlay"] = str(output_root / "baseline_agreement_overlay.png")
        paths["tile_agreement_overlay"] = str(output_root / "tile_agreement_overlay.png")
        panels = [
            ("input", base.resize((192, 192))),
            ("GT", _red_overlay(Image, base, record["_gt_mask"], shape).resize((192, 192))),
            ("baseline", _red_overlay(Image, base, record["_baseline_mask"], shape).resize((192, 192))),
            ("tile final", _red_overlay(Image, base, record["_final_mask"], shape).resize((192, 192))),
            ("base agree", _agreement_overlay(Image, base, record["_gt_mask"], record["_baseline_mask"], shape).resize((192, 192))),
            ("tile agree", _agreement_overlay(Image, base, record["_gt_mask"], record["_final_mask"], shape).resize((192, 192))),
        ]
        sheet = Image.new("RGB", (192 * len(panels), 216), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for index, (label, panel) in enumerate(panels):
            x = index * 192
            draw.text((x + 6, 6), label, fill=(0, 0, 0))
            sheet.paste(panel, (x, 24))
        sheet.save(output_root / "gt_comparison_sheet.jpg", quality=90)
        paths["gt_comparison_sheet"] = str(output_root / "gt_comparison_sheet.jpg")
    artifact = {"marker": MARKER, "output_paths": paths, "no_download": True, "no_network": True, "no_training": True, "no_sns_augmentation": True}
    paths["artifact_manifest"] = _write_json(output_root / "artifact_manifest.json", artifact)
    return paths


def _load_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("sample_list_path"):
        with open(config["sample_list_path"], "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        samples = raw.get("samples", raw) if isinstance(raw, dict) else raw
        return [sample for sample in samples if isinstance(sample, dict)]
    return [{"sample_id": "single_image", "image_path": config.get("image_path"), "gt_mask_path": config.get("gt_mask_path")}]


def run_integrated_report(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_exists=True)
    output_root = _real(config["output_root"])
    if _inside_repo(output_root):
        raise IntegratedReportError("output_root must be outside repository")
    started = time.perf_counter()
    samples = _load_samples(config)
    records: list[dict[str, Any]] = []
    all_paths: dict[str, Any] = {}
    for index, sample in enumerate(samples[: int(config.get("max_samples", len(samples)))]):
        record = build_integrated_record(config, sample, index)
        sample_root = output_root if len(samples) == 1 else output_root / record["sample_id"]
        record_paths = write_artifacts(sample_root, record)
        public = _public_record(record)
        public["output_paths"] = record_paths
        records.append(public)
        all_paths[record["sample_id"]] = record_paths
    summary = {
        "marker": MARKER,
        "record_count": len(records),
        "tile_activated_count": sum(1 for record in records if record["tile_localization_activated"]),
        "mean_final_mask_area_pct": sum(float(record["final_mask_area_pct"]) for record in records) / max(len(records), 1),
        "elapsed_sec": time.perf_counter() - started,
        "records": records,
        "output_paths": all_paths,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }
    _write_json(output_root / "integrated_report_summary.json", summary)
    return records[0] if len(records) == 1 else summary


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value

