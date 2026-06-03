"""Pre-SNS policy-gated integrated report using clean long256 and tile localizer v2."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .pre_sns_v3_dual_scale_report import (
    _err,
    _inside_repo,
    _is_under,
    _real,
    _validate_absolute_path,
    _validate_under_roots,
    mask_stats,
    postprocess_mask,
    resize_binary_mask_nearest,
)
from .pre_sns_v3_gt_iou_mining import dice_score, iou_score, to_binary_mask
from .pre_sns_v3_model import CLASS_LABELS, FAMILY_LABELS
from .pre_sns_v3_report import _image_tensor, confidence_map, load_v3_model
from .pre_sns_v3_tile_localizer_v2_model import build_pre_sns_v3_tile_localizer_v2

MARKER = "PRE_SNS_V3_V2_POLICY_GATED_REPORT_OK"
CONFIG_OK_MARKER = "PRE_SNS_V3_V2_POLICY_GATED_REPORT_CONFIG_OK"
APPROVED_KIND = "approved_pre_sns_v3_v2_policy_gated_report"
APPROVED_MODE = "approved_local_pre_sns_v3_v2_policy_gated_report"


class PolicyGatedReportError(ValueError):
    """Raised when v2 policy-gated report inputs or guardrails fail."""


def load_policy_gated_report_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise PolicyGatedReportError("policy-gated report config root must be a JSON object")
    return raw


def validate_policy_gated_report_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "long256_checkpoint_path",
        "tile_localizer_v2_checkpoint_path",
        "approved_input_roots",
        "output_root",
        "mask_threshold",
        "min_area_pct",
        "max_area_pct",
        "suppress_non_tampered_mask",
        "fallback_to_baseline_on_v2_unreliable",
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
    for field in ("suppress_non_tampered_mask", "fallback_to_baseline_on_v2_unreliable", "component_filtering"):
        if field in raw and raw.get(field) not in {True, False}:
            errors.append(_err(f"{field} must be boolean"))
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
    for field in ("long256_checkpoint_path", "tile_localizer_v2_checkpoint_path"):
        errors.extend(_validate_under_roots(raw.get(field), field, model_roots, require_file=require_exists, allow_parts={"checkpoints"}))
    for field in ("image_path", "gt_mask_path", "sample_list_path"):
        if raw.get(field):
            errors.extend(_validate_under_roots(raw.get(field), field, roots, require_file=require_exists))
    output_root = raw.get("output_root")
    errors.extend(_validate_absolute_path(output_root, "output_root", require_dir_parent=require_exists))
    if isinstance(output_root, str) and output_root.startswith("/") and any(_is_under(output_root, root) for root in roots):
        errors.append(_err("output_root must not be under approved input roots"))
    for field in ("mask_threshold", "min_area_pct", "max_area_pct", "min_largest_component_area_pct"):
        if field in raw and raw[field] is not None:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                errors.append(_err(f"{field} must be a non-negative finite number"))
    for field in ("max_component_count", "max_samples"):
        if field in raw and raw[field] is not None:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(_err(f"{field} must be a positive integer"))
    if "mask_threshold" in raw and float(raw["mask_threshold"]) > 1.0:
        errors.append(_err("mask_threshold must be <= 1.0"))
    if "min_area_pct" in raw and float(raw["min_area_pct"]) > 100.0:
        errors.append(_err("min_area_pct must be <= 100.0"))
    if "max_area_pct" in raw and float(raw["max_area_pct"]) > 100.0:
        errors.append(_err("max_area_pct must be <= 100.0"))
    if "min_area_pct" in raw and "max_area_pct" in raw and float(raw["min_area_pct"]) > float(raw["max_area_pct"]):
        errors.append(_err("min_area_pct must be <= max_area_pct"))
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_policy_gated_report_config(config, require_exists)
    if errors:
        raise PolicyGatedReportError("policy-gated report config validation failed:\n" + "\n".join(errors))


def _runtime_deps():
    try:
        import torch
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise PolicyGatedReportError("torch and PIL are required for policy-gated reporting") from exc
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


def threshold_mask(values: list[float], threshold: float) -> list[int]:
    return [1 if float(value) >= threshold else 0 for value in values]


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


def _write_probability_mask_png(Image: Any, path: Path, values: list[float], shape: tuple[int, int]) -> str:
    image = Image.new("L", shape)
    image.putdata([max(0, min(255, int(round(float(value) * 255.0)))) for value in values])
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


def _comparison_sheet(Image: Any, ImageDraw: Any, base: Any, gt: list[int], baseline: list[int], v2_mask: list[int], final_mask: list[int], shape: tuple[int, int]) -> Any:
    panels = [
        ("input", base.resize((192, 192))),
        ("gt", _red_overlay(Image, base, gt, shape).resize((192, 192))),
        ("baseline", _red_overlay(Image, base, baseline, shape).resize((192, 192))),
        ("tile_v2", _red_overlay(Image, base, v2_mask, shape).resize((192, 192))),
        ("final", _red_overlay(Image, base, final_mask, shape).resize((192, 192))),
    ]
    sheet = Image.new("RGB", (192 * len(panels), 216), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(panels):
        x = index * 192
        draw.text((x + 6, 6), label, fill=(0, 0, 0))
        sheet.paste(panel, (x, 24))
    return sheet


def _load_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("sample_list_path"):
        with open(config["sample_list_path"], "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        samples = raw.get("samples", raw) if isinstance(raw, dict) else raw
        return [sample for sample in samples if isinstance(sample, dict)]
    return [{"sample_id": "single_image", "image_path": config.get("image_path"), "gt_mask_path": config.get("gt_mask_path")}]


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
            "baseline_mask": _flatten_mask(fixture.get("baseline_mask"), width, height, float(config.get("mask_threshold", 0.4))),
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
    baseline_small = threshold_mask(mask_probs, float(config.get("mask_threshold", 0.4)))
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


def _v2_probability_values_from_fixture(sample: dict[str, Any], config: dict[str, Any], width: int, height: int) -> list[float] | None:
    raw = sample.get("fixture_v2_probability_mask")
    if raw is None:
        raw = config.get("fixture_v2_probability_mask")
    if raw is None:
        mask = sample.get("fixture_v2_mask")
        if mask is None:
            mask = config.get("fixture_v2_mask")
        if mask is not None:
            return [float(value) for value in _flatten_mask(mask, width, height, 0.5)]
        return None
    if isinstance(raw, (list, tuple)):
        values = [float(value) for value in raw]
        if len(values) == width * height:
            return values
    return [0.0] * (width * height)


def _extract_checkpoint_state(raw_obj: Any) -> dict[str, Any]:
    if not isinstance(raw_obj, dict):
        raise PolicyGatedReportError("tile localizer v2 checkpoint must be a dict")
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(raw_obj.get(key), dict):
            return raw_obj[key]
    tensor_keys = [key for key, value in raw_obj.items() if hasattr(value, "shape")]
    if tensor_keys:
        return raw_obj
    raise PolicyGatedReportError("could not find state_dict in tile localizer v2 checkpoint")


def _normalize_state_keys(state: dict[str, Any]) -> dict[str, Any]:
    if state and all(str(key).startswith("module.") for key in state):
        return {str(key)[len("module."):]: value for key, value in state.items()}
    return state


def _find_checkpoint_stem(state: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    stem = state.get("stem.0.weight")
    if hasattr(stem, "shape"):
        return "stem.0.weight", tuple(int(v) for v in stem.shape)
    for key, value in state.items():
        if hasattr(value, "shape") and len(value.shape) == 4:
            return str(key), tuple(int(v) for v in value.shape)
    raise PolicyGatedReportError("no 4D conv tensor found in tile localizer v2 checkpoint")


def _state_has_prefix(state: dict[str, Any], prefix: str) -> bool:
    return any(str(key).startswith(prefix) for key in state)


def _model_stem_shape(model: Any) -> tuple[int, ...]:
    state = model.state_dict()
    stem = state.get("stem.0.weight")
    if hasattr(stem, "shape"):
        return tuple(int(v) for v in stem.shape)
    for value in state.values():
        if hasattr(value, "shape") and len(value.shape) == 4:
            return tuple(int(v) for v in value.shape)
    raise PolicyGatedReportError("could not determine model stem shape")


def _tensor_from_model_output(torch: Any, out: Any) -> Any:
    if isinstance(out, dict):
        for key in ("mask_logits", "logits", "mask", "pred_mask"):
            if key in out and torch.is_tensor(out[key]):
                return out[key]
        for value in out.values():
            if torch.is_tensor(value):
                return value
    if isinstance(out, (list, tuple)):
        for value in out:
            if torch.is_tensor(value):
                return value
    if torch.is_tensor(out):
        return out
    raise PolicyGatedReportError(f"cannot find tensor output from model output type={type(out)}")


def _tile_boxes(width: int, height: int, tile_size: int, tile_stride: int) -> list[tuple[int, int, int, int]]:
    xs = list(range(0, max(1, width - tile_size + 1), tile_stride))
    ys = list(range(0, max(1, height - tile_size + 1), tile_stride))
    if not xs or xs[-1] != max(0, width - tile_size):
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] != max(0, height - tile_size):
        ys.append(max(0, height - tile_size))
    boxes: list[tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            boxes.append((x, y, min(width, x + tile_size), min(height, y + tile_size)))
    return boxes


def _pil_to_rgb_tensor(torch: Any, image: Any, device: str) -> Any:
    raw = torch.ByteTensor(torch.ByteStorage.from_buffer(image.convert("RGB").tobytes()))
    width, height = image.size
    return raw.reshape(1, height, width, 3).permute(0, 3, 1, 2).float().div(255.0).to(device)


def _rough_mask_image(Image: Any, mask: list[int] | None, shape: tuple[int, int]) -> Any | None:
    if mask is None:
        return None
    image = Image.new("L", shape)
    image.putdata([255 if value else 0 for value in mask])
    return image


def _model_forward_rgb(torch: Any, model: Any, model_info: dict[str, Any], tile_img: Any, device: str, rough_tile_img: Any | None = None) -> np.ndarray:
    rgb = _pil_to_rgb_tensor(torch, tile_img, device)
    with torch.no_grad():
        if bool(model_info.get("rough_mask_prior")) and rough_tile_img is not None:
            arr = np.asarray(rough_tile_img.convert("L")).astype("float32") / 255.0
            rough = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
            try:
                out = model(rgb, rough)
            except TypeError:
                try:
                    out = model(rgb, rough_mask_prior=rough)
                except TypeError:
                    out = model(images=rgb, rough_mask_prior=rough)
        else:
            out = model(rgb)
    tensor = _tensor_from_model_output(torch, out)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    logits = tensor.detach().float().cpu()
    return torch.sigmoid(logits).numpy()


def _load_v2_model(torch: Any, checkpoint_path: str, device: str, config: dict[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = _normalize_state_keys(_extract_checkpoint_state(checkpoint))
    stem_key, checkpoint_stem_shape = _find_checkpoint_stem(state)
    inferred_base_channels = int(checkpoint_stem_shape[0])
    expected_internal_feature_channels = int(checkpoint_stem_shape[1])
    tile_size = int(config.get("tile_size", checkpoint.get("tile_size", 768)))
    feature_mode = str(config.get("input_feature_mode") or checkpoint.get("input_feature_mode") or "rgb_edge_residual")
    boundary_head = True if _state_has_prefix(state, "boundary_head.") else bool(config.get("boundary_head", checkpoint.get("boundary_head", True)))
    confidence_head = True if _state_has_prefix(state, "confidence_head.") else bool(config.get("confidence_head", checkpoint.get("confidence_head", True)))

    selected_model = None
    selected_info = None
    for rough_mask_prior in (False, True):
        candidate = build_pre_sns_v3_tile_localizer_v2(
            torch,
            tile_size=tile_size,
            input_feature_mode=feature_mode,
            base_channels=inferred_base_channels,
            boundary_head=boundary_head,
            confidence_head=confidence_head,
            rough_mask_prior=rough_mask_prior,
        )
        model_stem_shape = _model_stem_shape(candidate)
        if tuple(model_stem_shape) == tuple(checkpoint_stem_shape):
            selected_model = candidate
            selected_info = {
                "checkpoint_stem_shape": [int(v) for v in checkpoint_stem_shape],
                "model_stem_shape": [int(v) for v in model_stem_shape],
                "base_channels": inferred_base_channels,
                "expected_internal_feature_channels": expected_internal_feature_channels,
                "input_feature_mode": feature_mode,
                "rough_mask_prior": bool(rough_mask_prior),
                "boundary_head": bool(boundary_head),
                "confidence_head": bool(confidence_head),
                "checkpoint_first_conv_key": stem_key,
                "tile_size": tile_size,
            }
            break
    if selected_model is None or selected_info is None:
        raise PolicyGatedReportError(
            f"could not match tile localizer v2 stem shape {checkpoint_stem_shape} using feature_mode={feature_mode}"
        )

    if tuple(_model_stem_shape(selected_model)) != tuple(checkpoint_stem_shape):
        raise PolicyGatedReportError("model stem shape does not match checkpoint stem shape before state load")
    result = selected_model.load_state_dict(state, strict=False)
    selected_info["missing_count"] = len(list(getattr(result, "missing_keys", [])))
    selected_info["unexpected_count"] = len(list(getattr(result, "unexpected_keys", [])))
    selected_model.to(device)
    selected_model.eval()
    return selected_model, checkpoint, selected_info


def run_tile_localizer_v2(
    config: dict[str, Any],
    sample: dict[str, Any],
    width: int,
    height: int,
    baseline_mask: list[int] | None = None,
) -> dict[str, Any]:
    fixture = _v2_probability_values_from_fixture(sample, config, width, height)
    if fixture is not None:
        return {"values": fixture, "model_info": {"source": "fixture_v2_probability_mask"}}
    torch, Image, _ImageDraw = _runtime_deps()
    device = "cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    model, checkpoint, model_info = _load_v2_model(torch, config["tile_localizer_v2_checkpoint_path"], device, config)
    tile_size = int(model_info.get("tile_size", checkpoint.get("tile_size", min(width, height))))
    tile_stride = int(config.get("tile_stride", max(1, tile_size // 2)))
    resample_bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    resample_nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    with Image.open(sample["image_path"]) as image:
        image = image.convert("RGB")
        rough_full = _rough_mask_image(Image, baseline_mask, (width, height))
        accum = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)
        boxes = _tile_boxes(width, height, tile_size, tile_stride)
        for x1, y1, x2, y2 in boxes:
            crop = image.crop((x1, y1, x2, y2))
            original_size = crop.size
            rough_crop = rough_full.crop((x1, y1, x2, y2)) if rough_full is not None else None
            if crop.size != (tile_size, tile_size):
                crop_for_model = crop.resize((tile_size, tile_size), resample_bilinear)
                rough_for_model = rough_crop.resize((tile_size, tile_size), resample_nearest) if rough_crop is not None else None
            else:
                crop_for_model = crop
                rough_for_model = rough_crop
            prob = _model_forward_rgb(torch, model, model_info, crop_for_model, device, rough_tile_img=rough_for_model)
            if prob.shape != (tile_size, tile_size):
                prob_img = Image.fromarray((prob * 255.0).clip(0, 255).astype("uint8"))
                prob_img = prob_img.resize((tile_size, tile_size), resample_bilinear)
                prob = np.asarray(prob_img).astype("float32") / 255.0
            prob_img = Image.fromarray((prob * 255.0).clip(0, 255).astype("uint8"))
            prob_img = prob_img.resize(original_size, resample_bilinear)
            prob_arr = np.asarray(prob_img).astype("float32") / 255.0
            accum[y1:y2, x1:x2] += prob_arr
            counts[y1:y2, x1:x2] += 1.0
    counts[counts == 0] = 1.0
    final_prob = accum / counts
    model_info = {**model_info, "tile_stride": tile_stride, "tile_box_count": len(_tile_boxes(width, height, tile_size, tile_stride))}
    return {"values": [float(value) for value in final_prob.reshape(-1).tolist()], "model_info": model_info}


def _apply_component_filtering(config: dict[str, Any], mask: list[int], shape: tuple[int, int]) -> list[int]:
    if not bool(config.get("component_filtering", False)):
        return mask
    keep_top_k = config.get("max_component_count")
    return postprocess_mask(mask, shape, min_component_area_px=0, keep_top_k_components=int(keep_top_k) if keep_top_k else None, morphology="none")


def _gt_metrics(gt_mask: list[int] | None, baseline_mask: list[int], v2_mask: list[int], final_mask: list[int]) -> dict[str, float | None]:
    if gt_mask is None:
        return {
            "baseline_iou": None,
            "baseline_dice": None,
            "v2_raw_iou": None,
            "v2_raw_dice": None,
            "final_iou": None,
            "final_dice": None,
            "final_iou_delta_vs_baseline": None,
        }
    baseline_iou = float(iou_score(gt_mask, baseline_mask))
    baseline_dice = float(dice_score(gt_mask, baseline_mask))
    v2_raw_iou = float(iou_score(gt_mask, v2_mask))
    v2_raw_dice = float(dice_score(gt_mask, v2_mask))
    final_iou = float(iou_score(gt_mask, final_mask))
    final_dice = float(dice_score(gt_mask, final_mask))
    return {
        "baseline_iou": baseline_iou,
        "baseline_dice": baseline_dice,
        "v2_raw_iou": v2_raw_iou,
        "v2_raw_dice": v2_raw_dice,
        "final_iou": final_iou,
        "final_dice": final_dice,
        "final_iou_delta_vs_baseline": float(final_iou - baseline_iou),
    }


def build_policy_gated_record(config: dict[str, Any], sample: dict[str, Any], index: int = 0) -> dict[str, Any]:
    torch, Image, _ImageDraw = _runtime_deps()
    image_path = str(sample.get("image_path") or config.get("image_path"))
    width, height = _load_image_size(Image, image_path)
    shape = (width, height)
    sample = {**sample, "image_path": image_path}
    long_report = _long256_report(config, sample, width, height)
    baseline_mask = list(long_report["baseline_mask"])
    threshold = float(config.get("mask_threshold", 0.4))
    gt_mask = None
    gt_path = sample.get("gt_mask_path") or config.get("gt_mask_path")
    if gt_path:
        gt_mask = _load_mask_from_path(Image, str(gt_path), width, height, 0.5)
    elif sample.get("gt_mask") is not None or config.get("gt_mask") is not None:
        gt_mask = _flatten_mask(sample.get("gt_mask", config.get("gt_mask")), width, height, 0.5)

    v2_values = [0.0] * (width * height)
    v2_raw_mask = [0] * (width * height)
    v2_filtered_mask = [0] * (width * height)
    v2_stats = mask_stats(v2_raw_mask, shape)
    localized_evidence_status = "suppressed_non_tampered"
    final_mask = [0] * (width * height)
    final_mask_source = "suppressed_non_tampered"
    reliability_reasons: list[str] = []
    v2_model_info: dict[str, Any] = {}

    if str(long_report["class"]) == "tampered":
        v2_result = run_tile_localizer_v2(config, sample, width, height, baseline_mask=baseline_mask)
        v2_values = list(v2_result.get("values", []))
        v2_model_info = dict(v2_result.get("model_info", {}))
        if len(v2_values) != width * height:
            v2_values = [0.0] * (width * height)
        v2_raw_mask = threshold_mask(v2_values, threshold)
        v2_filtered_mask = _apply_component_filtering(config, v2_raw_mask, shape)
        v2_stats = mask_stats(v2_filtered_mask, shape)
        mask_area = float(v2_stats["mask_area_pct"])
        component_count = int(v2_stats["component_count"])
        largest_component_area_pct = float(v2_stats["largest_component_area_pct"])
        if mask_area < float(config.get("min_area_pct", 0.01)):
            reliability_reasons.append("mask_area_too_small")
        if mask_area > float(config.get("max_area_pct", 35.0)):
            reliability_reasons.append("mask_area_too_large")
        if config.get("min_largest_component_area_pct") is not None and largest_component_area_pct < float(config["min_largest_component_area_pct"]):
            reliability_reasons.append("largest_component_area_too_small")
        if config.get("max_component_count") is not None and component_count > int(config["max_component_count"]):
            reliability_reasons.append("component_count_too_large")
        if reliability_reasons and bool(config.get("fallback_to_baseline_on_v2_unreliable", True)) and sum(baseline_mask) > 0:
            final_mask = baseline_mask
            final_mask_source = "baseline_fallback"
            localized_evidence_status = "baseline_fallback"
        else:
            final_mask = v2_filtered_mask
            final_mask_source = "tile_v2"
            localized_evidence_status = "localized_evidence_present" if sum(final_mask) > 0 else "v2_empty"
    else:
        if not bool(config.get("suppress_non_tampered_mask", True)):
            final_mask = baseline_mask
            final_mask_source = "baseline_unsuppressed"
            localized_evidence_status = "baseline_unsuppressed"

    metrics = _gt_metrics(gt_mask, baseline_mask, v2_filtered_mask, final_mask)
    final_stats = mask_stats(final_mask, shape)
    baseline_stats = mask_stats(baseline_mask, shape)
    return {
        "marker": MARKER,
        "sample_id": str(sample.get("sample_id") or f"sample_{index:06d}"),
        "image_path": image_path,
        "class": long_report["class"],
        "class_conf": long_report["class_conf"],
        "tampered_score": float(long_report["tampered_score"]),
        "family": long_report["family"],
        "family_conf": long_report["family_conf"],
        "primary_detector": "clean_long256",
        "conditional_localizer": "tile_localizer_v2",
        "tile_localization_activated": str(long_report["class"]) == "tampered",
        "localized_evidence_status": localized_evidence_status,
        "final_mask_source": final_mask_source,
        "mask_threshold": threshold,
        "mask_area_pct": float(v2_stats["mask_area_pct"]),
        "component_count": int(v2_stats["component_count"]),
        "largest_component_area_pct": float(v2_stats["largest_component_area_pct"]),
        "final_mask_area_pct": float(final_stats["mask_area_pct"]),
        "baseline_mask_area_pct": float(baseline_stats["mask_area_pct"]),
        "v2_model_info": v2_model_info,
        "reliability_reasons": reliability_reasons,
        "baseline_iou": metrics["baseline_iou"],
        "baseline_dice": metrics["baseline_dice"],
        "v2_raw_iou": metrics["v2_raw_iou"],
        "v2_raw_dice": metrics["v2_raw_dice"],
        "final_iou": metrics["final_iou"],
        "final_dice": metrics["final_dice"],
        "final_iou_delta_vs_baseline": metrics["final_iou_delta_vs_baseline"],
        "_shape": shape,
        "_baseline_mask": baseline_mask,
        "_v2_probability_values": v2_values,
        "_v2_mask": v2_filtered_mask,
        "_final_mask": final_mask,
        "_gt_mask": gt_mask,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def write_artifacts(output_root: Path, record: dict[str, Any]) -> dict[str, str]:
    _torch, Image, ImageDraw = _runtime_deps()
    output_root.mkdir(parents=True, exist_ok=True)
    shape = tuple(record["_shape"])
    paths: dict[str, str] = {}
    report_path = output_root / "policy_gated_report.json"
    paths["policy_gated_report"] = _write_json(report_path, _public_record(record))
    paths["final_mask"] = _write_mask_png(Image, output_root / "final_mask.png", record["_final_mask"], shape)
    paths["tile_v2_probability_mask"] = _write_probability_mask_png(Image, output_root / "tile_v2_probability_mask.png", record["_v2_probability_values"], shape)
    with Image.open(record["image_path"]) as image:
        base = image.convert("RGB")
    _red_overlay(Image, base, record["_final_mask"], shape).save(output_root / "final_clean_red_overlay.png")
    paths["final_clean_red_overlay"] = str(output_root / "final_clean_red_overlay.png")
    _red_overlay(Image, base, record["_v2_mask"], shape).save(output_root / "tile_v2_clean_red_overlay.png")
    paths["tile_v2_clean_red_overlay"] = str(output_root / "tile_v2_clean_red_overlay.png")
    _red_overlay(Image, base, record["_baseline_mask"], shape).save(output_root / "baseline_clean_red_overlay.png")
    paths["baseline_clean_red_overlay"] = str(output_root / "baseline_clean_red_overlay.png")
    if record.get("_gt_mask") is not None:
        sheet = _comparison_sheet(Image, ImageDraw, base, record["_gt_mask"], record["_baseline_mask"], record["_v2_mask"], record["_final_mask"], shape)
        sheet.save(output_root / "comparison_sheet.jpg", quality=90)
        paths["comparison_sheet"] = str(output_root / "comparison_sheet.jpg")
    artifact = {
        "marker": MARKER,
        "output_paths": paths,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }
    paths["artifact_manifest"] = _write_json(output_root / "artifact_manifest.json", artifact)
    return paths


def run_policy_gated_report(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_exists=True)
    output_root = _real(config["output_root"])
    if _inside_repo(output_root):
        raise PolicyGatedReportError("output_root must be outside repository")
    started = time.perf_counter()
    samples = _load_samples(config)
    records: list[dict[str, Any]] = []
    output_paths: dict[str, Any] = {}
    for index, sample in enumerate(samples[: int(config.get("max_samples", len(samples)))]):
        record = build_policy_gated_record(config, sample, index)
        sample_root = output_root if len(samples) == 1 else output_root / record["sample_id"]
        paths = write_artifacts(sample_root, record)
        public = _public_record(record)
        public["output_paths"] = paths
        records.append(public)
        output_paths[record["sample_id"]] = paths
    summary = {
        "marker": MARKER,
        "record_count": len(records),
        "suppressed_non_tampered_count": sum(1 for record in records if record["localized_evidence_status"] == "suppressed_non_tampered"),
        "tile_v2_used_count": sum(1 for record in records if record["final_mask_source"] == "tile_v2"),
        "baseline_fallback_count": sum(1 for record in records if record["final_mask_source"] == "baseline_fallback"),
        "elapsed_sec": time.perf_counter() - started,
        "records": records,
        "output_paths": output_paths,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }
    _write_json(output_root / "policy_gated_report_summary.json", summary)
    return records[0] if len(records) == 1 else summary


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value
