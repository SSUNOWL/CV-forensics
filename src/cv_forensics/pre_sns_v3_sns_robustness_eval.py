"""Basic SNS-like robustness evaluation for the frozen pre-SNS best bundle."""

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
)
from .pre_sns_v3_v2_policy_gated_report import build_policy_gated_record

MARKER = "PRE_SNS_V3_SNS_ROBUSTNESS_EVAL_OK"
CONFIG_OK_MARKER = "PRE_SNS_V3_SNS_ROBUSTNESS_EVAL_CONFIG_OK"
APPROVED_KIND = "approved_pre_sns_v3_sns_robustness_eval"
APPROVED_MODE = "approved_local_pre_sns_v3_sns_robustness_eval"
PERTURBATIONS = (
    "clean",
    "jpeg_q95",
    "jpeg_q85",
    "jpeg_q75",
    "jpeg_q60",
    "resize_long_1080",
    "resize_long_720",
    "resize_long_512",
    "resize_long_1080_jpeg_q85",
    "resize_long_720_jpeg_q75",
    "resize_long_512_jpeg_q75",
    "mild_center_crop_95pct_resize_back",
    "webp_q80",
)
CLASS_LABELS = ("real", "synthetic", "tampered")


class SNSRobustnessEvalError(ValueError):
    """Raised when SNS robustness evaluation inputs or guardrails fail."""


def load_sns_robustness_eval_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSRobustnessEvalError("SNS robustness eval config root must be a JSON object")
    return raw


def _as_roots(value: Any) -> list[str]:
    return value if isinstance(value, list) and all(isinstance(item, str) for item in value) else []


def validate_sns_robustness_eval_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "best_bundle_path",
        "validation_manifest_path",
        "approved_input_roots",
        "approved_output_roots",
        "output_root",
        "max_samples",
        "samples_per_class",
        "balanced_sampling",
        "perturbations",
        "device",
        "no_training",
        "no_download",
        "no_network",
        "no_sns_augmentation_training",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    if raw.get("config_kind") != APPROVED_KIND:
        errors.append(_err(f"config_kind must be {APPROVED_KIND}"))
    if raw.get("execution_mode") != APPROVED_MODE:
        errors.append(_err(f"execution_mode must be {APPROVED_MODE}"))
    for flag in ("no_training", "no_download", "no_network", "no_sns_augmentation_training"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if raw.get("balanced_sampling") not in {True, False}:
        errors.append(_err("balanced_sampling must be boolean"))
    roots = _as_roots(raw.get("approved_input_roots"))
    if not roots:
        errors.append(_err("approved_input_roots must be a non-empty list of absolute paths"))
    for index, root in enumerate(roots):
        errors.extend(_validate_absolute_path(root, f"approved_input_roots[{index}]"))
    output_roots = _as_roots(raw.get("approved_output_roots"))
    if not output_roots:
        errors.append(_err("approved_output_roots must be a non-empty list of absolute paths"))
    for index, root in enumerate(output_roots):
        errors.extend(_validate_absolute_path(root, f"approved_output_roots[{index}]"))
    for field in ("best_bundle_path", "validation_manifest_path"):
        errors.extend(_validate_under_roots(raw.get(field), field, roots, require_file=require_exists))
    output_root = raw.get("output_root")
    errors.extend(_validate_absolute_path(output_root, "output_root", require_dir_parent=require_exists))
    if isinstance(output_root, str) and output_root.startswith("/"):
        if _inside_repo(_real(output_root)):
            errors.append(_err("output_root must be outside repository"))
        if any(_is_under(output_root, root) for root in roots):
            errors.append(_err("output_root must not be under approved input roots"))
        if output_roots and not any(_is_under(output_root, root) or str(_real(output_root)) == str(_real(root)) for root in output_roots):
            errors.append(_err("output_root must be under an approved output root"))
    for field in ("max_samples", "samples_per_class", "top_n_worst_cases"):
        if field in raw:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(_err(f"{field} must be a positive integer"))
    perturbations = raw.get("perturbations")
    if not isinstance(perturbations, list) or not perturbations or not all(isinstance(item, str) for item in perturbations):
        errors.append(_err("perturbations must be a non-empty list of strings"))
    else:
        invalid = [item for item in perturbations if item not in PERTURBATIONS]
        if invalid:
            errors.append(_err(f"unsupported perturbations: {invalid}"))
    if raw.get("device") not in {"cpu", "cuda"}:
        errors.append(_err("device must be cpu or cuda"))
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_sns_robustness_eval_config(config, require_exists)
    if errors:
        raise SNSRobustnessEvalError("SNS robustness eval config validation failed:\n" + "\n".join(errors))


def _runtime_deps():
    try:
        from PIL import Image, features
    except Exception as exc:
        raise SNSRobustnessEvalError("PIL is required for SNS robustness evaluation") from exc
    return Image, features


def normalize_label(value: Any) -> str:
    text = str(value or "").lower()
    if text in {"synthetic", "full_synthetic"}:
        return "synthetic"
    if text in {"real", "tampered"}:
        return text
    return text


def load_best_bundle(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSRobustnessEvalError("best bundle must be a JSON object")
    return raw


def load_validation_samples(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    samples = raw.get("samples", raw) if isinstance(raw, dict) else raw
    if not isinstance(samples, list):
        raise SNSRobustnessEvalError("validation manifest must be a list or contain samples")
    return [sample for sample in samples if isinstance(sample, dict)]


def sample_label(sample: dict[str, Any]) -> str:
    return normalize_label(sample.get("class_label", sample.get("label", sample.get("class", ""))))


def sample_id(sample: dict[str, Any], index: int) -> str:
    raw = str(sample.get("sample_id") or sample.get("id") or f"sample_{index:06d}")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)[:96]


def select_samples(samples: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    clean = [sample for sample in samples if sample_label(sample) in CLASS_LABELS]
    max_samples = int(config["max_samples"])
    if config.get("balanced_sampling") is not True:
        return clean[:max_samples]
    per_class = int(config["samples_per_class"])
    selected: list[dict[str, Any]] = []
    for label in CLASS_LABELS:
        bucket = [sample for sample in clean if sample_label(sample) == label]
        selected.extend(bucket[:per_class])
    return selected[:max_samples]


def _mask_from_path(Image: Any, path: str | Path) -> Any:
    with Image.open(path) as image:
        return image.convert("L")


def _save_transformed(Image: Any, path: Path, image: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)


def apply_resize(Image: Any, image: Any, mask: Any | None, long_side: int) -> tuple[Any, Any | None]:
    width, height = image.size
    current_long = max(width, height)
    if current_long <= long_side:
        return image.copy(), mask.copy() if mask is not None else None
    scale = float(long_side) / float(current_long)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    resized_image = image.resize(new_size, bilinear)
    resized_mask = mask.resize(new_size, nearest) if mask is not None else None
    return resized_image, resized_mask


def apply_center_crop_resize_back(Image: Any, image: Any, mask: Any | None, keep_ratio: float) -> tuple[Any, Any | None]:
    width, height = image.size
    crop_w = max(1, int(round(width * keep_ratio)))
    crop_h = max(1, int(round(height * keep_ratio)))
    x1 = max(0, (width - crop_w) // 2)
    y1 = max(0, (height - crop_h) // 2)
    x2 = min(width, x1 + crop_w)
    y2 = min(height, y1 + crop_h)
    bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    cropped_image = image.crop((x1, y1, x2, y2)).resize((width, height), bilinear)
    cropped_mask = mask.crop((x1, y1, x2, y2)).resize((width, height), nearest) if mask is not None else None
    return cropped_image, cropped_mask


def apply_jpeg_transform(image: Any, quality: int) -> Any:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    Image, _features = _runtime_deps()
    return Image.open(buffer).convert("RGB")


def apply_webp_transform(image: Any, quality: int) -> Any:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=int(quality))
    buffer.seek(0)
    Image, _features = _runtime_deps()
    return Image.open(buffer).convert("RGB")


def transform_image_and_mask(perturbation: str, image: Any, mask: Any | None) -> tuple[Any, Any | None]:
    Image, features = _runtime_deps()
    if perturbation == "clean":
        return image.copy(), mask.copy() if mask is not None else None
    if perturbation.startswith("jpeg_q"):
        quality = int(perturbation.split("q", 1)[1])
        return apply_jpeg_transform(image, quality), mask.copy() if mask is not None else None
    if perturbation.startswith("resize_long_") and "_jpeg_" not in perturbation:
        long_side = int(perturbation.split("_")[2])
        return apply_resize(Image, image, mask, long_side)
    if perturbation == "mild_center_crop_95pct_resize_back":
        return apply_center_crop_resize_back(Image, image, mask, 0.95)
    if perturbation == "webp_q80":
        if not features.check("webp"):
            raise SNSRobustnessEvalError("PIL WebP support unavailable")
        return apply_webp_transform(image, 80), mask.copy() if mask is not None else None
    if perturbation.startswith("resize_long_") and "_jpeg_q" in perturbation:
        parts = perturbation.split("_")
        long_side = int(parts[2])
        quality = int(parts[-1][1:])
        resized_image, resized_mask = apply_resize(Image, image, mask, long_side)
        return apply_jpeg_transform(resized_image, quality), resized_mask
    raise SNSRobustnessEvalError(f"unsupported perturbation: {perturbation}")


def write_transformed_sample(output_root: Path, base_id: str, perturbation: str, image: Any, mask: Any | None) -> tuple[str, str | None]:
    ext = ".webp" if perturbation == "webp_q80" else ".jpg"
    image_path = output_root / "transformed_inputs" / base_id / f"{perturbation}{ext}"
    mask_path = output_root / "transformed_inputs" / base_id / f"{perturbation}_mask.png" if mask is not None else None
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    if mask is not None and mask_path is not None:
        mask.save(mask_path)
        return str(image_path), str(mask_path)
    return str(image_path), None


def policy_config_from_bundle(bundle: dict[str, Any], config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    policy = bundle.get("policy_gated_report", {})
    approved_inputs = list(config.get("approved_input_roots", []))
    checkpoint_roots = [str(Path(bundle["long256_checkpoint_path"]).parent), str(Path(bundle["tile_v2_checkpoint_path"]).parent)]
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_v3_v2_policy_gated_report",
        "execution_mode": "approved_local_pre_sns_v3_v2_policy_gated_report",
        "long256_checkpoint_path": bundle["long256_checkpoint_path"],
        "tile_localizer_v2_checkpoint_path": bundle["tile_v2_checkpoint_path"],
        "approved_input_roots": approved_inputs,
        "approved_checkpoint_roots": checkpoint_roots,
        "output_root": str(output_root),
        "device": config.get("device", "cpu"),
        "mask_threshold": float(policy.get("mask_threshold", 0.45)),
        "min_area_pct": float(policy.get("min_area_pct", 0.1)),
        "max_area_pct": float(policy.get("max_area_pct", 100.0)),
        "suppress_non_tampered_mask": bool(policy.get("suppress_non_tampered_mask", True)),
        "fallback_to_baseline_on_v2_unreliable": bool(policy.get("fallback_to_baseline_on_v2_unreliable", True)),
        "tile_size": int(policy.get("tile_size", 768)),
        "tile_stride": int(policy.get("tile_stride", 384)),
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }


def _safe_prob(prob_map: dict[str, Any], key: str) -> float:
    value = prob_map.get(key)
    if value is None and key == "synthetic":
        value = prob_map.get("full_synthetic")
    return float(value) if value is not None else 0.0


def evaluate_single_record(bundle: dict[str, Any], config: dict[str, Any], sample: dict[str, Any], base_id: str, perturbation: str, image_path: str, mask_path: str | None, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_config = policy_config_from_bundle(bundle, config, Path(config["output_root"]))
    eval_sample = dict(sample)
    eval_sample["image_path"] = image_path
    if mask_path:
        eval_sample["gt_mask_path"] = mask_path
    started = time.perf_counter()
    result = build_policy_gated_record(policy_config, eval_sample, index)
    latency_ms = (time.perf_counter() - started) * 1000.0
    pred_class = normalize_label(result.get("class"))
    gt_label = sample_label(sample)
    class_conf = result.get("class_conf", {})
    public = {
        "base_id": base_id,
        "sample_id": str(sample.get("sample_id") or base_id),
        "source_image_path": str(sample.get("image_path")),
        "source_mask_path": str(sample.get("mask_path") or sample.get("gt_mask_path")) if sample.get("mask_path") or sample.get("gt_mask_path") else None,
        "transformed_image_path": image_path,
        "transformed_mask_path": mask_path,
        "label": gt_label,
        "perturbation": perturbation,
        "pred_class": pred_class,
        "class_correct": pred_class == gt_label,
        "p_real": _safe_prob(class_conf, "real"),
        "p_synthetic": _safe_prob(class_conf, "synthetic"),
        "p_tampered": _safe_prob(class_conf, "tampered"),
        "tampered_score": float(result.get("tampered_score", 0.0)),
        "final_mask_source": result.get("final_mask_source"),
        "final_iou": result.get("final_iou"),
        "final_dice": result.get("final_dice"),
        "final_mask_area_pct": float(result.get("final_mask_area_pct", 0.0)),
        "latency_ms": float(latency_ms),
        "fps": float(1000.0 / latency_ms) if latency_ms > 0 else None,
        "localization_activated": bool(result.get("tile_localization_activated")),
        "activation_threshold": bundle.get("policy_gated_report", {}).get("mask_threshold"),
        "error_message": None,
        "_result": result,
    }
    return public, result


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def confusion_matrix(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {label: {pred: 0 for pred in CLASS_LABELS} for label in CLASS_LABELS}
    for record in records:
        gt = normalize_label(record.get("label"))
        pred = normalize_label(record.get("pred_class"))
        if gt in matrix and pred in matrix[gt]:
            matrix[gt][pred] += 1
    return matrix


def macro_f1_and_details(matrix: dict[str, dict[str, int]]) -> dict[str, Any]:
    per_class: dict[str, dict[str, float]] = {}
    for label in CLASS_LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in CLASS_LABELS if other != label)
        fn = sum(matrix[label][other] for other in CLASS_LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "per_class": per_class,
        "macro_f1": sum(per_class[label]["f1"] for label in CLASS_LABELS) / len(CLASS_LABELS),
    }


def aggregate_per_perturbation(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_perturbation: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_perturbation.setdefault(str(record["perturbation"]), []).append(record)
    out: dict[str, dict[str, Any]] = {}
    for perturbation, items in by_perturbation.items():
        matrix = confusion_matrix(items)
        cls = macro_f1_and_details(matrix)
        tampered = [item for item in items if item["label"] == "tampered" and item.get("final_iou") is not None]
        real = [item for item in items if item["label"] == "real"]
        synthetic = [item for item in items if item["label"] == "synthetic"]
        tp_latency = [float(item["latency_ms"]) for item in items if item.get("latency_ms") is not None]
        activation_den = [item for item in items if item["label"] == "tampered"]
        activation_num = sum(1 for item in activation_den if item.get("localization_activated") is True)
        out[perturbation] = {
            "sample_count": len(items),
            "class_accuracy": sum(1 for item in items if item["class_correct"]) / len(items) if items else 0.0,
            "macro_f1": cls["macro_f1"],
            "confusion_matrix": matrix,
            "real_false_positive_rate": (sum(1 for item in real if item["pred_class"] != "real") / len(real)) if real else 0.0,
            "synthetic_recall": cls["per_class"]["synthetic"]["recall"],
            "tampered_recall": cls["per_class"]["tampered"]["recall"],
            "final_mean_iou": _mean([float(item["final_iou"]) for item in tampered if item["final_iou"] is not None]),
            "final_median_iou": _median([float(item["final_iou"]) for item in tampered if item["final_iou"] is not None]),
            "final_mean_dice": _mean([float(item["final_dice"]) for item in tampered if item["final_dice"] is not None]),
            "final_median_dice": _median([float(item["final_dice"]) for item in tampered if item["final_dice"] is not None]),
            "failed_red_mask_count": sum(1 for item in tampered if float(item.get("final_iou") or 0.0) < 0.15),
            "localization_activation_recall": (activation_num / len(activation_den)) if activation_den else None,
            "mean_latency_ms": _mean(tp_latency),
            "fps": (1000.0 / _mean(tp_latency)) if tp_latency and _mean(tp_latency) not in {None, 0.0} else None,
        }
    return out


def join_clean_and_perturbed(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_by_base = {record["base_id"]: record for record in records if record["perturbation"] == "clean"}
    joined: list[dict[str, Any]] = []
    for record in records:
        public = {key: value for key, value in record.items() if key != "_result"}
        if record["perturbation"] == "clean":
            joined.append(public)
            continue
        clean = clean_by_base.get(record["base_id"])
        if clean is None:
            joined.append(public)
            continue
        clean_iou = clean.get("final_iou")
        clean_dice = clean.get("final_dice")
        pert_iou = record.get("final_iou")
        pert_dice = record.get("final_dice")
        joined_record = {
            **public,
            "clean_pred_class": clean.get("pred_class"),
            "clean_p_tampered": clean.get("p_tampered"),
            "clean_final_iou": clean_iou,
            "clean_final_dice": clean_dice,
            "clean_localization_activated": clean.get("localization_activated"),
            "pred_flip_from_clean": clean.get("pred_class") != record.get("pred_class"),
            "clean_correct": clean.get("class_correct"),
            "perturbed_correct": record.get("class_correct"),
            "correct_to_wrong": bool(clean.get("class_correct")) and not bool(record.get("class_correct")),
            "wrong_to_correct": (not bool(clean.get("class_correct"))) and bool(record.get("class_correct")),
            "p_tampered_drop": float(clean.get("p_tampered", 0.0)) - float(record.get("p_tampered", 0.0)),
            "iou_drop": (float(clean_iou) - float(pert_iou)) if clean_iou is not None and pert_iou is not None else None,
            "dice_drop": (float(clean_dice) - float(pert_dice)) if clean_dice is not None and pert_dice is not None else None,
            "activation_flip_off": bool(clean.get("localization_activated")) and not bool(record.get("localization_activated")),
        }
        joined_record["fragile_candidate"] = bool(
            joined_record["label"] == "tampered"
            and joined_record["clean_correct"] is True
            and (
                joined_record["perturbed_correct"] is False
                or float(joined_record["p_tampered_drop"]) >= 0.25
                or (joined_record["iou_drop"] is not None and float(joined_record["iou_drop"]) >= 0.20)
                or joined_record["activation_flip_off"] is True
            )
        )
        joined.append(joined_record)
    return joined


def robustness_drop_metrics(joined_records: list[dict[str, Any]], per_perturbation: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    clean = per_perturbation.get("clean", {})
    out: dict[str, dict[str, Any]] = {}
    for perturbation, metrics in per_perturbation.items():
        if perturbation == "clean":
            continue
        rows = [row for row in joined_records if row["perturbation"] == perturbation and row["label"] == "tampered"]
        p_drops = [float(row["p_tampered_drop"]) for row in rows]
        out[perturbation] = {
            "accuracy_drop": (clean.get("class_accuracy") or 0.0) - (metrics.get("class_accuracy") or 0.0),
            "macro_f1_drop": (clean.get("macro_f1") or 0.0) - (metrics.get("macro_f1") or 0.0),
            "synthetic_recall_drop": (clean.get("synthetic_recall") or 0.0) - (metrics.get("synthetic_recall") or 0.0),
            "tampered_recall_drop": (clean.get("tampered_recall") or 0.0) - (metrics.get("tampered_recall") or 0.0),
            "real_fpr_increase": (metrics.get("real_false_positive_rate") or 0.0) - (clean.get("real_false_positive_rate") or 0.0),
            "mean_iou_drop": (clean.get("final_mean_iou") or 0.0) - (metrics.get("final_mean_iou") or 0.0),
            "median_iou_drop": (clean.get("final_median_iou") or 0.0) - (metrics.get("final_median_iou") or 0.0),
            "mean_dice_drop": (clean.get("final_mean_dice") or 0.0) - (metrics.get("final_mean_dice") or 0.0),
            "localization_activation_recall_drop": (clean.get("localization_activation_recall") or 0.0) - (metrics.get("localization_activation_recall") or 0.0),
            "mean_p_tampered_drop_on_tampered": _mean(p_drops) or 0.0,
        }
    return out


def sort_worst_perturbations(drop_metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    items = [{"perturbation": perturbation, **metrics} for perturbation, metrics in drop_metrics.items()]
    items.sort(key=lambda item: (-float(item.get("tampered_recall_drop", 0.0)), -float(item.get("mean_iou_drop", 0.0)), -float(item.get("macro_f1_drop", 0.0)), item["perturbation"]))
    return items


def sort_worst_samples(joined_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in joined_records if row["perturbation"] != "clean"]
    rows.sort(
        key=lambda row: (
            0 if row.get("fragile_candidate") else 1,
            -float(row.get("p_tampered_drop") or 0.0),
            -float(row.get("iou_drop") or 0.0),
            0 if row.get("pred_flip_from_clean") else 1,
        )
    )
    return rows


def _red_overlay(Image: Any, base: Any, mask: list[int], shape: tuple[int, int]) -> Any:
    mask_image = Image.new("L", shape)
    mask_image.putdata([255 if value else 0 for value in mask])
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    if mask_image.size != base.size:
        mask_image = mask_image.resize(base.size, nearest)
    pixels = []
    for pixel, active in zip(base.getdata(), mask_image.getdata()):
        pixels.append(tuple(int(round(0.55 * pixel[i] + 0.45 * (255 if i == 0 else 0))) for i in range(3)) if active else pixel)
    out = Image.new("RGB", base.size)
    out.putdata(pixels)
    return out


def write_visual_gallery(output_root: Path, raw_records: list[dict[str, Any]], worst_samples: list[dict[str, Any]], top_n: int) -> dict[str, Any]:
    Image, _features = _runtime_deps()
    gallery_root = output_root / "visual_gallery"
    gallery_root.mkdir(parents=True, exist_ok=True)
    by_key = {(record["base_id"], record["perturbation"]): record for record in raw_records}
    items: list[dict[str, Any]] = []
    for row in worst_samples[:top_n]:
        raw = by_key.get((row["base_id"], row["perturbation"]))
        if raw is None:
            continue
        result = raw.get("_result", {})
        shape = tuple(result.get("_shape") or (0, 0))
        final_mask = list(result.get("_final_mask", []))
        gt_mask = result.get("_gt_mask")
        case_root = gallery_root / f"{row['base_id']}_{row['perturbation']}"
        case_root.mkdir(parents=True, exist_ok=True)
        with Image.open(row["transformed_image_path"]) as image:
            base = image.convert("RGB")
        _red_overlay(Image, base, final_mask, shape).save(case_root / "final_overlay.jpg", quality=90)
        paths = {"final_overlay": str(case_root / "final_overlay.jpg")}
        if gt_mask is not None:
            _red_overlay(Image, base, gt_mask, shape).save(case_root / "gt_overlay.jpg", quality=90)
            paths["gt_overlay"] = str(case_root / "gt_overlay.jpg")
        items.append({"base_id": row["base_id"], "perturbation": row["perturbation"], "paths": paths})
    return {"marker": MARKER, "record_count": len(items), "items": items}


def _write_json(path: Path, value: Any) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_safe(record), sort_keys=True))
            handle.write("\n")
    return str(path)


def run_sns_robustness_eval(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_exists=True)
    output_root = _real(config["output_root"])
    if _inside_repo(output_root):
        raise SNSRobustnessEvalError("output_root must be outside repository")
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = load_best_bundle(config["best_bundle_path"])
    samples = select_samples(load_validation_samples(config["validation_manifest_path"]), config)
    Image, _features = _runtime_deps()
    raw_records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        base_id = sample_id(sample, index)
        image_path = sample.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            continue
        mask_src = sample.get("mask_path") or sample.get("gt_mask_path")
        with Image.open(image_path) as image:
            base_image = image.convert("RGB")
        base_mask = _mask_from_path(Image, mask_src) if isinstance(mask_src, str) and mask_src else None
        for perturbation in config["perturbations"]:
            try:
                transformed_image, transformed_mask = transform_image_and_mask(str(perturbation), base_image, base_mask)
                transformed_image_path, transformed_mask_path = write_transformed_sample(output_root, base_id, str(perturbation), transformed_image, transformed_mask)
                record, _result = evaluate_single_record(bundle, config, sample, base_id, str(perturbation), transformed_image_path, transformed_mask_path, index)
            except Exception as exc:
                record = {
                    "base_id": base_id,
                    "sample_id": str(sample.get("sample_id") or base_id),
                    "source_image_path": str(image_path),
                    "source_mask_path": str(mask_src) if mask_src else None,
                    "transformed_image_path": None,
                    "transformed_mask_path": None,
                    "label": sample_label(sample),
                    "perturbation": str(perturbation),
                    "pred_class": None,
                    "class_correct": False,
                    "p_real": 0.0,
                    "p_synthetic": 0.0,
                    "p_tampered": 0.0,
                    "tampered_score": 0.0,
                    "final_mask_source": None,
                    "final_iou": None,
                    "final_dice": None,
                    "final_mask_area_pct": 0.0,
                    "latency_ms": None,
                    "fps": None,
                    "localization_activated": None,
                    "activation_threshold": bundle.get("policy_gated_report", {}).get("mask_threshold"),
                    "error_message": str(exc),
                    "_result": {},
                }
            raw_records.append(record)
    joined_records = join_clean_and_perturbed(raw_records)
    per_perturbation = aggregate_per_perturbation(joined_records)
    drop_metrics = robustness_drop_metrics(joined_records, per_perturbation)
    worst_perturbations = sort_worst_perturbations(drop_metrics)
    worst_samples = sort_worst_samples(joined_records)
    fragile_candidates = [row for row in worst_samples if row.get("fragile_candidate")]
    gallery = write_visual_gallery(output_root, raw_records, worst_samples, int(config.get("top_n_worst_cases", 8)))

    output_paths: dict[str, str] = {}
    output_paths["sns_robustness_records"] = _write_jsonl(output_root / "sns_robustness_records.jsonl", [{k: v for k, v in row.items() if k != "_result"} for row in joined_records])
    output_paths["per_perturbation_metrics"] = _write_json(output_root / "per_perturbation_metrics.json", per_perturbation)
    output_paths["robustness_drop_metrics"] = _write_json(output_root / "robustness_drop_metrics.json", drop_metrics)
    output_paths["worst_samples"] = _write_json(output_root / "worst_samples.json", worst_samples)
    output_paths["worst_perturbations"] = _write_json(output_root / "worst_perturbations.json", worst_perturbations)
    output_paths["fragile_candidates"] = _write_jsonl(output_root / "fragile_candidates.jsonl", fragile_candidates)
    output_paths["visual_gallery_manifest"] = _write_json(output_root / "visual_gallery_manifest.json", gallery)
    summary = {
        "marker": MARKER,
        "best_bundle_path": str(config["best_bundle_path"]),
        "sample_count": len(samples),
        "perturbation_count": len(config["perturbations"]),
        "per_perturbation_metrics_path": str(output_root / "per_perturbation_metrics.json"),
        "robustness_drop_metrics_path": str(output_root / "robustness_drop_metrics.json"),
        "worst_perturbations_path": str(output_root / "worst_perturbations.json"),
        "worst_samples_path": str(output_root / "worst_samples.json"),
        "fragile_candidate_count": len(fragile_candidates),
        "no_training": True,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation_training": True,
    }
    output_paths["sns_robustness_summary"] = _write_json(output_root / "sns_robustness_summary.json", summary)
    output_paths["artifact_manifest"] = str(output_root / "artifact_manifest.json")
    artifact = {
        "marker": MARKER,
        "output_paths": output_paths,
        "no_training": True,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation_training": True,
    }
    _write_json(output_root / "artifact_manifest.json", artifact)
    return {
        "marker": MARKER,
        "sample_count": len(samples),
        "records_written": len(joined_records),
        "output_paths": output_paths,
        "no_training": True,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation_training": True,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value
