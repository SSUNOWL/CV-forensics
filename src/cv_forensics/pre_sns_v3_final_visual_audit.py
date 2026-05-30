"""Final pre-SNS visual audit for the long256 + tile-localizer pipeline."""

from __future__ import annotations

import json
import math
import os
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
)
from .pre_sns_v3_long256_tile_integrated_report import (
    _agreement_overlay,
    _public_record,
    _red_overlay,
    build_integrated_record,
    json_safe,
)

MARKER = "PRE_SNS_V3_FINAL_VISUAL_AUDIT_OK"
CONFIG_OK_MARKER = "PRE_SNS_V3_FINAL_VISUAL_AUDIT_CONFIG_OK"
APPROVED_KIND = "approved_pre_sns_v3_final_visual_audit"
APPROVED_MODE = "approved_local_pre_sns_v3_final_visual_audit"
CLASS_LABELS = ("real", "full_synthetic", "tampered")


class FinalVisualAuditError(ValueError):
    """Raised when final visual audit inputs or guardrails fail."""


def load_final_visual_audit_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise FinalVisualAuditError("final visual audit config root must be a JSON object")
    return raw


def validate_final_visual_audit_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "approved_real_data_access",
        "long256_checkpoint_path",
        "tile_localizer_checkpoint_path",
        "approved_input_roots",
        "output_root",
        "max_samples",
        "balance_classes",
        "no_download",
        "no_network",
        "no_training",
        "no_sns_augmentation",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    if not raw.get("manifest_path") and not raw.get("sample_list_path"):
        errors.append(_err("manifest_path or sample_list_path is required"))
    if raw.get("config_kind") != APPROVED_KIND:
        errors.append(_err(f"config_kind must be {APPROVED_KIND}"))
    if raw.get("execution_mode") != APPROVED_MODE:
        errors.append(_err(f"execution_mode must be {APPROVED_MODE}"))
    for flag in ("approved_real_data_access", "no_download", "no_network", "no_training", "no_sns_augmentation"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if raw.get("balance_classes") not in {True, False}:
        errors.append(_err("balance_classes must be boolean"))
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
    for field in ("manifest_path", "sample_list_path", "hard_cases_path"):
        if raw.get(field):
            errors.extend(_validate_under_roots(raw.get(field), field, roots, require_file=require_exists))
    output_root = raw.get("output_root")
    errors.extend(_validate_absolute_path(output_root, "output_root", require_dir_parent=require_exists))
    if isinstance(output_root, str) and output_root.startswith("/") and any(_is_under(output_root, root) for root in roots):
        errors.append(_err("output_root must not be under approved input roots"))
    for field in ("max_samples", "per_class_max", "tile_size", "tile_stride", "max_tiles"):
        if field in raw:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(_err(f"{field} must be a positive integer"))
    if isinstance(raw.get("max_samples"), bool) or not isinstance(raw.get("max_samples"), int) or raw.get("max_samples", 0) <= 0:
        errors.append(_err("max_samples must be a positive integer"))
    for field in (
        "ready_macro_f1_threshold",
        "ready_tampered_recall_threshold",
        "ready_real_fpr_threshold",
        "ready_failed_red_mask_rate_threshold",
    ):
        if field in raw:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                errors.append(_err(f"{field} must be a non-negative finite number"))
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_final_visual_audit_config(config, require_exists)
    if errors:
        raise FinalVisualAuditError("final visual audit config validation failed:\n" + "\n".join(errors))


def load_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = config.get("sample_list_path") or config.get("manifest_path")
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    samples = raw.get("samples", raw) if isinstance(raw, dict) else raw
    if not isinstance(samples, list):
        raise FinalVisualAuditError("manifest/sample list must be a list or contain samples")
    return [sample for sample in samples if isinstance(sample, dict)]


def sample_class(sample: dict[str, Any]) -> str:
    value = sample.get("class_label", sample.get("label", sample.get("class", "")))
    if value == "synthetic":
        return "full_synthetic"
    return str(value)


def sample_id(sample: dict[str, Any], index: int) -> str:
    raw = str(sample.get("sample_id") or sample.get("id") or f"sample_{index:06d}")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)[:96]


def has_gt(sample: dict[str, Any]) -> bool:
    return bool(sample.get("gt_mask_path") or sample.get("mask_path") or sample.get("gt_mask"))


def select_samples(samples: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    max_samples = int(config["max_samples"])
    rng = random.Random(int(config.get("seed", 45)))
    clean = [sample for sample in samples if sample_class(sample) in CLASS_LABELS]
    if max_samples >= len(clean):
        selected = list(clean)
    elif config.get("balance_classes") is True:
        per_class = int(config.get("per_class_max", max(1, math.ceil(max_samples / len(CLASS_LABELS)))))
        selected = []
        for label in CLASS_LABELS:
            bucket = [sample for sample in clean if sample_class(sample) == label]
            if label == "tampered":
                bucket.sort(key=lambda sample: 0 if has_gt(sample) else 1)
            else:
                rng.shuffle(bucket)
            selected.extend(bucket[:per_class])
        selected = selected[:max_samples]
    else:
        selected = clean[:max_samples]
    selected.extend(load_hard_cases(config))
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, sample in enumerate(selected):
        key = sample_id(sample, index)
        if key not in seen:
            deduped.append(sample)
            seen.add(key)
    return deduped[:max_samples]


def load_hard_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = config.get("hard_cases_path")
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    cases = raw.get("samples", raw) if isinstance(raw, dict) else raw
    if not isinstance(cases, list):
        return []
    wanted = {"severe_iou_fail", "low_iou", "false-positive real", "false_positive_real", "class-mask inconsistent", "class_mask_inconsistent"}
    return [case for case in cases if isinstance(case, dict) and (not case.get("hard_case_type") or str(case.get("hard_case_type")) in wanted)]


def confusion_matrix(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {label: {pred: 0 for pred in CLASS_LABELS} for label in CLASS_LABELS}
    for record in records:
        gt = str(record.get("gt_class", ""))
        pred = str(record.get("pred_class", ""))
        if gt in matrix and pred in matrix[gt]:
            matrix[gt][pred] += 1
    return matrix


def class_metrics(matrix: dict[str, dict[str, int]]) -> dict[str, Any]:
    per_class: dict[str, dict[str, float]] = {}
    for label in CLASS_LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[gt][label] for gt in CLASS_LABELS if gt != label)
        fn = sum(matrix[label][pred] for pred in CLASS_LABELS if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "per_class": per_class,
        "macro_f1": sum(per_class[label]["f1"] for label in CLASS_LABELS) / len(CLASS_LABELS),
        "tampered_precision": per_class["tampered"]["precision"],
        "tampered_recall": per_class["tampered"]["recall"],
        "tampered_f1": per_class["tampered"]["f1"],
    }


def red_mask_bucket(record: dict[str, Any]) -> str:
    if record.get("gt_class") != "tampered":
        return "not_tampered"
    comparison = record.get("gt_comparison", {})
    iou = comparison.get("tile_final_iou")
    final_area = float(record.get("final_mask_area_pct", 0.0))
    if iou is None:
        return "needs_manual_review"
    if float(iou) >= 0.40:
        return "good"
    if float(iou) >= 0.25:
        return "acceptable"
    if float(iou) < 0.15 or final_area <= 0.0:
        return "failed"
    return "needs_manual_review"


def aggregate_metrics(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    matrix = confusion_matrix(records)
    cls = class_metrics(matrix)
    real_total = sum(matrix["real"].values())
    real_fp = real_total - matrix["real"]["real"]
    non_tampered = [record for record in records if record.get("gt_class") in {"real", "full_synthetic"}]
    non_tampered_false_activation = sum(1 for record in non_tampered if float(record.get("final_mask_area_pct", 0.0)) > 0.0 or record.get("tile_localization_activated") is True)
    gt_records = [record for record in records if record.get("gt_comparison")]
    baseline_ious = [float(record["gt_comparison"]["baseline_long256_iou"]) for record in gt_records]
    baseline_dices = [float(record["gt_comparison"]["baseline_long256_dice"]) for record in gt_records]
    tile_ious = [float(record["gt_comparison"]["tile_final_iou"]) for record in gt_records]
    tile_dices = [float(record["gt_comparison"]["tile_final_dice"]) for record in gt_records]
    deltas = [float(record["gt_comparison"]["iou_delta"]) for record in gt_records]
    buckets = {name: [] for name in ("good", "acceptable", "failed", "needs_manual_review")}
    for record in records:
        bucket = red_mask_bucket(record)
        if bucket in buckets:
            buckets[bucket].append(record)
    tampered_pred = [record for record in records if record.get("pred_class") == "tampered"]
    no_localized = sum(1 for record in tampered_pred if record.get("localized_evidence_status") in {"no_localized_evidence", "tile_unreliable_baseline_retained"})
    false_positive_real = [record for record in records if record.get("gt_class") == "real" and record.get("pred_class") == "tampered"]
    false_negative_tampered = [record for record in records if record.get("gt_class") == "tampered" and record.get("pred_class") != "tampered"]
    failed_rate = len(buckets["failed"]) / max(sum(1 for record in records if record.get("gt_class") == "tampered"), 1)
    metrics = {
        "confusion_matrix": matrix,
        **cls,
        "real_false_positive_rate": real_fp / real_total if real_total else 0.0,
        "non_tampered_false_activation_rate": non_tampered_false_activation / len(non_tampered) if non_tampered else 0.0,
        "baseline_long256_mask_mean_iou": sum(baseline_ious) / len(baseline_ious) if baseline_ious else 0.0,
        "baseline_long256_mask_mean_dice": sum(baseline_dices) / len(baseline_dices) if baseline_dices else 0.0,
        "tile_final_mask_mean_iou": sum(tile_ious) / len(tile_ious) if tile_ious else 0.0,
        "tile_final_mask_mean_dice": sum(tile_dices) / len(tile_dices) if tile_dices else 0.0,
        "mean_iou_delta_tile_minus_baseline": sum(deltas) / len(deltas) if deltas else 0.0,
        "good_red_mask_count": len(buckets["good"]),
        "acceptable_red_mask_count": len(buckets["acceptable"]),
        "failed_red_mask_count": len(buckets["failed"]),
        "no_localized_evidence_count": no_localized,
        "needs_manual_review_count": len(buckets["needs_manual_review"]),
        "false_positive_real_count": len(false_positive_real),
        "false_negative_tampered_count": len(false_negative_tampered),
        "failed_red_mask_rate": failed_rate,
    }
    metrics["recommendation"] = recommendation(metrics, config)
    return metrics


def recommendation(metrics: dict[str, Any], config: dict[str, Any]) -> str:
    if (
        float(metrics.get("macro_f1", 0.0)) >= float(config.get("ready_macro_f1_threshold", 0.85))
        and float(metrics.get("tampered_recall", 0.0)) >= float(config.get("ready_tampered_recall_threshold", 0.85))
        and float(metrics.get("real_false_positive_rate", 1.0)) <= float(config.get("ready_real_fpr_threshold", 0.10))
        and float(metrics.get("failed_red_mask_rate", 1.0)) <= float(config.get("ready_failed_red_mask_rate_threshold", 0.20))
    ):
        return "ready_for_sns_robustness_evaluation"
    if float(metrics.get("failed_red_mask_rate", 0.0)) > float(config.get("ready_failed_red_mask_rate_threshold", 0.20)):
        return "needs_more_tile_localization_training"
    return "needs_manual_review"


def _write_json(path: Path, value: Any) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_safe(record), sort_keys=True) + "\n")
    return str(path)


def write_case_visuals(output_root: Path, record: dict[str, Any]) -> dict[str, str]:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise FinalVisualAuditError("PIL is required for final visual audit visuals") from exc
    case_root = output_root / str(record["sample_id"])
    case_root.mkdir(parents=True, exist_ok=True)
    shape = tuple(record["_shape"])
    paths: dict[str, str] = {}
    with Image.open(record["image_path"]) as image:
        base = image.convert("RGB")
    base.save(case_root / "input_image.jpg", quality=92)
    paths["input_image"] = str(case_root / "input_image.jpg")
    _red_overlay(Image, base, record["_baseline_mask"], shape).save(case_root / "baseline_long256_red_overlay.jpg", quality=92)
    _red_overlay(Image, base, record["_final_mask"], shape).save(case_root / "tile_final_red_overlay.jpg", quality=92)
    paths["baseline_long256_red_overlay"] = str(case_root / "baseline_long256_red_overlay.jpg")
    paths["tile_final_red_overlay"] = str(case_root / "tile_final_red_overlay.jpg")
    panels = [("input", base.resize((192, 192)))]
    if record.get("_gt_mask") is not None:
        _red_overlay(Image, base, record["_gt_mask"], shape).save(case_root / "gt_red_overlay.jpg", quality=92)
        _agreement_overlay(Image, base, record["_gt_mask"], record["_baseline_mask"], shape).save(case_root / "gt_baseline_agreement_overlay.jpg", quality=92)
        _agreement_overlay(Image, base, record["_gt_mask"], record["_final_mask"], shape).save(case_root / "gt_tile_agreement_overlay.jpg", quality=92)
        paths["gt_red_overlay"] = str(case_root / "gt_red_overlay.jpg")
        paths["gt_baseline_agreement_overlay"] = str(case_root / "gt_baseline_agreement_overlay.jpg")
        paths["gt_tile_agreement_overlay"] = str(case_root / "gt_tile_agreement_overlay.jpg")
        panels.extend([
            ("GT", _red_overlay(Image, base, record["_gt_mask"], shape).resize((192, 192))),
            ("baseline", _red_overlay(Image, base, record["_baseline_mask"], shape).resize((192, 192))),
            ("tile final", _red_overlay(Image, base, record["_final_mask"], shape).resize((192, 192))),
            ("base agree", _agreement_overlay(Image, base, record["_gt_mask"], record["_baseline_mask"], shape).resize((192, 192))),
            ("tile agree", _agreement_overlay(Image, base, record["_gt_mask"], record["_final_mask"], shape).resize((192, 192))),
        ])
    else:
        panels.extend([
            ("baseline", _red_overlay(Image, base, record["_baseline_mask"], shape).resize((192, 192))),
            ("tile final", _red_overlay(Image, base, record["_final_mask"], shape).resize((192, 192))),
        ])
    sheet = Image.new("RGB", (192 * len(panels), 216), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(panels):
        x = index * 192
        draw.text((x + 6, 6), label, fill=(0, 0, 0))
        sheet.paste(panel, (x, 24))
    sheet.save(case_root / "comparison_sheet.jpg", quality=92)
    paths["comparison_sheet"] = str(case_root / "comparison_sheet.jpg")
    paths["case_summary"] = _write_json(case_root / "case_summary.json", _public_record(record))
    return paths


def build_gallery_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "marker": MARKER,
        "record_count": len(records),
        "cases": [
            {
                "sample_id": record["sample_id"],
                "gt_class": record["gt_class"],
                "pred_class": record["pred_class"],
                "red_mask_bucket": record["red_mask_bucket"],
                "visual_paths": record.get("visual_paths", {}),
            }
            for record in records
        ],
    }


def run_final_visual_audit(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_exists=True)
    output_root = _real(config["output_root"])
    if _inside_repo(output_root):
        raise FinalVisualAuditError("output_root must be outside repository")
    output_root.mkdir(parents=True, exist_ok=True)
    samples = select_samples(load_samples(config), config)
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        record_config = {
            **config,
            "image_path": sample.get("image_path"),
            "gt_mask_path": sample.get("gt_mask_path") or sample.get("mask_path"),
            "sample_list_path": None,
        }
        record = build_integrated_record(record_config, sample, index)
        record["sample_id"] = sample_id(sample, index)
        record["gt_class"] = sample_class(sample)
        record["pred_class"] = str(record.get("class"))
        record["red_mask_bucket"] = red_mask_bucket(record)
        record["visual_paths"] = write_case_visuals(output_root, record)
        records.append(record)
    public_records = [_public_record(record) for record in records]
    metrics = aggregate_metrics(public_records, config)
    summary = {
        "marker": MARKER,
        "record_count": len(public_records),
        "recommendation": metrics["recommendation"],
        "metrics": metrics,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }
    buckets = {
        "good_red_mask_cases": [record for record in public_records if record["red_mask_bucket"] == "good"],
        "acceptable_red_mask_cases": [record for record in public_records if record["red_mask_bucket"] == "acceptable"],
        "failed_red_mask_cases": [record for record in public_records if record["red_mask_bucket"] == "failed"],
        "false_positive_real_cases": [record for record in public_records if record["gt_class"] == "real" and record["pred_class"] == "tampered"],
        "false_negative_tampered_cases": [record for record in public_records if record["gt_class"] == "tampered" and record["pred_class"] != "tampered"],
        "needs_manual_review_cases": [record for record in public_records if record["red_mask_bucket"] == "needs_manual_review"],
    }
    output_paths: dict[str, str] = {
        "final_audit_records": _write_jsonl(output_root / "final_audit_records.jsonl", public_records),
        "final_audit_summary": _write_json(output_root / "final_audit_summary.json", summary),
        "confusion_matrix": _write_json(output_root / "confusion_matrix.json", metrics["confusion_matrix"]),
        "red_mask_gallery_manifest": _write_json(output_root / "red_mask_gallery_manifest.json", build_gallery_manifest(public_records)),
    }
    for name, items in buckets.items():
        output_paths[name] = _write_json(output_root / f"{name}.json", items)
    artifact = {
        "marker": MARKER,
        "output_paths": output_paths,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }
    output_paths["artifact_manifest"] = _write_json(output_root / "artifact_manifest.json", artifact)
    summary["output_paths"] = output_paths
    _write_json(output_root / "final_audit_summary.json", summary)
    return summary

