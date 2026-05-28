"""Hard-case mining for pre-SNS v3 calibrated reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .pre_sns_v3_dual_scale_report import (
    APPROVED_KIND as REPORT_APPROVED_KIND,
    _err,
    _inside_repo,
    _real,
    _run_one_model,
    _validate_absolute_path,
    _validate_under_roots,
    class_mask_consistency,
    mask_iou,
    run_dual_scale_report,
    select_final_mask,
)

MARKER = "PRE_SNS_V3_HARD_MINING_OK"
CONFIG_OK_MARKER = "PRE_SNS_V3_HARD_MINING_CONFIG_OK"
APPROVED_KIND = "approved_pre_sns_v3_hard_mining"
EXAMPLE_KIND = "example_symbolic"


class HardMiningError(ValueError):
    """Raised when hard mining inputs are invalid."""


def load_hard_mining_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise HardMiningError("hard mining config root must be a JSON object")
    return raw


def validate_hard_mining_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "manifest_path",
        "approved_input_roots",
        "approved_output_root",
        "max_samples",
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
        errors.append(_err("config_kind must be approved_pre_sns_v3_hard_mining or example_symbolic"))
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
    approved_roots = list(roots) + list(checkpoint_roots)
    errors.extend(_validate_under_roots(raw.get("manifest_path"), "manifest_path", roots, require_file=require_exists))
    for field in ("long224_checkpoint_path", "long256_checkpoint_path"):
        if raw.get(field):
            errors.extend(_validate_under_roots(raw.get(field), field, approved_roots, require_file=require_exists, allow_parts={"checkpoints"}))
    if not raw.get("long224_checkpoint_path") and not raw.get("long256_checkpoint_path") and not raw.get("use_manifest_predictions"):
        errors.append(_err("one checkpoint or use_manifest_predictions must be provided"))
    errors.extend(_validate_absolute_path(raw.get("approved_output_root"), "approved_output_root", require_dir_parent=kind == APPROVED_KIND and require_exists))
    if isinstance(raw.get("max_samples"), bool) or not isinstance(raw.get("max_samples"), int) or raw.get("max_samples", 0) <= 0:
        errors.append(_err("max_samples must be a positive integer"))
    for field in ("tampered_score_threshold", "low_iou_threshold"):
        if field in raw:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                errors.append(_err(f"{field} must be a number in [0, 1]"))
    if kind == APPROVED_KIND and raw.get("execution_mode") != "approved_local_pre_sns_v3_hard_mining":
        errors.append(_err("execution_mode must be approved_local_pre_sns_v3_hard_mining"))
    if kind == EXAMPLE_KIND and raw.get("execution_mode") != "example_only":
        errors.append(_err("example execution_mode must be example_only"))
    return errors


def _load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise HardMiningError("manifest must be a JSON object with samples list")
    return manifest


def _flatten_binary(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    flat: list[int] = []
    for item in value:
        if isinstance(item, list):
            flat.extend(_flatten_binary(item))
        else:
            flat.append(1 if item else 0)
    return flat


def mine_cases_from_records(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    hard_real: list[dict[str, Any]] = []
    hard_non_tampered: list[dict[str, Any]] = []
    hard_positive_low_iou: list[dict[str, Any]] = []
    inconsistent: list[dict[str, Any]] = []
    score_threshold = float(config.get("tampered_score_threshold", 0.5))
    low_iou_threshold = float(config.get("low_iou_threshold", 0.3))
    for record in records:
        gt = str(record.get("class_label", record.get("gt_class", "")))
        pred = str(record.get("final_decision", record.get("class", "")))
        tampered_score = float(record.get("tampered_score", 0.0))
        mask_area = float(record.get("final_mask_area_pct", record.get("mask_area_pct", 0.0)))
        item = {
            "sample_id": record.get("sample_id"),
            "image_path": record.get("image_path"),
            "gt_class": gt,
            "pred_or_decision": pred,
            "tampered_score": tampered_score,
            "mask_area_pct": mask_area,
        }
        if gt == "real" and (tampered_score >= score_threshold or "tampered" in pred):
            hard_real.append(item)
        if gt in {"real", "full_synthetic"} and (tampered_score >= score_threshold or "tampered" in pred):
            hard_non_tampered.append(item)
        if gt == "tampered":
            iou = record.get("localization_iou")
            if iou is None and record.get("pred_mask") is not None and record.get("gt_mask") is not None:
                iou = mask_iou(_flatten_binary(record["pred_mask"]), _flatten_binary(record["gt_mask"]))
            if iou is not None and float(iou) < low_iou_threshold:
                low = dict(item)
                low["localization_iou"] = float(iou)
                hard_positive_low_iou.append(low)
        if record.get("class_mask_consistency") in {"inconsistent", "partial"} or ("tampered" in pred and mask_area <= 0.0) or (gt in {"real", "full_synthetic"} and mask_area > 0.0):
            inconsistent.append(item)
    return {
        "hard_negative_real": hard_real,
        "hard_negative_non_tampered": hard_non_tampered,
        "hard_positive_tampered_low_iou": hard_positive_low_iou,
        "class_mask_inconsistent_cases": inconsistent,
    }


def _records_from_manifest_predictions(manifest: dict[str, Any], max_samples: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in manifest.get("samples", [])[:max_samples]:
        if not isinstance(sample, dict):
            continue
        prediction = sample.get("prediction", {})
        if not isinstance(prediction, dict):
            prediction = {}
        record = dict(sample)
        record.update(prediction)
        records.append(record)
    return records


def _sample_id(sample: dict[str, Any], index: int) -> str:
    raw = str(sample.get("sample_id") or sample.get("id") or f"sample_{index:06d}")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)[:80]


def _record_from_dual_report(sample: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    long256 = report.get("long256", {}) if isinstance(report.get("long256"), dict) else {}
    return {
        "sample_id": sample.get("sample_id"),
        "image_path": sample.get("image_path"),
        "class_label": sample.get("class_label"),
        "final_decision": report.get("final_decision"),
        "tampered_score": float(long256.get("tampered_score", 0.0)),
        "final_mask_area_pct": float(report.get("final_mask_area_pct", 0.0)),
        "class_mask_consistency": report.get("class_mask_consistency"),
        "localized_evidence_status": report.get("localized_evidence_status"),
    }


def _single_checkpoint_record(config: dict[str, Any], sample: dict[str, Any], checkpoint_field: str) -> dict[str, Any]:
    from .pre_sns_v3_dual_scale_report import _runtime_deps

    torch, Image, _ImageDraw = _runtime_deps()
    one_config = dict(config)
    one_config["image_path"] = sample["image_path"]
    one = _run_one_model(torch, Image, one_config, checkpoint_field, checkpoint_field.replace("_checkpoint_path", ""))
    empty = {
        "class": one["class"],
        "tampered_score": one["tampered_score"],
        "processed_mask": [0] * len(one["processed_mask"]),
        "mask_shape": one["mask_shape"],
    }
    selection = select_final_mask(empty, one, config)
    gate = class_mask_consistency(one["class"], one["tampered_score"], selection, config)
    return {
        "sample_id": sample.get("sample_id"),
        "image_path": sample.get("image_path"),
        "class_label": sample.get("class_label"),
        "final_decision": gate["final_decision"],
        "tampered_score": float(one["tampered_score"]),
        "final_mask_area_pct": float(selection["stats"]["mask_area_pct"]),
        "class_mask_consistency": gate["class_mask_consistency"],
        "localized_evidence_status": gate["localized_evidence_status"],
    }


def _records_from_checkpoint_runs(manifest: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    samples = [sample for sample in manifest.get("samples", [])[: int(config["max_samples"])] if isinstance(sample, dict)]
    has224 = bool(config.get("long224_checkpoint_path"))
    has256 = bool(config.get("long256_checkpoint_path"))
    for index, sample in enumerate(samples):
        if not isinstance(sample.get("image_path"), str):
            continue
        if has224 and has256:
            report_config = {
                "schema_version": config.get("schema_version", "1.0"),
                "config_kind": REPORT_APPROVED_KIND,
                "execution_mode": "approved_local_pre_sns_v3_dual_report",
                "image_path": sample["image_path"],
                "long224_checkpoint_path": config["long224_checkpoint_path"],
                "long256_checkpoint_path": config["long256_checkpoint_path"],
                "approved_input_roots": config["approved_input_roots"],
                "approved_checkpoint_roots": config.get("approved_checkpoint_roots", []),
                "approved_output_root": str(_real(config["approved_output_root"]) / _sample_id(sample, index)),
                "device": config.get("device", "cpu"),
                "mask_threshold": config.get("mask_threshold", 0.5),
                "min_component_area_px": config.get("min_component_area_px", 0),
                "keep_top_k_components": config.get("keep_top_k_components"),
                "morphology": config.get("morphology", "none"),
                "agreement_iou_threshold": config.get("agreement_iou_threshold", 0.15),
                "uncertain_tampered_score": config.get("uncertain_tampered_score", 0.35),
                "very_high_tampered_score": config.get("very_high_tampered_score", 0.85),
                "write_report": False,
                "write_visual_artifacts": False,
                "no_download": True,
                "no_network": True,
                "no_training": True,
                "no_checkpoint_writes": True,
                "no_sns_augmentation": True,
            }
            records.append(_record_from_dual_report(sample, run_dual_scale_report(report_config)))
        elif has256:
            records.append(_single_checkpoint_record(config, sample, "long256_checkpoint_path"))
        elif has224:
            records.append(_single_checkpoint_record(config, sample, "long224_checkpoint_path"))
    return records


def write_mining_outputs(output_root: str | Path, mined: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    root = _real(output_root)
    if _inside_repo(root):
        raise HardMiningError("approved_output_root must be outside repository")
    root.mkdir(parents=True, exist_ok=True)
    filenames = {
        "hard_negative_real": "hard_negative_real.json",
        "hard_negative_non_tampered": "hard_negative_non_tampered.json",
        "hard_positive_tampered_low_iou": "hard_positive_tampered_low_iou.json",
        "class_mask_inconsistent_cases": "class_mask_inconsistent_cases.json",
    }
    output_paths: dict[str, str] = {}
    for key, filename in filenames.items():
        path = root / filename
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(mined[key], handle, indent=2, sort_keys=True)
            handle.write("\n")
        output_paths[key] = str(path)
    summary = {
        "marker": MARKER,
        "schema_version": config.get("schema_version", "1.0"),
        "counts": {key: len(mined[key]) for key in filenames},
        "thresholds": {
            "tampered_score_threshold": float(config.get("tampered_score_threshold", 0.5)),
            "low_iou_threshold": float(config.get("low_iou_threshold", 0.3)),
        },
        "max_samples": int(config.get("max_samples", 0)),
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
        "output_paths": output_paths,
    }
    path = root / "mining_summary.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    output_paths["mining_summary"] = str(path)
    summary["output_paths"] = output_paths
    return summary


def run_hard_mining(config: dict[str, Any]) -> dict[str, Any]:
    errors = validate_hard_mining_config(config, require_exists=True)
    if errors:
        raise HardMiningError("hard mining config validation failed:\n" + "\n".join(errors))
    if config.get("config_kind") != APPROVED_KIND:
        raise HardMiningError("hard mining execution requires approved_pre_sns_v3_hard_mining")
    manifest = _load_manifest(config["manifest_path"])
    if config.get("use_manifest_predictions") is True:
        records = _records_from_manifest_predictions(manifest, int(config["max_samples"]))
    else:
        records = _records_from_checkpoint_runs(manifest, config)
    mined = mine_cases_from_records(records, config)
    return write_mining_outputs(config["approved_output_root"], mined, config)


def mining_summary_schema_ok(summary: dict[str, Any]) -> bool:
    return (
        summary.get("marker") == MARKER
        and isinstance(summary.get("counts"), dict)
        and isinstance(summary.get("thresholds"), dict)
        and isinstance(summary.get("output_paths"), dict)
        and summary.get("no_download") is True
        and summary.get("no_network") is True
        and summary.get("no_training") is True
        and summary.get("no_sns_augmentation") is True
    )
