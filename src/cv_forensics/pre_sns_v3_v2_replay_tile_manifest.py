"""Pre-SNS v2 replay tile-manifest builder and medium training-config generator."""

from __future__ import annotations

import json
import math
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
MARKER = "PRE_SNS_V3_V2_REPLAY_TILE_MANIFEST_OK"
CONFIG_OK_MARKER = "PRE_SNS_V3_V2_REPLAY_TILE_MANIFEST_CONFIG_OK"
APPROVED_KIND = "approved_pre_sns_v3_v2_replay_tile_manifest"
APPROVED_MODE = "approved_local_pre_sns_v3_v2_replay_tile_manifest"
TRAINING_KIND = "approved_pre_sns_v3_tile_localizer_v2_training"
TRAINING_MODE = "approved_local_pre_sns_v3_tile_localizer_v2_training"
TRAINING_APPROVAL_TEXT = "I_APPROVE_PRE_SNS_V3_TILE_LOCALIZER_V2_TRAINING"


class V2ReplayTileManifestError(ValueError):
    """Raised when replay-manifest inputs or guardrails fail."""


def load_v2_replay_tile_manifest_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise V2ReplayTileManifestError("replay tile-manifest config root must be a JSON object")
    return raw


def _as_roots(value: Any) -> list[str]:
    return value if isinstance(value, list) and all(isinstance(item, str) for item in value) else []


def validate_v2_replay_tile_manifest_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "base_tile_manifest_path",
        "approved_input_roots",
        "output_root",
        "no_download",
        "no_network",
        "no_training",
        "no_sns_augmentation",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    if raw.get("config_kind") != APPROVED_KIND:
        errors.append(_err(f"config_kind must be {APPROVED_KIND}"))
    if raw.get("execution_mode") != APPROVED_MODE:
        errors.append(_err(f"execution_mode must be {APPROVED_MODE}"))
    for flag in ("no_download", "no_network", "no_training", "no_sns_augmentation"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    roots = _as_roots(raw.get("approved_input_roots"))
    if not roots:
        errors.append(_err("approved_input_roots must be a non-empty list of absolute paths"))
    for index, root in enumerate(roots):
        errors.extend(_validate_absolute_path(root, f"approved_input_roots[{index}]"))
    errors.extend(_validate_under_roots(raw.get("base_tile_manifest_path"), "base_tile_manifest_path", roots, require_file=require_exists))
    for field in (
        "policy_still_failed_cases_path",
        "policy_worse_cases_path",
        "policy_non_tampered_high_mask_cases_path",
        "policy_fixed_cases_path",
        "policy_better_cases_path",
    ):
        if raw.get(field):
            errors.extend(_validate_under_roots(raw.get(field), field, roots, require_file=require_exists))
    output_root = raw.get("output_root")
    errors.extend(_validate_absolute_path(output_root, "output_root", require_dir_parent=require_exists))
    if isinstance(output_root, str) and output_root.startswith("/"):
        if _inside_repo(_real(output_root)):
            errors.append(_err("output_root must be outside repository"))
        if any(_is_under(output_root, root) for root in roots):
            errors.append(_err("output_root must not be under approved input roots"))
    for field in ("max_records",):
        if field in raw and raw[field] is not None:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(_err(f"{field} must be a positive integer"))
    return errors


def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_base_tile_manifest(path: str | Path) -> dict[str, Any]:
    raw = _load_json(path)
    if not isinstance(raw, dict) or not isinstance(raw.get("records"), list):
        raise V2ReplayTileManifestError("base tile manifest must contain a records list")
    return raw


def _sample_id_from_value(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("sample_id", "source_sample_id", "id"):
            raw = value.get(key)
            if isinstance(raw, str) and raw:
                return raw
    elif isinstance(value, str) and value:
        return value
    return None


def load_validation_case_ids(config: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for field in (
        "policy_still_failed_cases_path",
        "policy_worse_cases_path",
        "policy_non_tampered_high_mask_cases_path",
        "policy_fixed_cases_path",
        "policy_better_cases_path",
    ):
        path = config.get(field)
        if not path:
            continue
        raw = _load_json(path)
        if isinstance(raw, dict):
            candidates = raw.get("records", raw.get("cases", raw.get("items", raw)))
            if isinstance(candidates, dict):
                candidates = list(candidates.values())
        else:
            candidates = raw
        if isinstance(candidates, list):
            for item in candidates:
                sid = _sample_id_from_value(item)
                if sid:
                    ids.add(sid)
    return ids


def weight_for_record(record: dict[str, Any]) -> int:
    tile_class = str(record.get("tile_class", ""))
    bucket = str(record.get("mining_bucket", ""))
    failures = {str(item) for item in record.get("failure_types", []) if isinstance(item, str)}
    if tile_class == "hard_negative":
        return 8
    if tile_class in {"negative_real", "negative_synthetic"}:
        return 3
    if "empty_prediction" in failures or "wrong_region" in failures or bucket in {"empty_prediction", "wrong_region"}:
        return 12
    if bucket == "severe_iou_fail":
        return 12
    if bucket == "low_iou":
        return 10
    if bucket == "weak_iou":
        return 5
    if "undersegmented" in failures or "oversegmented" in failures or bucket in {"undersegmented", "oversegmented"}:
        return 4
    if bucket == "good_iou":
        return 1
    if tile_class == "positive_tampered":
        return 2
    return 1


def record_sample_id(record: dict[str, Any]) -> str:
    for key in ("source_sample_id", "sample_id", "id"):
        raw = record.get(key)
        if isinstance(raw, str) and raw:
            return raw
    return ""


def is_train_record(record: dict[str, Any], manifest: dict[str, Any]) -> bool:
    split = str(record.get("source_split", manifest.get("source_split", "train"))).lower()
    return "val" not in split and "valid" not in split and "test" not in split


def expand_replay_records(base_records: list[dict[str, Any]], excluded_ids: set[str], max_records: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    excluded_count = 0
    source_count = 0
    counts_by_tile_class: dict[str, int] = {}
    counts_by_weight: dict[str, int] = {}
    for record_index, record in enumerate(base_records):
        sid = record_sample_id(record)
        if sid and sid in excluded_ids:
            excluded_count += 1
            continue
        source_count += 1
        weight = weight_for_record(record)
        for repeat_index in range(weight):
            replay_record = dict(record)
            replay_record["replay_weight"] = weight
            replay_record["replay_source_record_index"] = record_index
            replay_record["replay_repeat_index"] = repeat_index
            expanded.append(replay_record)
            tile_class = str(replay_record.get("tile_class", "unknown"))
            counts_by_tile_class[tile_class] = counts_by_tile_class.get(tile_class, 0) + 1
        counts_by_weight[str(weight)] = counts_by_weight.get(str(weight), 0) + 1
        if max_records is not None and len(expanded) >= max_records:
            expanded = expanded[:max_records]
            break
    summary = {
        "source_record_count": source_count,
        "excluded_record_count": excluded_count,
        "expanded_record_count": len(expanded),
        "counts_by_tile_class": counts_by_tile_class,
        "counts_by_weight": counts_by_weight,
    }
    return expanded, summary


def build_replay_manifest(config: dict[str, Any], base_manifest: dict[str, Any], excluded_ids: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    records = [dict(record) for record in base_manifest.get("records", []) if isinstance(record, dict) and is_train_record(record, base_manifest)]
    expanded, summary = expand_replay_records(records, excluded_ids, config.get("max_records"))
    replay_manifest = {
        "marker": MARKER,
        "schema_version": str(config.get("schema_version", "1.0")),
        "source_base_tile_manifest_path": str(config["base_tile_manifest_path"]),
        "source_record_count": len(records),
        "record_count": len(expanded),
        "tile_size": int(base_manifest.get("tile_size", 768)),
        "counts_by_tile_class": summary["counts_by_tile_class"],
        "replay_weight_policy": {
            "severe_iou_fail": 12,
            "low_iou": 10,
            "weak_iou": 5,
            "empty_prediction_or_wrong_region": 12,
            "undersegmented_or_oversegmented": 4,
            "positive_tampered_normal": 2,
            "hard_negative": 8,
            "negative_real_or_negative_synthetic": 3,
            "good_iou": 1,
        },
        "excluded_validation_case_id_count": len(excluded_ids),
        "records": expanded,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }
    replay_summary = {
        "marker": MARKER,
        "base_record_count": len(records),
        "replay_record_count": len(expanded),
        "excluded_validation_case_id_count": len(excluded_ids),
        **summary,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }
    return replay_manifest, replay_summary


def recommended_training_config(config: dict[str, Any], replay_manifest_path: Path) -> dict[str, Any]:
    output_root = _real(config["output_root"])
    approved_roots = list(config.get("approved_input_roots", []))
    return {
        "schema_version": "1.0",
        "config_kind": TRAINING_KIND,
        "execution_mode": TRAINING_MODE,
        "required_approval_text": TRAINING_APPROVAL_TEXT,
        "user_approval_text": TRAINING_APPROVAL_TEXT,
        "tile_manifest_path": str(replay_manifest_path),
        "approved_input_roots": approved_roots + [str(output_root)],
        "approved_run_root": str(output_root / "recommended_v2_medium_run"),
        "approved_checkpoint_root": str(output_root / "recommended_v2_medium_ckpt"),
        "device": "cpu",
        "seed": 46,
        "tile_size": 768,
        "input_feature_mode": "rgb_edge_residual",
        "epochs": 60,
        "batch_size": 2,
        "learning_rate": 3e-5,
        "weight_decay": 1e-4,
        "base_channels": 8,
        "boundary_head": True,
        "confidence_head": True,
        "max_tiles_train": 24000,
        "max_tiles_val": 3000,
        "gradient_accumulation_steps": 2,
        "gradient_clip_norm": 1.0,
        "mixed_precision": False,
        "progress_log_interval_steps": 100,
        "progress_write_json": True,
        "progress_write_jsonl": True,
        "stdout_progress_interval_steps": 100,
        "severe_iou_oversample_factor": 1,
        "low_iou_oversample_factor": 1,
        "weak_iou_oversample_factor": 1,
        "hard_negative_oversample_factor": 1,
        "negative_oversample_factor": 1,
        "bce_loss_weight": 1.0,
        "dice_loss_weight": 2.0,
        "tversky_loss_weight": 1.0,
        "boundary_loss_weight": 0.5,
        "empty_mask_loss_weight": 3.0,
        "false_activation_area_loss_weight": 2.0,
        "focal_loss_weight": 0.0,
        "threshold_values": [0.2, 0.25, 0.35, 0.5, 0.65, 0.75],
        "negative_activation_area_pct": 0.1,
        "no_write_dry_run": True,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation": True,
    }


def _write_json(path: Path, value: Any) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def run_v2_replay_tile_manifest(config: dict[str, Any]) -> dict[str, Any]:
    errors = validate_v2_replay_tile_manifest_config(config, require_exists=True)
    if errors:
        raise V2ReplayTileManifestError("v2 replay tile-manifest config validation failed:\n" + "\n".join(errors))
    output_root = _real(config["output_root"])
    if _inside_repo(output_root):
        raise V2ReplayTileManifestError("output_root must be outside repository")
    output_root.mkdir(parents=True, exist_ok=True)
    base_manifest = load_base_tile_manifest(config["base_tile_manifest_path"])
    excluded_ids = load_validation_case_ids(config)
    replay_manifest, replay_summary = build_replay_manifest(config, base_manifest, excluded_ids)
    replay_manifest_path = output_root / "replay_tile_manifest.json"
    training_config = recommended_training_config(config, replay_manifest_path)

    output_paths: dict[str, str] = {}
    output_paths["replay_tile_manifest"] = _write_json(replay_manifest_path, replay_manifest)
    output_paths["replay_tile_manifest_summary"] = _write_json(output_root / "replay_tile_manifest_summary.json", replay_summary)
    output_paths["excluded_validation_case_ids"] = _write_json(
        output_root / "excluded_validation_case_ids.json",
        {"marker": MARKER, "excluded_validation_case_ids": sorted(excluded_ids), "count": len(excluded_ids)},
    )
    output_paths["recommended_v2_medium_train_config"] = _write_json(
        output_root / "recommended_v2_medium_train_config.json",
        training_config,
    )
    output_paths["artifact_manifest"] = str(output_root / "artifact_manifest.json")
    artifact_manifest = {
        "marker": MARKER,
        "output_paths": output_paths,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }
    _write_json(output_root / "artifact_manifest.json", artifact_manifest)
    return {
        "marker": MARKER,
        "output_root": str(output_root),
        "replay_manifest_summary": replay_summary,
        "output_paths": output_paths,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value
