"""Source manifest audit and generation config validation for SNSAug V2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .configs import PROFILES, SEVERITIES
from ..pre_sns_training_artifacts import validate_artifact_root
from ..pre_sns_v3_dual_scale_report import (
    _err,
    _inside_repo,
    _is_under,
    _real,
    _validate_absolute_path,
    _validate_under_roots,
)

MARKER = "SNSAUG_V2_GENERATION_AND_TINY_PAIRS_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_GENERATION_CONFIG_OK"
APPROVED_CONFIG_KINDS = {
    "approved_snsaug_v2_source_audit": "approved_local_snsaug_v2_source_audit",
    "approved_snsaug_v2_overlay_preview": "approved_local_snsaug_v2_overlay_preview",
    "approved_snsaug_v2_tiny_fixed_pairs": "approved_local_snsaug_v2_tiny_fixed_pairs",
}
LABEL_MAP = {
    "real": "real",
    "authentic": "real",
    "synthetic": "synthetic",
    "full_synthetic": "synthetic",
    "ai": "synthetic",
    "tampered": "tampered",
    "manipulated": "tampered",
}


class SNSAugV2GenerationError(ValueError):
    """Raised when source manifest audit or generation inputs are invalid."""


def load_snsaug_v2_generation_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2GenerationError("snsaug_v2 generation config root must be a JSON object")
    return raw


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _load_records_from_text(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    lines = [line for line in stripped.splitlines() if line.strip()]
    if len(lines) > 1:
        first = lines[0].lstrip()
        second = lines[1].lstrip()
        if first.startswith("{") and second.startswith("{"):
            rows: list[dict[str, Any]] = []
            for line in lines:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
            return rows
    if stripped.startswith("{") or stripped.startswith("["):
        raw = json.loads(stripped)
        if isinstance(raw, dict):
            for key in ("samples", "records", "items"):
                value = raw.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [raw] if "image_path" in raw else []
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def load_source_manifest_records(path: str | Path) -> list[dict[str, Any]]:
    return _load_records_from_text(Path(path).read_text(encoding="utf-8"))


def _normalize_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower().replace("-", "_").replace(" ", "_")
    return LABEL_MAP.get(text)


def _first_str(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_record(row: dict[str, Any], index: int) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    base_id = _first_str(row, ("base_id", "sample_id", "id", "img_id"))
    image_path = _first_str(row, ("image_path", "source_path", "img_path"))
    label = _normalize_label(_first_str(row, ("content_label", "label", "class_label")))
    source_dataset = _first_str(row, ("source_dataset", "dataset", "dataset_id"))
    split = _first_str(row, ("split", "split_name")) or "train"
    tamper_mask_path = _first_str(row, ("tamper_mask_path", "mask_path", "source_mask_path"))
    family_label = _first_str(row, ("family_label",))

    if base_id is None:
        base_id = f"sample_{index:06d}"
    if image_path is None:
        errors.append("image_path missing")
    if label is None:
        errors.append("content_label could not map to real/synthetic/tampered")
    if source_dataset is None:
        errors.append("source_dataset missing")
    if image_path is not None and not os.path.isfile(image_path):
        errors.append("image_path not readable")
    if tamper_mask_path is not None and not os.path.isfile(tamper_mask_path):
        errors.append("tamper_mask_path not readable")
    if errors:
        return None, errors
    return {
        "base_id": base_id,
        "image_path": image_path,
        "content_label": label,
        "tamper_mask_path": tamper_mask_path,
        "family_label": family_label,
        "split": split,
        "source_dataset": source_dataset,
        "mask_available": bool(tamper_mask_path),
    }, []


def audit_source_manifest(
    source_manifest_path: str | Path,
    *,
    output_root: str | Path,
    fail_fast: bool = False,
) -> dict[str, Any]:
    output_dir = validate_artifact_root(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_source_manifest_records(source_manifest_path)
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    class_counts = {"real": 0, "synthetic": 0, "tampered": 0}
    tampered_with_masks = 0

    for index, row in enumerate(records):
        normalized, errors = _normalize_record(row, index)
        if errors:
            invalid_rows.append(
                {
                    "row_index": index,
                    "base_id": row.get("base_id") or row.get("sample_id") or row.get("id"),
                    "errors": errors,
                }
            )
            if fail_fast:
                raise SNSAugV2GenerationError("; ".join(errors))
            continue
        assert normalized is not None
        valid_rows.append(normalized)
        class_counts[normalized["content_label"]] += 1
        if normalized["content_label"] == "tampered" and normalized["mask_available"]:
            tampered_with_masks += 1

    summary = {
        "source_manifest_path": str(_real(source_manifest_path)),
        "record_count": len(records),
        "valid_record_count": len(valid_rows),
        "invalid_record_count": len(invalid_rows),
        "class_counts": class_counts,
        "tampered_with_masks": tampered_with_masks,
        "fail_fast": bool(fail_fast),
    }
    _write_json(output_dir / "source_manifest_audit_summary.json", summary)
    _write_jsonl(output_dir / "source_manifest_valid_records.jsonl", valid_rows)
    _write_jsonl(output_dir / "source_manifest_invalid_records.jsonl", invalid_rows)
    _write_json(output_dir / "source_manifest_class_counts.json", class_counts)
    return {
        "summary": summary,
        "valid_records": valid_rows,
        "invalid_records": invalid_rows,
        "audit_summary_path": str(output_dir / "source_manifest_audit_summary.json"),
        "valid_records_path": str(output_dir / "source_manifest_valid_records.jsonl"),
        "invalid_records_path": str(output_dir / "source_manifest_invalid_records.jsonl"),
        "class_counts_path": str(output_dir / "source_manifest_class_counts.json"),
    }


def validate_snsaug_v2_generation_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "approved_input_roots",
        "approved_output_roots",
        "no_training",
        "no_finetune",
        "no_network",
        "no_download",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    kind = raw.get("config_kind")
    mode = raw.get("execution_mode")
    if kind not in APPROVED_CONFIG_KINDS:
        errors.append(_err("config_kind must be an approved snsaug_v2 generation kind"))
    elif mode != APPROVED_CONFIG_KINDS[kind]:
        errors.append(_err(f"execution_mode must be {APPROVED_CONFIG_KINDS[kind]}"))
    for flag in ("no_training", "no_finetune", "no_network", "no_download"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))

    input_roots = _as_str_list(raw.get("approved_input_roots"))
    output_roots = _as_str_list(raw.get("approved_output_roots"))
    if not input_roots:
        errors.append(_err("approved_input_roots must be a non-empty list of absolute paths"))
    if not output_roots:
        errors.append(_err("approved_output_roots must be a non-empty list of absolute paths"))
    for index, root in enumerate(input_roots):
        errors.extend(_validate_absolute_path(root, f"approved_input_roots[{index}]"))
    for index, root in enumerate(output_roots):
        errors.extend(_validate_absolute_path(root, f"approved_output_roots[{index}]"))

    if kind in {
        "approved_snsaug_v2_source_audit",
        "approved_snsaug_v2_tiny_fixed_pairs",
    }:
        errors.extend(_validate_under_roots(raw.get("source_manifest_path"), "source_manifest_path", input_roots, require_file=require_exists))

    if kind == "approved_snsaug_v2_overlay_preview":
        if raw.get("use_demo_image") is not True:
            errors.extend(_validate_under_roots(raw.get("image_path"), "image_path", input_roots, require_file=require_exists))
        if raw.get("mask_path") is not None:
            errors.extend(_validate_under_roots(raw.get("mask_path"), "mask_path", input_roots, require_file=require_exists))

    if "output_root" in raw:
        output_root = raw.get("output_root")
        errors.extend(_validate_absolute_path(output_root, "output_root", require_dir_parent=require_exists))
        if isinstance(output_root, str) and output_root.startswith("/"):
            if _inside_repo(_real(output_root)):
                errors.append(_err("output_root must be outside repository"))
            if input_roots and any(_is_under(output_root, root) for root in input_roots):
                errors.append(_err("output_root must not be under approved input roots"))
            if output_roots and not any(_is_under(output_root, root) or str(_real(output_root)) == str(_real(root)) for root in output_roots):
                errors.append(_err("output_root must be under an approved output root"))

    profiles = raw.get("profiles")
    if kind != "approved_snsaug_v2_source_audit":
        if not isinstance(profiles, list) or not profiles or not all(isinstance(item, str) for item in profiles):
            errors.append(_err("profiles must be a non-empty list of strings"))
        else:
            invalid_profiles = [item for item in profiles if item not in PROFILES]
            if invalid_profiles:
                errors.append(_err(f"profiles contain unsupported values: {sorted(set(invalid_profiles))}"))
    severity = raw.get("severity")
    if severity is not None and severity not in SEVERITIES:
        errors.append(_err("severity must be light, medium, or strong"))
    if "seed" in raw:
        seed = raw.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            errors.append(_err("seed must be a non-negative integer"))
    if kind == "approved_snsaug_v2_tiny_fixed_pairs" and "max_samples_per_class" not in raw:
        errors.append(_err("max_samples_per_class is required for tiny fixed-pairs benchmark generation"))
    if "max_samples_per_class" in raw:
        value = raw.get("max_samples_per_class")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(_err("max_samples_per_class must be a positive integer"))
    for flag in ("allow_tampered_without_mask", "allow_missing_tampered"):
        if flag in raw and not isinstance(raw.get(flag), bool):
            errors.append(_err(f"{flag} must be boolean when present"))
    return errors


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    return str(path)
