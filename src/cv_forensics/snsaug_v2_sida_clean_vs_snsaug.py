"""SIDA clean-vs-SNSAug paired diagnostic utilities."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

MARKER = "SNSAUG_V2_SIDA_CLEAN_VS_SNSAUG_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_SIDA_CLEAN_VS_SNSAUG_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_sida_clean_vs_snsaug"
APPROVED_MODE = "approved_local_snsaug_v2_sida_clean_vs_snsaug"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_SIDA_CLEAN_VS_SNSAUG"
EXPORT_MODE = "export_clean_counterpart_mode"
CACHED_MODE = "cached_clean_vs_snsaug_eval_mode"
REPO_ROOT = Path(__file__).resolve().parents[2]
CLASS_LABELS = ("real", "synthetic", "tampered")
TYPE_A_PROFILES = {
    "news_meme_overlay",
    "platform_ui_same_size",
    "tiktok_like",
    "instagram_story_like",
    "youtube_shorts_like",
    "combined_sns_realistic",
}
TYPE_B_PROFILES = {
    "canvas_9x16_only",
    "resize_crop_pad",
    "zoom_crop",
    "resize_jpeg",
    "screenshot_recapture_light",
    "recompression_light",
}
STRICT_PROFILES = TYPE_A_PROFILES | TYPE_B_PROFILES


class SNSAugV2SIDACleanVsSNSAugError(ValueError):
    """Raised when 0079 config or execution fails."""


def _real(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_under(path: str | Path, root: str | Path) -> bool:
    try:
        _real(path).relative_to(_real(root))
        return True
    except ValueError:
        return False


def _inside_repo(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT)


def _roots(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _abs_errors(value: Any, field: str, require_exists: bool = False) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, str) or not value.strip():
        return [f"{field} must be a non-empty absolute path"]
    path = Path(value).expanduser()
    errors: list[str] = []
    if not path.is_absolute():
        errors.append(f"{field} must be absolute")
    if "://" in str(value):
        errors.append(f"{field} must not use a remote scheme")
    if require_exists and not _real(path).exists():
        errors.append(f"{field} does not exist")
    return errors


def _under_errors(value: Any, field: str, roots: list[str], require_exists: bool = False) -> list[str]:
    errors = _abs_errors(value, field, require_exists=require_exists)
    if value in (None, ""):
        return errors
    if isinstance(value, str) and roots and not any(_is_under(value, root) or _real(value) == _real(root) for root in roots):
        errors.append(f"{field} must be under approved roots")
    return errors


def load_snsaug_v2_sida_clean_vs_snsaug_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2SIDACleanVsSNSAugError("0079 config root must be a JSON object")
    raw.setdefault("config_path", str(_real(path)))
    return raw


def validate_snsaug_v2_sida_clean_vs_snsaug_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "user_approval_text",
        "mode",
        "approved_input_roots",
        "approved_output_roots",
        "output_root",
        "no_training",
        "no_finetune",
        "no_network",
        "no_download",
    )
    for field in required:
        if field not in raw:
            errors.append(f"{field} is required")
    if raw.get("config_kind") != APPROVED_KIND:
        errors.append(f"config_kind must be {APPROVED_KIND}")
    if raw.get("execution_mode") != APPROVED_MODE:
        errors.append(f"execution_mode must be {APPROVED_MODE}")
    if raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(f"user_approval_text must equal {APPROVAL_TEXT}")
    if raw.get("mode") not in {EXPORT_MODE, CACHED_MODE}:
        errors.append(f"mode must be {EXPORT_MODE} or {CACHED_MODE}")
    for flag in ("no_training", "no_finetune", "no_network", "no_download"):
        if raw.get(flag) is not True:
            errors.append(f"{flag} must be true")
    input_roots = _roots(raw.get("approved_input_roots"))
    output_roots = _roots(raw.get("approved_output_roots"))
    if not input_roots:
        errors.append("approved_input_roots must be non-empty")
    if not output_roots:
        errors.append("approved_output_roots must be non-empty")
    for index, root in enumerate(input_roots):
        errors.extend(_abs_errors(root, f"approved_input_roots[{index}]", require_exists=require_exists))
    for index, root in enumerate(output_roots):
        errors.extend(_abs_errors(root, f"approved_output_roots[{index}]", require_exists=require_exists))
    path_fields = (
        "snsaug_sida_export_manifest_path",
        "meta_jsonl_path",
        "cached_clean_sida_outputs_path",
        "cached_snsaug_sida_outputs_path",
    )
    for field in path_fields:
        if raw.get(field):
            errors.extend(_under_errors(raw.get(field), field, input_roots, require_exists=require_exists))
    if raw.get("mode") == EXPORT_MODE:
        for field in ("snsaug_sida_export_manifest_path", "meta_jsonl_path"):
            if not raw.get(field):
                errors.append(f"{field} is required in {EXPORT_MODE}")
    if raw.get("mode") == CACHED_MODE:
        for field in ("cached_clean_sida_outputs_path", "cached_snsaug_sida_outputs_path"):
            if not raw.get(field):
                errors.append(f"{field} is required in {CACHED_MODE}")
    output_root = raw.get("output_root")
    errors.extend(_abs_errors(output_root, "output_root"))
    if isinstance(output_root, str) and output_root.strip():
        if _inside_repo(output_root):
            errors.append("output_root must be outside repository")
        if output_roots and not any(_is_under(output_root, root) or _real(output_root) == _real(root) for root in output_roots):
            errors.append("output_root must be under approved output roots")
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_sida_clean_vs_snsaug_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2SIDACleanVsSNSAugError("0079 config validation failed:\n" + "\n".join(errors))


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_safe(row), ensure_ascii=True, sort_keys=True) + "\n")
    return str(path)


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def normalize_class(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if "tamper" in text or "manipulat" in text or "altered" in text or "edited" in text:
        return "tampered"
    if "synthetic" in text or "generated" in text or "fake" in text:
        return "synthetic"
    if "real" in text or "authentic" in text or "natural" in text:
        return "real"
    return None


def parse_sida_class(row: dict[str, Any]) -> str | None:
    explicit = normalize_class(row.get("sida_pred_class") or row.get("pred_class"))
    if explicit:
        return explicit
    text = str(row.get("sida_text_output") or row.get("text") or "").lower()
    cls_part = text
    if "[cls]" in cls_part:
        cls_part = cls_part.split("[cls]", 1)[1].split("[seg]", 1)[0]
    return normalize_class(cls_part)


def profile_family(profile: str) -> str:
    if profile in TYPE_A_PROFILES:
        return "type_a_local_overlay"
    if profile in TYPE_B_PROFILES:
        return "type_b_global_geometry_degradation"
    if profile in {"clean", "basic", "original"}:
        return "clean_reference"
    return "other"


def _pair_key(row: dict[str, Any], include_profile: bool = False) -> tuple[str, str, str] | tuple[str, str, str, str]:
    base_id = str(row.get("base_id") or "")
    label = normalize_class(row.get("content_label")) or str(row.get("content_label") or "")
    source = str(row.get("source_dataset") or "")
    if include_profile:
        return base_id, label, source, str(row.get("profile") or "")
    return base_id, label, source


def _is_clean_row(row: dict[str, Any]) -> bool:
    profile = str(row.get("profile") or "").lower()
    view = str(row.get("view") or "").lower()
    return profile in {"clean", "basic", "original"} or view in {"clean", "basic", "original"}


def find_clean_counterparts(snsaug_rows: list[dict[str, Any]], meta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    loose_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in meta_rows:
        if not _is_clean_row(row):
            continue
        key = _pair_key(row)
        clean_by_key[key] = row
        loose_by_key[(key[0], key[1])] = row
    out: list[dict[str, Any]] = []
    for index, row in enumerate(snsaug_rows):
        key = _pair_key(row)
        clean = clean_by_key.get(key) or loose_by_key.get((key[0], key[1]))
        if not clean:
            continue
        label = normalize_class(row.get("content_label")) or normalize_class(clean.get("content_label"))
        sns_profile = str(row.get("profile") or "")
        image_id = f"{key[0]}__clean_for__{sns_profile}__{label}"
        out.append(
            {
                "marker": MARKER,
                "image_id": image_id,
                "row_index": index,
                "base_id": key[0],
                "content_label": label,
                "source_dataset": key[2],
                "sns_profile": sns_profile,
                "sns_profile_family": profile_family(sns_profile),
                "clean_profile": str(clean.get("profile") or "clean"),
                "clean_image_path": str(clean.get("image_path") or clean.get("clean_image_path") or ""),
                "sns_image_id": row.get("image_id"),
                "sns_image_path": str(row.get("image_path") or ""),
                "tamper_mask_path": str(clean.get("tamper_mask_path") or row.get("tamper_mask_path") or "") or None,
                "ignore_mask_path": str(clean.get("ignore_mask_path") or "") or None,
            }
        )
    return out


def make_prompt(row: dict[str, Any]) -> str:
    return (
        "You are evaluating a forensic image. Return SIDA-style output with [CLS] one of real, synthetic, tampered. "
        "If [CLS] is tampered, also return [SEG] with a mask description or mask path. "
        "Do not run any training or download step."
    )


def export_clean_counterparts(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    output_root = _real(config["output_root"])
    paths = _paths(output_root)
    if dry_run:
        return _plan(config, paths, dry_run=True)
    sns_rows = _load_jsonl(config["snsaug_sida_export_manifest_path"])
    meta_rows = _load_jsonl(config["meta_jsonl_path"])
    manifest = find_clean_counterparts(sns_rows, meta_rows)
    prompts = [{"marker": MARKER, "image_id": row["image_id"], "image_path": row["clean_image_path"], "prompt": make_prompt(row)} for row in manifest]
    _write_jsonl(Path(paths["sida_clean_counterpart_manifest"]), manifest)
    _write_jsonl(Path(paths["sida_clean_prompt_list"]), prompts)
    _write_text(Path(paths["run_sida_clean_external_template"]), _external_template(paths))
    artifact = {**_plan(config, paths, dry_run=False), "manifest_count": len(manifest), "prompt_count": len(prompts)}
    _write_json(Path(paths["artifact_manifest"]), artifact)
    return artifact


def _mask_available(row: dict[str, Any]) -> bool:
    return bool(row.get("sida_mask_path") or row.get("mask_path") or row.get("mask_available") is True)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _iou(row: dict[str, Any]) -> float | None:
    for key in ("valid_iou", "tamper_iou", "tampered_iou", "mask_iou"):
        value = _num(row.get(key))
        if value is not None:
            return value
    return None


def _ignore_capture(row: dict[str, Any]) -> float | None:
    return _num(row.get("ignore_capture") or row.get("ignore_capture_rate"))


def pair_cached_outputs(clean_rows: list[dict[str, Any]], sns_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    clean_loose: dict[tuple[str, str], dict[str, Any]] = {}
    for row in clean_rows:
        key = _pair_key(row)
        clean_by_key[key] = row
        clean_loose[(key[0], key[1])] = row
    pairs: list[dict[str, Any]] = []
    for sns in sns_rows:
        key = _pair_key(sns)
        clean = clean_by_key.get(key) or clean_loose.get((key[0], key[1]))
        if not clean:
            continue
        label = normalize_class(sns.get("content_label")) or normalize_class(clean.get("content_label"))
        profile = str(sns.get("profile") or sns.get("sns_profile") or "")
        clean_pred = parse_sida_class(clean)
        sns_pred = parse_sida_class(sns)
        clean_iou = _iou(clean) if label == "tampered" else None
        sns_iou = _iou(sns) if label == "tampered" else None
        clean_mask = _mask_available(clean)
        sns_mask = _mask_available(sns)
        clean_ignore = _ignore_capture(clean)
        sns_ignore = _ignore_capture(sns)
        pairs.append(
            {
                "marker": MARKER,
                "base_id": key[0],
                "content_label": label,
                "source_dataset": key[2],
                "profile": profile,
                "profile_family": profile_family(profile),
                "clean_pred_class": clean_pred,
                "sns_pred_class": sns_pred,
                "class_flip": bool(clean_pred and sns_pred and clean_pred != sns_pred),
                "clean_synthetic_recall": int(clean_pred == "synthetic") if label == "synthetic" else None,
                "sns_synthetic_recall": int(sns_pred == "synthetic") if label == "synthetic" else None,
                "synthetic_mask_false_positive_clean": int(clean_mask or clean_pred == "tampered") if label == "synthetic" else None,
                "synthetic_mask_false_positive_sns": int(sns_mask or sns_pred == "tampered") if label == "synthetic" else None,
                "clean_tampered_recall": int(clean_pred == "tampered") if label == "tampered" else None,
                "sns_tampered_recall": int(sns_pred == "tampered") if label == "tampered" else None,
                "clean_real_fpr": int(clean_pred != "real") if label == "real" and clean_pred else None,
                "sns_real_fpr": int(sns_pred != "real") if label == "real" and sns_pred else None,
                "clean_mask_available": clean_mask,
                "sns_mask_available": sns_mask,
                "clean_tamper_iou": clean_iou,
                "sns_tamper_iou": sns_iou,
                "delta_iou": (sns_iou - clean_iou) if clean_iou is not None and sns_iou is not None else None,
                "mask_missing_increase": int((not sns_mask) and clean_mask) if label == "tampered" else None,
                "ignore_capture_increase": (sns_ignore - clean_ignore) if clean_ignore is not None and sns_ignore is not None else None,
            }
        )
    return pairs


def _mean(values: list[Any]) -> float | None:
    nums = [_num(value) for value in values]
    vals = [value for value in nums if value is not None]
    return sum(vals) / len(vals) if vals else None


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "marker": MARKER,
        "clean_reference": _summary([row for row in pairs if row["profile_family"] == "clean_reference"]),
        "type_a_local_overlay": _summary([row for row in pairs if row["profile_family"] == "type_a_local_overlay"]),
        "type_b_global_geometry_degradation": _summary([row for row in pairs if row["profile_family"] == "type_b_global_geometry_degradation"]),
        "strict_type_a_plus_type_b": _summary([row for row in pairs if row["profile_family"] in {"type_a_local_overlay", "type_b_global_geometry_degradation"}]),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    synthetic = [row for row in rows if row.get("content_label") == "synthetic"]
    tampered = [row for row in rows if row.get("content_label") == "tampered"]
    real = [row for row in rows if row.get("content_label") == "real"]
    return {
        "row_count": len(rows),
        "synthetic_count": len(synthetic),
        "tampered_count": len(tampered),
        "real_count": len(real),
        "class_flip_rate": _mean([int(row.get("class_flip") is True) for row in rows]),
        "clean_synthetic_recall": _mean([row.get("clean_synthetic_recall") for row in synthetic]),
        "sns_synthetic_recall": _mean([row.get("sns_synthetic_recall") for row in synthetic]),
        "clean_synthetic_mask_false_positive_rate": _mean([row.get("synthetic_mask_false_positive_clean") for row in synthetic]),
        "sns_synthetic_mask_false_positive_rate": _mean([row.get("synthetic_mask_false_positive_sns") for row in synthetic]),
        "clean_tampered_recall": _mean([row.get("clean_tampered_recall") for row in tampered]),
        "sns_tampered_recall": _mean([row.get("sns_tampered_recall") for row in tampered]),
        "clean_real_fpr": _mean([row.get("clean_real_fpr") for row in real]),
        "sns_real_fpr": _mean([row.get("sns_real_fpr") for row in real]),
        "clean_tamper_iou": _mean([row.get("clean_tamper_iou") for row in tampered]),
        "sns_tamper_iou": _mean([row.get("sns_tamper_iou") for row in tampered]),
        "delta_iou": _mean([row.get("delta_iou") for row in tampered]),
        "mask_missing_increase": _mean([row.get("mask_missing_increase") for row in tampered]),
        "ignore_capture_increase": _mean([row.get("ignore_capture_increase") for row in tampered]),
    }


def cached_clean_vs_snsaug_eval(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    output_root = _real(config["output_root"])
    paths = _paths(output_root)
    if dry_run:
        return _plan(config, paths, dry_run=True)
    clean_rows = _load_jsonl(config["cached_clean_sida_outputs_path"])
    sns_rows = _load_jsonl(config["cached_snsaug_sida_outputs_path"])
    pairs = pair_cached_outputs(clean_rows, sns_rows)
    summary = summarize_pairs(pairs)
    metrics = {"marker": MARKER, "pair_count": len(pairs), "summary": summary}
    _write_jsonl(Path(paths["sida_clean_vs_snsaug_pairs"]), pairs)
    _write_json(Path(paths["sida_clean_vs_snsaug_metrics"]), metrics)
    _write_json(Path(paths["sida_clean_vs_snsaug_type_summary"]), summary)
    _write_text(Path(paths["sida_clean_vs_snsaug_report"]), _report(summary, len(pairs)))
    artifact = {**_plan(config, paths, dry_run=False), "pair_count": len(pairs)}
    _write_json(Path(paths["artifact_manifest"]), artifact)
    return artifact


def _paths(output_root: Path) -> dict[str, str]:
    return {
        "sida_clean_counterpart_manifest": str(output_root / "sida_clean_counterpart_manifest.jsonl"),
        "sida_clean_prompt_list": str(output_root / "sida_clean_prompt_list.jsonl"),
        "run_sida_clean_external_template": str(output_root / "run_sida_clean_external_template.sh"),
        "sida_clean_vs_snsaug_pairs": str(output_root / "sida_clean_vs_snsaug_pairs.jsonl"),
        "sida_clean_vs_snsaug_metrics": str(output_root / "sida_clean_vs_snsaug_metrics.json"),
        "sida_clean_vs_snsaug_type_summary": str(output_root / "sida_clean_vs_snsaug_type_summary.json"),
        "sida_clean_vs_snsaug_report": str(output_root / "sida_clean_vs_snsaug_report.md"),
        "artifact_manifest": str(output_root / "artifact_manifest.json"),
    }


def _plan(config: dict[str, Any], paths: dict[str, str], dry_run: bool) -> dict[str, Any]:
    return {
        "marker": MARKER,
        "dry_run": dry_run,
        "mode": config.get("mode"),
        "output_root": str(_real(config["output_root"])),
        "output_paths": paths,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def _external_template(paths: dict[str, str]) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
# Run this outside cv-forensics with an existing SIDA environment.
# This template does not train, download, or install anything.
SIDA_CLEAN_PROMPTS="{paths['sida_clean_prompt_list']}"
SIDA_CLEAN_MANIFEST="{paths['sida_clean_counterpart_manifest']}"
SIDA_CLEAN_OUTPUT_JSONL="/path/to/cached_clean_sida_outputs.jsonl"
echo "Use $SIDA_CLEAN_PROMPTS and $SIDA_CLEAN_MANIFEST with your external SIDA runner."
echo "Write cached outputs to $SIDA_CLEAN_OUTPUT_JSONL."
"""


def _report(summary: dict[str, Any], pair_count: int) -> str:
    lines = [
        "# SIDA Clean-vs-SNSAug Paired Diagnostic",
        "",
        MARKER,
        "",
        f"- Paired rows: `{pair_count}`",
        "",
        "This cached analysis keeps synthetic samples in 3-way classification. It does not compute tampered mask IoU for synthetic samples; synthetic rows report synthetic recall and mask false-positive rate instead.",
        "",
        "## Interpretation",
        "",
        "- SIDA clean weakness is visible when clean-reference metrics are already poor.",
        "- SNSAug-induced degradation is visible when SNS metrics drop relative to clean metrics.",
        "- Type A local overlay degradation and Type B geometry/degradation are reported separately.",
        "",
        "| Family | Rows | Clean Synthetic Recall | SNS Synthetic Recall | Clean Tampered Recall | SNS Tampered Recall | Clean IoU | SNS IoU | Delta IoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("clean_reference", "type_a_local_overlay", "type_b_global_geometry_degradation", "strict_type_a_plus_type_b"):
        item = summary.get(key, {}) if isinstance(summary.get(key), dict) else {}
        lines.append(
            f"| {key} | {item.get('row_count')} | {item.get('clean_synthetic_recall')} | {item.get('sns_synthetic_recall')} | {item.get('clean_tampered_recall')} | {item.get('sns_tampered_recall')} | {item.get('clean_tamper_iou')} | {item.get('sns_tamper_iou')} | {item.get('delta_iou')} |"
        )
    return "\n".join(lines) + "\n"


def run_snsaug_v2_sida_clean_vs_snsaug(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    assert_valid_config(config, require_exists=not dry_run)
    if config.get("mode") == EXPORT_MODE:
        return export_clean_counterparts(config, dry_run=dry_run)
    return cached_clean_vs_snsaug_eval(config, dry_run=dry_run)


__all__ = [
    "APPROVAL_TEXT",
    "CACHED_MODE",
    "CONFIG_OK_MARKER",
    "EXPORT_MODE",
    "MARKER",
    "SNSAugV2SIDACleanVsSNSAugError",
    "cached_clean_vs_snsaug_eval",
    "find_clean_counterparts",
    "load_snsaug_v2_sida_clean_vs_snsaug_config",
    "normalize_class",
    "pair_cached_outputs",
    "profile_family",
    "run_snsaug_v2_sida_clean_vs_snsaug",
    "summarize_pairs",
    "validate_snsaug_v2_sida_clean_vs_snsaug_config",
]
