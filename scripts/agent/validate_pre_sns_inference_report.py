#!/usr/bin/env python3
"""Validate pre-SNS single-image report configs and report JSON."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_inference_report import (  # noqa: E402
    APPROVED_CONFIG_KIND,
    CONFIG_OK_MARKER,
    LEGACY_APPROVED_CONFIG_KINDS,
    MARKER,
    REPORT_VALIDATED_MARKER,
    load_report_config,
    validate_report_config,
)
from cv_forensics.pre_sns_integrated_model import CLASS_LABELS, FAMILY_SMOKE_LABELS  # noqa: E402
from cv_forensics.model_output_schema import LOCALIZATION_STATES  # noqa: E402

REMOTE_PREFIXES = ("http://", "https://", "s3://", "gs://", "hf://")
VISUAL_PATH_FIELDS = {
    "predicted_mask_path": ".png",
    "heatmap_path": ".png",
    "overlay_path": ".png",
    "visual_artifacts_manifest_path": ".json",
}
CLEAN_OVERLAY_PATH_FIELDS = {
    "clean_focused_mask_path": ".png",
    "clean_red_overlay_path": ".png",
    "clean_red_overlay_manifest_path": ".json",
}


def _err(message: str) -> str:
    return f"- {message}"


def parse_report_json(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("report-json does not contain a JSON object")


def _is_under(path: str | Path, root: str | Path) -> bool:
    real_path = os.path.realpath(os.fspath(path))
    real_root = os.path.realpath(os.fspath(root))
    try:
        return os.path.commonpath([real_path, real_root]) == real_root
    except ValueError:
        return False


def _repo_outputs_or_checkpoints(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT / "outputs") or _is_under(path, REPO_ROOT / "checkpoints")


def _inside_repo(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT)


def _looks_remote(path: str) -> bool:
    return path.lower().startswith(REMOTE_PREFIXES)


def _has_protected_segment(path: str) -> bool:
    return any(part in {".env", "secrets", "data", "datasets", "outputs", "checkpoints"} for part in path.replace("\\", "/").split("/") if part)


def _validate_artifact_paths(
    fields: dict[str, str],
    report: dict[str, Any],
    report_root: str,
    errors: list[str],
    context: str,
) -> None:
    for field, suffix in fields.items():
        value = report.get(field)
        if not isinstance(value, str) or not value:
            errors.append(_err(f"{field} is required when {context} are written"))
            continue
        if _looks_remote(value):
            errors.append(_err(f"{field} must not be URL-like"))
        if ".." in value.replace("\\", "/").split("/"):
            errors.append(_err(f"{field} must not contain path traversal"))
        if _has_protected_segment(value):
            errors.append(_err(f"{field} must not contain protected path segments"))
        if not value.endswith(suffix):
            errors.append(_err(f"{field} must end with {suffix}"))
        if not os.path.isfile(value):
            errors.append(_err(f"{field} must exist"))
        elif os.path.getsize(value) <= 0:
            errors.append(_err(f"{field} must be non-empty"))
        if _inside_repo(value):
            errors.append(_err(f"{field} must be outside the repository"))
        if _repo_outputs_or_checkpoints(value):
            errors.append(_err(f"{field} must not be inside repository outputs/checkpoints"))
        if not _is_under(value, report_root):
            errors.append(_err(f"{field} must be under report_root"))


def _validate_clean_overlay_pixels(report: dict[str, Any], errors: list[str]) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    image_path = report.get("image_path")
    mask_path = report.get("clean_focused_mask_path")
    overlay_path = report.get("clean_red_overlay_path")
    if not all(isinstance(path, str) and os.path.isfile(path) for path in (image_path, mask_path, overlay_path)):
        return
    try:
        with Image.open(image_path) as original, Image.open(mask_path) as mask, Image.open(overlay_path) as overlay:
            original_rgb = original.convert("RGB")
            overlay_rgb = overlay.convert("RGB")
            mask_l = mask.convert("L")
        if original_rgb.size != overlay_rgb.size:
            errors.append(_err("clean_red_overlay_path must have the same size as image_path"))
            return
        if mask_l.size != overlay_rgb.size:
            errors.append(_err("clean_focused_mask_path must have the same size as clean_red_overlay_path"))
            return
        for index, mask_value in enumerate(mask_l.getdata()):
            if mask_value <= 0 and original_rgb.getdata()[index] != overlay_rgb.getdata()[index]:
                errors.append(_err("clean_red_overlay_path must not alter pixels outside clean_focused_mask_path"))
                return
    except Exception as exc:
        errors.append(_err(f"clean overlay pixel validation failed: {exc}"))


def _validate_conf(conf: Any, labels: tuple[str, ...], field: str, errors: list[str]) -> None:
    if not isinstance(conf, dict):
        errors.append(_err(f"{field} must be an object"))
        return
    for label in labels:
        if label not in conf:
            errors.append(_err(f"{field} missing {label}"))
    total = 0.0
    for label, value in conf.items():
        if label not in labels:
            errors.append(_err(f"{field} has unknown label {label}"))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(_err(f"{field}.{label} must be numeric"))
            continue
        fvalue = float(value)
        if not 0.0 <= fvalue <= 1.0:
            errors.append(_err(f"{field}.{label} must be in [0, 1]"))
        total += fvalue
    if conf and abs(total - 1.0) > 0.03:
        errors.append(_err(f"{field} must sum to approximately 1.0"))


def validate_report(config: dict[str, Any], report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("marker") != MARKER:
        errors.append(_err("marker must be PRE_SNS_SINGLE_IMAGE_REPORT_OK"))
    class_label = report.get("class")
    if class_label not in CLASS_LABELS:
        errors.append(_err("class must be real, full_synthetic, or tampered"))
    _validate_conf(report.get("class_conf"), CLASS_LABELS, "class_conf", errors)
    if report.get("family") not in FAMILY_SMOKE_LABELS:
        errors.append(_err("family is not supported"))
    _validate_conf(report.get("family_conf"), FAMILY_SMOKE_LABELS, "family_conf", errors)
    for field in ("tampered_score", "latency_ms", "fps_estimate"):
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(_err(f"{field} must be numeric"))
        elif field == "tampered_score" and not 0.0 <= float(value) <= 1.0:
            errors.append(_err("tampered_score must be in [0, 1]"))
        elif field != "tampered_score" and float(value) < 0.0:
            errors.append(_err(f"{field} must be non-negative"))
    if report.get("localization_head") not in LOCALIZATION_STATES:
        errors.append(_err("localization_head is invalid"))
    mask_area = report.get("mask_area_pct")
    if mask_area is not None:
        if isinstance(mask_area, bool) or not isinstance(mask_area, (int, float)) or not 0.0 <= float(mask_area) <= 100.0:
            errors.append(_err("mask_area_pct must be null or a number in [0, 100]"))
    if not isinstance(report.get("reason"), str) or not report.get("reason", "").strip():
        errors.append(_err("reason must be a non-empty string"))
    for flag in ("no_download", "no_network", "no_training", "no_sns_augmentation"):
        if report.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if config.get("config_kind") in {APPROVED_CONFIG_KIND, *LEGACY_APPROVED_CONFIG_KINDS}:
        for field in ("image_path", "checkpoint_path"):
            if report.get(field) != config.get(field):
                errors.append(_err(f"{field} must match config"))
        if config.get("write_report") is True:
            report_path = report.get("report_path")
            if not isinstance(report_path, str) or not report_path:
                errors.append(_err("report_path is required when write_report=true"))
            else:
                if _inside_repo(report_path):
                    errors.append(_err("report_path must be outside the repository"))
                if _repo_outputs_or_checkpoints(report_path):
                    errors.append(_err("report_path must not be inside repository outputs/checkpoints"))
                if not _is_under(report_path, config.get("report_root", "")):
                    errors.append(_err("report_path must be under report_root"))
        visual_requested = config.get("write_visual_artifacts") is True
        visual_written = report.get("visual_artifacts_written") is True
        if visual_requested and report.get("localization_head") == "activated" and not visual_written:
            errors.append(_err("visual_artifacts_written must be true when activated visual artifacts are requested"))
        if visual_written:
            report_root = config.get("report_root", "")
            _validate_artifact_paths(VISUAL_PATH_FIELDS, report, report_root, errors, "visual artifacts")
        clean_requested = config.get("write_clean_red_overlay") is True
        clean_written = report.get("clean_red_overlay_written") is True
        if clean_requested and report.get("localization_head") == "activated" and not clean_written:
            errors.append(_err("clean_red_overlay_written must be true when activated clean red overlay is requested"))
        if clean_written:
            report_root = config.get("report_root", "")
            _validate_artifact_paths(CLEAN_OVERLAY_PATH_FIELDS, report, report_root, errors, "clean red overlay artifacts")
            area = report.get("clean_overlay_area_pct")
            if isinstance(area, bool) or not isinstance(area, (int, float)) or not 0.0 <= float(area) <= 100.0:
                errors.append(_err("clean_overlay_area_pct must be numeric in [0, 100]"))
            _validate_clean_overlay_pixels(report, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in {1, 2}:
        print("usage: validate_pre_sns_inference_report.py <config.json> [report-json]", file=sys.stderr)
        return 2
    try:
        config = load_report_config(argv[0])
    except Exception as exc:
        print(f"failed to read config: {exc}", file=sys.stderr)
        return 1
    config_errors = validate_report_config(config)
    if config_errors:
        print("pre-SNS inference report config validation failed:", file=sys.stderr)
        for error in config_errors:
            print(error, file=sys.stderr)
        return 1
    if len(argv) == 1:
        print(f"{CONFIG_OK_MARKER}: config is valid")
        return 0
    try:
        report = parse_report_json(argv[1])
    except Exception as exc:
        print(f"failed to read report JSON: {exc}", file=sys.stderr)
        return 1
    report_errors = validate_report(config, report)
    if report_errors:
        print("pre-SNS single-image report validation failed:", file=sys.stderr)
        for error in report_errors:
            print(error, file=sys.stderr)
        return 1
    print(f"{REPORT_VALIDATED_MARKER}: report is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
