#!/usr/bin/env python3
"""Validate pre-SNS baseline report configs and generated summaries."""

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

from cv_forensics.pre_sns_baseline_report import (  # noqa: E402
    CONFIG_OK_MARKER,
    MARKER,
    REQUIRED_SECTIONS,
    RESULT_OK_MARKER,
    load_report_config,
    parse_json_with_logs,
    validate_report_config,
)


def _err(message: str) -> str:
    return f"- {message}"


def _is_under(path: str | Path, root: str | Path) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(os.fspath(path)), os.path.realpath(os.fspath(root))]) == os.path.realpath(os.fspath(root))
    except ValueError:
        return False


def _inside_repo(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT)


def _repo_outputs_or_checkpoints(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT / "outputs") or _is_under(path, REPO_ROOT / "checkpoints")


def validate_report_result(config: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if result.get("marker") != MARKER:
        errors.append(_err("marker must be PRE_SNS_BASELINE_REPORT_OK"))
    if result.get("no_sns_augmentation") is not True:
        errors.append(_err("no_sns_augmentation must be true"))
    markdown = result.get("markdown")
    sections = result.get("required_sections")
    for section in REQUIRED_SECTIONS:
        present_in_markdown = isinstance(markdown, str) and f"## {section}" in markdown
        present_in_sections = isinstance(sections, list) and section in sections
        if not present_in_markdown and not present_in_sections:
            errors.append(_err(f"required section missing: {section}"))
    metrics = result.get("evaluation_metrics")
    if not isinstance(metrics, dict):
        errors.append(_err("evaluation_metrics must be an object"))
    else:
        for field in ("accuracy_3way", "macro_f1_3way", "family_accuracy", "mask_iou_mean", "localization_activation_recall", "latency_ms_mean", "fps_estimate"):
            if field not in metrics:
                errors.append(_err(f"evaluation_metrics.{field} is required"))
    if not isinstance(result.get("single_image_report_summary"), dict):
        errors.append(_err("single_image_report_summary is required"))
    if config.get("write_report") is True:
        for field in ("report_markdown_path", "report_summary_path"):
            value = result.get(field)
            if not isinstance(value, str) or not value:
                errors.append(_err(f"{field} is required when write_report=true"))
                continue
            if not os.path.isfile(value):
                errors.append(_err(f"{field} must exist"))
            if _inside_repo(value):
                errors.append(_err(f"{field} must be outside the repository"))
            if _repo_outputs_or_checkpoints(value):
                errors.append(_err(f"{field} must not be inside repository outputs/checkpoints"))
            if not _is_under(value, config.get("approved_report_root", "")):
                errors.append(_err(f"{field} must be under approved_report_root"))
    return errors


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in {1, 2}:
        print("usage: validate_pre_sns_baseline_report.py <config.json> [report-result-json]", file=sys.stderr)
        return 2
    try:
        config = load_report_config(argv[0])
    except Exception as exc:
        print(f"failed to read config: {exc}", file=sys.stderr)
        return 1
    config_errors = validate_report_config(config)
    if config_errors:
        print("pre-SNS baseline report config validation failed:", file=sys.stderr)
        for error in config_errors:
            print(error, file=sys.stderr)
        return 1
    if len(argv) == 1:
        print(f"{CONFIG_OK_MARKER}: config is valid")
        return 0
    try:
        result = parse_json_with_logs(argv[1])
    except Exception as exc:
        print(f"failed to read report result JSON: {exc}", file=sys.stderr)
        return 1
    result_errors = validate_report_result(config, result)
    if result_errors:
        print("pre-SNS baseline report result validation failed:", file=sys.stderr)
        for error in result_errors:
            print(error, file=sys.stderr)
        return 1
    print(f"{RESULT_OK_MARKER}: report result is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
