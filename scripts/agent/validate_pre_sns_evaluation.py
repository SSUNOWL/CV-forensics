#!/usr/bin/env python3
"""Validate pre-SNS evaluation configs and result JSON."""

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

from cv_forensics.pre_sns_evaluation import (  # noqa: E402
    CONFIG_OK_MARKER,
    MARKER,
    RESULT_OK_MARKER,
    load_evaluation_config,
    validate_evaluation_config,
)

METRIC_FIELDS = (
    "accuracy_3way",
    "macro_f1_3way",
    "family_accuracy",
    "mask_iou_mean",
    "localization_activation_recall",
    "latency_ms_mean",
    "fps_estimate",
    "samples_evaluated",
)


def _err(message: str) -> str:
    return f"- {message}"


def parse_result_json(path: str | Path) -> dict[str, Any]:
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
    raise ValueError("evaluation-result-json does not contain a JSON object")


def _is_under(path: str | Path, root: str | Path) -> bool:
    real_path = os.path.realpath(os.fspath(path))
    real_root = os.path.realpath(os.fspath(root))
    try:
        return os.path.commonpath([real_path, real_root]) == real_root
    except ValueError:
        return False


def _inside_repo(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT)


def _repo_outputs_or_checkpoints(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT / "outputs") or _is_under(path, REPO_ROOT / "checkpoints")


def _validate_nullable_unit_metric(result: dict[str, Any], field: str, errors: list[str]) -> None:
    value = result.get(field)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(_err(f"{field} must be null or numeric"))
        return
    if not 0.0 <= float(value) <= 1.0:
        errors.append(_err(f"{field} must be in [0, 1]"))


def validate_evaluation_result(config: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if result.get("marker") != MARKER:
        errors.append(_err("marker must be PRE_SNS_EVALUATION_OK"))
    for field in METRIC_FIELDS:
        if field not in result:
            errors.append(_err(f"{field} is required"))
    for field in (
        "accuracy_3way",
        "macro_f1_3way",
        "family_accuracy",
        "mask_iou_mean",
        "localization_activation_recall",
    ):
        _validate_nullable_unit_metric(result, field, errors)
    for field in ("latency_ms_mean", "fps_estimate"):
        value = result.get(field)
        if value is None:
            errors.append(_err(f"{field} must be numeric"))
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0:
            errors.append(_err(f"{field} must be non-negative numeric"))
    samples = result.get("samples_evaluated")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
        errors.append(_err("samples_evaluated must be a non-negative integer"))
    for flag in ("no_download", "no_network", "no_training", "no_sns_augmentation"):
        if result.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if config.get("write_eval_artifact") is True:
        artifact = result.get("evaluation_artifact_path")
        if not isinstance(artifact, str) or not artifact:
            errors.append(_err("evaluation_artifact_path is required when write_eval_artifact=true"))
        else:
            if _inside_repo(artifact):
                errors.append(_err("evaluation_artifact_path must be outside the repository"))
            if _repo_outputs_or_checkpoints(artifact):
                errors.append(_err("evaluation_artifact_path must not be inside repository outputs/checkpoints"))
            if not _is_under(artifact, config.get("approved_eval_root", "")):
                errors.append(_err("evaluation_artifact_path must be under approved_eval_root"))
    return errors


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in {1, 2}:
        print("usage: validate_pre_sns_evaluation.py <config.json> [evaluation-result-json]", file=sys.stderr)
        return 2
    try:
        config = load_evaluation_config(argv[0])
    except Exception as exc:
        print(f"failed to read config: {exc}", file=sys.stderr)
        return 1
    config_errors = validate_evaluation_config(config)
    if config_errors:
        print("pre-SNS evaluation config validation failed:", file=sys.stderr)
        for error in config_errors:
            print(error, file=sys.stderr)
        return 1
    if len(argv) == 1:
        print(f"{CONFIG_OK_MARKER}: config is valid")
        return 0
    try:
        result = parse_result_json(argv[1])
    except Exception as exc:
        print(f"failed to read evaluation result JSON: {exc}", file=sys.stderr)
        return 1
    result_errors = validate_evaluation_result(config, result)
    if result_errors:
        print("pre-SNS evaluation result validation failed:", file=sys.stderr)
        for error in result_errors:
            print(error, file=sys.stderr)
        return 1
    print(f"{RESULT_OK_MARKER}: result is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
