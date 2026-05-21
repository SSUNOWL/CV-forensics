#!/usr/bin/env python3
"""Validate pre-SNS training artifacts from an approved actual run."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_training_artifacts import (  # noqa: E402
    REQUIRED_ARTIFACT_FILES,
    path_is_inside_repository,
    path_is_repo_outputs_or_checkpoints,
    validate_artifact_root,
)


OK_MARKER = "PRE_SNS_TRAINING_ARTIFACTS_OK"
RUN_MARKER = "PRE_SNS_BASELINE_TRAINING_RUN_OK"
DRY_RUN_MARKER = "PRE_SNS_BASELINE_TRAINING_DRY_RUN_OK"
ENTRYPOINT_MARKER = "PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK"
MAX_ARTIFACT_BYTES = 5 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 25 * 1024 * 1024


def _err(message: str) -> str:
    return f"- {message}"


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} root must be a JSON object")
    return raw


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
    raise ValueError("training result file does not contain a JSON object")


def _path_under(path: str | Path, root: str | Path) -> bool:
    real_path = os.path.realpath(os.fspath(path))
    real_root = os.path.realpath(os.fspath(root))
    try:
        return os.path.commonpath([real_path, real_root]) == real_root
    except ValueError:
        return False


def _validate_path_policy(path_value: Any, field: str, errors: list[str]) -> None:
    if not isinstance(path_value, str) or not path_value:
        errors.append(_err(f"{field} must be a non-empty path string"))
        return
    try:
        validate_artifact_root(path_value, REPO_ROOT)
    except Exception as exc:
        errors.append(_err(f"{field}: {exc}"))
    if path_is_inside_repository(path_value, REPO_ROOT):
        errors.append(_err(f"{field} must be outside the repository"))
    if path_is_repo_outputs_or_checkpoints(path_value, REPO_ROOT):
        errors.append(_err(f"{field} must not be inside repository outputs/checkpoints"))


def _validate_file(path_value: Any, field: str, max_bytes: int, errors: list[str]) -> None:
    if not isinstance(path_value, str) or not path_value:
        errors.append(_err(f"{field} must be a non-empty path string"))
        return
    path = Path(path_value)
    if not path.is_file():
        errors.append(_err(f"{field} does not exist as a file: {path_value}"))
        return
    if path.stat().st_size > max_bytes:
        errors.append(_err(f"{field} is too large: {path.stat().st_size} bytes"))
    if path_is_repo_outputs_or_checkpoints(path, REPO_ROOT):
        errors.append(_err(f"{field} must not be inside repository outputs/checkpoints"))
    if path_is_inside_repository(path, REPO_ROOT):
        errors.append(_err(f"{field} must be outside the repository"))


def _load_artifact_manifest(path: str, errors: list[str]) -> dict[str, Any]:
    try:
        manifest = load_json(path)
    except Exception as exc:
        errors.append(_err(f"failed to read artifact_manifest.json: {exc}"))
        return {}
    files = manifest.get("files")
    if not isinstance(files, dict):
        errors.append(_err("artifact_manifest.json must contain files object"))
        return manifest
    for filename in REQUIRED_ARTIFACT_FILES:
        if filename not in files:
            errors.append(_err(f"artifact_manifest.json missing listed file: {filename}"))
            continue
        record = files[filename]
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            errors.append(_err(f"artifact_manifest.json has invalid record for {filename}"))
            continue
        _validate_file(record["path"], f"artifact_manifest.files.{filename}.path", MAX_ARTIFACT_BYTES, errors)
    return manifest


def validate(config: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    no_write = result.get("no_write_dry_run") is True or config.get("no_write_dry_run") is True
    marker = result.get("marker")
    if no_write:
        if marker != DRY_RUN_MARKER:
            errors.append(_err("dry-run result marker must be PRE_SNS_BASELINE_TRAINING_DRY_RUN_OK"))
        if result.get("entrypoint_marker") != ENTRYPOINT_MARKER:
            errors.append(_err("dry-run result entrypoint_marker must be PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK"))
        if result.get("training_started") is not True:
            errors.append(_err("dry-run result must report training_started true after the smoke loop starts"))
        if result.get("training_completed") is not True:
            errors.append(_err("dry-run result must report training_completed true after the smoke loop completes"))
        if result.get("total_loss_finite") is not True:
            errors.append(_err("dry-run result must report total_loss_finite true"))
        return errors

    if marker != RUN_MARKER:
        errors.append(_err("actual run marker must be PRE_SNS_BASELINE_TRAINING_RUN_OK"))
    if result.get("entrypoint_marker") != ENTRYPOINT_MARKER:
        errors.append(_err("actual run entrypoint_marker must be PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK"))
    if result.get("total_loss_finite") is not True:
        errors.append(_err("total_loss_finite must be true"))
    if result.get("training_started") is not True:
        errors.append(_err("training_started must be true"))
    if result.get("training_completed") is not True:
        errors.append(_err("training_completed must be true"))
    for flag in ("no_download", "no_network", "no_sns_augmentation"):
        if result.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    run_root = result.get("approved_run_root", config.get("approved_run_root"))
    checkpoint_root = result.get("approved_checkpoint_root", config.get("approved_checkpoint_root"))
    _validate_path_policy(run_root, "approved_run_root", errors)
    _validate_path_policy(checkpoint_root, "approved_checkpoint_root", errors)

    artifact_manifest_path = result.get("artifact_manifest_path")
    _validate_file(artifact_manifest_path, "artifact_manifest_path", MAX_ARTIFACT_BYTES, errors)
    if isinstance(artifact_manifest_path, str) and isinstance(run_root, str) and not _path_under(artifact_manifest_path, run_root):
        errors.append(_err("artifact_manifest_path must be under approved_run_root"))
    if isinstance(artifact_manifest_path, str) and Path(artifact_manifest_path).is_file():
        _load_artifact_manifest(artifact_manifest_path, errors)

    checkpoint_path = result.get("checkpoint_path")
    _validate_file(checkpoint_path, "checkpoint_path", MAX_CHECKPOINT_BYTES, errors)
    if isinstance(checkpoint_path, str) and isinstance(checkpoint_root, str) and not _path_under(checkpoint_path, checkpoint_root):
        errors.append(_err("checkpoint_path must be under approved_checkpoint_root"))
    return errors


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: validate_pre_sns_training_artifacts.py <training-config.json> <training-result-json>", file=sys.stderr)
        return 2
    try:
        config = load_json(argv[0])
        result = parse_result_json(argv[1])
    except Exception as exc:
        print(f"failed to read validation inputs: {exc}", file=sys.stderr)
        return 1
    errors = validate(config, result)
    if errors:
        print("pre-SNS training artifact validation failed:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"{OK_MARKER}: artifacts are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
