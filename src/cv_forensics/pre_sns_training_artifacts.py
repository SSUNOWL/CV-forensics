"""Artifact helpers for guarded pre-SNS pilot training.

The helpers here only manage explicit run/checkpoint roots supplied by an
approved config. They do not read datasets, scan directories, download data, or
train models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ARTIFACT_FILES = (
    "config_snapshot.json",
    "manifest_snapshot.json",
    "metrics_summary.json",
    "run_summary.json",
    "artifact_manifest.json",
)
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}
CREDENTIAL_WORDS = ("secret", "credential", "password", "token", "private_key", "api_key")


class ArtifactPolicyError(ValueError):
    """Raised when an artifact path violates the pre-SNS artifact policy."""


def _real(path: str | Path) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def _is_under(path: str | Path, root: str | Path) -> bool:
    path_real = _real(path)
    root_real = _real(root)
    try:
        return os.path.commonpath([str(path_real), str(root_real)]) == str(root_real)
    except ValueError:
        return False


def _path_parts(path: str | Path) -> list[str]:
    return [part for part in Path(os.fspath(path)).parts if part not in {os.sep, ""}]


def path_is_inside_repository(path: str | Path, repo_root: str | Path = REPO_ROOT) -> bool:
    return _is_under(path, repo_root)


def path_is_repo_outputs_or_checkpoints(path: str | Path, repo_root: str | Path = REPO_ROOT) -> bool:
    repo = _real(repo_root)
    return _is_under(path, repo / "outputs") or _is_under(path, repo / "checkpoints")


def validate_artifact_path(path: str | Path, repo_root: str | Path = REPO_ROOT) -> Path:
    path_text = os.fspath(path)
    if not path_text or not os.path.isabs(path_text):
        raise ArtifactPolicyError(f"artifact path must be absolute: {path}")
    if any(part == ".." for part in Path(path_text).parts):
        raise ArtifactPolicyError(f"artifact path must not contain traversal: {path}")
    parts = _path_parts(path_text)
    lowered_parts = [part.lower() for part in parts]
    for part in lowered_parts:
        if part in PROTECTED_PARTS:
            raise ArtifactPolicyError(f"artifact path uses protected segment {part}: {path}")
        if part.startswith(".") and any(word in part for word in CREDENTIAL_WORDS):
            raise ArtifactPolicyError(f"artifact path uses hidden credential-like segment: {path}")
        if any(word in part for word in CREDENTIAL_WORDS):
            raise ArtifactPolicyError(f"artifact path uses credential-like segment: {path}")
    if path_is_repo_outputs_or_checkpoints(path_text, repo_root):
        raise ArtifactPolicyError(f"artifact path must not be inside repository outputs/checkpoints: {path}")
    return _real(path_text)


def validate_artifact_root(path: str | Path, repo_root: str | Path = REPO_ROOT) -> Path:
    root = validate_artifact_path(path, repo_root)
    if path_is_inside_repository(root, repo_root):
        raise ArtifactPolicyError(f"artifact root must be outside repository: {path}")
    return root


def _is_empty_directory(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is None


def prepare_artifact_root(path: str | Path, repo_root: str | Path = REPO_ROOT, overwrite: bool = False) -> Path:
    root = validate_artifact_root(path, repo_root)
    if root.exists():
        if not root.is_dir():
            raise ArtifactPolicyError(f"artifact root exists but is not a directory: {root}")
        if not overwrite and not _is_empty_directory(root):
            raise ArtifactPolicyError(f"artifact root is non-empty and overwrite is not enabled: {root}")
    else:
        root.mkdir(parents=True, exist_ok=False)
    return root


def prepare_artifact_roots(
    run_root: str | Path,
    checkpoint_root: str | Path,
    repo_root: str | Path = REPO_ROOT,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    run = validate_artifact_root(run_root, repo_root)
    checkpoint = validate_artifact_root(checkpoint_root, repo_root)
    if run == checkpoint:
        raise ArtifactPolicyError("approved_run_root and approved_checkpoint_root must be distinct")
    return (
        prepare_artifact_root(run, repo_root=repo_root, overwrite=overwrite),
        prepare_artifact_root(checkpoint, repo_root=repo_root, overwrite=overwrite),
    )


def write_json_atomic(path: str | Path, payload: Any, repo_root: str | Path = REPO_ROOT) -> Path:
    target = validate_artifact_path(path, repo_root)
    if path_is_inside_repository(target, repo_root):
        raise ArtifactPolicyError(f"artifact file must be outside repository: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, target)
    return target


def file_record(path: str | Path, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    target = validate_artifact_path(path, repo_root)
    if not target.is_file():
        raise ArtifactPolicyError(f"artifact file does not exist: {target}")
    return {"path": str(target), "size_bytes": int(target.stat().st_size)}


def write_training_artifacts(
    *,
    run_root: str | Path,
    checkpoint_root: str | Path,
    config_snapshot: dict[str, Any],
    manifest_snapshot: dict[str, Any],
    metrics_summary: dict[str, Any],
    run_summary: dict[str, Any],
    checkpoint_path: str | Path | None,
    repo_root: str | Path = REPO_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_dir, checkpoint_dir = prepare_artifact_roots(run_root, checkpoint_root, repo_root=repo_root, overwrite=overwrite)
    artifact_payloads = {
        "config_snapshot.json": config_snapshot,
        "manifest_snapshot.json": manifest_snapshot,
        "metrics_summary.json": metrics_summary,
        "run_summary.json": run_summary,
    }
    records: dict[str, dict[str, Any]] = {}
    for filename, payload in artifact_payloads.items():
        written = write_json_atomic(run_dir / filename, payload, repo_root=repo_root)
        records[filename] = file_record(written, repo_root=repo_root)
    checkpoint_record = None
    if checkpoint_path is not None:
        checkpoint_target = validate_artifact_path(checkpoint_path, repo_root)
        if not _is_under(checkpoint_target, checkpoint_dir):
            raise ArtifactPolicyError("checkpoint_path must be under approved_checkpoint_root")
        checkpoint_record = file_record(checkpoint_target, repo_root=repo_root)
    manifest = {
        "artifact_kind": "pre_sns_training_artifacts",
        "required_files": list(REQUIRED_ARTIFACT_FILES),
        "run_root": str(run_dir),
        "checkpoint_root": str(checkpoint_dir),
        "files": records,
        "checkpoint": checkpoint_record,
    }
    manifest_path = write_json_atomic(run_dir / "artifact_manifest.json", manifest, repo_root=repo_root)
    manifest["files"]["artifact_manifest.json"] = file_record(manifest_path, repo_root=repo_root)
    write_json_atomic(manifest_path, manifest, repo_root=repo_root)
    return {
        "artifact_manifest_path": str(manifest_path),
        "artifacts": manifest["files"],
        "checkpoint": checkpoint_record,
    }
