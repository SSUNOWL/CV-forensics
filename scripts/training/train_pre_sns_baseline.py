#!/usr/bin/env python3
"""Guarded pre-SNS baseline training entrypoint.

This script validates its config before loading any manifest, image, or mask.
The tracked tests exercise only gate behavior and no-write dry-run mode.
"""

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

from scripts.agent.validate_pre_sns_baseline_train_config import (  # noqa: E402
    APPROVAL_TEXT,
    load_config,
    validate_config,
)


ENTRYPOINT_MARKER = "PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK"
STARTED_MARKER = "PRE_SNS_BASELINE_TRAINING_STARTED"


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _load_optional_runtime_dependencies():
    try:
        import torch
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("torch and PIL are required for approved training execution") from exc
    return torch, Image


def _load_manifest(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("unified manifest must be a JSON object")
    if not isinstance(manifest.get("samples"), list):
        raise ValueError("unified manifest must contain a samples list")
    return manifest


def _manifest_path_summary(manifest: dict[str, Any]) -> dict[str, int]:
    image_paths = 0
    mask_paths = 0
    for sample in manifest.get("samples", []):
        if not isinstance(sample, dict):
            continue
        if isinstance(sample.get("image_path"), str):
            image_paths += 1
        if isinstance(sample.get("mask_path"), str):
            mask_paths += 1
    return {"image_paths_listed": image_paths, "mask_paths_listed": mask_paths}


def _select_device(raw: dict[str, Any], torch: Any) -> str:
    requested = raw.get("device", "cpu")
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def run_entrypoint(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("approved_training_run") is not True:
        return {
            "marker": ENTRYPOINT_MARKER,
            "training_started": False,
            "reason": "approved_training_run is false; exited before manifest loading, image loading, and writes",
            "no_download": raw.get("no_download") is True,
            "no_network": raw.get("no_network") is True,
            "no_sns_augmentation": raw.get("no_sns_augmentation") is True,
            "result_scope": "guard check only; no real training",
        }
    if raw.get("user_approval_text") != APPROVAL_TEXT:
        raise ValueError("missing exact approval text for approved training")

    no_write = raw.get("no_write_dry_run") is True
    manifest = _load_manifest(raw["unified_manifest_path"])
    path_summary = _manifest_path_summary(manifest)

    if no_write:
        return {
            "marker": ENTRYPOINT_MARKER,
            "training_started": False,
            "no_write_dry_run": True,
            "manifest_sample_count": len(manifest.get("samples", [])),
            **path_summary,
            "no_download": raw["no_download"],
            "no_network": raw["no_network"],
            "no_sns_augmentation": raw["no_sns_augmentation"],
            "result_scope": "approved config dry-run; no artifacts, image reads, masks reads, or training",
        }

    torch, _Image = _load_optional_runtime_dependencies()
    device = _select_device(raw, torch)
    os.makedirs(raw["approved_run_root"], exist_ok=False)
    os.makedirs(raw["approved_checkpoint_root"], exist_ok=False)
    return {
        "marker": ENTRYPOINT_MARKER,
        "started_marker": STARTED_MARKER,
        "training_started": True,
        "device": device,
        "manifest_sample_count": len(manifest.get("samples", [])),
        **path_summary,
        "max_samples": raw["max_samples"],
        "epochs": raw["epochs"],
        "result_scope": "guarded training entrypoint initialized; real training loop is intentionally minimal pending manual run policy",
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        return _fail("usage: train_pre_sns_baseline.py <config.json>")
    try:
        raw = load_config(argv[0])
    except Exception as exc:
        return _fail(f"failed to read config: {exc}")
    errors = validate_config(raw)
    if errors:
        print("config validation failed before loading manifest or artifacts:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    try:
        summary = run_entrypoint(raw)
    except Exception as exc:
        return _fail(str(exc))
    if summary.get("training_started"):
        print(STARTED_MARKER, file=sys.stderr)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
