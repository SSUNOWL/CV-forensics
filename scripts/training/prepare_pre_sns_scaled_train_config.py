#!/usr/bin/env python3
"""Prepare an approved scaled pre-SNS training config without training."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent.validate_pre_sns_scaled_training_plan import (  # noqa: E402
    APPROVAL_TEXT,
    OK_MARKER,
    validate_scaled_training_plan,
)

CREATED_MARKER = "PRE_SNS_SCALED_TRAIN_CONFIG_CREATED_OK"
STALE_OUTPUT_ROOT_FIELDS = {
    "run_root",
    "checkpoint_root",
    "artifact_root",
    "output_root",
    "approved_local_output_roots",
    "dataset_manifest_path",
    "manifest_path",
    "train_manifest_path",
    "val_manifest_path",
}


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} root must be a JSON object")
    return raw


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_scaled_config(
    base_config: dict[str, Any],
    manifest_path: str,
    scale: str,
    max_samples: int,
    epochs: int,
    batch_size: int,
    max_image_size: int,
    device: str,
    stamp: str | None = None,
) -> dict[str, Any]:
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path.home() / "cvf_runs" / f"pre_sns_{scale}_{stamp}"
    checkpoint_root = Path.home() / "cvf_checkpoints" / f"pre_sns_{scale}_{stamp}"
    approval = str(base_config.get("user_approval_text", ""))
    approved = approval == APPROVAL_TEXT

    scaled = {key: value for key, value in base_config.items() if key not in STALE_OUTPUT_ROOT_FIELDS}
    scaled.update(
        {
            "schema_version": "1.0",
            "config_kind": "approved_pre_sns_baseline_training",
            "execution_mode": "approved_local_pre_sns_scaled_training",
            "scale": scale,
            "marker": OK_MARKER,
            "train_name": f"pre_sns_{scale}",
            "approved_training_run": bool(approved),
            "required_approval_text": APPROVAL_TEXT,
            "user_approval_text": approval,
            "no_write_dry_run": False if approved else True,
            "no_download": True,
            "no_network": True,
            "no_sns_augmentation": True,
            "unified_manifest_path": str(Path(manifest_path).resolve()),
            "approved_run_root": str(run_root),
            "approved_checkpoint_root": str(checkpoint_root),
            "approved_local_roots": [str(Path(manifest_path).resolve().parent)],
            "device": device,
            "max_samples": int(max_samples),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "max_image_size": int(max_image_size),
            "result_scope": "scaled pre-SNS baseline training; not final full-dataset performance",
            "scaled_training_policy": {
                "config_generator_only": True,
                "training_started": False,
                "checkpoint_written": False,
                "sns_augmentation": False,
            },
            "validation_notes": [
                "Generated config only; training has not run.",
                "SNS augmentation is not part of this task.",
                OK_MARKER,
            ],
        }
    )
    return scaled


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--max-samples", type=_positive_int, required=True)
    parser.add_argument("--epochs", type=_positive_int, required=True)
    parser.add_argument("--batch-size", type=_positive_int, required=True)
    parser.add_argument("--max-image-size", type=_positive_int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_path = Path(args.base_config)
    manifest_path = Path(args.manifest)
    out_path = Path(args.out)
    if not base_path.is_file():
        print(f"base config does not exist: {base_path}", file=sys.stderr)
        return 1
    if not manifest_path.is_file():
        print(f"manifest does not exist: {manifest_path}", file=sys.stderr)
        return 1
    try:
        base = load_json(base_path)
        scaled = build_scaled_config(
            base,
            str(manifest_path),
            args.scale,
            args.max_samples,
            args.epochs,
            args.batch_size,
            args.max_image_size,
            args.device,
        )
        errors = validate_scaled_training_plan(scaled)
    except Exception as exc:
        print(f"failed to create scaled config: {exc}", file=sys.stderr)
        return 1
    if errors:
        print("generated scaled config failed validation:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(scaled, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(CREATED_MARKER)
    print(f"config: {out_path}")
    print(f"approved_run_root: {scaled['approved_run_root']}")
    print(f"approved_checkpoint_root: {scaled['approved_checkpoint_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
