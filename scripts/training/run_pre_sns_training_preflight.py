#!/usr/bin/env python3
"""Run the approved local pre-SNS training preflight dry-run."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_integrated_model import (  # noqa: E402
    CLASS_TO_INDEX,
    FAMILY_TO_INDEX,
    build_tiny_integrated_model,
    compute_integrated_losses,
)
from cv_forensics.pre_sns_preflight import ensure_ready, summarize_manifest_readiness  # noqa: E402
from scripts.agent.validate_pre_sns_training_preflight_config import load_config, validate_config  # noqa: E402


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _load_dependencies():
    try:
        import torch
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("torch and PIL are required for the preflight dry-run") from exc
    return torch, Image


def _image_to_flat_values(image: Any, image_size: int) -> list[float]:
    image = image.convert("RGB").resize((image_size, image_size))
    flat = []
    for pixel in image.getdata():
        flat.extend(channel / 255.0 for channel in pixel)
    return flat


def _mask_to_flat_values(mask: Any, image_size: int) -> list[float]:
    mask = mask.convert("L").resize((image_size, image_size))
    return [1.0 if value >= 128 else 0.0 for value in mask.getdata()]


def _prepare_batch(raw: dict[str, Any], manifest: dict[str, Any], torch: Any, Image: Any):
    limits = raw["tiny_preflight_limits"]
    image_size = int(limits["max_image_size"])
    samples = manifest["samples"][: int(limits["max_samples"])]
    images = []
    class_targets = []
    family_targets = []
    mask_targets = []
    class_labels = []
    has_mask = []
    for sample in samples:
        with Image.open(sample["image_path"]) as image:
            images.append(_image_to_flat_values(image, image_size))
        class_label = sample["class_label"]
        class_labels.append(class_label)
        class_targets.append(CLASS_TO_INDEX[class_label])
        family_label = sample.get("family_label")
        family_targets.append(FAMILY_TO_INDEX[family_label] if family_label in FAMILY_TO_INDEX else None)
        if class_label == "tampered" and isinstance(sample.get("mask_path"), str):
            with Image.open(sample["mask_path"]) as mask:
                mask_targets.append(_mask_to_flat_values(mask, image_size))
            has_mask.append(True)
        else:
            has_mask.append(False)
    return (
        torch.tensor(images, dtype=torch.float32),
        torch.tensor(class_targets, dtype=torch.long),
        family_targets,
        torch.tensor(mask_targets, dtype=torch.float32),
        class_labels,
        has_mask,
    )


def run_preflight(raw: dict[str, Any]) -> dict[str, Any]:
    torch, Image = _load_dependencies()
    torch.manual_seed(23)
    with open(raw["manifest_path"], "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    readiness = summarize_manifest_readiness(
        manifest,
        max_samples=int(raw["tiny_preflight_limits"]["max_samples"]),
        cpu_only=bool(raw["tiny_preflight_limits"]["cpu_only"]),
    )
    ensure_ready(readiness)
    x, y_class, family_targets, y_mask, class_labels, has_mask = _prepare_batch(raw, manifest, torch, Image)
    model = build_tiny_integrated_model(torch, x.shape[1], int(raw["tiny_preflight_limits"]["max_image_size"]) ** 2)
    model.train()
    outputs = model(x)
    losses = compute_integrated_losses(torch, outputs, y_class, family_targets, y_mask, class_labels, has_mask)
    if not losses["total_loss_finite"]:
        raise RuntimeError("non-finite total loss in preflight")
    model.zero_grad()
    losses["total_loss"].backward()
    return {
        "marker": "PRE_SNS_TRAINING_PREFLIGHT_OK",
        "readiness": readiness,
        "samples_seen": int(x.shape[0]),
        "class_loss_finite": losses["class_loss_finite"],
        "family_loss_finite": losses["family_loss_finite"],
        "localization_loss_finite": losses["localization_loss_finite"],
        "total_loss_finite": losses["total_loss_finite"],
        "loss_routing": losses["routing"],
        "optimizer_step_ran": False,
        "no_download": raw["no_download"],
        "no_network": raw["no_network"],
        "no_outputs": raw["no_outputs"],
        "no_checkpoints": raw["no_checkpoints"],
        "no_real_training": raw["no_real_training"],
        "no_sns_augmentation": raw["no_sns_augmentation"],
        "result_scope": "CPU-only forward/backward preflight; not real training or a performance claim",
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        return _fail("usage: run_pre_sns_training_preflight.py <approved-config.json>")
    try:
        raw = load_config(argv[0])
    except Exception as exc:
        return _fail(f"failed to read config: {exc}")
    errors = validate_config(raw)
    if errors:
        print("config validation failed before loading manifest:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    if raw.get("config_kind") != "approved_local_pre_sns_training_preflight":
        return _fail("runner requires config_kind approved_local_pre_sns_training_preflight")
    try:
        summary = run_preflight(raw)
    except Exception as exc:
        return _fail(str(exc))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
