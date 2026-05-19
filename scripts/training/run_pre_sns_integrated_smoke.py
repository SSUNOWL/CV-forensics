#!/usr/bin/env python3
"""Run a future approved pre-SNS integrated tiny smoke."""

from __future__ import annotations

import json
import random
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
    build_schema_output_summary,
    build_tiny_integrated_model,
    compute_integrated_losses,
)
from scripts.agent.validate_pre_sns_integrated_smoke_config import load_config, validate_config  # noqa: E402


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _load_dependencies():
    try:
        import torch
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("torch and PIL are required for the manual integrated smoke runner") from exc
    return torch, Image


def _run_limits(raw: dict[str, Any]) -> dict[str, int]:
    tiny_limits = raw["tiny_limits"]
    requested = raw.get("requested_run_limits") or {}
    return {
        "max_samples": min(int(requested.get("max_samples", tiny_limits["max_samples"])), int(tiny_limits["max_samples"])),
        "max_steps": min(int(requested.get("max_steps", tiny_limits["max_steps"])), int(tiny_limits["max_steps"])),
        "max_epochs": min(int(requested.get("max_epochs", tiny_limits["max_epochs"])), int(tiny_limits["max_epochs"])),
        "max_image_size": int(tiny_limits["max_image_size"]),
    }


def _image_to_flat_values(image: Any, image_size: int) -> list[float]:
    image = image.convert("RGB").resize((image_size, image_size))
    flat = []
    for pixel in image.getdata():
        flat.extend(channel / 255.0 for channel in pixel)
    return flat


def _mask_to_flat_values(mask: Any, image_size: int) -> list[float]:
    mask = mask.convert("L").resize((image_size, image_size))
    return [1.0 if value >= 128 else 0.0 for value in mask.getdata()]


def _prepare_batch(raw: dict[str, Any], torch: Any, Image: Any):
    limits = _run_limits(raw)
    samples = raw["sample_manifest"][: limits["max_samples"]]
    image_size = limits["max_image_size"]
    images = []
    class_targets = []
    family_targets = []
    mask_targets = []
    has_mask = []
    class_labels = []
    family_labels_seen: set[str] = set()
    for sample in samples:
        with Image.open(sample["image_path"]) as image:
            images.append(_image_to_flat_values(image, image_size))
        class_label = sample["class_label"]
        class_labels.append(class_label)
        class_targets.append(CLASS_TO_INDEX[class_label])
        family_label = sample.get("family_label")
        if family_label is None:
            family_targets.append(None)
        else:
            family_targets.append(FAMILY_TO_INDEX[family_label])
            family_labels_seen.add(family_label)
        if class_label == "tampered":
            with Image.open(sample["mask_path"]) as mask:
                mask_targets.append(_mask_to_flat_values(mask, image_size))
            has_mask.append(True)
        else:
            has_mask.append(False)
    x = torch.tensor(images, dtype=torch.float32)
    y_class = torch.tensor(class_targets, dtype=torch.long)
    y_mask = torch.tensor(mask_targets, dtype=torch.float32)
    return x, y_class, family_targets, y_mask, class_labels, has_mask, sorted(set(class_labels)), sorted(family_labels_seen)


def run_smoke(raw: dict[str, Any]) -> dict[str, Any]:
    torch, Image = _load_dependencies()
    torch.manual_seed(17)
    random.seed(17)
    if raw.get("tiny_limits", {}).get("cpu_only") is not True:
        raise RuntimeError("integrated smoke runner requires cpu_only true")
    limits = _run_limits(raw)
    x, y_class, family_targets, y_mask, class_labels, has_mask, class_seen, family_seen = _prepare_batch(raw, torch, Image)
    model = build_tiny_integrated_model(torch, x.shape[1], limits["max_image_size"] * limits["max_image_size"])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    family_weight = float(raw.get("training_smoke_policy", {}).get("family_loss_weight", 1.0))
    localization_weight = float(raw.get("training_smoke_policy", {}).get("localization_loss_weight", 1.0))

    initial_total_loss = None
    final_total_loss = None
    losses = None
    steps_completed = 0
    epochs_completed = 0
    last_outputs = None
    for epoch in range(limits["max_epochs"]):
        epochs_completed = epoch + 1
        outputs = model(x)
        losses = compute_integrated_losses(
            torch,
            outputs,
            y_class,
            family_targets,
            y_mask,
            class_labels,
            has_mask,
            family_loss_weight=family_weight,
            localization_loss_weight=localization_weight,
        )
        if initial_total_loss is None:
            initial_total_loss = float(losses["total_loss"].detach().item())
        if not losses["total_loss_finite"]:
            raise RuntimeError("non-finite integrated total loss encountered")
        optimizer.zero_grad()
        losses["total_loss"].backward()
        optimizer.step()
        steps_completed += 1
        final_total_loss = float(losses["total_loss"].detach().item())
        last_outputs = outputs
        if steps_completed >= limits["max_steps"]:
            break

    with torch.no_grad():
        eval_outputs = model(x)
        class_pred = eval_outputs["class_logits"].argmax(dim=1)
        class_accuracy = float((class_pred == y_class).float().mean().item())
        schema_output = build_schema_output_summary(torch, eval_outputs, sample_index=0)

    assert losses is not None
    assert last_outputs is not None
    return {
        "marker": "PRE_SNS_INTEGRATED_SMOKE_OK",
        "samples_seen": int(x.shape[0]),
        "class_labels_seen": class_seen,
        "family_labels_seen": family_seen,
        "steps_completed": steps_completed,
        "epochs_completed": epochs_completed,
        "initial_total_loss": initial_total_loss,
        "final_total_loss": final_total_loss,
        "class_loss_finite": losses["class_loss_finite"],
        "family_loss_finite": losses["family_loss_finite"],
        "localization_loss_finite": losses["localization_loss_finite"],
        "total_loss_finite": losses["total_loss_finite"],
        "class_smoke_accuracy": class_accuracy,
        "loss_routing": losses["routing"],
        "schema_output": schema_output,
        "no_download": raw["no_download"],
        "no_network": raw["no_network"],
        "no_outputs": raw["no_outputs"],
        "no_checkpoints": raw["no_checkpoints"],
        "no_sns_augmentation": raw["no_sns_augmentation"],
        "result_scope": "tiny smoke only; not a final metric or performance claim",
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        return _fail("usage: run_pre_sns_integrated_smoke.py <approved-local-config.json>")
    try:
        raw = load_config(argv[0])
    except Exception as exc:
        return _fail(f"failed to read config: {exc}")
    errors = validate_config(raw)
    if errors:
        print("config validation failed before image or mask loading:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    if raw.get("config_kind") != "approved_local_pre_sns_integrated_smoke":
        return _fail("runner requires config_kind approved_local_pre_sns_integrated_smoke")
    try:
        summary = run_smoke(raw)
    except Exception as exc:
        return _fail(str(exc))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
