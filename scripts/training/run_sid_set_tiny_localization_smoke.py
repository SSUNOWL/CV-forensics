#!/usr/bin/env python3
"""Run a future approved SID-Set tiny classification/localization smoke."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent.validate_sid_set_tiny_localization_smoke_config import (  # noqa: E402
    CLASS_LABELS,
    LABEL_ID_BY_CLASS,
    load_config,
    validate_config,
)


CLASS_TO_INDEX = LABEL_ID_BY_CLASS


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _load_dependencies():
    try:
        import torch
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on local optional packages.
        raise RuntimeError("torch and PIL are required for the manual tiny smoke runner") from exc
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


def _prepare_batch(raw: dict[str, Any], torch: Any, Image: Any) -> tuple[Any, Any, Any, Any, list[str], int, int]:
    limits = _run_limits(raw)
    samples = raw["sample_manifest"][: limits["max_samples"]]
    images = []
    class_targets = []
    mask_targets = []
    tampered_indices = []
    labels_seen: set[str] = set()
    image_size = limits["max_image_size"]

    for sample_index, sample in enumerate(samples):
        with Image.open(sample["image_path"]) as image:
            images.append(_image_to_flat_values(image, image_size))
        label = sample["label"]
        labels_seen.add(label)
        class_targets.append(CLASS_TO_INDEX[label])
        if label == "tampered":
            with Image.open(sample["mask_path"]) as mask:
                mask_targets.append(_mask_to_flat_values(mask, image_size))
            tampered_indices.append(sample_index)

    x = torch.tensor(images, dtype=torch.float32)
    y_class = torch.tensor(class_targets, dtype=torch.long)
    tampered_index_tensor = torch.tensor(tampered_indices, dtype=torch.long)
    if mask_targets:
        y_mask = torch.tensor(mask_targets, dtype=torch.float32)
    else:
        y_mask = torch.empty((0, image_size * image_size), dtype=torch.float32)
    return x, y_class, y_mask, tampered_index_tensor, sorted(labels_seen), len(tampered_indices), len(mask_targets)


def _build_model(torch: Any, input_dim: int, mask_dim: int):
    class TinySidLocalizationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 24),
                torch.nn.ReLU(),
            )
            self.class_head = torch.nn.Linear(24, len(CLASS_LABELS))
            self.localization_head = torch.nn.Linear(24, mask_dim)

        def forward(self, inputs):
            features = self.backbone(inputs)
            return self.class_head(features), self.localization_head(features)

    return TinySidLocalizationModel()


def _binary_iou(torch: Any, logits: Any, targets: Any) -> float:
    predicted = (torch.sigmoid(logits) >= 0.5).float()
    intersection = (predicted * targets).sum()
    union = ((predicted + targets) > 0).float().sum()
    if float(union.item()) == 0.0:
        return 1.0
    return float((intersection / union).item())


def run_smoke(raw: dict[str, Any]) -> dict[str, Any]:
    torch, Image = _load_dependencies()
    torch.manual_seed(11)
    random.seed(11)

    if raw.get("tiny_limits", {}).get("cpu_only") is not True:
        raise RuntimeError("tiny smoke runner requires cpu_only true")

    limits = _run_limits(raw)
    image_size = limits["max_image_size"]
    x, y_class, y_mask, tampered_indices, labels_seen, tampered_count, mask_count = _prepare_batch(raw, torch, Image)
    model = _build_model(torch, x.shape[1], image_size * image_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    class_loss_fn = torch.nn.CrossEntropyLoss()
    mask_loss_fn = torch.nn.BCEWithLogitsLoss()
    mask_loss_weight = float(raw.get("training_smoke_policy", {}).get("mask_loss_weight", 1.0))

    initial_total_loss = None
    final_total_loss = None
    class_loss_finite = False
    mask_loss_finite = False
    total_loss_finite = False
    localization_loss_applied_to_tampered_only = tampered_count == mask_count and tampered_count > 0
    steps_completed = 0
    epochs_completed = 0

    for epoch in range(limits["max_epochs"]):
        epochs_completed = epoch + 1
        class_logits, mask_logits = model(x)
        class_loss = class_loss_fn(class_logits, y_class)
        tampered_mask_logits = mask_logits.index_select(0, tampered_indices)
        mask_loss = mask_loss_fn(tampered_mask_logits, y_mask)
        total_loss = class_loss + mask_loss_weight * mask_loss
        if initial_total_loss is None:
            initial_total_loss = float(total_loss.detach().item())
        class_loss_finite = math.isfinite(float(class_loss.detach().item()))
        mask_loss_finite = math.isfinite(float(mask_loss.detach().item()))
        total_loss_finite = math.isfinite(float(total_loss.detach().item()))
        if not (class_loss_finite and mask_loss_finite and total_loss_finite):
            raise RuntimeError("non-finite loss encountered before optimizer step")
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        steps_completed += 1
        final_total_loss = float(total_loss.detach().item())
        if steps_completed >= limits["max_steps"]:
            break

    with torch.no_grad():
        class_logits, mask_logits = model(x)
        class_pred = class_logits.argmax(dim=1)
        class_accuracy = float((class_pred == y_class).float().mean().item())
        tampered_mask_logits = mask_logits.index_select(0, tampered_indices)
        localization_iou = _binary_iou(torch, tampered_mask_logits, y_mask)

    return {
        "marker": "SID_SET_TINY_LOCALIZATION_SMOKE_OK",
        "dataset_name": raw["dataset_name"],
        "samples_seen": int(x.shape[0]),
        "class_labels_seen": labels_seen,
        "tampered_samples_seen": tampered_count,
        "masks_seen": mask_count,
        "steps_completed": steps_completed,
        "epochs_completed": epochs_completed,
        "initial_total_loss": initial_total_loss,
        "final_total_loss": final_total_loss,
        "class_loss_finite": class_loss_finite,
        "mask_loss_finite": mask_loss_finite,
        "total_loss_finite": total_loss_finite,
        "class_smoke_accuracy": class_accuracy,
        "localization_smoke_iou": localization_iou,
        "localization_loss_applied_to_tampered_only": localization_loss_applied_to_tampered_only,
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
        return _fail("usage: run_sid_set_tiny_localization_smoke.py <approved-local-config.json>")
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
    if raw.get("config_kind") != "approved_local_sid_tiny_localization_smoke":
        return _fail("runner requires config_kind approved_local_sid_tiny_localization_smoke")
    try:
        summary = run_smoke(raw)
    except Exception as exc:
        return _fail(str(exc))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
