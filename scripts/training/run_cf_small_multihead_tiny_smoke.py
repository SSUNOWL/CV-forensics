#!/usr/bin/env python3
"""Run a future approved CF-Small multi-head tiny training smoke."""

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

from scripts.agent.validate_cf_small_multihead_tiny_smoke_config import (  # noqa: E402
    CLASS_LABELS,
    FAMILY_LABELS,
    load_config,
    validate_config,
)


CLASS_TO_INDEX = {"real": 0, "synthetic": 1}
FAMILY_TO_INDEX = {label: index for index, label in enumerate(["LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A"])}


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


def _prepare_batch(raw: dict[str, Any], torch: Any, Image: Any) -> tuple[Any, Any, Any, list[str], list[str]]:
    limits = _run_limits(raw)
    samples = raw["sample_manifest"][: limits["max_samples"]]
    images = []
    class_targets = []
    family_targets = []
    class_seen: set[str] = set()
    family_seen: set[str] = set()
    image_size = limits["max_image_size"]

    for sample in samples:
        with Image.open(sample["image_path"]) as image:
            image = image.convert("RGB").resize((image_size, image_size))
            values = list(image.getdata())
        flat = []
        for pixel in values:
            flat.extend(channel / 255.0 for channel in pixel)
        images.append(flat)
        label = sample["label"]
        family = sample["family_label"]
        class_targets.append(CLASS_TO_INDEX[label])
        family_targets.append(FAMILY_TO_INDEX[family])
        class_seen.add(label)
        family_seen.add(family)

    x = torch.tensor(images, dtype=torch.float32)
    y_class = torch.tensor(class_targets, dtype=torch.long)
    y_family = torch.tensor(family_targets, dtype=torch.long)
    return x, y_class, y_family, sorted(class_seen), sorted(family_seen)


def _build_model(torch: Any, input_dim: int):
    class TinyMultiHeadModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 16),
                torch.nn.ReLU(),
            )
            self.class_head = torch.nn.Linear(16, len(CLASS_LABELS))
            self.family_head = torch.nn.Linear(16, len(FAMILY_LABELS))

        def forward(self, inputs):
            features = self.backbone(inputs)
            return self.class_head(features), self.family_head(features)

    return TinyMultiHeadModel()


def run_smoke(raw: dict[str, Any]) -> dict[str, Any]:
    torch, Image = _load_dependencies()
    torch.manual_seed(7)
    random.seed(7)

    if raw.get("tiny_limits", {}).get("cpu_only") is not True:
        raise RuntimeError("tiny smoke runner requires cpu_only true")

    limits = _run_limits(raw)
    x, y_class, y_family, class_seen, family_seen = _prepare_batch(raw, torch, Image)
    model = _build_model(torch, x.shape[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    class_loss_fn = torch.nn.CrossEntropyLoss()
    family_loss_fn = torch.nn.CrossEntropyLoss()
    family_loss_weight = float(raw.get("training_smoke_policy", {}).get("family_loss_weight", 1.0))

    initial_total_loss = None
    final_total_loss = None
    class_loss_finite = False
    family_loss_finite = False
    total_loss_finite = False
    steps_completed = 0
    epochs_completed = 0

    for epoch in range(limits["max_epochs"]):
        epochs_completed = epoch + 1
        class_logits, family_logits = model(x)
        class_loss = class_loss_fn(class_logits, y_class)
        family_loss = family_loss_fn(family_logits, y_family)
        total_loss = class_loss + family_loss_weight * family_loss
        if initial_total_loss is None:
            initial_total_loss = float(total_loss.detach().item())
        class_loss_finite = math.isfinite(float(class_loss.detach().item()))
        family_loss_finite = math.isfinite(float(family_loss.detach().item()))
        total_loss_finite = math.isfinite(float(total_loss.detach().item()))
        if not (class_loss_finite and family_loss_finite and total_loss_finite):
            raise RuntimeError("non-finite loss encountered before optimizer step")
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        steps_completed += 1
        final_total_loss = float(total_loss.detach().item())
        if steps_completed >= limits["max_steps"]:
            break

    with torch.no_grad():
        class_logits, family_logits = model(x)
        class_pred = class_logits.argmax(dim=1)
        family_pred = family_logits.argmax(dim=1)
        class_accuracy = float((class_pred == y_class).float().mean().item())
        family_accuracy = float((family_pred == y_family).float().mean().item())

    return {
        "marker": "CF_SMALL_MULTIHEAD_TINY_SMOKE_OK",
        "dataset_name": raw["dataset_name"],
        "samples_seen": int(x.shape[0]),
        "class_labels_seen": class_seen,
        "family_labels_seen": family_seen,
        "steps_completed": steps_completed,
        "epochs_completed": epochs_completed,
        "initial_total_loss": initial_total_loss,
        "final_total_loss": final_total_loss,
        "class_loss_finite": class_loss_finite,
        "family_loss_finite": family_loss_finite,
        "total_loss_finite": total_loss_finite,
        "class_smoke_accuracy": class_accuracy,
        "family_smoke_accuracy": family_accuracy,
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
        return _fail("usage: run_cf_small_multihead_tiny_smoke.py <approved-local-config.json>")
    try:
        raw = load_config(argv[0])
    except Exception as exc:
        return _fail(f"failed to read config: {exc}")
    errors = validate_config(raw)
    if errors:
        print("config validation failed before image loading:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    if raw.get("config_kind") != "approved_local_multihead_smoke":
        return _fail("runner requires config_kind approved_local_multihead_smoke")
    try:
        summary = run_smoke(raw)
    except Exception as exc:
        return _fail(str(exc))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
