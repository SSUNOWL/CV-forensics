#!/usr/bin/env python3
"""Run a tiny approved local CF-Small binary training smoke.

This runner is for future manual execution with an approved local config. It
does not download data, recursively scan directories, write outputs, write
checkpoints, or use SNS augmentation.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "scripts" / "agent").is_dir()),
    None,
)
if _REPO_ROOT is not None:
    _AGENT_ROOT = str(_REPO_ROOT / "scripts" / "agent")
    if _AGENT_ROOT not in sys.path:
        sys.path.insert(0, _AGENT_ROOT)

from validate_cf_small_tiny_train_smoke_config import load_config  # noqa: E402


def _load_optional_deps():
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "torch and PIL are required for manual tiny training smoke execution. "
            "Do not install packages from this runner; use an environment where they are already available."
        ) from exc
    return torch, Image


def _image_to_tensor(path: str, size: int, torch: Any, Image: Any) -> Any:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size))
        pixels = list(img.getdata())
    values: List[float] = []
    for r, g, b in pixels:
        values.extend([r / 255.0, g / 255.0, b / 255.0])
    return torch.tensor(values, dtype=torch.float32)


def _prepare_batch(config: Dict[str, Any], torch: Any, Image: Any) -> Tuple[Any, Any, List[str]]:
    limits = config["tiny_limits"]
    max_samples = min(int(config.get("max_samples", limits["max_samples"])), int(limits["max_samples"]))
    image_size = int(limits["max_image_size"])
    label_to_index = {"real": 0, "synthetic": 1}
    xs = []
    ys = []
    labels_seen = []
    for entry in config["sample_manifest"][:max_samples]:
        xs.append(_image_to_tensor(entry["image_path"], image_size, torch, Image))
        ys.append(label_to_index[entry["label"]])
        labels_seen.append(entry["label"])
    if not xs:
        raise RuntimeError("sample_manifest must contain at least one sample.")
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long), sorted(set(labels_seen))


def run_smoke(config: Dict[str, Any]) -> Dict[str, Any]:
    if config.get("config_kind") != "approved_local_smoke":
        raise RuntimeError("Runner requires config_kind approved_local_smoke.")
    torch, Image = _load_optional_deps()
    torch.manual_seed(int(config.get("seed", 7)))
    random.seed(int(config.get("seed", 7)))
    x, y, classes_seen = _prepare_batch(config, torch, Image)
    limits = config["tiny_limits"]
    model = torch.nn.Linear(x.shape[1], 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config.get("learning_rate", 0.01)))
    criterion = torch.nn.CrossEntropyLoss()
    max_epochs = min(int(config.get("max_epochs", limits["max_epochs"])), int(limits["max_epochs"]))
    max_steps = min(int(config.get("max_steps", limits["max_steps"])), int(limits["max_steps"]))
    initial_loss = None
    final_loss = None
    steps = 0
    epochs_completed = 0
    for _epoch in range(max_epochs):
        epochs_completed += 1
        if steps >= max_steps:
            break
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss_value = float(loss.detach().cpu().item())
        if initial_loss is None:
            initial_loss = loss_value
        if not math.isfinite(loss_value):
            raise RuntimeError("Non-finite loss encountered during tiny smoke.")
        loss.backward()
        optimizer.step()
        final_loss = loss_value
        steps += 1
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
        accuracy = float((preds == y).float().mean().cpu().item())
    return {
        "marker": "CF_SMALL_TINY_TRAIN_SMOKE_OK",
        "dataset_name": config["dataset_name"],
        "samples_seen": int(x.shape[0]),
        "classes_seen": classes_seen,
        "steps_completed": steps,
        "epochs_completed": epochs_completed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "finite_loss": bool(final_loss is not None and math.isfinite(final_loss)),
        "smoke_accuracy": accuracy,
        "no_download": config["no_download"],
        "no_network": config["no_network"],
        "no_outputs": config["no_outputs"],
        "no_checkpoints": config["no_checkpoints"],
        "no_sns_augmentation": config["no_sns_augmentation"],
        "result_scope": "tiny smoke result only; not a final metric or performance claim"
    }


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <approved_cf_small_tiny_train_smoke.local.json>", file=sys.stderr)
        sys.exit(1)
    try:
        config = load_config(sys.argv[1])
        summary = run_smoke(config)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
