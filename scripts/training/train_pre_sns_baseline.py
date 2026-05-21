#!/usr/bin/env python3
"""Guarded pre-SNS baseline training entrypoint."""

from __future__ import annotations

import json
import math
import os
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
    CLASS_LABELS,
    CLASS_TO_INDEX,
    FAMILY_SMOKE_LABELS,
    FAMILY_TO_INDEX,
    build_tiny_integrated_model,
)
from cv_forensics.pre_sns_training_artifacts import write_training_artifacts  # noqa: E402
from scripts.agent.validate_pre_sns_baseline_train_config import (  # noqa: E402
    APPROVAL_TEXT,
    load_config,
    validate_config,
)


ENTRYPOINT_MARKER = "PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK"
DRY_RUN_MARKER = "PRE_SNS_BASELINE_TRAINING_DRY_RUN_OK"
RUN_MARKER = "PRE_SNS_BASELINE_TRAINING_RUN_OK"
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
    if not isinstance(manifest.get("samples"), list) and not isinstance(manifest.get("sample_manifest"), list):
        raise ValueError("unified manifest must contain samples or sample_manifest")
    return manifest


def _samples_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    selected = manifest.get("samples") if isinstance(manifest.get("samples"), list) else manifest.get("sample_manifest")
    samples = [sample for sample in selected if isinstance(sample, dict)]
    if not samples:
        raise ValueError("unified manifest contains no object samples")
    return samples


def _manifest_path_summary(samples: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "image_paths_listed": sum(1 for sample in samples if isinstance(sample.get("image_path"), str)),
        "mask_paths_listed": sum(1 for sample in samples if isinstance(sample.get("mask_path"), str)),
    }


def _select_device(raw: dict[str, Any], torch: Any) -> str:
    requested = raw.get("device", "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda was requested but CUDA is unavailable")
        return "cuda"
    return "cpu"


def _positive_int(raw: dict[str, Any], key: str, default: int) -> int:
    value = raw.get(key, default)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else default


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_label(sample: dict[str, Any]) -> str:
    label = sample.get("class_label", sample.get("label"))
    if label not in CLASS_TO_INDEX:
        raise ValueError(f"unsupported class label for sample {sample.get('sample_id', '<unknown>')}: {label}")
    return str(label)


def _synthetic_image_tensor(torch: Any, sample: dict[str, Any], image_size: int, device: str):
    label_offset = float(CLASS_TO_INDEX.get(str(sample.get("class_label")), 0)) / 10.0
    base = torch.linspace(0.0, 1.0, steps=3 * image_size * image_size, dtype=torch.float32)
    return (base.reshape(3, image_size, image_size) + label_offset).clamp(0.0, 1.0).to(device)


def _load_image_tensor(torch: Any, Image: Any, sample: dict[str, Any], image_size: int, device: str, no_write: bool):
    path = sample.get("image_path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"sample {sample.get('sample_id', '<unknown>')} missing explicit image_path")
    if not os.path.isfile(path):
        if no_write:
            return _synthetic_image_tensor(torch, sample, image_size, device)
        raise FileNotFoundError(f"explicit image_path does not exist as a file: {path}")
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        raw = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        tensor = raw.reshape(image_size, image_size, 3).permute(2, 0, 1).float().div(255.0)
    return tensor.to(device)


def _synthetic_mask_tensor(torch: Any, image_size: int, device: str):
    mask = torch.zeros((image_size, image_size), dtype=torch.float32)
    start = image_size // 4
    end = max(start + 1, image_size // 2)
    mask[start:end, start:end] = 1.0
    return mask.reshape(-1).to(device)


def _load_mask_tensor(torch: Any, Image: Any, sample: dict[str, Any], image_size: int, device: str, no_write: bool):
    if _safe_label(sample) != "tampered":
        return None
    path = sample.get("mask_path")
    if not isinstance(path, str) or not path:
        return None
    if not os.path.isfile(path):
        if no_write:
            return _synthetic_mask_tensor(torch, image_size, device)
        raise FileNotFoundError(f"explicit mask_path does not exist as a file: {path}")
    with Image.open(path) as image:
        image = image.convert("L").resize((image_size, image_size))
        raw = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        tensor = raw.reshape(image_size, image_size).float().div(255.0)
    return (tensor >= 0.5).float().reshape(-1).to(device)


def _batch_indices(count: int, batch_size: int) -> list[range]:
    return [range(start, min(start + batch_size, count)) for start in range(0, count, batch_size)]


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _compute_batch_losses(torch: Any, outputs: dict[str, Any], batch: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    class_loss = torch.nn.CrossEntropyLoss()(outputs["class_logits"], batch["class_targets"])
    family_indices = [index for index, target in enumerate(batch["family_targets"]) if target is not None]
    if family_indices:
        idx = torch.tensor(family_indices, dtype=torch.long, device=batch["class_targets"].device)
        targets = torch.tensor([int(batch["family_targets"][i]) for i in family_indices], dtype=torch.long, device=batch["class_targets"].device)
        family_loss = torch.nn.CrossEntropyLoss()(outputs["family_logits"].index_select(0, idx), targets)
    else:
        family_loss = outputs["family_logits"].sum() * 0.0
    localization_indices = batch["localization_indices"]
    if localization_indices:
        idx = torch.tensor(localization_indices, dtype=torch.long, device=batch["class_targets"].device)
        localization_loss = torch.nn.BCEWithLogitsLoss()(
            outputs["localization_logits"].index_select(0, idx),
            batch["mask_targets"],
        )
    else:
        localization_loss = outputs["localization_logits"].sum() * 0.0
    total_loss = (
        float(raw.get("class_loss_weight", 1.0)) * class_loss
        + float(raw.get("family_loss_weight", 1.0)) * family_loss
        + float(raw.get("localization_loss_weight", 1.0)) * localization_loss
    )
    return {
        "class_loss": class_loss,
        "family_loss": family_loss,
        "localization_loss": localization_loss,
        "total_loss": total_loss,
        "family_available": bool(family_indices),
        "localization_available": bool(localization_indices),
    }


def _build_batch(torch: Any, Image: Any, samples: list[dict[str, Any]], indices: range, image_size: int, device: str, no_write: bool) -> dict[str, Any]:
    image_tensors = []
    class_targets = []
    class_labels = []
    family_targets: list[int | None] = []
    mask_targets = []
    localization_indices: list[int] = []
    for batch_index, sample_index in enumerate(indices):
        sample = samples[sample_index]
        class_label = _safe_label(sample)
        image_tensors.append(_load_image_tensor(torch, Image, sample, image_size, device, no_write))
        class_targets.append(CLASS_TO_INDEX[class_label])
        class_labels.append(class_label)
        family_label = sample.get("family_label")
        family_targets.append(FAMILY_TO_INDEX[family_label] if family_label in FAMILY_TO_INDEX else None)
        mask_tensor = _load_mask_tensor(torch, Image, sample, image_size, device, no_write)
        if mask_tensor is not None:
            localization_indices.append(batch_index)
            mask_targets.append(mask_tensor)
    mask_dim = image_size * image_size
    if mask_targets:
        mask_batch = torch.stack(mask_targets, dim=0)
    else:
        mask_batch = torch.zeros((0, mask_dim), dtype=torch.float32, device=device)
    return {
        "inputs": torch.stack(image_tensors, dim=0).reshape(len(image_tensors), -1),
        "class_targets": torch.tensor(class_targets, dtype=torch.long, device=device),
        "class_labels": class_labels,
        "family_targets": family_targets,
        "localization_indices": localization_indices,
        "mask_targets": mask_batch,
        "masks_seen": len(mask_targets),
    }


def _run_training_loop(raw: dict[str, Any], manifest: dict[str, Any], no_write: bool) -> tuple[dict[str, Any], Any]:
    torch, Image = _load_optional_runtime_dependencies()
    seed = _positive_int(raw, "seed", 23)
    _seed_everything(torch, seed)
    device = _select_device(raw, torch)
    image_size = min(_positive_int(raw, "max_image_size", 64), 128)
    batch_size = _positive_int(raw, "batch_size", 2)
    epochs = _positive_int(raw, "epochs", 1)
    samples = _samples_from_manifest(manifest)
    max_samples = min(_positive_int(raw, "max_samples", len(samples)), len(samples))
    if max_samples < 1:
        raise ValueError("max_samples leaves no samples for training")
    pilot_samples = samples[:max_samples]
    model = build_tiny_integrated_model(torch, 3 * image_size * image_size, image_size * image_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(raw.get("learning_rate", 0.001) or 0.001))

    initial_total_loss: float | None = None
    final_total_loss: float | None = None
    last_class_loss = 0.0
    last_family_loss = 0.0
    last_localization_loss = 0.0
    family_available = False
    localization_available = False
    steps_completed = 0
    samples_seen = 0
    masks_seen = 0
    class_correct = 0
    family_correct = 0
    family_seen = 0
    iou_total = 0.0
    iou_seen = 0

    for _epoch in range(epochs):
        for indices in _batch_indices(len(pilot_samples), batch_size):
            batch = _build_batch(torch, Image, pilot_samples, indices, image_size, device, no_write)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["inputs"])
            losses = _compute_batch_losses(torch, outputs, batch, raw)
            total_loss = losses["total_loss"]
            if initial_total_loss is None:
                initial_total_loss = float(total_loss.detach().item())
            total_loss.backward()
            optimizer.step()
            final_total_loss = float(total_loss.detach().item())
            last_class_loss = float(losses["class_loss"].detach().item())
            last_family_loss = float(losses["family_loss"].detach().item())
            last_localization_loss = float(losses["localization_loss"].detach().item())
            family_available = family_available or losses["family_available"]
            localization_available = localization_available or losses["localization_available"]
            steps_completed += 1
            samples_seen += int(batch["class_targets"].shape[0])
            masks_seen += int(batch["masks_seen"])

            with torch.no_grad():
                class_pred = outputs["class_logits"].argmax(dim=1)
                class_correct += int((class_pred == batch["class_targets"]).sum().item())
                for idx, target in enumerate(batch["family_targets"]):
                    if target is not None:
                        family_seen += 1
                        family_correct += int(outputs["family_logits"][idx].argmax().item() == int(target))
                if batch["localization_indices"]:
                    selected_logits = outputs["localization_logits"].index_select(
                        0,
                        torch.tensor(batch["localization_indices"], dtype=torch.long, device=device),
                    )
                    predicted = (torch.sigmoid(selected_logits) >= 0.5).float()
                    target = batch["mask_targets"]
                    intersection = (predicted * target).sum(dim=1)
                    union = ((predicted + target) >= 1.0).float().sum(dim=1).clamp_min(1.0)
                    iou_total += float((intersection / union).sum().item())
                    iou_seen += int(target.shape[0])

    if initial_total_loss is None or final_total_loss is None:
        raise RuntimeError("training loop completed no steps")
    result = {
        "device": device,
        "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "epochs_completed": epochs,
        "steps_completed": steps_completed,
        "samples_seen": samples_seen,
        "masks_seen": masks_seen,
        "initial_total_loss": initial_total_loss,
        "final_total_loss": final_total_loss,
        "total_loss_finite": _finite(initial_total_loss) and _finite(final_total_loss),
        "class_loss_finite": _finite(last_class_loss),
        "family_loss_finite": True if not family_available else _finite(last_family_loss),
        "localization_loss_finite": True if not localization_available else _finite(last_localization_loss),
        "class_smoke_accuracy": float(class_correct / samples_seen) if samples_seen else 0.0,
        "manifest_sample_count": len(samples),
        "trained_sample_count": len(pilot_samples),
        **_manifest_path_summary(samples),
    }
    if family_seen:
        result["family_smoke_accuracy"] = float(family_correct / family_seen)
    if iou_seen:
        result["localization_smoke_iou"] = float(iou_total / iou_seen)
    return result, model


def _small_manifest_snapshot(manifest: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    snapshot = {key: value for key, value in manifest.items() if key not in {"samples", "sample_manifest"}}
    snapshot["sample_count"] = len(samples)
    snapshot["samples"] = samples[: min(25, len(samples))]
    return snapshot


def run_entrypoint(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("approved_training_run") is not True:
        return {
            "marker": ENTRYPOINT_MARKER,
            "training_started": False,
            "training_completed": False,
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
    training_metrics, model = _run_training_loop(raw, manifest, no_write)
    base = {
        "marker": DRY_RUN_MARKER if no_write else RUN_MARKER,
        "entrypoint_marker": ENTRYPOINT_MARKER,
        "training_started": True,
        "training_completed": True,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation": True,
        "no_write_dry_run": no_write,
        "approved_run_root": raw.get("approved_run_root"),
        "approved_checkpoint_root": raw.get("approved_checkpoint_root"),
        "result_scope": raw.get("result_scope", "pre-SNS baseline pilot training; not a final full-dataset performance claim"),
        **training_metrics,
    }
    if no_write:
        base["result_scope"] = "approved config no-write dry-run; no artifacts or checkpoints written"
        return base

    torch, _Image = _load_optional_runtime_dependencies()
    checkpoint_root = Path(raw["approved_checkpoint_root"])
    checkpoint_path = checkpoint_root / "pre_sns_baseline_pilot_checkpoint.pt"
    from cv_forensics.pre_sns_training_artifacts import prepare_artifact_roots

    overwrite = raw.get("overwrite_artifact_roots") is True
    run_dir, checkpoint_dir = prepare_artifact_roots(raw["approved_run_root"], raw["approved_checkpoint_root"], overwrite=overwrite)
    checkpoint_path = checkpoint_dir / checkpoint_path.name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_labels": list(CLASS_LABELS),
            "family_labels": list(FAMILY_SMOKE_LABELS),
            "image_size": min(_positive_int(raw, "max_image_size", 64), 128),
            "marker": RUN_MARKER,
        },
        checkpoint_path,
    )
    base["checkpoint_path"] = str(checkpoint_path)
    samples = _samples_from_manifest(manifest)
    artifact_info = write_training_artifacts(
        run_root=run_dir,
        checkpoint_root=checkpoint_dir,
        config_snapshot=raw,
        manifest_snapshot=_small_manifest_snapshot(manifest, samples),
        metrics_summary={key: value for key, value in base.items() if key.endswith("_loss") or key.endswith("_finite") or key.endswith("_accuracy") or key.endswith("_iou") or key in {"samples_seen", "masks_seen", "steps_completed", "epochs_completed"}},
        run_summary=base,
        checkpoint_path=checkpoint_path,
        overwrite=True,
    )
    base["artifact_manifest_path"] = artifact_info["artifact_manifest_path"]
    return base


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
    if summary.get("training_started") and not summary.get("no_write_dry_run"):
        print(STARTED_MARKER, file=sys.stderr)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
