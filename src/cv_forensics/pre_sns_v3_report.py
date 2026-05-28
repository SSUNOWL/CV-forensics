"""Single-image report adapter for pre-SNS v3 checkpoints."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .pre_sns_v3_model import CLASS_LABELS, FAMILY_LABELS, build_pre_sns_v3_model
from .pre_sns_visualization import focused_mask_from_values, write_clean_red_overlay

MARKER = "PRE_SNS_V3_SINGLE_IMAGE_REPORT_OK"


def _runtime_deps():
    try:
        import torch
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("torch and PIL are required for pre-SNS v3 report") from exc
    return torch, Image


def _image_tensor(torch: Any, Image: Any, image_path: str, image_size: int, device: str):
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        raw = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        return raw.reshape(1, image_size, image_size, 3).permute(0, 3, 1, 2).float().div(255.0).to(device)


def load_v3_model(torch: Any, checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_name") != "pre_sns_v3":
        raise RuntimeError("checkpoint is not a pre_sns_v3 checkpoint")
    image_size = int(checkpoint.get("image_size", 224))
    model = build_pre_sns_v3_model(torch, int(checkpoint.get("base_channels", 12))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, image_size


def confidence_map(labels: tuple[str, ...] | list[str], probs: list[float]) -> dict[str, float]:
    return {str(label): float(probs[index]) for index, label in enumerate(labels)}


def run_v3_single_image_report(config: dict[str, Any]) -> dict[str, Any]:
    torch, Image = _runtime_deps()
    device = "cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    model, checkpoint, image_size = load_v3_model(torch, config["checkpoint_path"], device)
    image = _image_tensor(torch, Image, config["image_path"], image_size, device)
    started = time.perf_counter()
    with torch.no_grad():
        outputs = model(image)
        class_probs = torch.softmax(outputs["class_logits"][0], dim=0).detach().cpu().tolist()
        tamper_probs = torch.softmax(outputs["tamper_binary_logits"][0], dim=0).detach().cpu().tolist()
        family_probs = torch.softmax(outputs["family_logits"][0], dim=0).detach().cpu().tolist()
        mask_probs = torch.sigmoid(outputs["localization_logits"][0]).detach().cpu()
    latency_ms = max((time.perf_counter() - started) * 1000.0, 0.000001)
    class_label = CLASS_LABELS[int(max(range(len(class_probs)), key=lambda idx: class_probs[idx]))]
    family_label = FAMILY_LABELS[int(max(range(len(family_probs)), key=lambda idx: family_probs[idx]))]
    tau = float(config.get("threshold_tau", checkpoint.get("selected_tau", 0.5)))
    active_binary, mask_shape = focused_mask_from_values(
        mask_probs,
        mask_size=(image_size, image_size),
        keep_ratio=float(config.get("clean_overlay_keep_ratio", 0.1)),
        threshold_mode=str(config.get("clean_overlay_threshold_mode", "fixed_0_5")),
        component_mode=str(config.get("clean_overlay_component_mode", "all_components")),
    )
    mask_area_pct = float(sum(active_binary) / max(len(active_binary), 1) * 100.0)
    localization_head = "activated" if float(tamper_probs[1]) >= tau and sum(active_binary) > 0 else "skipped_below_threshold_or_empty"
    report = {
        "marker": MARKER,
        "class": class_label,
        "class_conf": confidence_map(CLASS_LABELS, class_probs),
        "family": family_label,
        "family_conf": confidence_map(FAMILY_LABELS, family_probs),
        "tampered_score": float(tamper_probs[1]),
        "threshold_tau": tau,
        "localization_head": localization_head,
        "mask_area_pct": mask_area_pct,
        "device": device,
        "latency_ms": float(latency_ms),
        "fps_estimate": float(1000.0 / latency_ms),
        "image_path": config["image_path"],
        "checkpoint_path": config["checkpoint_path"],
        "clean_red_overlay_written": False,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
    }
    if config.get("write_clean_red_overlay") is True and localization_head == "activated":
        paths = write_clean_red_overlay(
            config["image_path"],
            active_binary,
            config["report_root"],
            mask_size=mask_shape,
            threshold_mode="fixed_0_5",
            component_mode="all_components",
            keep_ratio=1.0,
            alpha=float(config.get("clean_overlay_alpha", 0.45)),
        )
        report.update(paths)
        report["mask_area_pct"] = float(paths["clean_overlay_area_pct"])
    if config.get("write_report") is True:
        root = Path(config["report_root"])
        root.mkdir(parents=True, exist_ok=True)
        path = root / "pre_sns_v3_single_image_report.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        report["report_path"] = str(path)
    return report

