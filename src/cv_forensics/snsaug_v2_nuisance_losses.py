"""Loss helpers for SNSAug V2 nuisance-mask fine-tuning."""

from __future__ import annotations

import math
from typing import Any

from .snsaug_v2_losses import (
    CLASS_TO_INDEX,
    class_cross_entropy_loss,
    clean_sns_non_tampered_class_consistency_loss,
    non_tampered_tampered_suppression_loss,
    valid_tamper_mask_loss,
)


def _torch_modules():
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        return None, None
    return torch, F


def sns_nuisance_mask_target(ignore_mask: Any, *, has_local_overlay: bool = True):
    """Use local SNS ignore-mask supervision for benign nuisance overlays."""

    torch, _F = _torch_modules()
    if torch is not None and hasattr(ignore_mask, "shape"):
        target = ignore_mask.float()
        return target if has_local_overlay else torch.zeros_like(target)
    if not has_local_overlay:
        return [[0.0 for _ in row] for row in ignore_mask]
    return ignore_mask


def degradation_label_from_profile(profile: str, metadata: dict[str, Any] | None = None) -> list[float]:
    """Convert profile/metadata to the 0064 multi-hot degradation vector."""

    text = " ".join([str(profile or ""), " ".join(str(v) for v in (metadata or {}).values())]).lower()
    labels = {
        "jpeg": ("jpeg" in text or "jpg" in text or "recompression" in text),
        "resize": ("resize" in text or "rescale" in text),
        "crop": ("crop" in text),
        "screenshot_resampling": ("screenshot" in text or "resampling" in text),
        "platform_layout": any(token in text for token in ("platform", "tiktok", "instagram", "youtube", "story", "shorts", "layout")),
        "overlay_text": any(token in text for token in ("text", "caption", "watermark", "banner")),
        "sticker": any(token in text for token in ("sticker", "emoji")),
        "news_meme": any(token in text for token in ("news", "meme")),
        "combined_sns": ("combined" in text or "sns" in text),
        "severity_light": ("light" in text or "mild" in text),
        "severity_medium": ("medium" in text or "moderate" in text),
    }
    from .snsaug_v2_nuisance_model import DEGRADATION_LABELS

    return [1.0 if labels[name] else 0.0 for name in DEGRADATION_LABELS]


def sns_nuisance_mask_loss(pred_mask: Any, target_mask: Any, reduction: str = "mean"):
    """BCE plus Dice-style loss for local nuisance mask supervision."""

    torch, F = _torch_modules()
    if torch is not None and hasattr(pred_mask, "shape"):
        pred = pred_mask.float().clamp(1e-6, 1.0 - 1e-6)
        target = target_mask.float()
        bce = F.binary_cross_entropy(pred, target, reduction="mean")
        inter = (pred * target).sum()
        denom = pred.sum() + target.sum()
        dice = 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6))
        total = bce + dice
        if reduction == "sum":
            return total * pred.numel()
        if reduction == "none":
            return F.binary_cross_entropy(pred, target, reduction="none")
        return total
    values = [(float(p), float(t)) for row_p, row_t in zip(pred_mask, target_mask) for p, t in zip(row_p, row_t)]
    losses = [-(t * math.log(max(min(p, 1.0 - 1e-6), 1e-6)) + (1.0 - t) * math.log(max(1.0 - p, 1e-6))) for p, t in values]
    return sum(losses) / max(1, len(losses))


def global_degradation_loss(logits: Any, targets: Any, reduction: str = "mean"):
    """Multi-label BCE for global SNS/degradation metadata."""

    torch, F = _torch_modules()
    if torch is not None and hasattr(logits, "shape"):
        return F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction=reduction)
    total = 0.0
    count = 0
    for row, target_row in zip(logits, targets):
        for logit, target in zip(row, target_row):
            prob = 1.0 / (1.0 + math.exp(-float(logit)))
            total += -(float(target) * math.log(max(prob, 1e-6)) + (1.0 - float(target)) * math.log(max(1.0 - prob, 1e-6)))
            count += 1
    return total / max(count, 1)


def non_tampered_mask_suppression_loss(pred_mask: Any, labels: Any, reduction: str = "mean"):
    """Penalize real/synthetic rows that produce large tamper masks."""

    torch, _F = _torch_modules()
    if torch is not None and hasattr(pred_mask, "shape"):
        target = labels if hasattr(labels, "shape") else torch.tensor([CLASS_TO_INDEX[str(label)] for label in labels], device=pred_mask.device)
        mask = (target.long() != CLASS_TO_INDEX["tampered"]).float().view(-1, 1, 1, 1)
        per_sample = pred_mask.float().mean(dim=(1, 2, 3)).pow(2) * mask.view(-1)
        if reduction == "sum":
            return per_sample.sum()
        if reduction == "none":
            return per_sample
        return per_sample.sum() / mask.view(-1).sum().clamp(min=1.0)
    losses = []
    label_values = labels if isinstance(labels, (list, tuple)) else [labels]
    for pred, label in zip(pred_mask, label_values):
        if str(label) == "tampered" or int(label) == CLASS_TO_INDEX["tampered"]:
            losses.append(0.0)
            continue
        flat = [float(value) for row in pred for value in (row if isinstance(row, (list, tuple)) else [row])]
        mean_area = sum(flat) / max(len(flat), 1)
        losses.append(mean_area * mean_area)
    return sum(losses) / max(len(losses), 1)


def mask_area_regularization_loss(pred_mask: Any, reduction: str = "mean"):
    """Discourage all-one tamper masks without preventing true tamper activation."""

    torch, _F = _torch_modules()
    if torch is not None and hasattr(pred_mask, "shape"):
        per_sample = pred_mask.float().mean(dim=(1, 2, 3)).pow(2)
        if reduction == "sum":
            return per_sample.sum()
        if reduction == "none":
            return per_sample
        return per_sample.mean()
    flat = [float(value) for row in pred_mask for value in (row if isinstance(row, (list, tuple)) else [row])]
    mean_area = sum(flat) / max(len(flat), 1)
    return mean_area * mean_area


def synthetic_preservation_loss(logits: Any, labels: Any, *, floor: float = 0.35, reduction: str = "mean"):
    """Penalize synthetic rows whose synthetic probability falls below a floor."""

    torch, _F = _torch_modules()
    if torch is not None and hasattr(logits, "shape"):
        target = labels if hasattr(labels, "shape") else torch.tensor([CLASS_TO_INDEX[str(label)] for label in labels], device=logits.device)
        mask = (target.long() == CLASS_TO_INDEX["synthetic"]).float()
        probs = logits.float().softmax(dim=-1)[:, CLASS_TO_INDEX["synthetic"]]
        loss = (float(floor) - probs).clamp(min=0.0).pow(2) * mask
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        return loss.sum() / mask.sum().clamp(min=1.0)
    losses = []
    label_values = labels if isinstance(labels, (list, tuple)) else [labels]
    for row, label in zip(logits, label_values):
        label_index = CLASS_TO_INDEX[str(label)] if isinstance(label, str) else int(label)
        if label_index != CLASS_TO_INDEX["synthetic"]:
            losses.append(0.0)
            continue
        max_value = max(float(value) for value in row)
        exps = [math.exp(float(value) - max_value) for value in row]
        total = sum(exps) or 1.0
        p_synthetic = exps[CLASS_TO_INDEX["synthetic"]] / total
        losses.append(max(0.0, float(floor) - p_synthetic) ** 2)
    denom = sum(1 for label in label_values if (CLASS_TO_INDEX[str(label)] if isinstance(label, str) else int(label)) == CLASS_TO_INDEX["synthetic"]) or 1
    return sum(losses) / denom


def total_nuisance_loss(outputs: dict[str, Any], batch: dict[str, Any], *, weights: dict[str, float]) -> dict[str, Any]:
    """Compute finite 0064 loss components."""

    class_loss = class_cross_entropy_loss(outputs["class_logits"], batch["class_targets"])
    tamper_prob = outputs["tamper_mask_logits"].sigmoid() if hasattr(outputs["tamper_mask_logits"], "sigmoid") else outputs["tamper_mask_logits"]
    sns_prob = outputs["sns_nuisance_mask_logits"].sigmoid() if hasattr(outputs["sns_nuisance_mask_logits"], "sigmoid") else outputs["sns_nuisance_mask_logits"]
    tamper_mask_loss_raw = valid_tamper_mask_loss(tamper_prob, batch["tamper_mask"], batch["ignore_mask"])
    is_tampered = (batch["class_targets"].long() == CLASS_TO_INDEX["tampered"]).float().mean() if hasattr(batch["class_targets"], "shape") else 1.0
    tamper_mask_loss = tamper_mask_loss_raw * is_tampered
    nuisance_loss = sns_nuisance_mask_loss(sns_prob, batch["sns_nuisance_mask"])
    degradation_loss = global_degradation_loss(outputs["global_degradation_logits"], batch["degradation_targets"])
    consistency_loss = clean_sns_non_tampered_class_consistency_loss(batch["clean_class_logits"], outputs["class_logits"], batch["class_targets"])
    hardneg_loss = non_tampered_tampered_suppression_loss(
        outputs["class_logits"],
        batch["class_targets"],
        ceiling=float(batch.get("p_tampered_ceiling", 0.05)),
    )
    synthetic_loss = synthetic_preservation_loss(
        outputs["class_logits"],
        batch["class_targets"],
        floor=float(batch.get("synthetic_probability_floor", 0.35)),
    )
    mask_suppression_loss = non_tampered_mask_suppression_loss(tamper_prob, batch["class_targets"])
    mask_area_loss = mask_area_regularization_loss(tamper_prob)
    total = (
        class_loss
        + float(weights.get("lambda_tamper_mask", 1.0)) * tamper_mask_loss
        + float(weights.get("lambda_sns_mask", 1.0)) * nuisance_loss
        + float(weights.get("lambda_degradation", 0.3)) * degradation_loss
        + float(weights.get("lambda_gating_consistency", 0.5)) * consistency_loss
        + float(weights.get("lambda_hardneg", 1.0)) * hardneg_loss
        + float(weights.get("lambda_synthetic_preservation", 2.0)) * synthetic_loss
        + float(weights.get("lambda_non_tampered_mask_suppression", 1.0)) * mask_suppression_loss
        + float(weights.get("lambda_mask_area_regularization", 0.1)) * mask_area_loss
    )
    return {
        "total_loss": total,
        "class_loss": class_loss,
        "tamper_mask_loss": tamper_mask_loss,
        "sns_nuisance_mask_loss": nuisance_loss,
        "global_degradation_loss": degradation_loss,
        "clean_sns_class_consistency_loss": consistency_loss,
        "hardneg_loss": hardneg_loss,
        "synthetic_preservation_loss": synthetic_loss,
        "non_tampered_mask_suppression_loss": mask_suppression_loss,
        "mask_area_regularization_loss": mask_area_loss,
    }


__all__ = [
    "degradation_label_from_profile",
    "global_degradation_loss",
    "mask_area_regularization_loss",
    "non_tampered_mask_suppression_loss",
    "sns_nuisance_mask_loss",
    "sns_nuisance_mask_target",
    "synthetic_preservation_loss",
    "total_nuisance_loss",
    "valid_tamper_mask_loss",
]
