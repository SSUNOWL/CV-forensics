"""Loss helpers for SNSAug V2 smoke fine-tuning."""

from __future__ import annotations

import math
from typing import Any

CLASS_TO_INDEX = {"real": 0, "synthetic": 1, "tampered": 2}
TAMPERED_INDEX = 2


def _torch_modules():
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        return None, None
    return torch, F


def _flatten(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for child in value:
            out.extend(_flatten(child))
        return out
    return [float(value)]


def _rows(value: Any) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        return [[float(value)]]
    if value and isinstance(value[0], (list, tuple)):
        return [[float(item) for item in row] for row in value]
    return [[float(item) for item in value]]


def _label_index(value: Any) -> int:
    if isinstance(value, str):
        return CLASS_TO_INDEX[value]
    return int(value)


def _labels(values: Any) -> list[int]:
    if isinstance(values, (list, tuple)):
        return [_label_index(value) for value in values]
    return [_label_index(values)]


def _softmax(row: list[float]) -> list[float]:
    if not row:
        return []
    max_value = max(row)
    exps = [math.exp(value - max_value) for value in row]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def _prob_rows(logits_or_probs: Any) -> list[list[float]]:
    rows = _rows(logits_or_probs)
    out: list[list[float]] = []
    for row in rows:
        total = sum(row)
        if row and all(value >= 0.0 for value in row) and abs(total - 1.0) < 1e-6:
            out.append(row)
        else:
            out.append(_softmax(row))
    return out


def class_cross_entropy_loss(logits: Any, labels: Any, reduction: str = "mean"):
    """Cross entropy over real/synthetic/tampered."""

    torch, F = _torch_modules()
    if torch is not None and hasattr(logits, "shape"):
        target = labels if hasattr(labels, "shape") else torch.tensor(_labels(labels), device=logits.device)
        loss = F.cross_entropy(logits.float(), target.long(), reduction=reduction)
        return loss
    losses = []
    for row, label in zip(_prob_rows(logits), _labels(labels)):
        losses.append(-math.log(max(1e-6, min(1.0, row[label]))))
    if reduction == "sum":
        return sum(losses)
    if reduction == "none":
        return losses
    return sum(losses) / max(1, len(losses))


def valid_tamper_mask_loss(pred_mask: Any, tamper_mask: Any, ignore_mask: Any, reduction: str = "mean"):
    """BCE plus Dice-style loss on valid_region = 1 - ignore_mask."""

    torch, F = _torch_modules()
    if torch is not None and hasattr(pred_mask, "shape"):
        pred = pred_mask.float().clamp(1e-6, 1.0 - 1e-6)
        target = tamper_mask.float()
        valid = 1.0 - ignore_mask.float()
        bce = F.binary_cross_entropy(pred, target, reduction="none") * valid
        bce_value = bce.sum() / valid.sum().clamp(min=1.0)
        intersection = (pred * target * valid).sum()
        denom = (pred * valid).sum() + (target * valid).sum()
        dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (denom + 1e-6))
        total = bce_value + dice_loss
        if reduction == "sum":
            return total * valid.sum().clamp(min=1.0)
        if reduction == "none":
            return bce
        return total

    pred_values = [min(max(value, 1e-6), 1.0 - 1e-6) for value in _flatten(pred_mask)]
    target_values = _flatten(tamper_mask)
    valid_values = [1.0 - value for value in _flatten(ignore_mask)]
    bce_sum = 0.0
    valid_sum = 0.0
    intersection = 0.0
    pred_sum = 0.0
    target_sum = 0.0
    for pred, target, valid in zip(pred_values, target_values, valid_values):
        valid = max(0.0, min(1.0, valid))
        bce = -(target * math.log(pred) + (1.0 - target) * math.log(1.0 - pred))
        bce_sum += bce * valid
        valid_sum += valid
        intersection += pred * target * valid
        pred_sum += pred * valid
        target_sum += target * valid
    bce_value = bce_sum / (valid_sum or 1.0)
    dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6))
    total = bce_value + dice_loss
    if reduction == "sum":
        return total * (valid_sum or 1.0)
    if reduction == "none":
        return [0.0 if valid <= 0.0 else total for valid in valid_values]
    return total


def tampered_score_consistency_loss(
    clean_logits: Any,
    sns_logits: Any,
    labels: Any,
    *,
    floor: float = 0.50,
    reduction: str = "mean",
):
    """Penalize tampered SNS views whose p_tampered collapses below clean/floor target."""

    torch, F = _torch_modules()
    if torch is not None and hasattr(clean_logits, "shape"):
        target_labels = labels if hasattr(labels, "shape") else torch.tensor(_labels(labels), device=clean_logits.device)
        mask = (target_labels.long() == TAMPERED_INDEX).float()
        clean_prob = clean_logits.float().softmax(dim=-1)[:, TAMPERED_INDEX].detach()
        sns_prob = sns_logits.float().softmax(dim=-1)[:, TAMPERED_INDEX]
        target = torch.maximum(clean_prob, torch.full_like(clean_prob, float(floor)))
        loss = F.relu(target - sns_prob).pow(2) * mask
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        return loss.sum() / mask.sum().clamp(min=1.0)

    losses = []
    for clean_row, sns_row, label in zip(_prob_rows(clean_logits), _prob_rows(sns_logits), _labels(labels)):
        if label != TAMPERED_INDEX:
            losses.append(0.0)
            continue
        target = max(float(floor), clean_row[TAMPERED_INDEX])
        losses.append(max(0.0, target - sns_row[TAMPERED_INDEX]) ** 2)
    if reduction == "sum":
        return sum(losses)
    if reduction == "none":
        return losses
    denom = sum(1 for label in _labels(labels) if label == TAMPERED_INDEX) or 1
    return sum(losses) / denom


def clean_sns_class_consistency_loss(clean_logits: Any, sns_logits: Any, reduction: str = "mean"):
    """Symmetric KL over clean/SNS class distributions."""

    torch, F = _torch_modules()
    if torch is not None and hasattr(clean_logits, "shape"):
        clean_log = clean_logits.float().log_softmax(dim=-1)
        sns_log = sns_logits.float().log_softmax(dim=-1)
        clean_prob = clean_log.exp()
        sns_prob = sns_log.exp()
        loss = 0.5 * (
            F.kl_div(clean_log, sns_prob, reduction="none").sum(dim=-1)
            + F.kl_div(sns_log, clean_prob, reduction="none").sum(dim=-1)
        )
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        return loss.mean()

    losses = []
    for clean, sns in zip(_prob_rows(clean_logits), _prob_rows(sns_logits)):
        kl_left = sum(c * math.log(max(c, 1e-6) / max(s, 1e-6)) for c, s in zip(clean, sns))
        kl_right = sum(s * math.log(max(s, 1e-6) / max(c, 1e-6)) for c, s in zip(clean, sns))
        losses.append(0.5 * (kl_left + kl_right))
    if reduction == "sum":
        return sum(losses)
    if reduction == "none":
        return losses
    return sum(losses) / max(1, len(losses))


def hard_negative_tampered_loss(logits: Any, labels: Any, reduction: str = "mean"):
    """Penalize high p_tampered for real/synthetic SNSAug hard negatives."""

    torch, _F = _torch_modules()
    if torch is not None and hasattr(logits, "shape"):
        target_labels = labels if hasattr(labels, "shape") else torch.tensor(_labels(labels), device=logits.device)
        mask = (target_labels.long() != TAMPERED_INDEX).float()
        tampered_prob = logits.float().softmax(dim=-1)[:, TAMPERED_INDEX]
        loss = tampered_prob.pow(2) * mask
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        return loss.sum() / mask.sum().clamp(min=1.0)

    losses = []
    for row, label in zip(_prob_rows(logits), _labels(labels)):
        losses.append(0.0 if label == TAMPERED_INDEX else row[TAMPERED_INDEX] ** 2)
    if reduction == "sum":
        return sum(losses)
    if reduction == "none":
        return losses
    denom = sum(1 for label in _labels(labels) if label != TAMPERED_INDEX) or 1
    return sum(losses) / denom


def masked_family_loss(logits: Any, targets: Any, family_loss_mask: Any, reduction: str = "mean"):
    """Optional family CE applied only where family_loss_mask == 1."""

    torch, F = _torch_modules()
    if torch is not None and hasattr(logits, "shape"):
        target = targets if hasattr(targets, "shape") else torch.tensor([int(value) for value in targets], device=logits.device)
        mask = family_loss_mask.float() if hasattr(family_loss_mask, "shape") else torch.tensor(_flatten(family_loss_mask), device=logits.device).float()
        loss = F.cross_entropy(logits.float(), target.long(), reduction="none") * mask
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        return loss.sum() / mask.sum().clamp(min=1.0)

    losses = []
    masks = _flatten(family_loss_mask)
    for row, target, mask in zip(_prob_rows(logits), [int(value) for value in targets], masks):
        losses.append(0.0 if mask <= 0.0 else -math.log(max(1e-6, row[target])) * mask)
    if reduction == "sum":
        return sum(losses)
    if reduction == "none":
        return losses
    return sum(losses) / (sum(masks) or 1.0)


def compute_snsaug_v2_smoke_loss(
    *,
    class_logits: Any,
    labels: Any,
    pred_mask: Any,
    tamper_mask: Any,
    ignore_mask: Any,
    clean_logits: Any | None = None,
    sns_logits: Any | None = None,
    family_logits: Any | None = None,
    family_targets: Any | None = None,
    family_loss_mask: Any | None = None,
    lambda_mask: float = 1.0,
    lambda_score: float = 0.25,
    lambda_consistency: float = 0.10,
    lambda_hardneg: float = 0.20,
    lambda_family: float = 0.0,
    tampered_score_floor: float = 0.50,
) -> dict[str, Any]:
    class_loss = class_cross_entropy_loss(class_logits, labels)
    mask_loss = valid_tamper_mask_loss(pred_mask, tamper_mask, ignore_mask)
    score_loss = 0.0
    consistency_loss = 0.0
    if clean_logits is not None and sns_logits is not None:
        score_loss = tampered_score_consistency_loss(clean_logits, sns_logits, labels, floor=tampered_score_floor)
        consistency_loss = clean_sns_class_consistency_loss(clean_logits, sns_logits)
    hardneg_loss = hard_negative_tampered_loss(class_logits, labels)
    family_loss = 0.0
    if family_logits is not None and family_targets is not None and family_loss_mask is not None and float(lambda_family) != 0.0:
        family_loss = masked_family_loss(family_logits, family_targets, family_loss_mask)
    total = (
        class_loss
        + float(lambda_mask) * mask_loss
        + float(lambda_score) * score_loss
        + float(lambda_consistency) * consistency_loss
        + float(lambda_hardneg) * hardneg_loss
        + float(lambda_family) * family_loss
    )
    return {
        "total_loss": total,
        "class_loss": class_loss,
        "mask_loss": mask_loss,
        "tampered_score_consistency_loss": score_loss,
        "clean_sns_class_consistency_loss": consistency_loss,
        "hard_negative_loss": hardneg_loss,
        "family_loss": family_loss,
    }
