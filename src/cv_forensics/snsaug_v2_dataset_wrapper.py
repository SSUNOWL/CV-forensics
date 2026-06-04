"""SNSAug V2 dataset wrapper and masked localization loss."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from .snsaug_v2 import SNSAugV2Augmentor, SNSAugV2Config
from .snsaug_v2_training_manifest import DEFAULT_CURRICULUM, DEFAULT_VIEW_RATIOS


def _load_pil():
    from PIL import Image

    return Image


def _safe_seed(seed: Any) -> int:
    if isinstance(seed, int):
        return seed
    text = str(seed or "")
    return sum((index + 1) * ord(ch) for index, ch in enumerate(text))


def _curriculum_stage(epoch: int, schedule: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    schedule = schedule or list(DEFAULT_CURRICULUM)
    for stage in schedule:
        start = int(stage.get("start_epoch", 1))
        end = stage.get("end_epoch")
        if epoch >= start and (end is None or epoch <= int(end)):
            return dict(stage)
    return dict(schedule[-1])


def _load_image_from_item(item: dict[str, Any]) -> Any:
    Image = _load_pil()
    if item.get("image") is not None:
        return item["image"].copy()
    with Image.open(str(item["image_path"])) as image:
        return image.convert("RGB")


def _load_mask_from_item(item: dict[str, Any]) -> Any | None:
    Image = _load_pil()
    if item.get("tamper_mask") is not None:
        return item["tamper_mask"].copy()
    path = item.get("tamper_mask_path") or item.get("mask_path")
    if not path:
        return None
    with Image.open(str(path)) as image:
        return image.convert("L")


def _blank_mask(size: tuple[int, int]) -> Any:
    Image = _load_pil()
    return Image.new("L", size, 0)


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 0:
        return next(iter(weights))
    target = rng.random() * total
    current = 0.0
    for key, value in weights.items():
        current += max(0.0, float(value))
        if target <= current:
            return key
    return next(reversed(weights))


class SNSAugV2DatasetWrapper:
    def __init__(
        self,
        base_dataset: Any,
        *,
        seed: int = 1,
        clean_weight: float = DEFAULT_VIEW_RATIOS["clean_weight"],
        basic_aug_weight: float = DEFAULT_VIEW_RATIOS["basic_aug_weight"],
        sns_aug_weight: float = DEFAULT_VIEW_RATIOS["sns_aug_weight"],
        curriculum: list[dict[str, Any]] | None = None,
        basic_profile: str = "jpeg_resize",
        sns_profiles: list[str] | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.seed = int(seed)
        self.clean_weight = float(clean_weight)
        self.basic_aug_weight = float(basic_aug_weight)
        self.sns_aug_weight = float(sns_aug_weight)
        self.curriculum = list(curriculum or DEFAULT_CURRICULUM)
        self.basic_profile = basic_profile
        self.sns_profiles = list(sns_profiles or ["combined_sns_realistic", "tiktok_like", "instagram_story_like", "youtube_shorts_like", "annotation_sticker"])
        self.epoch = 1

    def __len__(self) -> int:
        return len(self.base_dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(1, int(epoch))

    def _severity_for_view(self, view: str) -> str:
        stage = _curriculum_stage(self.epoch, self.curriculum)
        if view == "sns_aug":
            return str(stage.get("severity", "medium"))
        if view == "basic_aug":
            return "medium"
        return "light"

    def _view_weights(self) -> dict[str, float]:
        explicit = {
            "clean": self.clean_weight,
            "basic_aug": self.basic_aug_weight,
            "sns_aug": self.sns_aug_weight,
        }
        default = {
            "clean": DEFAULT_VIEW_RATIOS["clean_weight"],
            "basic_aug": DEFAULT_VIEW_RATIOS["basic_aug_weight"],
            "sns_aug": DEFAULT_VIEW_RATIOS["sns_aug_weight"],
        }
        if any(abs(explicit[key] - default[key]) > 1e-9 for key in explicit):
            return explicit
        stage = _curriculum_stage(self.epoch, self.curriculum)
        return {"clean": float(stage["clean"]), "basic_aug": float(stage["basic_aug"]), "sns_aug": float(stage["sns"])}

    def _item_seed(self, index: int, base_id: str) -> int:
        return self.seed + index * 1009 + _safe_seed(base_id) + self.epoch * 97

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base_dataset[index]
        if not isinstance(item, dict):
            raise TypeError("base dataset items must be dict-like")
        base_id = str(item.get("base_id") or item.get("sample_id") or item.get("id") or f"sample_{index:06d}")
        rng = random.Random(self._item_seed(index, base_id))
        view = _weighted_choice(rng, self._view_weights())
        label = str(item.get("content_label") or item.get("label") or item.get("class_label") or "")
        family_label = item.get("family_label")
        image = _load_image_from_item(item)
        tamper_mask = _load_mask_from_item(item)
        profile = "clean"
        if view == "basic_aug":
            profile = str(item.get("basic_profile") or self.basic_profile)
        elif view == "sns_aug":
            recommended = item.get("recommended_profiles")
            if isinstance(recommended, list) and recommended:
                profiles = [str(value) for value in recommended]
            else:
                profiles = self.sns_profiles
            non_clean = [value for value in profiles if value != "clean"]
            profile = non_clean[rng.randrange(len(non_clean))] if non_clean else "combined_sns_realistic"
        if view == "clean":
            ignore_mask = _blank_mask(image.size)
            aug_meta = {"profile": "clean", "severity": "light", "label_preserved": True, "seed": self._item_seed(index, base_id)}
            out_image = image
            out_mask = tamper_mask
        else:
            severity = self._severity_for_view(view)
            result = SNSAugV2Augmentor(SNSAugV2Config(profile=profile, severity=severity, seed=self._item_seed(index, base_id)))(image, tamper_mask, label=label, base_id=base_id, seed=self._item_seed(index, base_id))
            out_image = result.image
            out_mask = result.tamper_mask
            ignore_mask = result.ignore_mask
            aug_meta = result.meta
        return {
            "image": out_image,
            "label": label,
            "tamper_mask": out_mask if out_mask is not None else _blank_mask(out_image.size),
            "ignore_mask": ignore_mask,
            "view": view,
            "profile": profile,
            "aug_meta": aug_meta,
            "base_id": base_id,
            "family_label": family_label,
            "family_loss_mask": 1 if isinstance(family_label, str) and family_label.strip() else 0,
        }


def masked_localization_loss(pred_mask, tamper_mask, ignore_mask, reduction: str = "mean"):
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        torch = None
        F = None
    if torch is not None and hasattr(pred_mask, "shape"):
        pred = pred_mask.float()
        target = tamper_mask.float()
        ignore = ignore_mask.float()
        valid = 1.0 - ignore
        loss = F.binary_cross_entropy(pred, target, reduction="none") * valid
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        denom = valid.sum().clamp(min=1.0)
        return loss.sum() / denom

    def _flatten(value):
        if isinstance(value, (list, tuple)):
            out = []
            for child in value:
                if isinstance(child, (list, tuple)):
                    out.extend(_flatten(child))
                else:
                    out.append(float(child))
            return out
        return [float(value)]

    pred_values = _flatten(pred_mask)
    target_values = _flatten(tamper_mask)
    ignore_values = _flatten(ignore_mask)
    losses = []
    for pred, target, ignore in zip(pred_values, target_values, ignore_values):
        valid = 1.0 - float(ignore)
        pred = min(max(float(pred), 1e-6), 1.0 - 1e-6)
        bce = -(float(target) * math.log(pred) + (1.0 - float(target)) * math.log(1.0 - pred))
        losses.append(bce * valid)
    if reduction == "sum":
        return sum(losses)
    if reduction == "none":
        return losses
    valid_count = sum(1.0 - float(value) for value in ignore_values) or 1.0
    return sum(losses) / valid_count
