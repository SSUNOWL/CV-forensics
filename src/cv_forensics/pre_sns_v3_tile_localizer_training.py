"""Guarded pre-SNS v3 tile-localizer training."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from .pre_sns_v3_tile_localizer_model import build_pre_sns_v3_tile_localizer

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER = "PRE_SNS_V3_TILE_LOCALIZER_TRAINING_OK"
CONFIG_OK_MARKER = "PRE_SNS_V3_TILE_LOCALIZER_TRAINING_CONFIG_OK"
DRY_RUN_MARKER = "PRE_SNS_V3_TILE_LOCALIZER_TRAINING_DRY_RUN_OK"
APPROVED_KIND = "approved_pre_sns_v3_tile_localizer_training"
APPROVED_MODE = "approved_local_pre_sns_v3_tile_localizer_training"
APPROVAL_TEXT = "I_APPROVE_PRE_SNS_V3_TILE_LOCALIZER_TRAINING"
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}


class TileLocalizerTrainingError(ValueError):
    """Raised when tile-localizer training guardrails fail."""


def _err(message: str) -> str:
    return f"- {message}"


def _real(path: str | Path) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def _is_under(path: str | Path, root: str | Path) -> bool:
    try:
        return os.path.commonpath([str(_real(path)), str(_real(root))]) == str(_real(root))
    except ValueError:
        return False


def _inside_repo(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT)


def _parts(text: str) -> list[str]:
    return [part for part in text.replace("\\", "/").split("/") if part]


def _has_remote(text: str) -> bool:
    lower = text.lower().strip()
    return "://" in lower or lower.startswith(("http:", "https:", "s3:", "gs:", "hf:"))


def _has_traversal(text: str) -> bool:
    return any(part == ".." for part in _parts(text))


def _has_protected(text: str) -> bool:
    return any(part in PROTECTED_PARTS or part.startswith(".env") for part in _parts(text))


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TileLocalizerTrainingError("config root must be a JSON object")
    return raw


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, target)


def _validate_abs_path(value: Any, field: str, *, require_file: bool = False, require_dir_parent: bool = False) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [_err(f"{field} must be a non-empty absolute local path")]
    text = value.strip()
    errors: list[str] = []
    if not text.startswith("/"):
        errors.append(_err(f"{field} must be absolute"))
    if _has_remote(text):
        errors.append(_err(f"{field} must not be a URL or remote scheme"))
    if _has_traversal(text):
        errors.append(_err(f"{field} must not contain path traversal"))
    if _has_protected(text):
        errors.append(_err(f"{field} must not contain protected path segments"))
    if text.startswith("/") and _inside_repo(text):
        errors.append(_err(f"{field} must be outside the repository"))
    if require_file and not os.path.isfile(text):
        errors.append(_err(f"{field} must exist as a file"))
    if require_dir_parent:
        parent = Path(text).parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not parent.is_dir() or not os.access(parent, os.W_OK):
            errors.append(_err(f"{field} parent must exist and be writable"))
    return errors


def _positive_int(raw: dict[str, Any], field: str, errors: list[str], *, maximum: int = 1_000_000) -> None:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(_err(f"{field} must be a positive integer"))
    elif value > maximum:
        errors.append(_err(f"{field} must be <= {maximum}"))


def _nonnegative_number(raw: dict[str, Any], field: str, errors: list[str], *, maximum: float = 1_000.0) -> None:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
        errors.append(_err(f"{field} must be a non-negative finite number"))
    elif float(value) > maximum:
        errors.append(_err(f"{field} must be <= {maximum}"))


def validate_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "required_approval_text",
        "user_approval_text",
        "tile_manifest_path",
        "approved_input_roots",
        "approved_run_root",
        "approved_checkpoint_root",
        "device",
        "seed",
        "tile_size",
        "epochs",
        "batch_size",
        "learning_rate",
        "max_tiles_train",
        "max_tiles_val",
        "no_write_dry_run",
        "no_download",
        "no_network",
        "no_sns_augmentation",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    if raw.get("config_kind") != APPROVED_KIND:
        errors.append(_err(f"config_kind must be {APPROVED_KIND}"))
    if raw.get("execution_mode") != APPROVED_MODE:
        errors.append(_err(f"execution_mode must be {APPROVED_MODE}"))
    if raw.get("required_approval_text") != APPROVAL_TEXT or raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(_err("approval text must match I_APPROVE_PRE_SNS_V3_TILE_LOCALIZER_TRAINING"))
    for flag in ("no_download", "no_network", "no_sns_augmentation"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if raw.get("no_write_dry_run") not in {True, False}:
        errors.append(_err("no_write_dry_run must be boolean"))
    if raw.get("device") not in {"cpu", "cuda"}:
        errors.append(_err("device must be cpu or cuda"))
    roots = raw.get("approved_input_roots")
    if not isinstance(roots, list) or not roots or not all(isinstance(root, str) for root in roots):
        errors.append(_err("approved_input_roots must be a non-empty list of absolute paths"))
        roots = []
    for index, root in enumerate(roots):
        errors.extend(_validate_abs_path(root, f"approved_input_roots[{index}]"))
    errors.extend(_validate_abs_path(raw.get("tile_manifest_path"), "tile_manifest_path", require_file=require_exists))
    errors.extend(_validate_abs_path(raw.get("approved_run_root"), "approved_run_root", require_dir_parent=require_exists))
    errors.extend(_validate_abs_path(raw.get("approved_checkpoint_root"), "approved_checkpoint_root", require_dir_parent=require_exists))
    if isinstance(raw.get("tile_manifest_path"), str) and raw["tile_manifest_path"].startswith("/") and roots:
        if not any(_is_under(raw["tile_manifest_path"], root) for root in roots):
            errors.append(_err("tile_manifest_path must be under approved_input_roots"))
    for field in ("approved_run_root", "approved_checkpoint_root"):
        value = raw.get(field)
        if isinstance(value, str) and value.startswith("/"):
            if any(_is_under(value, root) for root in roots):
                errors.append(_err(f"{field} must not be under approved_input_roots"))
    for field, maximum in (("seed", 2_147_483_647), ("tile_size", 4096), ("epochs", 1000), ("batch_size", 512), ("max_tiles_train", 1_000_000), ("max_tiles_val", 250_000), ("base_channels", 128)):
        if field in raw or field != "base_channels":
            _positive_int(raw, field, errors, maximum=maximum)
    for field in ("learning_rate", "weight_decay", "gradient_clip_norm", "bce_loss_weight", "dice_loss_weight", "focal_loss_weight", "empty_mask_loss_weight"):
        if field in raw or field in {"learning_rate"}:
            _nonnegative_number(raw, field, errors)
    if raw.get("learning_rate") == 0:
        errors.append(_err("learning_rate must be > 0"))
    values = raw.get("threshold_values", [0.3, 0.5, 0.7])
    if not isinstance(values, list) or not values:
        errors.append(_err("threshold_values must be a non-empty list"))
    else:
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
                errors.append(_err(f"threshold_values[{index}] must be between 0 and 1"))
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_config(config, require_exists)
    if errors:
        raise TileLocalizerTrainingError("tile localizer training config validation failed:\n" + "\n".join(errors))


def load_tile_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("records"), list):
        raise TileLocalizerTrainingError("tile manifest must be a JSON object with records")
    return raw


def _runtime_deps():
    try:
        import torch
        from PIL import Image
    except Exception as exc:
        raise TileLocalizerTrainingError("torch and PIL are required for tile-localizer training") from exc
    return torch, Image


def _validate_record_paths(records: list[dict[str, Any]], roots: list[str]) -> None:
    for record in records:
        for field in ("source_image_path", "source_mask_path"):
            value = record.get(field)
            if field == "source_mask_path" and record.get("expected_mask_type") == "empty_mask":
                continue
            if not isinstance(value, str) or not value:
                if field == "source_image_path":
                    raise TileLocalizerTrainingError("tile record missing source_image_path")
                continue
            if _has_remote(value) or _has_traversal(value) or _has_protected(value):
                raise TileLocalizerTrainingError(f"{field} violates path policy")
            if not os.path.isabs(value):
                raise TileLocalizerTrainingError(f"{field} must be absolute")
            if not any(_is_under(value, root) for root in roots):
                raise TileLocalizerTrainingError(f"{field} must be under approved_input_roots")


def split_records(records: list[dict[str, Any]], seed: int, max_train: int, max_val: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean = [record for record in records if isinstance(record, dict)]
    rng = random.Random(seed)
    rng.shuffle(clean)
    val_count = min(max_val, max(1, len(clean) // 5)) if clean else 0
    val = clean[:val_count]
    train = clean[val_count: val_count + max_train]
    if not train and clean:
        train = clean[: min(max_train, len(clean))]
    return train, val[:max_val]


class TileLocalizationDataset:
    """Manifest-backed tile dataset with on-the-fly image and mask crops."""

    def __init__(self, records: list[dict[str, Any]], tile_size: int, torch: Any, Image: Any) -> None:
        self.records = records
        self.tile_size = int(tile_size)
        self.torch = torch
        self.Image = Image

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = self._crop_image(record["source_image_path"], record["crop_box"], mode="RGB")
        image_tensor = self._image_to_tensor(image)
        if record.get("expected_mask_type") == "cropped_gt_mask" and record.get("source_mask_path"):
            mask = self._crop_image(record["source_mask_path"], record["crop_box"], mode="L")
            mask_tensor = self._mask_to_tensor(mask)
            is_positive = 1.0
        else:
            mask_tensor = self.torch.zeros((1, self.tile_size, self.tile_size), dtype=self.torch.float32)
            is_positive = 0.0
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "is_positive": self.torch.tensor([is_positive], dtype=self.torch.float32),
            "record": record,
        }

    def _crop_image(self, path: str, crop_box: list[int], mode: str) -> Any:
        with self.Image.open(path) as image:
            image = image.convert(mode)
            x1, y1, x2, y2 = [int(v) for v in crop_box]
            crop = image.crop((x1, y1, x2, y2))
            if crop.size != (self.tile_size, self.tile_size):
                resample = getattr(getattr(self.Image, "Resampling", self.Image), "NEAREST" if mode == "L" else "BILINEAR")
                crop = crop.resize((self.tile_size, self.tile_size), resample)
            return crop

    def _image_to_tensor(self, image: Any) -> Any:
        raw = self.torch.ByteTensor(self.torch.ByteStorage.from_buffer(image.tobytes()))
        return raw.reshape(self.tile_size, self.tile_size, 3).permute(2, 0, 1).float().div(255.0)

    def _mask_to_tensor(self, image: Any) -> Any:
        raw = self.torch.ByteTensor(self.torch.ByteStorage.from_buffer(image.tobytes()))
        return (raw.reshape(self.tile_size, self.tile_size).float().div(255.0) >= 0.5).float().unsqueeze(0)


def make_batch(torch: Any, dataset: TileLocalizationDataset, indices: list[int], device: str) -> dict[str, Any]:
    items = [dataset[index] for index in indices]
    return {
        "images": torch.stack([item["image"] for item in items]).to(device),
        "masks": torch.stack([item["mask"] for item in items]).to(device),
        "is_positive": torch.stack([item["is_positive"] for item in items]).to(device),
        "records": [item["record"] for item in items],
    }


def dice_loss(torch: Any, logits: Any, targets: Any, eps: float = 1e-6) -> Any:
    probs = torch.sigmoid(logits).reshape(logits.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1).float()
    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - ((2.0 * intersection + eps) / (denom + eps))).mean()


def focal_loss(torch: Any, logits: Any, targets: Any, gamma: float = 2.0) -> Any:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    return ((1.0 - pt) ** gamma * bce).mean()


def compute_losses(torch: Any, outputs: dict[str, Any], batch: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    logits = outputs["mask_logits"]
    targets = batch["masks"]
    bce = torch.nn.BCEWithLogitsLoss()(logits, targets)
    dloss = dice_loss(torch, logits, targets)
    floss = focal_loss(torch, logits, targets)
    negative = batch["is_positive"].reshape(-1) < 0.5
    if bool(negative.any()):
        empty_targets = torch.zeros_like(logits[negative])
        empty_loss = torch.nn.BCEWithLogitsLoss()(logits[negative], empty_targets)
    else:
        empty_loss = logits.sum() * 0.0
    total = (
        float(config.get("bce_loss_weight", 1.0)) * bce
        + float(config.get("dice_loss_weight", 1.0)) * dloss
        + float(config.get("focal_loss_weight", 0.0)) * floss
        + float(config.get("empty_mask_loss_weight", 1.0)) * empty_loss
    )
    return {"total_loss": total, "bce_loss": bce, "dice_loss": dloss, "focal_loss": floss, "empty_mask_loss": empty_loss}


def _mask_metrics(torch: Any, probs: Any, targets: Any, threshold: float) -> dict[str, float]:
    pred = (probs >= threshold).float()
    gt = (targets >= 0.5).float()
    inter = (pred * gt).reshape(pred.shape[0], -1).sum(dim=1)
    union = ((pred + gt) > 0).float().reshape(pred.shape[0], -1).sum(dim=1)
    pred_sum = pred.reshape(pred.shape[0], -1).sum(dim=1)
    gt_sum = gt.reshape(gt.shape[0], -1).sum(dim=1)
    iou = torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(union))
    dice = torch.where((pred_sum + gt_sum) > 0, (2.0 * inter) / (pred_sum + gt_sum).clamp_min(1.0), torch.ones_like(inter))
    return {"iou": float(iou.mean().item()), "dice": float(dice.mean().item()), "pred_area_pct": float(pred.mean().item() * 100.0)}


def evaluate_model(torch: Any, model: Any, dataset: TileLocalizationDataset, config: dict[str, Any], device: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    thresholds = [float(value) for value in config.get("threshold_values", [0.3, 0.5, 0.7])]
    rows: list[dict[str, Any]] = []
    by_threshold = {value: [] for value in thresholds}
    positives: list[float] = []
    negative_active = 0
    negative_count = 0
    area_values: list[float] = []
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = make_batch(torch, dataset, [index], device)
            outputs = model(batch["images"])
            probs = torch.sigmoid(outputs["mask_logits"])
            target = batch["masks"]
            is_positive = bool(float(batch["is_positive"].item()) >= 0.5)
            row = {"index": index, "tile_class": batch["records"][0].get("tile_class"), "is_positive": is_positive}
            for threshold in thresholds:
                metrics = _mask_metrics(torch, probs, target, threshold)
                by_threshold[threshold].append(metrics)
                row[f"iou_at_{threshold:.2f}"] = metrics["iou"]
                row[f"dice_at_{threshold:.2f}"] = metrics["dice"]
            selected = _mask_metrics(torch, probs, target, 0.5)
            row.update({"iou": selected["iou"], "dice": selected["dice"], "pred_area_pct": selected["pred_area_pct"], "gt_area_pct": float(target.mean().item() * 100.0)})
            if is_positive:
                positives.append(row["iou"])
            else:
                negative_count += 1
                if row["pred_area_pct"] > float(config.get("negative_activation_area_pct", 0.1)):
                    negative_active += 1
            area_values.append(row["pred_area_pct"])
            rows.append(row)
    sweep = {}
    best_threshold = thresholds[0]
    best_score = -1.0
    for threshold, values in by_threshold.items():
        mean_iou = sum(item["iou"] for item in values) / len(values) if values else 0.0
        mean_dice = sum(item["dice"] for item in values) / len(values) if values else 0.0
        sweep[str(threshold)] = {"tile_mean_iou": mean_iou, "tile_mean_dice": mean_dice}
        if mean_iou > best_score:
            best_threshold = threshold
            best_score = mean_iou
    ious = [row["iou"] for row in rows]
    dices = [row["dice"] for row in rows]
    metrics = {
        "tile_mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "tile_median_iou": sorted(ious)[len(ious) // 2] if ious else 0.0,
        "tile_mean_dice": sum(dices) / len(dices) if dices else 0.0,
        "positive_tile_iou": sum(positives) / len(positives) if positives else 0.0,
        "negative_tile_false_activation_rate": float(negative_active / negative_count) if negative_count else 0.0,
        "empty_mask_precision_proxy": float(1.0 - (negative_active / negative_count)) if negative_count else 1.0,
        "mask_area_pct_summary": {
            "mean": sum(area_values) / len(area_values) if area_values else 0.0,
            "max": max(area_values) if area_values else 0.0,
            "min": min(area_values) if area_values else 0.0,
        },
        "selected_mask_threshold": best_threshold,
        "threshold_sweep": sweep,
    }
    calibration = {"selected_mask_threshold": best_threshold, "threshold_sweep": sweep}
    return metrics, rows, calibration


def _safe_output_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    run_root = _real(config["approved_run_root"])
    ckpt_root = _real(config["approved_checkpoint_root"])
    if _inside_repo(run_root) or _inside_repo(ckpt_root):
        raise TileLocalizerTrainingError("run/checkpoint roots must be outside repository")
    if _has_protected(str(run_root)) or _has_protected(str(ckpt_root)):
        raise TileLocalizerTrainingError("run/checkpoint roots must not contain protected path segments")
    run_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    return run_root, ckpt_root


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_exists=config.get("no_write_dry_run") is False)
    if config.get("no_write_dry_run") is True:
        return {
            "marker": DRY_RUN_MARKER,
            "training_started": False,
            "artifact_writes": False,
            "checkpoint_writes": False,
            "no_write_dry_run": True,
            "no_download": True,
            "no_network": True,
            "no_sns_augmentation": True,
        }
    torch, Image = _runtime_deps()
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise TileLocalizerTrainingError("device=cuda requested but CUDA is unavailable")
    device = "cuda" if config["device"] == "cuda" else "cpu"
    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    manifest = load_tile_manifest(config["tile_manifest_path"])
    records = [record for record in manifest["records"] if isinstance(record, dict)]
    _validate_record_paths(records, list(config["approved_input_roots"]))
    train_records, val_records = split_records(records, int(config["seed"]), int(config["max_tiles_train"]), int(config["max_tiles_val"]))
    if not train_records or not val_records:
        raise TileLocalizerTrainingError("tile manifest must provide train and validation records")
    train_dataset = TileLocalizationDataset(train_records, int(config["tile_size"]), torch, Image)
    val_dataset = TileLocalizationDataset(val_records, int(config["tile_size"]), torch, Image)
    model = build_pre_sns_v3_tile_localizer(torch, int(config.get("base_channels", 8))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0)))
    batch_size = int(config["batch_size"])
    steps = 0
    train_metrics: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(int(config["epochs"])):
        order = list(range(len(train_dataset)))
        random.Random(int(config["seed"]) + epoch).shuffle(order)
        losses: list[float] = []
        model.train()
        for offset in range(0, len(order), batch_size):
            batch_indices = order[offset: offset + batch_size]
            batch = make_batch(torch, train_dataset, batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["images"])
            loss_map = compute_losses(torch, outputs, batch, config)
            loss_map["total_loss"].backward()
            if float(config.get("gradient_clip_norm", 0.0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
            steps += 1
            losses.append(float(loss_map["total_loss"].detach().cpu().item()))
        train_metrics.append({"epoch": epoch + 1, "mean_total_loss": sum(losses) / len(losses), "steps_completed": steps})
    val_metrics, tile_rows, calibration = evaluate_model(torch, model, val_dataset, config, device)
    run_root, ckpt_root = _safe_output_dirs(config)
    best_path = ckpt_root / "best_tile_localizer.pt"
    latest_path = ckpt_root / "latest_tile_localizer.pt"
    checkpoint = {
        "model_name": "pre_sns_v3_tile_localizer",
        "model_state_dict": model.state_dict(),
        "config": config,
        "tile_size": int(config["tile_size"]),
        "base_channels": int(config.get("base_channels", 8)),
        "selected_mask_threshold": val_metrics["selected_mask_threshold"],
    }
    torch.save(checkpoint, best_path)
    torch.save(checkpoint, latest_path)
    summary = {
        "marker": MARKER,
        "training_started": True,
        "training_completed": True,
        "device": device,
        "epochs_completed": int(config["epochs"]),
        "steps_completed": steps,
        "train_tile_count": len(train_dataset),
        "val_tile_count": len(val_dataset),
        "selected_mask_threshold": val_metrics["selected_mask_threshold"],
        "val_metrics": val_metrics,
        "train_metrics": train_metrics,
        "elapsed_sec": time.time() - start,
        "no_write_dry_run": False,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation": True,
    }
    write_json(run_root / "run_summary.json", summary)
    write_json(run_root / "val_metrics.json", val_metrics)
    write_json(run_root / "threshold_calibration.json", calibration)
    write_json(run_root / "config_snapshot.json", config)
    with open(run_root / "tile_metrics.jsonl", "w", encoding="utf-8") as handle:
        for row in tile_rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")
    artifact = {
        "marker": MARKER,
        "artifact_kind": "pre_sns_v3_tile_localizer_training",
        "files": {
            "run_summary": str(run_root / "run_summary.json"),
            "val_metrics": str(run_root / "val_metrics.json"),
            "threshold_calibration": str(run_root / "threshold_calibration.json"),
            "tile_metrics": str(run_root / "tile_metrics.jsonl"),
            "config_snapshot": str(run_root / "config_snapshot.json"),
            "artifact_manifest": str(run_root / "artifact_manifest.json"),
            "best_tile_localizer": str(best_path),
            "latest_tile_localizer": str(latest_path),
        },
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation": True,
    }
    write_json(run_root / "artifact_manifest.json", artifact)
    summary.update({
        "artifact_manifest_path": str(run_root / "artifact_manifest.json"),
        "run_summary_path": str(run_root / "run_summary.json"),
        "val_metrics_path": str(run_root / "val_metrics.json"),
        "threshold_calibration_path": str(run_root / "threshold_calibration.json"),
        "tile_metrics_path": str(run_root / "tile_metrics.jsonl"),
        "best_checkpoint_path": str(best_path),
        "latest_checkpoint_path": str(latest_path),
    })
    write_json(run_root / "run_summary.json", summary)
    return summary


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value

