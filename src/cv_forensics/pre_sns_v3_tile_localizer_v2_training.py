"""Guarded training for the high-resolution forensic tile localizer v2."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from .pre_sns_v3_tile_localizer_v2_model import build_pre_sns_v3_tile_localizer_v2

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER = "PRE_SNS_V3_TILE_LOCALIZER_V2_TRAINING_OK"
DRY_RUN_MARKER = "PRE_SNS_V3_TILE_LOCALIZER_V2_TRAINING_DRY_RUN_OK"
CONFIG_OK_MARKER = "PRE_SNS_V3_TILE_LOCALIZER_V2_TRAINING_CONFIG_OK"
APPROVED_KIND = "approved_pre_sns_v3_tile_localizer_v2_training"
APPROVED_MODE = "approved_local_pre_sns_v3_tile_localizer_v2_training"
APPROVAL_TEXT = "I_APPROVE_PRE_SNS_V3_TILE_LOCALIZER_V2_TRAINING"
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}


class TileLocalizerV2TrainingError(ValueError):
    """Raised when v2 tile localizer config or runtime guardrails fail."""


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


def _bad_path(text: str) -> bool:
    lower = text.lower().strip()
    return "://" in lower or any(part == ".." for part in _parts(text)) or any(part in PROTECTED_PARTS or part.startswith(".env") for part in _parts(text))


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TileLocalizerV2TrainingError("config root must be a JSON object")
    return raw


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, target)


def _validate_abs(value: Any, field: str, *, require_file: bool = False, require_parent: bool = False) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [_err(f"{field} must be a non-empty absolute path")]
    text = value.strip()
    errors: list[str] = []
    if not text.startswith("/"):
        errors.append(_err(f"{field} must be absolute"))
    if _bad_path(text):
        errors.append(_err(f"{field} must not contain remote, traversal, or protected path segments"))
    if text.startswith("/") and _inside_repo(text):
        errors.append(_err(f"{field} must be outside the repository"))
    if require_file and not os.path.isfile(text):
        errors.append(_err(f"{field} must exist as a file"))
    if require_parent:
        parent = Path(text).parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not parent.is_dir() or not os.access(parent, os.W_OK):
            errors.append(_err(f"{field} parent must exist and be writable"))
    return errors


def _positive_int(raw: dict[str, Any], field: str, errors: list[str], maximum: int = 1_000_000) -> None:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(_err(f"{field} must be a positive integer"))
    elif value > maximum:
        errors.append(_err(f"{field} must be <= {maximum}"))


def _nonnegative(raw: dict[str, Any], field: str, errors: list[str], maximum: float = 1000.0) -> None:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
        errors.append(_err(f"{field} must be a non-negative finite number"))
    elif float(value) > maximum:
        errors.append(_err(f"{field} must be <= {maximum}"))


def validate_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version", "config_kind", "execution_mode", "required_approval_text", "user_approval_text",
        "tile_manifest_path", "approved_input_roots", "approved_run_root", "approved_checkpoint_root",
        "device", "seed", "tile_size", "input_feature_mode", "epochs", "batch_size", "learning_rate",
        "max_tiles_train", "max_tiles_val", "gradient_accumulation_steps", "no_write_dry_run",
        "no_download", "no_network", "no_sns_augmentation",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    if raw.get("config_kind") != APPROVED_KIND:
        errors.append(_err(f"config_kind must be {APPROVED_KIND}"))
    if raw.get("execution_mode") != APPROVED_MODE:
        errors.append(_err(f"execution_mode must be {APPROVED_MODE}"))
    if raw.get("required_approval_text") != APPROVAL_TEXT or raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(_err(f"approval text must be {APPROVAL_TEXT}"))
    for flag in ("no_download", "no_network", "no_sns_augmentation"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if raw.get("no_write_dry_run") not in {True, False}:
        errors.append(_err("no_write_dry_run must be boolean"))
    if raw.get("device") not in {"cpu", "cuda"}:
        errors.append(_err("device must be cpu or cuda"))
    if raw.get("input_feature_mode") not in {"rgb_only", "rgb_residual", "rgb_edge_residual"}:
        errors.append(_err("input_feature_mode must be rgb_only, rgb_residual, or rgb_edge_residual"))
    roots = raw.get("approved_input_roots")
    if not isinstance(roots, list) or not roots or not all(isinstance(root, str) for root in roots):
        errors.append(_err("approved_input_roots must be a non-empty list"))
        roots = []
    for idx, root in enumerate(roots):
        errors.extend(_validate_abs(root, f"approved_input_roots[{idx}]"))
    errors.extend(_validate_abs(raw.get("tile_manifest_path"), "tile_manifest_path", require_file=require_exists))
    errors.extend(_validate_abs(raw.get("approved_run_root"), "approved_run_root", require_parent=require_exists))
    errors.extend(_validate_abs(raw.get("approved_checkpoint_root"), "approved_checkpoint_root", require_parent=require_exists))
    if isinstance(raw.get("tile_manifest_path"), str) and roots and not any(_is_under(raw["tile_manifest_path"], root) for root in roots):
        errors.append(_err("tile_manifest_path must be under approved_input_roots"))
    for field in ("approved_run_root", "approved_checkpoint_root"):
        value = raw.get(field)
        if isinstance(value, str) and value.startswith("/") and any(_is_under(value, root) for root in roots):
            errors.append(_err(f"{field} must not be under approved_input_roots"))
    for field, maximum in (
        ("seed", 2_147_483_647), ("tile_size", 4096), ("epochs", 1000), ("batch_size", 256),
        ("max_tiles_train", 1_000_000), ("max_tiles_val", 250_000), ("base_channels", 128),
        ("gradient_accumulation_steps", 1024),
    ):
        if field in raw or field != "base_channels":
            _positive_int(raw, field, errors, maximum)
    for field in (
        "learning_rate", "weight_decay", "gradient_clip_norm", "bce_loss_weight", "dice_loss_weight",
        "tversky_loss_weight", "boundary_loss_weight", "empty_mask_loss_weight",
        "false_activation_area_loss_weight", "focal_loss_weight",
    ):
        if field in raw or field == "learning_rate":
            _nonnegative(raw, field, errors)
    if raw.get("learning_rate") == 0:
        errors.append(_err("learning_rate must be > 0"))
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_config(config, require_exists)
    if errors:
        raise TileLocalizerV2TrainingError("v2 training config validation failed:\n" + "\n".join(errors))


def load_tile_records(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TileLocalizerV2TrainingError("tile manifest root must be a JSON object")
    records = raw.get("records", raw.get("tiles", raw.get("tile_records")))
    if not isinstance(records, list):
        raise TileLocalizerV2TrainingError("tile manifest must contain records, tiles, or tile_records")
    return [record for record in records if isinstance(record, dict)]


def expanded_crop_box(crop_box: list[int], image_size: tuple[int, int], tile_size: int) -> list[int]:
    width, height = image_size
    x1, y1, x2, y2 = [int(v) for v in crop_box]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    crop_w = min(int(tile_size), width)
    crop_h = min(int(tile_size), height)
    nx1 = max(0, min(width - crop_w, int(round(cx - crop_w / 2))))
    ny1 = max(0, min(height - crop_h, int(round(cy - crop_h / 2))))
    return [nx1, ny1, nx1 + crop_w, ny1 + crop_h]


def oversample_records(records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    factors = {
        "severe_iou_fail": int(config.get("severe_iou_oversample_factor", 1)),
        "low_iou": int(config.get("low_iou_oversample_factor", 1)),
        "weak_iou": int(config.get("weak_iou_oversample_factor", 1)),
        "hard_negative": int(config.get("hard_negative_oversample_factor", 1)),
        "negative": int(config.get("negative_oversample_factor", 1)),
    }
    out: list[dict[str, Any]] = []
    for record in records:
        tile_class = str(record.get("tile_class", ""))
        bucket = str(record.get("mining_bucket", ""))
        factor = 1
        if bucket in factors:
            factor = max(factor, factors[bucket])
        if tile_class == "hard_negative":
            factor = max(factor, factors["hard_negative"])
        if tile_class.startswith("negative"):
            factor = max(factor, factors["negative"])
        out.extend(dict(record) for _ in range(max(1, factor)))
    return out


def _runtime_deps():
    try:
        import torch
        from PIL import Image
    except Exception as exc:
        raise TileLocalizerV2TrainingError("torch and PIL are required for v2 training") from exc
    return torch, Image


class TileLocalizerV2Dataset:
    def __init__(self, records: list[dict[str, Any]], tile_size: int, torch: Any, Image: Any) -> None:
        self.records = records
        self.tile_size = int(tile_size)
        self.torch = torch
        self.Image = Image

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = record["source_image_path"]
        with self.Image.open(image_path) as image:
            image = image.convert("RGB")
            crop_box = expanded_crop_box(record["crop_box"], image.size, self.tile_size)
            crop = image.crop(tuple(crop_box))
            if crop.size != (self.tile_size, self.tile_size):
                crop = crop.resize((self.tile_size, self.tile_size), getattr(getattr(self.Image, "Resampling", self.Image), "BILINEAR"))
        raw = self.torch.frombuffer(bytearray(crop.tobytes()), dtype=self.torch.uint8)
        image_tensor = raw.reshape(self.tile_size, self.tile_size, 3).permute(2, 0, 1).float().div(255.0)
        if record.get("expected_mask_type") == "cropped_gt_mask" and record.get("source_mask_path"):
            with self.Image.open(record["source_mask_path"]) as mask:
                mask = mask.convert("L").crop(tuple(crop_box))
                if mask.size != (self.tile_size, self.tile_size):
                    mask = mask.resize((self.tile_size, self.tile_size), getattr(getattr(self.Image, "Resampling", self.Image), "NEAREST"))
            raw_mask = self.torch.frombuffer(bytearray(mask.tobytes()), dtype=self.torch.uint8)
            mask_tensor = (raw_mask.reshape(self.tile_size, self.tile_size).float().div(255.0) >= 0.5).float().unsqueeze(0)
            is_positive = 1.0
        else:
            mask_tensor = self.torch.zeros((1, self.tile_size, self.tile_size), dtype=self.torch.float32)
            is_positive = 0.0
        return {"image": image_tensor, "mask": mask_tensor, "is_positive": self.torch.tensor([is_positive]), "record": record}


def boundary_target(torch: Any, masks: Any) -> Any:
    nnf = torch.nn.functional
    pooled = nnf.max_pool2d(masks, 3, stride=1, padding=1)
    eroded = -nnf.max_pool2d(-masks, 3, stride=1, padding=1)
    return (pooled - eroded).clamp(0, 1)


def dice_loss(torch: Any, logits: Any, targets: Any, eps: float = 1e-6) -> Any:
    probs = torch.sigmoid(logits).reshape(logits.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def tversky_loss(torch: Any, logits: Any, targets: Any, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> Any:
    probs = torch.sigmoid(logits).reshape(logits.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    tp = (probs * targets).sum(dim=1)
    fp = (probs * (1 - targets)).sum(dim=1)
    fn = ((1 - probs) * targets).sum(dim=1)
    return (1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


def compute_losses(torch: Any, outputs: dict[str, Any], batch: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    logits = outputs["mask_logits"]
    masks = batch["masks"]
    bce = torch.nn.BCEWithLogitsLoss()(logits, masks)
    dloss = dice_loss(torch, logits, masks)
    tloss = tversky_loss(torch, logits, masks)
    neg = batch["is_positive"].reshape(-1) < 0.5
    empty_loss = torch.nn.BCEWithLogitsLoss()(logits[neg], torch.zeros_like(logits[neg])) if bool(neg.any()) else logits.sum() * 0.0
    false_area = torch.sigmoid(logits[neg]).mean() if bool(neg.any()) else logits.sum() * 0.0
    boundary = logits.sum() * 0.0
    if "boundary_logits" in outputs:
        boundary = torch.nn.BCEWithLogitsLoss()(outputs["boundary_logits"], boundary_target(torch, masks))
    total = (
        float(config.get("bce_loss_weight", 1.0)) * bce
        + float(config.get("dice_loss_weight", 2.0)) * dloss
        + float(config.get("tversky_loss_weight", 1.0)) * tloss
        + float(config.get("boundary_loss_weight", 0.5)) * boundary
        + float(config.get("empty_mask_loss_weight", 2.0)) * empty_loss
        + float(config.get("false_activation_area_loss_weight", 1.0)) * false_area
    )
    return {"total_loss": total, "bce_loss": bce, "dice_loss": dloss, "tversky_loss": tloss, "boundary_loss": boundary, "empty_mask_loss": empty_loss, "false_activation_area_loss": false_area}


def make_batch(torch: Any, dataset: TileLocalizerV2Dataset, indices: list[int], device: str) -> dict[str, Any]:
    items = [dataset[i] for i in indices]
    return {
        "images": torch.stack([item["image"] for item in items]).to(device),
        "masks": torch.stack([item["mask"] for item in items]).to(device),
        "is_positive": torch.stack([item["is_positive"] for item in items]).to(device),
        "records": [item["record"] for item in items],
    }


def _mask_metric(torch: Any, probs: Any, masks: Any, threshold: float) -> dict[str, float]:
    pred = (probs >= threshold).float()
    gt = (masks >= 0.5).float()
    inter = (pred * gt).reshape(pred.shape[0], -1).sum(dim=1)
    union = ((pred + gt) > 0).float().reshape(pred.shape[0], -1).sum(dim=1)
    ps = pred.reshape(pred.shape[0], -1).sum(dim=1)
    gs = gt.reshape(gt.shape[0], -1).sum(dim=1)
    iou = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(union))
    dice = torch.where(ps + gs > 0, 2 * inter / (ps + gs).clamp_min(1), torch.ones_like(inter))
    return {"iou": float(iou.mean().item()), "dice": float(dice.mean().item()), "area": float(pred.mean().item() * 100.0)}


def evaluate(torch: Any, model: Any, dataset: TileLocalizerV2Dataset, config: dict[str, Any], device: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    thresholds = [float(v) for v in config.get("threshold_values", [0.25, 0.5, 0.75])]
    rows: list[dict[str, Any]] = []
    sweep = {str(t): [] for t in thresholds}
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = make_batch(torch, dataset, [i], device)
            out = model(batch["images"])
            probs = torch.sigmoid(out["mask_logits"])
            pos = bool(float(batch["is_positive"].item()) >= 0.5)
            row = {"index": i, "is_positive": pos, "tile_class": batch["records"][0].get("tile_class"), "mining_bucket": batch["records"][0].get("mining_bucket")}
            for t in thresholds:
                metric = _mask_metric(torch, probs, batch["masks"], t)
                sweep[str(t)].append(metric)
            metric = _mask_metric(torch, probs, batch["masks"], 0.5)
            row.update({"iou": metric["iou"], "dice": metric["dice"], "pred_area_pct": metric["area"]})
            rows.append(row)
    best_t = thresholds[0]
    best_iou = -1.0
    sweep_summary: dict[str, Any] = {}
    for t in thresholds:
        vals = sweep[str(t)]
        mean_iou = sum(v["iou"] for v in vals) / len(vals) if vals else 0.0
        mean_dice = sum(v["dice"] for v in vals) / len(vals) if vals else 0.0
        sweep_summary[str(t)] = {"tile_mean_iou": mean_iou, "tile_mean_dice": mean_dice}
        if mean_iou > best_iou:
            best_t = t
            best_iou = mean_iou
    ious = [r["iou"] for r in rows]
    dices = [r["dice"] for r in rows]
    pos_rows = [r for r in rows if r["is_positive"]]
    neg_rows = [r for r in rows if not r["is_positive"]]
    bucket_metrics = failure_bucket_metrics(rows)
    metrics = {
        "tile_mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "tile_median_iou": sorted(ious)[len(ious) // 2] if ious else 0.0,
        "tile_mean_dice": sum(dices) / len(dices) if dices else 0.0,
        "positive_tile_iou": sum(r["iou"] for r in pos_rows) / len(pos_rows) if pos_rows else 0.0,
        "positive_tile_dice": sum(r["dice"] for r in pos_rows) / len(pos_rows) if pos_rows else 0.0,
        "negative_tile_false_activation_rate": sum(1 for r in neg_rows if r["pred_area_pct"] > float(config.get("negative_activation_area_pct", 0.1))) / len(neg_rows) if neg_rows else 0.0,
        "negative_tile_mean_mask_area_pct": sum(r["pred_area_pct"] for r in neg_rows) / len(neg_rows) if neg_rows else 0.0,
        "empty_mask_precision_proxy": 1.0 - (sum(1 for r in neg_rows if r["pred_area_pct"] > float(config.get("negative_activation_area_pct", 0.1))) / len(neg_rows) if neg_rows else 0.0),
        "boundary_iou": 0.0,
        "selected_mask_threshold": best_t,
        "threshold_sweep": sweep_summary,
        "severe_iou_bucket_metric": bucket_metrics.get("severe_iou_fail", {}),
        "low_iou_bucket_metric": bucket_metrics.get("low_iou", {}),
    }
    return metrics, rows, {"selected_mask_threshold": best_t, "threshold_sweep": sweep_summary}


def failure_bucket_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for bucket in ("severe_iou_fail", "low_iou", "weak_iou"):
        vals = [r for r in rows if r.get("mining_bucket") == bucket]
        if vals:
            out[bucket] = {"count": len(vals), "mean_iou": sum(r["iou"] for r in vals) / len(vals), "mean_dice": sum(r["dice"] for r in vals) / len(vals)}
    return out


def split_records(records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(int(config.get("seed", 46)))
    expanded = oversample_records(records, config)
    rng.shuffle(expanded)
    val_n = min(int(config["max_tiles_val"]), max(1, len(expanded) // 5)) if expanded else 0
    val = expanded[:val_n]
    train = expanded[val_n: val_n + int(config["max_tiles_train"])]
    if not train and expanded:
        train = expanded[: int(config["max_tiles_train"])]
    return train, val


def _safe_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    run_root = _real(config["approved_run_root"])
    ckpt_root = _real(config["approved_checkpoint_root"])
    if _inside_repo(run_root) or _inside_repo(ckpt_root):
        raise TileLocalizerV2TrainingError("run/checkpoint roots must be outside repository")
    run_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    return run_root, ckpt_root


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_exists=config.get("no_write_dry_run") is False)
    if config.get("no_write_dry_run") is True:
        return {"marker": DRY_RUN_MARKER, "training_started": False, "artifact_writes": False, "checkpoint_writes": False, "no_download": True, "no_network": True, "no_sns_augmentation": True}
    torch, Image = _runtime_deps()
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise TileLocalizerV2TrainingError("device=cuda requested but CUDA is unavailable")
    device = "cuda" if config["device"] == "cuda" else "cpu"
    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    train_records, val_records = split_records(load_tile_records(config["tile_manifest_path"]), config)
    if not train_records or not val_records:
        raise TileLocalizerV2TrainingError("training requires non-empty train and validation tiles")
    train_ds = TileLocalizerV2Dataset(train_records, int(config["tile_size"]), torch, Image)
    val_ds = TileLocalizerV2Dataset(val_records, int(config["tile_size"]), torch, Image)
    model = build_pre_sns_v3_tile_localizer_v2(torch, tile_size=int(config["tile_size"]), input_feature_mode=str(config["input_feature_mode"]), base_channels=int(config.get("base_channels", 8)), boundary_head=bool(config.get("boundary_head", True)), confidence_head=bool(config.get("confidence_head", True))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0)))
    use_amp = bool(config.get("mixed_precision", False)) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    batch_size = int(config["batch_size"])
    accum = int(config.get("gradient_accumulation_steps", 1))
    steps = 0
    train_metrics = []
    start = time.time()
    for epoch in range(int(config["epochs"])):
        order = list(range(len(train_ds)))
        random.Random(int(config["seed"]) + epoch).shuffle(order)
        losses: list[float] = []
        model.train()
        opt.zero_grad(set_to_none=True)
        for offset in range(0, len(order), batch_size):
            batch = make_batch(torch, train_ds, order[offset: offset + batch_size], device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss_map = compute_losses(torch, model(batch["images"]), batch, config)
                    loss = loss_map["total_loss"] / accum
                scaler.scale(loss).backward()
            else:
                loss_map = compute_losses(torch, model(batch["images"]), batch, config)
                loss = loss_map["total_loss"] / accum
                loss.backward()
            if (steps + 1) % accum == 0:
                if float(config.get("gradient_clip_norm", 0.0)) > 0:
                    if scaler is not None:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            steps += 1
            losses.append(float(loss_map["total_loss"].detach().cpu().item()))
        train_metrics.append({"epoch": epoch + 1, "mean_total_loss": sum(losses) / len(losses), "steps_completed": steps})
    val_metrics, rows, calibration = evaluate(torch, model, val_ds, config, device)
    run_root, ckpt_root = _safe_dirs(config)
    best = ckpt_root / "best_tile_localizer_v2.pt"
    latest = ckpt_root / "latest_tile_localizer_v2.pt"
    checkpoint = {"model_name": "pre_sns_v3_tile_localizer_v2", "model_state_dict": model.state_dict(), "config": config, "tile_size": int(config["tile_size"]), "base_channels": int(config.get("base_channels", 8)), "input_feature_mode": config["input_feature_mode"], "selected_mask_threshold": val_metrics["selected_mask_threshold"]}
    torch.save(checkpoint, best)
    torch.save(checkpoint, latest)
    summary = {"marker": MARKER, "training_started": True, "training_completed": True, "device": device, "epochs_completed": int(config["epochs"]), "steps_completed": steps, "train_tile_count": len(train_ds), "val_tile_count": len(val_ds), "val_metrics": val_metrics, "train_metrics": train_metrics, "elapsed_sec": time.time() - start, "no_download": True, "no_network": True, "no_sns_augmentation": True}
    write_json(run_root / "run_summary.json", summary)
    write_json(run_root / "val_metrics.json", val_metrics)
    write_json(run_root / "threshold_calibration.json", calibration)
    write_json(run_root / "failure_bucket_metrics.json", failure_bucket_metrics(rows))
    write_json(run_root / "config_snapshot.json", config)
    (run_root / "visual_samples").mkdir(parents=True, exist_ok=True)
    with open(run_root / "tile_metrics.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")
    artifact = {"marker": MARKER, "files": {"run_summary": str(run_root / "run_summary.json"), "val_metrics": str(run_root / "val_metrics.json"), "threshold_calibration": str(run_root / "threshold_calibration.json"), "tile_metrics": str(run_root / "tile_metrics.jsonl"), "failure_bucket_metrics": str(run_root / "failure_bucket_metrics.json"), "visual_samples": str(run_root / "visual_samples"), "config_snapshot": str(run_root / "config_snapshot.json"), "best_tile_localizer_v2": str(best), "latest_tile_localizer_v2": str(latest)}, "no_download": True, "no_network": True, "no_sns_augmentation": True}
    write_json(run_root / "artifact_manifest.json", artifact)
    summary.update({"artifact_manifest_path": str(run_root / "artifact_manifest.json"), "best_checkpoint_path": str(best), "latest_checkpoint_path": str(latest)})
    write_json(run_root / "run_summary.json", summary)
    return summary


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value
