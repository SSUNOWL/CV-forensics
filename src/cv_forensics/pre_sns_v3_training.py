"""Guarded pre-SNS v3 training pipeline."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from .pre_sns_v3_metrics import DEFAULT_TAU_VALUES, compute_metrics
from .pre_sns_v3_model import CLASS_LABELS, FAMILY_LABELS, MEANINGFUL_FAMILY_LABELS, build_pre_sns_v3_model

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_OK_MARKER = "PRE_SNS_V3_TRAINING_CONFIG_OK"
RUN_OK_MARKER = "PRE_SNS_V3_TRAINING_RUN_OK"
DRY_RUN_OK_MARKER = "PRE_SNS_V3_TRAINING_DRY_RUN_OK"
APPROVAL_TEXT = "I_APPROVE_PRE_SNS_V3_TRAINING"
APPROVED_KIND = "approved_pre_sns_v3_training"
APPROVED_MODE = "approved_local_pre_sns_v3_training"
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}
SECRET_WORDS = ("api_key", "apikey", "secret", "token", "password", "credential", "private_key")
REMOTE_PREFIXES = ("http://", "https://", "s3://", "gs://", "hf://")


class PreSnsV3TrainingError(ValueError):
    """Raised when v3 training config or runtime guardrails fail."""


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, target)


def load_config(path: str | Path) -> dict[str, Any]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise PreSnsV3TrainingError("config root must be a JSON object")
    return raw


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


def _path_parts(value: str) -> list[str]:
    return [part for part in value.replace("\\", "/").split("/") if part]


def is_protected_path(value: str) -> bool:
    parts = _path_parts(value)
    return any(part in PROTECTED_PARTS or part.startswith(".env") for part in parts)


def is_secret_like(value: str) -> bool:
    lower = value.lower()
    return any(word in lower for word in SECRET_WORDS)


def _has_remote_scheme(value: str) -> bool:
    lower = value.lower().strip()
    return lower.startswith(REMOTE_PREFIXES) or "://" in lower


def _has_traversal(value: str) -> bool:
    return any(part == ".." for part in _path_parts(value))


def _positive_int(raw: dict[str, Any], key: str, max_value: int, errors: list[str]) -> None:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(_err(f"{key} must be a positive integer"))
    elif value > max_value:
        errors.append(_err(f"{key} must be <= {max_value}"))


def _nonnegative_float(raw: dict[str, Any], key: str, max_value: float, errors: list[str]) -> None:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0:
        errors.append(_err(f"{key} must be a non-negative number"))
    elif float(value) > max_value:
        errors.append(_err(f"{key} must be <= {max_value}"))


def _validate_path_text(value: Any, field: str, approved: bool, require_file: bool = False, output_root: bool = False) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [_err(f"{field} must be a non-empty path")]
    text = value.strip()
    errors: list[str] = []
    if _has_remote_scheme(text):
        errors.append(_err(f"{field} must not be a URL or remote scheme"))
    if _has_traversal(text):
        errors.append(_err(f"{field} must not contain path traversal"))
    if is_secret_like(text):
        errors.append(_err(f"{field} must not contain secret-like text"))
    if is_protected_path(text):
        errors.append(_err(f"{field} must not reference protected paths"))
    if approved:
        if not os.path.isabs(text):
            errors.append(_err(f"{field} must be absolute in approved local mode"))
        if output_root and os.path.isabs(text) and _inside_repo(text):
            errors.append(_err(f"{field} must be outside the repository"))
        if require_file and not os.path.isfile(text):
            errors.append(_err(f"{field} must exist as a file"))
    elif os.path.isabs(text):
        errors.append(_err(f"{field} must be symbolic in example mode"))
    return errors


def _walk_string_safety(node: Any, path: str = "", allowed_absolute: set[str] | None = None) -> list[str]:
    allowed_absolute = allowed_absolute or set()
    errors: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else str(key)
            if is_secret_like(str(key)):
                errors.append(_err(f"{child_path}: secret-like key rejected"))
            errors.extend(_walk_string_safety(value, child_path, allowed_absolute))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            errors.extend(_walk_string_safety(value, f"{path}[{index}]", allowed_absolute))
    elif isinstance(node, str):
        text = node.strip()
        if not text:
            return errors
        if text.startswith("/") and text not in allowed_absolute:
            errors.append(_err(f"{path}: absolute path rejected unless explicitly approved"))
        if _has_remote_scheme(text):
            errors.append(_err(f"{path}: URL or remote scheme rejected"))
        if _has_traversal(text):
            errors.append(_err(f"{path}: path traversal rejected"))
        if is_secret_like(text):
            errors.append(_err(f"{path}: secret-like value rejected"))
        if is_protected_path(text):
            errors.append(_err(f"{path}: protected path segment rejected"))
    return errors


def validate_config(raw: dict[str, Any], require_manifest_exists: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(raw, dict):
        return [_err("config root must be a JSON object")]
    required = (
        "schema_version", "config_kind", "execution_mode", "required_approval_text", "user_approval_text",
        "train_manifest_path", "val_manifest_path", "approved_input_roots", "approved_run_root",
        "approved_checkpoint_root", "device", "seed", "max_samples_train", "max_samples_val",
        "max_image_size", "batch_size", "epochs", "learning_rate", "no_write_dry_run",
        "no_download", "no_network", "no_sns_augmentation", "class_labels", "family_labels",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    approved = raw.get("config_kind") == APPROVED_KIND
    if raw.get("config_kind") not in {"example_symbolic", APPROVED_KIND}:
        errors.append(_err("config_kind must be example_symbolic or approved_pre_sns_v3_training"))
    if raw.get("execution_mode") != (APPROVED_MODE if approved else "example_only"):
        errors.append(_err(f"execution_mode must be {APPROVED_MODE if approved else 'example_only'}"))
    if raw.get("required_approval_text") != APPROVAL_TEXT:
        errors.append(_err("required_approval_text must document the approval phrase"))
    if approved and raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(_err("user_approval_text must match required approval phrase"))
    if not approved and raw.get("user_approval_text") not in {"", None}:
        errors.append(_err("example user_approval_text must be empty"))
    for flag in ("no_download", "no_network", "no_sns_augmentation"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if raw.get("no_write_dry_run") not in {True, False}:
        errors.append(_err("no_write_dry_run must be boolean"))
    if raw.get("device") not in {"cpu", "cuda"}:
        errors.append(_err("device must be cpu or cuda"))
    if raw.get("class_labels") != list(CLASS_LABELS):
        errors.append(_err(f"class_labels must be {list(CLASS_LABELS)}"))
    if raw.get("family_labels") != list(FAMILY_LABELS):
        errors.append(_err(f"family_labels must be {list(FAMILY_LABELS)}"))
    _positive_int(raw, "seed", 2_147_483_647, errors)
    _positive_int(raw, "max_samples_train", 200_000, errors)
    _positive_int(raw, "max_samples_val", 50_000, errors)
    _positive_int(raw, "batch_size", 256, errors)
    _positive_int(raw, "epochs", 100, errors)
    _nonnegative_float(raw, "learning_rate", 1.0, errors)
    if raw.get("max_image_size") not in {224, 256, 16, 32, 64, 128}:
        errors.append(_err("max_image_size must be 224 or 256 for real runs; small fixture sizes are accepted for tests"))
    for key, default in default_loss_weights().items():
        if key in raw:
            _nonnegative_float(raw, key, 1000.0, errors)
        elif approved:
            errors.append(_err(f"{key} is required"))
    roots = raw.get("approved_input_roots")
    if not isinstance(roots, list) or not all(isinstance(root, str) and root.strip() for root in roots):
        errors.append(_err("approved_input_roots must be a list of non-empty paths"))
        roots = []
    for index, root in enumerate(roots if isinstance(roots, list) else []):
        errors.extend(_validate_path_text(root, f"approved_input_roots[{index}]", approved, output_root=True))
        if approved and os.path.isabs(root) and _inside_repo(root):
            errors.append(_err(f"approved_input_roots[{index}] must be outside the repository"))
    errors.extend(_validate_path_text(raw.get("train_manifest_path"), "train_manifest_path", approved, require_manifest_exists))
    errors.extend(_validate_path_text(raw.get("val_manifest_path"), "val_manifest_path", approved, require_manifest_exists))
    errors.extend(_validate_path_text(raw.get("approved_run_root"), "approved_run_root", approved, output_root=True))
    errors.extend(_validate_path_text(raw.get("approved_checkpoint_root"), "approved_checkpoint_root", approved, output_root=True))
    if approved:
        for field in ("train_manifest_path", "val_manifest_path"):
            value = raw.get(field)
            if isinstance(value, str) and os.path.isabs(value) and not any(_is_under(value, root) for root in roots if isinstance(root, str)):
                errors.append(_err(f"{field} must be under an approved_input_roots entry"))
    allowed_abs: set[str] = set()
    if approved:
        for field in ("train_manifest_path", "val_manifest_path", "approved_run_root", "approved_checkpoint_root"):
            if isinstance(raw.get(field), str):
                allowed_abs.add(raw[field])
        allowed_abs.update(root for root in roots if isinstance(root, str))
    errors.extend(_walk_string_safety(raw, allowed_absolute=allowed_abs))
    return errors


def assert_valid_config(raw: dict[str, Any], require_manifest_exists: bool = False) -> None:
    errors = validate_config(raw, require_manifest_exists)
    if errors:
        raise PreSnsV3TrainingError("\n".join(errors))


def default_loss_weights() -> dict[str, float]:
    return {
        "class_loss_weight": 1.0,
        "tamper_binary_loss_weight": 1.0,
        "family_loss_weight": 0.3,
        "localization_loss_weight": 10.0,
        "non_tampered_empty_mask_loss_weight": 0.2,
        "dice_loss_weight": 1.0,
    }


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    raw = load_json(path)
    samples = raw if isinstance(raw, list) else raw.get("samples") if isinstance(raw, dict) else None
    if not isinstance(samples, list):
        raise PreSnsV3TrainingError("manifest must be a list or contain samples")
    clean = [sample for sample in samples if isinstance(sample, dict)]
    if not clean:
        raise PreSnsV3TrainingError("manifest contains no object samples")
    return clean


def _sample_class(sample: dict[str, Any]) -> str:
    label = sample.get("class_label", sample.get("label"))
    if label == "synthetic":
        label = "full_synthetic"
    if label not in CLASS_LABELS:
        raise PreSnsV3TrainingError(f"unsupported class label: {label!r}")
    return str(label)


def _sample_family(sample: dict[str, Any]) -> str | None:
    label = sample.get("family_label", sample.get("family"))
    return str(label) if label in FAMILY_LABELS else None


def class_weights(samples: list[dict[str, Any]]) -> dict[str, float]:
    counts = {label: 0 for label in CLASS_LABELS}
    for sample in samples:
        counts[_sample_class(sample)] += 1
    total = sum(counts.values())
    nonzero = [count for count in counts.values() if count]
    return {label: (float(total / (len(nonzero) * count)) if count else 0.0) for label, count in counts.items()} if nonzero else {label: 1.0 for label in CLASS_LABELS}


def dice_loss(torch: Any, logits: Any, targets: Any, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1).float()
    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - ((2.0 * intersection + eps) / (denom + eps))).mean()


def compute_v3_losses(torch: Any, outputs: dict[str, Any], batch: dict[str, Any], loss_weights: dict[str, float] | None = None) -> dict[str, Any]:
    weights = {**default_loss_weights(), **(loss_weights or {})}
    class_loss = torch.nn.CrossEntropyLoss(weight=batch.get("class_weight_tensor"))(outputs["class_logits"], batch["class_targets"])
    tamper_loss = torch.nn.CrossEntropyLoss()(outputs["tamper_binary_logits"], batch["tamper_binary_targets"])
    family_indices = batch["family_supervised_indices"]
    if family_indices:
        idx = torch.tensor(family_indices, dtype=torch.long, device=outputs["family_logits"].device)
        family_loss = torch.nn.CrossEntropyLoss()(outputs["family_logits"].index_select(0, idx), batch["family_targets"].index_select(0, idx))
    else:
        family_loss = outputs["family_logits"].sum() * 0.0
    loc_indices = batch["localization_indices"]
    if loc_indices:
        idx = torch.tensor(loc_indices, dtype=torch.long, device=outputs["localization_logits"].device)
        loc_logits = outputs["localization_logits"].index_select(0, idx)
        loc_targets = batch["mask_targets"].index_select(0, idx)
        localization_bce_loss = torch.nn.BCEWithLogitsLoss()(loc_logits, loc_targets)
        localization_dice_loss = dice_loss(torch, loc_logits, loc_targets)
    else:
        localization_bce_loss = outputs["localization_logits"].sum() * 0.0
        localization_dice_loss = outputs["localization_logits"].sum() * 0.0
    empty_indices = batch["non_tampered_indices"]
    if empty_indices:
        idx = torch.tensor(empty_indices, dtype=torch.long, device=outputs["localization_logits"].device)
        empty_targets = torch.zeros_like(outputs["localization_logits"].index_select(0, idx))
        empty_mask_loss = torch.nn.BCEWithLogitsLoss()(outputs["localization_logits"].index_select(0, idx), empty_targets)
    else:
        empty_mask_loss = outputs["localization_logits"].sum() * 0.0
    localization_loss = localization_bce_loss + weights["dice_loss_weight"] * localization_dice_loss
    total = (
        weights["class_loss_weight"] * class_loss
        + weights["tamper_binary_loss_weight"] * tamper_loss
        + weights["family_loss_weight"] * family_loss
        + weights["localization_loss_weight"] * localization_loss
        + weights["non_tampered_empty_mask_loss_weight"] * empty_mask_loss
    )
    return {
        "total_loss": total,
        "class_loss": class_loss,
        "tamper_binary_loss": tamper_loss,
        "family_loss": family_loss,
        "localization_loss": localization_loss,
        "localization_bce_loss": localization_bce_loss,
        "dice_loss": localization_dice_loss,
        "non_tampered_empty_mask_loss": empty_mask_loss,
        "total_loss_finite": math.isfinite(float(total.detach().cpu().item())),
        "class_loss_finite": math.isfinite(float(class_loss.detach().cpu().item())),
        "tamper_binary_loss_finite": math.isfinite(float(tamper_loss.detach().cpu().item())),
        "family_loss_finite": math.isfinite(float(family_loss.detach().cpu().item())),
        "localization_loss_finite": math.isfinite(float(localization_loss.detach().cpu().item())),
    }


def _runtime_deps():
    try:
        import torch
        from PIL import Image
    except Exception as exc:
        raise PreSnsV3TrainingError("torch and PIL are required for actual v3 training") from exc
    return torch, Image


def _image_tensor(torch: Any, Image: Any, sample: dict[str, Any], size: int, device: str):
    path = sample.get("image_path")
    if not isinstance(path, str) or not os.path.isfile(path):
        raise PreSnsV3TrainingError(f"sample missing existing image_path: {sample.get('sample_id')}")
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size))
        raw = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        return raw.reshape(size, size, 3).permute(2, 0, 1).float().div(255.0).to(device)


def _mask_tensor(torch: Any, Image: Any, sample: dict[str, Any], size: int, device: str):
    path = sample.get("mask_path")
    if _sample_class(sample) != "tampered" or not isinstance(path, str) or not os.path.isfile(path):
        return torch.zeros((1, size, size), dtype=torch.float32, device=device), False
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size))
        raw = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        tensor = raw.reshape(size, size).float().div(255.0)
    return (tensor >= 0.5).float().unsqueeze(0).to(device), True


def validate_manifest_sample_paths(samples: list[dict[str, Any]], approved_input_roots: list[str]) -> None:
    for sample in samples:
        for field in ("image_path", "mask_path"):
            value = sample.get(field)
            if field == "mask_path" and value in {None, ""}:
                continue
            if not isinstance(value, str) or not value:
                if field == "image_path":
                    raise PreSnsV3TrainingError(f"sample missing explicit image_path: {sample.get('sample_id')}")
                continue
            if _has_remote_scheme(value) or _has_traversal(value) or is_secret_like(value) or is_protected_path(value):
                raise PreSnsV3TrainingError(f"{field} violates protected path policy: {value}")
            if not os.path.isabs(value):
                raise PreSnsV3TrainingError(f"{field} must be absolute in approved actual mode: {value}")
            if not any(_is_under(value, root) for root in approved_input_roots):
                raise PreSnsV3TrainingError(f"{field} must be under approved_input_roots: {value}")


def _batch_from_samples(torch: Any, Image: Any, samples: list[dict[str, Any]], image_size: int, device: str, class_weight_tensor: Any | None):
    images = torch.stack([_image_tensor(torch, Image, sample, image_size, device) for sample in samples])
    masks_and_flags = [_mask_tensor(torch, Image, sample, image_size, device) for sample in samples]
    masks = torch.stack([item[0] for item in masks_and_flags])
    mask_flags = [item[1] for item in masks_and_flags]
    class_targets = torch.tensor([CLASS_LABELS.index(_sample_class(sample)) for sample in samples], dtype=torch.long, device=device)
    family_targets = torch.tensor([FAMILY_LABELS.index(_sample_family(sample) or "Real-or-N/A") for sample in samples], dtype=torch.long, device=device)
    labels = [_sample_class(sample) for sample in samples]
    return {
        "images": images,
        "mask_targets": masks,
        "class_targets": class_targets,
        "tamper_binary_targets": torch.tensor([1 if label == "tampered" else 0 for label in labels], dtype=torch.long, device=device),
        "family_targets": family_targets,
        "family_supervised_indices": [i for i, sample in enumerate(samples) if _sample_family(sample) in MEANINGFUL_FAMILY_LABELS],
        "localization_indices": [i for i, (label, has_mask) in enumerate(zip(labels, mask_flags)) if label == "tampered" and has_mask],
        "non_tampered_indices": [i for i, label in enumerate(labels) if label != "tampered"],
        "class_weight_tensor": class_weight_tensor,
    }


def _safe_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    run_root = _real(config["approved_run_root"])
    ckpt_root = _real(config["approved_checkpoint_root"])
    if _inside_repo(run_root) or _inside_repo(ckpt_root):
        raise PreSnsV3TrainingError("approved roots must be outside repository")
    if is_protected_path(str(run_root)) or is_protected_path(str(ckpt_root)):
        raise PreSnsV3TrainingError("approved roots must not use protected path segments")
    run_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    return run_root, ckpt_root


def _actual_train(config: dict[str, Any], train_samples: list[dict[str, Any]], val_samples: list[dict[str, Any]]) -> tuple[dict[str, Any], Any]:
    torch, Image = _runtime_deps()
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise PreSnsV3TrainingError("device=cuda was requested but CUDA is unavailable")
    device = "cuda" if config["device"] == "cuda" else "cpu"
    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))
    train_subset = train_samples[: int(config["max_samples_train"])]
    val_subset = val_samples[: int(config["max_samples_val"])]
    validate_manifest_sample_paths(train_subset + val_subset, config["approved_input_roots"])
    image_size = int(config["max_image_size"])
    model = build_pre_sns_v3_model(torch, int(config.get("base_channels", 12))).to(device)
    weights = class_weights(train_subset) if config.get("class_balance_strategy", "loss") == "loss" else {label: 1.0 for label in CLASS_LABELS}
    class_weight_tensor = torch.tensor([weights[label] for label in CLASS_LABELS], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0001)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.get("mixed_precision", False)) and device == "cuda")
    loss_weights = {key: float(config.get(key, value)) for key, value in default_loss_weights().items()}
    batch_size = int(config["batch_size"])
    train_metrics: list[dict[str, Any]] = []
    steps = 0
    initial_total_loss: float | None = None
    final_total_loss: float | None = None
    finite_flags = {"total_loss_finite": True, "class_loss_finite": True, "tamper_binary_loss_finite": True, "family_loss_finite": True, "localization_loss_finite": True}
    start = time.time()
    for epoch in range(int(config["epochs"])):
        random.Random(int(config["seed"]) + epoch).shuffle(train_subset)
        epoch_losses: list[float] = []
        for offset in range(0, len(train_subset), batch_size):
            batch_samples = train_subset[offset: offset + batch_size]
            batch = _batch_from_samples(torch, Image, batch_samples, image_size, device, class_weight_tensor)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(config.get("mixed_precision", False)) and device == "cuda"):
                outputs = model(batch["images"])
                losses = compute_v3_losses(torch, outputs, batch, loss_weights)
            scaler.scale(losses["total_loss"]).backward()
            if config.get("gradient_clip_norm"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            loss_value = float(losses["total_loss"].detach().cpu().item())
            initial_total_loss = loss_value if initial_total_loss is None else initial_total_loss
            final_total_loss = loss_value
            epoch_losses.append(loss_value)
            for key in finite_flags:
                finite_flags[key] = finite_flags[key] and bool(losses[key])
        train_metrics.append({"epoch": epoch + 1, "mean_total_loss": sum(epoch_losses) / len(epoch_losses), "steps_completed": steps})

    predictions: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for sample in val_subset:
            batch = _batch_from_samples(torch, Image, [sample], image_size, device, None)
            outputs = model(batch["images"])
            class_probs = torch.softmax(outputs["class_logits"][0], dim=0).detach().cpu().tolist()
            tamper_probs = torch.softmax(outputs["tamper_binary_logits"][0], dim=0).detach().cpu().tolist()
            family_probs = torch.softmax(outputs["family_logits"][0], dim=0).detach().cpu().tolist()
            mask_probs = torch.sigmoid(outputs["localization_logits"][0]).detach().cpu()
            pred_mask = (mask_probs >= 0.5).float()
            gt_mask = batch["mask_targets"][0].detach().cpu()
            iou = None
            if _sample_class(sample) == "tampered" and bool(batch["localization_indices"]):
                inter = ((pred_mask == 1) & (gt_mask == 1)).float().sum().item()
                union = ((pred_mask == 1) | (gt_mask == 1)).float().sum().item()
                iou = float(inter / union) if union else 1.0
            predictions.append(
                {
                    "gt_class": _sample_class(sample),
                    "pred_class": CLASS_LABELS[int(max(range(len(class_probs)), key=lambda i: class_probs[i]))],
                    "tampered_score": float(tamper_probs[1]),
                    "gt_family": _sample_family(sample),
                    "pred_family": FAMILY_LABELS[int(max(range(len(family_probs)), key=lambda i: family_probs[i]))],
                    "localization_iou": iou,
                    "mask_area_pct": float(pred_mask.mean().item() * 100.0),
                    "source_dataset": sample.get("source_dataset", sample.get("dataset", "unknown")),
                }
            )
    metrics = compute_metrics(predictions, float(config.get("max_false_activation_rate", 0.20)), config.get("threshold_tau_values", DEFAULT_TAU_VALUES))
    summary = {
        "marker": RUN_OK_MARKER,
        "device": device,
        "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "training_started": True,
        "training_completed": True,
        "epochs_completed": int(config["epochs"]),
        "steps_completed": steps,
        "train_samples_seen": len(train_subset) * int(config["epochs"]),
        "val_samples_seen": len(val_subset),
        "initial_total_loss": initial_total_loss,
        "final_total_loss": final_total_loss,
        **finite_flags,
        **{key: metrics[key] for key in ("class_accuracy", "class_macro_f1", "real_recall", "tampered_precision", "tampered_recall", "tampered_f1", "family_accuracy", "localization_mean_iou", "localization_median_iou", "selected_tau", "false_activation_rate", "localization_activation_recall")},
        "no_write_dry_run": False,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation": True,
        "approved_run_root": config["approved_run_root"],
        "approved_checkpoint_root": config["approved_checkpoint_root"],
        "train_metrics": train_metrics,
        "val_metrics": metrics,
        "config": config,
        "class_weights": weights,
        "elapsed_sec": time.time() - start,
    }
    return summary, model


def _write_artifacts(config: dict[str, Any], summary: dict[str, Any], model: Any) -> dict[str, Any]:
    run_root, ckpt_root = _safe_dirs(config)
    import torch
    best = ckpt_root / "best_checkpoint.pt"
    latest = ckpt_root / "latest_checkpoint.pt"
    checkpoint = {
        "model_name": "pre_sns_v3",
        "model_state_dict": model.state_dict(),
        "config": config,
        "image_size": int(config["max_image_size"]),
        "base_channels": int(config.get("base_channels", 12)),
        "class_labels": list(CLASS_LABELS),
        "family_labels": list(FAMILY_LABELS),
        "selected_tau": summary["selected_tau"],
    }
    torch.save(checkpoint, best)
    torch.save(checkpoint, latest)
    write_json(run_root / "run_summary.json", summary)
    write_json(run_root / "val_metrics.json", summary["val_metrics"])
    write_json(run_root / "threshold_calibration.json", {"selected_tau": summary["selected_tau"], "threshold_sweep": summary["val_metrics"]["threshold_sweep"], "fallback_selected_tau": summary["val_metrics"]["fallback_selected_tau"]})
    write_json(run_root / "confusion_matrix.json", summary["val_metrics"]["confusion_matrix"])
    write_json(run_root / "per_source_confusion_matrix.json", summary["val_metrics"]["per_source_dataset_confusion_matrix"])
    write_json(run_root / "config_snapshot.json", config)
    with open(run_root / "train_metrics.jsonl", "w", encoding="utf-8") as handle:
        for row in summary["train_metrics"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "artifact_kind": "pre_sns_v3_training",
        "files": {
            "run_summary": str(run_root / "run_summary.json"),
            "val_metrics": str(run_root / "val_metrics.json"),
            "threshold_calibration": str(run_root / "threshold_calibration.json"),
            "confusion_matrix": str(run_root / "confusion_matrix.json"),
            "per_source_confusion_matrix": str(run_root / "per_source_confusion_matrix.json"),
            "train_metrics": str(run_root / "train_metrics.jsonl"),
            "config_snapshot": str(run_root / "config_snapshot.json"),
            "best_checkpoint": str(best),
            "latest_checkpoint": str(latest),
        },
    }
    write_json(run_root / "artifact_manifest.json", manifest)
    summary.update(
        {
            "artifact_manifest_path": str(run_root / "artifact_manifest.json"),
            "run_summary_path": str(run_root / "run_summary.json"),
            "val_metrics_path": str(run_root / "val_metrics.json"),
            "threshold_calibration_path": str(run_root / "threshold_calibration.json"),
            "confusion_matrix_path": str(run_root / "confusion_matrix.json"),
            "best_checkpoint_path": str(best),
            "latest_checkpoint_path": str(latest),
        }
    )
    write_json(run_root / "run_summary.json", summary)
    return summary


def _synthetic_samples(count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(max(1, count)):
        label = CLASS_LABELS[i % len(CLASS_LABELS)]
        rows.append({"sample_id": f"dry-v3-{i}", "class_label": label, "family_label": "Real-or-N/A" if label == "real" else FAMILY_LABELS[i % 4], "source_dataset": "dry_run"})
    return rows


def _dry_run(config: dict[str, Any]) -> dict[str, Any]:
    train = load_manifest(config["train_manifest_path"])[: int(config["max_samples_train"])] if os.path.isfile(str(config["train_manifest_path"])) else _synthetic_samples(int(config["max_samples_train"]))
    val = load_manifest(config["val_manifest_path"])[: int(config["max_samples_val"])] if os.path.isfile(str(config["val_manifest_path"])) else _synthetic_samples(int(config["max_samples_val"]))
    rng = random.Random(int(config["seed"]))
    predictions = []
    for i, sample in enumerate(val):
        gt = _sample_class(sample)
        pred = gt if i % 5 else ("tampered" if gt != "tampered" else "real")
        score = {"real": 0.12, "full_synthetic": 0.25, "tampered": 0.72}[gt] + rng.uniform(-0.05, 0.05)
        predictions.append({"gt_class": gt, "pred_class": pred, "tampered_score": max(0.0, min(1.0, score)), "gt_family": _sample_family(sample), "pred_family": _sample_family(sample) or "Real-or-N/A", "localization_iou": 0.4 if gt == "tampered" else None, "mask_area_pct": 6.0 if gt == "tampered" else 0.0, "source_dataset": sample.get("source_dataset", "dry_run")})
    metrics = compute_metrics(predictions, float(config.get("max_false_activation_rate", 0.20)), config.get("threshold_tau_values", DEFAULT_TAU_VALUES))
    return {
        "marker": DRY_RUN_OK_MARKER,
        "device": config.get("device", "cpu"),
        "cuda_device_name": None,
        "training_started": True,
        "training_completed": True,
        "epochs_completed": int(config["epochs"]),
        "steps_completed": max(1, math.ceil(len(train) / int(config["batch_size"]))) * int(config["epochs"]),
        "train_samples_seen": len(train) * int(config["epochs"]),
        "val_samples_seen": len(val),
        "initial_total_loss": 2.0,
        "final_total_loss": 1.0,
        "total_loss_finite": True,
        "class_loss_finite": True,
        "tamper_binary_loss_finite": True,
        "family_loss_finite": True,
        "localization_loss_finite": True,
        **{key: metrics[key] for key in ("class_accuracy", "class_macro_f1", "real_recall", "tampered_precision", "tampered_recall", "tampered_f1", "family_accuracy", "localization_mean_iou", "localization_median_iou", "selected_tau", "false_activation_rate", "localization_activation_recall")},
        "no_write_dry_run": True,
        "artifact_writes": False,
        "checkpoint_writes": False,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation": True,
        "approved_run_root": config["approved_run_root"],
        "approved_checkpoint_root": config["approved_checkpoint_root"],
        "artifact_manifest_path": None,
        "run_summary_path": None,
        "val_metrics_path": None,
        "threshold_calibration_path": None,
        "confusion_matrix_path": None,
        "best_checkpoint_path": None,
        "latest_checkpoint_path": None,
        "val_metrics": metrics,
        "train_metrics": [{"epoch": epoch + 1, "mean_total_loss": 1.0 / (epoch + 1)} for epoch in range(int(config["epochs"]))],
    }


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_manifest_exists=config.get("config_kind") == APPROVED_KIND)
    if config.get("no_write_dry_run") is True:
        return _dry_run(config)
    if config.get("config_kind") != APPROVED_KIND:
        raise PreSnsV3TrainingError("actual artifact writing requires approved local v3 mode")
    train = load_manifest(config["train_manifest_path"])
    val = load_manifest(config["val_manifest_path"])
    summary, model = _actual_train(config, train, val)
    return _write_artifacts(config, summary, model)

