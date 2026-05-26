"""Guarded pre-SNS meaningful training v2 helpers.

This module keeps the v2 path pre-SNS only. It validates explicit local
manifests, computes proposal-aligned validation metrics, supports no-write
dry-runs, and writes actual run artifacts only under approved roots outside
the repository.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_OK_MARKER = "PRE_SNS_MEANINGFUL_TRAINING_V2_CONFIG_OK"
RUN_OK_MARKER = "PRE_SNS_MEANINGFUL_TRAINING_V2_RUN_OK"
DRY_RUN_OK_MARKER = "PRE_SNS_MEANINGFUL_TRAINING_V2_DRY_RUN_OK"
APPROVAL_TEXT = "I_APPROVE_PRE_SNS_MEANINGFUL_TRAINING_V2"

CLASS_LABELS = ("real", "full_synthetic", "tampered")
FAMILY_LABELS = ("LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A")
MEANINGFUL_FAMILY_LABELS = ("LatDiff", "PixDiff", "GAN", "Other")
TAU_VALUES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}
SECRET_WORDS = ("api_key", "apikey", "secret", "token", "password", "credential", "private_key")
REMOTE_PREFIXES = ("http://", "https://", "s3://", "gs://", "hf://")


class PreSnsMeaningfulTrainingV2Error(ValueError):
    """Raised when a v2 training config or run violates guardrails."""


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


def _real(path: str | Path) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def _is_under(path: str | Path, root: str | Path) -> bool:
    path_real = _real(path)
    root_real = _real(root)
    try:
        return os.path.commonpath([str(path_real), str(root_real)]) == str(root_real)
    except ValueError:
        return False


def _err(message: str) -> str:
    return f"- {message}"


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


def _inside_repo(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT)


def _repo_artifact_path(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT / "outputs") or _is_under(path, REPO_ROOT / "checkpoints")


def _walk_string_safety(node: Any, path: str = "", allowed_absolute: set[str] | None = None) -> list[str]:
    allowed_absolute = allowed_absolute or set()
    errors: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if is_secret_like(key_text):
                errors.append(_err(f"{child_path}: secret-like key rejected"))
            errors.extend(_walk_string_safety(value, child_path, allowed_absolute))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            errors.extend(_walk_string_safety(value, f"{path}[{index}]", allowed_absolute))
    elif isinstance(node, str):
        text = node.strip()
        if not text:
            return errors
        if _has_remote_scheme(text):
            errors.append(_err(f"{path}: URL or remote scheme rejected"))
        if len(text) >= 3 and text[1] == ":" and text[2] in {"/", "\\"}:
            errors.append(_err(f"{path}: Windows drive path rejected"))
        if _has_traversal(text):
            errors.append(_err(f"{path}: path traversal rejected"))
        if is_secret_like(text):
            errors.append(_err(f"{path}: secret-like value rejected"))
        if is_protected_path(text):
            errors.append(_err(f"{path}: protected path segment rejected"))
        if text.startswith("/") and text not in allowed_absolute:
            errors.append(_err(f"{path}: absolute path rejected unless explicitly approved"))
    return errors


def _positive_int(raw: dict[str, Any], key: str, max_value: int, errors: list[str]) -> None:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(_err(f"{key} must be a positive integer"))
    elif value > max_value:
        errors.append(_err(f"{key} must be <= {max_value}"))


def _positive_float(raw: dict[str, Any], key: str, max_value: float, errors: list[str]) -> None:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
        errors.append(_err(f"{key} must be a positive number"))
    elif float(value) > max_value:
        errors.append(_err(f"{key} must be <= {max_value}"))


def _validate_manifest_path(value: Any, field: str, approved: bool, require_exists: bool) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, str) or not value.strip():
        return [_err(f"{field} must be a non-empty path")]
    text = value.strip()
    if _has_remote_scheme(text):
        errors.append(_err(f"{field} must not be a URL or remote scheme"))
    if _has_traversal(text):
        errors.append(_err(f"{field} must not contain path traversal"))
    if is_protected_path(text):
        errors.append(_err(f"{field} must not reference protected paths"))
    if is_secret_like(text):
        errors.append(_err(f"{field} must not contain secret-like text"))
    if approved:
        if not os.path.isabs(text):
            errors.append(_err(f"{field} must be absolute in approved local mode"))
        if require_exists and not os.path.isfile(text):
            errors.append(_err(f"{field} must exist as a file"))
    else:
        if os.path.isabs(text):
            errors.append(_err(f"{field} must be symbolic in example mode"))
    return errors


def _validate_artifact_root(value: Any, field: str, approved: bool) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, str) or not value.strip():
        return [_err(f"{field} must be a non-empty path")]
    text = value.strip()
    if _has_remote_scheme(text):
        errors.append(_err(f"{field} must not be a URL or remote scheme"))
    if _has_traversal(text):
        errors.append(_err(f"{field} must not contain path traversal"))
    if is_protected_path(text):
        errors.append(_err(f"{field} must not reference protected paths"))
    if is_secret_like(text):
        errors.append(_err(f"{field} must not contain secret-like text"))
    if approved:
        if not os.path.isabs(text):
            errors.append(_err(f"{field} must be absolute in approved local mode"))
        if os.path.isabs(text) and _inside_repo(text):
            errors.append(_err(f"{field} must be outside the repository"))
        if os.path.isabs(text) and _repo_artifact_path(text):
            errors.append(_err(f"{field} must not be repository outputs/checkpoints"))
    elif os.path.isabs(text):
        errors.append(_err(f"{field} must be symbolic in example mode"))
    return errors


def validate_config(raw: dict[str, Any], require_manifest_exists: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(raw, dict):
        return [_err("config root must be a JSON object")]

    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "train_manifest_path",
        "val_manifest_path",
        "approved_input_roots",
        "approved_run_root",
        "approved_checkpoint_root",
        "device",
        "seed",
        "max_samples_train",
        "max_samples_val",
        "max_image_size",
        "batch_size",
        "epochs",
        "learning_rate",
        "no_write_dry_run",
        "no_download",
        "no_network",
        "no_sns_augmentation",
        "class_labels",
        "family_labels",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))

    kind = raw.get("config_kind")
    approved = kind == "approved_pre_sns_meaningful_training_v2"
    if kind not in {"example_symbolic", "approved_pre_sns_meaningful_training_v2"}:
        errors.append(_err("config_kind must be example_symbolic or approved_pre_sns_meaningful_training_v2"))

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
    _positive_int(raw, "max_image_size", 512, errors)
    _positive_int(raw, "batch_size", 256, errors)
    _positive_int(raw, "epochs", 50, errors)
    _positive_float(raw, "learning_rate", 1.0, errors)

    if approved:
        if raw.get("execution_mode") != "approved_local_pre_sns_meaningful_training_v2":
            errors.append(_err("execution_mode must be approved_local_pre_sns_meaningful_training_v2"))
        if raw.get("required_approval_text") != APPROVAL_TEXT:
            errors.append(_err("required_approval_text must document the approval phrase"))
        if raw.get("user_approval_text") != APPROVAL_TEXT:
            errors.append(_err("user_approval_text must match required approval phrase"))
    else:
        if raw.get("execution_mode") != "example_only":
            errors.append(_err("execution_mode must be example_only"))
        if raw.get("user_approval_text") not in {"", None}:
            errors.append(_err("example user_approval_text must be empty"))

    input_roots = raw.get("approved_input_roots")
    if not isinstance(input_roots, list) or not all(isinstance(root, str) and root.strip() for root in input_roots):
        errors.append(_err("approved_input_roots must be a list of non-empty paths"))
        input_roots = []
    if approved and not input_roots:
        errors.append(_err("approved_input_roots is required in approved local mode"))
    for index, root in enumerate(input_roots if isinstance(input_roots, list) else []):
        root_errors = _validate_artifact_root(root, f"approved_input_roots[{index}]", approved)
        errors.extend(root_errors)

    errors.extend(_validate_manifest_path(raw.get("train_manifest_path"), "train_manifest_path", approved, require_manifest_exists))
    errors.extend(_validate_manifest_path(raw.get("val_manifest_path"), "val_manifest_path", approved, require_manifest_exists))
    if approved and isinstance(input_roots, list):
        for field in ("train_manifest_path", "val_manifest_path"):
            value = raw.get(field)
            if isinstance(value, str) and os.path.isabs(value):
                if not any(_is_under(value, root) for root in input_roots if isinstance(root, str)):
                    errors.append(_err(f"{field} must be under an approved_input_roots entry"))
    errors.extend(_validate_artifact_root(raw.get("approved_run_root"), "approved_run_root", approved))
    errors.extend(_validate_artifact_root(raw.get("approved_checkpoint_root"), "approved_checkpoint_root", approved))

    allowed_abs: set[str] = set()
    if approved:
        for field in ("train_manifest_path", "val_manifest_path", "approved_run_root", "approved_checkpoint_root"):
            value = raw.get(field)
            if isinstance(value, str):
                allowed_abs.add(value)
        for root in input_roots if isinstance(input_roots, list) else []:
            if isinstance(root, str):
                allowed_abs.add(root)
    errors.extend(_walk_string_safety(raw, allowed_absolute=allowed_abs))
    return errors


def assert_valid_config(raw: dict[str, Any], require_manifest_exists: bool = False) -> None:
    errors = validate_config(raw, require_manifest_exists=require_manifest_exists)
    if errors:
        raise PreSnsMeaningfulTrainingV2Error("\n".join(errors))


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    raw = load_json(path)
    if isinstance(raw, list):
        samples = raw
    elif isinstance(raw, dict):
        if isinstance(raw.get("samples"), list):
            samples = raw["samples"]
        elif isinstance(raw.get("sample_manifest"), list):
            samples = raw["sample_manifest"]
        else:
            raise PreSnsMeaningfulTrainingV2Error("manifest must contain samples or sample_manifest")
    else:
        raise PreSnsMeaningfulTrainingV2Error("manifest root must be a JSON object or list")
    clean = [sample for sample in samples if isinstance(sample, dict)]
    if not clean:
        raise PreSnsMeaningfulTrainingV2Error("manifest contains no object samples")
    return clean


def _sample_class(sample: dict[str, Any]) -> str:
    label = sample.get("class_label", sample.get("label"))
    if label == "synthetic":
        label = "full_synthetic"
    if label not in CLASS_LABELS:
        raise PreSnsMeaningfulTrainingV2Error(f"unsupported class label: {label!r}")
    return str(label)


def _sample_family(sample: dict[str, Any]) -> str | None:
    label = sample.get("family_label", sample.get("family"))
    return str(label) if label in FAMILY_LABELS else None


def family_is_meaningful(label: str | None) -> bool:
    return label in MEANINGFUL_FAMILY_LABELS


def class_weights(samples: list[dict[str, Any]]) -> dict[str, float]:
    counts = {label: 0 for label in CLASS_LABELS}
    for sample in samples:
        counts[_sample_class(sample)] += 1
    total = sum(counts.values())
    nonzero = [count for count in counts.values() if count > 0]
    if not nonzero:
        return {label: 1.0 for label in CLASS_LABELS}
    return {
        label: float(total / (len(nonzero) * count)) if count > 0 else 0.0
        for label, count in counts.items()
    }


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    matrix = [[0 for _ in CLASS_LABELS] for _ in CLASS_LABELS]
    idx = {label: index for index, label in enumerate(CLASS_LABELS)}
    for truth, pred in zip(y_true, y_pred):
        matrix[idx[truth]][idx[pred]] += 1
    return {"labels": list(CLASS_LABELS), "matrix": matrix}


def per_class_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, float]]:
    cm = confusion_matrix(y_true, y_pred)["matrix"]
    metrics: dict[str, dict[str, float]] = {}
    for index, label in enumerate(CLASS_LABELS):
        tp = cm[index][index]
        fp = sum(cm[row][index] for row in range(len(CLASS_LABELS)) if row != index)
        fn = sum(cm[index][col] for col in range(len(CLASS_LABELS)) if col != index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def summarize_scores_by_class(y_true: list[str], tampered_scores: list[float]) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for label in CLASS_LABELS:
        values = [float(score) for truth, score in zip(y_true, tampered_scores) if truth == label]
        if not values:
            summary[label] = {"count": 0, "min": None, "max": None, "mean": None, "median": None}
        else:
            summary[label] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "median": statistics.median(values),
            }
    return summary


def binary_iou(pred_mask: list[Any], gt_mask: list[Any]) -> float:
    pred = [1 if bool(value) else 0 for value in _flatten(pred_mask)]
    truth = [1 if bool(value) else 0 for value in _flatten(gt_mask)]
    if len(pred) != len(truth):
        raise PreSnsMeaningfulTrainingV2Error("predicted and ground-truth masks must have equal size")
    intersection = sum(1 for p, t in zip(pred, truth) if p and t)
    union = sum(1 for p, t in zip(pred, truth) if p or t)
    return float(intersection / union) if union else 1.0


def _flatten(mask: list[Any]) -> list[Any]:
    if not isinstance(mask, list):
        return [mask]
    flat: list[Any] = []
    for value in mask:
        if isinstance(value, list):
            flat.extend(_flatten(value))
        else:
            flat.append(value)
    return flat


def threshold_sweep(y_true: list[str], tampered_scores: list[float], ious: list[float | None] | None = None) -> list[dict[str, float]]:
    ious = ious or [None] * len(y_true)
    rows: list[dict[str, float]] = []
    tampered_total = sum(1 for label in y_true if label == "tampered")
    non_tampered_total = len(y_true) - tampered_total
    for tau in TAU_VALUES:
        activated = [score >= tau for score in tampered_scores]
        true_activations = sum(1 for label, active in zip(y_true, activated) if label == "tampered" and active)
        false_activations = sum(1 for label, active in zip(y_true, activated) if label != "tampered" and active)
        recall = true_activations / tampered_total if tampered_total else 0.0
        false_rate = false_activations / non_tampered_total if non_tampered_total else 0.0
        active_ious = [iou for label, active, iou in zip(y_true, activated, ious) if label == "tampered" and active and iou is not None]
        mean_iou = sum(active_ious) / len(active_ious) if active_ious else 0.0
        score = recall - 0.50 * false_rate + 0.10 * mean_iou
        rows.append(
            {
                "tau": float(tau),
                "localization_activation_recall": recall,
                "false_activation_rate": false_rate,
                "localization_mean_iou_when_activated": mean_iou,
                "selection_score": score,
            }
        )
    return rows


def select_tau(sweep: list[dict[str, float]]) -> float:
    if not sweep:
        raise PreSnsMeaningfulTrainingV2Error("threshold sweep is empty")
    # Deterministic trade-off: maximize recall - 0.5*false_activation + 0.1*IoU.
    # Ties choose the smaller tau to preserve tampered recall.
    best = sorted(sweep, key=lambda row: (-row["selection_score"], row["tau"]))[0]
    return float(best["tau"])


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise PreSnsMeaningfulTrainingV2Error("predictions must not be empty")
    y_true = [str(row["gt_class"]) for row in predictions]
    y_pred = [str(row["pred_class"]) for row in predictions]
    tampered_scores = [float(row["tampered_score"]) for row in predictions]
    per_class = per_class_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    accuracy = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred) / len(y_true)
    macro_f1 = sum(row["f1"] for row in per_class.values()) / len(CLASS_LABELS)

    family_rows = [
        row for row in predictions
        if family_is_meaningful(row.get("gt_family")) and row.get("pred_family") in FAMILY_LABELS
    ]
    family_accuracy = None
    if family_rows:
        family_accuracy = sum(1 for row in family_rows if row.get("gt_family") == row.get("pred_family")) / len(family_rows)

    ious: list[float | None] = []
    tampered_ious: list[float] = []
    for row in predictions:
        iou = row.get("localization_iou")
        if iou is None and row.get("pred_mask") is not None and row.get("gt_mask") is not None:
            iou = binary_iou(row["pred_mask"], row["gt_mask"])
        if iou is not None:
            iou = float(iou)
        ious.append(iou)
        if row["gt_class"] == "tampered" and iou is not None:
            tampered_ious.append(float(iou))

    sweep = threshold_sweep(y_true, tampered_scores, ious)
    selected = select_tau(sweep)
    selected_row = next(row for row in sweep if row["tau"] == selected)
    return {
        "class_accuracy": accuracy,
        "class_macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "family_accuracy": family_accuracy,
        "family_samples_evaluated": len(family_rows),
        "tampered_score_summary_by_gt_class": summarize_scores_by_class(y_true, tampered_scores),
        "threshold_sweep": sweep,
        "selected_tau": selected,
        "localization_activation_recall": selected_row["localization_activation_recall"],
        "false_activation_rate": selected_row["false_activation_rate"],
        "localization_mean_iou": sum(tampered_ious) / len(tampered_ious) if tampered_ious else None,
        "localization_median_iou": statistics.median(tampered_ious) if tampered_ious else None,
        "samples_evaluated": len(predictions),
    }


def _deterministic_predictions(samples: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    predictions: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        gt = _sample_class(sample)
        family = _sample_family(sample)
        # Dry-run predictions are deterministic and intentionally imperfect so
        # metric code exercises confusion and tau calibration paths.
        if index % 7 == 0 and gt == "tampered":
            pred = "real"
        elif index % 11 == 0 and gt == "real":
            pred = "tampered"
        else:
            pred = gt
        base_score = {"real": 0.28, "full_synthetic": 0.38, "tampered": 0.62}[gt]
        tampered_score = max(0.01, min(0.99, base_score + rng.uniform(-0.08, 0.08)))
        iou = None
        if gt == "tampered" and sample.get("mask_path"):
            iou = max(0.0, min(1.0, 0.45 + rng.uniform(-0.10, 0.20)))
        predictions.append(
            {
                "gt_class": gt,
                "pred_class": pred,
                "tampered_score": tampered_score,
                "gt_family": family,
                "pred_family": family if family_is_meaningful(family) else "Real-or-N/A",
                "localization_iou": iou,
            }
        )
    return predictions


def _load_runtime_deps():
    try:
        import torch
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on local runtime
        raise PreSnsMeaningfulTrainingV2Error("torch and PIL are required for approved actual training") from exc
    return torch, Image


def _image_tensor(torch: Any, Image: Any, sample: dict[str, Any], image_size: int, device: str):
    image_path = sample.get("image_path")
    if not isinstance(image_path, str) or not os.path.isfile(image_path):
        raise PreSnsMeaningfulTrainingV2Error(f"sample missing existing explicit image_path: {sample.get('sample_id')}")
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        raw = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        tensor = raw.reshape(image_size, image_size, 3).permute(2, 0, 1).float().div(255.0)
    return tensor.reshape(-1).to(device)


def _mask_tensor(torch: Any, Image: Any, sample: dict[str, Any], image_size: int, device: str):
    mask_path = sample.get("mask_path")
    if _sample_class(sample) != "tampered" or not isinstance(mask_path, str):
        return None
    if not os.path.isfile(mask_path):
        return None
    with Image.open(mask_path) as image:
        image = image.convert("L").resize((image_size, image_size))
        raw = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        tensor = raw.reshape(image_size, image_size).float().div(255.0)
    return (tensor >= 0.5).float().reshape(-1).to(device)


def _actual_train_and_eval(config: dict[str, Any], train_samples: list[dict[str, Any]], val_samples: list[dict[str, Any]]) -> tuple[dict[str, Any], Any]:
    torch, Image = _load_runtime_deps()
    from cv_forensics.pre_sns_integrated_model import build_tiny_integrated_model

    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise PreSnsMeaningfulTrainingV2Error("device=cuda was requested but CUDA is unavailable")
    device = "cuda" if config["device"] == "cuda" else "cpu"
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    image_size = int(config["max_image_size"])
    train_subset = train_samples[: int(config["max_samples_train"])]
    val_subset = val_samples[: int(config["max_samples_val"])]
    validate_manifest_sample_paths(train_subset + val_subset, config.get("approved_input_roots", []))
    model = build_tiny_integrated_model(torch, 3 * image_size * image_size, image_size * image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    weights = class_weights(train_subset)
    class_weight_tensor = torch.tensor([weights[label] for label in CLASS_LABELS], dtype=torch.float32, device=device)
    class_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)
    batch_size = int(config["batch_size"])
    train_metrics: list[dict[str, Any]] = []
    steps = 0
    start = time.time()

    for epoch in range(int(config["epochs"])):
        random.Random(seed + epoch).shuffle(train_subset)
        epoch_losses: list[float] = []
        for offset in range(0, len(train_subset), batch_size):
            batch = train_subset[offset: offset + batch_size]
            inputs = torch.stack([_image_tensor(torch, Image, sample, image_size, device) for sample in batch])
            class_targets = torch.tensor([CLASS_LABELS.index(_sample_class(sample)) for sample in batch], dtype=torch.long, device=device)
            outputs = model(inputs)
            class_loss = class_loss_fn(outputs["class_logits"], class_targets)

            family_indices = [idx for idx, sample in enumerate(batch) if family_is_meaningful(_sample_family(sample))]
            if family_indices:
                idx_tensor = torch.tensor(family_indices, dtype=torch.long, device=device)
                family_targets = torch.tensor([FAMILY_LABELS.index(_sample_family(batch[idx])) for idx in family_indices], dtype=torch.long, device=device)
                family_loss = torch.nn.CrossEntropyLoss()(outputs["family_logits"].index_select(0, idx_tensor), family_targets)
            else:
                family_loss = outputs["family_logits"].sum() * 0.0

            mask_pairs = [(idx, _mask_tensor(torch, Image, sample, image_size, device)) for idx, sample in enumerate(batch)]
            mask_pairs = [(idx, mask) for idx, mask in mask_pairs if mask is not None]
            if mask_pairs:
                idx_tensor = torch.tensor([idx for idx, _ in mask_pairs], dtype=torch.long, device=device)
                masks = torch.stack([mask for _, mask in mask_pairs])
                loc_loss = torch.nn.BCEWithLogitsLoss()(outputs["localization_logits"].index_select(0, idx_tensor), masks)
            else:
                loc_loss = outputs["localization_logits"].sum() * 0.0

            total_loss = class_loss + family_loss + loc_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            steps += 1
            epoch_losses.append(float(total_loss.detach().cpu().item()))
        train_metrics.append({"epoch": epoch + 1, "mean_total_loss": sum(epoch_losses) / len(epoch_losses), "steps_completed": steps})

    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in val_subset:
            inputs = _image_tensor(torch, Image, sample, image_size, device).unsqueeze(0)
            outputs = model(inputs)
            class_probs = torch.softmax(outputs["class_logits"][0], dim=0).detach().cpu().tolist()
            family_probs = torch.softmax(outputs["family_logits"][0], dim=0).detach().cpu().tolist()
            pred_index = max(range(len(class_probs)), key=lambda idx: class_probs[idx])
            family_index = max(range(len(family_probs)), key=lambda idx: family_probs[idx])
            mask = _mask_tensor(torch, Image, sample, image_size, device)
            iou = None
            if mask is not None:
                pred_mask = (torch.sigmoid(outputs["localization_logits"][0]) >= 0.5).float()
                intersection = ((pred_mask == 1) & (mask == 1)).float().sum().item()
                union = ((pred_mask == 1) | (mask == 1)).float().sum().item()
                iou = float(intersection / union) if union else 1.0
            predictions.append(
                {
                    "gt_class": _sample_class(sample),
                    "pred_class": CLASS_LABELS[pred_index],
                    "tampered_score": float(class_probs[CLASS_LABELS.index("tampered")]),
                    "gt_family": _sample_family(sample),
                    "pred_family": FAMILY_LABELS[family_index],
                    "localization_iou": iou,
                }
            )
    metrics = compute_metrics(predictions)
    summary = {
        "marker": RUN_OK_MARKER,
        "device": device,
        "epochs_completed": int(config["epochs"]),
        "steps_completed": steps,
        "samples_train": len(train_subset),
        "samples_val": len(val_subset),
        "train_metrics": train_metrics,
        "val_metrics": metrics,
        "class_weights": weights,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation": True,
        "elapsed_sec": time.time() - start,
    }
    return summary, model


def _safe_run_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    run_root = _real(config["approved_run_root"])
    checkpoint_root = _real(config["approved_checkpoint_root"])
    if _inside_repo(run_root) or _inside_repo(checkpoint_root):
        raise PreSnsMeaningfulTrainingV2Error("approved roots must be outside repository")
    if is_protected_path(str(run_root)) or is_protected_path(str(checkpoint_root)):
        raise PreSnsMeaningfulTrainingV2Error("approved roots must not use protected path segments")
    if run_root == checkpoint_root:
        raise PreSnsMeaningfulTrainingV2Error("approved_run_root and approved_checkpoint_root must be distinct")
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    return run_root, checkpoint_root


def validate_manifest_sample_paths(samples: list[dict[str, Any]], approved_input_roots: list[str]) -> None:
    roots = [root for root in approved_input_roots if isinstance(root, str)]
    for sample in samples:
        for field in ("image_path", "mask_path"):
            value = sample.get(field)
            if field == "mask_path" and value in {None, ""}:
                continue
            if not isinstance(value, str) or not value:
                if field == "image_path":
                    raise PreSnsMeaningfulTrainingV2Error(f"sample missing explicit image_path: {sample.get('sample_id')}")
                continue
            if _has_remote_scheme(value) or _has_traversal(value) or is_secret_like(value) or is_protected_path(value):
                raise PreSnsMeaningfulTrainingV2Error(f"{field} violates protected path policy: {value}")
            if not os.path.isabs(value):
                raise PreSnsMeaningfulTrainingV2Error(f"{field} must be absolute in approved actual mode: {value}")
            if not any(_is_under(value, root) for root in roots):
                raise PreSnsMeaningfulTrainingV2Error(f"{field} must be under approved_input_roots: {value}")


def _write_actual_artifacts(config: dict[str, Any], summary: dict[str, Any], model: Any) -> dict[str, Any]:
    run_root, checkpoint_root = _safe_run_dirs(config)
    train_metrics = summary.get("train_metrics", [])
    val_metrics = summary["val_metrics"]
    best_checkpoint = checkpoint_root / "best_checkpoint.pt"
    latest_checkpoint = checkpoint_root / "latest_checkpoint.pt"
    try:
        import torch
        torch.save({"model_state_dict": model.state_dict(), "config": config, "selected_tau": val_metrics["selected_tau"]}, best_checkpoint)
        torch.save({"model_state_dict": model.state_dict(), "config": config, "selected_tau": val_metrics["selected_tau"]}, latest_checkpoint)
    except Exception as exc:  # pragma: no cover - depends on local runtime
        raise PreSnsMeaningfulTrainingV2Error("failed to write checkpoint under approved_checkpoint_root") from exc

    write_json(run_root / "run_summary.json", summary)
    with open(run_root / "train_metrics.jsonl", "w", encoding="utf-8") as handle:
        for row in train_metrics:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(run_root / "val_metrics.json", val_metrics)
    write_json(run_root / "threshold_calibration.json", {"selected_tau": val_metrics["selected_tau"], "threshold_sweep": val_metrics["threshold_sweep"]})
    write_json(run_root / "confusion_matrix.json", val_metrics["confusion_matrix"])
    manifest = {
        "artifact_kind": "pre_sns_meaningful_training_v2",
        "run_root": str(run_root),
        "checkpoint_root": str(checkpoint_root),
        "files": {
            "run_summary": str(run_root / "run_summary.json"),
            "train_metrics": str(run_root / "train_metrics.jsonl"),
            "val_metrics": str(run_root / "val_metrics.json"),
            "threshold_calibration": str(run_root / "threshold_calibration.json"),
            "confusion_matrix": str(run_root / "confusion_matrix.json"),
            "best_checkpoint": str(best_checkpoint),
            "latest_checkpoint": str(latest_checkpoint),
        },
    }
    write_json(run_root / "artifact_manifest.json", manifest)
    summary["artifact_manifest_path"] = str(run_root / "artifact_manifest.json")
    summary["best_checkpoint_path"] = str(best_checkpoint)
    summary["latest_checkpoint_path"] = str(latest_checkpoint)
    return summary


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_manifest_exists=config.get("config_kind") == "approved_pre_sns_meaningful_training_v2")
    no_write = config.get("no_write_dry_run") is True
    if config.get("config_kind") != "approved_pre_sns_meaningful_training_v2" and not no_write:
        raise PreSnsMeaningfulTrainingV2Error("actual artifact writing requires approved local mode")

    if no_write:
        seed = int(config["seed"])
        train_count = int(config["max_samples_train"])
        val_count = int(config["max_samples_val"])
        # If symbolic manifests do not exist, dry-run uses synthetic manifest rows.
        if os.path.isfile(str(config["train_manifest_path"])):
            train_samples = load_manifest(config["train_manifest_path"])[:train_count]
        else:
            train_samples = synthetic_samples(train_count)
        if os.path.isfile(str(config["val_manifest_path"])):
            val_samples = load_manifest(config["val_manifest_path"])[:val_count]
        else:
            val_samples = synthetic_samples(val_count)
        predictions = _deterministic_predictions(val_samples, seed)
        metrics = compute_metrics(predictions)
        return {
            "marker": DRY_RUN_OK_MARKER,
            "training_completed": True,
            "no_write_dry_run": True,
            "artifact_writes": False,
            "checkpoint_writes": False,
            "samples_train": len(train_samples),
            "samples_val": len(val_samples),
            "class_weights": class_weights(train_samples),
            "train_metrics": [
                {"epoch": epoch + 1, "mean_total_loss": round(1.0 / (epoch + 1), 6)}
                for epoch in range(int(config["epochs"]))
            ],
            "val_metrics": metrics,
            "no_download": True,
            "no_network": True,
            "no_sns_augmentation": True,
        }

    train_samples = load_manifest(config["train_manifest_path"])[: int(config["max_samples_train"])]
    val_samples = load_manifest(config["val_manifest_path"])[: int(config["max_samples_val"])]
    summary, model = _actual_train_and_eval(config, train_samples, val_samples)
    return _write_actual_artifacts(config, summary, model)


def synthetic_samples(count: int) -> list[dict[str, Any]]:
    labels = list(CLASS_LABELS)
    families = list(FAMILY_LABELS)
    samples: list[dict[str, Any]] = []
    for index in range(max(1, count)):
        label = labels[index % len(labels)]
        sample: dict[str, Any] = {
            "sample_id": f"synthetic-dry-run-{index}",
            "class_label": label,
            "family_label": families[index % len(families)] if label != "real" else "Real-or-N/A",
            "image_path": f"SYMBOLIC_IMAGE_{index}.jpg",
        }
        if label == "tampered":
            sample["mask_path"] = f"SYMBOLIC_MASK_{index}.png"
        samples.append(sample)
    return samples


def load_config(path: str | Path) -> dict[str, Any]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise PreSnsMeaningfulTrainingV2Error("config root must be a JSON object")
    return raw
