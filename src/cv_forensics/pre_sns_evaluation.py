"""Pre-SNS evaluation helpers for approved local manifest subsets."""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from .model_output_schema import LOCALIZATION_ACTIVATED
from .pre_sns_inference_report import (
    confidence_map,
    load_trained_model,
    localization_summary,
    prepare_image_tensor,
    select_device,
)
from .pre_sns_integrated_model import CLASS_LABELS, FAMILY_SMOKE_LABELS

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER = "PRE_SNS_EVALUATION_OK"
CONFIG_OK_MARKER = "PRE_SNS_EVALUATION_CONFIG_OK"
RESULT_OK_MARKER = "PRE_SNS_EVALUATION_RESULT_OK"
APPROVAL_TEXT = "I_APPROVE_PRE_SNS_EVALUATION"
APPROVED_CONFIG_KIND = "approved_pre_sns_evaluation"
LEGACY_APPROVED_CONFIG_KINDS = {"approved_local_pre_sns_evaluation"}
CONFIG_KINDS = {"example_symbolic", APPROVED_CONFIG_KIND, *LEGACY_APPROVED_CONFIG_KINDS}
REMOTE_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
SECRET_KEY_RE = re.compile(r"(^|_)(api[_-]?key|secret|token|password|credential|private[_-]?key)($|_)", re.I)
SECRET_VALUE_RE = re.compile(r"(api[_-]?key|secret|password|token=|bearer |private[_-]?key)", re.I)
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}
REQUIRED_FIELDS = (
    "schema_version",
    "config_kind",
    "execution_mode",
    "required_approval_text",
    "user_approval_text",
    "manifest_path",
    "checkpoint_path",
    "approved_eval_root",
    "write_eval_artifact",
    "device",
    "cuda_device_index",
    "max_samples",
    "batch_size",
    "max_image_size",
    "threshold_tau",
    "no_download",
    "no_network",
    "no_training",
    "no_checkpoint_writes",
    "no_sns_augmentation",
    "result_scope",
)


class EvaluationError(ValueError):
    """Raised when pre-SNS evaluation inputs are invalid."""


def _err(message: str) -> str:
    return f"- {message}"


def _real(path: str | Path) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def _is_under(path: str | Path, root: str | Path) -> bool:
    path_real = _real(path)
    root_real = _real(root)
    try:
        return os.path.commonpath([str(path_real), str(root_real)]) == str(root_real)
    except ValueError:
        return False


def _inside_repo(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT)


def _repo_outputs_or_checkpoints(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT / "outputs") or _is_under(path, REPO_ROOT / "checkpoints")


def _has_path_traversal(value: str) -> bool:
    return any(part == ".." for part in value.replace("\\", "/").split("/"))


def _contains_protected_part(value: str, allowed_parts: set[str] | None = None) -> bool:
    allowed_parts = allowed_parts or set()
    return any(
        part in PROTECTED_PARTS and part not in allowed_parts
        for part in value.replace("\\", "/").split("/")
        if part
    )


def _walk_safety(value: Any, path: str = "", allowed_abs_values: set[str] | None = None) -> list[str]:
    allowed_abs_values = allowed_abs_values or set()
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if SECRET_KEY_RE.search(key_text):
                errors.append(_err(f"{child_path}: secret-like key rejected"))
            errors.extend(_walk_safety(child, child_path, allowed_abs_values))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_walk_safety(child, f"{path}[{index}]", allowed_abs_values))
    elif isinstance(value, str):
        text = value.strip()
        if REMOTE_RE.search(text):
            errors.append(_err(f"{path}: URL or remote scheme rejected"))
        if WINDOWS_DRIVE_RE.search(text):
            errors.append(_err(f"{path}: Windows drive path rejected"))
        if text in allowed_abs_values:
            if _has_path_traversal(text):
                errors.append(_err(f"{path}: path traversal rejected"))
            if _repo_outputs_or_checkpoints(text):
                errors.append(_err(f"{path}: repository outputs/checkpoints path rejected"))
            if _contains_protected_part(text):
                errors.append(_err(f"{path}: protected path segment rejected"))
            if SECRET_VALUE_RE.search(text):
                errors.append(_err(f"{path}: secret-like value rejected"))
            return errors
        if text.startswith("/"):
            errors.append(_err(f"{path}: absolute path rejected"))
        if _has_path_traversal(text):
            errors.append(_err(f"{path}: path traversal rejected"))
        if _contains_protected_part(text):
            errors.append(_err(f"{path}: protected path segment rejected"))
        if SECRET_VALUE_RE.search(text):
            errors.append(_err(f"{path}: secret-like value rejected"))
    return errors


def load_evaluation_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise EvaluationError("evaluation config root must be a JSON object")
    return raw


def _validate_explicit_file(path_value: Any, field: str, require_exists: bool) -> list[str]:
    errors: list[str] = []
    if not isinstance(path_value, str) or not path_value.strip():
        return [_err(f"{field} must be a non-empty local file path")]
    text = path_value.strip()
    if REMOTE_RE.search(text):
        errors.append(_err(f"{field} must not be a URL or remote scheme"))
    if WINDOWS_DRIVE_RE.search(text):
        errors.append(_err(f"{field} must not be a Windows drive path"))
    if not text.startswith("/"):
        errors.append(_err(f"{field} must be absolute in approved local mode"))
    if _has_path_traversal(text):
        errors.append(_err(f"{field} must not contain path traversal"))
    if _contains_protected_part(text):
        errors.append(_err(f"{field} must not contain protected path segments"))
    if _repo_outputs_or_checkpoints(text):
        errors.append(_err(f"{field} must not be inside repository outputs/checkpoints"))
    if require_exists and not os.path.isfile(text):
        errors.append(_err(f"{field} must exist as a file"))
    return errors


def _nearest_existing_parent(path_value: str) -> Path:
    current = Path(path_value)
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def validate_eval_root(path_value: Any, require_parent: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(path_value, str) or not path_value.strip():
        return [_err("approved_eval_root must be a non-empty local directory path")]
    text = path_value.strip()
    if REMOTE_RE.search(text):
        errors.append(_err("approved_eval_root must not be a URL or remote scheme"))
    if WINDOWS_DRIVE_RE.search(text):
        errors.append(_err("approved_eval_root must not be a Windows drive path"))
    if not text.startswith("/"):
        errors.append(_err("approved_eval_root must be absolute in approved local mode"))
    if _has_path_traversal(text):
        errors.append(_err("approved_eval_root must not contain path traversal"))
    if _contains_protected_part(text):
        errors.append(_err("approved_eval_root must not contain protected path segments"))
    if text.startswith("/") and _inside_repo(text):
        errors.append(_err("approved_eval_root must be outside the repository"))
    if _repo_outputs_or_checkpoints(text):
        errors.append(_err("approved_eval_root must not be inside repository outputs/checkpoints"))
    if require_parent:
        if os.path.exists(text) and not os.path.isdir(text):
            errors.append(_err("approved_eval_root exists but is not a directory"))
        ancestor = _nearest_existing_parent(os.path.dirname(text.rstrip(os.sep)) or os.sep)
        if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
            errors.append(_err("approved_eval_root parent must be valid or creatable"))
    return errors


def validate_evaluation_config(raw: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    kind = raw.get("config_kind")
    if kind not in CONFIG_KINDS:
        errors.append(_err("config_kind must be example_symbolic or approved_pre_sns_evaluation"))
        errors.extend(_walk_safety(raw))
        return errors
    for flag in ("no_download", "no_network", "no_training", "no_checkpoint_writes", "no_sns_augmentation"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if raw.get("required_approval_text") != APPROVAL_TEXT:
        errors.append(_err("required_approval_text must document the exact approval phrase"))
    if raw.get("device") not in {"cpu", "cuda"}:
        errors.append(_err("device must be cpu or cuda"))
    if not isinstance(raw.get("cuda_device_index"), int) or isinstance(raw.get("cuda_device_index"), bool) or raw.get("cuda_device_index", 0) < 0:
        errors.append(_err("cuda_device_index must be a non-negative integer"))
    for field in ("max_samples", "batch_size", "max_image_size"):
        value = raw.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(_err(f"{field} must be a positive integer"))
    if isinstance(raw.get("max_image_size"), int) and raw["max_image_size"] > 512:
        errors.append(_err("max_image_size must be <= 512"))
    tau = raw.get("threshold_tau")
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not 0.0 <= float(tau) <= 1.0:
        errors.append(_err("threshold_tau must be a number in [0.0, 1.0]"))
    if raw.get("recursive_scan") is True or raw.get("recursive_directory_scan") is True:
        errors.append(_err("recursive scan flags are rejected"))
    if kind == "example_symbolic":
        if raw.get("execution_mode") != "example_only":
            errors.append(_err("execution_mode must be example_only"))
        if raw.get("user_approval_text") not in {"", APPROVAL_TEXT}:
            errors.append(_err("example user_approval_text must be empty or the documented approval phrase"))
        errors.extend(_walk_safety(raw))
        return errors

    if raw.get("execution_mode") != "approved_local_pre_sns_evaluation":
        errors.append(_err("execution_mode must be approved_local_pre_sns_evaluation"))
    if raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(_err("user_approval_text does not match required approval phrase"))
    if raw.get("write_eval_artifact") not in {True, False}:
        errors.append(_err("write_eval_artifact must be boolean"))
    allowed_abs = {
        value
        for value in (raw.get("manifest_path"), raw.get("checkpoint_path"), raw.get("approved_eval_root"))
        if isinstance(value, str)
    }
    errors.extend(_walk_safety(raw, allowed_abs_values=allowed_abs))
    errors.extend(_validate_explicit_file(raw.get("manifest_path"), "manifest_path", require_exists=True))
    errors.extend(_validate_explicit_file(raw.get("checkpoint_path"), "checkpoint_path", require_exists=True))
    errors.extend(validate_eval_root(raw.get("approved_eval_root"), require_parent=True))
    return errors


def load_approved_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise EvaluationError("manifest root must be a JSON object")
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise EvaluationError("manifest must contain a samples list")
    return manifest


def validate_manifest_samples(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    samples = manifest.get("samples", [])
    if not isinstance(samples, list) or not samples:
        return [_err("manifest samples must be a non-empty list")]
    allowed_abs: set[str] = set()
    for sample in samples:
        if isinstance(sample, dict):
            for field in ("image_path", "mask_path"):
                value = sample.get(field)
                if isinstance(value, str):
                    allowed_abs.add(value)
    errors.extend(_walk_safety(samples, "samples", allowed_abs_values=allowed_abs))
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            errors.append(_err(f"samples[{index}] must be an object"))
            continue
        image_path = sample.get("image_path")
        errors.extend(_validate_explicit_file(image_path, f"samples[{index}].image_path", require_exists=True))
        label = sample.get("class_label")
        if label not in CLASS_LABELS:
            errors.append(_err(f"samples[{index}].class_label must be real, full_synthetic, or tampered"))
        family = sample.get("family_label")
        if family is not None and family not in FAMILY_SMOKE_LABELS:
            errors.append(_err(f"samples[{index}].family_label is invalid"))
        mask_path = sample.get("mask_path")
        if mask_path is not None:
            errors.extend(_validate_explicit_file(mask_path, f"samples[{index}].mask_path", require_exists=True))
    return errors


def select_samples(manifest: dict[str, Any], max_samples: int) -> list[dict[str, Any]]:
    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        raise EvaluationError("manifest samples must be a list")
    return [sample for sample in samples[:max_samples] if isinstance(sample, dict)]


def accuracy_score(y_true: list[str], y_pred: list[str]) -> float | None:
    if not y_true:
        return None
    return sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)


def macro_f1_score(y_true: list[str], y_pred: list[str], labels: tuple[str, ...] = CLASS_LABELS) -> float | None:
    if not y_true:
        return None
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append((2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def family_accuracy_score(y_true: list[str | None], y_pred: list[str]) -> float | None:
    pairs = [(true, pred) for true, pred in zip(y_true, y_pred) if true]
    if not pairs:
        return None
    return sum(1 for true, pred in pairs if true == pred) / len(pairs)


def binary_mask_iou(pred_mask: Any, true_mask: Any) -> float | None:
    pred_flat = _flatten_binary_mask(pred_mask)
    true_flat = _flatten_binary_mask(true_mask)
    if len(pred_flat) != len(true_flat) or not pred_flat:
        return None
    intersection = sum(1 for pred, true in zip(pred_flat, true_flat) if pred and true)
    union = sum(1 for pred, true in zip(pred_flat, true_flat) if pred or true)
    return 1.0 if union == 0 else intersection / union


def _flatten_binary_mask(mask: Any) -> list[bool]:
    if not isinstance(mask, list):
        return []
    flat: list[bool] = []
    for row in mask:
        if isinstance(row, list):
            flat.extend(bool(value) for value in row)
        else:
            flat.append(bool(row))
    return flat


def mean_mask_iou(pred_masks: list[Any], true_masks: list[Any]) -> float | None:
    scores = [score for score in (binary_mask_iou(pred, true) for pred, true in zip(pred_masks, true_masks)) if score is not None]
    return (sum(scores) / len(scores)) if scores else None


def localization_activation_recall_score(class_labels: list[str], localization_states: list[str]) -> float | None:
    tampered_states = [state for label, state in zip(class_labels, localization_states) if label == "tampered"]
    if not tampered_states:
        return None
    return sum(1 for state in tampered_states if state == LOCALIZATION_ACTIVATED) / len(tampered_states)


def sample_counts(samples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_dataset: dict[str, int] = {}
    by_class: dict[str, int] = {}
    for sample in samples:
        dataset = str(sample.get("source_dataset", "unknown"))
        label = str(sample.get("class_label", "unknown"))
        by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
        by_class[label] = by_class.get(label, 0) + 1
    return {"by_dataset": by_dataset, "by_class": by_class}


def _load_mask_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_evaluation(raw: dict[str, Any]) -> dict[str, Any]:
    errors = validate_evaluation_config(raw)
    if errors:
        raise EvaluationError("evaluation config validation failed:\n" + "\n".join(errors))
    if raw.get("config_kind") not in {APPROVED_CONFIG_KIND, *LEGACY_APPROVED_CONFIG_KINDS}:
        raise EvaluationError("evaluation requires approved_pre_sns_evaluation config")
    torch, Image = _runtime_deps()
    device = select_device(torch, raw)
    manifest = load_approved_manifest(raw["manifest_path"])
    manifest_errors = validate_manifest_samples(manifest)
    if manifest_errors:
        raise EvaluationError("manifest validation failed:\n" + "\n".join(manifest_errors))
    samples = select_samples(manifest, int(raw["max_samples"]))
    if not samples:
        raise EvaluationError("no samples selected for evaluation")
    model, image_size, checkpoint = load_trained_model(torch, raw["checkpoint_path"], int(raw["max_image_size"]), device)
    family_labels = tuple(checkpoint.get("family_labels", list(FAMILY_SMOKE_LABELS)))
    if len(family_labels) != len(FAMILY_SMOKE_LABELS):
        family_labels = FAMILY_SMOKE_LABELS

    y_true: list[str] = []
    y_pred: list[str] = []
    family_true: list[str | None] = []
    family_pred: list[str] = []
    true_masks: list[Any] = []
    pred_masks: list[Any] = []
    loc_true_labels: list[str] = []
    loc_states: list[str] = []
    latencies: list[float] = []

    with torch.no_grad():
        for sample in samples:
            image_tensor = prepare_image_tensor(torch, Image, sample["image_path"], image_size, device)
            started = time.perf_counter()
            outputs = model(image_tensor)
            class_probs = torch.softmax(outputs["class_logits"][0], dim=0).detach().cpu().tolist()
            family_probs = torch.softmax(outputs["family_logits"][0], dim=0).detach().cpu().tolist()
            mask_probs = torch.sigmoid(outputs["localization_logits"][0]).detach().cpu()
            latency_ms = max((time.perf_counter() - started) * 1000.0, 0.000001)
            latencies.append(float(latency_ms))
            class_conf = confidence_map(CLASS_LABELS, class_probs)
            pred_label = CLASS_LABELS[int(max(range(len(class_probs)), key=lambda idx: class_probs[idx]))]
            pred_family = family_labels[int(max(range(len(family_probs)), key=lambda idx: family_probs[idx]))]
            tampered_score = float(class_conf["tampered"])
            loc_state, _area = localization_summary(tampered_score, float(raw["threshold_tau"]), mask_probs, torch)
            y_true.append(sample["class_label"])
            y_pred.append(pred_label)
            family_true.append(sample.get("family_label"))
            family_pred.append(pred_family)
            loc_true_labels.append(sample["class_label"])
            loc_states.append(loc_state)
            if sample.get("class_label") == "tampered" and sample.get("mask_path"):
                true_masks.append(_load_mask_json(sample["mask_path"]))
                pred_masks.append(mask_probs.ge(0.5).int().reshape(image_size, image_size).tolist())

    latency_mean = sum(latencies) / len(latencies)
    result = {
        "marker": MARKER,
        "schema_version": raw.get("schema_version", "1.0"),
        "accuracy_3way": accuracy_score(y_true, y_pred),
        "macro_f1_3way": macro_f1_score(y_true, y_pred),
        "family_accuracy": family_accuracy_score(family_true, family_pred),
        "mask_iou_mean": mean_mask_iou(pred_masks, true_masks),
        "localization_activation_recall": localization_activation_recall_score(loc_true_labels, loc_states),
        "latency_ms_mean": float(latency_mean),
        "fps_estimate": float(1000.0 / latency_mean) if latency_mean > 0 else None,
        "samples_evaluated": len(samples),
        "sample_counts": sample_counts(samples),
        "write_eval_artifact": bool(raw.get("write_eval_artifact")),
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
        "result_scope": raw.get("result_scope", "pre-SNS evaluation; not final full-dataset performance"),
    }
    if raw.get("write_eval_artifact") is True:
        result["evaluation_artifact_path"] = str(write_evaluation_artifact(raw["approved_eval_root"], result))
    return _json_safe(result)


def write_evaluation_artifact(eval_root: str | Path, result: dict[str, Any]) -> Path:
    errors = validate_eval_root(str(eval_root), require_parent=True)
    if errors:
        raise EvaluationError("\n".join(errors))
    root = _real(eval_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "evaluation_summary.json"
    tmp = root / ".evaluation_summary.json.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)
    return path


def _runtime_deps():
    try:
        import torch
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("torch and PIL are required for pre-SNS evaluation") from exc
    return torch, Image


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value
