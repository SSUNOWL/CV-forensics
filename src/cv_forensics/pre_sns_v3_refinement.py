"""Guarded pre-SNS v3 hard-case refinement."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from .pre_sns_v3_metrics import DEFAULT_TAU_VALUES, compute_metrics
from .pre_sns_v3_model import CLASS_LABELS, FAMILY_LABELS
from .pre_sns_v3_training import (
    _batch_from_samples,
    _err,
    _has_remote_scheme,
    _has_traversal,
    _inside_repo,
    _is_under,
    _runtime_deps,
    _safe_dirs,
    _sample_class,
    _sample_family,
    class_weights,
    compute_v3_losses,
    is_protected_path,
    is_secret_like,
    load_json,
    load_manifest,
    validate_manifest_sample_paths,
    write_json,
)
from .pre_sns_v3_model import build_pre_sns_v3_model

CONFIG_OK_MARKER = "PRE_SNS_V3_REFINEMENT_CONFIG_OK"
RUN_OK_MARKER = "PRE_SNS_V3_REFINEMENT_RUN_OK"
DRY_RUN_OK_MARKER = "PRE_SNS_V3_REFINEMENT_DRY_RUN_OK"
APPROVAL_TEXT = "I_APPROVE_PRE_SNS_V3_REFINEMENT"
APPROVED_KIND = "approved_pre_sns_v3_refinement"
APPROVED_MODE = "approved_local_pre_sns_v3_refinement"
EXAMPLE_KIND = "example_symbolic"

HARD_CASE_KEYS = (
    "hard_negative_real",
    "hard_negative_non_tampered",
    "hard_positive_tampered_low_iou",
    "class_mask_inconsistent_cases",
)

HARD_CASE_PATH_FIELDS = {
    "hard_negative_real": "hard_negative_real_path",
    "hard_negative_non_tampered": "hard_negative_non_tampered_path",
    "hard_positive_tampered_low_iou": "hard_positive_tampered_low_iou_path",
    "class_mask_inconsistent_cases": "class_mask_inconsistent_cases_path",
}

COUNT_FIELDS = {
    "hard_negative_real": "hard_negative_real_count_used",
    "hard_negative_non_tampered": "hard_negative_non_tampered_count_used",
    "class_mask_inconsistent_cases": "hard_inconsistent_count_used",
    "hard_positive_tampered_low_iou": "hard_positive_low_iou_count_used",
}


class PreSnsV3RefinementError(ValueError):
    """Raised when refinement guardrails or runtime checks fail."""


def load_config(path: str | Path) -> dict[str, Any]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise PreSnsV3RefinementError("config root must be a JSON object")
    return raw


def default_loss_weights() -> dict[str, float]:
    return {
        "class_loss_weight": 1.0,
        "tamper_binary_loss_weight": 1.5,
        "family_loss_weight": 0.2,
        "localization_loss_weight": 12.0,
        "non_tampered_empty_mask_loss_weight": 0.8,
        "dice_loss_weight": 1.0,
    }


def default_oversample_weights() -> dict[str, int]:
    return {
        "hard_negative_real_oversample_weight": 5,
        "hard_negative_non_tampered_oversample_weight": 4,
        "hard_inconsistent_oversample_weight": 3,
        "hard_positive_low_iou_oversample_weight": 3,
    }


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


def _path_parts(value: str) -> list[str]:
    return [part for part in value.replace("\\", "/").split("/") if part]


def _looks_like_val_hard_cases(value: str) -> bool:
    lower = value.lower().replace("\\", "/")
    parts = _path_parts(lower)
    if "hard_mining_full_repaired" in lower and any(part in {"val", "valid", "validation"} or part.startswith("val_") for part in parts):
        return True
    if any(part in {"val", "valid", "validation"} for part in parts) and "hard_mining" in lower:
        return True
    if "hard_mining_val" in lower or "val_hard_mining" in lower or "validation_hard_mining" in lower:
        return True
    return False


def _validate_path_text(
    value: Any,
    field: str,
    approved: bool,
    approved_roots: list[str] | None = None,
    require_file: bool = False,
    output_root: bool = False,
    allow_checkpoint_part: bool = False,
) -> list[str]:
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
    protected = is_protected_path(text)
    if protected and not allow_checkpoint_part:
        errors.append(_err(f"{field} must not reference protected paths"))
    if _looks_like_val_hard_cases(text):
        errors.append(_err(f"{field} must not reference validation hard-mining outputs"))
    if approved:
        if not os.path.isabs(text):
            errors.append(_err(f"{field} must be absolute in approved local mode"))
        if output_root and os.path.isabs(text) and _inside_repo(text):
            errors.append(_err(f"{field} must be outside the repository"))
        if approved_roots and os.path.isabs(text) and not any(_is_under(text, root) for root in approved_roots):
            errors.append(_err(f"{field} must be under an approved root"))
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
            child = f"{path}.{key}" if path else str(key)
            if is_secret_like(str(key)):
                errors.append(_err(f"{child}: secret-like key rejected"))
            errors.extend(_walk_string_safety(value, child, allowed_absolute))
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
        if is_protected_path(text) and "base_checkpoint_path" not in path and "approved_checkpoint_roots" not in path:
            errors.append(_err(f"{path}: protected path segment rejected"))
        if _looks_like_val_hard_cases(text):
            errors.append(_err(f"{path}: validation hard-mining output rejected"))
    return errors


def validate_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(raw, dict):
        return [_err("config root must be a JSON object")]
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "required_approval_text",
        "user_approval_text",
        "hard_cases_split",
        "train_manifest_path",
        "val_manifest_path",
        "base_checkpoint_path",
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
    for field in HARD_CASE_PATH_FIELDS.values():
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    approved = raw.get("config_kind") == APPROVED_KIND
    if raw.get("config_kind") not in {EXAMPLE_KIND, APPROVED_KIND}:
        errors.append(_err("config_kind must be example_symbolic or approved_pre_sns_v3_refinement"))
    if raw.get("execution_mode") != (APPROVED_MODE if approved else "example_only"):
        errors.append(_err(f"execution_mode must be {APPROVED_MODE if approved else 'example_only'}"))
    if raw.get("required_approval_text") != APPROVAL_TEXT:
        errors.append(_err("required_approval_text must document the approval phrase"))
    if approved and raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(_err("user_approval_text must match required approval phrase"))
    if not approved and raw.get("user_approval_text") not in {"", None}:
        errors.append(_err("example user_approval_text must be empty"))
    if raw.get("hard_cases_split") != "train":
        errors.append(_err("hard_cases_split must be train"))
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
    if raw.get("max_image_size") not in {16, 32, 64, 128, 224, 256}:
        errors.append(_err("max_image_size must be 224 or 256 for real runs; small fixture sizes are accepted for tests"))
    for key in default_loss_weights():
        if key in raw:
            _nonnegative_float(raw, key, 1000.0, errors)
        elif approved:
            errors.append(_err(f"{key} is required"))
    for key in default_oversample_weights():
        if key in raw:
            _positive_int(raw, key, 1000, errors)
        elif approved:
            errors.append(_err(f"{key} is required"))
    roots = raw.get("approved_input_roots")
    if not isinstance(roots, list) or not all(isinstance(root, str) and root.strip() for root in roots):
        errors.append(_err("approved_input_roots must be a list of non-empty paths"))
        roots = []
    checkpoint_roots = raw.get("approved_checkpoint_roots", [])
    if checkpoint_roots is None:
        checkpoint_roots = []
    if not isinstance(checkpoint_roots, list) or not all(isinstance(root, str) and root.strip() for root in checkpoint_roots):
        errors.append(_err("approved_checkpoint_roots must be a list of non-empty paths when present"))
        checkpoint_roots = []
    for index, root in enumerate(roots):
        errors.extend(_validate_path_text(root, f"approved_input_roots[{index}]", approved, output_root=True))
        if approved and os.path.isabs(root) and _inside_repo(root):
            errors.append(_err(f"approved_input_roots[{index}] must be outside the repository"))
    for index, root in enumerate(checkpoint_roots):
        errors.extend(_validate_path_text(root, f"approved_checkpoint_roots[{index}]", approved, output_root=True, allow_checkpoint_part=True))
        if approved and os.path.isabs(root) and _inside_repo(root):
            errors.append(_err(f"approved_checkpoint_roots[{index}] must be outside the repository"))
    input_roots = [root for root in roots if isinstance(root, str)]
    base_roots = input_roots + [root for root in checkpoint_roots if isinstance(root, str)]
    errors.extend(_validate_path_text(raw.get("train_manifest_path"), "train_manifest_path", approved, input_roots, require_exists))
    errors.extend(_validate_path_text(raw.get("val_manifest_path"), "val_manifest_path", approved, input_roots, require_exists))
    for key, field in HARD_CASE_PATH_FIELDS.items():
        errors.extend(_validate_path_text(raw.get(field), field, approved, input_roots, require_exists))
    errors.extend(_validate_path_text(raw.get("base_checkpoint_path"), "base_checkpoint_path", approved, base_roots, require_exists, allow_checkpoint_part=True))
    errors.extend(_validate_path_text(raw.get("approved_run_root"), "approved_run_root", approved, output_root=True))
    errors.extend(_validate_path_text(raw.get("approved_checkpoint_root"), "approved_checkpoint_root", approved, output_root=True, allow_checkpoint_part=True))
    allowed_abs: set[str] = set()
    if approved:
        for field in (
            "train_manifest_path",
            "val_manifest_path",
            "base_checkpoint_path",
            "approved_run_root",
            "approved_checkpoint_root",
            *HARD_CASE_PATH_FIELDS.values(),
        ):
            if isinstance(raw.get(field), str):
                allowed_abs.add(raw[field])
        allowed_abs.update(input_roots)
        allowed_abs.update(root for root in checkpoint_roots if isinstance(root, str))
    errors.extend(_walk_string_safety(raw, allowed_absolute=allowed_abs))
    return errors


def assert_valid_config(raw: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_config(raw, require_exists)
    if errors:
        raise PreSnsV3RefinementError("\n".join(errors))


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    return (str(record.get("sample_id") or ""), str(record.get("image_path") or ""))


def _sample_keys(sample: dict[str, Any]) -> set[tuple[str, str]]:
    sample_id = str(sample.get("sample_id") or sample.get("id") or "")
    image_path = str(sample.get("image_path") or "")
    return {(sample_id, image_path), (sample_id, ""), ("", image_path)}


def load_hard_case_records(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for key, field in HARD_CASE_PATH_FIELDS.items():
        path = config.get(field)
        if not path or not os.path.isfile(str(path)):
            records[key] = []
            continue
        raw = load_json(path)
        if not isinstance(raw, list):
            raise PreSnsV3RefinementError(f"{field} must contain a JSON list")
        records[key] = [row for row in raw if isinstance(row, dict)]
    return records


def map_hard_cases_to_train(
    train_samples: list[dict[str, Any]],
    hard_cases: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in train_samples:
        for key in _sample_keys(sample):
            if key != ("", ""):
                by_key[key] = sample
    matched: dict[str, list[dict[str, Any]]] = {key: [] for key in HARD_CASE_KEYS}
    dropped: dict[str, int] = {key: 0 for key in HARD_CASE_KEYS}
    seen: dict[str, set[int]] = {key: set() for key in HARD_CASE_KEYS}
    for key in HARD_CASE_KEYS:
        for record in hard_cases.get(key, []):
            sample = by_key.get(_record_key(record)) or by_key.get((str(record.get("sample_id") or ""), "")) or by_key.get(("", str(record.get("image_path") or "")))
            if sample is None:
                dropped[key] += 1
                continue
            identity = id(sample)
            if identity in seen[key]:
                continue
            seen[key].add(identity)
            matched[key].append(sample)
    return matched, dropped


def build_oversampled_train_samples(
    train_samples: list[dict[str, Any]],
    matched_hard_cases: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    weights = {**default_oversample_weights(), **{key: config[key] for key in default_oversample_weights() if key in config}}
    expanded = list(train_samples)
    usage_counts = {
        "hard_negative_real_count_used": len(matched_hard_cases.get("hard_negative_real", [])),
        "hard_negative_non_tampered_count_used": len(matched_hard_cases.get("hard_negative_non_tampered", [])),
        "hard_inconsistent_count_used": len(matched_hard_cases.get("class_mask_inconsistent_cases", [])),
        "hard_positive_low_iou_count_used": len(matched_hard_cases.get("hard_positive_tampered_low_iou", [])),
    }
    mapping = (
        ("hard_negative_real", "hard_negative_real_oversample_weight"),
        ("hard_negative_non_tampered", "hard_negative_non_tampered_oversample_weight"),
        ("class_mask_inconsistent_cases", "hard_inconsistent_oversample_weight"),
        ("hard_positive_tampered_low_iou", "hard_positive_low_iou_oversample_weight"),
    )
    for case_key, weight_key in mapping:
        repetitions = max(0, int(weights[weight_key]) - 1)
        for sample in matched_hard_cases.get(case_key, []):
            expanded.extend([sample] * repetitions)
    non_tampered = [sample for sample in expanded if _sample_class(sample) != "tampered"]
    proxy_den = len(matched_hard_cases.get("class_mask_inconsistent_cases", [])) + len(train_samples)
    summary = {
        **usage_counts,
        "base_train_sample_count": len(train_samples),
        "oversampled_train_sample_count": len(expanded),
        "oversample_weights": weights,
        "class_mask_inconsistency_proxy": len(matched_hard_cases.get("class_mask_inconsistent_cases", [])) / proxy_den if proxy_den else 0.0,
        "non_tampered_mask_activation_rate": None,
        "non_tampered_oversampled_sample_count": len(non_tampered),
    }
    return expanded, summary


def _predictions_from_samples(samples: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    predictions: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        gt = _sample_class(sample)
        pred = gt if index % 6 else ("tampered" if gt != "tampered" else "real")
        base_score = {"real": 0.10, "full_synthetic": 0.18, "tampered": 0.78}[gt]
        score = max(0.0, min(1.0, base_score + rng.uniform(-0.04, 0.04)))
        mask_area = 4.0 if gt == "tampered" else (1.0 if pred == "tampered" else 0.0)
        predictions.append(
            {
                "gt_class": gt,
                "pred_class": pred,
                "tampered_score": score,
                "gt_family": _sample_family(sample),
                "pred_family": _sample_family(sample) or "Real-or-N/A",
                "localization_iou": 0.55 if gt == "tampered" else None,
                "mask_area_pct": mask_area,
                "source_dataset": sample.get("source_dataset", sample.get("dataset", "fixture")),
            }
        )
    return predictions


def _add_refinement_metrics(metrics: dict[str, Any], hard_summary: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(metrics)
    for field in (
        "hard_negative_real_count_used",
        "hard_negative_non_tampered_count_used",
        "hard_inconsistent_count_used",
        "hard_positive_low_iou_count_used",
        "class_mask_inconsistency_proxy",
        "non_tampered_mask_activation_rate",
    ):
        enriched[field] = hard_summary.get(field)
    return enriched


def _hard_summary_with_prediction_metrics(hard_summary: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    enriched = dict(hard_summary)
    non_tampered = [row for row in predictions if row.get("gt_class") != "tampered"]
    if non_tampered:
        enriched["non_tampered_mask_activation_rate"] = sum(1 for row in non_tampered if float(row.get("mask_area_pct") or 0.0) > 0.0) / len(non_tampered)
    return enriched


def _write_refinement_artifacts(config: dict[str, Any], summary: dict[str, Any], model: Any | None = None) -> dict[str, Any]:
    run_root, ckpt_root = _safe_dirs(config)
    best = ckpt_root / "best_checkpoint.pt"
    latest = ckpt_root / "latest_checkpoint.pt"
    if model is not None:
        import torch

        checkpoint = {
            "model_name": "pre_sns_v3_refinement",
            "model_state_dict": model.state_dict(),
            "config": config,
            "image_size": int(config["max_image_size"]),
            "class_labels": list(CLASS_LABELS),
            "family_labels": list(FAMILY_LABELS),
            "selected_tau": summary["selected_tau"],
        }
        torch.save(checkpoint, best)
        torch.save(checkpoint, latest)
    else:
        write_json(best, {"model_name": "pre_sns_v3_refinement_fixture", "selected_tau": summary["selected_tau"]})
        write_json(latest, {"model_name": "pre_sns_v3_refinement_fixture", "selected_tau": summary["selected_tau"]})
    write_json(run_root / "val_metrics.json", summary["val_metrics"])
    write_json(run_root / "threshold_calibration.json", {"selected_tau": summary["selected_tau"], "threshold_sweep": summary["val_metrics"]["threshold_sweep"], "fallback_selected_tau": summary["val_metrics"]["fallback_selected_tau"]})
    write_json(run_root / "confusion_matrix.json", summary["val_metrics"]["confusion_matrix"])
    if summary["val_metrics"].get("per_source_dataset_confusion_matrix") is not None:
        write_json(run_root / "per_source_confusion_matrix.json", summary["val_metrics"]["per_source_dataset_confusion_matrix"])
    write_json(run_root / "refinement_hard_case_summary.json", summary["refinement_hard_case_summary"])
    write_json(run_root / "config_snapshot.json", config)
    with open(run_root / "train_metrics.jsonl", "w", encoding="utf-8") as handle:
        for row in summary["train_metrics"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "artifact_kind": "pre_sns_v3_refinement",
        "files": {
            "run_summary": str(run_root / "run_summary.json"),
            "val_metrics": str(run_root / "val_metrics.json"),
            "threshold_calibration": str(run_root / "threshold_calibration.json"),
            "confusion_matrix": str(run_root / "confusion_matrix.json"),
            "per_source_confusion_matrix": str(run_root / "per_source_confusion_matrix.json"),
            "refinement_hard_case_summary": str(run_root / "refinement_hard_case_summary.json"),
            "train_metrics": str(run_root / "train_metrics.jsonl"),
            "artifact_manifest": str(run_root / "artifact_manifest.json"),
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
            "refinement_hard_case_summary_path": str(run_root / "refinement_hard_case_summary.json"),
            "best_checkpoint_path": str(best),
            "latest_checkpoint_path": str(latest),
        }
    )
    write_json(run_root / "run_summary.json", summary)
    return summary


def _load_base_checkpoint(torch: Any, model: Any, config: dict[str, Any]) -> None:
    checkpoint = torch.load(config["base_checkpoint_path"], map_location=config["device"])
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise PreSnsV3RefinementError("base checkpoint does not contain a model state dict")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise PreSnsV3RefinementError(f"base checkpoint has unexpected keys: {unexpected[:5]}")
    if len(missing) > 20:
        raise PreSnsV3RefinementError("base checkpoint is not compatible with pre-SNS v3 model")


def _actual_refine(config: dict[str, Any], train_samples: list[dict[str, Any]], val_samples: list[dict[str, Any]], hard_summary: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    torch, Image = _runtime_deps()
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise PreSnsV3RefinementError("device=cuda was requested but CUDA is unavailable")
    device = "cuda" if config["device"] == "cuda" else "cpu"
    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))
    train_subset = list(train_samples)
    val_subset = val_samples[: int(config["max_samples_val"])]
    validate_manifest_sample_paths(train_subset + val_subset, config["approved_input_roots"])
    model = build_pre_sns_v3_model(torch, int(config.get("base_channels", 12))).to(device)
    _load_base_checkpoint(torch, model, {**config, "device": device})
    image_size = int(config["max_image_size"])
    weights = class_weights(train_subset) if config.get("class_balance_strategy", "loss") == "loss" else {label: 1.0 for label in CLASS_LABELS}
    class_weight_tensor = torch.tensor([weights[label] for label in CLASS_LABELS], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0001)))
    loss_weights = {key: float(config.get(key, value)) for key, value in default_loss_weights().items()}
    batch_size = int(config["batch_size"])
    train_metrics: list[dict[str, Any]] = []
    steps = 0
    initial_total_loss: float | None = None
    final_total_loss: float | None = None
    start = time.time()
    for epoch in range(int(config["epochs"])):
        random.Random(int(config["seed"]) + epoch).shuffle(train_subset)
        epoch_losses: list[float] = []
        for offset in range(0, len(train_subset), batch_size):
            batch_samples = train_subset[offset: offset + batch_size]
            batch = _batch_from_samples(torch, Image, batch_samples, image_size, device, class_weight_tensor)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["images"])
            losses = compute_v3_losses(torch, outputs, batch, loss_weights)
            losses["total_loss"].backward()
            if config.get("gradient_clip_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
            steps += 1
            loss_value = float(losses["total_loss"].detach().cpu().item())
            initial_total_loss = loss_value if initial_total_loss is None else initial_total_loss
            final_total_loss = loss_value
            epoch_losses.append(loss_value)
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
            pred_mask = (torch.sigmoid(outputs["localization_logits"][0]).detach().cpu() >= 0.5).float()
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
    hard_summary = _hard_summary_with_prediction_metrics(hard_summary, predictions)
    metrics = _add_refinement_metrics(compute_metrics(predictions, float(config.get("max_false_activation_rate", 0.20)), config.get("threshold_tau_values", DEFAULT_TAU_VALUES)), hard_summary)
    summary = {
        "marker": RUN_OK_MARKER,
        "device": device,
        "training_started": True,
        "training_completed": True,
        "epochs_completed": int(config["epochs"]),
        "steps_completed": steps,
        "train_samples_seen": len(train_subset) * int(config["epochs"]),
        "val_samples_seen": len(val_subset),
        "initial_total_loss": initial_total_loss,
        "final_total_loss": final_total_loss,
        **{key: metrics[key] for key in ("class_accuracy", "class_macro_f1", "real_recall", "tampered_precision", "tampered_recall", "tampered_f1", "family_accuracy", "localization_mean_iou", "localization_median_iou", "selected_tau", "false_activation_rate", "localization_activation_recall")},
        **{key: metrics[key] for key in COUNT_FIELDS.values()},
        "class_mask_inconsistency_proxy": metrics["class_mask_inconsistency_proxy"],
        "non_tampered_mask_activation_rate": metrics["non_tampered_mask_activation_rate"],
        "no_write_dry_run": False,
        "artifact_writes": True,
        "checkpoint_writes": True,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation": True,
        "approved_run_root": config["approved_run_root"],
        "approved_checkpoint_root": config["approved_checkpoint_root"],
        "refinement_hard_case_summary": hard_summary,
        "train_metrics": train_metrics,
        "val_metrics": metrics,
        "elapsed_sec": time.time() - start,
    }
    return summary, model


def _dry_or_fixture_refine(config: dict[str, Any], train_samples: list[dict[str, Any]], val_samples: list[dict[str, Any]], hard_summary: dict[str, Any], write_artifacts: bool) -> dict[str, Any]:
    train_subset = list(train_samples)
    val_subset = val_samples[: int(config["max_samples_val"])]
    predictions = _predictions_from_samples(val_subset, int(config["seed"]))
    hard_summary = _hard_summary_with_prediction_metrics(hard_summary, predictions)
    metrics = _add_refinement_metrics(compute_metrics(predictions, float(config.get("max_false_activation_rate", 0.20)), config.get("threshold_tau_values", DEFAULT_TAU_VALUES)), hard_summary)
    summary = {
        "marker": RUN_OK_MARKER if write_artifacts else DRY_RUN_OK_MARKER,
        "device": config.get("device", "cpu"),
        "training_started": True,
        "training_completed": True,
        "epochs_completed": int(config["epochs"]),
        "steps_completed": max(1, math.ceil(len(train_subset) / int(config["batch_size"]))) * int(config["epochs"]),
        "train_samples_seen": len(train_subset) * int(config["epochs"]),
        "val_samples_seen": len(val_subset),
        "initial_total_loss": 1.5,
        "final_total_loss": 0.9,
        **{key: metrics[key] for key in ("class_accuracy", "class_macro_f1", "real_recall", "tampered_precision", "tampered_recall", "tampered_f1", "family_accuracy", "localization_mean_iou", "localization_median_iou", "selected_tau", "false_activation_rate", "localization_activation_recall")},
        **{key: metrics[key] for key in COUNT_FIELDS.values()},
        "class_mask_inconsistency_proxy": metrics["class_mask_inconsistency_proxy"],
        "non_tampered_mask_activation_rate": metrics["non_tampered_mask_activation_rate"],
        "no_write_dry_run": not write_artifacts,
        "artifact_writes": write_artifacts,
        "checkpoint_writes": write_artifacts,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation": True,
        "approved_run_root": config["approved_run_root"],
        "approved_checkpoint_root": config["approved_checkpoint_root"],
        "refinement_hard_case_summary": hard_summary,
        "train_metrics": [{"epoch": epoch + 1, "mean_total_loss": 0.9 / (epoch + 1)} for epoch in range(int(config["epochs"]))],
        "val_metrics": metrics,
    }
    if write_artifacts:
        return _write_refinement_artifacts(config, summary, None)
    return {
        **summary,
        "artifact_manifest_path": None,
        "run_summary_path": None,
        "val_metrics_path": None,
        "threshold_calibration_path": None,
        "confusion_matrix_path": None,
        "refinement_hard_case_summary_path": None,
        "best_checkpoint_path": None,
        "latest_checkpoint_path": None,
    }


def prepare_refinement_samples(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train = load_manifest(config["train_manifest_path"])[: int(config["max_samples_train"])]
    val = load_manifest(config["val_manifest_path"])[: int(config["max_samples_val"])]
    hard_cases = load_hard_case_records(config)
    matched, dropped = map_hard_cases_to_train(train, hard_cases)
    oversampled, hard_summary = build_oversampled_train_samples(train, matched, config)
    hard_summary["dropped_hard_cases_not_in_train"] = dropped
    return oversampled, val, hard_summary


def run_refinement(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_exists=config.get("config_kind") == APPROVED_KIND and config.get("no_write_dry_run") is not True)
    if config.get("hard_cases_split") != "train":
        raise PreSnsV3RefinementError("hard_cases_split must be train")
    train, val, hard_summary = prepare_refinement_samples(config)
    if config.get("no_write_dry_run") is True:
        return _dry_or_fixture_refine(config, train, val, hard_summary, write_artifacts=False)
    if config.get("config_kind") != APPROVED_KIND:
        raise PreSnsV3RefinementError("actual refinement requires approved_pre_sns_v3_refinement")
    summary, model = _actual_refine(config, train, val, hard_summary)
    return _write_refinement_artifacts(config, summary, model)
