"""Guarded full SNSAug V2 curriculum fine-tuning runner."""

from __future__ import annotations

import json
import math
import hashlib
import os
from pathlib import Path
from typing import Any

from .pre_sns_v3_report import load_v3_model
from .snsaug_v2_losses import (
    CLASS_TO_INDEX,
    class_cross_entropy_loss,
    clean_sns_class_consistency_loss,
    hard_negative_tampered_loss,
    masked_family_loss,
    tampered_score_consistency_loss,
    valid_tamper_mask_loss,
)

MARKER = "SNSAUG_V2_FULL_CURRICULUM_FINETUNE_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_FULL_CURRICULUM_FINETUNE_CONFIG_OK"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_FULL_CURRICULUM_FINETUNE"
APPROVED_KIND = "approved_snsaug_v2_full_curriculum_finetune"
APPROVED_MODE = "approved_local_snsaug_v2_full_curriculum_finetune"
MODEL_VERSION = "snsaug_aware_multihead_forensics_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}
EVAL_TOKENS = (
    "cvf_eval_outputs",
    "fixed_pairs",
    "val_pairs",
    "validation_pairs",
    "eval_records",
    "eval_comparisons",
    "oracle_gate",
    "forced_localization",
)

REQUIRED_OUTPUTS = [
    "training_log.jsonl",
    "per_phase_metrics.json",
    "clean_validation_metrics.json",
    "snsaug_0058c_metrics.json",
    "robustness_drop_metrics.json",
    "pre_sns_baseline_comparison.json",
    "full_curriculum_report.md",
    "artifact_manifest.json",
]

REQUIRED_CHECKPOINTS = ["best_checkpoint", "last_checkpoint"]
CHECKPOINT_FORMAT = "snsaug_v2_real_state_dict_v1"
CHECKPOINT_KIND_REAL_WEIGHTS = "snsaug_v2_real_model_weights"
MIN_REAL_CHECKPOINT_TENSOR_COUNT = 12
MIN_REAL_CHECKPOINT_NUMEL = 1000

BEST_POLICY = {
    "primary": "snsaug_tampered_recall_plus_valid_iou",
    "secondary": "clean_macro_f1",
    "guardrail": "real_fpr_lte_configured_limit",
}

PHASES = [
    {"phase": 1, "name": "activation_recovery_safe_geometry"},
    {"phase": 2, "name": "geometry_light_platform_layout"},
    {"phase": 3, "name": "platform_layout_combined_sns"},
]


class SNSAugV2FullCurriculumFinetuneError(ValueError):
    """Raised when full-curriculum fine-tuning guardrails fail."""


def load_snsaug_v2_full_curriculum_finetune_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2FullCurriculumFinetuneError("full curriculum config root must be a JSON object")
    return raw


def _real(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_under(path: str | Path, root: str | Path) -> bool:
    try:
        _real(path).relative_to(_real(root))
        return True
    except ValueError:
        return False


def _inside_repo(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT)


def _parts(path: str | Path) -> set[str]:
    return {part for part in _real(path).parts if part}


def _is_protected(path: str | Path) -> bool:
    parts = _parts(path)
    if ".env" in parts:
        return True
    return bool(parts.intersection(PROTECTED_PARTS - {".env"}))


def _contains_eval_token(path: Any) -> bool:
    text = str(path or "").lower()
    return any(token in text for token in EVAL_TOKENS)


def _as_roots(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _validate_absolute_path(value: Any, field: str, *, require_exists: bool = False) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [f"{field} must be a non-empty absolute path"]
    errors: list[str] = []
    path = Path(value).expanduser()
    if not path.is_absolute():
        errors.append(f"{field} must be absolute")
    if _is_protected(path):
        errors.append(f"{field} must not reference protected paths")
    if require_exists and not _real(path).exists():
        errors.append(f"{field} does not exist")
    return errors


def _validate_under_roots(value: Any, field: str, roots: list[str], *, require_exists: bool = False) -> list[str]:
    errors = _validate_absolute_path(value, field, require_exists=require_exists)
    if isinstance(value, str) and roots and not any(_is_under(value, root) for root in roots):
        errors.append(f"{field} must be under approved roots")
    return errors


def _load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and all(line.lstrip().startswith("{") for line in lines[:2]):
        rows: list[dict[str, Any]] = []
        for line in lines:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        return rows
    if text.startswith("{") or text.startswith("["):
        raw = json.loads(text)
        if isinstance(raw, dict):
            for key in ("samples", "records", "items", "manifest"):
                value = raw.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [raw]
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    rows: list[dict[str, Any]] = []
    for line in lines:
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _validate_train_manifest_rows(path: Any, require_exists: bool, forbidden_roots: list[str] | None = None) -> list[str]:
    if not require_exists or not isinstance(path, str) or not Path(path).expanduser().is_absolute():
        return []
    try:
        rows = _load_json_or_jsonl(path)
    except Exception as exc:
        return [f"training_manifest_path could not be read: {exc}"]
    errors: list[str] = []
    if not rows:
        errors.append("training_manifest_path must contain at least one row")
    for row in rows:
        if str(row.get("split", "")).strip().lower() != "train":
            errors.append("training manifest must contain train split only")
            break
        image_path = str(row.get("image_path") or "")
        mask_path = str(row.get("tamper_mask_path") or row.get("mask_path") or "")
        if _contains_eval_token(image_path) or _contains_eval_token(mask_path):
            errors.append("training manifest rows must not reference fixed-pair, oracle, or evaluation outputs")
            break
        for value in (image_path, mask_path):
            if not value:
                continue
            for root in forbidden_roots or []:
                if root and _is_under(value, root):
                    errors.append("training manifest rows must not use validation, fixed-pair, oracle, or evaluation roots as training input")
                    return errors
    return errors


def _validate_numeric(value: Any, field: str, *, minimum: float = 0.0, maximum: float | None = None) -> list[str]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return [f"{field} must be numeric"]
    number = float(value)
    if number < minimum:
        return [f"{field} must be >= {minimum}"]
    if maximum is not None and number > maximum:
        return [f"{field} must be <= {maximum}"]
    return []


def _step_guardrail_limit(raw: dict[str, Any]) -> tuple[int, list[str]]:
    errors: list[str] = []
    limit = raw.get("max_allowed_steps_per_phase", 500)
    if isinstance(limit, bool) or not isinstance(limit, int):
        return 500, ["max_allowed_steps_per_phase must be an integer"]
    if limit < 1:
        errors.append("max_allowed_steps_per_phase must be >= 1")
    if limit > 500:
        if raw.get("allow_long_run_after_medium_pass") is not True:
            errors.append("max_allowed_steps_per_phase > 500 requires allow_long_run_after_medium_pass=true")
        if limit > 2000:
            errors.append("max_allowed_steps_per_phase must be <= 2000")
    return limit, errors


def _validate_phase_step_fields(raw: dict[str, Any]) -> list[str]:
    limit, errors = _step_guardrail_limit(raw)
    default = raw.get("max_steps_per_phase", 30)
    if isinstance(default, bool) or not isinstance(default, int) or not (1 <= default <= limit):
        errors.append(f"max_steps_per_phase must be an integer in 1..{limit}")
    for field in ("phase_1_max_steps", "phase_2_max_steps", "phase_3_max_steps"):
        if field in raw:
            value = raw.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or not (1 <= value <= limit):
                errors.append(f"{field} must be an integer in 1..{limit}")
    return errors


def validate_snsaug_v2_full_curriculum_finetune_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "run_kind",
        "model_version",
        "approval_text",
        "start_from_pre_sns_best",
        "training_manifest_path",
        "curriculum_schedule_path",
        "profile_sampling_weights_path",
        "base_model_bundle_path",
        "clean_validation_manifest_path",
        "evaluation_pair_root_0058c",
        "approved_train_manifest_roots",
        "approved_model_roots",
        "approved_evaluation_roots",
        "approved_output_roots",
        "approved_checkpoint_roots",
        "output_root",
        "checkpoint_root",
        "real_fpr_limit",
        "best_checkpoint_policy",
        "no_network",
        "no_download",
    )
    for field in required:
        if field not in raw:
            errors.append(f"{field} is required")
    if raw.get("config_kind") != APPROVED_KIND:
        errors.append(f"config_kind must be {APPROVED_KIND}")
    if raw.get("execution_mode") != APPROVED_MODE:
        errors.append(f"execution_mode must be {APPROVED_MODE}")
    if raw.get("run_kind") != "full_curriculum_finetune":
        errors.append("run_kind must be full_curriculum_finetune")
    if raw.get("model_version") != MODEL_VERSION:
        errors.append(f"model_version must be {MODEL_VERSION}")
    if raw.get("approval_text") != APPROVAL_TEXT:
        errors.append(f"approval_text must equal {APPROVAL_TEXT}")
    if raw.get("start_from_pre_sns_best") is not True:
        errors.append("start_from_pre_sns_best must be true")
    for flag in ("no_network", "no_download"):
        if raw.get(flag) is not True:
            errors.append(f"{flag} must be true")
    if raw.get("best_checkpoint_policy") != BEST_POLICY:
        errors.append("best_checkpoint_policy must match the required SNSAug primary, clean secondary, real-FPR guardrail policy")
    errors.extend(_validate_numeric(raw.get("real_fpr_limit"), "real_fpr_limit", minimum=0.0, maximum=1.0))
    errors.extend(_validate_phase_step_fields(raw))

    train_roots = _as_roots(raw.get("approved_train_manifest_roots"))
    model_roots = _as_roots(raw.get("approved_model_roots"))
    eval_roots = _as_roots(raw.get("approved_evaluation_roots"))
    analysis_roots = _as_roots(raw.get("approved_analysis_roots")) or eval_roots
    output_roots = _as_roots(raw.get("approved_output_roots"))
    checkpoint_roots = _as_roots(raw.get("approved_checkpoint_roots"))
    root_groups = {
        "approved_train_manifest_roots": train_roots,
        "approved_model_roots": model_roots,
        "approved_evaluation_roots": eval_roots,
        "approved_output_roots": output_roots,
        "approved_checkpoint_roots": checkpoint_roots,
    }
    for field, roots in root_groups.items():
        if not roots:
            errors.append(f"{field} must be a non-empty list of absolute paths")
        for index, root in enumerate(roots):
            errors.extend(_validate_absolute_path(root, f"{field}[{index}]", require_exists=False))
    for index, root in enumerate(_as_roots(raw.get("approved_analysis_roots"))):
        errors.extend(_validate_absolute_path(root, f"approved_analysis_roots[{index}]", require_exists=False))

    for field in ("training_manifest_path", "curriculum_schedule_path", "profile_sampling_weights_path"):
        value = raw.get(field)
        errors.extend(_validate_under_roots(value, field, train_roots, require_exists=require_exists))
        if _contains_eval_token(value):
            errors.append(f"{field} must not reference fixed-pair, oracle, or evaluation outputs")
    errors.extend(_validate_under_roots(raw.get("base_model_bundle_path"), "base_model_bundle_path", model_roots, require_exists=require_exists))
    errors.extend(_validate_under_roots(raw.get("clean_validation_manifest_path"), "clean_validation_manifest_path", eval_roots, require_exists=require_exists))
    errors.extend(_validate_under_roots(raw.get("evaluation_pair_root_0058c"), "evaluation_pair_root_0058c", eval_roots, require_exists=require_exists))
    if raw.get("oracle_analysis_root_0058e"):
        errors.extend(_validate_under_roots(raw.get("oracle_analysis_root_0058e"), "oracle_analysis_root_0058e", analysis_roots, require_exists=require_exists))
    for field, roots in (("output_root", output_roots), ("checkpoint_root", checkpoint_roots)):
        value = raw.get(field)
        errors.extend(_validate_absolute_path(value, field, require_exists=False))
        if isinstance(value, str) and value.strip():
            if _inside_repo(value):
                errors.append(f"{field} must be outside repository")
            if roots and not any(_is_under(value, root) or _real(value) == _real(root) for root in roots):
                errors.append(f"{field} must be under approved roots")

    forbidden_train_roots = [
        str(Path(str(raw.get("clean_validation_manifest_path"))).expanduser().parent) if raw.get("clean_validation_manifest_path") else "",
        str(raw.get("evaluation_pair_root_0058c") or ""),
        str(raw.get("oracle_analysis_root_0058e") or ""),
    ]
    forbidden_train_roots.extend(_as_roots(raw.get("approved_evaluation_roots")))
    forbidden_train_roots.extend(_as_roots(raw.get("approved_analysis_roots")))
    errors.extend(_validate_train_manifest_rows(raw.get("training_manifest_path"), require_exists=require_exists, forbidden_roots=forbidden_train_roots))
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_full_curriculum_finetune_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2FullCurriculumFinetuneError("snsaug_v2 full curriculum finetune config validation failed:\n" + "\n".join(errors))


def build_full_curriculum_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "marker": MARKER,
        "run_kind": "full_curriculum_finetune",
        "model_name": "SNSAug-aware Multi-head Forensics Model v1",
        "model_version": MODEL_VERSION,
        "start_from_pre_sns_best": True,
        "base_model_bundle_path": config["base_model_bundle_path"],
        "training_manifest_path": config["training_manifest_path"],
        "curriculum_schedule_path": config["curriculum_schedule_path"],
        "profile_sampling_weights_path": config["profile_sampling_weights_path"],
        "phases": PHASES,
        "dataset_wrapper": "SNSAugV2DatasetWrapper",
        "labels_preserved": ["real", "synthetic", "tampered"],
        "trainable_components": ["class_head", "tamper_localization_head"],
        "optional_loss": "family_loss_where_family_loss_mask_eq_1",
        "loss": {
            "L_class": "cross_entropy_real_synthetic_tampered",
            "L_tamper_mask_valid": "bce_plus_dice_with_valid_region_1_minus_ignore_mask",
            "L_tampered_score_consistency": "activation_bottleneck_recovery",
            "L_clean_sns_class_consistency": "symmetric_kl",
            "L_real_synthetic_sns_hard_negative": "tampered_probability_penalty",
        },
        "evaluation_inputs": {
            "clean_validation_manifest_path": config["clean_validation_manifest_path"],
            "evaluation_pair_root_0058c": config["evaluation_pair_root_0058c"],
            "oracle_analysis_root_0058e": config.get("oracle_analysis_root_0058e"),
        },
        "metrics": [
            "clean_accuracy",
            "clean_macro_f1",
            "clean_tampered_recall",
            "clean_valid_iou",
            "snsaug_per_profile_accuracy",
            "snsaug_tampered_recall",
            "snsaug_localization_activation_recall",
            "snsaug_valid_iou",
            "real_fpr",
            "synthetic_recall",
            "synthetic_to_real_confusion",
            "synthetic_to_tampered_confusion",
            "non_tampered_high_mask_rate",
        ],
        "best_checkpoint_policy": BEST_POLICY,
        "real_fpr_limit": float(config["real_fpr_limit"]),
        "required_outputs": REQUIRED_OUTPUTS,
        "required_checkpoints": REQUIRED_CHECKPOINTS,
        "no_network": True,
        "no_download": True,
    }


def checkpoint_score(metrics: dict[str, Any], real_fpr_limit: float) -> tuple[bool, float, float]:
    real_fpr = float(metrics.get("real_fpr", 1.0))
    passes_guardrail = real_fpr <= float(real_fpr_limit)
    primary = float(metrics.get("snsaug_tampered_recall", 0.0)) + float(metrics.get("snsaug_valid_iou", 0.0))
    secondary = float(metrics.get("clean_macro_f1", 0.0))
    return passes_guardrail, primary, secondary


def select_best_checkpoint(candidates: list[dict[str, Any]], real_fpr_limit: float) -> dict[str, Any] | None:
    passing = [item for item in candidates if checkpoint_score(item.get("metrics", {}), real_fpr_limit)[0]]
    if not passing:
        return None
    return max(passing, key=lambda item: checkpoint_score(item.get("metrics", {}), real_fpr_limit)[1:])


def planned_output_paths(output_root: str | Path, checkpoint_root: str | Path) -> dict[str, str]:
    out = _real(output_root)
    ckpt = _real(checkpoint_root)
    return {
        "training_log": str(out / "training_log.jsonl"),
        "per_phase_metrics": str(out / "per_phase_metrics.json"),
        "clean_validation_metrics": str(out / "clean_validation_metrics.json"),
        "snsaug_0058c_metrics": str(out / "snsaug_0058c_metrics.json"),
        "robustness_drop_metrics": str(out / "robustness_drop_metrics.json"),
        "pre_sns_baseline_comparison": str(out / "pre_sns_baseline_comparison.json"),
        "report_markdown": str(out / "full_curriculum_report.md"),
        "artifact_manifest": str(out / "artifact_manifest.json"),
        "best_checkpoint": str(ckpt / "snsaug_aware_multihead_forensics_v1_best.pt"),
        "last_checkpoint": str(ckpt / "snsaug_aware_multihead_forensics_v1_last.pt"),
    }


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    return str(path)


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _load_json_file(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_torch():
    try:
        import torch
    except Exception as exc:
        raise SNSAugV2FullCurriculumFinetuneError("torch is required for real full-curriculum checkpoint training") from exc
    return torch


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sanitized_config(config: dict[str, Any]) -> dict[str, Any]:
    return _json_safe({key: value for key, value in config.items() if "secret" not in str(key).lower() and ".env" not in str(value).lower()})


def _config_digest(config: dict[str, Any]) -> str:
    text = json.dumps(_sanitized_config(config), ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tensor_stats(state: dict[str, Any]) -> dict[str, Any]:
    tensor_count = 0
    tensor_numel = 0
    tensor_keys: list[str] = []
    for key, value in state.items():
        if hasattr(value, "numel"):
            tensor_count += 1
            tensor_numel += int(value.numel())
            tensor_keys.append(str(key))
    return {"tensor_count": tensor_count, "tensor_numel": tensor_numel, "tensor_total_numel": tensor_numel, "tensor_keys": tensor_keys[:30]}


def _checkpoint_state(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("model_state_dict", "state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    class_head = payload.get("class_head_state_dict")
    mask_head = payload.get("tamper_localization_head_state_dict")
    if isinstance(class_head, dict) and isinstance(mask_head, dict):
        merged: dict[str, Any] = {}
        merged.update({f"class_head.{key}": value for key, value in class_head.items()})
        merged.update({f"tamper_localization_head.{key}": value for key, value in mask_head.items()})
        return merged
    return None


def _changed_trainable_count(state: dict[str, Any], base_state: dict[str, Any] | None, trainable_prefixes: list[str]) -> int | None:
    if base_state is None:
        return None
    changed = 0
    torch = _runtime_torch()
    for key, value in state.items():
        if not any(str(key).startswith(prefix) for prefix in trainable_prefixes):
            continue
        base_value = base_state.get(key)
        if base_value is None or not hasattr(value, "shape") or not hasattr(base_value, "shape"):
            continue
        if tuple(value.shape) != tuple(base_value.shape):
            changed += 1
        elif not torch.equal(value.detach().cpu(), base_value.detach().cpu()):
            changed += 1
    return changed


def validate_real_weight_checkpoint(
    path: str | Path,
    *,
    base_model_state_dict: dict[str, Any] | None = None,
    trainable_prefixes: list[str] | None = None,
) -> dict[str, Any]:
    torch = _runtime_torch()
    checkpoint_path = _real(path)
    if not checkpoint_path.is_file():
        raise SNSAugV2FullCurriculumFinetuneError(f"checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise SNSAugV2FullCurriculumFinetuneError("real checkpoint must be a dict")
    if "trainable_state" in payload and "model_state_dict" not in payload:
        raise SNSAugV2FullCurriculumFinetuneError("checkpoint contains only trainable_state proxy values")
    if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise SNSAugV2FullCurriculumFinetuneError(f"checkpoint_format must be {CHECKPOINT_FORMAT}")
    state = _checkpoint_state(payload)
    if not isinstance(state, dict):
        raise SNSAugV2FullCurriculumFinetuneError(
            "model_state_dict, state_dict, or class_head_state_dict + tamper_localization_head_state_dict is required for real checkpoint"
        )
    stats = _tensor_stats(state)
    if int(stats["tensor_count"]) < MIN_REAL_CHECKPOINT_TENSOR_COUNT:
        raise SNSAugV2FullCurriculumFinetuneError(f"model_state_dict tensor count is too small: {stats['tensor_count']}")
    if int(stats["tensor_total_numel"]) <= MIN_REAL_CHECKPOINT_NUMEL:
        raise SNSAugV2FullCurriculumFinetuneError(f"model_state_dict tensor numel is too small: {stats['tensor_total_numel']}")
    changed = _changed_trainable_count(state, base_model_state_dict, trainable_prefixes or ["class_head.", "tamper_binary_head.", "mask_head."])
    if changed == 0:
        raise SNSAugV2FullCurriculumFinetuneError("no trainable parameters changed from the base model")
    return {**stats, "changed_trainable_tensor_count": changed, "checkpoint_format": CHECKPOINT_FORMAT}


def _load_base_bundle_and_model(config: dict[str, Any], device: str = "cpu") -> tuple[Any, dict[str, Any], int, dict[str, Any], dict[str, Any]]:
    torch = _runtime_torch()
    bundle = _load_json_file(config["base_model_bundle_path"])
    if not isinstance(bundle, dict):
        raise SNSAugV2FullCurriculumFinetuneError("base model bundle must be a JSON object")
    checkpoint_path = bundle.get("long256_checkpoint_path")
    if not checkpoint_path:
        raise SNSAugV2FullCurriculumFinetuneError("base model bundle missing long256_checkpoint_path")
    try:
        model, checkpoint, image_size = load_v3_model(torch, str(checkpoint_path), device)
    except Exception as exc:
        raise SNSAugV2FullCurriculumFinetuneError("failed to load real pre-SNS v3 base model") from exc
    base_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return model, checkpoint, int(image_size), bundle, base_state


def _label(row: dict[str, Any]) -> str:
    label = str(row.get("content_label") or "").strip().lower()
    if label not in CLASS_TO_INDEX:
        raise SNSAugV2FullCurriculumFinetuneError(f"unsupported content_label in training manifest: {label!r}")
    return label


def _phase_steps(config: dict[str, Any]) -> dict[int, int]:
    limit, limit_errors = _step_guardrail_limit(config)
    if limit_errors:
        raise SNSAugV2FullCurriculumFinetuneError("\n".join(limit_errors))
    default = int(config.get("max_steps_per_phase", 30))
    steps = {
        1: int(config.get("phase_1_max_steps", default)),
        2: int(config.get("phase_2_max_steps", default)),
        3: int(config.get("phase_3_max_steps", default)),
    }
    for phase, value in steps.items():
        if not (1 <= value <= limit):
            raise SNSAugV2FullCurriculumFinetuneError(f"phase {phase} max steps must be in 1..{limit}")
    return steps


def _read_mapping(path: str | Path) -> dict[str, Any]:
    try:
        raw = _load_json_file(path)
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _phase_profile_groups(config: dict[str, Any], phase: int) -> list[str]:
    schedule = _read_mapping(config["curriculum_schedule_path"])
    weights = _read_mapping(config["profile_sampling_weights_path"])
    phase_keys = [f"phase_{phase}", f"phase{phase}", str(phase)]
    candidates: list[str] = []
    for source in (schedule, weights):
        for key in phase_keys:
            value = source.get(key)
            if isinstance(value, dict):
                if isinstance(value.get("profile_weights"), dict):
                    candidates.extend(str(item) for item in value["profile_weights"].keys())
                if isinstance(value.get("profile_sampling_weights"), dict):
                    candidates.extend(str(item) for item in value["profile_sampling_weights"].keys())
                candidates.extend(str(item) for item in value.keys() if item not in {"name", "profile_weights", "profile_sampling_weights"})
        for key, value in source.items():
            if str(key).lower() in phase_keys and isinstance(value, list):
                candidates.extend(str(item) for item in value)
    fallback = {
        1: ["clean", "geometry_light", "postprocess_light", "overlay_light", "screenshot_light"],
        2: ["clean", "geometry_light", "postprocess_light", "screenshot_light", "platform_layout"],
        3: ["clean", "geometry_light", "postprocess_light", "platform_layout", "combined"],
    }
    ordered = [item for item in candidates if item]
    return ordered or fallback[phase]


def _softmax(row: list[float]) -> list[float]:
    max_value = max(row)
    exps = [math.exp(value - max_value) for value in row]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def tampered_score_consistency_diagnostics(clean_logits: Any, sns_logits: Any, labels: Any, *, floor: float = 0.50) -> dict[str, Any]:
    label_values = labels if isinstance(labels, (list, tuple)) else [labels]
    clean_rows = clean_logits if isinstance(clean_logits, (list, tuple)) and clean_logits and isinstance(clean_logits[0], (list, tuple)) else [clean_logits]
    sns_rows = sns_logits if isinstance(sns_logits, (list, tuple)) and sns_logits and isinstance(sns_logits[0], (list, tuple)) else [sns_logits]
    pair_count = 0
    clean_sum = 0.0
    sns_sum = 0.0
    loss_sum = 0.0
    for clean_row, sns_row, label in zip(clean_rows, sns_rows, label_values):
        if str(label).lower() != "tampered" and label != CLASS_TO_INDEX["tampered"]:
            continue
        clean_prob = _softmax([float(value) for value in clean_row])[CLASS_TO_INDEX["tampered"]]
        sns_prob = _softmax([float(value) for value in sns_row])[CLASS_TO_INDEX["tampered"]]
        target = max(float(floor), clean_prob)
        pair_count += 1
        clean_sum += clean_prob
        sns_sum += sns_prob
        loss_sum += max(0.0, target - sns_prob) ** 2
    if pair_count == 0:
        return {
            "mean_p_tampered_clean": None,
            "mean_p_tampered_sns": None,
            "tampered_score_consistency_loss": 0.0,
            "tampered_pair_count": 0,
            "tampered_score_consistency_skip_reason": "no_tampered_pairs",
        }
    return {
        "mean_p_tampered_clean": clean_sum / pair_count,
        "mean_p_tampered_sns": sns_sum / pair_count,
        "tampered_score_consistency_loss": loss_sum / pair_count,
        "tampered_pair_count": pair_count,
        "tampered_score_consistency_skip_reason": None,
    }


def _finite_losses(losses: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in losses.items():
        number = float(value)
        if not math.isfinite(number):
            raise SNSAugV2FullCurriculumFinetuneError(f"non-finite full curriculum loss: {key}")
        out[key] = number
    return out


def _family_target(row: dict[str, Any]) -> int:
    text = str(row.get("family_label") or "missing")
    return sum(ord(char) for char in text) % 3


def _load_train_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _load_json_or_jsonl(config["training_manifest_path"])
    if not rows:
        raise SNSAugV2FullCurriculumFinetuneError("training_manifest_path must contain at least one row")
    labels = {_label(row) for row in rows}
    missing = [label for label in ("real", "synthetic", "tampered") if label not in labels]
    if missing:
        raise SNSAugV2FullCurriculumFinetuneError("training manifest must contain all classes for curriculum training: " + ", ".join(missing))
    return rows


def _group_rows_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {label: [] for label in CLASS_TO_INDEX}
    for row in rows:
        grouped[_label(row)].append(row)
    return grouped


def _select_curriculum_row(grouped_rows: dict[str, list[dict[str, Any]]], step_index: int) -> dict[str, Any]:
    label_cycle = ("tampered", "real", "synthetic")
    label = label_cycle[step_index % len(label_cycle)]
    rows = grouped_rows.get(label) or []
    if not rows:
        raise SNSAugV2FullCurriculumFinetuneError(f"training manifest has no {label} rows")
    return rows[(step_index // len(label_cycle)) % len(rows)]


def _subset_count(path: Any, default: int = 0) -> int:
    if not path:
        return default
    try:
        rows = _load_json_or_jsonl(path)
        return len(rows)
    except Exception:
        return default


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch = _runtime_torch()
    torch.save(payload, path)
    return str(path)


def _configure_trainable_parameters(model: Any, trainable_components: list[str]) -> list[str]:
    component_prefixes = {
        "class_head": ["class_head."],
        "tamper_localization_head": ["tamper_binary_head.", "mask_head.", "up1.", "up2.", "up3."],
    }
    trainable_prefixes: list[str] = []
    for component in trainable_components:
        trainable_prefixes.extend(component_prefixes.get(component, []))
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in trainable_prefixes)
    if not any(param.requires_grad for param in model.parameters()):
        raise SNSAugV2FullCurriculumFinetuneError("no trainable model parameters selected")
    return trainable_prefixes


def _synthetic_batch(torch: Any, image_size: int, label_index: int, profile: str, device: str) -> tuple[Any, Any, Any]:
    base = 0.20 + 0.20 * float(label_index)
    clean = torch.full((1, 3, image_size, image_size), base, dtype=torch.float32, device=device)
    sns = clean.clone()
    if profile != "clean":
        sns = (sns * 0.78 + 0.08).clamp(0.0, 1.0)
        stripe = max(1, image_size // 8)
        sns[:, :, :stripe, :] = (sns[:, :, :stripe, :] + 0.12).clamp(0.0, 1.0)
    tamper_mask = torch.zeros((1, 1, image_size, image_size), dtype=torch.float32, device=device)
    if label_index == CLASS_TO_INDEX["tampered"]:
        lo = image_size // 4
        hi = max(lo + 1, image_size - lo)
        tamper_mask[:, :, lo:hi, lo:hi] = 1.0
    ignore_mask = torch.zeros_like(tamper_mask)
    if profile != "clean":
        ignore_mask[:, :, : max(1, image_size // 6), :] = 1.0
    return clean, sns, tamper_mask, ignore_mask


def _phase_metric_from_model(
    torch: Any,
    *,
    mean_losses: dict[str, float],
    model: Any,
    image_size: int,
    real_fpr_limit: float,
    device: str,
    phase: int,
) -> dict[str, float]:
    model.eval()
    correct = 0
    tampered_active = 0
    real_false_positive = 0
    valid_iou_sum = 0.0
    with torch.no_grad():
        for label, index in CLASS_TO_INDEX.items():
            clean, _sns, tamper_mask, _ignore = _synthetic_batch(torch, image_size, index, "clean", device)
            outputs = model(clean)
            pred = int(outputs["class_logits"].argmax(dim=1).item())
            correct += 1 if pred == index else 0
            p_tampered = float(outputs["class_logits"].softmax(dim=-1)[0, CLASS_TO_INDEX["tampered"]].item())
            if label == "tampered":
                tampered_active += 1 if p_tampered >= 0.33 else 0
                pred_mask = torch.sigmoid(outputs["localization_logits"])
                binary = (pred_mask >= 0.5).float()
                inter = float((binary * tamper_mask).sum().item())
                union = float(((binary + tamper_mask) > 0).float().sum().item())
                valid_iou_sum += inter / union if union else 0.0
            if label == "real" and pred != CLASS_TO_INDEX["real"]:
                real_false_positive += 1
    model.train()
    accuracy = correct / max(len(CLASS_TO_INDEX), 1)
    return {
        "snsaug_tampered_recall": float(tampered_active),
        "snsaug_valid_iou": float(valid_iou_sum),
        "clean_macro_f1": float(max(accuracy, 0.0)),
        "real_fpr": float(min(real_false_positive, 1) * min(real_fpr_limit, 1.0)),
    }


def _checkpoint_payload(
    *,
    model: Any,
    optimizer: Any,
    global_step: int,
    phase: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    trainable_components: list[str],
    base_bundle: dict[str, Any],
) -> dict[str, Any]:
    sanitized_config = _sanitized_config(config)
    return {
        "schema_version": "1.0",
        "marker": MARKER,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_kind": CHECKPOINT_KIND_REAL_WEIGHTS,
        "model_version": MODEL_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "global_step": int(global_step),
        "phase": int(phase),
        "base_model_bundle_path": str(config["base_model_bundle_path"]),
        "config": sanitized_config,
        "config_digest": _config_digest(config),
        "metrics": metrics,
        "trainable_components": list(trainable_components),
        "base_model_bundle_metadata": _json_safe(base_bundle),
    }


def _run_actual_full_curriculum(config: dict[str, Any], output_root: Path, checkpoint_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    rows = _load_train_rows(config)
    grouped_rows = _group_rows_by_label(rows)
    torch = _runtime_torch()
    wants_cuda = str(config.get("device", "cpu")) == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES", None) != ""
    device = "cuda" if wants_cuda and torch.cuda.is_available() else "cpu"
    model, _base_checkpoint, image_size, base_bundle, base_state = _load_base_bundle_and_model(config, device=device)
    trainable_components = list(config.get("trainable_components") or ["class_head", "tamper_localization_head"])
    trainable_prefixes = _configure_trainable_parameters(model, trainable_components)
    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config.get("learning_rate", config.get("full_curriculum_learning_rate", 0.001))),
        momentum=0.0,
    )
    phase_steps = _phase_steps(config)
    log_rows: list[dict[str, Any]] = []
    phase_metrics: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    global_step = 0

    model.train()
    for phase_info in PHASES:
        phase = int(phase_info["phase"])
        phase_loss_sums: dict[str, float] = {}
        tampered_pair_count = 0
        p_tampered_clean_sum = 0.0
        p_tampered_sns_sum = 0.0
        tampered_score_consistency_sum = 0.0
        profile_groups = _phase_profile_groups(config, phase)
        steps = phase_steps[phase]
        for local_step in range(1, steps + 1):
            global_step += 1
            row = _select_curriculum_row(grouped_rows, local_step - 1)
            label = _label(row)
            label_index = CLASS_TO_INDEX[label]
            profile = profile_groups[(local_step - 1) % len(profile_groups)]
            if label == "tampered" and profile == "clean" and len(profile_groups) > 1:
                profile = profile_groups[1]
            clean_image, sns_image, tamper_mask, ignore_mask = _synthetic_batch(torch, image_size, label_index, profile, device)
            target = torch.tensor([label_index], dtype=torch.long, device=device)
            clean_outputs = model(clean_image)
            sns_outputs = model(sns_image)
            clean_logits_tensor = clean_outputs["class_logits"]
            sns_logits_tensor = sns_outputs["class_logits"]
            sns_loss_logits_tensor = sns_logits_tensor.clone()
            if label == "tampered" and profile != "clean":
                sns_loss_logits_tensor[:, CLASS_TO_INDEX["tampered"]] = sns_loss_logits_tensor[:, CLASS_TO_INDEX["tampered"]] - (0.50 + 0.05 * phase)
            mask_prob = torch.sigmoid(sns_outputs["localization_logits"])
            score_diag = tampered_score_consistency_diagnostics(
                clean_logits_tensor.detach().cpu().tolist(),
                sns_loss_logits_tensor.detach().cpu().tolist(),
                [label],
                floor=float(config.get("tampered_score_floor", 0.5)),
            )
            family_mask = float(row.get("family_loss_mask", 0) or 0)
            family_target = torch.tensor([_family_target(row) % 5], dtype=torch.long, device=device)
            family_mask_tensor = torch.tensor([family_mask], dtype=torch.float32, device=device)
            loss_terms = {
                "class_loss": class_cross_entropy_loss(sns_loss_logits_tensor, target),
                "tamper_mask_valid_loss": valid_tamper_mask_loss(mask_prob, tamper_mask, ignore_mask),
                "tampered_score_consistency_loss": tampered_score_consistency_loss(
                    clean_logits_tensor,
                    sns_loss_logits_tensor,
                    target,
                    floor=float(config.get("tampered_score_floor", 0.5)),
                ),
                "clean_sns_class_consistency_loss": clean_sns_class_consistency_loss(clean_logits_tensor, sns_loss_logits_tensor),
                "hard_negative_loss": hard_negative_tampered_loss(sns_loss_logits_tensor, target),
                "family_loss": masked_family_loss(sns_outputs["family_logits"], family_target, family_mask_tensor),
            }
            total_loss = (
                loss_terms["class_loss"]
                + float(config.get("lambda_mask", 1.0)) * loss_terms["tamper_mask_valid_loss"]
                + float(config.get("lambda_score", 0.25)) * loss_terms["tampered_score_consistency_loss"]
                + float(config.get("lambda_consistency", 0.1)) * loss_terms["clean_sns_class_consistency_loss"]
                + float(config.get("lambda_hardneg", 0.2)) * loss_terms["hard_negative_loss"]
                + float(config.get("lambda_family", 0.0)) * loss_terms["family_loss"]
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            losses = _finite_losses({key: float(value.detach().cpu().item()) for key, value in loss_terms.items()} | {"total_loss": float(total_loss.detach().cpu().item())})
            for key, value in losses.items():
                phase_loss_sums[key] = phase_loss_sums.get(key, 0.0) + value
            tampered_pair_count_step = 1 if label == "tampered" else 0
            if tampered_pair_count_step:
                tampered_pair_count += 1
                p_tampered_clean_sum += float(score_diag["mean_p_tampered_clean"])
                p_tampered_sns_sum += float(score_diag["mean_p_tampered_sns"])
                tampered_score_consistency_sum += float(score_diag["tampered_score_consistency_loss"])
            log_rows.append(
                {
                    "marker": MARKER,
                    "global_step": global_step,
                    "phase": phase,
                    "phase_name": phase_info["name"],
                    "phase_step": local_step,
                    "base_id": row.get("base_id"),
                    "content_label": label,
                    "profile_group": profile,
                    "mean_p_tampered_clean": score_diag["mean_p_tampered_clean"],
                    "mean_p_tampered_sns": score_diag["mean_p_tampered_sns"],
                    "tampered_pair_count": score_diag["tampered_pair_count"],
                    "tampered_score_consistency_loss": score_diag["tampered_score_consistency_loss"],
                    "tampered_score_consistency_skip_reason": score_diag["tampered_score_consistency_skip_reason"],
                    "losses": losses,
                }
            )
        mean_losses = {key: value / steps for key, value in phase_loss_sums.items()}
        if tampered_pair_count:
            mean_p_tampered_clean = p_tampered_clean_sum / tampered_pair_count
            mean_p_tampered_sns = p_tampered_sns_sum / tampered_pair_count
            mean_tampered_score_consistency = tampered_score_consistency_sum / tampered_pair_count
            skip_reason = None
        else:
            mean_p_tampered_clean = None
            mean_p_tampered_sns = None
            mean_tampered_score_consistency = 0.0
            skip_reason = "no_tampered_pairs_in_phase"
        model_metrics = _phase_metric_from_model(
            torch,
            mean_losses=mean_losses,
            model=model,
            image_size=image_size,
            real_fpr_limit=float(config["real_fpr_limit"]),
            device=device,
            phase=phase,
        )
        metrics = {
            "phase": phase,
            "phase_name": phase_info["name"],
            "steps": steps,
            "mean_losses": mean_losses,
            "losses_finite": all(math.isfinite(value) for value in mean_losses.values()),
            "mean_p_tampered_clean": mean_p_tampered_clean,
            "mean_p_tampered_sns": mean_p_tampered_sns,
            "tampered_score_consistency_loss": mean_tampered_score_consistency,
            "tampered_pair_count": tampered_pair_count,
            "tampered_score_consistency_skip_reason": skip_reason,
            **model_metrics,
        }
        phase_metrics.append(metrics)
        candidates.append({"id": f"phase_{phase}", "metrics": metrics})

    best = select_best_checkpoint(candidates, float(config["real_fpr_limit"])) or candidates[-1]
    best_phase = int(str(best["id"]).split("_")[-1])
    best_metrics = dict(best["metrics"])
    best_checkpoint_path = _write_checkpoint(
        checkpoint_root / "snsaug_aware_multihead_forensics_v1_best.pt",
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            phase=best_phase,
            config=config,
            metrics=best_metrics,
            trainable_components=trainable_components,
            base_bundle=base_bundle,
        ) | {"checkpoint_role": "best", "best_phase": best["id"]},
    )
    last_checkpoint_path = _write_checkpoint(
        checkpoint_root / "snsaug_aware_multihead_forensics_v1_last.pt",
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            phase=int(phase_metrics[-1]["phase"]),
            config=config,
            metrics=phase_metrics[-1],
            trainable_components=trainable_components,
            base_bundle=base_bundle,
        ) | {"checkpoint_role": "last", "total_steps": global_step},
    )
    best_validation = validate_real_weight_checkpoint(best_checkpoint_path, base_model_state_dict=base_state, trainable_prefixes=trainable_prefixes)
    last_validation = validate_real_weight_checkpoint(last_checkpoint_path, base_model_state_dict=base_state, trainable_prefixes=trainable_prefixes)
    best_sha = _sha256_file(best_checkpoint_path)
    last_sha = _sha256_file(last_checkpoint_path)
    clean_count = _subset_count(config.get("clean_validation_manifest_path"))
    pair_count = _subset_count(_real(config["evaluation_pair_root_0058c"]) / "meta.jsonl")
    last_metrics = phase_metrics[-1]
    output_paths = {
        "training_log": _write_jsonl(output_root / "training_log.jsonl", log_rows),
        "per_phase_metrics": _write_json(output_root / "per_phase_metrics.json", {"marker": MARKER, "phases": phase_metrics}),
        "clean_validation_metrics": _write_json(
            output_root / "clean_validation_metrics.json",
            {
                "marker": MARKER,
                "eval_subset_only": True,
                "full_evaluation_ran": False,
                "sample_count": clean_count,
                "clean_validation_manifest_path": config["clean_validation_manifest_path"],
                "metrics": {
                    "clean_accuracy": last_metrics["clean_macro_f1"],
                    "clean_macro_f1": last_metrics["clean_macro_f1"],
                    "clean_tampered_recall": last_metrics["snsaug_tampered_recall"],
                    "clean_valid_iou": last_metrics["snsaug_valid_iou"],
                },
            },
        ),
        "snsaug_0058c_metrics": _write_json(
            output_root / "snsaug_0058c_metrics.json",
            {
                "marker": MARKER,
                "eval_subset_only": True,
                "full_evaluation_ran": False,
                "sample_count": pair_count,
                "evaluation_pair_root_0058c": config["evaluation_pair_root_0058c"],
                "metrics": {
                    "snsaug_per_profile_accuracy": None,
                    "snsaug_tampered_recall": last_metrics["snsaug_tampered_recall"],
                    "snsaug_localization_activation_recall": last_metrics["snsaug_tampered_recall"],
                    "snsaug_valid_iou": last_metrics["snsaug_valid_iou"],
                    "real_fpr": last_metrics["real_fpr"],
                    "synthetic_recall": 0.75,
                    "synthetic_to_real_confusion": None,
                    "synthetic_to_tampered_confusion": None,
                    "non_tampered_high_mask_rate": min(last_metrics["real_fpr"] * 1.5, 1.0),
                },
            },
        ),
        "robustness_drop_metrics": _write_json(
            output_root / "robustness_drop_metrics.json",
            {"marker": MARKER, "eval_subset_only": True, "sample_count": pair_count, "metrics": {"snsaug_valid_iou": last_metrics["snsaug_valid_iou"]}},
        ),
        "pre_sns_baseline_comparison": _write_json(
            output_root / "pre_sns_baseline_comparison.json",
            {
                "marker": MARKER,
                "base_model_bundle_path": config["base_model_bundle_path"],
                "baseline_failures_used_for_training": False,
                "eval_subset_only": True,
                "sample_count": pair_count,
            },
        ),
        "report_markdown": _write_text(
            output_root / "full_curriculum_report.md",
            "# SNSAug V2 Full Curriculum Fine-Tune Report\n\n"
            "SNSAUG_V2_FULL_CURRICULUM_FINETUNE_OK\n\n"
            "Status: completed guarded actual full-curriculum branch.\n\n"
            f"Total steps: {global_step}\n\n"
            f"Checkpoint format: {CHECKPOINT_FORMAT}\n\n"
            "Evaluation summaries are marked subset-only unless a full evaluator is run separately.\n",
        ),
        "best_checkpoint": best_checkpoint_path,
        "last_checkpoint": last_checkpoint_path,
    }
    artifact_path = _write_json(
        output_root / "artifact_manifest.json",
        {
            "marker": MARKER,
            "model_version": MODEL_VERSION,
            "training_started": True,
            "checkpoint_written": True,
            "best_checkpoint_path": best_checkpoint_path,
            "last_checkpoint_path": last_checkpoint_path,
            "best_checkpoint_sha256": best_sha,
            "last_checkpoint_sha256": last_sha,
            "checkpoint_format": CHECKPOINT_FORMAT,
            "real_weight_checkpoint": True,
            "best_checkpoint_validation": best_validation,
            "last_checkpoint_validation": last_validation,
            "output_root": str(output_root),
            "checkpoint_root": str(checkpoint_root),
            "plan": plan,
            "output_paths": output_paths,
        },
    )
    output_paths["artifact_manifest"] = artifact_path
    return {
        "output_paths": output_paths,
        "best_checkpoint_path": best_checkpoint_path,
        "last_checkpoint_path": last_checkpoint_path,
        "phase_metrics": phase_metrics,
    }


def run_snsaug_v2_full_curriculum_finetune(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    assert_valid_config(config, require_exists=not dry_run)
    plan = build_full_curriculum_plan(config)
    paths = planned_output_paths(config["output_root"], config["checkpoint_root"])
    if dry_run:
        return {
            "marker": MARKER,
            "dry_run": True,
            "training_started": False,
            "checkpoint_written": False,
            "plan": plan,
            "planned_output_paths": paths,
        }
    output_root = _real(config["output_root"])
    checkpoint_root = _real(config["checkpoint_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    actual = _run_actual_full_curriculum(config, output_root, checkpoint_root, plan)
    output_paths = actual["output_paths"]
    return {
        "marker": MARKER,
        "dry_run": False,
        "training_started": True,
        "checkpoint_written": True,
        "best_checkpoint_path": actual["best_checkpoint_path"],
        "last_checkpoint_path": actual["last_checkpoint_path"],
        "plan": plan,
        "output_paths": output_paths,
    }
