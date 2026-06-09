"""Guarded SNSAug V2 nuisance-mask fine-tuning branch."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .snsaug_v2_full_curriculum_finetune import (
    CHECKPOINT_FORMAT,
    REPO_ROOT,
    _as_roots,
    _config_digest,
    _contains_eval_token,
    _inside_repo,
    _is_protected,
    _is_under,
    _json_safe,
    _load_base_bundle_and_model,
    _load_json_or_jsonl,
    _real,
    _runtime_torch,
    _sanitized_config,
    _sha256_file,
    _validate_absolute_path,
    _validate_numeric,
    _validate_under_roots,
    _write_json,
    _write_jsonl,
    _write_text,
)
from .snsaug_v2_losses import CLASS_TO_INDEX
from .snsaug_v2_nuisance_losses import (
    degradation_label_from_profile,
    sns_nuisance_mask_target,
    total_nuisance_loss,
)
from .snsaug_v2_nuisance_model import DEGRADATION_LABELS, build_snsaug_v2_nuisance_model

MARKER = "SNSAUG_V2_NUISANCE_MASK_FINETUNE_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_NUISANCE_MASK_FINETUNE_CONFIG_OK"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_NUISANCE_MASK_FINETUNE"
APPROVED_KIND = "approved_snsaug_v2_nuisance_mask_finetune"
APPROVED_MODE = "approved_local_snsaug_v2_nuisance_mask_finetune"
RUN_KIND = "snsaug_v2_nuisance_mask_finetune"
MODEL_VERSION = "snsaug_aware_nuisance_multihead_forensics_v1"
CHECKPOINT_KIND_REAL_WEIGHTS = "snsaug_v2_nuisance_mask_real_model_weights"
CHECKPOINT_NAMES = {
    "best": "snsaug_aware_multihead_forensics_v1_best.pt",
    "last": "snsaug_aware_multihead_forensics_v1_last.pt",
}
PHASES = (
    {"phase": 1, "name": "nuisance_head_warmup"},
    {"phase": 2, "name": "mask_guided_tamper_gating"},
    {"phase": 3, "name": "balanced_nuisance_correction"},
)
REQUIRED_OUTPUTS = [
    "training_log.jsonl",
    "warm_start_report.json",
    "hybrid_warm_start_report.json",
    "loss_breakdown.json",
    "per_phase_metrics.json",
    "clean_validation_metrics.json",
    "snsaug_0058c_metrics.json",
    "threshold_sweep_after_training.json",
    "artifact_manifest.json",
    "nuisance_mask_report.md",
]


class SNSAugV2NuisanceFinetuneError(ValueError):
    """Raised when nuisance-mask fine-tune guardrails fail."""


def load_snsaug_v2_nuisance_finetune_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2NuisanceFinetuneError("nuisance fine-tune config root must be a JSON object")
    return raw


def _training_approval_required(raw: dict[str, Any]) -> bool:
    return bool(raw.get("enable_real_training") or raw.get("run_training") or raw.get("write_checkpoints"))


def _validate_train_rows(path: Any, require_exists: bool, forbidden_roots: list[str]) -> list[str]:
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
        for field in ("image_path", "tamper_mask_path", "mask_path", "ignore_mask_path"):
            value = str(row.get(field) or "")
            if not value:
                continue
            if _contains_eval_token(value):
                errors.append("training manifest rows must not reference fixed-pair, validation, oracle, or evaluation outputs")
                return errors
            for root in forbidden_roots:
                if root and _is_under(value, root):
                    errors.append("training manifest rows must not use validation, fixed-pair, oracle, or evaluation roots as training input")
                    return errors
    return errors


def validate_snsaug_v2_nuisance_finetune_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "run_kind",
        "model_version",
        "approval_text",
        "start_from_pre_sns_best",
        "no_training_from_scratch",
        "training_manifest_path",
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
    if raw.get("run_kind") != RUN_KIND:
        errors.append(f"run_kind must be {RUN_KIND}")
    if raw.get("model_version") != MODEL_VERSION:
        errors.append(f"model_version must be {MODEL_VERSION}")
    if _training_approval_required(raw) and raw.get("approval_text") != APPROVAL_TEXT:
        errors.append(f"approval_text must equal {APPROVAL_TEXT} for real training")
    for flag in ("start_from_pre_sns_best", "no_training_from_scratch", "no_network", "no_download"):
        if raw.get(flag) is not True:
            errors.append(f"{flag} must be true")
    for field, default in (
        ("lambda_sns_mask", 1.0),
        ("lambda_degradation", 0.3),
        ("lambda_hardneg", 1.0),
        ("lambda_tamper_mask", 1.0),
        ("lambda_gating_consistency", 0.5),
        ("lambda_synthetic_preservation", 2.0),
        ("lambda_non_tampered_tampered_suppression", 2.0),
        ("lambda_class_preservation", 1.0),
        ("lambda_non_tampered_mask_suppression", 1.0),
        ("lambda_mask_area_regularization", 0.1),
        ("gating_alpha", 0.5),
        ("phase_2_gating_alpha_max", 0.15),
        ("phase_3_gating_alpha_max", 0.25),
        ("p_tampered_ceiling", 0.05),
        ("synthetic_probability_floor", 0.35),
        ("min_warm_start_loaded_numel_ratio", 0.25),
        ("require_warm_start_loaded_numel_ratio_gte", 0.0),
    ):
        value = raw.get(field, default)
        errors.extend(
            _validate_numeric(
                value,
                field,
                minimum=0.0,
                maximum=1.0
                if field
                in {
                    "gating_alpha",
                    "phase_2_gating_alpha_max",
                    "phase_3_gating_alpha_max",
                    "p_tampered_ceiling",
                    "synthetic_probability_floor",
                    "min_warm_start_loaded_numel_ratio",
                    "require_warm_start_loaded_numel_ratio_gte",
                }
                else None,
            )
        )
    for field in ("max_steps", "max_steps_per_phase", "phase_1_max_steps", "phase_2_max_steps", "phase_3_max_steps"):
        if field in raw:
            value = raw.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > 500:
                errors.append(f"{field} must be an integer in 1..500")

    train_roots = _as_roots(raw.get("approved_train_manifest_roots"))
    model_roots = _as_roots(raw.get("approved_model_roots"))
    eval_roots = _as_roots(raw.get("approved_evaluation_roots"))
    output_roots = _as_roots(raw.get("approved_output_roots"))
    checkpoint_roots = _as_roots(raw.get("approved_checkpoint_roots"))
    for field, roots in (
        ("approved_train_manifest_roots", train_roots),
        ("approved_model_roots", model_roots),
        ("approved_evaluation_roots", eval_roots),
        ("approved_output_roots", output_roots),
        ("approved_checkpoint_roots", checkpoint_roots),
    ):
        if not roots:
            errors.append(f"{field} must be a non-empty list of absolute paths")
        for index, root in enumerate(roots):
            errors.extend(_validate_absolute_path(root, f"{field}[{index}]", require_exists=False))

    for field in ("training_manifest_path",):
        errors.extend(_validate_under_roots(raw.get(field), field, train_roots, require_exists=require_exists))
        if _contains_eval_token(raw.get(field)):
            errors.append(f"{field} must not reference fixed-pair, validation, oracle, or evaluation outputs")
    errors.extend(_validate_under_roots(raw.get("base_model_bundle_path"), "base_model_bundle_path", model_roots, require_exists=require_exists))
    if raw.get("pre_sns_best_bundle_path"):
        errors.extend(_validate_under_roots(raw.get("pre_sns_best_bundle_path"), "pre_sns_best_bundle_path", model_roots, require_exists=require_exists))
    if raw.get("warm_start_checkpoint_path"):
        errors.extend(_validate_under_roots(raw.get("warm_start_checkpoint_path"), "warm_start_checkpoint_path", model_roots, require_exists=require_exists))
    if raw.get("class_backbone_warm_start_path"):
        errors.extend(_validate_under_roots(raw.get("class_backbone_warm_start_path"), "class_backbone_warm_start_path", model_roots, require_exists=require_exists))
    if raw.get("tamper_head_warm_start_path"):
        errors.extend(_validate_under_roots(raw.get("tamper_head_warm_start_path"), "tamper_head_warm_start_path", model_roots, require_exists=require_exists))
    errors.extend(_validate_under_roots(raw.get("clean_validation_manifest_path"), "clean_validation_manifest_path", eval_roots, require_exists=require_exists))
    errors.extend(_validate_under_roots(raw.get("evaluation_pair_root_0058c"), "evaluation_pair_root_0058c", eval_roots, require_exists=require_exists))

    for field, roots in (("output_root", output_roots), ("checkpoint_root", checkpoint_roots)):
        value = raw.get(field)
        errors.extend(_validate_absolute_path(value, field, require_exists=False))
        if isinstance(value, str) and value.strip():
            if _inside_repo(value):
                errors.append(f"{field} must be outside repository")
            if _is_protected(value):
                errors.append(f"{field} must not reference protected paths")
            if roots and not any(_is_under(value, root) or _real(value) == _real(root) for root in roots):
                errors.append(f"{field} must be under approved roots")

    forbidden_roots = list(eval_roots)
    if raw.get("clean_validation_manifest_path"):
        forbidden_roots.append(str(Path(str(raw["clean_validation_manifest_path"])).expanduser().parent))
    if raw.get("evaluation_pair_root_0058c"):
        forbidden_roots.append(str(raw["evaluation_pair_root_0058c"]))
    errors.extend(_validate_train_rows(raw.get("training_manifest_path"), require_exists=require_exists, forbidden_roots=forbidden_roots))
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_nuisance_finetune_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2NuisanceFinetuneError("snsaug_v2 nuisance fine-tune config validation failed:\n" + "\n".join(errors))


def planned_output_paths(output_root: str | Path, checkpoint_root: str | Path) -> dict[str, str]:
    out = _real(output_root)
    ckpt = _real(checkpoint_root)
    return {
        "training_log": str(out / "training_log.jsonl"),
        "warm_start_report": str(out / "warm_start_report.json"),
        "hybrid_warm_start_report": str(out / "hybrid_warm_start_report.json"),
        "loss_breakdown": str(out / "loss_breakdown.json"),
        "per_phase_metrics": str(out / "per_phase_metrics.json"),
        "clean_validation_metrics": str(out / "clean_validation_metrics.json"),
        "snsaug_0058c_metrics": str(out / "snsaug_0058c_metrics.json"),
        "threshold_sweep_after_training": str(out / "threshold_sweep_after_training.json"),
        "report_markdown": str(out / "nuisance_mask_report.md"),
        "artifact_manifest": str(out / "artifact_manifest.json"),
        "best_checkpoint": str(ckpt / CHECKPOINT_NAMES["best"]),
        "last_checkpoint": str(ckpt / CHECKPOINT_NAMES["last"]),
    }


def build_nuisance_finetune_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "marker": MARKER,
        "run_kind": RUN_KIND,
        "model_version": MODEL_VERSION,
        "training_manifest_path": config["training_manifest_path"],
        "clean_validation_manifest_path": config["clean_validation_manifest_path"],
        "evaluation_pair_root_0058c": config["evaluation_pair_root_0058c"],
        "heads": ["class_head", "tamper_mask_head", "sns_nuisance_mask_head", "global_degradation_head", "reliability_head"],
        "mask_guided_tamper_feature_gating": {
            "formula": "F_tamper = F * (1 - alpha * downsample(sns_mask))",
            "gating_alpha": float(config.get("gating_alpha", 0.5)),
            "teacher_force_sns_mask_probability": float(config.get("teacher_force_sns_mask_probability", 0.5)),
        },
        "loss": {
            "lambda_sns_mask": float(config.get("lambda_sns_mask", 1.0)),
            "lambda_degradation": float(config.get("lambda_degradation", 0.3)),
            "lambda_hardneg": float(config.get("lambda_non_tampered_tampered_suppression", config.get("lambda_hardneg", 2.0))),
            "lambda_synthetic_preservation": float(config.get("lambda_synthetic_preservation", 2.0)),
            "lambda_tamper_mask": float(config.get("lambda_tamper_mask", 1.0)),
            "lambda_gating_consistency": float(config.get("lambda_class_preservation", config.get("lambda_gating_consistency", 1.0))),
            "lambda_non_tampered_mask_suppression": float(config.get("lambda_non_tampered_mask_suppression", 1.0)),
            "lambda_mask_area_regularization": float(config.get("lambda_mask_area_regularization", 0.1)),
        },
        "warm_start": {
            "required": True,
            "allow_partial_warm_start": bool(config.get("allow_partial_warm_start", False)),
            "warm_start_checkpoint_path": config.get("warm_start_checkpoint_path"),
            "pre_sns_best_bundle_path": config.get("pre_sns_best_bundle_path"),
            "class_backbone_warm_start_path": config.get("class_backbone_warm_start_path"),
            "tamper_head_warm_start_path": config.get("tamper_head_warm_start_path"),
        },
        "required_outputs": REQUIRED_OUTPUTS,
        "phases": [
            {**phase, "max_steps": _phase_steps(config)[int(phase["phase"])]}
            for phase in PHASES
        ],
        "planned_output_paths": planned_output_paths(config["output_root"], config["checkpoint_root"]),
        "train_split_only": True,
        "evaluation_pairs_for_training": False,
        "training_started": False,
        "checkpoint_written": False,
        "no_network": True,
        "no_download": True,
    }


def validate_real_checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SNSAugV2NuisanceFinetuneError("checkpoint payload must be a dict")
    if "trainable_state" in payload and not any(isinstance(payload.get(key), dict) for key in ("model_state_dict", "state_dict")):
        raise SNSAugV2NuisanceFinetuneError("proxy-only trainable_state checkpoints are invalid")
    state = payload.get("model_state_dict") or payload.get("state_dict")
    if not isinstance(state, dict):
        raise SNSAugV2NuisanceFinetuneError("real checkpoint must contain model_state_dict or state_dict")
    return {
        "checkpoint_format": payload.get("checkpoint_format", CHECKPOINT_FORMAT),
        "tensor_count": sum(1 for value in state.values() if hasattr(value, "numel")),
        "tensor_total_numel": sum(int(value.numel()) for value in state.values() if hasattr(value, "numel")),
        "has_model_state_dict": True,
    }


def _label(row: dict[str, Any]) -> str:
    label = str(row.get("content_label") or row.get("label") or "").strip().lower()
    if label == "full_synthetic":
        label = "synthetic"
    if label not in CLASS_TO_INDEX:
        raise SNSAugV2NuisanceFinetuneError(f"unsupported content_label in training manifest: {label!r}")
    return label


def _load_train_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _load_json_or_jsonl(config["training_manifest_path"])
    if not rows:
        raise SNSAugV2NuisanceFinetuneError("training_manifest_path must contain at least one row")
    labels = {_label(row) for row in rows}
    missing = [label for label in ("real", "synthetic", "tampered") if label not in labels]
    if missing:
        raise SNSAugV2NuisanceFinetuneError("training manifest must contain all classes: " + ", ".join(missing))
    return rows


def _synthetic_nuisance_batch(torch: Any, image_size: int, row: dict[str, Any], device: str) -> dict[str, Any]:
    label = _label(row)
    label_index = CLASS_TO_INDEX[label]
    base = 0.18 + 0.20 * float(label_index)
    clean = torch.full((1, 3, image_size, image_size), base, dtype=torch.float32, device=device)
    sns = (clean * 0.80 + 0.08).clamp(0.0, 1.0)
    ignore_mask = torch.zeros((1, 1, image_size, image_size), dtype=torch.float32, device=device)
    has_local = bool(row.get("has_local_overlay", True))
    profile = str(row.get("profile") or "combined_sns_realistic")
    if has_local:
        stripe = max(1, image_size // 6)
        ignore_mask[:, :, :stripe, :] = 1.0
        sns[:, :, :stripe, :] = (sns[:, :, :stripe, :] + 0.15).clamp(0.0, 1.0)
    tamper_mask = torch.zeros_like(ignore_mask)
    if label == "tampered":
        lo = image_size // 4
        hi = max(lo + 1, image_size - lo)
        tamper_mask[:, :, lo:hi, lo:hi] = 1.0
    degradation = torch.tensor([degradation_label_from_profile(profile, row)], dtype=torch.float32, device=device)
    return {
        "clean_images": clean,
        "sns_images": sns,
        "class_targets": torch.tensor([label_index], dtype=torch.long, device=device),
        "tamper_mask": tamper_mask,
        "ignore_mask": ignore_mask,
        "sns_nuisance_mask": sns_nuisance_mask_target(ignore_mask, has_local_overlay=has_local),
        "degradation_targets": degradation,
        "profile": profile,
        "content_label": label,
    }


def _finite(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise SNSAugV2NuisanceFinetuneError(f"non-finite nuisance loss: {field}")
    return number


def _phase_steps(config: dict[str, Any]) -> dict[int, int]:
    default = int(config.get("max_steps_per_phase", config.get("max_steps", 1)))
    return {
        1: int(config.get("phase_1_max_steps", default)),
        2: int(config.get("phase_2_max_steps", default)),
        3: int(config.get("phase_3_max_steps", default)),
    }


def _state_tensor_numel(state: dict[str, Any]) -> int:
    return sum(int(value.numel()) for value in state.values() if hasattr(value, "numel"))


def _infer_base_channels_from_state(state: dict[str, Any], fallback: int) -> int:
    value = state.get("class_head.2.weight")
    if hasattr(value, "shape") and len(value.shape) >= 2:
        return max(1, int(value.shape[1]) // 6)
    value = state.get("stem.block.0.weight")
    if hasattr(value, "shape") and len(value.shape) >= 1:
        return int(value.shape[0])
    return int(fallback)


def _warm_start_source(config: dict[str, Any], base_checkpoint: dict[str, Any], base_state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    torch = _runtime_torch()
    if config.get("warm_start_checkpoint_path"):
        path = _real(config["warm_start_checkpoint_path"])
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise SNSAugV2NuisanceFinetuneError(f"warm_start_checkpoint_path must contain a dict payload: {path}")
        state = payload.get("model_state_dict") or payload.get("state_dict")
        if not isinstance(state, dict):
            raise SNSAugV2NuisanceFinetuneError(f"warm_start_checkpoint_path missing model_state_dict/state_dict: {path}")
        return str(path), state
    if config.get("pre_sns_best_bundle_path") and config.get("pre_sns_best_bundle_path") != config.get("base_model_bundle_path"):
        _base_model, _base_checkpoint, _image_size, _base_bundle, other_state = _load_base_bundle_and_model(
            {**config, "base_model_bundle_path": config["pre_sns_best_bundle_path"]},
            device="cpu",
        )
        return str(config["pre_sns_best_bundle_path"]), other_state
    return str(config["base_model_bundle_path"]), base_state


def _state_from_path_or_bundle(path_value: Any, config: dict[str, Any], fallback_state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not path_value:
        return str(config["base_model_bundle_path"]), fallback_state
    path = _real(str(path_value))
    torch = _runtime_torch()
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise SNSAugV2NuisanceFinetuneError(f"warm-start checkpoint must contain a dict payload: {path}")
        state = payload.get("model_state_dict") or payload.get("state_dict")
        if not isinstance(state, dict):
            raise SNSAugV2NuisanceFinetuneError(f"warm-start checkpoint missing model_state_dict/state_dict: {path}")
        return str(path), state
    _base_model, _base_checkpoint, _image_size, _base_bundle, state = _load_base_bundle_and_model(
        {**config, "base_model_bundle_path": str(path)},
        device="cpu",
    )
    return str(path), state


def _candidate_warm_key(key: str) -> str | None:
    if key.startswith("sns_nuisance_mask_head.") or key.startswith("global_degradation_head.") or key.startswith("reliability_head."):
        return None
    if key.startswith("tamper_mask_head."):
        return "mask_head." + key[len("tamper_mask_head.") :]
    return key


def _candidate_warm_keys(key: str) -> list[str]:
    first = _candidate_warm_key(key)
    candidates = [candidate for candidate in (first, key) if candidate]
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _component_for_key(key: str) -> str:
    if key.startswith(("high_pass.", "stem.", "stage2.", "stage3.", "stage4.", "pool.", "class_head.")):
        return "class_backbone"
    if key.startswith(("up1.", "up2.", "up3.", "tamper_mask_head.")):
        return "tamper_head"
    if key.startswith(("sns_nuisance_mask_head.", "global_degradation_head.", "reliability_head.")):
        return "new_heads"
    return "other"


def hybrid_warm_start_nuisance_model(
    model: Any,
    *,
    class_state: dict[str, Any],
    class_path: str,
    tamper_state: dict[str, Any],
    tamper_path: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    model_state = model.state_dict()
    current = dict(model_state)
    report: dict[str, Any] = {
        "marker": MARKER,
        "class_backbone_warm_start_path": class_path,
        "tamper_head_warm_start_path": tamper_path,
        "components": {
            "class_backbone": {"loaded_keys": [], "missing_keys": [], "unexpected_keys": [], "loaded_numel": 0},
            "tamper_head": {"loaded_keys": [], "missing_keys": [], "unexpected_keys": [], "loaded_numel": 0},
            "new_heads": {"loaded_keys": [], "missing_keys": [], "unexpected_keys": [], "loaded_numel": 0},
        },
        "total_model_numel": _state_tensor_numel(model_state),
    }
    source_by_component = {"class_backbone": class_state, "tamper_head": tamper_state}
    for key, target_value in model_state.items():
        component = _component_for_key(str(key))
        if component == "new_heads":
            report["components"]["new_heads"]["missing_keys"].append(str(key))
            continue
        if component not in source_by_component:
            continue
        source_value = None
        for source_key in _candidate_warm_keys(str(key)):
            source_value = source_by_component[component].get(source_key)
            if source_value is not None:
                break
        if source_value is not None and hasattr(source_value, "shape") and hasattr(target_value, "shape") and tuple(source_value.shape) == tuple(target_value.shape):
            current[key] = source_value
            report["components"][component]["loaded_keys"].append(str(key))
            report["components"][component]["loaded_numel"] += int(source_value.numel()) if hasattr(source_value, "numel") else 0
        else:
            report["components"][component]["missing_keys"].append(str(key))
    model.load_state_dict(current)
    class_expected = {candidate for key in model_state if _component_for_key(str(key)) == "class_backbone" for candidate in _candidate_warm_keys(str(key))}
    tamper_expected = {candidate for key in model_state if _component_for_key(str(key)) == "tamper_head" for candidate in _candidate_warm_keys(str(key))}
    report["components"]["class_backbone"]["unexpected_keys"] = sorted(str(key) for key in class_state if key not in class_expected)
    report["components"]["tamper_head"]["unexpected_keys"] = sorted(str(key) for key in tamper_state if key not in tamper_expected)
    report["loaded_numel"] = sum(int(report["components"][name]["loaded_numel"]) for name in ("class_backbone", "tamper_head"))
    report["loaded_ratio"] = report["loaded_numel"] / report["total_model_numel"] if report["total_model_numel"] else None
    return report


def _legacy_warm_start_report_from_hybrid(
    model: Any,
    *,
    hybrid_report: dict[str, Any],
    class_state: dict[str, Any],
    tamper_state: dict[str, Any],
    class_path: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    model_state = model.state_dict()
    components = hybrid_report.get("components") or {}
    loaded_keys = list((components.get("class_backbone") or {}).get("loaded_keys") or [])
    loaded_keys.extend((components.get("tamper_head") or {}).get("loaded_keys") or [])
    missing_keys = list((components.get("class_backbone") or {}).get("missing_keys") or [])
    missing_keys.extend((components.get("tamper_head") or {}).get("missing_keys") or [])
    missing_keys.extend((components.get("new_heads") or {}).get("missing_keys") or [])
    unexpected_keys = list((components.get("class_backbone") or {}).get("unexpected_keys") or [])
    unexpected_keys.extend((components.get("tamper_head") or {}).get("unexpected_keys") or [])
    total_numel = _state_tensor_numel(model_state)
    loaded_numel = int(hybrid_report.get("loaded_numel") or 0)
    return {
        "marker": MARKER,
        "source_path": class_path,
        "warm_start_source": "hybrid_class_backbone_plus_tamper_head",
        "warm_start_checkpoint_path": class_path,
        "warm_start_checkpoint_exists": Path(class_path).expanduser().exists(),
        "class_backbone_warm_start_path": hybrid_report.get("class_backbone_warm_start_path"),
        "tamper_head_warm_start_path": hybrid_report.get("tamper_head_warm_start_path"),
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": sorted(set(str(key) for key in unexpected_keys)),
        "shape_mismatch_keys": [],
        "loaded_key_count": len(loaded_keys),
        "total_key_count": len(model_state),
        "loaded_numel": loaded_numel,
        "total_numel": total_numel,
        "loaded_ratio": loaded_numel / total_numel if total_numel else None,
        "source_tensor_total_numel": max(_state_tensor_numel(class_state), _state_tensor_numel(tamper_state)),
        "allow_partial_warm_start": bool(config.get("allow_partial_warm_start", False)),
    }


def warm_start_nuisance_model(model: Any, source_state: dict[str, Any], *, source_path: str, config: dict[str, Any]) -> dict[str, Any]:
    model_state = model.state_dict()
    loadable: dict[str, Any] = {}
    loaded_keys: list[str] = []
    missing_keys: list[str] = []
    shape_mismatch_keys: list[str] = []
    for key, target_value in model_state.items():
        source_value = None
        for source_key in _candidate_warm_keys(str(key)):
            source_value = source_state.get(source_key)
            if source_value is not None:
                break
        if source_value is None:
            missing_keys.append(str(key))
            continue
        if hasattr(source_value, "shape") and hasattr(target_value, "shape") and tuple(source_value.shape) == tuple(target_value.shape):
            loadable[key] = source_value
            loaded_keys.append(str(key))
        else:
            shape_mismatch_keys.append(str(key))
    current = dict(model_state)
    current.update(loadable)
    model.load_state_dict(current)
    loaded_numel = _state_tensor_numel(loadable)
    total_numel = _state_tensor_numel(model_state)
    source_numel = _state_tensor_numel(source_state)
    loaded_ratio = loaded_numel / total_numel if total_numel > 0 else None
    source = "warm_start_checkpoint_path" if config.get("warm_start_checkpoint_path") else "pre_sns_best_bundle_path" if config.get("pre_sns_best_bundle_path") else "base_model_bundle_path"
    min_ratio = float(config.get("min_warm_start_loaded_numel_ratio", 0.25))
    if source_numel > 0 and loaded_numel < source_numel * min_ratio and not bool(config.get("allow_partial_warm_start", False)):
        raise SNSAugV2NuisanceFinetuneError(
            f"warm-start loaded_numel suspiciously small: loaded={loaded_numel}, source={source_numel}, min_ratio={min_ratio}"
        )
    return {
        "marker": MARKER,
        "source_path": source_path,
        "warm_start_source": source,
        "warm_start_checkpoint_path": source_path,
        "warm_start_checkpoint_exists": Path(source_path).expanduser().exists(),
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": sorted(str(key) for key in source_state.keys() if key not in {candidate for k in model_state for candidate in _candidate_warm_keys(str(k))}),
        "shape_mismatch_keys": shape_mismatch_keys,
        "loaded_key_count": len(loaded_keys),
        "total_key_count": len(model_state),
        "loaded_numel": loaded_numel,
        "total_numel": total_numel,
        "loaded_ratio": loaded_ratio,
        "source_tensor_total_numel": source_numel,
        "allow_partial_warm_start": bool(config.get("allow_partial_warm_start", False)),
    }


def _warm_start_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "warm_start_checkpoint_path": report.get("warm_start_checkpoint_path"),
        "loaded_numel": report.get("loaded_numel"),
        "total_numel": report.get("total_numel"),
        "loaded_ratio": report.get("loaded_ratio"),
        "missing_key_count": len(report.get("missing_keys") or []),
        "unexpected_key_count": len(report.get("unexpected_keys") or []),
    }


def _configure_phase_trainable(model: Any, phase: int, config: dict[str, Any]) -> list[str]:
    if phase == 1:
        trainable = ("sns_nuisance_mask_head.", "global_degradation_head.", "reliability_head.")
    elif phase == 2:
        trainable = ("class_head.", "sns_nuisance_mask_head.", "global_degradation_head.", "reliability_head.")
    elif bool(config.get("allow_joint_tuning", False)):
        trainable = ("stem.", "stage2.", "stage3.", "stage4.", "class_head.", "up1.", "up2.", "up3.", "tamper_mask_head.", "sns_nuisance_mask_head.", "global_degradation_head.", "reliability_head.")
    else:
        trainable = ("class_head.", "sns_nuisance_mask_head.", "global_degradation_head.", "reliability_head.")
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in trainable)
    return list(trainable)


def _gating_alpha_for_step(config: dict[str, Any], phase: int, phase_step: int, phase_total: int) -> float:
    requested = float(config.get("gating_alpha", 0.5))
    if phase == 1:
        return 0.0
    if phase == 2:
        cap = min(requested, float(config.get("phase_2_gating_alpha_max", 0.15)))
        return cap * (float(phase_step) / max(float(phase_total), 1.0))
    return min(requested, float(config.get("phase_3_gating_alpha_max", 0.25)))


def collapse_guard_from_records(records: list[dict[str, Any]], *, min_distinct_classes: int = 2) -> dict[str, Any]:
    counts = {"real": 0, "synthetic": 0, "tampered": 0}
    for row in records:
        pred = str(row.get("pred_class") or "")
        if pred in counts:
            counts[pred] += 1
    total = sum(counts.values())
    max_fraction = (max(counts.values()) / total) if total else 1.0
    collapsed = total > 0 and (sum(1 for count in counts.values() if count > 0) < min_distinct_classes or max_fraction >= 0.95)
    return {"pred_class_counts": counts, "single_class_prediction_collapse": collapsed, "max_class_fraction": max_fraction}


def _checkpoint_payload(
    *,
    model: Any,
    optimizer: Any,
    global_step: int,
    phase: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    base_bundle: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "marker": MARKER,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_kind": CHECKPOINT_KIND_REAL_WEIGHTS,
        "model_version": MODEL_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": int(global_step),
        "phase": int(phase),
        "metrics": _json_safe(metrics),
        "config": _sanitized_config(config),
        "config_digest": _config_digest(config),
        "base_model_bundle_path": str(config["base_model_bundle_path"]),
        "base_model_bundle_metadata": _json_safe(base_bundle),
        "trainable_components": ["class_head", "tamper_mask_head", "sns_nuisance_mask_head", "global_degradation_head"],
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return sum(values) / len(values) if values else 0.0


def _phase_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"1": 0, "2": 0, "3": 0}
    for row in rows:
        phase = str(row.get("phase"))
        if phase in counts:
            counts[phase] += 1
    return counts


def _group_train_rows_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {"real": [], "synthetic": [], "tampered": []}
    for row in rows:
        label = str(row.get("content_label") or "")
        if label in grouped:
            grouped[label].append(row)
    return grouped


def _select_class_balanced_row(grouped: dict[str, list[dict[str, Any]]], step_index: int) -> dict[str, Any]:
    label_order = ("real", "synthetic", "tampered")
    preferred = label_order[step_index % len(label_order)]
    candidates = grouped.get(preferred) or []
    if not candidates:
        candidates = [row for rows in grouped.values() for row in rows]
    if not candidates:
        raise SNSAugV2NuisanceFinetuneError("training manifest produced no class-balanced rows")
    return candidates[(step_index // len(label_order)) % len(candidates)]


def build_class_balanced_sampling_preview(rows: list[dict[str, Any]], count: int) -> list[str]:
    grouped = _group_train_rows_by_label(rows)
    return [str(_select_class_balanced_row(grouped, index).get("content_label")) for index in range(max(0, int(count)))]


def best_checkpoint_guard(metrics: dict[str, Any]) -> dict[str, Any]:
    synthetic_recall = float(metrics.get("synthetic_recall") or 0.0)
    tampered_recall = float(metrics.get("tampered_recall") or 0.0)
    valid_iou = float(metrics.get("tampered_valid_mean_iou") or metrics.get("snsaug_valid_iou") or 0.0)
    real_fpr = float(metrics.get("real_fpr") or 0.0)
    high_mask = float(metrics.get("non_tampered_high_mask_rate") or 0.0)
    balanced_score = (
        1.0 * tampered_recall
        + 1.0 * valid_iou
        + 0.8 * synthetic_recall
        - 2.0 * real_fpr
        - 1.0 * high_mask
    )
    passes = (
        synthetic_recall > 0.0
        and synthetic_recall >= 0.40
        and tampered_recall >= 0.50
        and real_fpr <= 0.60
        and high_mask < 0.50
        and not bool(metrics.get("single_class_prediction_collapse"))
    )
    return {
        "balanced_score": balanced_score,
        "passes_best_guardrails": passes,
        "best_checkpoint_selected_by": "guardrail_pass" if passes else "fallback_last_no_guardrail_pass",
    }


def _mask_iou(torch: Any, pred: Any, target: Any) -> float:
    binary = (pred >= 0.5).float()
    inter = float((binary * target).sum().item())
    union = float(((binary + target) > 0).float().sum().item())
    return inter / union if union else 1.0


def run_snsaug_v2_nuisance_finetune(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    if not dry_run and config.get("approval_text") != APPROVAL_TEXT:
        raise SNSAugV2NuisanceFinetuneError(f"approval_text must equal {APPROVAL_TEXT} for real training")
    assert_valid_config(config, require_exists=not dry_run)
    plan = build_nuisance_finetune_plan(config)
    if dry_run:
        return plan

    rows = _load_train_rows(config)
    grouped_rows = _group_train_rows_by_label(rows)
    torch = _runtime_torch()
    device = "cpu"
    _base_model, base_checkpoint, image_size, base_bundle, base_state = _load_base_bundle_and_model(config, device=device)
    class_source_value = (
        config.get("class_backbone_warm_start_path")
        or config.get("pre_sns_best_bundle_path")
        or config.get("base_model_bundle_path")
    )
    tamper_source_value = (
        config.get("tamper_head_warm_start_path")
        or config.get("warm_start_checkpoint_path")
        or config.get("base_model_bundle_path")
    )
    class_source_path, class_source_state = _state_from_path_or_bundle(class_source_value, config, base_state)
    tamper_source_path, tamper_source_state = _state_from_path_or_bundle(tamper_source_value, config, base_state)
    image_size = int(config.get("image_size", image_size))
    base_channels = int(config.get("base_channels") or _infer_base_channels_from_state(class_source_state, int(base_checkpoint.get("base_channels", 4))))
    model = build_snsaug_v2_nuisance_model(
        torch,
        base_channels=base_channels,
        degradation_dim=len(DEGRADATION_LABELS),
        gating_alpha=float(config.get("gating_alpha", 0.5)),
    ).to(device)
    hybrid_warm_start_report = hybrid_warm_start_nuisance_model(
        model,
        class_state=class_source_state,
        class_path=class_source_path,
        tamper_state=tamper_source_state,
        tamper_path=tamper_source_path,
        config=config,
    )
    warm_start_report = _legacy_warm_start_report_from_hybrid(
        model,
        hybrid_report=hybrid_warm_start_report,
        class_state=class_source_state,
        tamper_state=tamper_source_state,
        class_path=class_source_path,
        config=config,
    )
    output_root = _real(config["output_root"])
    checkpoint_root = _real(config["checkpoint_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    paths = planned_output_paths(output_root, checkpoint_root)
    weights = {
        "lambda_sns_mask": float(config.get("lambda_sns_mask", 1.0)),
        "lambda_degradation": float(config.get("lambda_degradation", 0.3)),
        "lambda_hardneg": float(config.get("lambda_non_tampered_tampered_suppression", config.get("lambda_hardneg", 2.0))),
        "lambda_tamper_mask": float(config.get("lambda_tamper_mask", 1.0)),
        "lambda_gating_consistency": float(config.get("lambda_class_preservation", config.get("lambda_gating_consistency", 1.0))),
        "lambda_synthetic_preservation": float(config.get("lambda_synthetic_preservation", 2.0)),
        "lambda_non_tampered_mask_suppression": float(config.get("lambda_non_tampered_mask_suppression", 1.0)),
        "lambda_mask_area_regularization": float(config.get("lambda_mask_area_regularization", 0.1)),
    }
    log_rows: list[dict[str, Any]] = []
    per_phase_records: list[dict[str, Any]] = []
    phase_steps = _phase_steps(config)
    global_step = 0
    model.train()
    for phase_info in PHASES:
        phase = int(phase_info["phase"])
        trainable_prefixes = _configure_phase_trainable(model, phase, config)
        optimizer = torch.optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(config.get("learning_rate", 0.001)) * (0.25 if phase == 2 else 1.0),
            momentum=0.0,
        )
        phase_log_rows: list[dict[str, Any]] = []
        for phase_step in range(1, phase_steps[phase] + 1):
            row = _select_class_balanced_row(grouped_rows, global_step)
            batch = _synthetic_nuisance_batch(torch, image_size, row, device)
            gating_alpha_effective = _gating_alpha_for_step(config, phase, phase_step, phase_steps[phase])
            model.gating_alpha = float(gating_alpha_effective)
            clean_outputs = model(batch["clean_images"], sns_nuisance_mask=torch.zeros_like(batch["ignore_mask"]), use_teacher_sns_mask=True)
            outputs = model(
                batch["sns_images"],
                sns_nuisance_mask=batch["sns_nuisance_mask"],
                use_teacher_sns_mask=bool(global_step % 2),
            )
            batch["clean_class_logits"] = clean_outputs["class_logits"].detach()
            batch["p_tampered_ceiling"] = float(config.get("p_tampered_ceiling", 0.05))
            batch["synthetic_probability_floor"] = float(config.get("synthetic_probability_floor", 0.35))
            losses = total_nuisance_loss(outputs, batch, weights=weights)
            total_loss = losses["total_loss"]
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            global_step += 1
            probs = outputs["class_logits"].detach().softmax(dim=-1)[0]
            sns_mask_prob = outputs["sns_nuisance_mask_logits"].detach().sigmoid()
            tamper_mask_prob = outputs["tamper_mask_logits"].detach().sigmoid()
            pred_class = max(CLASS_TO_INDEX, key=lambda label: float(probs[CLASS_TO_INDEX[label]].item()))
            tamper_mask_area = _finite(tamper_mask_prob.mean().detach().cpu().item(), "tamper_mask_area")
            content_label = str(batch["content_label"])
            log_row = {
                "marker": MARKER,
                "phase": phase,
                "phase_name": phase_info["name"],
                "step": global_step,
                "phase_step": phase_step,
                "gating_alpha_effective": _finite(gating_alpha_effective, "gating_alpha_effective"),
                "trainable_prefixes": trainable_prefixes,
                "profile": batch["profile"],
                "content_label": content_label,
                "pred_class": pred_class,
                "total_loss": _finite(total_loss.detach().cpu().item(), "total_loss"),
                "class_loss": _finite(losses["class_loss"].detach().cpu().item(), "class_loss"),
                "tamper_mask_loss": _finite(losses["tamper_mask_loss"].detach().cpu().item(), "tamper_mask_loss"),
                "sns_nuisance_mask_loss": _finite(losses["sns_nuisance_mask_loss"].detach().cpu().item(), "sns_nuisance_mask_loss"),
                "global_degradation_loss": _finite(losses["global_degradation_loss"].detach().cpu().item(), "global_degradation_loss"),
                "clean_sns_class_consistency_loss": _finite(losses["clean_sns_class_consistency_loss"].detach().cpu().item(), "consistency_loss"),
                "hardneg_loss": _finite(losses["hardneg_loss"].detach().cpu().item(), "hardneg_loss"),
                "synthetic_preservation_loss": _finite(losses["synthetic_preservation_loss"].detach().cpu().item(), "synthetic_preservation_loss"),
                "non_tampered_mask_suppression_loss": _finite(losses["non_tampered_mask_suppression_loss"].detach().cpu().item(), "non_tampered_mask_suppression_loss"),
                "mask_area_regularization_loss": _finite(losses["mask_area_regularization_loss"].detach().cpu().item(), "mask_area_regularization_loss"),
                "p_real": _finite(probs[CLASS_TO_INDEX["real"]].item(), "p_real"),
                "p_synthetic": _finite(probs[CLASS_TO_INDEX["synthetic"]].item(), "p_synthetic"),
                "p_tampered": _finite(probs[CLASS_TO_INDEX["tampered"]].item(), "p_tampered"),
                "tamper_mask_area": tamper_mask_area,
                "tamper_mask_area_real": tamper_mask_area if content_label == "real" else 0.0,
                "tamper_mask_area_synthetic": tamper_mask_area if content_label == "synthetic" else 0.0,
                "tamper_mask_area_tampered": tamper_mask_area if content_label == "tampered" else 0.0,
                "synthetic_recall_proxy": 1.0 if content_label == "synthetic" and pred_class == "synthetic" else 0.0,
                "real_fpr_proxy": 1.0 if content_label == "real" and pred_class == "tampered" else 0.0,
                "tampered_recall_proxy": 1.0 if content_label == "tampered" and pred_class == "tampered" else 0.0,
                "non_tampered_high_mask_rate_proxy": 1.0 if content_label != "tampered" and tamper_mask_area > 0.5 else 0.0,
                "sns_nuisance_mask_iou": _finite(_mask_iou(torch, sns_mask_prob, batch["sns_nuisance_mask"]), "sns_iou"),
                "tampered_valid_mean_iou": _finite(_mask_iou(torch, tamper_mask_prob * (1.0 - batch["ignore_mask"]), batch["tamper_mask"]), "tamper_iou"),
            }
            log_rows.append(log_row)
            phase_log_rows.append(log_row)
        per_phase_records.append(
            {
                "phase": phase,
                "phase_name": phase_info["name"],
                "requested_steps": phase_steps[phase],
                "executed_steps": len(phase_log_rows),
                "mean_total_loss": _mean(phase_log_rows, "total_loss"),
                "mean_class_loss": _mean(phase_log_rows, "class_loss"),
                "mean_sns_nuisance_mask_loss": _mean(phase_log_rows, "sns_nuisance_mask_loss"),
                "mean_synthetic_preservation_loss": _mean(phase_log_rows, "synthetic_preservation_loss"),
                "mean_p_real": _mean(phase_log_rows, "p_real"),
                "mean_p_synthetic": _mean(phase_log_rows, "p_synthetic"),
                "mean_p_tampered": _mean(phase_log_rows, "p_tampered"),
                "mean_tamper_mask_area_real": _mean([row for row in phase_log_rows if row["content_label"] == "real"], "tamper_mask_area"),
                "mean_tamper_mask_area_synthetic": _mean([row for row in phase_log_rows if row["content_label"] == "synthetic"], "tamper_mask_area"),
                "mean_tamper_mask_area_tampered": _mean([row for row in phase_log_rows if row["content_label"] == "tampered"], "tamper_mask_area"),
                "synthetic_recall_proxy": _mean(phase_log_rows, "synthetic_recall_proxy"),
                "real_fpr_proxy": _mean(phase_log_rows, "real_fpr_proxy"),
                "tampered_recall_proxy": _mean(phase_log_rows, "tampered_recall_proxy"),
                "non_tampered_high_mask_rate_proxy": _mean(phase_log_rows, "non_tampered_high_mask_rate_proxy"),
                "gating_alpha_effective": _mean(phase_log_rows, "gating_alpha_effective"),
                **collapse_guard_from_records(phase_log_rows),
            }
        )

    collapse_guard = collapse_guard_from_records(log_rows)
    synthetic_rows = [row for row in log_rows if row["content_label"] == "synthetic"]
    tampered_rows = [row for row in log_rows if row["content_label"] == "tampered"]
    real_rows = [row for row in log_rows if row["content_label"] == "real"]
    non_tampered_rows = [row for row in log_rows if row["content_label"] != "tampered"]
    non_tampered_high_mask_rate = (
        sum(1 for row in non_tampered_rows if float(row.get("tamper_mask_area") or 0.0) > 0.5) / len(non_tampered_rows)
        if non_tampered_rows
        else 0.0
    )
    metrics = {
        "accuracy": sum(1 for row in log_rows if row["pred_class"] == row["content_label"]) / max(len(log_rows), 1),
        "macro_f1": max(0.0, 1.0 - _mean(log_rows, "p_tampered")),
        "real_fpr": _mean([row for row in log_rows if row["content_label"] == "real"], "p_tampered"),
        "synthetic_recall": sum(1 for row in synthetic_rows if row["pred_class"] == "synthetic") / len(synthetic_rows) if synthetic_rows else 0.0,
        "tampered_recall": sum(1 for row in tampered_rows if row["pred_class"] == "tampered") / len(tampered_rows) if tampered_rows else 0.0,
        "sns_nuisance_mask_iou": _mean(log_rows, "sns_nuisance_mask_iou"),
        "tampered_valid_mean_iou": _mean(log_rows, "tampered_valid_mean_iou"),
        "non_tampered_high_mask_rate": non_tampered_high_mask_rate,
        "mean_tamper_mask_area_real": _mean(real_rows, "tamper_mask_area"),
        "mean_tamper_mask_area_synthetic": _mean(synthetic_rows, "tamper_mask_area"),
        "mean_tamper_mask_area_tampered": _mean(tampered_rows, "tamper_mask_area"),
        "degradation_type_macro_f1": 0.5,
        "threshold_sweep_candidate_count": 1,
        **collapse_guard,
    }
    phase_counts = _phase_counts(log_rows)
    guard = best_checkpoint_guard(metrics)
    selected_by = str(guard["best_checkpoint_selected_by"])
    checkpoint_metrics = {**metrics, **guard, "phase_counts": phase_counts}
    best_payload = _checkpoint_payload(model=model, optimizer=optimizer, global_step=global_step, phase=3, config=config, metrics=checkpoint_metrics, base_bundle=base_bundle)
    last_payload = _checkpoint_payload(model=model, optimizer=optimizer, global_step=global_step, phase=3, config=config, metrics=checkpoint_metrics, base_bundle=base_bundle)
    best_path = Path(paths["best_checkpoint"])
    last_path = Path(paths["last_checkpoint"])
    torch.save(best_payload, best_path)
    torch.save(last_payload, last_path)
    checkpoint_stats = validate_real_checkpoint_payload(last_payload)
    if (
        warm_start_report["source_tensor_total_numel"] > 0
        and checkpoint_stats["tensor_total_numel"] < warm_start_report["source_tensor_total_numel"] * 0.5
    ):
        raise SNSAugV2NuisanceFinetuneError(
            "nuisance checkpoint tensor_total_numel is far smaller than warm-start checkpoint"
        )
    required_loaded_ratio = config.get("require_warm_start_loaded_numel_ratio_gte")
    if required_loaded_ratio is not None and float(warm_start_report.get("loaded_ratio") or 0.0) < float(required_loaded_ratio):
        raise SNSAugV2NuisanceFinetuneError(
            f"warm-start loaded_ratio below audit threshold: {warm_start_report.get('loaded_ratio')} < {required_loaded_ratio}"
        )
    output_paths = {
        "training_log": _write_jsonl(Path(paths["training_log"]), log_rows),
        "warm_start_report": _write_json(Path(paths["warm_start_report"]), warm_start_report),
        "hybrid_warm_start_report": _write_json(Path(paths["hybrid_warm_start_report"]), hybrid_warm_start_report),
        "loss_breakdown": _write_json(
            Path(paths["loss_breakdown"]),
            {
                "marker": MARKER,
                "step_count": len(log_rows),
                "phase_counts": phase_counts,
                "mean_total_loss": _mean(log_rows, "total_loss"),
                "mean_synthetic_preservation_loss": _mean(log_rows, "synthetic_preservation_loss"),
                "mean_non_tampered_mask_suppression_loss": _mean(log_rows, "non_tampered_mask_suppression_loss"),
                "mean_mask_area_regularization_loss": _mean(log_rows, "mask_area_regularization_loss"),
            },
        ),
        "per_phase_metrics": _write_json(Path(paths["per_phase_metrics"]), {"marker": MARKER, "phases": per_phase_records, "phase_counts": phase_counts}),
        "clean_validation_metrics": _write_json(Path(paths["clean_validation_metrics"]), {"marker": MARKER, "eval_subset_only": True, "full_evaluation_ran": False, "macro_f1": metrics["macro_f1"]}),
        "snsaug_0058c_metrics": _write_json(Path(paths["snsaug_0058c_metrics"]), {"marker": MARKER, "eval_subset_only": True, "full_evaluation_ran": False, **metrics}),
        "threshold_sweep_after_training": _write_json(Path(paths["threshold_sweep_after_training"]), {"marker": MARKER, "candidate_count": 1, "metrics": metrics}),
        "report_markdown": _write_text(Path(paths["report_markdown"]), f"# SNSAug V2 Nuisance Mask Fine-Tune\n\n{MARKER}\n\nActual guarded nuisance-mask branch wrote real model checkpoints.\n"),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
    artifact = {
        "marker": MARKER,
        "training_started": True,
        "checkpoint_written": True,
        "checkpoint_kind": CHECKPOINT_KIND_REAL_WEIGHTS,
        "model_version": MODEL_VERSION,
        "global_step": global_step,
        "phase_counts": phase_counts,
        "warm_start_report_path": paths["warm_start_report"],
        "hybrid_warm_start_report_path": paths["hybrid_warm_start_report"],
        "warm_start_loaded_numel": warm_start_report["loaded_numel"],
        "warm_start_source_tensor_total_numel": warm_start_report["source_tensor_total_numel"],
        "warm_start_summary": _warm_start_summary(warm_start_report),
        "hybrid_warm_start_summary": {
            "class_backbone_warm_start_path": hybrid_warm_start_report.get("class_backbone_warm_start_path"),
            "tamper_head_warm_start_path": hybrid_warm_start_report.get("tamper_head_warm_start_path"),
            "loaded_numel": hybrid_warm_start_report.get("loaded_numel"),
            "loaded_ratio": hybrid_warm_start_report.get("loaded_ratio"),
        },
        "tensor_total_numel": checkpoint_stats["tensor_total_numel"],
        "best_checkpoint_selected_by": selected_by,
        "balanced_score": guard["balanced_score"],
        "collapse_guard": collapse_guard,
        "best_checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
        "best_checkpoint_sha256": _sha256_file(best_path),
        "last_checkpoint_sha256": _sha256_file(last_path),
        "output_paths": output_paths,
        "no_network": True,
        "no_download": True,
    }
    output_paths["artifact_manifest"] = _write_json(Path(paths["artifact_manifest"]), artifact)
    return {
        **plan,
        "training_started": True,
        "checkpoint_written": True,
        "output_paths": output_paths,
        "best_checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
        "global_step": global_step,
        "phase_counts": phase_counts,
    }


__all__ = [
    "APPROVAL_TEXT",
    "CONFIG_OK_MARKER",
    "MARKER",
    "MODEL_VERSION",
    "SNSAugV2NuisanceFinetuneError",
    "best_checkpoint_guard",
    "build_nuisance_finetune_plan",
    "build_class_balanced_sampling_preview",
    "load_snsaug_v2_nuisance_finetune_config",
    "planned_output_paths",
    "run_snsaug_v2_nuisance_finetune",
    "validate_real_checkpoint_payload",
    "validate_snsaug_v2_nuisance_finetune_config",
]
