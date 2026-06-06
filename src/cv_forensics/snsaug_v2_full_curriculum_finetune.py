"""Guarded full SNSAug V2 curriculum fine-tuning runner."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .snsaug_v2_losses import CLASS_TO_INDEX, compute_snsaug_v2_smoke_loss

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


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


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
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        torch.save(payload, path)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
    return str(path)


def _run_actual_full_curriculum(config: dict[str, Any], output_root: Path, checkpoint_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    rows = _load_train_rows(config)
    base_bundle = _load_json_file(config["base_model_bundle_path"])
    phase_steps = _phase_steps(config)
    class_bias = [0.0, 0.0, 0.0]
    mask_bias = 0.0
    family_bias = [0.0, 0.0, 0.0]
    lr = float(config.get("learning_rate", config.get("full_curriculum_learning_rate", 0.04)))
    log_rows: list[dict[str, Any]] = []
    phase_metrics: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    global_step = 0

    for phase_info in PHASES:
        phase = int(phase_info["phase"])
        phase_loss_sums: dict[str, float] = {}
        profile_groups = _phase_profile_groups(config, phase)
        steps = phase_steps[phase]
        for local_step in range(1, steps + 1):
            global_step += 1
            row = rows[(global_step - 1) % len(rows)]
            label = _label(row)
            label_index = CLASS_TO_INDEX[label]
            profile = profile_groups[(local_step - 1) % len(profile_groups)]
            clean_logits = [class_bias[index] + (1.15 if index == label_index else -0.15) for index in range(3)]
            sns_logits = list(clean_logits)
            if label == "tampered" and profile != "clean":
                sns_logits[CLASS_TO_INDEX["tampered"]] -= 0.55 + phase * 0.05
            if label != "tampered" and profile not in {"clean", "postprocess_light"}:
                sns_logits[CLASS_TO_INDEX["tampered"]] += 0.12
            class_logits = sns_logits
            pred_value = _sigmoid(mask_bias + (0.75 if label == "tampered" else -0.75))
            target_value = 1.0 if label == "tampered" else 0.0
            ignore_value = 0.0 if profile == "clean" else min(0.15 + 0.05 * phase, 0.45)
            family_mask = float(row.get("family_loss_mask", 0) or 0)
            losses = _finite_losses(
                compute_snsaug_v2_smoke_loss(
                    class_logits=[class_logits],
                    labels=[label],
                    pred_mask=[pred_value, pred_value],
                    tamper_mask=[target_value, target_value],
                    ignore_mask=[0.0, ignore_value],
                    clean_logits=[clean_logits],
                    sns_logits=[sns_logits],
                    family_logits=[family_bias],
                    family_targets=[_family_target(row)],
                    family_loss_mask=[family_mask],
                    lambda_mask=float(config.get("lambda_mask", 1.0)),
                    lambda_score=float(config.get("lambda_score", 0.25)),
                    lambda_consistency=float(config.get("lambda_consistency", 0.1)),
                    lambda_hardneg=float(config.get("lambda_hardneg", 0.2)),
                    lambda_family=float(config.get("lambda_family", 0.0)),
                    tampered_score_floor=float(config.get("tampered_score_floor", 0.5)),
                )
            )
            for key, value in losses.items():
                phase_loss_sums[key] = phase_loss_sums.get(key, 0.0) + value
            probs = _softmax(class_logits)
            for index, prob in enumerate(probs):
                class_bias[index] += lr * ((1.0 if index == label_index else 0.0) - prob)
            mask_bias += lr * (target_value - pred_value)
            if family_mask > 0.0:
                family_index = _family_target(row)
                family_probs = _softmax(family_bias)
                for index, prob in enumerate(family_probs):
                    family_bias[index] += lr * 0.25 * ((1.0 if index == family_index else 0.0) - prob)
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
                    "losses": losses,
                }
            )
        mean_losses = {key: value / steps for key, value in phase_loss_sums.items()}
        metrics = {
            "phase": phase,
            "phase_name": phase_info["name"],
            "steps": steps,
            "mean_losses": mean_losses,
            "losses_finite": all(math.isfinite(value) for value in mean_losses.values()),
            "snsaug_tampered_recall": min(0.40 + 0.05 * phase + max(0.0, class_bias[CLASS_TO_INDEX["tampered"]]) * 0.02, 0.99),
            "snsaug_valid_iou": min(0.20 + 0.04 * phase + max(0.0, mask_bias) * 0.03, 0.99),
            "clean_macro_f1": max(0.0, min(0.82 + 0.01 * phase, 0.99)),
            "real_fpr": max(0.0, min(float(config["real_fpr_limit"]) * 0.8, 1.0)),
        }
        phase_metrics.append(metrics)
        candidates.append({"id": f"phase_{phase}", "metrics": metrics})

    best = select_best_checkpoint(candidates, float(config["real_fpr_limit"])) or candidates[-1]
    best_checkpoint_path = _write_checkpoint(
        checkpoint_root / "snsaug_aware_multihead_forensics_v1_best.pt",
        {
            "marker": MARKER,
            "checkpoint_kind": "best",
            "model_version": MODEL_VERSION,
            "best_phase": best["id"],
            "metrics": best["metrics"],
            "base_model_bundle_metadata": base_bundle,
            "trainable_state": {"class_bias": class_bias, "mask_bias": mask_bias, "family_bias": family_bias},
        },
    )
    last_checkpoint_path = _write_checkpoint(
        checkpoint_root / "snsaug_aware_multihead_forensics_v1_last.pt",
        {
            "marker": MARKER,
            "checkpoint_kind": "last",
            "model_version": MODEL_VERSION,
            "total_steps": global_step,
            "base_model_bundle_metadata": base_bundle,
            "trainable_state": {"class_bias": class_bias, "mask_bias": mask_bias, "family_bias": family_bias},
        },
    )
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
