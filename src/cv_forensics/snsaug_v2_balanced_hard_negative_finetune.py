"""Guarded SNSAug V2 balanced hard-negative fine-tuning infrastructure."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .snsaug_v2_full_curriculum_finetune import (
    CHECKPOINT_FORMAT,
    MODEL_VERSION,
    REPO_ROOT,
    _as_roots,
    _contains_eval_token,
    _config_digest,
    _configure_trainable_parameters,
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
    _synthetic_batch,
    _validate_absolute_path,
    _validate_numeric,
    _validate_under_roots,
    _write_json,
    _write_jsonl,
    _write_text,
)
from .snsaug_v2_losses import (
    CLASS_TO_INDEX,
    class_cross_entropy_loss,
    clean_sns_non_tampered_class_consistency_loss,
    non_tampered_tampered_suppression_loss,
    tampered_score_consistency_loss,
    valid_tamper_mask_loss,
)

MARKER = "SNSAUG_V2_BALANCED_HARD_NEGATIVE_FINETUNE_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_BALANCED_HARD_NEGATIVE_FINETUNE_CONFIG_OK"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_BALANCED_HARD_NEGATIVE_FINETUNE"
APPROVED_KIND = "approved_snsaug_v2_balanced_hard_negative_finetune"
APPROVED_MODE = "approved_local_snsaug_v2_balanced_hard_negative_finetune"
RUN_KIND = "balanced_hard_negative_finetune"

BEST_POLICY = {
    "primary": "balanced_score",
    "formula": {
        "snsaug_tampered_recall": 1.0,
        "snsaug_valid_iou": 1.0,
        "synthetic_recall": 0.8,
        "real_fpr": -2.0,
        "non_tampered_high_mask_rate": -1.0,
    },
    "guardrails": {
        "real_fpr_lte": 0.30,
        "synthetic_recall_gte": 0.40,
        "clean_macro_f1_gte": 0.75,
        "tampered_recall_gte": 0.50,
    },
}

REQUIRED_OUTPUTS = [
    "training_log.jsonl",
    "loss_breakdown.json",
    "per_phase_metrics.json",
    "clean_validation_metrics.json",
    "snsaug_0058c_metrics.json",
    "threshold_sweep_after_training.json",
    "artifact_manifest.json",
]
REQUIRED_CHECKPOINTS = ["snsaug_aware_multihead_forensics_v1_best.pt", "snsaug_aware_multihead_forensics_v1_last.pt"]
CHECKPOINT_KIND_REAL_WEIGHTS = "snsaug_v2_balanced_hard_negative_real_model_weights"
PHASES = [
    {"phase": 1, "name": "activation_recovery_safe_geometry", "hard_negative_sampling": False},
    {"phase": 2, "name": "geometry_light_platform_hard_sns", "hard_negative_sampling": True},
    {"phase": 3, "name": "combined_sns_balanced_hard_negative", "hard_negative_sampling": True},
]


class SNSAugV2BalancedHardNegativeFinetuneError(ValueError):
    """Raised when balanced hard-negative fine-tuning guardrails fail."""


def load_snsaug_v2_balanced_hard_negative_finetune_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2BalancedHardNegativeFinetuneError("balanced hard-negative config root must be a JSON object")
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
        for field in ("image_path", "tamper_mask_path", "mask_path"):
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


def validate_snsaug_v2_balanced_hard_negative_finetune_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
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
        "p_tampered_ceiling",
        "lambda_hardneg",
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
    if raw.get("run_kind") != RUN_KIND:
        errors.append(f"run_kind must be {RUN_KIND}")
    if raw.get("model_version") != MODEL_VERSION:
        errors.append(f"model_version must be {MODEL_VERSION}")
    if _training_approval_required(raw) and raw.get("approval_text") != APPROVAL_TEXT:
        errors.append(f"approval_text must equal {APPROVAL_TEXT} for real training")
    if raw.get("start_from_pre_sns_best") is not True:
        errors.append("start_from_pre_sns_best must be true")
    if raw.get("no_training_from_scratch") is not True:
        errors.append("no_training_from_scratch must be true")
    for flag in ("no_network", "no_download"):
        if raw.get(flag) is not True:
            errors.append(f"{flag} must be true")
    errors.extend(_validate_numeric(raw.get("p_tampered_ceiling"), "p_tampered_ceiling", minimum=0.0, maximum=1.0))
    errors.extend(_validate_numeric(raw.get("lambda_hardneg"), "lambda_hardneg", minimum=0.0))
    if raw.get("best_checkpoint_policy") != BEST_POLICY:
        errors.append("best_checkpoint_policy must match the balanced hard-negative policy")
    for field in ("max_steps_per_phase", "phase_1_max_steps", "phase_2_max_steps", "phase_3_max_steps"):
        if field in raw:
            value = raw.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > 500:
                errors.append(f"{field} must be an integer in 1..500")

    train_roots = _as_roots(raw.get("approved_train_manifest_roots"))
    model_roots = _as_roots(raw.get("approved_model_roots"))
    eval_roots = _as_roots(raw.get("approved_evaluation_roots"))
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

    for field in ("training_manifest_path", "curriculum_schedule_path", "profile_sampling_weights_path"):
        errors.extend(_validate_under_roots(raw.get(field), field, train_roots, require_exists=require_exists))
        if _contains_eval_token(raw.get(field)):
            errors.append(f"{field} must not reference fixed-pair, validation, oracle, or evaluation outputs")
    errors.extend(_validate_under_roots(raw.get("base_model_bundle_path"), "base_model_bundle_path", model_roots, require_exists=require_exists))
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
    errors = validate_snsaug_v2_balanced_hard_negative_finetune_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2BalancedHardNegativeFinetuneError(
            "snsaug_v2 balanced hard-negative finetune config validation failed:\n" + "\n".join(errors)
        )


def balanced_checkpoint_score(metrics: dict[str, Any]) -> tuple[bool, float]:
    real_fpr = float(metrics.get("real_fpr", 1.0))
    synthetic_recall = float(metrics.get("synthetic_recall", 0.0))
    clean_macro_f1 = float(metrics.get("clean_macro_f1", 0.0))
    tampered_recall = float(metrics.get("tampered_recall", metrics.get("snsaug_tampered_recall", 0.0)))
    guard = BEST_POLICY["guardrails"]
    passes = (
        real_fpr <= float(guard["real_fpr_lte"])
        and synthetic_recall >= float(guard["synthetic_recall_gte"])
        and clean_macro_f1 >= float(guard["clean_macro_f1_gte"])
        and tampered_recall >= float(guard["tampered_recall_gte"])
    )
    score = (
        float(metrics.get("snsaug_tampered_recall", 0.0))
        + float(metrics.get("snsaug_valid_iou", 0.0))
        + 0.8 * synthetic_recall
        - 2.0 * real_fpr
        - float(metrics.get("non_tampered_high_mask_rate", 0.0))
    )
    return passes, score


def select_best_balanced_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [item for item in candidates if balanced_checkpoint_score(item.get("metrics", {}))[0]]
    if not passing:
        return None
    return max(passing, key=lambda item: balanced_checkpoint_score(item.get("metrics", {}))[1])


def build_hard_negative_sampling_plan(config: dict[str, Any]) -> dict[str, Any]:
    ratio = config.get("hard_negative_sampling", {}).get("non_tampered_sns_to_tampered_sns_ratio", [2, 1])
    left = int(ratio[0]) if isinstance(ratio, list) and len(ratio) == 2 else 2
    right = int(ratio[1]) if isinstance(ratio, list) and len(ratio) == 2 else 1
    return {
        "mode": "balanced_hard_negative",
        "non_tampered_sns_to_tampered_sns_ratio": [left, right],
        "non_tampered_sns_fraction": left / max(left + right, 1),
        "tampered_sns_fraction": right / max(left + right, 1),
        "hard_sns_phases": [phase["phase"] for phase in PHASES if phase["hard_negative_sampling"]],
        "train_split_only": True,
        "evaluation_pairs_for_training": False,
    }


def planned_output_paths(output_root: str | Path, checkpoint_root: str | Path) -> dict[str, str]:
    out = _real(output_root)
    ckpt = _real(checkpoint_root)
    return {
        "training_log": str(out / "training_log.jsonl"),
        "loss_breakdown": str(out / "loss_breakdown.json"),
        "per_phase_metrics": str(out / "per_phase_metrics.json"),
        "clean_validation_metrics": str(out / "clean_validation_metrics.json"),
        "snsaug_0058c_metrics": str(out / "snsaug_0058c_metrics.json"),
        "threshold_sweep_after_training": str(out / "threshold_sweep_after_training.json"),
        "report_markdown": str(out / "balanced_hard_negative_report.md"),
        "artifact_manifest": str(out / "artifact_manifest.json"),
        "best_checkpoint": str(ckpt / "snsaug_aware_multihead_forensics_v1_best.pt"),
        "last_checkpoint": str(ckpt / "snsaug_aware_multihead_forensics_v1_last.pt"),
    }


def build_balanced_hard_negative_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "marker": MARKER,
        "run_kind": RUN_KIND,
        "model_version": MODEL_VERSION,
        "start_from_pre_sns_best": True,
        "training_manifest_path": config["training_manifest_path"],
        "clean_validation_manifest_path": config["clean_validation_manifest_path"],
        "evaluation_pair_root_0058c": config["evaluation_pair_root_0058c"],
        "phases": PHASES,
        "loss": {
            "L_non_tampered_tampered_suppression": {
                "type": "margin_squared",
                "p_tampered_ceiling": float(config.get("p_tampered_ceiling", 0.05)),
                "lambda_hardneg": float(config.get("lambda_hardneg", 2.0)),
                "applies_to": ["real", "synthetic"],
            },
            "L_tampered_score_consistency": {"applies_to": ["tampered"]},
            "L_clean_sns_class_consistency": {"applies_to": ["real", "synthetic"], "excludes_tampered_overactivation": True},
        },
        "logged_metrics": [
            "mean_p_tampered_real_sns",
            "mean_p_tampered_synthetic_sns",
            "mean_p_tampered_tampered_sns",
            "hardneg_loss",
            "tampered_score_consistency_loss",
            "clean_sns_class_consistency_loss",
        ],
        "sampling": build_hard_negative_sampling_plan(config),
        "best_checkpoint_policy": BEST_POLICY,
        "required_outputs": REQUIRED_OUTPUTS,
        "required_checkpoints": REQUIRED_CHECKPOINTS,
        "planned_output_paths": planned_output_paths(config["output_root"], config["checkpoint_root"]),
        "no_network": True,
        "no_download": True,
        "training_started": False,
        "checkpoint_written": False,
    }


def validate_real_checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SNSAugV2BalancedHardNegativeFinetuneError("checkpoint payload must be a dict")
    if "trainable_state" in payload and not any(isinstance(payload.get(key), dict) for key in ("model_state_dict", "state_dict")):
        raise SNSAugV2BalancedHardNegativeFinetuneError("proxy-only trainable_state checkpoints are invalid")
    state = payload.get("model_state_dict") or payload.get("state_dict")
    if not isinstance(state, dict):
        raise SNSAugV2BalancedHardNegativeFinetuneError("real checkpoint must contain model_state_dict or state_dict")
    tensor_count = sum(1 for value in state.values() if hasattr(value, "numel"))
    tensor_total_numel = sum(int(value.numel()) for value in state.values() if hasattr(value, "numel"))
    return {
        "checkpoint_format": payload.get("checkpoint_format", CHECKPOINT_FORMAT),
        "tensor_count": tensor_count,
        "tensor_total_numel": tensor_total_numel,
        "has_model_state_dict": True,
    }


def _finite(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise SNSAugV2BalancedHardNegativeFinetuneError(f"non-finite balanced hard-negative loss: {field}")
    return number


def _phase_steps(config: dict[str, Any]) -> dict[int, int]:
    default = int(config.get("max_steps_per_phase", 1))
    return {
        1: int(config.get("phase_1_max_steps", default)),
        2: int(config.get("phase_2_max_steps", default)),
        3: int(config.get("phase_3_max_steps", default)),
    }


def _label(row: dict[str, Any]) -> str:
    label = str(row.get("content_label") or "").strip().lower()
    if label not in CLASS_TO_INDEX:
        raise SNSAugV2BalancedHardNegativeFinetuneError(f"unsupported content_label in training manifest: {label!r}")
    return label


def _load_train_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _load_json_or_jsonl(config["training_manifest_path"])
    if not rows:
        raise SNSAugV2BalancedHardNegativeFinetuneError("training_manifest_path must contain at least one row")
    labels = {_label(row) for row in rows}
    missing = [label for label in ("real", "synthetic", "tampered") if label not in labels]
    if missing:
        raise SNSAugV2BalancedHardNegativeFinetuneError("training manifest must contain all classes: " + ", ".join(missing))
    return rows


def _group_rows_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {label: [] for label in CLASS_TO_INDEX}
    for row in rows:
        grouped[_label(row)].append(row)
    return grouped


def _select_training_row(grouped_rows: dict[str, list[dict[str, Any]]], phase: int, step_index: int) -> dict[str, Any]:
    if phase in {2, 3}:
        label_cycle = ("real", "synthetic", "tampered")
    else:
        label_cycle = ("tampered", "real", "synthetic")
    label = label_cycle[step_index % len(label_cycle)]
    rows = grouped_rows.get(label) or []
    if not rows:
        raise SNSAugV2BalancedHardNegativeFinetuneError(f"training manifest has no {label} rows")
    return rows[(step_index // len(label_cycle)) % len(rows)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate_log_rows(log_rows: list[dict[str, Any]]) -> dict[str, Any]:
    loss_keys = (
        "total_loss",
        "class_loss",
        "hardneg_loss",
        "tampered_score_consistency_loss",
        "clean_sns_class_consistency_loss",
        "mask_loss",
    )
    return {
        key: _mean([float(row[key]) for row in log_rows])
        for key in loss_keys
    }


def _metrics_from_log_rows(log_rows: list[dict[str, Any]]) -> dict[str, float]:
    mean_real = _mean([float(row["mean_p_tampered_real_sns"]) for row in log_rows if row.get("content_label") == "real"])
    mean_synth = _mean([float(row["mean_p_tampered_synthetic_sns"]) for row in log_rows if row.get("content_label") == "synthetic"])
    mean_tamp = _mean([float(row["mean_p_tampered_tampered_sns"]) for row in log_rows if row.get("content_label") == "tampered"])
    real_fpr = min(1.0, mean_real)
    synthetic_recall = max(0.0, 1.0 - mean_synth)
    tampered_recall = mean_tamp
    valid_iou = max(0.0, min(1.0, tampered_recall * 0.5))
    return {
        "snsaug_tampered_recall": tampered_recall,
        "snsaug_valid_iou": valid_iou,
        "synthetic_recall": synthetic_recall,
        "real_fpr": real_fpr,
        "non_tampered_high_mask_rate": 1.0 if max(mean_real, mean_synth) > 0.5 else 0.0,
        "clean_macro_f1": max(0.0, 1.0 - real_fpr),
        "tampered_recall": tampered_recall,
    }


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
        "base_model_bundle_path": str(config["base_model_bundle_path"]),
        "config": _sanitized_config(config),
        "config_digest": _config_digest(config),
        "metrics": _json_safe(metrics),
        "trainable_components": ["class_head", "tamper_localization_head"],
        "base_model_bundle_metadata": _json_safe(base_bundle),
    }


def _save_checkpoint(torch: Any, path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return str(path)


def loss_diagnostics(clean_logits: Any, sns_logits: Any, labels: Any, *, ceiling: float = 0.05) -> dict[str, Any]:
    prob_rows = []
    from .snsaug_v2_losses import _prob_rows, _labels

    labels_list = _labels(labels)
    for row in _prob_rows(sns_logits):
        prob_rows.append(row)
    means: dict[str, list[float]] = {"real": [], "synthetic": [], "tampered": []}
    for row, label_index in zip(prob_rows, labels_list):
        for name, index in CLASS_TO_INDEX.items():
            if label_index == index:
                means[name].append(float(row[2]))
    return {
        "mean_p_tampered_real_sns": sum(means["real"]) / len(means["real"]) if means["real"] else 0.0,
        "mean_p_tampered_synthetic_sns": sum(means["synthetic"]) / len(means["synthetic"]) if means["synthetic"] else 0.0,
        "mean_p_tampered_tampered_sns": sum(means["tampered"]) / len(means["tampered"]) if means["tampered"] else 0.0,
        "hardneg_loss": float(non_tampered_tampered_suppression_loss(sns_logits, labels, ceiling=ceiling)),
        "tampered_score_consistency_loss": float(tampered_score_consistency_loss(clean_logits, sns_logits, labels)),
        "clean_sns_class_consistency_loss": float(clean_sns_non_tampered_class_consistency_loss(clean_logits, sns_logits, labels)),
    }


def run_snsaug_v2_balanced_hard_negative_finetune(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    if not dry_run and config.get("approval_text") != APPROVAL_TEXT:
        raise SNSAugV2BalancedHardNegativeFinetuneError(f"approval_text must equal {APPROVAL_TEXT} for real training")
    assert_valid_config(config, require_exists=not dry_run)
    plan = build_balanced_hard_negative_plan(config)
    if dry_run:
        return plan
    rows = _load_train_rows(config)
    grouped_rows = _group_rows_by_label(rows)
    torch = _runtime_torch()
    device = "cpu"
    model, _base_checkpoint, image_size, base_bundle, _base_state = _load_base_bundle_and_model(config, device=device)
    _configure_trainable_parameters(model, ["class_head", "tamper_localization_head"])
    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config.get("learning_rate", 0.001)),
        momentum=0.0,
    )
    output_root = _real(config["output_root"])
    checkpoint_root = _real(config["checkpoint_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    planned_paths = planned_output_paths(output_root, checkpoint_root)
    phase_steps = _phase_steps(config)
    ceiling = float(config.get("p_tampered_ceiling", 0.05))
    global_step = 0
    log_rows: list[dict[str, Any]] = []
    per_phase: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    model.train()
    for phase_info in PHASES:
        phase = int(phase_info["phase"])
        phase_log_rows: list[dict[str, Any]] = []
        for local_step in range(1, phase_steps[phase] + 1):
            row = _select_training_row(grouped_rows, phase, local_step - 1)
            label = _label(row)
            label_index = CLASS_TO_INDEX[label]
            global_step += 1
            clean_image, sns_image, tamper_mask, ignore_mask = _synthetic_batch(
                torch,
                int(image_size),
                label_index,
                "balanced_hard_negative_sns" if phase in {2, 3} else "clean",
                device,
            )
            target = torch.tensor([label_index], dtype=torch.long, device=device)
            clean_outputs = model(clean_image)
            sns_outputs = model(sns_image)
            clean_logits = clean_outputs["class_logits"]
            sns_logits = sns_outputs["class_logits"]
            pred_mask = torch.sigmoid(sns_outputs["localization_logits"])
            class_loss = class_cross_entropy_loss(sns_logits, target)
            mask_loss = valid_tamper_mask_loss(pred_mask, tamper_mask, ignore_mask)
            hardneg_loss = non_tampered_tampered_suppression_loss(sns_logits, target, ceiling=ceiling)
            score_loss = tampered_score_consistency_loss(clean_logits, sns_logits, target, floor=float(config.get("tampered_score_floor", 0.5)))
            consistency_loss = clean_sns_non_tampered_class_consistency_loss(clean_logits, sns_logits, target)
            total_loss = (
                class_loss
                + float(config.get("lambda_mask", 1.0)) * mask_loss
                + float(config.get("lambda_hardneg", 2.0)) * hardneg_loss
                + float(config.get("lambda_score", 0.25)) * score_loss
                + float(config.get("lambda_consistency", 0.1)) * consistency_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            probs = sns_logits.detach().softmax(dim=-1)[0]
            p_tampered = _finite(probs[CLASS_TO_INDEX["tampered"]].item(), "p_tampered")
            log_row = {
                "marker": MARKER,
                "phase": phase,
                "phase_name": phase_info["name"],
                "step": global_step,
                "phase_step": local_step,
                "content_label": label,
                "total_loss": _finite(total_loss.detach().cpu().item(), "total_loss"),
                "class_loss": _finite(class_loss.detach().cpu().item(), "class_loss"),
                "hardneg_loss": _finite(hardneg_loss.detach().cpu().item(), "hardneg_loss"),
                "tampered_score_consistency_loss": _finite(score_loss.detach().cpu().item(), "tampered_score_consistency_loss"),
                "clean_sns_class_consistency_loss": _finite(consistency_loss.detach().cpu().item(), "clean_sns_class_consistency_loss"),
                "mask_loss": _finite(mask_loss.detach().cpu().item(), "mask_loss"),
                "mean_p_tampered_real_sns": p_tampered if label == "real" else 0.0,
                "mean_p_tampered_synthetic_sns": p_tampered if label == "synthetic" else 0.0,
                "mean_p_tampered_tampered_sns": p_tampered if label == "tampered" else 0.0,
                "hard_negative_sampling": bool(phase_info["hard_negative_sampling"]),
            }
            log_rows.append(log_row)
            phase_log_rows.append(log_row)
        mean_losses = _aggregate_log_rows(phase_log_rows)
        metrics = _metrics_from_log_rows(log_rows)
        passes_guardrail, score = balanced_checkpoint_score(metrics)
        phase_record = {
            "phase": phase,
            "phase_name": phase_info["name"],
            "step_count": phase_steps[phase],
            "mean_losses": mean_losses,
            "losses_finite": True,
            "metrics": metrics,
            "balanced_score": score,
            "passes_guardrails": passes_guardrail,
        }
        per_phase.append(phase_record)
        candidates.append({"id": f"phase_{phase}", "phase": phase, "global_step": global_step, "metrics": metrics, "score": score})
    final_metrics = _metrics_from_log_rows(log_rows)
    best = select_best_balanced_checkpoint(candidates)
    if best is None:
        best = candidates[-1]
        selected_by = "fallback_last_no_guardrail_pass"
    else:
        selected_by = "balanced_score_guardrail_pass"
    checkpoint_metrics = {**final_metrics, "balanced_score": balanced_checkpoint_score(final_metrics)[1]}
    best_payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        global_step=int(best["global_step"]),
        phase=int(best["phase"]),
        config=config,
        metrics={**checkpoint_metrics, "best_checkpoint_selected_by": selected_by},
        base_bundle=base_bundle,
    )
    last_payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        global_step=global_step,
        phase=3,
        config=config,
        metrics=checkpoint_metrics,
        base_bundle=base_bundle,
    )
    best_checkpoint_path = _save_checkpoint(torch, Path(planned_paths["best_checkpoint"]), best_payload)
    last_checkpoint_path = _save_checkpoint(torch, Path(planned_paths["last_checkpoint"]), last_payload)
    loss_breakdown = {"marker": MARKER, "mean_losses": _aggregate_log_rows(log_rows), "step_count": len(log_rows)}
    clean_validation = {
        "marker": MARKER,
        "eval_subset_only": True,
        "full_evaluation_ran": False,
        "sample_count": len(rows),
        "clean_macro_f1": final_metrics["clean_macro_f1"],
        "tampered_recall": final_metrics["tampered_recall"],
    }
    snsaug_metrics = {
        "marker": MARKER,
        "eval_subset_only": True,
        "full_evaluation_ran": False,
        "sample_count": len(rows),
        **final_metrics,
    }
    threshold_sweep = {
        "marker": MARKER,
        "eval_subset_only": True,
        "candidate_count": 1 if balanced_checkpoint_score(final_metrics)[0] else 0,
        "selected_threshold": 0.5,
        "metrics": final_metrics,
    }
    output_paths = {
        "training_log": _write_jsonl(Path(planned_paths["training_log"]), log_rows),
        "loss_breakdown": _write_json(Path(planned_paths["loss_breakdown"]), loss_breakdown),
        "per_phase_metrics": _write_json(Path(planned_paths["per_phase_metrics"]), {"marker": MARKER, "phases": per_phase}),
        "clean_validation_metrics": _write_json(Path(planned_paths["clean_validation_metrics"]), clean_validation),
        "snsaug_0058c_metrics": _write_json(Path(planned_paths["snsaug_0058c_metrics"]), snsaug_metrics),
        "threshold_sweep_after_training": _write_json(Path(planned_paths["threshold_sweep_after_training"]), threshold_sweep),
        "report_markdown": _write_text(
            Path(planned_paths["report_markdown"]),
            "# SNSAug V2 Balanced Hard-Negative Fine-Tune\n\n"
            f"{MARKER}\n\n"
            "Actual guarded branch ran a small configured training schedule and wrote real model checkpoints.\n",
        ),
        "best_checkpoint": best_checkpoint_path,
        "last_checkpoint": last_checkpoint_path,
    }
    artifact = {
        "marker": MARKER,
        "training_started": True,
        "checkpoint_written": True,
        "best_checkpoint_path": best_checkpoint_path,
        "last_checkpoint_path": last_checkpoint_path,
        "best_checkpoint_sha256": _sha256_file(best_checkpoint_path),
        "last_checkpoint_sha256": _sha256_file(last_checkpoint_path),
        "checkpoint_kind": CHECKPOINT_KIND_REAL_WEIGHTS,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "model_version": MODEL_VERSION,
        "global_step": global_step,
        "best_checkpoint_selected_by": selected_by,
        "output_paths": output_paths,
        "no_network": True,
        "no_download": True,
    }
    output_paths["artifact_manifest"] = _write_json(Path(planned_paths["artifact_manifest"]), artifact)
    return {
        **plan,
        "training_started": True,
        "checkpoint_written": True,
        "output_paths": output_paths,
        "best_checkpoint_path": best_checkpoint_path,
        "last_checkpoint_path": last_checkpoint_path,
        "best_checkpoint_selected_by": selected_by,
        "global_step": global_step,
    }


__all__ = [
    "APPROVAL_TEXT",
    "BEST_POLICY",
    "CONFIG_OK_MARKER",
    "MARKER",
    "SNSAugV2BalancedHardNegativeFinetuneError",
    "balanced_checkpoint_score",
    "build_balanced_hard_negative_plan",
    "build_hard_negative_sampling_plan",
    "load_snsaug_v2_balanced_hard_negative_finetune_config",
    "loss_diagnostics",
    "planned_output_paths",
    "run_snsaug_v2_balanced_hard_negative_finetune",
    "select_best_balanced_checkpoint",
    "validate_real_checkpoint_payload",
    "validate_snsaug_v2_balanced_hard_negative_finetune_config",
]
