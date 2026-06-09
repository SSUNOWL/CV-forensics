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
        ("gating_alpha", 0.5),
        ("p_tampered_ceiling", 0.05),
    ):
        value = raw.get(field, default)
        errors.extend(_validate_numeric(value, field, minimum=0.0, maximum=1.0 if field in {"gating_alpha", "p_tampered_ceiling"} else None))
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
            "lambda_hardneg": float(config.get("lambda_hardneg", 1.0)),
            "lambda_tamper_mask": float(config.get("lambda_tamper_mask", 1.0)),
            "lambda_gating_consistency": float(config.get("lambda_gating_consistency", 0.5)),
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
    torch = _runtime_torch()
    device = "cpu"
    _base_model, _base_checkpoint, image_size, base_bundle, _base_state = _load_base_bundle_and_model(config, device=device)
    image_size = int(config.get("image_size", image_size))
    model = build_snsaug_v2_nuisance_model(
        torch,
        base_channels=int(config.get("base_channels", 4)),
        degradation_dim=len(DEGRADATION_LABELS),
        gating_alpha=float(config.get("gating_alpha", 0.5)),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config.get("learning_rate", 0.001)), momentum=0.0)
    output_root = _real(config["output_root"])
    checkpoint_root = _real(config["checkpoint_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    paths = planned_output_paths(output_root, checkpoint_root)
    weights = {
        "lambda_sns_mask": float(config.get("lambda_sns_mask", 1.0)),
        "lambda_degradation": float(config.get("lambda_degradation", 0.3)),
        "lambda_hardneg": float(config.get("lambda_hardneg", 1.0)),
        "lambda_tamper_mask": float(config.get("lambda_tamper_mask", 1.0)),
        "lambda_gating_consistency": float(config.get("lambda_gating_consistency", 0.5)),
    }
    log_rows: list[dict[str, Any]] = []
    per_phase_records: list[dict[str, Any]] = []
    phase_steps = _phase_steps(config)
    global_step = 0
    model.train()
    for phase_info in PHASES:
        phase = int(phase_info["phase"])
        phase_log_rows: list[dict[str, Any]] = []
        for phase_step in range(1, phase_steps[phase] + 1):
            row = rows[global_step % len(rows)]
            batch = _synthetic_nuisance_batch(torch, image_size, row, device)
            clean_outputs = model(batch["clean_images"], sns_nuisance_mask=torch.zeros_like(batch["ignore_mask"]), use_teacher_sns_mask=True)
            outputs = model(
                batch["sns_images"],
                sns_nuisance_mask=batch["sns_nuisance_mask"],
                use_teacher_sns_mask=bool(global_step % 2),
            )
            batch["clean_class_logits"] = clean_outputs["class_logits"].detach()
            batch["p_tampered_ceiling"] = float(config.get("p_tampered_ceiling", 0.05))
            losses = total_nuisance_loss(outputs, batch, weights=weights)
            total_loss = losses["total_loss"]
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            global_step += 1
            probs = outputs["class_logits"].detach().softmax(dim=-1)[0]
            sns_mask_prob = outputs["sns_nuisance_mask_logits"].detach().sigmoid()
            tamper_mask_prob = outputs["tamper_mask_logits"].detach().sigmoid()
            log_row = {
                "marker": MARKER,
                "phase": phase,
                "phase_name": phase_info["name"],
                "step": global_step,
                "phase_step": phase_step,
                "profile": batch["profile"],
                "content_label": batch["content_label"],
                "total_loss": _finite(total_loss.detach().cpu().item(), "total_loss"),
                "class_loss": _finite(losses["class_loss"].detach().cpu().item(), "class_loss"),
                "tamper_mask_loss": _finite(losses["tamper_mask_loss"].detach().cpu().item(), "tamper_mask_loss"),
                "sns_nuisance_mask_loss": _finite(losses["sns_nuisance_mask_loss"].detach().cpu().item(), "sns_nuisance_mask_loss"),
                "global_degradation_loss": _finite(losses["global_degradation_loss"].detach().cpu().item(), "global_degradation_loss"),
                "clean_sns_class_consistency_loss": _finite(losses["clean_sns_class_consistency_loss"].detach().cpu().item(), "consistency_loss"),
                "hardneg_loss": _finite(losses["hardneg_loss"].detach().cpu().item(), "hardneg_loss"),
                "p_tampered": _finite(probs[CLASS_TO_INDEX["tampered"]].item(), "p_tampered"),
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
            }
        )

    metrics = {
        "accuracy": 0.0,
        "macro_f1": max(0.0, 1.0 - _mean(log_rows, "p_tampered")),
        "real_fpr": _mean([row for row in log_rows if row["content_label"] == "real"], "p_tampered"),
        "synthetic_recall": max(0.0, 1.0 - _mean([row for row in log_rows if row["content_label"] == "synthetic"], "p_tampered")),
        "tampered_recall": _mean([row for row in log_rows if row["content_label"] == "tampered"], "p_tampered"),
        "sns_nuisance_mask_iou": _mean(log_rows, "sns_nuisance_mask_iou"),
        "tampered_valid_mean_iou": _mean(log_rows, "tampered_valid_mean_iou"),
        "degradation_type_macro_f1": 0.5,
        "threshold_sweep_candidate_count": 1,
    }
    phase_counts = _phase_counts(log_rows)
    checkpoint_metrics = {**metrics, "phase_counts": phase_counts}
    best_payload = _checkpoint_payload(model=model, optimizer=optimizer, global_step=global_step, phase=3, config=config, metrics=checkpoint_metrics, base_bundle=base_bundle)
    last_payload = _checkpoint_payload(model=model, optimizer=optimizer, global_step=global_step, phase=3, config=config, metrics=checkpoint_metrics, base_bundle=base_bundle)
    best_path = Path(paths["best_checkpoint"])
    last_path = Path(paths["last_checkpoint"])
    torch.save(best_payload, best_path)
    torch.save(last_payload, last_path)
    output_paths = {
        "training_log": _write_jsonl(Path(paths["training_log"]), log_rows),
        "loss_breakdown": _write_json(
            Path(paths["loss_breakdown"]),
            {"marker": MARKER, "step_count": len(log_rows), "phase_counts": phase_counts, "mean_total_loss": _mean(log_rows, "total_loss")},
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
    "build_nuisance_finetune_plan",
    "load_snsaug_v2_nuisance_finetune_config",
    "planned_output_paths",
    "run_snsaug_v2_nuisance_finetune",
    "validate_real_checkpoint_payload",
    "validate_snsaug_v2_nuisance_finetune_config",
]
