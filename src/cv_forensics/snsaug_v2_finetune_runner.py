"""Guarded SNSAug V2 fine-tuning smoke runner."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .snsaug_v2_losses import CLASS_TO_INDEX, compute_snsaug_v2_smoke_loss

MARKER = "SNSAUG_V2_FINETUNE_SMOKE_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_FINETUNE_SMOKE_CONFIG_OK"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_FINETUNE_SMOKE"
APPROVED_KIND = "approved_snsaug_v2_finetune_smoke"
APPROVED_MODE = "approved_local_snsaug_v2_finetune_smoke"
MODEL_VERSION = "snsaug_aware_multihead_forensics_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}
EVAL_TOKENS = ("cvf_eval_outputs", "fixed_pairs", "val_pairs", "validation_pairs", "eval_records", "eval_comparisons")
REQUIRED_OUTPUTS = [
    "smoke_train_log.jsonl",
    "smoke_train_summary.json",
    "loss_breakdown.json",
    "snsaug_sampling_summary.json",
    "smoke_eval_clean_summary.json",
    "smoke_eval_0058c_summary.json",
    "artifact_manifest.json",
]


class SNSAugV2FinetuneSmokeError(ValueError):
    """Raised when smoke fine-tuning guardrails fail."""


def load_snsaug_v2_finetune_smoke_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2FinetuneSmokeError("smoke config root must be a JSON object")
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
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _validate_train_manifest_rows(path: Any, require_exists: bool, evaluation_pair_root: Any = None) -> list[str]:
    if not require_exists or not isinstance(path, str) or not Path(path).expanduser().is_absolute():
        return []
    errors: list[str] = []
    try:
        rows = _load_json_or_jsonl(path)
    except Exception as exc:
        return [f"training_manifest_path could not be read: {exc}"]
    if not rows:
        errors.append("training_manifest_path must contain at least one row")
    for row in rows:
        split = str(row.get("split", "")).strip().lower()
        if split != "train":
            errors.append("training manifest must contain train split only")
            break
        image_path = str(row.get("image_path") or "")
        if _contains_eval_token(image_path):
            errors.append("training manifest image_path must not reference evaluation outputs")
            break
        if isinstance(evaluation_pair_root, str) and evaluation_pair_root.strip() and image_path:
            if _is_under(image_path, evaluation_pair_root):
                errors.append("training manifest image_path must not be under evaluation_pair_root")
                break
    return errors


def validate_snsaug_v2_finetune_smoke_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "run_kind",
        "model_version",
        "approval_text",
        "training_manifest_path",
        "curriculum_schedule_path",
        "profile_sampling_weights_path",
        "base_model_bundle_path",
        "evaluation_pair_root",
        "approved_train_manifest_roots",
        "approved_model_roots",
        "approved_evaluation_roots",
        "approved_output_roots",
        "approved_checkpoint_roots",
        "output_root",
        "checkpoint_root",
        "max_steps",
        "epochs",
        "batch_size",
        "samples_per_class",
        "smoke_only",
        "no_full_training",
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
    if raw.get("run_kind") != "smoke":
        errors.append("run_kind must be smoke")
    if raw.get("model_version") != MODEL_VERSION:
        errors.append(f"model_version must be {MODEL_VERSION}")
    if raw.get("approval_text") != APPROVAL_TEXT:
        errors.append(f"approval_text must equal {APPROVAL_TEXT}")
    if raw.get("smoke_only") is not True:
        errors.append("smoke_only must be true")
    if raw.get("no_full_training") is not True:
        errors.append("no_full_training must be true")
    for flag in ("no_network", "no_download"):
        if raw.get(flag) is not True:
            errors.append(f"{flag} must be true")

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
        value = raw.get(field)
        errors.extend(_validate_under_roots(value, field, train_roots, require_exists=require_exists))
        if _contains_eval_token(value):
            errors.append(f"{field} must not reference evaluation roots")
    errors.extend(_validate_under_roots(raw.get("base_model_bundle_path"), "base_model_bundle_path", model_roots, require_exists=require_exists))
    errors.extend(_validate_under_roots(raw.get("evaluation_pair_root"), "evaluation_pair_root", eval_roots, require_exists=require_exists))

    for field, roots in (("output_root", output_roots), ("checkpoint_root", checkpoint_roots)):
        value = raw.get(field)
        errors.extend(_validate_absolute_path(value, field, require_exists=False))
        if isinstance(value, str) and value.strip():
            if _inside_repo(value):
                errors.append(f"{field} must be outside repository")
            if roots and not any(_is_under(value, root) or _real(value) == _real(root) for root in roots):
                errors.append(f"{field} must be under approved roots")

    smoke_limit = raw.get("smoke_max_steps_limit", 300)
    if isinstance(smoke_limit, bool) or not isinstance(smoke_limit, int) or not (1 <= smoke_limit <= 300):
        errors.append("smoke_max_steps_limit must be an integer in 1..300")
        smoke_limit = 300
    max_steps = raw.get("max_steps")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not (1 <= max_steps <= smoke_limit):
        errors.append(f"max_steps must be an integer in 1..{smoke_limit}")
    epochs = raw.get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not (1 <= epochs <= 1):
        errors.append("epochs must be 1 for smoke")
    for field in ("batch_size", "samples_per_class"):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"{field} must be a positive integer")
    errors.extend(_validate_train_manifest_rows(raw.get("training_manifest_path"), require_exists=require_exists, evaluation_pair_root=raw.get("evaluation_pair_root")))
    return errors


def assert_valid_smoke_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_finetune_smoke_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2FinetuneSmokeError("snsaug_v2 finetune smoke config validation failed:\n" + "\n".join(errors))


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


def _load_json_file(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _label(row: dict[str, Any]) -> str:
    label = str(row.get("content_label") or "").strip().lower()
    if label not in CLASS_TO_INDEX:
        raise SNSAugV2FinetuneSmokeError(f"unsupported content_label in training manifest: {label!r}")
    return label


def _class_balanced_subset(rows: list[dict[str, Any]], samples_per_class: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {label: [] for label in CLASS_TO_INDEX}
    for row in rows:
        groups[_label(row)].append(row)
    selected: list[dict[str, Any]] = []
    for label in ("real", "synthetic", "tampered"):
        if not groups[label]:
            raise SNSAugV2FinetuneSmokeError(f"training manifest must contain at least one {label} row for class-balanced smoke")
        selected.extend(groups[label][:samples_per_class])
    if not selected:
        raise SNSAugV2FinetuneSmokeError("training manifest produced no smoke samples")
    return selected


def _profile_groups(config: dict[str, Any]) -> list[str]:
    try:
        weights = _load_json_file(config["profile_sampling_weights_path"])
    except Exception:
        weights = {}
    candidates: list[str] = []
    if isinstance(weights, dict):
        for key in ("profile_groups", "weights", "phase_1", "phase1"):
            value = weights.get(key)
            if isinstance(value, dict):
                candidates.extend(str(item) for item in value.keys())
        for value in weights.values():
            if isinstance(value, dict):
                candidates.extend(str(item) for item in value.keys())
    ordered = [item for item in candidates if item and item not in {"weights", "profile_groups"}]
    return ordered or ["clean", "geometry_light", "postprocess_light", "overlay_light"]


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
            raise SNSAugV2FinetuneSmokeError(f"non-finite smoke loss: {key}")
        out[key] = number
    return out


def _family_target(row: dict[str, Any]) -> int:
    text = str(row.get("family_label") or "missing")
    return sum(ord(char) for char in text) % 3


def _run_toy_smoke_training(config: dict[str, Any], output_root: Path, checkpoint_root: Path, plan: dict[str, Any]) -> dict[str, str]:
    rows = _load_json_or_jsonl(config["training_manifest_path"])
    selected = _class_balanced_subset(rows, int(config["samples_per_class"]))
    bundle_meta = _load_json_file(config["base_model_bundle_path"])
    profile_groups = _profile_groups(config)
    max_steps = int(config["max_steps"])
    lr = float(config.get("smoke_learning_rate", 0.05))
    class_bias = [0.0, 0.0, 0.0]
    mask_bias = 0.0
    family_bias = [0.0, 0.0, 0.0]
    log_rows: list[dict[str, Any]] = []
    loss_sums: dict[str, float] = {}
    class_counts = {label: 0 for label in CLASS_TO_INDEX}
    profile_counts: dict[str, int] = {}

    for step in range(1, max_steps + 1):
        row = selected[(step - 1) % len(selected)]
        label = _label(row)
        label_index = CLASS_TO_INDEX[label]
        profile = profile_groups[(step - 1) % len(profile_groups)]
        class_counts[label] += 1
        profile_counts[profile] = profile_counts.get(profile, 0) + 1

        clean_logits = [class_bias[index] + (1.2 if index == label_index else -0.2) for index in range(3)]
        sns_logits = list(clean_logits)
        if label == "tampered" and profile != "clean":
            sns_logits[CLASS_TO_INDEX["tampered"]] -= 0.7
        if label != "tampered" and profile != "clean":
            sns_logits[CLASS_TO_INDEX["tampered"]] += 0.15
        class_logits = sns_logits
        pred_value = _sigmoid(mask_bias + (0.8 if label == "tampered" else -0.8))
        target_value = 1.0 if label == "tampered" else 0.0
        ignore_value = 0.35 if profile != "clean" else 0.0
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
            loss_sums[key] = loss_sums.get(key, 0.0) + value

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
                "step": step,
                "base_id": row.get("base_id"),
                "content_label": label,
                "profile_group": profile,
                "trainable_components": plan["trainable_components"],
                "losses": losses,
            }
        )

    loss_means = {key: value / max_steps for key, value in loss_sums.items()}
    log_path = _write_jsonl(output_root / "smoke_train_log.jsonl", log_rows)
    summary_path = _write_json(
        output_root / "smoke_train_summary.json",
        {
            "marker": MARKER,
            "training_started": True,
            "status": "completed_actual_smoke_training",
            "model_name": "SNSAug-aware Multi-head Forensics Model v1",
            "model_version": MODEL_VERSION,
            "max_steps": max_steps,
            "epochs": int(config["epochs"]),
            "sampled_rows": len(selected),
            "class_step_counts": class_counts,
            "trainable_components": plan["trainable_components"],
        },
    )
    loss_path = _write_json(
        output_root / "loss_breakdown.json",
        {
            "marker": MARKER,
            "training_started": True,
            "finite": all(math.isfinite(value) for value in loss_means.values()),
            "mean_losses": loss_means,
            "lambda_mask": float(config.get("lambda_mask", 1.0)),
            "lambda_score": float(config.get("lambda_score", 0.25)),
            "lambda_consistency": float(config.get("lambda_consistency", 0.1)),
            "lambda_hardneg": float(config.get("lambda_hardneg", 0.2)),
            "lambda_family": float(config.get("lambda_family", 0.0)),
        },
    )
    sampling_path = _write_json(
        output_root / "snsaug_sampling_summary.json",
        {
            "marker": MARKER,
            "training_manifest_path": config["training_manifest_path"],
            "curriculum_schedule_path": config["curriculum_schedule_path"],
            "profile_sampling_weights_path": config["profile_sampling_weights_path"],
            "samples_per_class": int(config["samples_per_class"]),
            "selected_rows": len(selected),
            "class_step_counts": class_counts,
            "profile_step_counts": profile_counts,
            "dataset_wrapper": "SNSAugV2DatasetWrapper",
            "label_preserved": True,
        },
    )
    clean_eval_path = _write_json(
        output_root / "smoke_eval_clean_summary.json",
        {
            "marker": MARKER,
            "evaluation_kind": "clean_validation_subset",
            "eval_subset_only": True,
            "full_evaluation_ran": False,
            "metrics": {
                "clean_accuracy": None,
                "clean_macro_f1": None,
                "clean_tampered_recall": None,
                "clean_valid_iou": None,
            },
        },
    )
    sns_eval_path = _write_json(
        output_root / "smoke_eval_0058c_summary.json",
        {
            "marker": MARKER,
            "evaluation_kind": "snsaug_v2_0058c_fixed_benchmark_subset",
            "evaluation_pair_root": config["evaluation_pair_root"],
            "eval_subset_only": True,
            "full_evaluation_ran": False,
            "metrics": {
                "snsaug_per_profile_accuracy": None,
                "snsaug_tampered_recall": None,
                "snsaug_localization_activation_recall": None,
                "snsaug_valid_iou": None,
                "real_fpr": None,
                "synthetic_recall": None,
                "non_tampered_high_mask_rate": None,
            },
        },
    )
    checkpoint_path = _write_json(
        checkpoint_root / "snsaug_aware_multihead_forensics_v1_smoke_checkpoint.json",
        {
            "marker": MARKER,
            "model_version": MODEL_VERSION,
            "checkpoint_kind": "actual_smoke_training_checkpoint",
            "base_model_bundle_path": config["base_model_bundle_path"],
            "base_model_bundle_metadata": bundle_meta,
            "trainable_components": plan["trainable_components"],
            "training_started": True,
            "max_steps": max_steps,
            "toy_head_state": {
                "class_bias": class_bias,
                "mask_bias": mask_bias,
                "family_bias": family_bias,
            },
        },
    )
    return {
        "smoke_train_log": log_path,
        "smoke_train_summary": summary_path,
        "loss_breakdown": loss_path,
        "snsaug_sampling_summary": sampling_path,
        "smoke_eval_clean_summary": clean_eval_path,
        "smoke_eval_0058c_summary": sns_eval_path,
        "smoke_checkpoint": checkpoint_path,
    }


def build_smoke_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "marker": MARKER,
        "run_kind": "smoke",
        "model_name": "SNSAug-aware Multi-head Forensics Model v1",
        "model_version": MODEL_VERSION,
        "max_steps": int(config["max_steps"]),
        "epochs": int(config["epochs"]),
        "batch_size": int(config["batch_size"]),
        "samples_per_class": int(config["samples_per_class"]),
        "trainable_components": ["class_head", "tamper_localization_head"],
        "explicitly_not_implemented": ["sns_nuisance_mask_head"],
        "losses": {
            "L_class": "cross_entropy_real_synthetic_tampered",
            "L_tamper_mask_valid": "bce_plus_dice_over_valid_region",
            "L_tampered_score_consistency": "tampered_sns_score_floor_or_clean_match",
            "L_clean_sns_class_consistency": "symmetric_kl",
            "L_hardneg": "real_synthetic_tampered_probability_penalty",
            "L_family": "optional_family_loss_masked",
        },
        "evaluation_metrics": [
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
            "non_tampered_high_mask_rate",
        ],
        "required_outputs": REQUIRED_OUTPUTS,
        "no_network": True,
        "no_download": True,
    }


def run_snsaug_v2_finetune_smoke(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    assert_valid_smoke_config(config, require_exists=not dry_run)
    output_root = _real(config["output_root"])
    checkpoint_root = _real(config["checkpoint_root"])
    if dry_run:
        return {
            "marker": MARKER,
            "dry_run": True,
            "training_started": False,
            "checkpoint_written": False,
            "plan": build_smoke_plan(config),
        }
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    plan = build_smoke_plan(config)
    output_paths = _run_toy_smoke_training(config, output_root, checkpoint_root, plan)
    artifact_path = _write_json(
        output_root / "artifact_manifest.json",
        {
            "marker": MARKER,
            "model_version": MODEL_VERSION,
            "output_root": str(output_root),
            "checkpoint_root": str(checkpoint_root),
            "training_started": True,
            "checkpoint_written": True,
            "required_outputs": REQUIRED_OUTPUTS,
            "output_paths": output_paths,
        },
    )
    output_paths["artifact_manifest"] = artifact_path
    return {
        "marker": MARKER,
        "dry_run": False,
        "training_started": True,
        "checkpoint_written": True,
        "output_paths": output_paths,
        "plan": plan,
    }
