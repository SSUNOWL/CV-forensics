"""SNS-aware fine-tuning helpers and guarded runner."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .pre_sns_v3_dual_scale_report import (
    _err,
    _inside_repo,
    _is_under,
    _real,
    _validate_absolute_path,
    _validate_under_roots,
)
from .pre_sns_v3_sns_robustness_eval import json_safe
from .snsaug_v2_dataset_wrapper import masked_localization_loss
from .snsaug_v2_training_manifest import DEFAULT_CURRICULUM

MARKER = "SNSAUG_V2_AWARE_FINETUNING_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_AWARE_FINETUNING_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_finetune"
APPROVED_MODE = "approved_local_snsaug_v2_finetune"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_FINETUNE"


class SNSAugV2FinetuneError(ValueError):
    """Raised when snsaug_v2 finetune inputs or guardrails fail."""


def load_snsaug_v2_finetune_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2FinetuneError("snsaug_v2 finetune config root must be a JSON object")
    return raw


def _as_roots(value: Any) -> list[str]:
    return value if isinstance(value, list) and all(isinstance(item, str) for item in value) else []


def _load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and all(line.lstrip().startswith("{") for line in lines[:2]):
        rows = []
        for line in lines:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        return rows
    raw = json.loads(text)
    items = raw.get("samples", raw) if isinstance(raw, dict) else raw
    return [item for item in items if isinstance(item, dict)]


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def validate_snsaug_v2_finetune_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "training_manifest_path",
        "best_bundle_path",
        "approved_input_roots",
        "approved_checkpoint_roots",
        "approved_output_roots",
        "output_root",
        "approval_text",
        "seed",
        "epochs",
        "device",
        "no_network",
        "no_download",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    if raw.get("config_kind") != APPROVED_KIND:
        errors.append(_err(f"config_kind must be {APPROVED_KIND}"))
    if raw.get("execution_mode") != APPROVED_MODE:
        errors.append(_err(f"execution_mode must be {APPROVED_MODE}"))
    for flag in ("no_network", "no_download"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if raw.get("approval_text") != APPROVAL_TEXT:
        errors.append(_err(f"approval_text must equal {APPROVAL_TEXT}"))
    roots = _as_roots(raw.get("approved_input_roots"))
    checkpoint_roots = _as_roots(raw.get("approved_checkpoint_roots"))
    output_roots = _as_roots(raw.get("approved_output_roots"))
    if not roots:
        errors.append(_err("approved_input_roots must be a non-empty list of absolute paths"))
    if not checkpoint_roots:
        errors.append(_err("approved_checkpoint_roots must be a non-empty list of absolute paths"))
    if not output_roots:
        errors.append(_err("approved_output_roots must be a non-empty list of absolute paths"))
    for index, root in enumerate(roots):
        errors.extend(_validate_absolute_path(root, f"approved_input_roots[{index}]"))
    for index, root in enumerate(checkpoint_roots):
        errors.extend(_validate_absolute_path(root, f"approved_checkpoint_roots[{index}]"))
    for index, root in enumerate(output_roots):
        errors.extend(_validate_absolute_path(root, f"approved_output_roots[{index}]"))
    errors.extend(_validate_under_roots(raw.get("training_manifest_path"), "training_manifest_path", roots, require_file=require_exists))
    errors.extend(_validate_under_roots(raw.get("best_bundle_path"), "best_bundle_path", roots + checkpoint_roots, require_file=require_exists))
    output_root = raw.get("output_root")
    errors.extend(_validate_absolute_path(output_root, "output_root", require_dir_parent=require_exists))
    if isinstance(output_root, str) and output_root.startswith("/"):
        if _inside_repo(_real(output_root)):
            errors.append(_err("output_root must be outside repository"))
        if any(_is_under(output_root, root) for root in roots + checkpoint_roots):
            errors.append(_err("output_root must not be under approved input/checkpoint roots"))
        if output_roots and not any(_is_under(output_root, root) or str(_real(output_root)) == str(_real(root)) for root in output_roots):
            errors.append(_err("output_root must be under an approved output root"))
    for field in ("seed", "epochs", "batch_size"):
        if field in raw:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(_err(f"{field} must be a positive integer"))
    if raw.get("device") not in {"cpu", "cuda"}:
        errors.append(_err("device must be cpu or cuda"))
    if require_exists and raw.get("training_manifest_path"):
        try:
            rows = _load_json_or_jsonl(raw["training_manifest_path"])
        except Exception as exc:
            errors.append(_err(f"training_manifest_path could not be read: {exc}"))
            rows = []
        for row in rows:
            split = str(row.get("split", ""))
            if split != "train":
                errors.append(_err("training manifest must contain train split only"))
                break
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_finetune_config(config, require_exists)
    if errors:
        raise SNSAugV2FinetuneError("snsaug_v2 finetune config validation failed:\n" + "\n".join(errors))


def curriculum_stage(epoch: int, schedule: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    schedule = list(schedule or DEFAULT_CURRICULUM)
    for stage in schedule:
        start = int(stage.get("start_epoch", 1))
        end = stage.get("end_epoch")
        if epoch >= start and (end is None or epoch <= int(end)):
            return dict(stage)
    return dict(schedule[-1])


def masked_family_loss(logits, targets, family_loss_mask, reduction: str = "mean"):
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        torch = None
        F = None
    if torch is not None and hasattr(logits, "shape"):
        mask = family_loss_mask.float()
        losses = F.cross_entropy(logits, targets.long(), reduction="none") * mask
        if reduction == "sum":
            return losses.sum()
        if reduction == "none":
            return losses
        denom = mask.sum().clamp(min=1.0)
        return losses.sum() / denom
    flat_logits = list(logits)
    flat_targets = list(targets)
    flat_mask = [float(value) for value in family_loss_mask]
    losses = []
    for probs, target, mask_value in zip(flat_logits, flat_targets, flat_mask):
        if mask_value <= 0.0:
            losses.append(0.0)
            continue
        prob = max(1e-6, min(1.0, float(probs[int(target)])))
        losses.append(-math.log(prob) * mask_value)
    if reduction == "sum":
        return sum(losses)
    if reduction == "none":
        return losses
    denom = sum(flat_mask) or 1.0
    return sum(losses) / denom


def consistency_loss(clean_logits, aug_logits, reduction: str = "mean"):
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        torch = None
        F = None
    if torch is not None and hasattr(clean_logits, "shape"):
        loss = F.mse_loss(clean_logits.float(), aug_logits.float(), reduction="none")
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        return loss.mean()
    losses = []
    for left, right in zip(clean_logits, aug_logits):
        for lval, rval in zip(left, right):
            losses.append((float(lval) - float(rval)) ** 2)
    if reduction == "sum":
        return sum(losses)
    if reduction == "none":
        return losses
    return sum(losses) / max(1, len(losses))


def build_validation_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "clean_validation": bool(config.get("run_clean_validation", True)),
        "basic_validation": bool(config.get("run_basic_validation", True)),
        "snsaug_validation": bool(config.get("run_snsaug_validation", True)),
        "reported_metrics": [
            "accuracy_3way",
            "macro_f1",
            "real_fpr",
            "synthetic_recall",
            "tampered_recall",
            "tampered_mask_iou",
            "localization_activation_recall",
            "latency_ms",
            "fps",
        ],
    }


def compute_total_loss(
    *,
    class_loss,
    mask_pred,
    tamper_mask,
    ignore_mask,
    lambda_mask: float,
    family_logits=None,
    family_targets=None,
    family_loss_mask=None,
    lambda_family: float = 0.0,
    clean_logits=None,
    aug_logits=None,
    lambda_consistency: float = 0.0,
):
    mask_loss = masked_localization_loss(mask_pred, tamper_mask, ignore_mask, reduction="mean")
    family_loss = 0.0
    if family_logits is not None and family_targets is not None and family_loss_mask is not None and float(lambda_family) != 0.0:
        family_loss = masked_family_loss(family_logits, family_targets, family_loss_mask, reduction="mean")
    consistency = 0.0
    if clean_logits is not None and aug_logits is not None and float(lambda_consistency) != 0.0:
        consistency = consistency_loss(clean_logits, aug_logits, reduction="mean")
    total = class_loss + float(lambda_mask) * mask_loss + float(lambda_family) * family_loss + float(lambda_consistency) * consistency
    return {
        "total_loss": total,
        "class_loss": class_loss,
        "mask_loss": mask_loss,
        "family_loss": family_loss,
        "consistency_loss": consistency,
    }


def run_snsaug_v2_finetune(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_exists=True)
    output_root = _real(config["output_root"])
    if _inside_repo(output_root):
        raise SNSAugV2FinetuneError("output_root must be outside repository")
    output_root.mkdir(parents=True, exist_ok=True)
    config_copy = _write_json(output_root / "snsaug_v2_finetune_config.json", config)
    summary = {
        "marker": MARKER,
        "approval_text": config["approval_text"],
        "epochs": int(config["epochs"]),
        "device": config["device"],
        "training_manifest_path": config["training_manifest_path"],
        "best_bundle_path": config["best_bundle_path"],
        "curriculum": list(config.get("curriculum", DEFAULT_CURRICULUM)),
        "validation_plan": build_validation_plan(config),
        "output_paths": {
            "config_copy": config_copy,
        },
        "no_network": True,
        "no_download": True,
    }
    artifact_path = _write_json(
        output_root / "artifact_manifest.json",
        {
            "marker": MARKER,
            "output_paths": summary["output_paths"],
            "training_allowed": True,
            "no_network": True,
            "no_download": True,
        },
    )
    summary["output_paths"]["artifact_manifest"] = artifact_path
    return summary
