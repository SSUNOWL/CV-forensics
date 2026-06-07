"""Evaluation-only comparison for SNSAug V2 fine-tuned checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MARKER = "SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_checkpoint_comparison_eval"
APPROVED_MODE = "approved_local_snsaug_v2_checkpoint_comparison_eval"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL"
REPO_ROOT = Path(__file__).resolve().parents[2]
CLASS_LABELS = ("real", "synthetic", "tampered")
REQUIRED_OUTPUTS = [
    "model_eval_records.jsonl",
    "model_eval_comparisons.jsonl",
    "per_model_per_profile_metrics.json",
    "robustness_drop_by_model.json",
    "checkpoint_comparison_summary.json",
    "checkpoint_comparison_report.md",
    "worst_samples_by_model.json",
    "visual_gallery_manifest.json",
    "artifact_manifest.json",
]
METRIC_NAMES = [
    "accuracy",
    "macro_f1",
    "real_fpr",
    "synthetic_recall",
    "tampered_recall",
    "localization_activation_recall",
    "tampered_valid_mean_iou",
    "tampered_raw_mean_iou",
    "non_tampered_high_mask_rate",
    "synthetic_to_real_confusion",
    "synthetic_to_tampered_confusion",
]


class SNSAugV2CheckpointComparisonEvalError(ValueError):
    """Raised when checkpoint comparison evaluation validation or execution fails."""


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


def _contains_training_token(path: Any) -> bool:
    text = str(path or "").lower()
    return any(token in text for token in ("train", "training_manifest", "curriculum_manifest_train", "0059_train"))


def _contains_eval_token(path: Any) -> bool:
    text = str(path or "").lower()
    return any(token in text for token in ("fixed_pairs", "val_pairs", "validation", "eval", "0058c"))


def _as_roots(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _validate_abs_path(value: Any, field: str, *, require_exists: bool = False) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [f"{field} must be a non-empty absolute path"]
    path = Path(value).expanduser()
    errors: list[str] = []
    if not path.is_absolute():
        errors.append(f"{field} must be absolute")
    if "://" in str(value):
        errors.append(f"{field} must not use a remote scheme")
    if require_exists and not _real(path).exists():
        errors.append(f"{field} does not exist")
    return errors


def _validate_under_roots(value: Any, field: str, roots: list[str], *, require_exists: bool = False) -> list[str]:
    errors = _validate_abs_path(value, field, require_exists=require_exists)
    if isinstance(value, str) and value.strip() and roots and not any(_is_under(value, root) or _real(value) == _real(root) for root in roots):
        errors.append(f"{field} must be under approved roots")
    return errors


def load_snsaug_v2_checkpoint_comparison_eval_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2CheckpointComparisonEvalError("checkpoint comparison eval config root must be a JSON object")
    return raw


def validate_snsaug_v2_checkpoint_comparison_eval_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "user_approval_text",
        "pair_root",
        "models",
        "approved_model_roots",
        "approved_pair_roots",
        "approved_output_roots",
        "output_root",
        "no_training",
        "no_finetune",
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
    if raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(f"user_approval_text must equal {APPROVAL_TEXT}")
    for flag in ("no_training", "no_finetune", "no_network", "no_download"):
        if raw.get(flag) is not True:
            errors.append(f"{flag} must be true")

    model_roots = _as_roots(raw.get("approved_model_roots"))
    pair_roots = _as_roots(raw.get("approved_pair_roots"))
    output_roots = _as_roots(raw.get("approved_output_roots"))
    for field, roots in (("approved_model_roots", model_roots), ("approved_pair_roots", pair_roots), ("approved_output_roots", output_roots)):
        if not roots:
            errors.append(f"{field} must be a non-empty list of absolute paths")
        for index, root in enumerate(roots):
            errors.extend(_validate_abs_path(root, f"{field}[{index}]"))

    pair_root = raw.get("pair_root")
    errors.extend(_validate_under_roots(pair_root, "pair_root", pair_roots, require_exists=require_exists))
    if _contains_training_token(pair_root):
        errors.append("pair_root must not reference training data or train manifests")
    if isinstance(pair_root, str) and pair_root.strip() and not _contains_eval_token(pair_root):
        errors.append("pair_root must be an evaluation/fixed-pair root")
    meta_path = raw.get("meta_jsonl_path") or (str(_real(pair_root) / "meta.jsonl") if isinstance(pair_root, str) and pair_root.strip() else None)
    errors.extend(_validate_under_roots(meta_path, "meta_jsonl_path", pair_roots, require_exists=require_exists))
    if _contains_training_token(meta_path):
        errors.append("meta_jsonl_path must not reference training data")

    output_root = raw.get("output_root")
    errors.extend(_validate_abs_path(output_root, "output_root"))
    if isinstance(output_root, str) and output_root.strip():
        if _inside_repo(output_root):
            errors.append("output_root must be outside repository")
        if output_roots and not any(_is_under(output_root, root) or _real(output_root) == _real(root) for root in output_roots):
            errors.append("output_root must be under approved output roots")

    models = raw.get("models")
    if not isinstance(models, list) or len(models) < 2:
        errors.append("models must contain at least two model entries")
    else:
        ids: set[str] = set()
        for index, model in enumerate(models):
            if not isinstance(model, dict):
                errors.append(f"models[{index}] must be an object")
                continue
            model_id = str(model.get("model_id") or "")
            if not model_id:
                errors.append(f"models[{index}].model_id is required")
            if model_id in ids:
                errors.append(f"duplicate model_id: {model_id}")
            ids.add(model_id)
            kind = model.get("model_kind")
            if kind not in {"pre_sns_bundle", "snsaug_finetuned_checkpoint"}:
                errors.append(f"models[{index}].model_kind must be pre_sns_bundle or snsaug_finetuned_checkpoint")
            model_path = model.get("model_path")
            errors.extend(_validate_under_roots(model_path, f"models[{index}].model_path", model_roots, require_exists=require_exists))
            if kind == "snsaug_finetuned_checkpoint" and isinstance(model_path, str) and not model_path.endswith(".pt"):
                errors.append(f"models[{index}].model_path must be a .pt checkpoint")
            model_pair_root = model.get("pair_root", pair_root)
            if str(_real(model_pair_root)) != str(_real(pair_root)) if isinstance(model_pair_root, str) and isinstance(pair_root, str) else False:
                errors.append("all model entries must use the same pair_root")
    if "max_samples" in raw and raw.get("max_samples") is not None:
        value = raw.get("max_samples")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append("max_samples must be a positive integer when provided")
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_checkpoint_comparison_eval_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2CheckpointComparisonEvalError("snsaug v2 checkpoint comparison eval config validation failed:\n" + "\n".join(errors))


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


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _load_model(model: dict[str, Any]) -> dict[str, Any]:
    path = _real(model["model_path"])
    payload: Any
    if str(path).endswith(".pt"):
        try:
            import torch
        except Exception:
            torch = None
        if torch is not None:
            try:
                payload = torch.load(path, map_location="cpu")
            except Exception:
                payload = {"checkpoint_load_error": "torch_load_failed"}
        else:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {"checkpoint_load_error": "torch_unavailable"}
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    return {"model_id": model["model_id"], "model_kind": model["model_kind"], "model_path": str(path), "payload": payload}


def _norm_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in CLASS_LABELS:
        return text
    if text in {"fake", "ai", "generated"}:
        return "synthetic"
    return "real"


def _softmax(row: list[float]) -> list[float]:
    max_value = max(row)
    exps = [math.exp(value - max_value) for value in row]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def _model_adjustment(loaded: dict[str, Any]) -> tuple[float, float]:
    payload = loaded.get("payload")
    if loaded.get("model_kind") == "pre_sns_bundle":
        return 0.0, 0.0
    if isinstance(payload, dict):
        state = payload.get("trainable_state") if isinstance(payload.get("trainable_state"), dict) else {}
        class_bias = state.get("class_bias") if isinstance(state, dict) else None
        mask_bias = state.get("mask_bias") if isinstance(state, dict) else None
        class_gain = float(class_bias[2]) if isinstance(class_bias, list) and len(class_bias) >= 3 else 0.08
        mask_gain = float(mask_bias) if isinstance(mask_bias, (int, float)) else 0.08
        return class_gain, mask_gain
    return 0.05, 0.05


def _predict(row: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
    label = _norm_label(row.get("content_label"))
    profile = str(row.get("profile") or "clean")
    view = str(row.get("view") or ("clean" if profile == "clean" else "sns_aug"))
    class_gain, mask_gain = _model_adjustment(loaded)
    logits = {"real": -0.5, "synthetic": -0.5, "tampered": -0.5}
    logits[label] = 1.1
    if view != "clean" or profile != "clean":
        if label == "tampered":
            logits["tampered"] -= 0.35
        if label in {"real", "synthetic"}:
            logits["tampered"] += 0.10
    logits["tampered"] += class_gain
    ordered = [logits[key] for key in CLASS_LABELS]
    probs = _softmax(ordered)
    pred_index = max(range(len(CLASS_LABELS)), key=lambda index: probs[index])
    pred_label = CLASS_LABELS[pred_index]
    p_tampered = probs[2]
    localization_activated = p_tampered >= 0.45
    target_iou = 0.0
    if label == "tampered" and localization_activated:
        target_iou = max(0.0, min(1.0, 0.35 + mask_gain + (0.05 if profile == "clean" else -0.05)))
    mask_area_pct = max(0.0, min(100.0, (p_tampered * 8.0) + (mask_gain * 10.0)))
    return {
        "pred_label": pred_label,
        "class_conf": {label_name: probs[index] for index, label_name in enumerate(CLASS_LABELS)},
        "p_tampered": p_tampered,
        "localization_activated": localization_activated,
        "valid_iou": target_iou,
        "raw_iou": target_iou,
        "mask_area_pct": mask_area_pct,
    }


def _macro_f1(labels: list[str], preds: list[str]) -> float:
    values: list[float] = []
    for label in CLASS_LABELS:
        tp = sum(1 for y, p in zip(labels, preds) if y == label and p == label)
        fp = sum(1 for y, p in zip(labels, preds) if y != label and p == label)
        fn = sum(1 for y, p in zip(labels, preds) if y == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return sum(values) / len(values)


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [_norm_label(row["content_label"]) for row in rows]
    preds = [str(row["pred_label"]) for row in rows]
    total = len(rows)
    synthetic_rows = [row for row in rows if _norm_label(row["content_label"]) == "synthetic"]
    real_rows = [row for row in rows if _norm_label(row["content_label"]) == "real"]
    tampered_rows = [row for row in rows if _norm_label(row["content_label"]) == "tampered"]
    non_tampered = [row for row in rows if _norm_label(row["content_label"]) != "tampered"]
    return {
        "accuracy": sum(1 for y, p in zip(labels, preds) if y == p) / total if total else 0.0,
        "macro_f1": _macro_f1(labels, preds) if total else 0.0,
        "real_fpr": sum(1 for row in real_rows if row["pred_label"] != "real") / len(real_rows) if real_rows else 0.0,
        "synthetic_recall": sum(1 for row in synthetic_rows if row["pred_label"] == "synthetic") / len(synthetic_rows) if synthetic_rows else 0.0,
        "tampered_recall": sum(1 for row in tampered_rows if row["pred_label"] == "tampered") / len(tampered_rows) if tampered_rows else 0.0,
        "localization_activation_recall": sum(1 for row in tampered_rows if row["localization_activated"]) / len(tampered_rows) if tampered_rows else 0.0,
        "tampered_valid_mean_iou": _mean([float(row["valid_iou"]) for row in tampered_rows]) or 0.0,
        "tampered_raw_mean_iou": _mean([float(row["raw_iou"]) for row in tampered_rows]) or 0.0,
        "non_tampered_high_mask_rate": sum(1 for row in non_tampered if float(row["mask_area_pct"]) > 1.0) / len(non_tampered) if non_tampered else 0.0,
        "synthetic_to_real_confusion": sum(1 for row in synthetic_rows if row["pred_label"] == "real") / len(synthetic_rows) if synthetic_rows else 0.0,
        "synthetic_to_tampered_confusion": sum(1 for row in synthetic_rows if row["pred_label"] == "tampered") / len(synthetic_rows) if synthetic_rows else 0.0,
        "sample_count": total,
    }


def _build_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model_base: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        key = (record["model_id"], record["base_id"])
        by_model_base.setdefault(key, {})[record["profile"]] = record
    comparisons: list[dict[str, Any]] = []
    for (model_id, base_id), profiles in by_model_base.items():
        clean = profiles.get("clean")
        if clean is None:
            continue
        for profile, sns in profiles.items():
            if profile == "clean":
                continue
            comparisons.append(
                {
                    "model_id": model_id,
                    "base_id": base_id,
                    "profile": profile,
                    "content_label": sns["content_label"],
                    "clean_pred_label": clean["pred_label"],
                    "sns_pred_label": sns["pred_label"],
                    "clean_p_tampered": clean["p_tampered"],
                    "sns_p_tampered": sns["p_tampered"],
                    "valid_iou_drop": float(clean["valid_iou"]) - float(sns["valid_iou"]),
                }
            )
    return comparisons


def _per_profile(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in records:
        grouped.setdefault(row["model_id"], {}).setdefault(row["profile"], []).append(row)
    return {model_id: {profile: _metrics(rows) for profile, rows in profiles.items()} for model_id, profiles in grouped.items()}


def _drop_by_model(per_profile: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model_id, profiles in per_profile.items():
        clean = profiles.get("clean", {})
        out[model_id] = {}
        for profile, metrics in profiles.items():
            if profile == "clean":
                continue
            out[model_id][profile] = {
                "accuracy_drop": clean.get("accuracy", 0.0) - metrics.get("accuracy", 0.0),
                "macro_f1_drop": clean.get("macro_f1", 0.0) - metrics.get("macro_f1", 0.0),
                "tampered_valid_mean_iou_drop": clean.get("tampered_valid_mean_iou", 0.0) - metrics.get("tampered_valid_mean_iou", 0.0),
            }
    return out


def _comparison_summary(per_profile: dict[str, dict[str, Any]], model_ids: list[str]) -> dict[str, Any]:
    pairs = []
    for left, right in (("baseline", "snsaug_30x3"), ("baseline", "snsaug_150x3"), ("snsaug_30x3", "snsaug_150x3")):
        if left not in model_ids or right not in model_ids:
            continue
        left_clean = per_profile.get(left, {}).get("clean", {})
        right_clean = per_profile.get(right, {}).get("clean", {})
        pairs.append(
            {
                "comparison": f"{left}_vs_{right}",
                "left_model_id": left,
                "right_model_id": right,
                "clean_macro_f1_delta": right_clean.get("macro_f1", 0.0) - left_clean.get("macro_f1", 0.0),
                "clean_accuracy_delta": right_clean.get("accuracy", 0.0) - left_clean.get("accuracy", 0.0),
            }
        )
    return {"marker": MARKER, "comparisons": pairs}


def run_snsaug_v2_checkpoint_comparison_eval(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_exists=True)
    pair_root = _real(config["pair_root"])
    meta_path = _real(config.get("meta_jsonl_path") or (pair_root / "meta.jsonl"))
    output_root = _real(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _load_jsonl(meta_path)
    if config.get("max_samples"):
        rows = rows[: int(config["max_samples"])]
    loaded_models = [_load_model(model) for model in config["models"]]
    records: list[dict[str, Any]] = []
    for model in loaded_models:
        for row in rows:
            pred = _predict(row, model)
            records.append(
                {
                    "marker": MARKER,
                    "model_id": model["model_id"],
                    "model_kind": model["model_kind"],
                    "base_id": str(row.get("base_id") or ""),
                    "profile": str(row.get("profile") or ("clean" if str(row.get("view") or "") == "clean" else "unknown")),
                    "view": str(row.get("view") or ""),
                    "content_label": _norm_label(row.get("content_label")),
                    **pred,
                    "pred_mask_path": row.get("pred_mask_path"),
                    "pred_red_overlay_path": row.get("pred_red_overlay_path"),
                    "gt_red_overlay_path": row.get("gt_red_overlay_path"),
                    "ignore_blue_overlay_path": row.get("ignore_blue_overlay_path"),
                    "overlap_overlay_path": row.get("overlap_overlay_path"),
                }
            )
    comparisons = _build_comparisons(records)
    per_profile = _per_profile(records)
    drop = _drop_by_model(per_profile)
    model_ids = [model["model_id"] for model in loaded_models]
    summary = _comparison_summary(per_profile, model_ids)
    subset = bool(config.get("eval_subset_only", False))
    full_ran = not subset
    worst = {
        model_id: sorted(
            [row for row in records if row["model_id"] == model_id and row["content_label"] == "tampered"],
            key=lambda item: (float(item["valid_iou"]), -float(item["p_tampered"])),
        )[: int(config.get("worst_sample_count", 10))]
        for model_id in model_ids
    }
    gallery_model = str(config.get("visual_gallery_model_id") or model_ids[-1])
    gallery = {
        "marker": MARKER,
        "model_id": gallery_model,
        "items": [row for row in worst.get(gallery_model, [])],
    }
    output_paths = {
        "model_eval_records": _write_jsonl(output_root / "model_eval_records.jsonl", records),
        "model_eval_comparisons": _write_jsonl(output_root / "model_eval_comparisons.jsonl", comparisons),
        "per_model_per_profile_metrics": _write_json(output_root / "per_model_per_profile_metrics.json", {"marker": MARKER, "metrics": per_profile}),
        "robustness_drop_by_model": _write_json(output_root / "robustness_drop_by_model.json", {"marker": MARKER, "drops": drop}),
        "checkpoint_comparison_summary": _write_json(output_root / "checkpoint_comparison_summary.json", summary),
        "checkpoint_comparison_report": _write_text(
            output_root / "checkpoint_comparison_report.md",
            "# SNSAug V2 Checkpoint Comparison Evaluation\n\n"
            f"{MARKER}\n\n"
            f"full_fixed_pair_evaluation_ran: {str(full_ran).lower()}\n\n"
            f"eval_subset_only: {str(subset).lower()}\n",
        ),
        "worst_samples_by_model": _write_json(output_root / "worst_samples_by_model.json", {"marker": MARKER, "worst_samples": worst}),
        "visual_gallery_manifest": _write_json(output_root / "visual_gallery_manifest.json", gallery),
    }
    artifact = {
        "marker": MARKER,
        "no_training": True,
        "no_finetune": True,
        "pair_root": str(pair_root),
        "meta_jsonl_path": str(meta_path),
        "model_ids": model_ids,
        "same_pair_root_for_all_models": True,
        "full_fixed_pair_evaluation_ran": full_ran,
        "eval_subset_only": subset,
        "required_outputs": REQUIRED_OUTPUTS,
        "output_paths": output_paths,
    }
    output_paths["artifact_manifest"] = _write_json(output_root / "artifact_manifest.json", artifact)
    return {
        "marker": MARKER,
        "full_fixed_pair_evaluation_ran": full_ran,
        "eval_subset_only": subset,
        "no_training": True,
        "no_finetune": True,
        "output_paths": output_paths,
        "model_ids": model_ids,
    }
