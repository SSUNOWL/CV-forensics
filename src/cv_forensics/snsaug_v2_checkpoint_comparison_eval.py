"""Evaluation-only comparison for SNSAug V2 fine-tuned checkpoints."""

from __future__ import annotations

import json
import hashlib
import warnings
from pathlib import Path
from typing import Any

from .pre_sns_v3_sns_robustness_eval import load_best_bundle
from .snsaug_v2_fixed_pairs_eval import (
    evaluate_rows as fixed_pair_evaluate_rows,
    parse_fixed_pair_rows,
)

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    return {"checkpoint_sha256": _sha256(path), "checkpoint_size_bytes": int(path.stat().st_size)}


def _torch_runtime():
    try:
        import torch
    except Exception as exc:
        raise SNSAugV2CheckpointComparisonEvalError("torch is required for real checkpoint comparison inference") from exc
    return torch


def _torch_load(path: Path) -> Any:
    torch = _torch_runtime()
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise SNSAugV2CheckpointComparisonEvalError(f"failed to load checkpoint: {path}") from exc


def _extract_state(payload: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any] | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, dict):
            return value
    return None


def _state_delta(base_state: dict[str, Any], candidate_state: dict[str, Any]) -> dict[str, Any]:
    torch = _torch_runtime()
    compared = 0
    changed = 0
    missing: list[str] = []
    shape_mismatch: list[str] = []
    for key, candidate_value in candidate_state.items():
        base_value = base_state.get(key)
        if base_value is None:
            missing.append(str(key))
            continue
        if hasattr(candidate_value, "shape") and hasattr(base_value, "shape"):
            if tuple(candidate_value.shape) != tuple(base_value.shape):
                shape_mismatch.append(str(key))
                continue
            compared += 1
            if not torch.equal(candidate_value.detach().cpu(), base_value.detach().cpu()):
                changed += 1
    return {
        "tensor_count_compared": compared,
        "changed_tensor_count": changed,
        "missing_key_count": len(missing),
        "shape_mismatch_count": len(shape_mismatch),
        "missing_keys": missing[:20],
        "shape_mismatch_keys": shape_mismatch[:20],
        "has_parameter_delta": changed > 0,
    }


def _write_derived_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch = _torch_runtime()
    torch.save(payload, path)
    return str(path)


def _resolve_baseline_bundle(model: dict[str, Any]) -> dict[str, Any]:
    path = _real(model["model_path"])
    bundle = load_best_bundle(path)
    for field in ("long256_checkpoint_path", "tile_v2_checkpoint_path"):
        if field not in bundle:
            raise SNSAugV2CheckpointComparisonEvalError(f"pre-SNS bundle missing {field}")
    return bundle


def _bundle_for_finetuned_checkpoint(
    *,
    model: dict[str, Any],
    baseline_bundle: dict[str, Any],
    derived_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _real(model["model_path"])
    if not path.is_file():
        raise SNSAugV2CheckpointComparisonEvalError(f"fine-tuned checkpoint does not exist: {path}")
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise SNSAugV2CheckpointComparisonEvalError(f"fine-tuned checkpoint must contain a dict payload: {path}")
    if payload.get("trainable_state") and not any(
        isinstance(payload.get(key), dict)
        for key in ("long256_model_state_dict", "long256_state_dict", "model_state_dict", "state_dict", "tile_v2_model_state_dict")
    ) and not payload.get("long256_checkpoint_path"):
        raise SNSAugV2CheckpointComparisonEvalError(
            f"fine-tuned checkpoint has only trainable_state proxy values, not real model weights: {path}"
        )

    baseline_long_path = _real(baseline_bundle["long256_checkpoint_path"])
    baseline_tile_path = _real(baseline_bundle["tile_v2_checkpoint_path"])
    baseline_long = _torch_load(baseline_long_path)
    if not isinstance(baseline_long, dict) or not isinstance(baseline_long.get("model_state_dict"), dict):
        raise SNSAugV2CheckpointComparisonEvalError("baseline long256 checkpoint missing model_state_dict")
    baseline_tile = _torch_load(baseline_tile_path)

    no_weight_delta = bool(model.get("no_weight_delta") or payload.get("no_weight_delta"))
    long_state = _extract_state(payload, ("long256_model_state_dict", "long256_state_dict", "model_state_dict", "state_dict"))
    tile_state = _extract_state(payload, ("tile_v2_model_state_dict", "tile_v2_state_dict", "tile_model_state_dict"))
    delta: dict[str, Any] = {"long256": None, "tile_v2": None}

    if long_state is not None:
        delta["long256"] = _state_delta(baseline_long["model_state_dict"], long_state)
        if not delta["long256"]["has_parameter_delta"] and not no_weight_delta:
            raise SNSAugV2CheckpointComparisonEvalError(f"fine-tuned long256 checkpoint has no parameter delta: {path}")
        derived_long = dict(baseline_long)
        derived_long["model_state_dict"] = long_state
        derived_long["model_name"] = "pre_sns_v3"
        long_path = _write_derived_checkpoint(derived_root / model["model_id"] / "long256_finetuned.pt", derived_long)
    elif payload.get("long256_checkpoint_path"):
        long_path = str(_real(payload["long256_checkpoint_path"]))
        if not Path(long_path).is_file():
            raise SNSAugV2CheckpointComparisonEvalError(f"fine-tuned long256_checkpoint_path does not exist: {long_path}")
        if _sha256(Path(long_path)) == _sha256(baseline_long_path) and not no_weight_delta:
            raise SNSAugV2CheckpointComparisonEvalError(f"fine-tuned long256 checkpoint is byte-identical to baseline: {long_path}")
        delta["long256"] = {"has_parameter_delta": True, "changed_tensor_count": None, "source": "external_checkpoint"}
    else:
        raise SNSAugV2CheckpointComparisonEvalError(f"fine-tuned checkpoint missing real long256 model weights: {path}")

    if tile_state is not None:
        base_tile_state = _extract_state(baseline_tile, ("model_state_dict", "state_dict", "model"))
        if base_tile_state is not None:
            delta["tile_v2"] = _state_delta(base_tile_state, tile_state)
        derived_tile = dict(baseline_tile) if isinstance(baseline_tile, dict) else {}
        derived_tile["model_state_dict"] = tile_state
        tile_path = _write_derived_checkpoint(derived_root / model["model_id"] / "tile_v2_finetuned.pt", derived_tile)
    elif payload.get("tile_v2_checkpoint_path"):
        tile_path = str(_real(payload["tile_v2_checkpoint_path"]))
        if not Path(tile_path).is_file():
            raise SNSAugV2CheckpointComparisonEvalError(f"fine-tuned tile_v2_checkpoint_path does not exist: {tile_path}")
    else:
        tile_path = str(baseline_tile_path)

    bundle = dict(payload.get("base_model_bundle_metadata") if isinstance(payload.get("base_model_bundle_metadata"), dict) else baseline_bundle)
    bundle["long256_checkpoint_path"] = long_path
    bundle["tile_v2_checkpoint_path"] = tile_path
    return bundle, {**_checkpoint_metadata(path), "weight_delta": delta, "no_weight_delta": no_weight_delta}


def _norm_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in CLASS_LABELS:
        return text
    if text in {"fake", "ai", "generated"}:
        return "synthetic"
    return "real"


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
    preds = [str(row["pred_class"]) for row in rows]
    total = len(rows)
    synthetic_rows = [row for row in rows if _norm_label(row["content_label"]) == "synthetic"]
    real_rows = [row for row in rows if _norm_label(row["content_label"]) == "real"]
    tampered_rows = [row for row in rows if _norm_label(row["content_label"]) == "tampered"]
    non_tampered = [row for row in rows if _norm_label(row["content_label"]) != "tampered"]
    return {
        "accuracy": sum(1 for y, p in zip(labels, preds) if y == p) / total if total else 0.0,
        "macro_f1": _macro_f1(labels, preds) if total else 0.0,
        "real_fpr": sum(1 for row in real_rows if row["pred_class"] != "real") / len(real_rows) if real_rows else 0.0,
        "synthetic_recall": sum(1 for row in synthetic_rows if row["pred_class"] == "synthetic") / len(synthetic_rows) if synthetic_rows else 0.0,
        "tampered_recall": sum(1 for row in tampered_rows if row["pred_class"] == "tampered") / len(tampered_rows) if tampered_rows else 0.0,
        "localization_activation_recall": sum(1 for row in tampered_rows if row["localization_activated"]) / len(tampered_rows) if tampered_rows else 0.0,
        "tampered_valid_mean_iou": _mean([float(row["valid_iou"]) for row in tampered_rows if row.get("valid_iou") is not None]) or 0.0,
        "tampered_raw_mean_iou": _mean([float(row["raw_iou"]) for row in tampered_rows if row.get("raw_iou") is not None]) or 0.0,
        "non_tampered_high_mask_rate": sum(1 for row in non_tampered if float(row.get("pred_mask_area_pct") or 0.0) > 1.0) / len(non_tampered) if non_tampered else 0.0,
        "synthetic_to_real_confusion": sum(1 for row in synthetic_rows if row["pred_class"] == "real") / len(synthetic_rows) if synthetic_rows else 0.0,
        "synthetic_to_tampered_confusion": sum(1 for row in synthetic_rows if row["pred_class"] == "tampered") / len(synthetic_rows) if synthetic_rows else 0.0,
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
                    "clean_pred_class": clean["pred_class"],
                    "sns_pred_class": sns["pred_class"],
                    "clean_p_tampered": clean["p_tampered"],
                    "sns_p_tampered": sns["p_tampered"],
                    "valid_iou_drop": (
                        float(clean["valid_iou"]) - float(sns["valid_iou"])
                        if clean.get("valid_iou") is not None and sns.get("valid_iou") is not None
                        else None
                    ),
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
    desired_pairs = (
        ("pre_sns_baseline", "snsaug_guarded_short_30x3"),
        ("pre_sns_baseline", "snsaug_medium_150x3"),
        ("snsaug_guarded_short_30x3", "snsaug_medium_150x3"),
        ("baseline", "snsaug_30x3"),
        ("baseline", "snsaug_150x3"),
        ("snsaug_30x3", "snsaug_150x3"),
    )
    seen: set[tuple[str, str]] = set()
    for left, right in desired_pairs:
        if left not in model_ids or right not in model_ids:
            continue
        if (left, right) in seen:
            continue
        seen.add((left, right))
        left_clean = per_profile.get(left, {}).get("clean", {})
        right_clean = per_profile.get(right, {}).get("clean", {})
        left_sns = _mean([float(metrics.get("tampered_valid_mean_iou", 0.0) or 0.0) for profile, metrics in per_profile.get(left, {}).items() if profile != "clean"])
        right_sns = _mean([float(metrics.get("tampered_valid_mean_iou", 0.0) or 0.0) for profile, metrics in per_profile.get(right, {}).items() if profile != "clean"])
        pairs.append(
            {
                "comparison": f"{left}_vs_{right}",
                "left_model_id": left,
                "right_model_id": right,
                "clean_macro_f1_delta": right_clean.get("macro_f1", 0.0) - left_clean.get("macro_f1", 0.0),
                "clean_accuracy_delta": right_clean.get("accuracy", 0.0) - left_clean.get("accuracy", 0.0),
                "snsaug_tampered_valid_mean_iou_delta": (right_sns or 0.0) - (left_sns or 0.0),
            }
        )
    return {"marker": MARKER, "comparisons": pairs}


def _profiles_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    profiles = sorted({str(row.get("profile") or "clean") for row in rows})
    if "clean" in profiles:
        profiles.remove("clean")
    return ["clean", *profiles]


def _mask_area_pct(path: Any) -> float | None:
    if not path:
        return None
    try:
        from PIL import Image
        with Image.open(str(path)) as image:
            image = image.convert("L")
            values = list(image.getdata())
        return float(sum(1 for value in values if int(value) > 0) / max(len(values), 1) * 100.0)
    except Exception:
        return None


def _normalize_record(row: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    label = _norm_label(row.get("content_label"))
    valid_iou = row.get("valid_iou") if label == "tampered" else None
    raw_iou = row.get("raw_iou") if label == "tampered" else None
    p_real = row.get("p_real")
    p_synthetic = row.get("p_synthetic")
    p_tampered = row.get("p_tampered")
    return {
        "marker": MARKER,
        "model_id": model["model_id"],
        "model_kind": model["model_kind"],
        "base_id": str(row.get("base_id") or ""),
        "profile": str(row.get("profile") or ("clean" if str(row.get("view") or "") == "clean" else "unknown")),
        "view": str(row.get("view") or ""),
        "content_label": label,
        "pred_class": _norm_label(row.get("pred_class")),
        "p_real": float(p_real) if p_real is not None else None,
        "p_synthetic": float(p_synthetic) if p_synthetic is not None else None,
        "p_tampered": float(p_tampered) if p_tampered is not None else None,
        "localization_activated": bool(row.get("localization_activated")),
        "pred_mask_area_pct": float(row.get("final_mask_area_pct") or 0.0),
        "gt_mask_area_pct": _mask_area_pct(row.get("tamper_mask_path")),
        "valid_iou": valid_iou,
        "raw_iou": raw_iou,
        "ignore_mask_path": row.get("ignore_mask_path"),
        "tamper_mask_path": row.get("tamper_mask_path"),
        "image_path": row.get("image_path"),
        "pred_mask_path": row.get("pred_mask_path"),
        "pred_red_overlay_path": row.get("pred_red_overlay_path"),
        "gt_red_overlay_path": row.get("gt_red_overlay_path"),
        "ignore_blue_overlay_path": row.get("ignore_blue_overlay_path"),
        "overlap_overlay_path": row.get("overlap_overlay_path"),
        "pred_mask_available": bool(row.get("pred_mask_available")),
        "error": row.get("error"),
    }


def _evaluate_model_records(
    *,
    model: dict[str, Any],
    bundle: dict[str, Any],
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    model_output_root = output_root / "model_visuals" / str(model["model_id"])
    fixed_config = {
        "output_root": str(model_output_root),
        "approved_input_roots": list(config.get("approved_pair_roots", [])) + list(config.get("approved_model_roots", [])),
        "device": str(config.get("device", "cpu")),
        "write_empty_pred_mask": bool(config.get("write_empty_pred_mask", True)),
        "mask_threshold": float(config.get("mask_threshold", 0.45)),
    }
    raw_records = fixed_pair_evaluate_rows(bundle, fixed_config, rows)
    records = [_normalize_record(row, model) for row in raw_records]
    if any(record.get("p_tampered") is None for record in records):
        raise SNSAugV2CheckpointComparisonEvalError(f"model {model['model_id']} produced records without p_tampered")
    return records


def _prediction_signature(records: list[dict[str, Any]], model_id: str) -> list[tuple[Any, ...]]:
    return [
        (
            row.get("base_id"),
            row.get("profile"),
            row.get("view"),
            row.get("pred_class"),
            round(float(row.get("p_real") or 0.0), 8),
            round(float(row.get("p_synthetic") or 0.0), 8),
            round(float(row.get("p_tampered") or 0.0), 8),
            bool(row.get("localization_activated")),
        )
        for row in records
        if row.get("model_id") == model_id
    ]


def _metric_signature(metrics: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        round(float(metrics.get(name) or 0.0), 8)
        for name in METRIC_NAMES
        if name != "sample_count"
    )


def sanity_check_outputs(
    *,
    records: list[dict[str, Any]],
    per_profile: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    model_infos: list[dict[str, Any]],
) -> list[str]:
    if not summary.get("comparisons"):
        raise SNSAugV2CheckpointComparisonEvalError("checkpoint comparison sanity failed: comparisons is empty")
    if records and any("p_tampered" not in row for row in records):
        raise SNSAugV2CheckpointComparisonEvalError("checkpoint comparison sanity failed: model_eval_records missing p_tampered")
    profile_metrics = [metrics for profiles in per_profile.values() for metrics in profiles.values()]
    if profile_metrics and all(float(metrics.get("accuracy") or 0.0) == 1.0 for metrics in profile_metrics):
        raise SNSAugV2CheckpointComparisonEvalError("checkpoint comparison sanity failed: all model/profile accuracy values are exactly 1.0")
    signatures = {_metric_signature(metrics) for metrics in profile_metrics}
    if len(profile_metrics) > 1 and len(signatures) == 1:
        raise SNSAugV2CheckpointComparisonEvalError("checkpoint comparison sanity failed: all profiles have identical metrics across all models")
    warnings_out: list[str] = []
    for model_id, profiles in per_profile.items():
        for profile, metrics in profiles.items():
            if float(metrics.get("non_tampered_high_mask_rate") or 0.0) == 1.0 and float(metrics.get("real_fpr") or 0.0) == 0.0:
                message = f"{model_id}/{profile}: non_tampered_high_mask_rate=1.0 with real_fpr=0.0"
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                warnings_out.append(message)
    baseline_ids = [info["model_id"] for info in model_infos if info.get("model_kind") == "pre_sns_bundle"]
    baseline_id = baseline_ids[0] if baseline_ids else None
    baseline_sig = _prediction_signature(records, baseline_id) if baseline_id else []
    for info in model_infos:
        if info.get("model_kind") != "snsaug_finetuned_checkpoint":
            continue
        if not info.get("checkpoint_sha256"):
            raise SNSAugV2CheckpointComparisonEvalError(f"checkpoint_sha256 missing for {info['model_id']}")
        if baseline_sig and _prediction_signature(records, info["model_id"]) == baseline_sig and not info.get("no_weight_delta"):
            raise SNSAugV2CheckpointComparisonEvalError(
                f"fine-tuned checkpoint predictions are byte-identical to baseline for {info['model_id']}"
            )
    return warnings_out


def run_snsaug_v2_checkpoint_comparison_eval(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_exists=True)
    pair_root = _real(config["pair_root"])
    meta_path = _real(config.get("meta_jsonl_path") or (pair_root / "meta.jsonl"))
    output_root = _real(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    meta_rows = _load_jsonl(meta_path)
    profiles = list(config.get("profiles") or _profiles_from_rows(meta_rows))
    rows, parse_warnings = parse_fixed_pair_rows(meta_rows, profiles, config.get("max_samples"))
    if not rows:
        raise SNSAugV2CheckpointComparisonEvalError("fixed pair meta produced no evaluation rows")

    baseline_models = [model for model in config["models"] if model.get("model_kind") == "pre_sns_bundle"]
    if not baseline_models:
        raise SNSAugV2CheckpointComparisonEvalError("at least one pre_sns_bundle model is required")
    baseline_bundle = _resolve_baseline_bundle(baseline_models[0])
    derived_root = output_root / "_derived_checkpoints"
    model_bundles: dict[str, dict[str, Any]] = {}
    model_infos: list[dict[str, Any]] = []
    for model in config["models"]:
        path = _real(model["model_path"])
        info: dict[str, Any] = {
            "model_id": model["model_id"],
            "model_kind": model["model_kind"],
            "model_path": str(path),
        }
        if model["model_kind"] == "pre_sns_bundle":
            bundle = _resolve_baseline_bundle(model)
            model_bundles[model["model_id"]] = bundle
            if path.is_file():
                info.update(_checkpoint_metadata(path))
        else:
            bundle, checkpoint_info = _bundle_for_finetuned_checkpoint(model=model, baseline_bundle=baseline_bundle, derived_root=derived_root)
            model_bundles[model["model_id"]] = bundle
            info.update(checkpoint_info)
        model_infos.append(info)

    records: list[dict[str, Any]] = []
    for model in config["models"]:
        records.extend(
            _evaluate_model_records(
                model=model,
                bundle=model_bundles[model["model_id"]],
                config=config,
                rows=rows,
                output_root=output_root,
            )
        )
    comparisons = _build_comparisons(records)
    per_profile = _per_profile(records)
    drop = _drop_by_model(per_profile)
    model_ids = [model["model_id"] for model in config["models"]]
    summary = _comparison_summary(per_profile, model_ids)
    sanity_warnings = sanity_check_outputs(records=records, per_profile=per_profile, summary=summary, model_infos=model_infos)
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
        "checkpoint_comparison_summary": _write_json(
            output_root / "checkpoint_comparison_summary.json",
            {**summary, "model_checkpoint_info": model_infos, "sanity_warnings": sanity_warnings},
        ),
        "checkpoint_comparison_report": _write_text(
            output_root / "checkpoint_comparison_report.md",
            "# SNSAug V2 Checkpoint Comparison Evaluation\n\n"
            f"{MARKER}\n\n"
            "Real checkpoint inference was used. Runs with empty comparisons or all-perfect metrics are invalid.\n\n"
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
        "model_checkpoint_info": model_infos,
        "same_pair_root_for_all_models": True,
        "full_fixed_pair_evaluation_ran": full_ran,
        "eval_subset_only": subset,
        "sanity_warnings": sanity_warnings,
        "parse_warnings": parse_warnings,
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
