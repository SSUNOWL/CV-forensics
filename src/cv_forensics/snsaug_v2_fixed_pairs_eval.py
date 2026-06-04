"""Evaluation-only SNSAug v2 fixed-pairs audit for the frozen pre-SNS bundle."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from .pre_sns_v3_sns_robustness_eval import (
    _mean,
    _median,
    _safe_prob,
    confusion_matrix,
    json_safe,
    load_best_bundle,
    macro_f1_and_details,
    normalize_label,
    policy_config_from_bundle,
)
from .pre_sns_v3_v2_policy_gated_report import build_policy_gated_record

MARKER = "SNSAUG_V2_FIXED_PAIRS_EVAL_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_FIXED_PAIRS_EVAL_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_fixed_pairs_eval"
APPROVED_MODE = "approved_local_snsaug_v2_fixed_pairs_eval"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_FIXED_PAIRS_EVAL"
CLASS_LABELS = ("real", "synthetic", "tampered")
SUPPORTED_PROFILES = (
    "clean",
    "tiktok_like",
    "instagram_story_like",
    "youtube_shorts_like",
    "news_meme_overlay",
    "combined_sns_realistic",
)
NON_TAMPERED_HIGH_MASK_THRESHOLD_PCT = 1.0


class SNSAugV2FixedPairsEvalError(ValueError):
    """Raised when snsaug v2 fixed-pairs evaluation fails validation or execution."""


def _err(message: str) -> str:
    return f"- {message}"


def _real(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_under(path: str | Path, root: str | Path) -> bool:
    try:
        return str(_real(path)).startswith(str(_real(root)).rstrip("/") + "/") or _real(path) == _real(root)
    except Exception:
        return False


def _inside_repo(path: str | Path) -> bool:
    repo_root = Path(__file__).resolve().parents[2]
    return _is_under(path, repo_root)


def _as_roots(value: Any) -> list[str]:
    return value if isinstance(value, list) and all(isinstance(item, str) for item in value) else []


def _validate_local_path(value: Any, field: str, *, require_exists: bool = False) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [_err(f"{field} must be a non-empty absolute local path")]
    text = value.strip()
    errors: list[str] = []
    if not text.startswith("/"):
        errors.append(_err(f"{field} must be absolute"))
    if "://" in text:
        errors.append(_err(f"{field} must not use a remote scheme"))
    if require_exists and not Path(text).exists():
        errors.append(_err(f"{field} must exist"))
    return errors


def load_snsaug_v2_fixed_pairs_eval_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2FixedPairsEvalError("snsaug v2 fixed-pairs eval config root must be a JSON object")
    return raw


def validate_snsaug_v2_fixed_pairs_eval_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "user_approval_text",
        "best_bundle_path",
        "pair_root",
        "meta_jsonl_path",
        "approved_input_roots",
        "approved_output_roots",
        "output_root",
        "profiles",
        "device",
        "no_training",
        "no_finetune",
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
    if raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(_err(f"user_approval_text must be {APPROVAL_TEXT}"))
    for flag in ("no_training", "no_finetune", "no_network", "no_download"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    input_roots = _as_roots(raw.get("approved_input_roots"))
    output_roots = _as_roots(raw.get("approved_output_roots"))
    if not input_roots:
        errors.append(_err("approved_input_roots must be a non-empty list of absolute paths"))
    if not output_roots:
        errors.append(_err("approved_output_roots must be a non-empty list of absolute paths"))
    for index, root in enumerate(input_roots):
        errors.extend(_validate_local_path(root, f"approved_input_roots[{index}]", require_exists=False))
    for index, root in enumerate(output_roots):
        errors.extend(_validate_local_path(root, f"approved_output_roots[{index}]", require_exists=False))
    for field in ("best_bundle_path", "pair_root", "meta_jsonl_path"):
        errors.extend(_validate_local_path(raw.get(field), field, require_exists=require_exists))
        value = raw.get(field)
        if isinstance(value, str) and value.startswith("/") and input_roots and not any(_is_under(value, root) or _real(value) == _real(root) for root in input_roots):
            errors.append(_err(f"{field} must be under an approved input root"))
    output_root = raw.get("output_root")
    errors.extend(_validate_local_path(output_root, "output_root", require_exists=False))
    if isinstance(output_root, str) and output_root.startswith("/"):
        if _inside_repo(output_root):
            errors.append(_err("output_root must be outside repository"))
        if input_roots and any(_is_under(output_root, root) for root in input_roots):
            errors.append(_err("output_root must not be under approved input roots"))
        if output_roots and not any(_is_under(output_root, root) or _real(output_root) == _real(root) for root in output_roots):
            errors.append(_err("output_root must be under an approved output root"))
    profiles = raw.get("profiles")
    if not isinstance(profiles, list) or not profiles or not all(isinstance(item, str) for item in profiles):
        errors.append(_err("profiles must be a non-empty list of strings"))
    else:
        invalid = [item for item in profiles if item not in SUPPORTED_PROFILES]
        if invalid:
            errors.append(_err(f"unsupported profiles: {invalid}"))
        if "clean" not in profiles:
            errors.append(_err("profiles must include clean"))
    if raw.get("device") not in {"cpu", "cuda"}:
        errors.append(_err("device must be cpu or cuda"))
    if "max_samples" in raw and raw.get("max_samples") is not None:
        value = raw.get("max_samples")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(_err("max_samples must be a positive integer when present"))
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_fixed_pairs_eval_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2FixedPairsEvalError("snsaug v2 fixed-pairs eval config validation failed:\n" + "\n".join(errors))


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True))
            handle.write("\n")
    return str(path)


def load_meta_rows(path: str | Path) -> list[dict[str, Any]]:
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


def parse_fixed_pair_rows(rows: list[dict[str, Any]], profiles: list[str], max_samples: int | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    grouped: dict[str, dict[str, Any]] = {}
    duplicate_clean_ignored = 0
    for row in rows:
        base_id = str(row.get("base_id") or "")
        if not base_id:
            continue
        profile = str(row.get("profile") or "")
        view = str(row.get("view") or "")
        content_label = normalize_label(row.get("content_label"))
        if content_label not in CLASS_LABELS:
            continue
        bucket = grouped.setdefault(base_id, {"clean": None, "sns": [], "label": content_label})
        if view == "clean":
            if bucket["clean"] is None:
                bucket["clean"] = row
        elif view == "sns_aug":
            if profile == "clean":
                duplicate_clean_ignored += 1
                continue
            if profile in profiles:
                bucket["sns"].append(row)
    if duplicate_clean_ignored:
        warnings.append(f"ignored {duplicate_clean_ignored} sns_aug rows with profile=clean")
    parsed: list[dict[str, Any]] = []
    for base_id in sorted(grouped):
        bucket = grouped[base_id]
        clean = bucket["clean"]
        if not isinstance(clean, dict):
            continue
        parsed.append(clean)
        parsed.extend(bucket["sns"])
    if max_samples is not None:
        seen: set[str] = set()
        limited: list[dict[str, Any]] = []
        for row in parsed:
            base_id = str(row["base_id"])
            if row.get("view") == "clean":
                if len(seen) >= max_samples:
                    break
                seen.add(base_id)
            if base_id in seen:
                limited.append(row)
        parsed = limited
    return parsed, warnings


def group_by_base_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        base_id = str(row["base_id"])
        bucket = grouped.setdefault(base_id, {"clean": None, "sns": []})
        if row.get("view") == "clean":
            bucket["clean"] = row
        else:
            bucket["sns"].append(row)
    return grouped


def _runtime_deps():
    try:
        from PIL import Image
    except Exception as exc:
        raise SNSAugV2FixedPairsEvalError("PIL is required for snsaug v2 fixed-pairs eval") from exc
    return Image


def _load_mask(Image: Any, path: str | None) -> list[int] | None:
    if not path:
        return None
    with Image.open(path) as image:
        image = image.convert("L")
        return [1 if int(value) > 0 else 0 for value in image.getdata()]


def _shape_from_path(Image: Any, path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.size[0]), int(image.size[1])


def _iou(pred: list[int], gt: list[int]) -> float | None:
    if len(pred) != len(gt) or not pred:
        return None
    inter = sum(1 for p, g in zip(pred, gt) if p and g)
    union = sum(1 for p, g in zip(pred, gt) if p or g)
    return float(inter / union) if union else 0.0


def _dice(pred: list[int], gt: list[int]) -> float | None:
    if len(pred) != len(gt) or not pred:
        return None
    inter = sum(1 for p, g in zip(pred, gt) if p and g)
    total = sum(pred) + sum(gt)
    return float((2.0 * inter) / total) if total else 0.0


def compute_mask_metrics(pred_mask: list[int] | None, gt_mask: list[int] | None, ignore_mask: list[int] | None) -> dict[str, float | None]:
    if pred_mask is None or gt_mask is None:
        return {"raw_iou": None, "raw_dice": None, "valid_iou": None, "valid_dice": None}
    raw_iou = _iou(pred_mask, gt_mask)
    raw_dice = _dice(pred_mask, gt_mask)
    if ignore_mask is None or len(ignore_mask) != len(pred_mask):
        return {"raw_iou": raw_iou, "raw_dice": raw_dice, "valid_iou": raw_iou, "valid_dice": raw_dice}
    valid_pred = [int(p and not ig) for p, ig in zip(pred_mask, ignore_mask)]
    valid_gt = [int(g and not ig) for g, ig in zip(gt_mask, ignore_mask)]
    return {
        "raw_iou": raw_iou,
        "raw_dice": raw_dice,
        "valid_iou": _iou(valid_pred, valid_gt),
        "valid_dice": _dice(valid_pred, valid_gt),
    }


def evaluate_pair_record(bundle: dict[str, Any], config: dict[str, Any], row: dict[str, Any], index: int) -> dict[str, Any]:
    policy_config = policy_config_from_bundle(bundle, config, Path(config["output_root"]))
    sample = dict(row)
    sample["image_path"] = str(row["image_path"])
    if row.get("tamper_mask_path"):
        sample["gt_mask_path"] = str(row["tamper_mask_path"])
    started = time.perf_counter()
    result = build_policy_gated_record(policy_config, sample, index)
    latency_ms = (time.perf_counter() - started) * 1000.0
    Image = _runtime_deps()
    pred_mask = result.get("_final_mask")
    gt_mask = _load_mask(Image, str(row.get("tamper_mask_path"))) if row.get("tamper_mask_path") else None
    ignore_mask = _load_mask(Image, str(row.get("ignore_mask_path"))) if row.get("ignore_mask_path") else None
    metrics = compute_mask_metrics(pred_mask, gt_mask, ignore_mask)
    pred_class = normalize_label(result.get("class"))
    label = normalize_label(row.get("content_label"))
    return {
        "base_id": str(row["base_id"]),
        "content_label": label,
        "view": str(row.get("view")),
        "profile": str(row.get("profile")),
        "seed": row.get("seed"),
        "image_path": str(row.get("image_path")),
        "tamper_mask_path": str(row.get("tamper_mask_path")) if row.get("tamper_mask_path") else None,
        "ignore_mask_path": str(row.get("ignore_mask_path")) if row.get("ignore_mask_path") else None,
        "pred_class": pred_class,
        "class_correct": pred_class == label,
        "p_real": _safe_prob(result.get("class_conf", {}), "real"),
        "p_synthetic": _safe_prob(result.get("class_conf", {}), "synthetic"),
        "p_tampered": _safe_prob(result.get("class_conf", {}), "tampered"),
        "tampered_score": float(result.get("tampered_score", 0.0)),
        "localization_activated": bool(result.get("tile_localization_activated")),
        "final_mask_source": result.get("final_mask_source"),
        "final_mask_area_pct": float(result.get("final_mask_area_pct", 0.0)),
        "raw_iou": metrics["raw_iou"],
        "raw_dice": metrics["raw_dice"],
        "valid_iou": metrics["valid_iou"],
        "valid_dice": metrics["valid_dice"],
        "latency_ms": float(latency_ms),
        "error": None,
        "_result": result,
    }


def evaluate_rows(bundle: dict[str, Any], config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            records.append(evaluate_pair_record(bundle, config, row, index))
        except Exception as exc:
            records.append(
                {
                    "base_id": str(row.get("base_id") or f"row_{index:06d}"),
                    "content_label": normalize_label(row.get("content_label")),
                    "view": str(row.get("view")),
                    "profile": str(row.get("profile")),
                    "seed": row.get("seed"),
                    "image_path": str(row.get("image_path")),
                    "tamper_mask_path": str(row.get("tamper_mask_path")) if row.get("tamper_mask_path") else None,
                    "ignore_mask_path": str(row.get("ignore_mask_path")) if row.get("ignore_mask_path") else None,
                    "pred_class": None,
                    "class_correct": False,
                    "p_real": None,
                    "p_synthetic": None,
                    "p_tampered": None,
                    "tampered_score": None,
                    "localization_activated": None,
                    "final_mask_source": None,
                    "final_mask_area_pct": None,
                    "raw_iou": None,
                    "raw_dice": None,
                    "valid_iou": None,
                    "valid_dice": None,
                    "latency_ms": None,
                    "error": str(exc),
                }
            )
    return records


def join_clean_and_sns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_by_base = {row["base_id"]: row for row in records if row["view"] == "clean"}
    comparisons: list[dict[str, Any]] = []
    for row in records:
        if row["view"] == "clean":
            continue
        clean = clean_by_base.get(row["base_id"])
        if clean is None:
            continue
        p_tampered_drop = None
        if clean.get("p_tampered") is not None and row.get("p_tampered") is not None:
            p_tampered_drop = float(clean["p_tampered"]) - float(row["p_tampered"])
        valid_iou_drop = None
        if clean.get("valid_iou") is not None and row.get("valid_iou") is not None:
            valid_iou_drop = float(clean["valid_iou"]) - float(row["valid_iou"])
        raw_iou_drop = None
        if clean.get("raw_iou") is not None and row.get("raw_iou") is not None:
            raw_iou_drop = float(clean["raw_iou"]) - float(row["raw_iou"])
        activation_flip_off = bool(clean.get("localization_activated")) and not bool(row.get("localization_activated"))
        comparison = {
            "base_id": row["base_id"],
            "content_label": row["content_label"],
            "profile": row["profile"],
            "seed": row.get("seed"),
            "clean_pred_class": clean.get("pred_class"),
            "sns_pred_class": row.get("pred_class"),
            "pred_flip": clean.get("pred_class") != row.get("pred_class"),
            "clean_correct": clean.get("class_correct"),
            "sns_correct": row.get("class_correct"),
            "correct_to_wrong": bool(clean.get("class_correct")) and not bool(row.get("class_correct")),
            "wrong_to_correct": (not bool(clean.get("class_correct"))) and bool(row.get("class_correct")),
            "clean_p_tampered": clean.get("p_tampered"),
            "sns_p_tampered": row.get("p_tampered"),
            "p_tampered_drop": p_tampered_drop,
            "clean_valid_iou": clean.get("valid_iou"),
            "sns_valid_iou": row.get("valid_iou"),
            "valid_iou_drop": valid_iou_drop,
            "clean_raw_iou": clean.get("raw_iou"),
            "sns_raw_iou": row.get("raw_iou"),
            "raw_iou_drop": raw_iou_drop,
            "clean_localization_activated": clean.get("localization_activated"),
            "sns_localization_activated": row.get("localization_activated"),
            "activation_flip_off": activation_flip_off,
        }
        comparison["fragile_class_flip"] = bool(clean.get("class_correct")) and not bool(row.get("class_correct"))
        comparison["fragile_confidence_drop"] = row["content_label"] == "tampered" and p_tampered_drop is not None and float(p_tampered_drop) >= 0.25
        comparison["fragile_mask_drop"] = row["content_label"] == "tampered" and valid_iou_drop is not None and float(valid_iou_drop) >= 0.20
        comparison["fragile_activation_flip"] = activation_flip_off
        comparisons.append(comparison)
    return comparisons


def aggregate_per_profile(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_profile.setdefault(str(row["profile"]), []).append(row)
    for profile, items in by_profile.items():
        matrix = confusion_matrix([{"label": row["content_label"], "pred_class": row.get("pred_class")} for row in items if row.get("pred_class")])
        cls = macro_f1_and_details(matrix)
        real_items = [row for row in items if row["content_label"] == "real"]
        synthetic_items = [row for row in items if row["content_label"] == "synthetic"]
        tampered_items = [row for row in items if row["content_label"] == "tampered"]
        valid_ious = [float(row["valid_iou"]) for row in tampered_items if row.get("valid_iou") is not None]
        raw_ious = [float(row["raw_iou"]) for row in tampered_items if row.get("raw_iou") is not None]
        valid_dices = [float(row["valid_dice"]) for row in tampered_items if row.get("valid_dice") is not None]
        latencies = [float(row["latency_ms"]) for row in items if row.get("latency_ms") is not None]
        out[profile] = {
            "sample_count": len(items),
            "accuracy": sum(1 for row in items if row.get("class_correct")) / len(items) if items else 0.0,
            "macro_f1": cls["macro_f1"],
            "confusion_matrix": matrix,
            "real_fpr": (sum(1 for row in real_items if row.get("pred_class") != "real") / len(real_items)) if real_items else 0.0,
            "synthetic_recall": cls["per_class"]["synthetic"]["recall"],
            "tampered_recall": cls["per_class"]["tampered"]["recall"],
            "tampered_raw_mean_iou": _mean(raw_ious),
            "tampered_raw_median_iou": _median(raw_ious),
            "tampered_valid_mean_iou": _mean(valid_ious),
            "tampered_valid_median_iou": _median(valid_ious),
            "tampered_valid_mean_dice": _mean(valid_dices),
            "localization_activation_recall": (sum(1 for row in tampered_items if row.get("localization_activated")) / len(tampered_items)) if tampered_items else None,
            "non_tampered_high_mask_rate": (
                sum(1 for row in items if row["content_label"] != "tampered" and float(row.get("final_mask_area_pct") or 0.0) > NON_TAMPERED_HIGH_MASK_THRESHOLD_PCT)
                / max(1, sum(1 for row in items if row["content_label"] != "tampered"))
            ) if any(row["content_label"] != "tampered" for row in items) else 0.0,
            "mean_latency_ms": _mean(latencies),
            "fps": (1000.0 / _mean(latencies)) if latencies and _mean(latencies) not in {None, 0.0} else None,
        }
    return out


def robustness_drop_metrics(comparisons: list[dict[str, Any]], per_profile: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    clean = per_profile.get("clean", {})
    out: dict[str, dict[str, Any]] = {}
    for profile, metrics in per_profile.items():
        if profile == "clean":
            continue
        tampered_rows = [row for row in comparisons if row["profile"] == profile and row["content_label"] == "tampered"]
        p_drops = [float(row["p_tampered_drop"]) for row in tampered_rows if row.get("p_tampered_drop") is not None]
        out[profile] = {
            "accuracy_drop": float(clean.get("accuracy") or 0.0) - float(metrics.get("accuracy") or 0.0),
            "macro_f1_drop": float(clean.get("macro_f1") or 0.0) - float(metrics.get("macro_f1") or 0.0),
            "real_fpr_increase": float(metrics.get("real_fpr") or 0.0) - float(clean.get("real_fpr") or 0.0),
            "synthetic_recall_drop": float(clean.get("synthetic_recall") or 0.0) - float(metrics.get("synthetic_recall") or 0.0),
            "tampered_recall_drop": float(clean.get("tampered_recall") or 0.0) - float(metrics.get("tampered_recall") or 0.0),
            "valid_iou_drop": float(clean.get("tampered_valid_mean_iou") or 0.0) - float(metrics.get("tampered_valid_mean_iou") or 0.0),
            "raw_iou_drop": float(clean.get("tampered_raw_mean_iou") or 0.0) - float(metrics.get("tampered_raw_mean_iou") or 0.0),
            "localization_activation_recall_drop": float(clean.get("localization_activation_recall") or 0.0) - float(metrics.get("localization_activation_recall") or 0.0),
            "mean_p_tampered_drop_on_tampered": _mean(p_drops),
        }
    return out


def worst_samples(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(comparisons)
    rows.sort(
        key=lambda row: (
            0 if row.get("fragile_class_flip") else 1,
            -float(row.get("valid_iou_drop") or 0.0),
            -float(row.get("p_tampered_drop") or 0.0),
            0 if row.get("activation_flip_off") else 1,
            0 if (row["content_label"] == "real" and row.get("sns_pred_class") != "real") else 1,
            0 if (row["content_label"] == "synthetic" and row.get("sns_pred_class") != "synthetic") else 1,
            str(row["base_id"]),
            str(row["profile"]),
        )
    )
    return rows


def fragile_candidates(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in comparisons
        if row.get("fragile_class_flip") or row.get("fragile_confidence_drop") or row.get("fragile_mask_drop") or row.get("fragile_activation_flip")
    ]


def run_snsaug_v2_fixed_pairs_eval(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_exists=True)
    output_root = _real(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    meta_rows = load_meta_rows(config["meta_jsonl_path"])
    parsed_rows, warnings = parse_fixed_pair_rows(meta_rows, list(config["profiles"]), config.get("max_samples"))
    bundle = load_best_bundle(config["best_bundle_path"])
    started = time.perf_counter()
    records = evaluate_rows(bundle, config, parsed_rows)
    comparisons = join_clean_and_sns(records)
    per_profile = aggregate_per_profile(records)
    drop_metrics = robustness_drop_metrics(comparisons, per_profile)
    worst = worst_samples(comparisons)
    fragile = fragile_candidates(comparisons)
    summary = {
        "marker": MARKER,
        "pair_root": str(_real(config["pair_root"])),
        "record_count": len(records),
        "comparison_count": len(comparisons),
        "warning_count": len(warnings),
        "warnings": warnings,
        "profiles": list(config["profiles"]),
        "elapsed_sec": time.perf_counter() - started,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    visual_gallery = {
        "top_worst_cases": [
            {
                "base_id": row["base_id"],
                "profile": row["profile"],
                "content_label": row["content_label"],
                "clean_valid_iou": row.get("clean_valid_iou"),
                "sns_valid_iou": row.get("sns_valid_iou"),
            }
            for row in worst[: min(12, len(worst))]
        ]
    }
    artifact_manifest = {
        "marker": MARKER,
        "records_jsonl": str(output_root / "snsaug_v2_eval_records.jsonl"),
        "comparisons_jsonl": str(output_root / "snsaug_v2_eval_comparisons.jsonl"),
        "per_profile_metrics_json": str(output_root / "snsaug_v2_per_profile_metrics.json"),
        "robustness_drop_metrics_json": str(output_root / "snsaug_v2_robustness_drop_metrics.json"),
        "summary_json": str(output_root / "snsaug_v2_eval_summary.json"),
        "worst_samples_json": str(output_root / "snsaug_v2_worst_samples.json"),
        "fragile_candidates_jsonl": str(output_root / "snsaug_v2_fragile_candidates.jsonl"),
        "visual_gallery_manifest_json": str(output_root / "visual_gallery_manifest.json"),
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    _write_jsonl(output_root / "snsaug_v2_eval_records.jsonl", [{k: v for k, v in row.items() if not str(k).startswith("_")} for row in records])
    _write_jsonl(output_root / "snsaug_v2_eval_comparisons.jsonl", comparisons)
    _write_json(output_root / "snsaug_v2_per_profile_metrics.json", per_profile)
    _write_json(output_root / "snsaug_v2_robustness_drop_metrics.json", drop_metrics)
    _write_json(output_root / "snsaug_v2_eval_summary.json", summary)
    _write_json(output_root / "snsaug_v2_worst_samples.json", worst)
    _write_jsonl(output_root / "snsaug_v2_fragile_candidates.jsonl", fragile)
    _write_json(output_root / "visual_gallery_manifest.json", visual_gallery)
    _write_json(output_root / "artifact_manifest.json", artifact_manifest)
    return summary
