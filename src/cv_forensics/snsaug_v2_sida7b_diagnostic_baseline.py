"""Offline SIDA-7B/SIDA-style diagnostic baseline for SNSAug fixed pairs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .snsaug_v2_fixed_pairs_eval import compute_mask_metrics, load_meta_rows, parse_fixed_pair_rows

MARKER = "SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_sida7b_diagnostic_baseline"
APPROVED_MODE = "approved_local_snsaug_v2_sida7b_diagnostic_baseline"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE"
REPO_ROOT = Path(__file__).resolve().parents[2]
CLASS_LABELS = ("real", "synthetic", "tampered")
EXPORT_MODE = "export_sida_eval_subset_mode"
CACHED_MODE = "cached_sida_output_mode"
LOCAL_OVERLAY_PROFILES = {
    "news_meme_overlay",
    "platform_ui_same_size",
    "tiktok_like",
    "instagram_story_like",
    "youtube_shorts_like",
    "combined_sns_realistic",
}
GLOBAL_GEOMETRY_PROFILES = {
    "canvas_9x16_only",
    "resize_crop_pad",
    "zoom_crop",
    "resize_jpeg",
    "screenshot_recapture_light",
    "recompression_light",
}
SUPPORTED_PROFILES = tuple(sorted({"clean"} | LOCAL_OVERLAY_PROFILES | GLOBAL_GEOMETRY_PROFILES))
STRICT_COMPARISON_PROFILES = tuple(sorted(LOCAL_OVERLAY_PROFILES | GLOBAL_GEOMETRY_PROFILES))


class SNSAugV2SIDA7BDiagnosticBaselineError(ValueError):
    """Raised when SIDA diagnostic baseline validation or execution fails."""


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


def _roots(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _abs_errors(value: Any, field: str, require_exists: bool = False) -> list[str]:
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


def _under_errors(value: Any, field: str, roots: list[str], require_exists: bool = False) -> list[str]:
    errors = _abs_errors(value, field, require_exists=require_exists)
    if isinstance(value, str) and roots and not any(_is_under(value, root) or _real(value) == _real(root) for root in roots):
        errors.append(f"{field} must be under approved roots")
    return errors


def load_snsaug_v2_sida7b_diagnostic_baseline_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2SIDA7BDiagnosticBaselineError("SIDA diagnostic baseline config root must be a JSON object")
    raw.setdefault("config_path", str(_real(path)))
    return raw


def validate_snsaug_v2_sida7b_diagnostic_baseline_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "user_approval_text",
        "mode",
        "pair_root",
        "meta_jsonl_path",
        "approved_input_roots",
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
    if raw.get("mode") not in {EXPORT_MODE, CACHED_MODE}:
        errors.append(f"mode must be {EXPORT_MODE} or {CACHED_MODE}")
    for flag in ("no_training", "no_finetune", "no_network", "no_download"):
        if raw.get(flag) is not True:
            errors.append(f"{flag} must be true")
    input_roots = _roots(raw.get("approved_input_roots"))
    output_roots = _roots(raw.get("approved_output_roots"))
    if not input_roots:
        errors.append("approved_input_roots must be a non-empty list of absolute paths")
    if not output_roots:
        errors.append("approved_output_roots must be a non-empty list of absolute paths")
    for index, root in enumerate(input_roots):
        errors.extend(_abs_errors(root, f"approved_input_roots[{index}]", require_exists=require_exists))
    for index, root in enumerate(output_roots):
        errors.extend(_abs_errors(root, f"approved_output_roots[{index}]", require_exists=require_exists))
    for field in ("pair_root", "meta_jsonl_path", "cached_sida_outputs_path", "final_policy_gate_records_path", "final_policy_gate_metrics_path"):
        if raw.get(field):
            errors.extend(_under_errors(raw.get(field), field, input_roots, require_exists=require_exists))
    if raw.get("mode") == CACHED_MODE and not raw.get("cached_sida_outputs_path"):
        errors.append("cached_sida_outputs_path is required in cached_sida_output_mode")
    output_root = raw.get("output_root")
    errors.extend(_abs_errors(output_root, "output_root"))
    if isinstance(output_root, str) and output_root.strip():
        if _inside_repo(output_root):
            errors.append("output_root must be outside repository")
        if any(_is_under(output_root, root) or _real(output_root) == _real(root) for root in input_roots):
            errors.append("output_root must not be under approved input roots")
        if output_roots and not any(_is_under(output_root, root) or _real(output_root) == _real(root) for root in output_roots):
            errors.append("output_root must be under approved output roots")
    if "max_per_label_profile" in raw:
        value = raw.get("max_per_label_profile")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append("max_per_label_profile must be a positive integer")
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_sida7b_diagnostic_baseline_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2SIDA7BDiagnosticBaselineError("SIDA diagnostic baseline config validation failed:\n" + "\n".join(errors))


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_safe(row), ensure_ascii=True, sort_keys=True) + "\n")
    return str(path)


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def profile_family(profile: str) -> str:
    if profile == "clean":
        return "clean"
    if profile in LOCAL_OVERLAY_PROFILES:
        return "type_a_local_overlay"
    if profile in GLOBAL_GEOMETRY_PROFILES:
        return "type_b_global_geometry_degradation"
    return "other"


def normalize_class(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if "tamper" in text or "manipulated" in text or "manipulation" in text or "altered" in text or "edited" in text:
        return "tampered"
    if "synthetic" in text or "fully synthetic" in text or "full synthetic" in text or "fake" in text or "ai-generated" in text or "ai generated" in text or "generated" in text:
        return "synthetic"
    if "real" in text or "authentic" in text or "natural" in text:
        return "real"
    return None


def raw_output_audit(rows: list[dict[str, Any]]) -> dict[str, int]:
    counters = {
        "raw_contains_cls_count": 0,
        "raw_contains_seg_count": 0,
        "raw_contains_real_count": 0,
        "raw_contains_synthetic_count": 0,
        "raw_contains_fully_synthetic_count": 0,
        "raw_contains_generated_count": 0,
        "raw_contains_tampered_count": 0,
        "raw_contains_manipulated_count": 0,
    }
    for row in rows:
        raw = str(row.get("sida_text_output") or "").lower()
        counters["raw_contains_cls_count"] += int("[cls]" in raw)
        counters["raw_contains_seg_count"] += int("[seg]" in raw)
        counters["raw_contains_real_count"] += int(bool(re.search(r"\breal\b|\bauthentic\b|\bnatural\b", raw)))
        counters["raw_contains_synthetic_count"] += int("synthetic" in raw)
        counters["raw_contains_fully_synthetic_count"] += int("fully synthetic" in raw or "full synthetic" in raw)
        counters["raw_contains_generated_count"] += int("generated" in raw)
        counters["raw_contains_tampered_count"] += int("tamper" in raw or "altered" in raw)
        counters["raw_contains_manipulated_count"] += int("manipulated" in raw or "manipulation" in raw)
    return counters


def _class_hits(text: str) -> set[str]:
    lowered = text.lower()
    tamper_probe = re.sub(r"\b(?:not|no|without)\s+(?:signs?\s+of\s+)?(?:tamper(?:ed|ing)?|manipulat(?:ed|ion)|alter(?:ed|ation)|edit(?:ed|ing)?)\b", " ", lowered)
    hits: set[str] = set()
    if re.search(r"\btamper(?:ed|ing)?\b|\bmanipulat(?:ed|ion)\b|\baltered\b|\bedited\b", tamper_probe):
        hits.add("tampered")
    if re.search(r"\bfull(?:y)? synthetic\b|\bsynthetic\b|\bfake\b|\bai[- ]generated\b|\bgenerated\b", lowered):
        hits.add("synthetic")
    if re.search(r"\breal\b|\bauthentic\b|\bnatural\b", lowered):
        hits.add("real")
    return hits


def _explicit_cls_span(text: str) -> str | None:
    lowered = text.lower()
    marker = "[cls]"
    if marker not in lowered:
        return None
    after = text[lowered.index(marker) + len(marker):]
    lowered_after = after.lower()
    for stop in ("[seg]", "\n", ";", "."):
        if stop in lowered_after:
            after = after[: lowered_after.index(stop)]
            break
    return after


def parse_sida_text_output_detail(text: Any) -> dict[str, Any]:
    raw = str(text or "")
    explicit = _explicit_cls_span(raw)
    spans = [explicit] if explicit is not None else []
    spans.append(raw)
    for index, span in enumerate(spans):
        hits = _class_hits(str(span or ""))
        if "tampered" in hits:
            if "synthetic" in hits or "real" in hits:
                return {"pred_class": "unknown", "parse_error": True, "parse_warning": "ambiguous_class_terms", "class_hits": sorted(hits), "used_explicit_cls": explicit is not None and index == 0}
            return {"pred_class": "tampered", "parse_error": False, "parse_warning": None, "class_hits": sorted(hits), "used_explicit_cls": explicit is not None and index == 0}
        if hits == {"synthetic"}:
            return {"pred_class": "synthetic", "parse_error": False, "parse_warning": None, "class_hits": sorted(hits), "used_explicit_cls": explicit is not None and index == 0}
        if hits == {"real"}:
            return {"pred_class": "real", "parse_error": False, "parse_warning": None, "class_hits": sorted(hits), "used_explicit_cls": explicit is not None and index == 0}
        if len(hits) > 1:
            return {"pred_class": "unknown", "parse_error": True, "parse_warning": "ambiguous_class_terms", "class_hits": sorted(hits), "used_explicit_cls": explicit is not None and index == 0}
    return {"pred_class": "unknown", "parse_error": True, "parse_warning": "no_class_terms", "class_hits": [], "used_explicit_cls": explicit is not None}


def parse_sida_text_output(text: Any) -> str | None:
    parsed = parse_sida_text_output_detail(text).get("pred_class")
    return parsed if parsed in CLASS_LABELS else None


def _image_id(row: dict[str, Any]) -> str:
    return "__".join(str(row.get(key) or "") for key in ("base_id", "profile", "view", "content_label"))


def load_subset_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    profiles = list(dict.fromkeys(["clean", *sorted(LOCAL_OVERLAY_PROFILES), *sorted(GLOBAL_GEOMETRY_PROFILES)]))
    rows, warnings = parse_fixed_pair_rows(load_meta_rows(config["meta_jsonl_path"]), profiles, config.get("max_samples"))
    return rows, warnings


def build_balanced_subset(rows: list[dict[str, Any]], max_per_label_profile: int = 2) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        label = normalize_class(row.get("content_label"))
        profile = str(row.get("profile") or "")
        if label not in CLASS_LABELS or profile not in SUPPORTED_PROFILES:
            continue
        buckets.setdefault((label, profile), []).append(row)
    out: list[dict[str, Any]] = []
    for key in sorted(buckets):
        for row in sorted(buckets[key], key=lambda item: str(item.get("base_id")))[:max_per_label_profile]:
            out.append(row)
    return out


def make_prompt(row: dict[str, Any]) -> str:
    return (
        "You are evaluating a social-media forensic image. "
        "Return SIDA-style output with [CLS] one of real, synthetic, tampered. "
        "If [CLS] is tampered, also return [SEG] with a mask description or mask reference. "
        "Optionally include a short explanation only if your SIDA runner supports description output."
    )


def export_subset_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        image_id = _image_id(row)
        item = {
            "marker": MARKER,
            "image_id": image_id,
            "row_index": index,
            "base_id": str(row.get("base_id") or ""),
            "profile": str(row.get("profile") or ""),
            "profile_family": profile_family(str(row.get("profile") or "")),
            "view": str(row.get("view") or ""),
            "content_label": normalize_class(row.get("content_label")),
            "image_path": str(row.get("image_path") or ""),
            "tamper_mask_path": str(row.get("tamper_mask_path")) if row.get("tamper_mask_path") else None,
            "ignore_mask_path": str(row.get("ignore_mask_path")) if row.get("ignore_mask_path") else None,
        }
        manifest.append(item)
        prompts.append({"marker": MARKER, "image_id": image_id, "image_path": item["image_path"], "prompt": make_prompt(row)})
    return manifest, prompts


def _runtime_image():
    try:
        from PIL import Image
    except Exception as exc:
        raise SNSAugV2SIDA7BDiagnosticBaselineError("PIL is required for mask IoU evaluation") from exc
    return Image


def _mask_values(path: str | None, size: tuple[int, int] | None = None) -> tuple[list[int], tuple[int, int]] | tuple[None, None]:
    if not path:
        return None, None
    Image = _runtime_image()
    with Image.open(path) as image:
        mask = image.convert("L")
        if size is not None:
            resample = getattr(getattr(Image, "Resampling", Image), "NEAREST")
            mask = mask.resize(size, resample)
        return [1 if int(value) > 0 else 0 for value in mask.getdata()], (int(mask.size[0]), int(mask.size[1]))


def mask_valid_iou(pred_mask_path: str | None, gt_mask_path: str | None, ignore_mask_path: str | None = None) -> float | None:
    gt, shape = _mask_values(gt_mask_path)
    if gt is None or shape is None:
        return None
    pred, _pred_shape = _mask_values(pred_mask_path, shape)
    if pred is None:
        return None
    ignore, _ignore_shape = _mask_values(ignore_mask_path, shape)
    return compute_mask_metrics(pred, gt, ignore)["valid_iou"]


def _response_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("base_id") or ""), str(row.get("profile") or ""), str(row.get("content_label") or ""))


def parse_cached_sida_outputs(cached_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_by_id = {str(row.get("image_id")): row for row in manifest_rows}
    manifest_by_key = {_response_key(row): row for row in manifest_rows}
    out: list[dict[str, Any]] = []
    for row in cached_rows:
        manifest = manifest_by_id.get(str(row.get("image_id") or "")) or manifest_by_key.get(_response_key(row)) or {}
        parsed_detail = parse_sida_text_output_detail(row.get("sida_text_output"))
        pred = parsed_detail["pred_class"] if parsed_detail["pred_class"] in CLASS_LABELS else normalize_class(row.get("sida_pred_class"))
        if pred not in CLASS_LABELS:
            pred = None
        label = normalize_class(row.get("content_label")) or normalize_class(manifest.get("content_label"))
        mask_path = row.get("sida_mask_path")
        raw_contains_seg = "[seg]" in str(row.get("sida_text_output") or "").lower()
        mask_missing = bool(row.get("mask_missing")) or (label == "tampered" and not mask_path)
        valid_iou = mask_valid_iou(str(mask_path), manifest.get("tamper_mask_path"), manifest.get("ignore_mask_path")) if mask_path else None
        parse_error = bool(row.get("parse_error")) or bool(parsed_detail.get("parse_error")) or pred is None
        out.append(
            {
                "marker": MARKER,
                "image_id": row.get("image_id") or manifest.get("image_id"),
                "image_path": row.get("image_path") or manifest.get("image_path"),
                "base_id": row.get("base_id") or manifest.get("base_id"),
                "profile": row.get("profile") or manifest.get("profile"),
                "profile_family": row.get("profile_family") or manifest.get("profile_family"),
                "content_label": label,
                "sida_text_output": row.get("sida_text_output"),
                "sida_pred_class": pred,
                "sida_p_real": row.get("sida_p_real"),
                "sida_p_synthetic": row.get("sida_p_synthetic"),
                "sida_p_tampered": row.get("sida_p_tampered"),
                "sida_mask_path": mask_path,
                "valid_iou": valid_iou,
                "mask_missing": mask_missing,
                "raw_contains_seg": raw_contains_seg,
                "mask_extraction_failed": bool(raw_contains_seg and not mask_path),
                "mask_not_requested_or_not_generated": bool(not raw_contains_seg),
                "parse_error": parse_error,
                "parse_warning": parsed_detail.get("parse_warning"),
                "class_parse_hits": parsed_detail.get("class_hits", []),
                "class_correct": pred == label if pred and label else False,
            }
        )
    return out


def _confusion(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {label: {pred: 0 for pred in CLASS_LABELS} for label in CLASS_LABELS}
    for row in records:
        label = row.get("content_label")
        pred = row.get("sida_pred_class")
        if label in CLASS_LABELS and pred in CLASS_LABELS:
            matrix[label][pred] += 1
    return matrix


def _recall(matrix: dict[str, dict[str, int]], label: str) -> float | None:
    total = sum(matrix[label].values())
    return matrix[label][label] / total if total else None


def _precision(matrix: dict[str, dict[str, int]], label: str) -> float | None:
    total = sum(matrix[truth][label] for truth in CLASS_LABELS)
    return matrix[label][label] / total if total else None


def _macro_f1(matrix: dict[str, dict[str, int]]) -> float | None:
    scores: list[float] = []
    for label in CLASS_LABELS:
        p = _precision(matrix, label)
        r = _recall(matrix, label)
        if p is None or r is None or p + r == 0:
            continue
        scores.append((2 * p * r) / (p + r))
    return sum(scores) / len(scores) if scores else None


def _mean(values: list[Any]) -> float | None:
    xs = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return sum(xs) / len(xs) if xs else None


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = _confusion(records)
    total = len(records)
    real = [row for row in records if row.get("content_label") == "real"]
    synthetic = [row for row in records if row.get("content_label") == "synthetic"]
    tampered = [row for row in records if row.get("content_label") == "tampered"]
    return {
        "row_count": total,
        "accuracy": (sum(1 for row in records if row.get("class_correct")) / total) if total else None,
        "macro_f1": _macro_f1(matrix),
        "confusion_matrix": matrix,
        "real_fpr": (sum(1 for row in real if row.get("sida_pred_class") != "real") / len(real)) if real else None,
        "synthetic_recall": _recall(matrix, "synthetic") if synthetic else None,
        "tampered_recall": _recall(matrix, "tampered") if tampered else None,
        "tampered_valid_mean_iou": _mean([row.get("valid_iou") for row in tampered]),
        "mask_missing_rate": (sum(1 for row in tampered if row.get("mask_missing")) / len(tampered)) if tampered else None,
        "raw_seg_rate": (sum(1 for row in records if row.get("raw_contains_seg")) / total) if total else None,
        "mask_extraction_failed_rate": (sum(1 for row in records if row.get("mask_extraction_failed")) / total) if total else None,
        "mask_not_requested_or_not_generated_rate": (sum(1 for row in records if row.get("mask_not_requested_or_not_generated")) / total) if total else None,
        "parse_error_rate": (sum(1 for row in records if row.get("parse_error")) / total) if total else None,
    }


def per_profile_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for profile in sorted({str(row.get("profile") or "") for row in records}):
        out[profile] = compute_metrics([row for row in records if row.get("profile") == profile])
    return out


def type_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type_a_local_overlay": compute_metrics([row for row in records if row.get("profile_family") == "type_a_local_overlay"]),
        "type_b_global_geometry_degradation": compute_metrics([row for row in records if row.get("profile_family") == "type_b_global_geometry_degradation"]),
        "clean": compute_metrics([row for row in records if row.get("profile") == "clean"]),
    }


def clean_to_sns_drops(metrics: dict[str, Any]) -> dict[str, Any]:
    clean = metrics.get("clean", {})
    out: dict[str, Any] = {}
    for profile, item in metrics.items():
        if profile == "clean":
            continue
        out[profile] = {
            "tampered_recall_drop": None if clean.get("tampered_recall") is None or item.get("tampered_recall") is None else clean["tampered_recall"] - item["tampered_recall"],
            "valid_iou_drop": None if clean.get("tampered_valid_mean_iou") is None or item.get("tampered_valid_mean_iou") is None else clean["tampered_valid_mean_iou"] - item["tampered_valid_mean_iou"],
            "accuracy_drop": None if clean.get("accuracy") is None or item.get("accuracy") is None else clean["accuracy"] - item["accuracy"],
        }
    return out


def _mixed_gate_metrics_from_records(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    rows = _load_jsonl(path)
    out: dict[str, Any] = {}
    for profile in sorted({str(row.get("profile") or "") for row in rows if row.get("profile")}):
        selected = [row for row in rows if str(row.get("profile") or "") == profile]
        tampered = [row for row in selected if normalize_class(row.get("content_label")) == "tampered"]
        synthetic = [row for row in selected if normalize_class(row.get("content_label")) == "synthetic"]
        real = [row for row in selected if normalize_class(row.get("content_label")) == "real"]
        out[profile] = {
            "row_count": len(selected),
            "tampered_recall": (sum(1 for row in tampered if str(row.get("pred_class") or "") == "tampered" or row.get("pred_tampered") is True) / len(tampered)) if tampered else None,
            "synthetic_recall": (sum(1 for row in synthetic if str(row.get("pred_class") or "") == "synthetic") / len(synthetic)) if synthetic else None,
            "real_fpr": (sum(1 for row in real if str(row.get("pred_class") or "") != "real" or row.get("pred_tampered") is True) / len(real)) if real else None,
            "tampered_valid_mean_iou": _mean([row.get("valid_iou") for row in tampered]),
            "mean_p_tampered": _mean([row.get("p_tampered") for row in selected]),
        }
    return out


def load_mixed_gate_metrics(path: str | None, records_path: str | None = None) -> dict[str, Any]:
    if not path:
        return _mixed_gate_metrics_from_records(records_path)
    payload = _load_json(path)
    comparison = payload.get("comparison_against_fixed_original") if isinstance(payload.get("comparison_against_fixed_original"), dict) else {}
    out: dict[str, Any] = {}
    for profile, item in comparison.items():
        if not isinstance(item, dict):
            continue
        selected = item.get("selected")
        if isinstance(selected, dict):
            out[str(profile)] = selected
    if out:
        return out
    per_profile = payload.get("per_profile") if isinstance(payload.get("per_profile"), dict) else {}
    if per_profile:
        return per_profile
    profiles = payload.get("profiles") if isinstance(payload.get("profiles"), dict) else {}
    if profiles:
        return profiles
    return _mixed_gate_metrics_from_records(records_path)


def _metric_delta(left: Any, right: Any) -> float | None:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) - float(right)
    return None


def _metric_value(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    return None


def compare_sida_to_mixed_gate(sida_metrics: dict[str, Any], mixed_gate_metrics: dict[str, Any]) -> dict[str, Any]:
    profiles = sorted(profile for profile in STRICT_COMPARISON_PROFILES if profile in sida_metrics or profile in mixed_gate_metrics)
    out: dict[str, Any] = {}
    for profile in profiles:
        sida = sida_metrics.get(profile, {}) if isinstance(sida_metrics.get(profile), dict) else {}
        gate = mixed_gate_metrics.get(profile, {}) if isinstance(mixed_gate_metrics.get(profile), dict) else {}
        sida_tampered_recall = _metric_value(sida, "tampered_recall")
        sida_synthetic_recall = _metric_value(sida, "synthetic_recall")
        sida_real_fpr = _metric_value(sida, "real_fpr")
        sida_valid_iou = _metric_value(sida, "tampered_valid_mean_iou", "valid_iou")
        gate_tampered_recall = _metric_value(gate, "tampered_recall")
        gate_synthetic_recall = _metric_value(gate, "synthetic_recall")
        gate_real_fpr = _metric_value(gate, "real_fpr")
        gate_valid_iou = _metric_value(gate, "tampered_valid_mean_iou", "valid_iou")
        out[profile] = {
            "sida_tampered_recall": sida_tampered_recall,
            "sida_synthetic_recall": sida_synthetic_recall,
            "sida_real_fpr": sida_real_fpr,
            "sida_valid_iou": sida_valid_iou,
            "mixed_gate_tampered_recall": gate_tampered_recall,
            "mixed_gate_synthetic_recall": gate_synthetic_recall,
            "mixed_gate_real_fpr": gate_real_fpr,
            "mixed_gate_valid_iou": gate_valid_iou,
            "delta_sida_minus_mixed_gate": {
                "tampered_recall": _metric_delta(sida_tampered_recall, gate_tampered_recall),
                "synthetic_recall": _metric_delta(sida_synthetic_recall, gate_synthetic_recall),
                "real_fpr": _metric_delta(sida_real_fpr, gate_real_fpr),
                "valid_iou": _metric_delta(sida_valid_iou, gate_valid_iou),
            },
            "tampered_recall_delta_sida_minus_gate": _metric_delta(sida_tampered_recall, gate_tampered_recall),
            "valid_iou_delta_sida_minus_gate": _metric_delta(sida_valid_iou, gate_valid_iou),
        }
    clean_metrics = sida_metrics.get("clean", {}) if isinstance(sida_metrics.get("clean"), dict) else {}
    clean_reference = {
        "profile": "clean",
        "reference_only": True,
        "sida_metrics": clean_metrics,
        "mixed_gate_available": "clean" in mixed_gate_metrics,
        "reason": "SIDA subset includes clean reference rows; final mixed_feature_gate contains SNSAug perturbation profiles only.",
    }
    missing_strict = [profile for profile in STRICT_COMPARISON_PROFILES if profile not in mixed_gate_metrics]
    missing_reference_only = ["clean"] if "clean" in sida_metrics and "clean" not in mixed_gate_metrics else []
    return {
        "marker": MARKER,
        "strict_profile_set": list(STRICT_COMPARISON_PROFILES),
        "profiles": out,
        "clean_reference": clean_reference,
        "reference_only_profiles": ["clean"] if "clean" in sida_metrics else [],
        "missing_reference_only_profiles": missing_reference_only,
        "missing_mixed_gate_profiles": missing_strict,
        "strict_comparison_available": not missing_strict,
        "clean_safe": True,
    }


def decide_sida(type_metrics: dict[str, Any], cached_available: bool) -> str:
    if not cached_available:
        return "sida_cached_outputs_missing"
    a_ok = (type_metrics.get("type_a_local_overlay", {}).get("tampered_recall") or 0.0) >= 0.5
    b_ok = (type_metrics.get("type_b_global_geometry_degradation", {}).get("tampered_recall") or 0.0) >= 0.5
    if a_ok and b_ok:
        return "sida_handles_both_types"
    if not a_ok and b_ok:
        return "sida_fails_local_overlay_only"
    if a_ok and not b_ok:
        return "sida_fails_global_degradation_only"
    return "sida_fails_both_types"


def decide_sida_findings(type_metrics: dict[str, Any], comparison: dict[str, Any], cached_available: bool) -> list[str]:
    if not cached_available:
        return ["sida_cached_outputs_missing"]
    findings: list[str] = []
    all_groups = [item for item in type_metrics.values() if isinstance(item, dict) and item.get("row_count")]
    if all_groups and all((item.get("synthetic_recall") or 0.0) == 0.0 for item in all_groups):
        findings.append("sida_classification_collapse_synthetic")
    type_a = type_metrics.get("type_a_local_overlay", {})
    type_b = type_metrics.get("type_b_global_geometry_degradation", {})
    a_t = type_a.get("tampered_recall")
    b_t = type_b.get("tampered_recall")
    if isinstance(a_t, (int, float)) and isinstance(b_t, (int, float)) and a_t < b_t:
        findings.append("sida_local_overlay_worse_than_global")
    if isinstance(b_t, (int, float)) and b_t < 0.5:
        findings.append("sida_global_degradation_partially_failed")
    if any((item.get("mask_missing_rate") or 0.0) >= 1.0 for item in all_groups):
        findings.append("sida_localization_unavailable_mask_missing")
    missing = comparison.get("missing_mixed_gate_profiles", []) if isinstance(comparison, dict) else []
    clean_missing = comparison.get("missing_reference_only_profiles", []) if isinstance(comparison, dict) else []
    findings.append("sida_vs_mixed_gate_incomplete" if missing else "sida_vs_mixed_gate_strict_comparison_available")
    if clean_missing == ["clean"]:
        findings.append("clean_missing_only_expected_for_mixed_gate")
    return findings or ["sida_handles_both_types"]


def _paths(output_root: Path) -> dict[str, str]:
    return {
        "sida_eval_subset_manifest": str(output_root / "sida_eval_subset_manifest.jsonl"),
        "sida_prompt_list": str(output_root / "sida_prompt_list.jsonl"),
        "run_sida_external_template": str(output_root / "run_sida_external_template.sh"),
        "sida_external_output_schema": str(output_root / "sida_external_output_schema.md"),
        "sida7b_diagnostic_records": str(output_root / "sida7b_diagnostic_records.jsonl"),
        "sida7b_per_profile_metrics": str(output_root / "sida7b_per_profile_metrics.json"),
        "sida7b_type_a_type_b_summary": str(output_root / "sida7b_type_a_type_b_summary.json"),
        "sida7b_vs_mixed_gate_comparison": str(output_root / "sida7b_vs_mixed_gate_comparison.json"),
        "sida7b_diagnostic_report": str(output_root / "sida7b_diagnostic_report.md"),
        "sida7b_raw_output_audit": str(output_root / "sida7b_raw_output_audit.json"),
        "sida7b_corrected_type_a_type_b_summary": str(output_root / "sida7b_corrected_type_a_type_b_summary.json"),
        "sida7b_corrected_vs_mixed_gate_comparison": str(output_root / "sida7b_corrected_vs_mixed_gate_comparison.json"),
        "sida7b_clean_safe_vs_mixed_gate_comparison": str(output_root / "sida7b_clean_safe_vs_mixed_gate_comparison.json"),
        "sida7b_clean_safe_vs_mixed_gate_comparison_report": str(output_root / "sida7b_clean_safe_vs_mixed_gate_comparison.md"),
        "sida7b_corrected_diagnostic_report": str(output_root / "sida7b_corrected_diagnostic_report.md"),
        "artifact_manifest": str(output_root / "artifact_manifest.json"),
    }


def _schema_doc() -> str:
    return """# Cached SIDA Output JSONL Schema

Each row should contain `image_id`, `image_path`, `base_id`, `profile`, `profile_family`, `content_label`, `sida_text_output`, `sida_pred_class`, optional `sida_p_real`, `sida_p_synthetic`, `sida_p_tampered`, optional `sida_mask_path`, optional `parse_error`, and optional `mask_missing`.
"""


def _external_template(paths: dict[str, str]) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
# Run this outside cv-forensics in an environment where SIDA-7B is already installed.
# No network/download step is performed by this template.
SIDA_PROMPTS=\"{paths['sida_prompt_list']}\"
SIDA_MANIFEST=\"{paths['sida_eval_subset_manifest']}\"
SIDA_OUTPUT_JSONL=\"/path/to/cached_sida_outputs.jsonl\"
echo \"Use $SIDA_PROMPTS and $SIDA_MANIFEST with your external SIDA runner.\"
echo \"Write cached outputs to $SIDA_OUTPUT_JSONL using sida_external_output_schema.md.\"
"""


def _report(decision: str, mode: str, manifest_count: int, records_count: int, type_metrics: dict[str, Any]) -> str:
    lines = [
        "# SIDA-7B SNSAug Diagnostic Baseline",
        "",
        MARKER,
        "",
        f"- Mode: `{mode}`",
        f"- Exported subset rows: `{manifest_count}`",
        f"- Cached SIDA records evaluated: `{records_count}`",
        f"- Decision: `{decision}`",
        "",
    ]
    if decision == "sida_cached_outputs_missing":
        lines.append("SIDA was not run yet. This is an export-only result; no performance conclusion is allowed.")
    else:
        lines.append("Cached SIDA outputs were evaluated, so performance conclusions are allowed for this cached run.")
        lines += [
            "",
            "| Failure Type | Accuracy | Tampered Recall | Valid IoU | Parse Error Rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for key in ("type_a_local_overlay", "type_b_global_geometry_degradation", "clean"):
            item = type_metrics.get(key, {})
            lines.append(f"| {key} | {item.get('accuracy')} | {item.get('tampered_recall')} | {item.get('tampered_valid_mean_iou')} | {item.get('parse_error_rate')} |")
    return "\n".join(lines) + "\n"


def _clean_safe_comparison_report(comparison: dict[str, Any]) -> str:
    missing_strict = comparison.get("missing_mixed_gate_profiles", []) if isinstance(comparison, dict) else []
    missing_reference = comparison.get("missing_reference_only_profiles", []) if isinstance(comparison, dict) else []
    clean_reference = comparison.get("clean_reference", {}) if isinstance(comparison.get("clean_reference"), dict) else {}
    lines = [
        "# Clean-Safe SIDA vs Mixed Gate Comparison",
        "",
        MARKER,
        "",
        f"- Strict comparison available: `{not bool(missing_strict)}`",
        f"- Missing strict Type A/B mixed-gate profiles: `{missing_strict}`",
        f"- Missing reference-only profiles: `{missing_reference}`",
        f"- Clean reference-only: `{clean_reference.get('reference_only', False)}`",
        "",
        "Strict comparison is limited to the 12 Type A/B SNSAug perturbation profiles. Clean rows are retained as SIDA reference context and are not fatal for mixed_feature_gate comparison availability.",
        "",
        "| Profile | SIDA Tampered Recall | Gate Tampered Recall | SIDA Valid IoU | Gate Valid IoU |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for profile, row in sorted((comparison.get("profiles") or {}).items()):
        if isinstance(row, dict):
            lines.append(f"| {profile} | {row.get('sida_tampered_recall')} | {row.get('mixed_gate_tampered_recall')} | {row.get('sida_valid_iou')} | {row.get('mixed_gate_valid_iou')} |")
    return "\n".join(lines) + "\n"


def _corrected_report(findings: list[str], mode: str, manifest_count: int, records_count: int, type_metrics: dict[str, Any], raw_audit: dict[str, Any], comparison: dict[str, Any]) -> str:
    lines = [
        "# Corrected SIDA-7B SNSAug Diagnostic Report",
        "",
        MARKER,
        "",
        f"- Mode: `{mode}`",
        f"- Exported subset rows: `{manifest_count}`",
        f"- Cached SIDA records evaluated: `{records_count}`",
        f"- Findings: `{', '.join(findings) if findings else 'none'}`",
        f"- Raw [CLS] count: `{raw_audit.get('raw_contains_cls_count', 0)}`",
        f"- Raw [SEG] count: `{raw_audit.get('raw_contains_seg_count', 0)}`",
        "",
    ]
    if not records_count:
        lines.append("SIDA was not run yet. This is an export-only result; no performance conclusion is allowed.")
        return "\n".join(lines) + "\n"
    lines.append("Cached SIDA outputs were evaluated as a classification diagnostic.")
    if "sida_localization_unavailable_mask_missing" in findings:
        lines.append("Localization is unavailable for this cached run because masks are missing; this report does not claim localization failure.")
    lines += [
        "",
        "| Failure Type | Accuracy | Synthetic Recall | Tampered Recall | Valid IoU | Mask Missing Rate | Raw SEG Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("type_a_local_overlay", "type_b_global_geometry_degradation", "clean"):
        item = type_metrics.get(key, {})
        lines.append(f"| {key} | {item.get('accuracy')} | {item.get('synthetic_recall')} | {item.get('tampered_recall')} | {item.get('tampered_valid_mean_iou')} | {item.get('mask_missing_rate')} | {item.get('raw_seg_rate')} |")
    missing = comparison.get("missing_mixed_gate_profiles", []) if isinstance(comparison, dict) else []
    lines += [
        "",
        f"- Mixed-gate comparison status: `{'incomplete' if missing else 'available'}`",
        f"- Missing mixed-gate profiles: `{missing}`",
    ]
    return "\n".join(lines) + "\n"


def run_snsaug_v2_sida7b_diagnostic_baseline(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    assert_valid_config(config, require_exists=not dry_run)
    output_root = _real(config["output_root"])
    paths = _paths(output_root)
    plan = {
        "marker": MARKER,
        "dry_run": dry_run,
        "mode": config.get("mode"),
        "output_root": str(output_root),
        "output_paths": paths,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    if dry_run:
        return plan
    output_root.mkdir(parents=True, exist_ok=True)
    parsed_rows, warnings = load_subset_rows(config)
    subset = build_balanced_subset(parsed_rows, int(config.get("max_per_label_profile", 2)))
    manifest, prompts = export_subset_rows(subset)
    _write_jsonl(Path(paths["sida_eval_subset_manifest"]), manifest)
    _write_jsonl(Path(paths["sida_prompt_list"]), prompts)
    _write_text(Path(paths["run_sida_external_template"]), _external_template(paths))
    _write_text(Path(paths["sida_external_output_schema"]), _schema_doc())
    cached_available = config.get("mode") == CACHED_MODE and bool(config.get("cached_sida_outputs_path"))
    records: list[dict[str, Any]] = []
    per_profile: dict[str, Any] = {}
    type_metrics: dict[str, Any] = {}
    raw_audit: dict[str, Any] = raw_output_audit([])
    comparison: dict[str, Any] = {"marker": MARKER, "profiles": {}, "missing_mixed_gate_profiles": []}
    if cached_available:
        cached_rows = _load_jsonl(config["cached_sida_outputs_path"])
        raw_audit = raw_output_audit(cached_rows)
        records = parse_cached_sida_outputs(cached_rows, manifest)
        per_profile = per_profile_metrics(records)
        type_metrics = type_summary(records)
        comparison = compare_sida_to_mixed_gate(per_profile, load_mixed_gate_metrics(config.get("final_policy_gate_metrics_path"), config.get("final_policy_gate_records_path")))
    decision = decide_sida(type_metrics, cached_available)
    decision_findings = decide_sida_findings(type_metrics, comparison, cached_available)
    _write_jsonl(Path(paths["sida7b_diagnostic_records"]), records)
    _write_json(Path(paths["sida7b_per_profile_metrics"]), {"marker": MARKER, "metrics": per_profile, "clean_to_sns_drops": clean_to_sns_drops(per_profile)})
    _write_json(Path(paths["sida7b_type_a_type_b_summary"]), {"marker": MARKER, "summary": type_metrics, "decision": decision})
    _write_json(Path(paths["sida7b_vs_mixed_gate_comparison"]), comparison)
    _write_text(Path(paths["sida7b_diagnostic_report"]), _report(decision, str(config.get("mode")), len(manifest), len(records), type_metrics))
    _write_json(Path(paths["sida7b_raw_output_audit"]), {"marker": MARKER, **raw_audit})
    _write_json(Path(paths["sida7b_corrected_type_a_type_b_summary"]), {"marker": MARKER, "summary": type_metrics, "decision_findings": decision_findings})
    _write_json(Path(paths["sida7b_corrected_vs_mixed_gate_comparison"]), comparison)
    _write_json(Path(paths["sida7b_clean_safe_vs_mixed_gate_comparison"]), comparison)
    _write_text(Path(paths["sida7b_clean_safe_vs_mixed_gate_comparison_report"]), _clean_safe_comparison_report(comparison))
    _write_text(Path(paths["sida7b_corrected_diagnostic_report"]), _corrected_report(decision_findings, str(config.get("mode")), len(manifest), len(records), type_metrics, raw_audit, comparison))
    artifact = {
        **plan,
        "dry_run": False,
        "analysis_started": True,
        "pair_root": str(_real(config["pair_root"])),
        "meta_jsonl_path": str(_real(config["meta_jsonl_path"])),
        "manifest_count": len(manifest),
        "record_count": len(records),
        "warning_count": len(warnings),
        "warnings": warnings,
        "decision": decision,
        "decision_findings": decision_findings,
        "raw_output_audit": raw_audit,
        "cached_sida_outputs_path": str(_real(config["cached_sida_outputs_path"])) if config.get("cached_sida_outputs_path") else None,
    }
    _write_json(Path(paths["artifact_manifest"]), artifact)
    return artifact


__all__ = [
    "APPROVAL_TEXT",
    "CACHED_MODE",
    "CONFIG_OK_MARKER",
    "EXPORT_MODE",
    "MARKER",
    "SNSAugV2SIDA7BDiagnosticBaselineError",
    "build_balanced_subset",
    "compare_sida_to_mixed_gate",
    "load_mixed_gate_metrics",
    "mask_valid_iou",
    "normalize_class",
    "parse_cached_sida_outputs",
    "parse_sida_text_output",
    "parse_sida_text_output_detail",
    "raw_output_audit",
    "run_snsaug_v2_sida7b_diagnostic_baseline",
    "validate_snsaug_v2_sida7b_diagnostic_baseline_config",
]
