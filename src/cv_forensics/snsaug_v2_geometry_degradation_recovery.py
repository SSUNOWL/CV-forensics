"""Geometry-normalized and degradation-aware SNSAug inference recovery."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .pre_sns_v3_sns_robustness_eval import (
    _mean,
    _safe_prob,
    confusion_matrix,
    load_best_bundle,
    macro_f1_and_details,
    normalize_label,
    policy_config_from_bundle,
)
from .pre_sns_v3_v2_policy_gated_report import build_policy_gated_record
from .snsaug_v2_fixed_pairs_eval import compute_mask_metrics, load_meta_rows, parse_fixed_pair_rows
from .snsaug_v2_masked_nuisance_inference import resolve_pair_mask_path

MARKER = "SNSAUG_V2_GEOMETRY_DEGRADATION_RECOVERY_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_GEOMETRY_DEGRADATION_RECOVERY_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_geometry_degradation_recovery"
APPROVED_MODE = "approved_local_snsaug_v2_geometry_degradation_recovery"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_GEOMETRY_DEGRADATION_RECOVERY"
REPO_ROOT = Path(__file__).resolve().parents[2]
CLASS_LABELS = ("real", "synthetic", "tampered")
RECOVERY_POLICIES = (
    "original",
    "content_box_crop_resize",
    "geometry_normalized",
    "deblock_mild",
    "resize_to_clean_proxy",
    "geometry_plus_deblock",
    "oracle_clean_geometry_diagnostic",
)
FOCUS_PROFILES = {
    "canvas_9x16_only",
    "resize_crop_pad",
    "zoom_crop",
    "resize_jpeg",
    "screenshot_recapture_light",
    "combined_sns_realistic",
}
OUTPUT_NAMES = (
    "geometry_degradation_recovery_records.jsonl",
    "per_policy_per_profile_metrics.json",
    "recovery_delta_summary.json",
    "geometry_degradation_recovery_report.md",
    "visual_gallery_manifest.json",
    "artifact_manifest.json",
)


class SNSAugV2GeometryDegradationRecoveryError(ValueError):
    """Raised when geometry/degradation recovery validation or execution fails."""


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


def load_snsaug_v2_geometry_degradation_recovery_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2GeometryDegradationRecoveryError("geometry degradation recovery config root must be a JSON object")
    raw.setdefault("config_path", str(_real(path)))
    return raw


def validate_snsaug_v2_geometry_degradation_recovery_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
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
        "policies",
        "profiles",
        "device",
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
    input_roots = _as_roots(raw.get("approved_input_roots"))
    output_roots = _as_roots(raw.get("approved_output_roots"))
    if not input_roots:
        errors.append("approved_input_roots must be a non-empty list of absolute paths")
    if not output_roots:
        errors.append("approved_output_roots must be a non-empty list of absolute paths")
    for index, root in enumerate(input_roots):
        errors.extend(_validate_abs_path(root, f"approved_input_roots[{index}]"))
    for index, root in enumerate(output_roots):
        errors.extend(_validate_abs_path(root, f"approved_output_roots[{index}]"))
    for field in ("best_bundle_path", "pair_root", "meta_jsonl_path"):
        errors.extend(_validate_under_roots(raw.get(field), field, input_roots, require_exists=require_exists))
    output_root = raw.get("output_root")
    errors.extend(_validate_abs_path(output_root, "output_root"))
    if isinstance(output_root, str) and output_root.strip():
        if _inside_repo(output_root):
            errors.append("output_root must be outside repository")
        if any(_is_under(output_root, root) or _real(output_root) == _real(root) for root in input_roots):
            errors.append("output_root must not be under approved input roots")
        if output_roots and not any(_is_under(output_root, root) or _real(output_root) == _real(root) for root in output_roots):
            errors.append("output_root must be under approved output roots")
    policies = raw.get("policies")
    if not isinstance(policies, list) or not policies or not all(isinstance(item, str) for item in policies):
        errors.append("policies must be a non-empty list of strings")
    else:
        invalid = [item for item in policies if item not in RECOVERY_POLICIES]
        if invalid:
            errors.append(f"unsupported policies: {invalid}")
        if "original" not in policies:
            errors.append("policies must include original")
    profiles = raw.get("profiles")
    if not isinstance(profiles, list) or not profiles or not all(isinstance(item, str) for item in profiles):
        errors.append("profiles must be a non-empty list of strings")
    elif "clean" not in profiles:
        errors.append("profiles must include clean")
    if raw.get("device") not in {"cpu", "cuda"}:
        errors.append("device must be cpu or cuda")
    for field in ("max_samples", "top_n_gallery", "content_background_tolerance"):
        if field in raw and raw.get(field) is not None:
            value = raw.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
                errors.append(f"{field} must be positive when provided")
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_geometry_degradation_recovery_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2GeometryDegradationRecoveryError("geometry degradation recovery config validation failed:\n" + "\n".join(errors))


def _runtime_deps():
    try:
        from PIL import Image, ImageChops, ImageFilter, ImageStat
    except Exception as exc:
        raise SNSAugV2GeometryDegradationRecoveryError("PIL is required for geometry degradation recovery") from exc
    return Image, ImageChops, ImageFilter, ImageStat


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    return str(path)


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _safe_stem(row: dict[str, Any], policy: str, index: int) -> str:
    raw = "__".join([f"{index:06d}", str(row.get("base_id") or "base"), str(row.get("profile") or "profile"), str(row.get("view") or "view"), policy])
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)


def _resample_bilinear(Image: Any) -> Any:
    return getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def estimate_content_box(image: Any, *, tolerance: int = 10) -> tuple[int, int, int, int]:
    """Estimate the main non-border content rectangle using border background color."""

    Image, _ImageChops, _ImageFilter, _ImageStat = _runtime_deps()
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width <= 2 or height <= 2:
        return (0, 0, width, height)
    corners = [rgb.getpixel((0, 0)), rgb.getpixel((width - 1, 0)), rgb.getpixel((0, height - 1)), rgb.getpixel((width - 1, height - 1))]
    bg = tuple(int(round(sum(pixel[channel] for pixel in corners) / len(corners))) for channel in range(3))
    pixels = rgb.load()
    xs: list[int] = []
    ys: list[int] = []
    for y in range(height):
        for x in range(width):
            pix = pixels[x, y]
            if sum(abs(int(pix[channel]) - bg[channel]) for channel in range(3)) > int(tolerance):
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return (0, 0, width, height)
    left, right = max(0, min(xs)), min(width, max(xs) + 1)
    top, bottom = max(0, min(ys)), min(height, max(ys) + 1)
    if right <= left or bottom <= top:
        return (0, 0, width, height)
    return (left, top, right, bottom)


def content_box_crop_resize(image: Any, target_size: tuple[int, int] | None = None, *, tolerance: int = 10) -> Any:
    Image, _ImageChops, _ImageFilter, _ImageStat = _runtime_deps()
    rgb = image.convert("RGB")
    target = target_size or rgb.size
    box = estimate_content_box(rgb, tolerance=tolerance)
    return rgb.crop(box).resize(target, _resample_bilinear(Image))


def deblock_mild(image: Any) -> Any:
    _Image, _ImageChops, ImageFilter, _ImageStat = _runtime_deps()
    return image.convert("RGB").filter(ImageFilter.MedianFilter(size=3))


def resize_to_clean_proxy(image: Any, clean_image: Any | None) -> Any:
    Image, _ImageChops, _ImageFilter, _ImageStat = _runtime_deps()
    if clean_image is None:
        return image.convert("RGB").copy()
    return image.convert("RGB").resize(clean_image.convert("RGB").size, _resample_bilinear(Image))


def _fixture_for_policy(row: dict[str, Any], policy: str) -> dict[str, Any]:
    sample = dict(row)
    fixtures = row.get("fixture_by_policy")
    if isinstance(fixtures, dict) and isinstance(fixtures.get(policy), dict):
        sample.update(fixtures[policy])
    return sample


def _group_clean_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("base_id")): row for row in rows if row.get("view") == "clean"}


def apply_recovery_policy(
    image: Any,
    *,
    policy: str,
    clean_image: Any | None = None,
    tolerance: int = 10,
) -> tuple[Any, dict[str, Any]]:
    rgb = image.convert("RGB")
    clean_rgb = clean_image.convert("RGB") if clean_image is not None else None
    diagnostic = policy == "oracle_clean_geometry_diagnostic"
    if policy == "original":
        out = rgb.copy()
    elif policy == "content_box_crop_resize":
        out = content_box_crop_resize(rgb, target_size=rgb.size, tolerance=tolerance)
    elif policy == "geometry_normalized":
        out = content_box_crop_resize(rgb, target_size=clean_rgb.size if clean_rgb is not None else rgb.size, tolerance=tolerance)
    elif policy == "deblock_mild":
        out = deblock_mild(rgb)
    elif policy == "resize_to_clean_proxy":
        out = resize_to_clean_proxy(rgb, clean_rgb)
    elif policy == "geometry_plus_deblock":
        out = deblock_mild(content_box_crop_resize(rgb, target_size=clean_rgb.size if clean_rgb is not None else rgb.size, tolerance=tolerance))
    elif policy == "oracle_clean_geometry_diagnostic":
        out = clean_rgb.copy() if clean_rgb is not None else rgb.copy()
    else:
        raise SNSAugV2GeometryDegradationRecoveryError(f"unsupported policy: {policy}")
    return out, {"diagnostic_oracle": diagnostic, "output_width": int(out.size[0]), "output_height": int(out.size[1])}


def build_policy_input(
    row: dict[str, Any],
    *,
    clean_row: dict[str, Any] | None,
    output_root: str | Path,
    policy: str,
    index: int,
    tolerance: int = 10,
) -> dict[str, Any]:
    Image, _ImageChops, _ImageFilter, _ImageStat = _runtime_deps()
    image_path = str(row.get("image_path") or "")
    if not image_path:
        raise SNSAugV2GeometryDegradationRecoveryError("row missing image_path")
    with Image.open(image_path) as image:
        base = image.convert("RGB")
    clean_image = None
    if clean_row and clean_row.get("image_path"):
        with Image.open(str(clean_row["image_path"])) as image:
            clean_image = image.convert("RGB")
    transformed, info = apply_recovery_policy(base, policy=policy, clean_image=clean_image, tolerance=tolerance)
    if policy == "original":
        transformed_path = image_path
    else:
        path = _real(output_root) / "policy_inputs" / policy / f"{_safe_stem(row, policy, index)}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        transformed.save(path)
        transformed_path = str(path)
    return {"image_path": transformed_path, **info}


def _load_binary_mask_list(Image: Any, path: str | None, size: tuple[int, int]) -> list[int] | None:
    if not path:
        return None
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    with Image.open(path) as image:
        mask = image.convert("L").resize(size, nearest)
        return [1 if int(value) > 0 else 0 for value in mask.getdata()]


def _evaluate_one(
    bundle: dict[str, Any],
    config: dict[str, Any],
    row: dict[str, Any],
    clean_row: dict[str, Any] | None,
    policy: str,
    index: int,
) -> dict[str, Any]:
    Image, _ImageChops, _ImageFilter, _ImageStat = _runtime_deps()
    transformed = build_policy_input(
        row,
        clean_row=clean_row,
        output_root=config["output_root"],
        policy=policy,
        index=index,
        tolerance=int(config.get("content_background_tolerance", 10)),
    )
    sample = _fixture_for_policy(row, policy)
    sample["image_path"] = transformed["image_path"]
    tamper_mask_path = resolve_pair_mask_path(row, config["pair_root"], "tamper_mask_path")
    ignore_mask_path = resolve_pair_mask_path(row, config["pair_root"], "ignore_mask_path")
    if tamper_mask_path:
        sample["gt_mask_path"] = tamper_mask_path
    policy_config = policy_config_from_bundle(bundle, config, _real(config["output_root"]))
    started = time.perf_counter()
    result = build_policy_gated_record(policy_config, sample, index)
    latency_ms = (time.perf_counter() - started) * 1000.0
    width, height = int(transformed["output_width"]), int(transformed["output_height"])
    pred_mask = result.get("_final_mask")
    gt_mask = _load_binary_mask_list(Image, tamper_mask_path, (width, height)) if tamper_mask_path else None
    ignore_mask = _load_binary_mask_list(Image, ignore_mask_path, (width, height)) if ignore_mask_path else None
    mask_metrics = compute_mask_metrics(pred_mask, gt_mask, ignore_mask)
    pred_class = normalize_label(result.get("class"))
    label = normalize_label(row.get("content_label"))
    return {
        "marker": MARKER,
        "base_id": str(row.get("base_id") or f"row_{index:06d}"),
        "content_label": label,
        "view": str(row.get("view") or ""),
        "profile": str(row.get("profile") or ""),
        "policy": policy,
        "diagnostic_oracle": bool(transformed["diagnostic_oracle"]),
        "image_path": str(row.get("image_path")),
        "policy_image_path": transformed["image_path"],
        "tamper_mask_path": tamper_mask_path,
        "ignore_mask_path": ignore_mask_path,
        "pred_class": pred_class,
        "class_correct": pred_class == label,
        "p_real": _safe_prob(result.get("class_conf", {}), "real"),
        "p_synthetic": _safe_prob(result.get("class_conf", {}), "synthetic"),
        "p_tampered": _safe_prob(result.get("class_conf", {}), "tampered"),
        "tampered_score": float(result.get("tampered_score", 0.0)),
        "localization_activated": bool(result.get("tile_localization_activated")),
        "final_mask_area_pct": float(result.get("final_mask_area_pct", 0.0)),
        "raw_iou": mask_metrics["raw_iou"],
        "raw_dice": mask_metrics["raw_dice"],
        "valid_iou": mask_metrics["valid_iou"],
        "valid_dice": mask_metrics["valid_dice"],
        "latency_ms": float(latency_ms),
        "error": None,
    }


def evaluate_recovery_rows(bundle: dict[str, Any], config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    policies = list(config.get("policies") or RECOVERY_POLICIES)
    clean_by_base = _group_clean_rows(rows)
    for row_index, row in enumerate(rows):
        clean_row = clean_by_base.get(str(row.get("base_id")))
        for policy_index, policy in enumerate(policies):
            index = row_index * len(policies) + policy_index
            try:
                records.append(_evaluate_one(bundle, config, row, clean_row, str(policy), index))
            except Exception as exc:
                records.append(
                    {
                        "marker": MARKER,
                        "base_id": str(row.get("base_id") or f"row_{row_index:06d}"),
                        "content_label": normalize_label(row.get("content_label")),
                        "view": str(row.get("view") or ""),
                        "profile": str(row.get("profile") or ""),
                        "policy": str(policy),
                        "diagnostic_oracle": str(policy) == "oracle_clean_geometry_diagnostic",
                        "image_path": str(row.get("image_path")),
                        "policy_image_path": None,
                        "tamper_mask_path": resolve_pair_mask_path(row, config["pair_root"], "tamper_mask_path"),
                        "ignore_mask_path": resolve_pair_mask_path(row, config["pair_root"], "ignore_mask_path"),
                        "pred_class": None,
                        "class_correct": False,
                        "p_real": 0.0,
                        "p_synthetic": 0.0,
                        "p_tampered": 0.0,
                        "tampered_score": 0.0,
                        "localization_activated": False,
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


def aggregate_per_policy_profile(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in records:
        grouped.setdefault(str(row["policy"]), {}).setdefault(str(row["profile"]), []).append(row)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for policy, by_profile in grouped.items():
        out[policy] = {}
        for profile, items in by_profile.items():
            matrix = confusion_matrix([{"label": row["content_label"], "pred_class": row.get("pred_class")} for row in items if row.get("pred_class")])
            cls = macro_f1_and_details(matrix)
            real = [row for row in items if row["content_label"] == "real"]
            synthetic = [row for row in items if row["content_label"] == "synthetic"]
            tampered = [row for row in items if row["content_label"] == "tampered"]
            non_tampered = [row for row in items if row["content_label"] != "tampered"]
            valid_ious = [float(row["valid_iou"]) for row in tampered if row.get("valid_iou") is not None]
            tampered_probs = [float(row["p_tampered"]) for row in tampered if row.get("p_tampered") is not None]
            out[policy][profile] = {
                "sample_count": len(items),
                "accuracy": sum(1 for row in items if row.get("class_correct")) / len(items) if items else 0.0,
                "macro_f1": cls["macro_f1"],
                "confusion_matrix": matrix,
                "real_fpr": (sum(1 for row in real if row.get("pred_class") != "real") / len(real)) if real else None,
                "synthetic_recall": cls["per_class"]["synthetic"]["recall"] if synthetic else None,
                "tampered_recall": cls["per_class"]["tampered"]["recall"] if tampered else None,
                "localization_activation_recall": (sum(1 for row in tampered if row.get("localization_activated")) / len(tampered)) if tampered else None,
                "tampered_valid_mean_iou": _mean(valid_ious),
                "mean_p_tampered_on_tampered": _mean(tampered_probs),
                "non_tampered_high_mask_rate": (
                    sum(1 for row in non_tampered if float(row.get("final_mask_area_pct") or 0.0) > 1.0) / len(non_tampered)
                ) if non_tampered else None,
            }
    return out


def recovery_delta_summary(per_policy_profile: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    original = per_policy_profile.get("original", {})
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for policy, by_profile in per_policy_profile.items():
        if policy == "original":
            continue
        out[policy] = {}
        for profile, metrics in by_profile.items():
            base = original.get(profile, {})

            def delta(key: str) -> float | None:
                left = metrics.get(key)
                right = base.get(key)
                if left is None or right is None:
                    return None
                return float(left) - float(right)

            out[policy][profile] = {
                "tampered_recall_recovery": delta("tampered_recall"),
                "valid_iou_recovery": delta("tampered_valid_mean_iou"),
                "p_tampered_recovery_on_tampered": delta("mean_p_tampered_on_tampered"),
                "localization_activation_recall_recovery": delta("localization_activation_recall"),
                "synthetic_recall_change": delta("synthetic_recall"),
                "real_fpr_change": delta("real_fpr"),
            }
    return out


def promising_policies(recovery: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for policy, by_profile in recovery.items():
        focus_profiles = [profile for profile in by_profile if profile in FOCUS_PROFILES]
        recall_profiles = [
            profile for profile in focus_profiles
            if float(by_profile[profile].get("tampered_recall_recovery") or 0.0) >= 0.10
        ]
        iou_profiles = [
            profile for profile in focus_profiles
            if float(by_profile[profile].get("valid_iou_recovery") or 0.0) >= 0.05
        ]
        synthetic_ok = all(float(by_profile[profile].get("synthetic_recall_change") or 0.0) >= -0.05 for profile in focus_profiles)
        real_ok = all(float(by_profile[profile].get("real_fpr_change") or 0.0) <= 0.20 for profile in focus_profiles)
        if (len(recall_profiles) >= 2 or len(iou_profiles) >= 2) and synthetic_ok and real_ok:
            out.append(
                {
                    "policy": policy,
                    "tampered_recall_recovered_profiles": recall_profiles,
                    "valid_iou_recovered_profiles": iou_profiles,
                    "synthetic_recall_not_collapsed": synthetic_ok,
                    "real_fpr_not_severely_increased": real_ok,
                }
            )
    return out


def _counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        value = str(row.get(key) or "")
        if value:
            counts[value] = counts.get(value, 0) + 1
    return counts


def _output_plan(output_root: Path) -> dict[str, str]:
    return {
        "geometry_degradation_recovery_records": str(output_root / "geometry_degradation_recovery_records.jsonl"),
        "per_policy_per_profile_metrics": str(output_root / "per_policy_per_profile_metrics.json"),
        "recovery_delta_summary": str(output_root / "recovery_delta_summary.json"),
        "geometry_degradation_recovery_report": str(output_root / "geometry_degradation_recovery_report.md"),
        "visual_gallery_manifest": str(output_root / "visual_gallery_manifest.json"),
        "artifact_manifest": str(output_root / "artifact_manifest.json"),
    }


def build_plan(config: dict[str, Any]) -> dict[str, Any]:
    output_root = _real(config["output_root"])
    return {
        "marker": MARKER,
        "training_started": False,
        "inference_started": False,
        "record_count": 0,
        "pair_root": str(_real(config["pair_root"])),
        "policies": list(config.get("policies") or []),
        "profiles": list(config.get("profiles") or []),
        "output_paths": _output_plan(output_root),
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def _render_report(summary: dict[str, Any], recovery: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [
        "# SNSAug V2 Geometry Degradation Recovery",
        "",
        MARKER,
        "",
        f"- Records: `{summary['record_count']}`",
        f"- Promising policies: `{len(summary['promising_policies'])}`",
        "",
        "| Policy | Profile | Tampered Recall Recovery | Valid IoU Recovery | p_tampered Recovery | Activation Recovery |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for policy, by_profile in recovery.items():
        for profile, metrics in by_profile.items():
            lines.append(
                f"| {policy} | {profile} | {metrics.get('tampered_recall_recovery')} | {metrics.get('valid_iou_recovery')} | {metrics.get('p_tampered_recovery_on_tampered')} | {metrics.get('localization_activation_recall_recovery')} |"
            )
    return "\n".join(lines) + "\n"


def run_snsaug_v2_geometry_degradation_recovery(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    assert_valid_config(config, require_exists=not dry_run)
    plan = build_plan(config)
    if dry_run:
        return plan
    output_root = _real(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    meta_rows = load_meta_rows(config["meta_jsonl_path"])
    parsed_rows, parse_warnings = parse_fixed_pair_rows(meta_rows, list(config["profiles"]), config.get("max_samples"))
    bundle = load_best_bundle(config["best_bundle_path"])
    started = time.perf_counter()
    records = evaluate_recovery_rows(bundle, config, parsed_rows)
    per_policy_profile = aggregate_per_policy_profile(records)
    recovery = recovery_delta_summary(per_policy_profile)
    promising = promising_policies(recovery)
    summary = {
        **plan,
        "inference_started": True,
        "record_count": len(records),
        "row_count": len(parsed_rows),
        "warning_count": len(parse_warnings),
        "warnings": parse_warnings,
        "elapsed_sec": time.perf_counter() - started,
        "promising_policies": promising,
    }
    paths = _output_plan(output_root)
    gallery = {
        "marker": MARKER,
        "items": [
            {
                "base_id": row["base_id"],
                "profile": row["profile"],
                "policy": row["policy"],
                "policy_image_path": row.get("policy_image_path"),
                "diagnostic_oracle": row.get("diagnostic_oracle"),
            }
            for row in records
            if row.get("policy") != "original"
        ][: int(config.get("top_n_gallery", 12))]
    }
    _write_jsonl(Path(paths["geometry_degradation_recovery_records"]), records)
    _write_json(Path(paths["per_policy_per_profile_metrics"]), per_policy_profile)
    _write_json(Path(paths["recovery_delta_summary"]), {"marker": MARKER, "recovery": recovery, "promising_policies": promising})
    _write_text(Path(paths["geometry_degradation_recovery_report"]), _render_report(summary, recovery))
    _write_json(Path(paths["visual_gallery_manifest"]), gallery)
    artifact = {
        "marker": MARKER,
        "config_path": config.get("config_path"),
        "output_root": str(output_root),
        "pair_root": str(_real(config["pair_root"])),
        "best_bundle_path": str(_real(config["best_bundle_path"])),
        "policies": list(config.get("policies") or []),
        "profiles": list(config.get("profiles") or []),
        "subset_only": config.get("max_samples") is not None,
        "max_rows": config.get("max_samples"),
        "record_count": len(records),
        "row_count": len(parsed_rows),
        "policy_counts": _counts(records, "policy"),
        "profile_counts": _counts(records, "profile"),
        "label_counts": _counts(records, "content_label"),
        "output_paths": paths,
        "warning_count": len(parse_warnings),
        "warnings": parse_warnings,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
        "inference_started": True,
        "training_started": False,
    }
    _write_json(Path(paths["artifact_manifest"]), artifact)
    return {**summary, "output_paths": paths}


__all__ = [
    "APPROVAL_TEXT",
    "CONFIG_OK_MARKER",
    "MARKER",
    "RECOVERY_POLICIES",
    "SNSAugV2GeometryDegradationRecoveryError",
    "aggregate_per_policy_profile",
    "apply_recovery_policy",
    "build_policy_input",
    "content_box_crop_resize",
    "deblock_mild",
    "estimate_content_box",
    "load_snsaug_v2_geometry_degradation_recovery_config",
    "recovery_delta_summary",
    "resize_to_clean_proxy",
    "run_snsaug_v2_geometry_degradation_recovery",
    "validate_snsaug_v2_geometry_degradation_recovery_config",
]
