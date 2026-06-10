"""SNSAug v2 global degradation and geometry shift analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .pre_sns_v3_sns_robustness_eval import normalize_label
from .snsaug_v2_fixed_pairs_eval import load_meta_rows, parse_fixed_pair_rows

MARKER = "SNSAUG_V2_GLOBAL_DEGRADATION_GEOMETRY_ANALYSIS_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_GLOBAL_DEGRADATION_GEOMETRY_ANALYSIS_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_global_degradation_geometry_analysis"
APPROVED_MODE = "approved_local_snsaug_v2_global_degradation_geometry_analysis"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_GLOBAL_DEGRADATION_GEOMETRY_ANALYSIS"
REPO_ROOT = Path(__file__).resolve().parents[2]
CLASS_LABELS = ("real", "synthetic", "tampered")
PROFILE_FAMILIES = {
    "clean": {"clean"},
    "local_overlay": {
        "platform_ui_same_size",
        "news_meme_overlay",
        "tiktok_like",
        "instagram_story_like",
        "youtube_shorts_like",
        "combined_sns_realistic",
    },
    "geometry": {"canvas_9x16_only", "resize_crop_pad", "zoom_crop"},
    "postprocess": {"recompression_light", "resize_jpeg"},
    "screenshot": {"screenshot_recapture_light"},
}
OUTPUT_NAMES = (
    "global_degradation_geometry_records.jsonl",
    "profile_family_shift_summary.json",
    "feature_response_correlation.json",
    "global_degradation_geometry_report.md",
    "artifact_manifest.json",
)
SHIFT_FEATURES = (
    "aspect_ratio_delta_abs",
    "area_ratio_delta_abs",
    "mean_delta_abs",
    "std_delta_abs",
    "histogram_l1",
    "edge_energy_delta_abs",
    "highpass_energy_delta_abs",
    "laplacian_variance_delta_abs",
    "blockiness_delta_abs",
    "crop_scale_proxy",
    "ignore_mask_area",
    "tamper_area",
    "tamper_occluded_by_ignore",
)
RESPONSE_FEATURES = (
    "delta_p_tampered",
    "delta_valid_iou",
    "activation_flip_off",
    "pred_flip",
)


class SNSAugV2GlobalDegradationGeometryAnalysisError(ValueError):
    """Raised when global degradation/geometry analysis validation fails."""


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


def load_snsaug_v2_global_degradation_geometry_analysis_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2GlobalDegradationGeometryAnalysisError("global degradation geometry config root must be a JSON object")
    raw.setdefault("config_path", str(_real(path)))
    return raw


def validate_snsaug_v2_global_degradation_geometry_analysis_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "user_approval_text",
        "pair_root",
        "meta_jsonl_path",
        "approved_input_roots",
        "approved_output_roots",
        "output_root",
        "profiles",
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
    for field in ("pair_root", "meta_jsonl_path"):
        errors.extend(_validate_under_roots(raw.get(field), field, input_roots, require_exists=require_exists))
    if raw.get("baseline_records_path"):
        errors.extend(_validate_under_roots(raw.get("baseline_records_path"), "baseline_records_path", input_roots, require_exists=require_exists))
    output_root = raw.get("output_root")
    errors.extend(_validate_abs_path(output_root, "output_root"))
    if isinstance(output_root, str) and output_root.strip():
        if _inside_repo(output_root):
            errors.append("output_root must be outside repository")
        if any(_is_under(output_root, root) or _real(output_root) == _real(root) for root in input_roots):
            errors.append("output_root must not be under approved input roots")
        if output_roots and not any(_is_under(output_root, root) or _real(output_root) == _real(root) for root in output_roots):
            errors.append("output_root must be under approved output roots")
    profiles = raw.get("profiles")
    if not isinstance(profiles, list) or not profiles or not all(isinstance(item, str) for item in profiles):
        errors.append("profiles must be a non-empty list of strings")
    elif "clean" not in profiles:
        errors.append("profiles must include clean")
    if "max_samples" in raw and raw.get("max_samples") is not None:
        value = raw.get("max_samples")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append("max_samples must be a positive integer when provided")
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_global_degradation_geometry_analysis_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2GlobalDegradationGeometryAnalysisError(
            "global degradation geometry analysis config validation failed:\n" + "\n".join(errors)
        )


def _runtime_deps():
    try:
        from PIL import Image, ImageFilter
    except Exception as exc:
        raise SNSAugV2GlobalDegradationGeometryAnalysisError("PIL is required for global degradation geometry analysis") from exc
    return Image, ImageFilter


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
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


def profile_family(profile: str) -> str:
    for family, profiles in PROFILE_FAMILIES.items():
        if profile in profiles:
            return family
    return "other"


def _load_gray_values(path: str | Path) -> tuple[list[int], int, int]:
    Image, _ImageFilter = _runtime_deps()
    with Image.open(path) as image:
        gray = image.convert("L")
        width, height = gray.size
        return [int(value) for value in gray.getdata()], int(width), int(height)


def _load_mask_values(path: str | Path | None, width: int, height: int) -> list[int] | None:
    if not path:
        return None
    Image, _ImageFilter = _runtime_deps()
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    with Image.open(path) as image:
        mask = image.convert("L").resize((width, height), nearest)
        return [1 if int(value) > 0 else 0 for value in mask.getdata()]


def _mean_std(values: list[int]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((float(value) - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(var)


def _histogram(values: list[int], bins: int = 32) -> list[float]:
    hist = [0] * bins
    for value in values:
        idx = min(bins - 1, max(0, int(value) * bins // 256))
        hist[idx] += 1
    total = float(sum(hist)) or 1.0
    return [count / total for count in hist]


def histogram_l1(left_values: list[int], right_values: list[int]) -> float:
    left = _histogram(left_values)
    right = _histogram(right_values)
    return sum(abs(a - b) for a, b in zip(left, right))


def edge_energy(values: list[int], width: int, height: int) -> float:
    if width <= 1 or height <= 1:
        return 0.0
    total = 0.0
    count = 0
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if x + 1 < width:
                total += abs(values[idx] - values[idx + 1])
                count += 1
            if y + 1 < height:
                total += abs(values[idx] - values[idx + width])
                count += 1
    return total / max(count, 1)


def highpass_residual_energy(values: list[int], width: int, height: int) -> float:
    if width <= 2 or height <= 2:
        return 0.0
    total = 0.0
    count = 0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            idx = y * width + x
            neighborhood = [
                values[(y + dy) * width + (x + dx)]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ]
            total += abs(values[idx] - (sum(neighborhood) / 9.0))
            count += 1
    return total / max(count, 1)


def laplacian_variance(values: list[int], width: int, height: int) -> float:
    if width <= 2 or height <= 2:
        return 0.0
    responses: list[float] = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            idx = y * width + x
            response = (
                4.0 * values[idx]
                - values[idx - 1]
                - values[idx + 1]
                - values[idx - width]
                - values[idx + width]
            )
            responses.append(response)
    if not responses:
        return 0.0
    mean = sum(responses) / len(responses)
    return sum((value - mean) ** 2 for value in responses) / len(responses)


def blockiness_score(values: list[int], width: int, height: int, block: int = 8) -> float:
    if width <= 1 or height <= 1:
        return 0.0
    boundary = []
    interior = []
    for y in range(height):
        for x in range(1, width):
            diff = abs(values[y * width + x] - values[y * width + x - 1])
            (boundary if x % block == 0 else interior).append(diff)
    for y in range(1, height):
        for x in range(width):
            diff = abs(values[y * width + x] - values[(y - 1) * width + x])
            (boundary if y % block == 0 else interior).append(diff)
    boundary_mean = sum(boundary) / len(boundary) if boundary else 0.0
    interior_mean = sum(interior) / len(interior) if interior else 0.0
    return max(0.0, boundary_mean - interior_mean)


def _mask_area(mask: list[int] | None) -> float:
    return (sum(mask) / len(mask)) if mask else 0.0


def _mask_overlap_ratio(mask: list[int] | None, other: list[int] | None) -> float:
    if not mask or not other or len(mask) != len(other) or sum(mask) == 0:
        return 0.0
    overlap = sum(1 for left, right in zip(mask, other) if left and right)
    return overlap / sum(mask)


def extract_image_features(row: dict[str, Any]) -> dict[str, Any]:
    values, width, height = _load_gray_values(row["image_path"])
    mean, std = _mean_std(values)
    ignore = _load_mask_values(row.get("ignore_mask_path"), width, height)
    tamper = _load_mask_values(row.get("tamper_mask_path"), width, height)
    return {
        "width": width,
        "height": height,
        "aspect_ratio": width / height if height else 0.0,
        "image_mean": mean,
        "image_std": std,
        "edge_energy": edge_energy(values, width, height),
        "highpass_residual_energy": highpass_residual_energy(values, width, height),
        "laplacian_variance": laplacian_variance(values, width, height),
        "blockiness_score": blockiness_score(values, width, height),
        "histogram_values": values,
        "ignore_mask_area": _mask_area(ignore),
        "tamper_area": _mask_area(tamper),
        "tamper_occluded_by_ignore": _mask_overlap_ratio(tamper, ignore),
    }


def join_clean_sns_pairs(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    clean_by_base = {str(row.get("base_id")): row for row in rows if row.get("view") == "clean"}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        if row.get("view") == "clean":
            continue
        clean = clean_by_base.get(str(row.get("base_id")))
        if clean is not None:
            pairs.append((clean, row))
    return pairs


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


def _response_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("base_id") or ""),
        str(row.get("profile") or ""),
        str(row.get("view") or ""),
        normalize_label(row.get("content_label", row.get("label"))),
    )


def load_response_records(path: str | Path | None) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if not path:
        return {}
    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in _load_jsonl(path):
        policy = row.get("policy")
        if policy not in {None, "", "original"}:
            continue
        records[_response_key(row)] = row
    return records


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    try:
        return float(left) - float(right)
    except Exception:
        return None


def _bool_drop(clean: Any, sns: Any) -> bool:
    return bool(clean) and not bool(sns)


def build_shift_records(rows: list[dict[str, Any]], response_by_key: dict[tuple[str, str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for clean, sns in join_clean_sns_pairs(rows):
        clean_features = extract_image_features(clean)
        sns_features = extract_image_features(sns)
        clean_response = response_by_key.get(_response_key(clean), {})
        sns_response = response_by_key.get(_response_key(sns), {})
        clean_p = clean_response.get("p_tampered")
        sns_p = sns_response.get("p_tampered")
        clean_iou = clean_response.get("valid_iou")
        sns_iou = sns_response.get("valid_iou")
        clean_pred = clean_response.get("pred_class")
        sns_pred = sns_response.get("pred_class")
        profile = str(sns.get("profile") or "")
        area_ratio = (sns_features["width"] * sns_features["height"]) / max(clean_features["width"] * clean_features["height"], 1)
        record = {
            "marker": MARKER,
            "base_id": str(sns.get("base_id") or ""),
            "content_label": normalize_label(sns.get("content_label")),
            "profile": profile,
            "profile_family": profile_family(profile),
            "clean_image_path": str(clean.get("image_path") or ""),
            "sns_image_path": str(sns.get("image_path") or ""),
            "width_clean": clean_features["width"],
            "height_clean": clean_features["height"],
            "width_sns": sns_features["width"],
            "height_sns": sns_features["height"],
            "aspect_ratio_clean": clean_features["aspect_ratio"],
            "aspect_ratio_sns": sns_features["aspect_ratio"],
            "aspect_ratio_delta_abs": abs(sns_features["aspect_ratio"] - clean_features["aspect_ratio"]),
            "area_ratio": area_ratio,
            "area_ratio_delta_abs": abs(1.0 - area_ratio),
            "image_mean_clean": clean_features["image_mean"],
            "image_mean_sns": sns_features["image_mean"],
            "mean_delta_abs": abs(sns_features["image_mean"] - clean_features["image_mean"]),
            "image_std_clean": clean_features["image_std"],
            "image_std_sns": sns_features["image_std"],
            "std_delta_abs": abs(sns_features["image_std"] - clean_features["image_std"]),
            "histogram_l1": histogram_l1(clean_features["histogram_values"], sns_features["histogram_values"]),
            "edge_energy_clean": clean_features["edge_energy"],
            "edge_energy_sns": sns_features["edge_energy"],
            "edge_energy_delta_abs": abs(sns_features["edge_energy"] - clean_features["edge_energy"]),
            "highpass_energy_clean": clean_features["highpass_residual_energy"],
            "highpass_energy_sns": sns_features["highpass_residual_energy"],
            "highpass_energy_delta_abs": abs(sns_features["highpass_residual_energy"] - clean_features["highpass_residual_energy"]),
            "laplacian_variance_clean": clean_features["laplacian_variance"],
            "laplacian_variance_sns": sns_features["laplacian_variance"],
            "laplacian_variance_delta_abs": abs(sns_features["laplacian_variance"] - clean_features["laplacian_variance"]),
            "blockiness_clean": clean_features["blockiness_score"],
            "blockiness_sns": sns_features["blockiness_score"],
            "blockiness_delta_abs": abs(sns_features["blockiness_score"] - clean_features["blockiness_score"]),
            "crop_scale_proxy": abs(1.0 - area_ratio) + abs(sns_features["aspect_ratio"] - clean_features["aspect_ratio"]),
            "ignore_mask_area": sns_features["ignore_mask_area"],
            "tamper_area": sns_features["tamper_area"],
            "tamper_occluded_by_ignore": sns_features["tamper_occluded_by_ignore"],
            "p_tampered_clean": clean_p,
            "p_tampered_sns": sns_p,
            "delta_p_tampered": _delta(clean_p, sns_p),
            "clean_valid_iou": clean_iou,
            "sns_valid_iou": sns_iou,
            "delta_valid_iou": _delta(clean_iou, sns_iou),
            "activation_flip_off": _bool_drop(clean_response.get("localization_activated"), sns_response.get("localization_activated")),
            "pred_flip": clean_pred is not None and sns_pred is not None and clean_pred != sns_pred,
        }
        record.pop("histogram_values", None)
        records.append(record)
    return records


def pearson_correlation(left: list[float], right: list[float]) -> float | None:
    pairs = [(float(a), float(b)) for a, b in zip(left, right) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((value - mean_x) ** 2 for value in xs)
    var_y = sum((value - mean_y) ** 2 for value in ys)
    if var_x == 0.0 or var_y == 0.0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def correlation_table(records: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for feature in SHIFT_FEATURES:
        table[feature] = {}
        for response in RESPONSE_FEATURES:
            feature_values: list[float] = []
            response_values: list[float] = []
            for row in records:
                left = row.get(feature)
                right = row.get(response)
                if left is None or right is None:
                    continue
                feature_values.append(float(left))
                response_values.append(1.0 if isinstance(right, bool) and right else 0.0 if isinstance(right, bool) else float(right))
            corr = pearson_correlation(feature_values, response_values)
            table[feature][response] = {
                "pearson_r": corr,
                "abs_pearson_r": abs(corr) if corr is not None else None,
                "pair_count": len(feature_values),
            }
    return table


def summarize_by_profile_family(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row.get("profile_family") or "other"), []).append(row)
    out: dict[str, Any] = {}
    for family, items in grouped.items():
        out[family] = {
            "pair_count": len(items),
            "profiles": sorted(set(str(row.get("profile")) for row in items)),
            "mean_delta_p_tampered": _mean_present(items, "delta_p_tampered"),
            "mean_delta_valid_iou": _mean_present(items, "delta_valid_iou"),
            "activation_flip_off_rate": sum(1 for row in items if row.get("activation_flip_off")) / len(items) if items else None,
            "pred_flip_rate": sum(1 for row in items if row.get("pred_flip")) / len(items) if items else None,
            "mean_histogram_l1": _mean_present(items, "histogram_l1"),
            "mean_crop_scale_proxy": _mean_present(items, "crop_scale_proxy"),
            "mean_ignore_mask_area": _mean_present(items, "ignore_mask_area"),
            "mean_tamper_occluded_by_ignore": _mean_present(items, "tamper_occluded_by_ignore"),
        }
    return out


def _mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def dominant_factor(correlations: dict[str, Any]) -> dict[str, Any]:
    scores = {"global_degradation_geometry": 0.0, "local_nuisance": 0.0}
    geometry_features = {
        "aspect_ratio_delta_abs",
        "area_ratio_delta_abs",
        "histogram_l1",
        "edge_energy_delta_abs",
        "highpass_energy_delta_abs",
        "laplacian_variance_delta_abs",
        "blockiness_delta_abs",
        "crop_scale_proxy",
    }
    local_features = {"ignore_mask_area", "tamper_occluded_by_ignore"}
    for feature, response_map in correlations.items():
        feature_score = max((float(entry.get("abs_pearson_r") or 0.0) for entry in response_map.values()), default=0.0)
        if feature in geometry_features:
            scores["global_degradation_geometry"] = max(scores["global_degradation_geometry"], feature_score)
        if feature in local_features:
            scores["local_nuisance"] = max(scores["local_nuisance"], feature_score)
    conclusion = (
        "global_degradation_geometry_dominant"
        if scores["global_degradation_geometry"] > scores["local_nuisance"]
        else "local_nuisance_or_tamper_occlusion_dominant"
        if scores["local_nuisance"] > scores["global_degradation_geometry"]
        else "inconclusive"
    )
    return {"scores": scores, "conclusion": conclusion}


def _render_report(summary: dict[str, Any], family_summary: dict[str, Any], dominance: dict[str, Any]) -> str:
    lines = [
        "# SNSAug V2 Global Degradation Geometry Analysis",
        "",
        MARKER,
        "",
        f"- Records: `{summary['record_count']}`",
        f"- Pair root: `{summary['pair_root']}`",
        f"- Conclusion: `{dominance['conclusion']}`",
        "",
        "| Profile Family | Pairs | Mean delta p_tampered | Mean delta valid IoU | Activation flip-off rate | Pred flip rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, metrics in sorted(family_summary.items()):
        lines.append(
            f"| {family} | {metrics.get('pair_count')} | {metrics.get('mean_delta_p_tampered')} | {metrics.get('mean_delta_valid_iou')} | {metrics.get('activation_flip_off_rate')} | {metrics.get('pred_flip_rate')} |"
        )
    return "\n".join(lines) + "\n"


def _output_plan(output_root: Path) -> dict[str, str]:
    return {
        "global_degradation_geometry_records": str(output_root / "global_degradation_geometry_records.jsonl"),
        "profile_family_shift_summary": str(output_root / "profile_family_shift_summary.json"),
        "feature_response_correlation": str(output_root / "feature_response_correlation.json"),
        "global_degradation_geometry_report": str(output_root / "global_degradation_geometry_report.md"),
        "artifact_manifest": str(output_root / "artifact_manifest.json"),
    }


def build_plan(config: dict[str, Any]) -> dict[str, Any]:
    output_root = _real(config["output_root"])
    return {
        "marker": MARKER,
        "analysis_started": False,
        "training_started": False,
        "record_count": 0,
        "pair_root": str(_real(config["pair_root"])),
        "profiles": list(config.get("profiles") or []),
        "output_paths": _output_plan(output_root),
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def run_snsaug_v2_global_degradation_geometry_analysis(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    assert_valid_config(config, require_exists=not dry_run)
    plan = build_plan(config)
    if dry_run:
        return plan
    output_root = _real(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    meta_rows = load_meta_rows(config["meta_jsonl_path"])
    parsed_rows, warnings = parse_fixed_pair_rows(meta_rows, list(config["profiles"]), config.get("max_samples"))
    responses = load_response_records(config.get("baseline_records_path"))
    records = build_shift_records(parsed_rows, responses)
    family_summary = summarize_by_profile_family(records)
    correlations = correlation_table(records)
    dominance = dominant_factor(correlations)
    summary = {
        **plan,
        "analysis_started": True,
        "record_count": len(records),
        "row_count": len(parsed_rows),
        "warning_count": len(warnings),
        "warnings": warnings,
        "profile_family_counts": {family: metrics["pair_count"] for family, metrics in family_summary.items()},
        "dominant_factor": dominance,
    }
    paths = _output_plan(output_root)
    _write_jsonl(Path(paths["global_degradation_geometry_records"]), records)
    _write_json(Path(paths["profile_family_shift_summary"]), family_summary)
    _write_json(Path(paths["feature_response_correlation"]), {"marker": MARKER, "correlations": correlations, "dominant_factor": dominance})
    _write_text(Path(paths["global_degradation_geometry_report"]), _render_report(summary, family_summary, dominance))
    artifact = {
        "marker": MARKER,
        "config_path": config.get("config_path"),
        "output_root": str(output_root),
        "pair_root": str(_real(config["pair_root"])),
        "meta_jsonl_path": str(_real(config["meta_jsonl_path"])),
        "baseline_records_path": str(_real(config["baseline_records_path"])) if config.get("baseline_records_path") else None,
        "profiles": list(config.get("profiles") or []),
        "record_count": len(records),
        "row_count": len(parsed_rows),
        "profile_family_counts": summary["profile_family_counts"],
        "output_paths": paths,
        "warning_count": len(warnings),
        "warnings": warnings,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
        "analysis_started": True,
        "training_started": False,
    }
    _write_json(Path(paths["artifact_manifest"]), artifact)
    return {**summary, "output_paths": paths}


__all__ = [
    "APPROVAL_TEXT",
    "CONFIG_OK_MARKER",
    "MARKER",
    "SNSAugV2GlobalDegradationGeometryAnalysisError",
    "blockiness_score",
    "build_shift_records",
    "correlation_table",
    "edge_energy",
    "extract_image_features",
    "highpass_residual_energy",
    "join_clean_sns_pairs",
    "laplacian_variance",
    "load_snsaug_v2_global_degradation_geometry_analysis_config",
    "pearson_correlation",
    "profile_family",
    "run_snsaug_v2_global_degradation_geometry_analysis",
    "validate_snsaug_v2_global_degradation_geometry_analysis_config",
]
