"""SNSAug v2 residual, SRM, and DCT degradation analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .pre_sns_v3_sns_robustness_eval import normalize_label
from .snsaug_v2_fixed_pairs_eval import load_meta_rows, parse_fixed_pair_rows

MARKER = "SNSAUG_V2_RESIDUAL_DEGRADATION_ANALYSIS_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_RESIDUAL_DEGRADATION_ANALYSIS_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_residual_degradation_analysis"
APPROVED_MODE = "approved_local_snsaug_v2_residual_degradation_analysis"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_RESIDUAL_DEGRADATION_ANALYSIS"
REPO_ROOT = Path(__file__).resolve().parents[2]

PROFILE_FAMILIES = {
    "clean": {"clean"},
    "geometry": {"canvas_9x16_only", "resize_crop_pad", "zoom_crop"},
    "postprocess": {"recompression_light", "resize_jpeg"},
    "screenshot": {"screenshot_recapture_light"},
    "local_overlay": {
        "news_meme_overlay",
        "platform_ui_same_size",
        "tiktok_like",
        "instagram_story_like",
        "youtube_shorts_like",
        "combined_sns_realistic",
    },
}
OUTPUT_NAMES = (
    "residual_degradation_records.jsonl",
    "residual_feature_summary.json",
    "residual_feature_response_correlation.json",
    "residual_degradation_report.md",
    "artifact_manifest.json",
)
FEATURE_COLUMNS = (
    "highpass_energy_delta_abs",
    "srm_energy_delta_abs",
    "srm_mean_delta_abs",
    "srm_std_delta_abs",
    "laplacian_variance_delta_abs",
    "sobel_edge_energy_delta_abs",
    "blockiness_delta_abs",
    "dct_total_energy_delta_abs",
    "dct_low_energy_delta_abs",
    "dct_high_energy_delta_abs",
    "dct_high_low_ratio_delta_abs",
    "histogram_l1",
    "aspect_ratio_delta_abs",
    "area_ratio_delta_abs",
    "crop_scale_proxy",
    "ignore_mask_area",
    "tamper_area",
    "tamper_occluded_by_ignore",
)
RESPONSE_TARGETS = (
    "delta_p_tampered",
    "delta_valid_iou",
    "activation_flip_off",
    "pred_flip",
    "correct_to_wrong",
)


class SNSAugV2ResidualDegradationAnalysisError(ValueError):
    """Raised when residual degradation analysis validation fails."""


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


def load_snsaug_v2_residual_degradation_analysis_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2ResidualDegradationAnalysisError("residual degradation config root must be a JSON object")
    raw.setdefault("config_path", str(_real(path)))
    return raw


def validate_snsaug_v2_residual_degradation_analysis_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "user_approval_text",
        "pair_root",
        "meta_jsonl_path",
        "baseline_records_path",
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
    for field in ("pair_root", "meta_jsonl_path", "baseline_records_path"):
        errors.extend(_validate_under_roots(raw.get(field), field, input_roots, require_exists=require_exists))
    for field in ("global_geometry_analysis_root", "geometry_recovery_root"):
        if raw.get(field):
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
    errors = validate_snsaug_v2_residual_degradation_analysis_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2ResidualDegradationAnalysisError("residual degradation analysis config validation failed:\n" + "\n".join(errors))


def _runtime_deps():
    try:
        from PIL import Image
    except Exception as exc:
        raise SNSAugV2ResidualDegradationAnalysisError("PIL is required for residual degradation analysis") from exc
    return Image


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


def _load_gray_values(path: str | Path) -> tuple[list[float], int, int]:
    Image = _runtime_deps()
    with Image.open(path) as image:
        gray = image.convert("L")
        width, height = gray.size
        return [float(value) for value in gray.getdata()], int(width), int(height)


def _load_mask_values(path: str | Path | None, width: int, height: int) -> list[int] | None:
    if not path:
        return None
    Image = _runtime_deps()
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    with Image.open(path) as image:
        mask = image.convert("L").resize((width, height), nearest)
        return [1 if int(value) > 0 else 0 for value in mask.getdata()]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else None


def _histogram(values: list[float], bins: int = 32) -> list[float]:
    hist = [0] * bins
    for value in values:
        idx = min(bins - 1, max(0, int(value) * bins // 256))
        hist[idx] += 1
    total = float(sum(hist)) or 1.0
    return [count / total for count in hist]


def histogram_l1(left_values: list[float], right_values: list[float]) -> float:
    left = _histogram(left_values)
    right = _histogram(right_values)
    return sum(abs(a - b) for a, b in zip(left, right))


def _convolve_valid(values: list[float], width: int, height: int, kernel: tuple[tuple[float, ...], ...]) -> list[float]:
    k_height = len(kernel)
    k_width = len(kernel[0]) if kernel else 0
    pad_y = k_height // 2
    pad_x = k_width // 2
    if width <= pad_x * 2 or height <= pad_y * 2:
        return []
    out: list[float] = []
    for y in range(pad_y, height - pad_y):
        for x in range(pad_x, width - pad_x):
            total = 0.0
            for ky in range(k_height):
                for kx in range(k_width):
                    yy = y + ky - pad_y
                    xx = x + kx - pad_x
                    total += values[yy * width + xx] * kernel[ky][kx]
            out.append(total)
    return out


def highpass_residual_energy(values: list[float], width: int, height: int) -> float:
    kernel = ((-1.0, -1.0, -1.0), (-1.0, 8.0, -1.0), (-1.0, -1.0, -1.0))
    responses = _convolve_valid(values, width, height, kernel)
    return _mean([abs(value) for value in responses])


def srm_like_features(values: list[float], width: int, height: int) -> dict[str, float]:
    filters = (
        ((0.0, 0.0, 0.0), (-1.0, 2.0, -1.0), (0.0, 0.0, 0.0)),
        ((0.0, -1.0, 0.0), (0.0, 2.0, 0.0), (0.0, -1.0, 0.0)),
        ((-1.0, 2.0, -1.0), (2.0, -4.0, 2.0), (-1.0, 2.0, -1.0)),
        ((1.0, -2.0, 1.0), (-2.0, 4.0, -2.0), (1.0, -2.0, 1.0)),
    )
    responses: list[float] = []
    for kernel in filters:
        responses.extend(_convolve_valid(values, width, height, kernel))
    abs_values = [abs(value) for value in responses]
    return {
        "srm_residual_mean": _mean(responses),
        "srm_residual_std": _std(responses),
        "srm_residual_energy": _mean(abs_values),
    }


def laplacian_variance(values: list[float], width: int, height: int) -> float:
    responses = _convolve_valid(values, width, height, ((0.0, -1.0, 0.0), (-1.0, 4.0, -1.0), (0.0, -1.0, 0.0)))
    if not responses:
        return 0.0
    mean = _mean(responses)
    return sum((value - mean) ** 2 for value in responses) / len(responses)


def sobel_edge_energy(values: list[float], width: int, height: int) -> float:
    gx = _convolve_valid(values, width, height, ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)))
    gy = _convolve_valid(values, width, height, ((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0)))
    return _mean([math.sqrt(x * x + y * y) for x, y in zip(gx, gy)])


def blockiness_score(values: list[float], width: int, height: int, block: int = 8) -> float:
    if width <= 1 or height <= 1:
        return 0.0
    boundary: list[float] = []
    interior: list[float] = []
    for y in range(height):
        for x in range(1, width):
            diff = abs(values[y * width + x] - values[y * width + x - 1])
            (boundary if x % block == 0 else interior).append(diff)
    for y in range(1, height):
        for x in range(width):
            diff = abs(values[y * width + x] - values[(y - 1) * width + x])
            (boundary if y % block == 0 else interior).append(diff)
    return max(0.0, _mean(boundary) - _mean(interior))


def dct_energy_summary(values: list[float], width: int, height: int, block: int = 8) -> dict[str, float]:
    if width < block or height < block:
        return {"dct_total_energy": 0.0, "dct_low_energy": 0.0, "dct_high_energy": 0.0, "dct_high_low_ratio": 0.0}
    cos_cache = {
        (u, x): math.cos(((2 * x + 1) * u * math.pi) / (2 * block))
        for u in range(block)
        for x in range(block)
    }
    total = 0.0
    low = 0.0
    high = 0.0
    block_count = 0
    for by in range(0, height - block + 1, block):
        for bx in range(0, width - block + 1, block):
            block_count += 1
            centered = [values[(by + y) * width + (bx + x)] - 128.0 for y in range(block) for x in range(block)]
            for u in range(block):
                for v in range(block):
                    coeff = 0.0
                    for y in range(block):
                        for x in range(block):
                            coeff += centered[y * block + x] * cos_cache[(u, x)] * cos_cache[(v, y)]
                    energy = coeff * coeff
                    total += energy
                    if u + v <= 2:
                        low += energy
                    elif u + v >= 6:
                        high += energy
    scale = float(block_count) or 1.0
    total /= scale
    low /= scale
    high /= scale
    return {
        "dct_total_energy": total,
        "dct_low_energy": low,
        "dct_high_energy": high,
        "dct_high_low_ratio": high / max(low, 1e-9),
    }


def _mask_area(mask: list[int] | None) -> float:
    return (sum(mask) / len(mask)) if mask else 0.0


def _mask_overlap_ratio(mask: list[int] | None, other: list[int] | None) -> float:
    if not mask or not other or len(mask) != len(other) or sum(mask) == 0:
        return 0.0
    overlap = sum(1 for left, right in zip(mask, other) if left and right)
    return overlap / sum(mask)


def extract_residual_features(row: dict[str, Any]) -> dict[str, Any]:
    values, width, height = _load_gray_values(row["image_path"])
    ignore = _load_mask_values(row.get("ignore_mask_path"), width, height)
    tamper = _load_mask_values(row.get("tamper_mask_path"), width, height)
    srm = srm_like_features(values, width, height)
    dct = dct_energy_summary(values, width, height)
    return {
        "width": width,
        "height": height,
        "aspect_ratio": width / height if height else 0.0,
        "image_mean": _mean(values),
        "image_std": _std(values),
        "highpass_residual_energy": highpass_residual_energy(values, width, height),
        "laplacian_variance": laplacian_variance(values, width, height),
        "sobel_edge_energy": sobel_edge_energy(values, width, height),
        "blockiness_score": blockiness_score(values, width, height),
        "histogram_values": values,
        "ignore_mask_area": _mask_area(ignore),
        "tamper_area": _mask_area(tamper),
        "tamper_occluded_by_ignore": _mask_overlap_ratio(tamper, ignore),
        **srm,
        **dct,
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


def load_response_records(path: str | Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in _load_jsonl(path):
        policy = row.get("policy")
        if policy not in {None, "", "original"}:
            continue
        records[_response_key(row)] = row
    return records


def _delta(clean: Any, sns: Any) -> float | None:
    if clean is None or sns is None:
        return None
    try:
        return float(clean) - float(sns)
    except Exception:
        return None


def _bool_drop(clean: Any, sns: Any) -> bool:
    return bool(clean) and not bool(sns)


def _correct(row: dict[str, Any], label: str) -> bool | None:
    pred = row.get("pred_class")
    if pred is None:
        return None
    return normalize_label(pred) == label


def build_residual_degradation_records(
    rows: list[dict[str, Any]],
    response_by_key: dict[tuple[str, str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for clean, sns in join_clean_sns_pairs(rows):
        clean_features = extract_residual_features(clean)
        sns_features = extract_residual_features(sns)
        clean_response = response_by_key.get(_response_key(clean), {})
        sns_response = response_by_key.get(_response_key(sns), {})
        profile = str(sns.get("profile") or "")
        label = normalize_label(sns.get("content_label"))
        area_ratio = (sns_features["width"] * sns_features["height"]) / max(clean_features["width"] * clean_features["height"], 1)
        clean_correct = _correct(clean_response, label)
        sns_correct = _correct(sns_response, label)
        clean_pred = clean_response.get("pred_class")
        sns_pred = sns_response.get("pred_class")
        record = {
            "marker": MARKER,
            "base_id": str(sns.get("base_id") or ""),
            "content_label": label,
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
            "crop_scale_proxy": abs(1.0 - area_ratio) + abs(sns_features["aspect_ratio"] - clean_features["aspect_ratio"]),
            "image_mean_clean": clean_features["image_mean"],
            "image_mean_sns": sns_features["image_mean"],
            "image_std_clean": clean_features["image_std"],
            "image_std_sns": sns_features["image_std"],
            "histogram_l1": histogram_l1(clean_features["histogram_values"], sns_features["histogram_values"]),
            "highpass_energy_clean": clean_features["highpass_residual_energy"],
            "highpass_energy_sns": sns_features["highpass_residual_energy"],
            "highpass_energy_delta_abs": abs(sns_features["highpass_residual_energy"] - clean_features["highpass_residual_energy"]),
            "srm_mean_clean": clean_features["srm_residual_mean"],
            "srm_mean_sns": sns_features["srm_residual_mean"],
            "srm_mean_delta_abs": abs(sns_features["srm_residual_mean"] - clean_features["srm_residual_mean"]),
            "srm_std_clean": clean_features["srm_residual_std"],
            "srm_std_sns": sns_features["srm_residual_std"],
            "srm_std_delta_abs": abs(sns_features["srm_residual_std"] - clean_features["srm_residual_std"]),
            "srm_energy_clean": clean_features["srm_residual_energy"],
            "srm_energy_sns": sns_features["srm_residual_energy"],
            "srm_energy_delta_abs": abs(sns_features["srm_residual_energy"] - clean_features["srm_residual_energy"]),
            "laplacian_variance_clean": clean_features["laplacian_variance"],
            "laplacian_variance_sns": sns_features["laplacian_variance"],
            "laplacian_variance_delta_abs": abs(sns_features["laplacian_variance"] - clean_features["laplacian_variance"]),
            "sobel_edge_energy_clean": clean_features["sobel_edge_energy"],
            "sobel_edge_energy_sns": sns_features["sobel_edge_energy"],
            "sobel_edge_energy_delta_abs": abs(sns_features["sobel_edge_energy"] - clean_features["sobel_edge_energy"]),
            "blockiness_clean": clean_features["blockiness_score"],
            "blockiness_sns": sns_features["blockiness_score"],
            "blockiness_delta_abs": abs(sns_features["blockiness_score"] - clean_features["blockiness_score"]),
            "dct_total_energy_clean": clean_features["dct_total_energy"],
            "dct_total_energy_sns": sns_features["dct_total_energy"],
            "dct_total_energy_delta_abs": abs(sns_features["dct_total_energy"] - clean_features["dct_total_energy"]),
            "dct_low_energy_clean": clean_features["dct_low_energy"],
            "dct_low_energy_sns": sns_features["dct_low_energy"],
            "dct_low_energy_delta_abs": abs(sns_features["dct_low_energy"] - clean_features["dct_low_energy"]),
            "dct_high_energy_clean": clean_features["dct_high_energy"],
            "dct_high_energy_sns": sns_features["dct_high_energy"],
            "dct_high_energy_delta_abs": abs(sns_features["dct_high_energy"] - clean_features["dct_high_energy"]),
            "dct_high_low_ratio_clean": clean_features["dct_high_low_ratio"],
            "dct_high_low_ratio_sns": sns_features["dct_high_low_ratio"],
            "dct_high_low_ratio_delta_abs": abs(sns_features["dct_high_low_ratio"] - clean_features["dct_high_low_ratio"]),
            "ignore_mask_area": sns_features["ignore_mask_area"],
            "tamper_area": sns_features["tamper_area"],
            "tamper_occluded_by_ignore": sns_features["tamper_occluded_by_ignore"],
            "p_tampered_clean": clean_response.get("p_tampered"),
            "p_tampered_sns": sns_response.get("p_tampered"),
            "delta_p_tampered": _delta(clean_response.get("p_tampered"), sns_response.get("p_tampered")),
            "clean_valid_iou": clean_response.get("valid_iou"),
            "sns_valid_iou": sns_response.get("valid_iou"),
            "delta_valid_iou": _delta(clean_response.get("valid_iou"), sns_response.get("valid_iou")),
            "activation_flip_off": _bool_drop(clean_response.get("localization_activated"), sns_response.get("localization_activated")),
            "pred_flip": clean_pred is not None and sns_pred is not None and normalize_label(clean_pred) != normalize_label(sns_pred),
            "correct_to_wrong": clean_correct is True and sns_correct is False,
        }
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


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for _value, index in ordered[i:j]:
            ranks[index] = rank
        i = j
    return ranks


def spearman_correlation(left: list[float], right: list[float]) -> float | None:
    pairs = [(float(a), float(b)) for a, b in zip(left, right) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 2:
        return None
    return pearson_correlation(_ranks([pair[0] for pair in pairs]), _ranks([pair[1] for pair in pairs]))


def _target_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except Exception:
        return None


def _correlate_subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for feature in FEATURE_COLUMNS:
        table[feature] = {}
        for target in RESPONSE_TARGETS:
            feature_values: list[float] = []
            target_values: list[float] = []
            for row in records:
                left = row.get(feature)
                right = _target_value(row.get(target))
                if left is None or right is None:
                    continue
                feature_values.append(float(left))
                target_values.append(float(right))
            pearson = pearson_correlation(feature_values, target_values)
            spearman = spearman_correlation(feature_values, target_values)
            table[feature][target] = {
                "pearson_r": pearson,
                "abs_pearson_r": abs(pearson) if pearson is not None else None,
                "spearman_r": spearman,
                "abs_spearman_r": abs(spearman) if spearman is not None else None,
                "pair_count": len(feature_values),
            }
    return table


def correlation_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, Any] = {}
    by_family: dict[str, Any] = {}
    labels = sorted(set(str(row.get("content_label") or "") for row in records if row.get("content_label")))
    families = sorted(set(str(row.get("profile_family") or "") for row in records if row.get("profile_family")))
    for label in labels:
        by_label[label] = _correlate_subset([row for row in records if row.get("content_label") == label])
    for family in families:
        by_family[family] = _correlate_subset([row for row in records if row.get("profile_family") == family])
    overall = _correlate_subset(records)
    return {"marker": MARKER, "overall": overall, "by_label": by_label, "by_profile_family": by_family}


def feature_group(feature: str) -> str:
    lowered = str(feature or "").lower()
    if any(token in lowered for token in ("dct", "srm", "residual", "highpass", "high_pass", "blockiness", "laplacian", "sobel", "edge")):
        return "residual_dct"
    if any(token in lowered for token in ("ignore", "occlud", "mask_area", "tamper_occluded", "tamper_area")):
        return "local_nuisance"
    if any(token in lowered for token in ("crop", "scale", "aspect", "area_ratio", "width", "height", "geometry")):
        return "geometry"
    if any(token in lowered for token in ("histogram", "mean", "std", "color")):
        return "color_histogram"
    return "other"


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _looks_like_correlation_entry(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return any(key in value for key in ("abs_pearson_r", "abs_spearman_r", "pearson_r", "spearman_r"))


def _correlation_row_from_entry(entry: dict[str, Any], path: list[str]) -> dict[str, Any] | None:
    feature = entry.get("feature")
    target = entry.get("target")
    if feature is None or target is None:
        if len(path) < 2:
            return None
        feature = path[-2]
        target = path[-1]
    pearson = _float_or_none(entry.get("pearson_r"))
    spearman = _float_or_none(entry.get("spearman_r"))
    abs_pearson = _float_or_none(entry.get("abs_pearson_r"))
    abs_spearman = _float_or_none(entry.get("abs_spearman_r"))
    score = max(value for value in (abs_pearson, abs_spearman, abs(pearson) if pearson is not None else None, abs(spearman) if spearman is not None else None, 0.0) if value is not None)
    if score <= 0.0:
        return None
    pair_count = entry.get("pair_count")
    return {
        "feature": str(feature),
        "target": str(target),
        "score": float(score),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "pair_count": int(pair_count) if isinstance(pair_count, int) or (isinstance(pair_count, float) and pair_count.is_integer()) else pair_count,
        "feature_group": feature_group(str(feature)),
        "path": ".".join(path),
    }


def flatten_correlation_schema(payload: Any, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def walk(value: Any, path: list[str]) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, [*path, str(index)])
            return
        if not isinstance(value, dict):
            return
        if _looks_like_correlation_entry(value):
            row = _correlation_row_from_entry(value, path)
            if row is not None:
                rows.append(row)
            return
        for key, item in value.items():
            if key in {"marker", "decision", "top_correlations"}:
                continue
            walk(item, [*path, str(key)])

    walk(payload, [])
    rows.sort(key=lambda row: (float(row["score"]), int(row["pair_count"] or 0) if isinstance(row.get("pair_count"), int) else 0, row["feature"], row["target"]), reverse=True)
    if limit is not None:
        return rows[: int(limit)]
    return rows


def summarize_features(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row.get("profile_family") or "other"), []).append(row)
    families: dict[str, Any] = {}
    for family, items in grouped.items():
        feature_means = {feature: _mean_present(items, feature) for feature in FEATURE_COLUMNS}
        families[family] = {
            "pair_count": len(items),
            "profiles": sorted(set(str(row.get("profile")) for row in items)),
            "mean_delta_p_tampered": _mean_present(items, "delta_p_tampered"),
            "mean_delta_valid_iou": _mean_present(items, "delta_valid_iou"),
            "activation_flip_off_rate": sum(1 for row in items if row.get("activation_flip_off")) / len(items) if items else None,
            "pred_flip_rate": sum(1 for row in items if row.get("pred_flip")) / len(items) if items else None,
            "correct_to_wrong_rate": sum(1 for row in items if row.get("correct_to_wrong")) / len(items) if items else None,
            "feature_means": feature_means,
        }
    return {"marker": MARKER, "record_count": len(records), "profile_families": families}


def decide_next_step(correlations: dict[str, Any]) -> dict[str, Any]:
    top = flatten_correlation_schema(correlations, limit=25)
    scores = {
        "residual_dct": 0.0,
        "geometry": 0.0,
        "local_nuisance": 0.0,
        "color_histogram": 0.0,
        "other": 0.0,
    }
    for row in top:
        group = str(row.get("feature_group") or "other")
        scores[group] = max(float(scores.get(group, 0.0)), float(row.get("score") or 0.0))
    if not top:
        decision = "correlation_parser_failed_or_empty"
    elif scores["residual_dct"] >= 0.35 and scores["residual_dct"] >= scores["geometry"] and scores["residual_dct"] >= scores["local_nuisance"]:
        decision = "residual_dct_signal_dominant"
    elif scores["geometry"] >= 0.35 and scores["residual_dct"] >= 0.25:
        decision = "mixed_low_level_and_geometry_signal"
    elif scores["geometry"] >= 0.35:
        decision = "geometry_still_dominant_or_mixed"
    elif max(scores.values()) < 0.20:
        decision = "weak_low_level_signal"
    else:
        decision = "mixed_low_level_and_geometry_signal"
    recommendations = {
        "residual_dct_signal_dominant": "train_0071_lightweight_residual_dct_branch",
        "geometry_still_dominant_or_mixed": "refine_geometry_normalized_preprocessing",
        "mixed_low_level_and_geometry_signal": "compare_residual_branch_against_geometry_preprocessing",
        "weak_low_level_signal": "revisit_model_calibration_or_class_head_robustness",
        "correlation_parser_failed_or_empty": "fix_correlation_reporting_or_collect_more_records",
    }
    return {
        "decision": decision,
        "scores": scores,
        "top_correlations": top[:10],
        "recommendation": recommendations[decision],
    }


def _output_plan(output_root: Path) -> dict[str, str]:
    return {
        "residual_degradation_records": str(output_root / "residual_degradation_records.jsonl"),
        "residual_feature_summary": str(output_root / "residual_feature_summary.json"),
        "residual_feature_response_correlation": str(output_root / "residual_feature_response_correlation.json"),
        "residual_degradation_report": str(output_root / "residual_degradation_report.md"),
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


def _render_report(summary: dict[str, Any], feature_summary: dict[str, Any], decision: dict[str, Any]) -> str:
    lines = [
        "# SNSAug V2 Residual Degradation Analysis",
        "",
        MARKER,
        "",
        f"- Records: `{summary['record_count']}`",
        f"- Pair root: `{summary['pair_root']}`",
        f"- Decision: `{decision.get('decision')}`",
        f"- Recommendation: `{decision['recommendation']}`",
        "",
        "## Top Correlations",
        "",
        "| Feature | Target | Score | Pearson r | Spearman r | Pairs | Group | Path |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in decision.get("top_correlations") or []:
        lines.append(
            f"| {row.get('feature')} | {row.get('target')} | {row.get('score')} | {row.get('pearson_r')} | {row.get('spearman_r')} | {row.get('pair_count')} | {row.get('feature_group')} | {row.get('path')} |"
        )
    lines.extend([
        "",
        "## Profile Families",
        "",
        "| Profile Family | Pairs | Mean delta p_tampered | Mean delta valid IoU | Activation flip-off rate | Pred flip rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for family, metrics in sorted((feature_summary.get("profile_families") or {}).items()):
        lines.append(
            f"| {family} | {metrics.get('pair_count')} | {metrics.get('mean_delta_p_tampered')} | {metrics.get('mean_delta_valid_iou')} | {metrics.get('activation_flip_off_rate')} | {metrics.get('pred_flip_rate')} |"
        )
    return "\n".join(lines) + "\n"


def run_snsaug_v2_residual_degradation_analysis(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    assert_valid_config(config, require_exists=not dry_run)
    plan = build_plan(config)
    if dry_run:
        return plan
    output_root = _real(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    meta_rows = load_meta_rows(config["meta_jsonl_path"])
    parsed_rows, warnings = parse_fixed_pair_rows(meta_rows, list(config["profiles"]), config.get("max_samples"))
    responses = load_response_records(config["baseline_records_path"])
    records = build_residual_degradation_records(parsed_rows, responses)
    feature_summary = summarize_features(records)
    correlations = correlation_report(records)
    decision = decide_next_step(correlations)
    summary = {
        **plan,
        "analysis_started": True,
        "record_count": len(records),
        "row_count": len(parsed_rows),
        "warning_count": len(warnings),
        "warnings": warnings,
        "profile_family_counts": {
            family: metrics["pair_count"]
            for family, metrics in (feature_summary.get("profile_families") or {}).items()
        },
        "decision": decision,
    }
    paths = _output_plan(output_root)
    _write_jsonl(Path(paths["residual_degradation_records"]), records)
    _write_json(Path(paths["residual_feature_summary"]), feature_summary)
    _write_json(Path(paths["residual_feature_response_correlation"]), {**correlations, "decision": decision, "top_correlations": decision["top_correlations"]})
    _write_text(Path(paths["residual_degradation_report"]), _render_report(summary, feature_summary, decision))
    artifact = {
        "marker": MARKER,
        "config_path": config.get("config_path"),
        "output_root": str(output_root),
        "pair_root": str(_real(config["pair_root"])),
        "meta_jsonl_path": str(_real(config["meta_jsonl_path"])),
        "baseline_records_path": str(_real(config["baseline_records_path"])),
        "global_geometry_analysis_root": str(_real(config["global_geometry_analysis_root"])) if config.get("global_geometry_analysis_root") else None,
        "geometry_recovery_root": str(_real(config["geometry_recovery_root"])) if config.get("geometry_recovery_root") else None,
        "profiles": list(config.get("profiles") or []),
        "record_count": len(records),
        "row_count": len(parsed_rows),
        "profile_family_counts": summary["profile_family_counts"],
        "decision": decision,
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
    "SNSAugV2ResidualDegradationAnalysisError",
    "blockiness_score",
    "build_residual_degradation_records",
    "correlation_report",
    "dct_energy_summary",
    "extract_residual_features",
    "highpass_residual_energy",
    "join_clean_sns_pairs",
    "load_response_records",
    "load_snsaug_v2_residual_degradation_analysis_config",
    "pearson_correlation",
    "run_snsaug_v2_residual_degradation_analysis",
    "srm_like_features",
    "validate_snsaug_v2_residual_degradation_analysis_config",
]
