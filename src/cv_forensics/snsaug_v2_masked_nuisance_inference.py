"""Inference-time SNSAug nuisance masking evaluation."""

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

MARKER = "SNSAUG_V2_MASKED_NUISANCE_INFERENCE_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_MASKED_NUISANCE_INFERENCE_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_masked_nuisance_inference"
APPROVED_MODE = "approved_local_snsaug_v2_masked_nuisance_inference"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_MASKED_NUISANCE_INFERENCE"
REPO_ROOT = Path(__file__).resolve().parents[2]
CLASS_LABELS = ("real", "synthetic", "tampered")
MASKING_POLICIES = (
    "original",
    "gray_fill",
    "blur_fill",
    "mean_fill",
    "black_fill",
    "dilated_gray_fill",
)
REQUIRED_OUTPUTS = (
    "model_eval_records_masked.jsonl",
    "per_policy_per_profile_metrics.json",
    "clean_vs_sns_masked_delta.json",
    "masked_nuisance_inference_summary.json",
    "masked_nuisance_inference_report.md",
    "visual_gallery_manifest.json",
    "artifact_manifest.json",
)


class SNSAugV2MaskedNuisanceInferenceError(ValueError):
    """Raised when masked nuisance inference validation or execution fails."""


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


def load_snsaug_v2_masked_nuisance_inference_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2MaskedNuisanceInferenceError("masked nuisance inference config root must be a JSON object")
    raw.setdefault("config_path", str(_real(path)))
    return raw


def validate_snsaug_v2_masked_nuisance_inference_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
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
        invalid = [item for item in policies if item not in MASKING_POLICIES]
        if invalid:
            errors.append(f"unsupported masking policies: {invalid}")
        if "original" not in policies:
            errors.append("policies must include original")
    profiles = raw.get("profiles")
    if not isinstance(profiles, list) or not profiles or not all(isinstance(item, str) for item in profiles):
        errors.append("profiles must be a non-empty list of strings")
    elif "clean" not in profiles:
        errors.append("profiles must include clean")
    if raw.get("device") not in {"cpu", "cuda"}:
        errors.append("device must be cpu or cuda")
    for field in ("max_samples", "dilation_radius", "top_n_gallery"):
        if field in raw and raw.get(field) is not None:
            value = raw.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(f"{field} must be a positive integer when provided")
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_masked_nuisance_inference_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2MaskedNuisanceInferenceError("masked nuisance inference config validation failed:\n" + "\n".join(errors))


def _runtime_deps():
    try:
        from PIL import Image, ImageFilter
    except Exception as exc:
        raise SNSAugV2MaskedNuisanceInferenceError("PIL is required for masked nuisance inference") from exc
    return Image, ImageFilter


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


def resolve_pair_mask_path(row: dict[str, Any], pair_root: str | Path, field: str) -> str | None:
    direct = row.get(field)
    if isinstance(direct, str) and direct:
        path = Path(direct)
        return str(path if path.is_absolute() else _real(pair_root) / path)
    rel_field = field.replace("_path", "_relpath")
    rel = row.get(rel_field)
    if isinstance(rel, str) and rel:
        return str(_real(pair_root) / rel)
    directory = "ignore_masks" if field == "ignore_mask_path" else "tamper_masks"
    image_name = Path(str(row.get("image_path") or "")).name
    candidates = []
    if image_name:
        candidates.append(_real(pair_root) / directory / image_name)
    base_id = str(row.get("base_id") or "")
    profile = str(row.get("profile") or "")
    view = str(row.get("view") or "")
    for suffix in (".png", ".jpg", ".jpeg"):
        if base_id and view:
            candidates.append(_real(pair_root) / directory / f"{base_id}__{view}{suffix}")
        if base_id and profile:
            candidates.append(_real(pair_root) / directory / f"{base_id}__{profile}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _mask_to_binary(mask_image: Any, size: tuple[int, int]) -> Any:
    Image, _ImageFilter = _runtime_deps()
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    return mask_image.convert("L").resize(size, nearest).point(lambda px: 255 if int(px) > 0 else 0)


def dilate_mask(mask_image: Any, radius: int = 1) -> Any:
    _Image, ImageFilter = _runtime_deps()
    size = max(3, int(radius) * 2 + 1)
    return mask_image.convert("L").filter(ImageFilter.MaxFilter(size=size)).point(lambda px: 255 if int(px) > 0 else 0)


def mask_area_ratio(mask_image: Any) -> float:
    values = list(mask_image.convert("L").getdata())
    return sum(1 for value in values if int(value) > 0) / len(values) if values else 0.0


def apply_masking_policy(image: Any, ignore_mask: Any | None, policy: str, *, dilation_radius: int = 1) -> tuple[Any, Any | None]:
    Image, ImageFilter = _runtime_deps()
    base = image.convert("RGB")
    if policy == "original" or ignore_mask is None:
        return base.copy(), _mask_to_binary(ignore_mask, base.size) if ignore_mask is not None else None
    mask = _mask_to_binary(ignore_mask, base.size)
    active_mask = dilate_mask(mask, dilation_radius) if policy == "dilated_gray_fill" else mask
    if policy in {"gray_fill", "dilated_gray_fill"}:
        replacement = Image.new("RGB", base.size, (128, 128, 128))
    elif policy == "black_fill":
        replacement = Image.new("RGB", base.size, (0, 0, 0))
    elif policy == "mean_fill":
        pixels = list(base.getdata())
        mean = tuple(int(round(sum(pixel[channel] for pixel in pixels) / max(len(pixels), 1))) for channel in range(3))
        replacement = Image.new("RGB", base.size, mean)
    elif policy == "blur_fill":
        replacement = base.filter(ImageFilter.GaussianBlur(radius=3))
    else:
        raise SNSAugV2MaskedNuisanceInferenceError(f"unsupported masking policy: {policy}")
    out = base.copy()
    out.paste(replacement, mask=active_mask)
    return out, active_mask


def _load_binary_mask_list(Image: Any, path: str | None, size: tuple[int, int]) -> list[int] | None:
    if not path:
        return None
    with Image.open(path) as image:
        mask = _mask_to_binary(image, size)
    return [1 if int(value) > 0 else 0 for value in mask.getdata()]


def _safe_stem(row: dict[str, Any], policy: str, index: int) -> str:
    raw = "__".join([f"{index:06d}", str(row.get("base_id") or "base"), str(row.get("profile") or "profile"), str(row.get("view") or "view"), policy])
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)


def _fixture_for_policy(row: dict[str, Any], policy: str) -> dict[str, Any]:
    sample = dict(row)
    fixtures = row.get("fixture_by_policy")
    if isinstance(fixtures, dict) and isinstance(fixtures.get(policy), dict):
        sample.update(fixtures[policy])
    return sample


def build_masked_input(
    row: dict[str, Any],
    *,
    pair_root: str | Path,
    output_root: str | Path,
    policy: str,
    index: int,
    dilation_radius: int = 1,
) -> dict[str, Any]:
    Image, _ImageFilter = _runtime_deps()
    image_path = str(row.get("image_path") or "")
    if not image_path:
        raise SNSAugV2MaskedNuisanceInferenceError("row missing image_path")
    ignore_path = resolve_pair_mask_path(row, pair_root, "ignore_mask_path")
    with Image.open(image_path) as image:
        base = image.convert("RGB")
    ignore_image = None
    if ignore_path:
        with Image.open(ignore_path) as mask:
            ignore_image = _mask_to_binary(mask, base.size)
    masked, effective_mask = apply_masking_policy(base, ignore_image, policy, dilation_radius=dilation_radius)
    if policy == "original":
        masked_path = image_path
    else:
        masked_path_obj = _real(output_root) / "masked_inputs" / policy / f"{_safe_stem(row, policy, index)}.png"
        masked_path_obj.parent.mkdir(parents=True, exist_ok=True)
        masked.save(masked_path_obj)
        masked_path = str(masked_path_obj)
    return {
        "image_path": masked_path,
        "ignore_mask_path": ignore_path,
        "effective_ignore_mask": effective_mask,
        "mask_area_ratio": mask_area_ratio(effective_mask) if effective_mask is not None else 0.0,
        "image_size": [int(base.size[0]), int(base.size[1])],
    }


def _evaluate_one(bundle: dict[str, Any], config: dict[str, Any], row: dict[str, Any], policy: str, index: int) -> dict[str, Any]:
    Image, _ImageFilter = _runtime_deps()
    masked = build_masked_input(
        row,
        pair_root=config["pair_root"],
        output_root=config["output_root"],
        policy=policy,
        index=index,
        dilation_radius=int(config.get("dilation_radius", 1)),
    )
    sample = _fixture_for_policy(row, policy)
    sample["image_path"] = masked["image_path"]
    tamper_mask_path = resolve_pair_mask_path(row, config["pair_root"], "tamper_mask_path")
    if tamper_mask_path:
        sample["gt_mask_path"] = tamper_mask_path
    policy_config = policy_config_from_bundle(bundle, config, _real(config["output_root"]))
    started = time.perf_counter()
    result = build_policy_gated_record(policy_config, sample, index)
    latency_ms = (time.perf_counter() - started) * 1000.0
    width, height = int(masked["image_size"][0]), int(masked["image_size"][1])
    pred_mask = result.get("_final_mask")
    gt_mask = _load_binary_mask_list(Image, tamper_mask_path, (width, height)) if tamper_mask_path else None
    ignore_mask = _load_binary_mask_list(Image, masked["ignore_mask_path"], (width, height)) if masked["ignore_mask_path"] else None
    metrics = compute_mask_metrics(pred_mask, gt_mask, ignore_mask)
    pred_class = normalize_label(result.get("class"))
    label = normalize_label(row.get("content_label"))
    return {
        "marker": MARKER,
        "row_id": row.get("row_id"),
        "base_id": str(row.get("base_id") or f"row_{index:06d}"),
        "content_label": label,
        "view": str(row.get("view") or ""),
        "profile": str(row.get("profile") or ""),
        "policy": policy,
        "image_path": str(row.get("image_path")),
        "masked_image_path": str(masked["image_path"]),
        "tamper_mask_path": tamper_mask_path,
        "ignore_mask_path": masked["ignore_mask_path"],
        "pred_class": pred_class,
        "class_correct": pred_class == label,
        "p_real": _safe_prob(result.get("class_conf", {}), "real"),
        "p_synthetic": _safe_prob(result.get("class_conf", {}), "synthetic"),
        "p_tampered": _safe_prob(result.get("class_conf", {}), "tampered"),
        "tampered_score": float(result.get("tampered_score", 0.0)),
        "localization_activated": bool(result.get("tile_localization_activated")),
        "final_mask_area_pct": float(result.get("final_mask_area_pct", 0.0)),
        "mask_area_ratio": float(masked["mask_area_ratio"]),
        "raw_iou": metrics["raw_iou"],
        "raw_dice": metrics["raw_dice"],
        "valid_iou": metrics["valid_iou"],
        "valid_dice": metrics["valid_dice"],
        "latency_ms": float(latency_ms),
        "error": None,
    }


def evaluate_masked_rows(bundle: dict[str, Any], config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    policies = list(config.get("policies") or MASKING_POLICIES)
    for row_index, row in enumerate(rows):
        for policy in policies:
            index = row_index * len(policies) + policies.index(policy)
            try:
                records.append(_evaluate_one(bundle, config, row, str(policy), index))
            except Exception as exc:
                records.append(
                    {
                        "marker": MARKER,
                        "row_id": row.get("row_id"),
                        "base_id": str(row.get("base_id") or f"row_{row_index:06d}"),
                        "content_label": normalize_label(row.get("content_label")),
                        "view": str(row.get("view") or ""),
                        "profile": str(row.get("profile") or ""),
                        "policy": str(policy),
                        "image_path": str(row.get("image_path")),
                        "masked_image_path": None,
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
                        "mask_area_ratio": 0.0,
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
            latencies = [float(row["latency_ms"]) for row in items if row.get("latency_ms") is not None]
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
                "non_tampered_high_mask_rate": (
                    sum(1 for row in non_tampered if float(row.get("final_mask_area_pct") or 0.0) > 1.0) / len(non_tampered)
                ) if non_tampered else None,
                "mean_p_tampered_on_tampered": _mean(tampered_probs),
                "mask_area_ratio": _mean([float(row.get("mask_area_ratio") or 0.0) for row in items]),
                "mean_latency_ms": _mean(latencies),
            }
    return out


def compute_recovery_metrics(per_policy_profile: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    original = per_policy_profile.get("original", {})
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for policy, by_profile in per_policy_profile.items():
        if policy == "original":
            continue
        out[policy] = {}
        for profile, metrics in by_profile.items():
            base = original.get(profile, {})
            def delta(name: str) -> float | None:
                left = metrics.get(name)
                right = base.get(name)
                if left is None or right is None:
                    return None
                return float(left) - float(right)

            out[policy][profile] = {
                "tampered_recall_recovery": delta("tampered_recall"),
                "valid_iou_recovery": delta("tampered_valid_mean_iou"),
                "p_tampered_recovery_on_tampered": delta("mean_p_tampered_on_tampered"),
                "synthetic_recall_change": delta("synthetic_recall"),
                "real_fpr_change": delta("real_fpr"),
            }
    return out


def promising_policies(recovery: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for policy, by_profile in recovery.items():
        profiles = [profile for profile in by_profile if profile != "clean"]
        tampered_profiles = [profile for profile in profiles if float((by_profile[profile].get("tampered_recall_recovery") or 0.0)) > 0.0]
        iou_profiles = [profile for profile in profiles if float((by_profile[profile].get("valid_iou_recovery") or 0.0)) > 0.0]
        synthetic_ok = all(float((by_profile[profile].get("synthetic_recall_change") or 0.0)) >= -0.05 for profile in profiles)
        real_ok = all(float((by_profile[profile].get("real_fpr_change") or 0.0)) <= 0.20 for profile in profiles)
        if len(tampered_profiles) >= 2 and len(iou_profiles) >= 2 and synthetic_ok and real_ok:
            out.append(
                {
                    "policy": policy,
                    "tampered_recall_improved_profiles": tampered_profiles,
                    "valid_iou_improved_profiles": iou_profiles,
                    "synthetic_recall_not_collapsed": synthetic_ok,
                    "real_fpr_not_severely_increased": real_ok,
                }
            )
    return out


def _counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        value = str(row.get(key) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


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
        "output_paths": {name: str(output_root / name) for name in REQUIRED_OUTPUTS},
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def _render_report(summary: dict[str, Any], recovery: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [
        "# SNSAug V2 Masked Nuisance Inference",
        "",
        MARKER,
        "",
        f"- Records: `{summary['record_count']}`",
        f"- Promising policies: `{len(summary['promising_policies'])}`",
        "",
        "| Policy | Profile | Tampered Recall Recovery | Valid IoU Recovery | Synthetic Recall Change | Real FPR Change |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for policy, by_profile in recovery.items():
        for profile, metrics in by_profile.items():
            lines.append(
                f"| {policy} | {profile} | {metrics.get('tampered_recall_recovery')} | {metrics.get('valid_iou_recovery')} | {metrics.get('synthetic_recall_change')} | {metrics.get('real_fpr_change')} |"
            )
    return "\n".join(lines) + "\n"


def run_snsaug_v2_masked_nuisance_inference(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
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
    records = evaluate_masked_rows(bundle, config, parsed_rows)
    per_policy_profile = aggregate_per_policy_profile(records)
    recovery = compute_recovery_metrics(per_policy_profile)
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
    gallery = {
        "marker": MARKER,
        "items": [
            {
                "base_id": row["base_id"],
                "profile": row["profile"],
                "policy": row["policy"],
                "masked_image_path": row.get("masked_image_path"),
                "ignore_mask_path": row.get("ignore_mask_path"),
            }
            for row in records
            if row.get("policy") != "original" and row.get("ignore_mask_path")
        ][: int(config.get("top_n_gallery", 12))]
    }
    output_paths = {
        "model_eval_records_masked": _write_jsonl(output_root / "model_eval_records_masked.jsonl", records),
        "per_policy_per_profile_metrics": _write_json(output_root / "per_policy_per_profile_metrics.json", per_policy_profile),
        "clean_vs_sns_masked_delta": _write_json(output_root / "clean_vs_sns_masked_delta.json", recovery),
        "masked_nuisance_inference_summary": _write_json(output_root / "masked_nuisance_inference_summary.json", summary),
        "masked_nuisance_inference_report": _write_text(output_root / "masked_nuisance_inference_report.md", _render_report(summary, recovery)),
        "visual_gallery_manifest": _write_json(output_root / "visual_gallery_manifest.json", gallery),
    }
    artifact_path = str(output_root / "artifact_manifest.json")
    artifact_output_paths = {**output_paths, "artifact_manifest": artifact_path}
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
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
        "inference_started": True,
        "training_started": False,
        "record_count": len(records),
        "row_count": len(parsed_rows),
        "policy_counts": _counts(records, "policy"),
        "profile_counts": _counts(records, "profile"),
        "label_counts": _counts(records, "content_label"),
        "output_paths": artifact_output_paths,
        "warning_count": len(parse_warnings),
        "warnings": parse_warnings,
    }
    output_paths["artifact_manifest"] = _write_json(output_root / "artifact_manifest.json", artifact)
    return {**summary, "output_paths": output_paths}


__all__ = [
    "APPROVAL_TEXT",
    "CONFIG_OK_MARKER",
    "MARKER",
    "MASKING_POLICIES",
    "SNSAugV2MaskedNuisanceInferenceError",
    "aggregate_per_policy_profile",
    "apply_masking_policy",
    "build_masked_input",
    "compute_recovery_metrics",
    "dilate_mask",
    "load_snsaug_v2_masked_nuisance_inference_config",
    "mask_area_ratio",
    "resolve_pair_mask_path",
    "run_snsaug_v2_masked_nuisance_inference",
    "validate_snsaug_v2_masked_nuisance_inference_config",
]
