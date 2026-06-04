"""SNSAug V2 robustness evaluation and fragile mining."""

from __future__ import annotations

import json
import math
import time
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
from .pre_sns_v3_sns_robustness_eval import (
    _mean,
    _median,
    _safe_prob,
    confusion_matrix,
    json_safe,
    load_best_bundle,
    load_validation_samples,
    macro_f1_and_details,
    normalize_label,
    policy_config_from_bundle,
    sample_id,
    sample_label,
    select_samples,
)
from .pre_sns_v3_v2_policy_gated_report import build_policy_gated_record
from .snsaug_v2 import PROFILES, SEVERITIES, SNSAugV2Augmentor, SNSAugV2Config

MARKER = "SNSAUG_V2_ROBUSTNESS_EVAL_AND_MINING_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_ROBUSTNESS_EVAL_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_robustness_eval"
APPROVED_MODE = "approved_local_snsaug_v2_robustness_eval"
CLASS_LABELS = ("real", "synthetic", "tampered")


class SNSAugV2RobustnessEvalError(ValueError):
    """Raised when snsaug v2 robustness evaluation inputs or guardrails fail."""


def load_snsaug_v2_robustness_eval_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2RobustnessEvalError("snsaug_v2 robustness config root must be a JSON object")
    return raw


def _as_roots(value: Any) -> list[str]:
    return value if isinstance(value, list) and all(isinstance(item, str) for item in value) else []


def validate_snsaug_v2_robustness_eval_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "best_bundle_path",
        "validation_manifest_path",
        "approved_input_roots",
        "approved_output_roots",
        "output_root",
        "profiles",
        "severity",
        "seed",
        "max_samples",
        "samples_per_class",
        "balanced_sampling",
        "device",
        "no_training",
        "no_download",
        "no_network",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    if raw.get("config_kind") != APPROVED_KIND:
        errors.append(_err(f"config_kind must be {APPROVED_KIND}"))
    if raw.get("execution_mode") != APPROVED_MODE:
        errors.append(_err(f"execution_mode must be {APPROVED_MODE}"))
    for flag in ("no_training", "no_download", "no_network"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if raw.get("balanced_sampling") not in {True, False}:
        errors.append(_err("balanced_sampling must be boolean"))
    roots = _as_roots(raw.get("approved_input_roots"))
    if not roots:
        errors.append(_err("approved_input_roots must be a non-empty list of absolute paths"))
    for index, root in enumerate(roots):
        errors.extend(_validate_absolute_path(root, f"approved_input_roots[{index}]"))
    output_roots = _as_roots(raw.get("approved_output_roots"))
    if not output_roots:
        errors.append(_err("approved_output_roots must be a non-empty list of absolute paths"))
    for index, root in enumerate(output_roots):
        errors.extend(_validate_absolute_path(root, f"approved_output_roots[{index}]"))
    for field in ("best_bundle_path", "validation_manifest_path", "train_manifest_path"):
        if raw.get(field):
            errors.extend(_validate_under_roots(raw.get(field), field, roots, require_file=require_exists))
    output_root = raw.get("output_root")
    errors.extend(_validate_absolute_path(output_root, "output_root", require_dir_parent=require_exists))
    if isinstance(output_root, str) and output_root.startswith("/"):
        if _inside_repo(_real(output_root)):
            errors.append(_err("output_root must be outside repository"))
        if any(_is_under(output_root, root) for root in roots):
            errors.append(_err("output_root must not be under approved input roots"))
        if output_roots and not any(_is_under(output_root, root) or str(_real(output_root)) == str(_real(root)) for root in output_roots):
            errors.append(_err("output_root must be under an approved output root"))
    for field in ("seed", "max_samples", "samples_per_class", "top_n_worst_cases"):
        if field in raw:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(_err(f"{field} must be a positive integer"))
    profiles = raw.get("profiles")
    if not isinstance(profiles, list) or not profiles or not all(isinstance(item, str) for item in profiles):
        errors.append(_err("profiles must be a non-empty list of strings"))
    else:
        invalid = [item for item in profiles if item not in PROFILES or item == "screenshot_basic"]
        if invalid:
            errors.append(_err(f"unsupported evaluation profiles: {invalid}"))
        if "clean" not in profiles:
            errors.append(_err("profiles must include clean"))
    if raw.get("severity") not in SEVERITIES:
        errors.append(_err(f"severity must be one of {SEVERITIES}"))
    if raw.get("device") not in {"cpu", "cuda"}:
        errors.append(_err("device must be cpu or cuda"))
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_robustness_eval_config(config, require_exists)
    if errors:
        raise SNSAugV2RobustnessEvalError("snsaug_v2 robustness config validation failed:\n" + "\n".join(errors))


def _runtime_deps():
    try:
        from PIL import Image
    except Exception as exc:
        raise SNSAugV2RobustnessEvalError("PIL is required for snsaug_v2 robustness evaluation") from exc
    return Image


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


def _load_mask(Image: Any, path: str | None) -> Any | None:
    if not path:
        return None
    with Image.open(path) as image:
        return image.convert("L")


def _save_augmented_sample(output_root: Path, split: str, base_id_value: str, profile: str, image: Any, tamper_mask: Any | None, ignore_mask: Any) -> dict[str, str | None]:
    case_root = output_root / "transformed_inputs" / split / base_id_value
    case_root.mkdir(parents=True, exist_ok=True)
    image_path = case_root / f"{profile}.png"
    image.save(image_path)
    mask_path = None
    if tamper_mask is not None:
        mask_path = case_root / f"{profile}_tamper_mask.png"
        tamper_mask.save(mask_path)
    ignore_path = case_root / f"{profile}_ignore_mask.png"
    ignore_mask.save(ignore_path)
    return {
        "image_path": str(image_path),
        "tamper_mask_path": str(mask_path) if mask_path is not None else None,
        "ignore_mask_path": str(ignore_path),
    }


def _mask_area_pct(mask_image: Any | None) -> float:
    if mask_image is None:
        return 0.0
    values = list(mask_image.getdata())
    if not values:
        return 0.0
    active = sum(1 for value in values if int(value) > 0)
    return (100.0 * float(active)) / float(len(values))


def _profile_seed(config: dict[str, Any], split: str, base_id_value: str, profile: str) -> int:
    base = int(config.get("seed", 1))
    token = f"{split}:{base_id_value}:{profile}"
    return base + sum((index + 1) * ord(ch) for index, ch in enumerate(token))


def _apply_fixture_profile(sample: dict[str, Any], profile: str) -> dict[str, Any]:
    merged = dict(sample)
    fixture_by_profile = sample.get("fixture_by_profile")
    if isinstance(fixture_by_profile, dict):
        profile_fixture = fixture_by_profile.get(profile)
        if isinstance(profile_fixture, dict):
            if "fixture_long256_report" in profile_fixture:
                merged["fixture_long256_report"] = profile_fixture["fixture_long256_report"]
            if "fixture_v2_probability_mask" in profile_fixture:
                merged["fixture_v2_probability_mask"] = profile_fixture["fixture_v2_probability_mask"]
            if "fixture_v2_mask" in profile_fixture:
                merged["fixture_v2_mask"] = profile_fixture["fixture_v2_mask"]
    return merged


def _evaluate_augmented_record(
    bundle: dict[str, Any],
    config: dict[str, Any],
    sample: dict[str, Any],
    split: str,
    base_id_value: str,
    profile: str,
    saved_paths: dict[str, str | None],
    index: int,
    aug_meta: dict[str, Any],
) -> dict[str, Any]:
    policy_config = policy_config_from_bundle(bundle, config, Path(config["output_root"]))
    eval_sample = _apply_fixture_profile(sample, profile)
    eval_sample["image_path"] = str(saved_paths["image_path"])
    if saved_paths.get("tamper_mask_path"):
        eval_sample["gt_mask_path"] = str(saved_paths["tamper_mask_path"])
    started = time.perf_counter()
    result = build_policy_gated_record(policy_config, eval_sample, index)
    latency_ms = (time.perf_counter() - started) * 1000.0
    class_conf = result.get("class_conf", {})
    return {
        "base_id": base_id_value,
        "split": split,
        "label": sample_label(sample),
        "profile": profile,
        "seed": int(aug_meta["seed"]),
        "pred_class": normalize_label(result.get("class")),
        "class_correct": normalize_label(result.get("class")) == sample_label(sample),
        "p_real": _safe_prob(class_conf, "real"),
        "p_synthetic": _safe_prob(class_conf, "synthetic"),
        "p_tampered": _safe_prob(class_conf, "tampered"),
        "final_iou": result.get("final_iou"),
        "final_dice": result.get("final_dice"),
        "localization_activated": bool(result.get("tile_localization_activated")),
        "final_mask_source": result.get("final_mask_source"),
        "latency_ms": float(latency_ms),
        "fps": float(1000.0 / latency_ms) if latency_ms > 0 else None,
        "ignore_mask_area_pct": _mask_area_pct(_load_mask(_runtime_deps(), saved_paths.get("ignore_mask_path"))),
        "overlay_area_pct": _mask_area_pct(_load_mask(_runtime_deps(), saved_paths.get("ignore_mask_path"))),
        "source_image_path": str(sample.get("image_path")),
        "transformed_image_path": str(saved_paths["image_path"]),
        "transformed_mask_path": saved_paths.get("tamper_mask_path"),
        "ignore_mask_path": saved_paths.get("ignore_mask_path"),
        "overlay_boxes": aug_meta.get("overlay_boxes", []),
        "_result": result,
    }


def evaluate_manifest(
    bundle: dict[str, Any],
    config: dict[str, Any],
    manifest_path: str,
    split: str,
    output_root: Path,
) -> list[dict[str, Any]]:
    Image = _runtime_deps()
    samples = select_samples(load_validation_samples(manifest_path), config)
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        base_id_value = sample_id(sample, index)
        image_path = sample.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            continue
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
        tamper_mask = _load_mask(Image, str(sample.get("mask_path") or sample.get("gt_mask_path") or "")) if sample.get("mask_path") or sample.get("gt_mask_path") else None
        for profile in config["profiles"]:
            aug_seed = _profile_seed(config, split, base_id_value, str(profile))
            aug_config = SNSAugV2Config(
                profile=str(profile),
                severity=str(config["severity"]),
                seed=aug_seed,
                output_size=config.get("output_size"),
                platform_template=config.get("platform_template"),
            )
            aug_result = SNSAugV2Augmentor(aug_config)(image, tamper_mask, label=sample_label(sample), base_id=base_id_value, seed=aug_seed)
            saved_paths = _save_augmented_sample(output_root, split, base_id_value, str(profile), aug_result.image, aug_result.tamper_mask, aug_result.ignore_mask)
            records.append(_evaluate_augmented_record(bundle, config, sample, split, base_id_value, str(profile), saved_paths, index, aug_result.meta))
    return records


def join_clean_and_profiles(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_lookup = {(row["split"], row["base_id"]): row for row in records if row["profile"] == "clean"}
    joined: list[dict[str, Any]] = []
    for row in records:
        clean = clean_lookup.get((row["split"], row["base_id"]))
        if clean is None:
            continue
        clean_iou = clean.get("final_iou")
        profile_iou = row.get("final_iou")
        clean_dice = clean.get("final_dice")
        profile_dice = row.get("final_dice")
        joined_row = {
            "base_id": row["base_id"],
            "split": row["split"],
            "label": row["label"],
            "profile": row["profile"],
            "seed": row["seed"],
            "clean_pred": clean.get("pred_class"),
            "profile_pred": row.get("pred_class"),
            "clean_correct": clean.get("class_correct"),
            "profile_correct": row.get("class_correct"),
            "p_real_clean": clean.get("p_real"),
            "p_synthetic_clean": clean.get("p_synthetic"),
            "p_tampered_clean": clean.get("p_tampered"),
            "p_real_profile": row.get("p_real"),
            "p_synthetic_profile": row.get("p_synthetic"),
            "p_tampered_profile": row.get("p_tampered"),
            "p_tampered_drop": float(clean.get("p_tampered") or 0.0) - float(row.get("p_tampered") or 0.0),
            "clean_iou": clean_iou,
            "profile_iou": profile_iou,
            "iou_drop": (float(clean_iou) - float(profile_iou)) if clean_iou is not None and profile_iou is not None else None,
            "clean_dice": clean_dice,
            "profile_dice": profile_dice,
            "dice_drop": (float(clean_dice) - float(profile_dice)) if clean_dice is not None and profile_dice is not None else None,
            "clean_localization_activated": clean.get("localization_activated"),
            "profile_localization_activated": row.get("localization_activated"),
            "activation_flip_off": bool(clean.get("localization_activated")) and not bool(row.get("localization_activated")),
            "pred_flip": clean.get("pred_class") != row.get("pred_class"),
            "correct_to_wrong": bool(clean.get("class_correct")) and not bool(row.get("class_correct")),
            "ignore_mask_area_pct": row.get("ignore_mask_area_pct"),
            "overlay_area_pct": row.get("overlay_area_pct"),
            "final_mask_source": row.get("final_mask_source"),
            "latency_ms": row.get("latency_ms"),
            "fps": row.get("fps"),
            "source_image_path": row.get("source_image_path"),
            "transformed_image_path": row.get("transformed_image_path"),
            "transformed_mask_path": row.get("transformed_mask_path"),
            "ignore_mask_path": row.get("ignore_mask_path"),
            "overlay_boxes": row.get("overlay_boxes", []),
        }
        joined.append(joined_row)
    return joined


def assign_mining_groups(joined_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions_by_base: dict[tuple[str, str], set[str]] = {}
    for row in joined_rows:
        if row["profile"] == "clean":
            continue
        key = (str(row["split"]), str(row["base_id"]))
        predictions_by_base.setdefault(key, set()).add(str(row.get("profile_pred")))
    out: list[dict[str, Any]] = []
    for row in joined_rows:
        key = (str(row["split"]), str(row["base_id"]))
        clean_iou = float(row.get("clean_iou") or 0.0)
        stable_correct_anchor = bool(
            row["label"] == "tampered"
            and row["clean_pred"] == "tampered"
            and row["clean_correct"] is True
            and float(row.get("p_tampered_clean") or 0.0) >= 0.75
            and clean_iou >= 0.35
        )
        fragile_correct_to_fail = bool(stable_correct_anchor and row["profile"] != "clean" and row["profile_correct"] is False)
        confidence_fragile = bool(stable_correct_anchor and row["profile"] != "clean" and float(row.get("p_tampered_drop") or 0.0) >= 0.25)
        mask_iou_fragile = bool(stable_correct_anchor and row["profile"] != "clean" and row.get("iou_drop") is not None and float(row["iou_drop"]) >= 0.20)
        threshold_flip = bool(row.get("clean_localization_activated") is True and row.get("profile_localization_activated") is False)
        clean_fail = bool(row["label"] == "tampered" and row["profile"] == "clean" and row["clean_correct"] is False)
        noisy_candidate = bool(
            row["label"] == "tampered"
            and clean_iou < 0.10
            and (
                len(predictions_by_base.get(key, set())) >= 2
                or sum(1 for pred in predictions_by_base.get(key, set()) if pred != row.get("clean_pred")) >= 1
            )
        )
        groups = []
        if stable_correct_anchor:
            groups.append("stable_correct_anchor")
        if fragile_correct_to_fail:
            groups.append("fragile_correct_to_fail")
        if confidence_fragile:
            groups.append("confidence_fragile")
        if mask_iou_fragile:
            groups.append("mask_iou_fragile")
        if threshold_flip:
            groups.append("threshold_flip")
        if clean_fail:
            groups.append("clean_fail")
        if noisy_candidate:
            groups.append("noisy_candidate")
        out.append(
            {
                **row,
                "stable_correct_anchor": stable_correct_anchor,
                "fragile_correct_to_fail": fragile_correct_to_fail,
                "confidence_fragile": confidence_fragile,
                "mask_iou_fragile": mask_iou_fragile,
                "threshold_flip": threshold_flip,
                "clean_fail": clean_fail,
                "noisy_candidate": noisy_candidate,
                "groups": groups,
            }
        )
    return out


def aggregate_per_profile(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_profile.setdefault(str(row["profile"]), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for profile, items in by_profile.items():
        class_rows = [{"label": row["label"], "pred_class": row["profile_pred"], "class_correct": row["profile_correct"]} for row in items]
        matrix = confusion_matrix(class_rows)
        cls = macro_f1_and_details(matrix)
        tampered = [row for row in items if row["label"] == "tampered"]
        real = [row for row in items if row["label"] == "real"]
        latencies = [float(row["latency_ms"]) for row in items if row.get("latency_ms") is not None]
        iou_values = [float(row["profile_iou"]) for row in tampered if row.get("profile_iou") is not None]
        dice_values = [float(row["profile_dice"]) for row in tampered if row.get("profile_dice") is not None]
        out[profile] = {
            "sample_count": len(items),
            "class_accuracy": sum(1 for row in items if row["profile_correct"]) / len(items) if items else 0.0,
            "macro_f1": cls["macro_f1"],
            "confusion_matrix": matrix,
            "real_false_positive_rate": (sum(1 for row in real if row["profile_pred"] != "real") / len(real)) if real else 0.0,
            "synthetic_recall": cls["per_class"]["synthetic"]["recall"],
            "tampered_recall": cls["per_class"]["tampered"]["recall"],
            "tampered_mean_iou": _mean(iou_values),
            "tampered_median_iou": _median(iou_values),
            "tampered_mean_dice": _mean(dice_values),
            "localization_activation_recall": (sum(1 for row in tampered if row.get("profile_localization_activated")) / len(tampered)) if tampered else None,
            "mean_p_tampered_drop_on_tampered": _mean([float(row["p_tampered_drop"]) for row in tampered if row["profile"] != "clean"]),
            "mean_iou_drop": _mean([float(row["iou_drop"]) for row in tampered if row.get("iou_drop") is not None and row["profile"] != "clean"]),
            "fragile_sample_count": sum(1 for row in items if row.get("fragile_correct_to_fail") or row.get("confidence_fragile") or row.get("mask_iou_fragile")),
            "threshold_flip_count": sum(1 for row in items if row.get("threshold_flip")),
            "mean_latency_ms": _mean(latencies),
            "fps": (1000.0 / _mean(latencies)) if latencies and _mean(latencies) not in {None, 0.0} else None,
        }
    return out


def sort_worst_profiles(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    clean = metrics.get("clean", {})
    items = []
    for profile, row in metrics.items():
        if profile == "clean":
            continue
        items.append(
            {
                "profile": profile,
                "tampered_recall_drop": float(clean.get("tampered_recall") or 0.0) - float(row.get("tampered_recall") or 0.0),
                "mean_iou_drop": float(clean.get("tampered_mean_iou") or 0.0) - float(row.get("tampered_mean_iou") or 0.0),
                "macro_f1_drop": float(clean.get("macro_f1") or 0.0) - float(row.get("macro_f1") or 0.0),
            }
        )
    items.sort(key=lambda item: (-item["tampered_recall_drop"], -item["mean_iou_drop"], -item["macro_f1_drop"], item["profile"]))
    return items


def sort_worst_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = [row for row in rows if row["profile"] != "clean"]
    items.sort(
        key=lambda row: (
            0 if row.get("fragile_correct_to_fail") else 1,
            -float(row.get("p_tampered_drop") or 0.0),
            -float(row.get("iou_drop") or 0.0),
            0 if row.get("pred_flip") else 1,
        )
    )
    return items


def build_training_mining_manifest(train_rows: list[dict[str, Any]], val_base_ids: set[str]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        base_id_value = str(row["base_id"])
        if base_id_value in val_base_ids:
            continue
        if row["split"] != "train":
            continue
        entry = grouped.setdefault(
            base_id_value,
            {
                "base_id": base_id_value,
                "split": "train",
                "label": row["label"],
                "source_image_path": row.get("source_image_path"),
                "groups": set(),
            },
        )
        for group in row.get("groups", []):
            entry["groups"].add(group)
    manifest = []
    for value in grouped.values():
        groups = sorted(str(group) for group in value["groups"])
        if not groups:
            continue
        manifest.append({**value, "groups": groups, "label_preserved": True})
    manifest.sort(key=lambda row: row["base_id"])
    return manifest


def write_visual_gallery(output_root: Path, rows: list[dict[str, Any]], top_n: int) -> dict[str, Any]:
    items = []
    for row in sort_worst_samples(rows)[:top_n]:
        items.append(
            {
                "base_id": row["base_id"],
                "split": row["split"],
                "profile": row["profile"],
                "image_path": row.get("transformed_image_path"),
                "ignore_mask_path": row.get("ignore_mask_path"),
            }
        )
    return {"marker": MARKER, "record_count": len(items), "items": items}


def run_snsaug_v2_robustness_eval(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_exists=True)
    output_root = _real(config["output_root"])
    if _inside_repo(output_root):
        raise SNSAugV2RobustnessEvalError("output_root must be outside repository")
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = load_best_bundle(config["best_bundle_path"])

    all_records = evaluate_manifest(bundle, config, config["validation_manifest_path"], "val", output_root)
    if config.get("train_manifest_path"):
        all_records.extend(evaluate_manifest(bundle, config, config["train_manifest_path"], "train", output_root))

    joined = assign_mining_groups(join_clean_and_profiles(all_records))
    val_rows = [row for row in joined if row["split"] == "val"]
    train_rows = [row for row in joined if row["split"] == "train"]
    per_profile_metrics = aggregate_per_profile(val_rows)
    worst_profiles = sort_worst_profiles(per_profile_metrics)
    worst_samples = sort_worst_samples(val_rows)
    fragile_val = [row for row in val_rows if any(group in row["groups"] for group in ("fragile_correct_to_fail", "confidence_fragile", "mask_iou_fragile", "threshold_flip", "clean_fail", "noisy_candidate"))]
    fragile_train = [row for row in train_rows if any(group in row["groups"] for group in ("fragile_correct_to_fail", "confidence_fragile", "mask_iou_fragile", "threshold_flip", "clean_fail", "noisy_candidate"))]
    training_manifest = build_training_mining_manifest(train_rows, {str(row["base_id"]) for row in val_rows})
    gallery = write_visual_gallery(output_root, val_rows, int(config.get("top_n_worst_cases", 8)))

    paths = {
        "records": _write_jsonl(output_root / "snsaug_v2_robustness_records.jsonl", joined),
        "per_profile_metrics": _write_json(output_root / "snsaug_v2_per_profile_metrics.json", per_profile_metrics),
        "worst_profiles": _write_json(output_root / "snsaug_v2_worst_profiles.json", worst_profiles),
        "worst_samples": _write_json(output_root / "snsaug_v2_worst_samples.json", worst_samples),
        "fragile_candidates_val": _write_jsonl(output_root / "snsaug_v2_fragile_candidates_val.jsonl", fragile_val),
        "visual_gallery_manifest": _write_json(output_root / "visual_gallery_manifest.json", gallery),
    }
    if train_rows:
        paths["fragile_candidates_train"] = _write_jsonl(output_root / "snsaug_v2_fragile_candidates_train.jsonl", fragile_train)
        paths["training_mining_manifest_train_only"] = _write_jsonl(output_root / "snsaug_v2_training_mining_manifest_train_only.jsonl", training_manifest)
    summary = {
        "marker": MARKER,
        "validation_record_count": len(val_rows),
        "train_record_count": len(train_rows),
        "fragile_val_count": len(fragile_val),
        "fragile_train_count": len(fragile_train),
        "profiles": config["profiles"],
        "severity": config["severity"],
        "worst_profiles": worst_profiles[:5],
        "output_paths": paths,
        "no_training": True,
        "no_download": True,
        "no_network": True,
    }
    paths["summary"] = _write_json(output_root / "snsaug_v2_robustness_summary.json", summary)
    artifact = {
        "marker": MARKER,
        "output_paths": paths,
        "no_training": True,
        "no_download": True,
        "no_network": True,
    }
    paths["artifact_manifest"] = _write_json(output_root / "artifact_manifest.json", artifact)
    return summary
