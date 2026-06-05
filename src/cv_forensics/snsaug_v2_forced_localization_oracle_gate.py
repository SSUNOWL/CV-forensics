"""Evaluation-only SNSAug V2 forced-localization oracle-gate diagnostic."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .pre_sns_v3_sns_robustness_eval import _mean, _median, _safe_prob, json_safe, load_best_bundle, normalize_label
from .pre_sns_v3_v2_policy_gated_report import build_policy_gated_record
from .snsaug_v2_fixed_pairs_eval import (
    CLASS_LABELS,
    NON_TAMPERED_HIGH_MASK_THRESHOLD_PCT,
    _export_visual_artifacts,
    _inside_repo,
    _load_mask,
    _real,
    _runtime_deps,
    compute_mask_metrics,
    join_clean_and_sns,
    load_meta_rows,
    parse_fixed_pair_rows,
    worst_samples,
)

MARKER = "SNSAUG_V2_FORCED_LOCALIZATION_ORACLE_GATE_OK"
DEFAULT_THRESHOLDS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50)
MODES = ("normal_gate", "oracle_tampered_gate")


class SNSAugV2ForcedOracleGateError(ValueError):
    """Raised when forced-localization oracle-gate diagnostic fails."""


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


def load_snsaug_v2_forced_oracle_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2ForcedOracleGateError("forced oracle gate config root must be a JSON object")
    return raw


def validate_snsaug_v2_forced_oracle_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
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
            errors.append(f"- {field} is required")
    for flag in ("no_training", "no_finetune", "no_network", "no_download"):
        if raw.get(flag) is not True:
            errors.append(f"- {flag} must be true")
    for field in ("best_bundle_path", "pair_root", "meta_jsonl_path"):
        value = raw.get(field)
        if not isinstance(value, str) or not value.startswith("/"):
            errors.append(f"- {field} must be an absolute local path")
        elif require_exists and not Path(value).exists():
            errors.append(f"- {field} must exist")
    output_root = raw.get("output_root")
    if not isinstance(output_root, str) or not output_root.startswith("/"):
        errors.append("- output_root must be an absolute local path")
    elif _inside_repo(output_root):
        errors.append("- output_root must be outside repository")
    profiles = raw.get("profiles")
    if not isinstance(profiles, list) or not profiles or not all(isinstance(item, str) for item in profiles):
        errors.append("- profiles must be a non-empty list of strings")
    if raw.get("device") not in {"cpu", "cuda"}:
        errors.append("- device must be cpu or cuda")
    if "max_samples" in raw and raw.get("max_samples") is not None:
        value = raw.get("max_samples")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append("- max_samples must be a positive integer when present")
    thresholds = raw.get("thresholds", list(DEFAULT_THRESHOLDS))
    if not isinstance(thresholds, list) or not thresholds:
        errors.append("- thresholds must be a non-empty list")
    else:
        for value in thresholds:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                errors.append("- thresholds must contain numbers in [0, 1]")
                break
    if require_exists:
        roots = raw.get("approved_input_roots")
        if not isinstance(roots, list) or not roots:
            errors.append("- approved_input_roots must be a non-empty list")
        output_roots = raw.get("approved_output_roots")
        if not isinstance(output_roots, list) or not output_roots:
            errors.append("- approved_output_roots must be a non-empty list")
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_forced_oracle_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2ForcedOracleGateError("forced oracle gate config validation failed:\n" + "\n".join(errors))


def _policy_config_from_bundle(bundle: dict[str, Any], config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    policy = bundle.get("policy_gated_report", {})
    checkpoint_roots = [str(Path(bundle["long256_checkpoint_path"]).parent), str(Path(bundle["tile_v2_checkpoint_path"]).parent)]
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_v3_v2_policy_gated_report",
        "execution_mode": "approved_local_pre_sns_v3_v2_policy_gated_report",
        "long256_checkpoint_path": bundle["long256_checkpoint_path"],
        "tile_localizer_v2_checkpoint_path": bundle["tile_v2_checkpoint_path"],
        "approved_input_roots": list(config.get("approved_input_roots", [])),
        "approved_checkpoint_roots": checkpoint_roots,
        "output_root": str(output_root),
        "device": config.get("device", "cpu"),
        "mask_threshold": float(policy.get("mask_threshold", 0.45)),
        "min_area_pct": float(policy.get("min_area_pct", 0.1)),
        "max_area_pct": float(policy.get("max_area_pct", 100.0)),
        "suppress_non_tampered_mask": bool(policy.get("suppress_non_tampered_mask", True)),
        "fallback_to_baseline_on_v2_unreliable": bool(policy.get("fallback_to_baseline_on_v2_unreliable", True)),
        "tile_size": int(policy.get("tile_size", 768)),
        "tile_stride": int(policy.get("tile_stride", 384)),
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }


def _evaluate_row(bundle: dict[str, Any], config: dict[str, Any], row: dict[str, Any], index: int, mode: str, output_root: Path) -> dict[str, Any]:
    Image = _runtime_deps()
    policy_config = _policy_config_from_bundle(bundle, config, output_root)
    sample = dict(row)
    sample["image_path"] = str(row["image_path"])
    if row.get("tamper_mask_path"):
        sample["gt_mask_path"] = str(row["tamper_mask_path"])
    if mode == "oracle_tampered_gate" and normalize_label(row.get("content_label")) == "tampered":
        sample["force_tile_localization"] = True
    started = time.perf_counter()
    result = build_policy_gated_record(policy_config, sample, index)
    latency_ms = (time.perf_counter() - started) * 1000.0
    pred_mask = result.get("_final_mask")
    gt_mask = _load_mask(Image, str(row.get("tamper_mask_path"))) if row.get("tamper_mask_path") else None
    ignore_mask = _load_mask(Image, str(row.get("ignore_mask_path"))) if row.get("ignore_mask_path") else None
    metrics = compute_mask_metrics(pred_mask, gt_mask, ignore_mask)
    label = normalize_label(row.get("content_label"))
    pred_class = normalize_label(result.get("class"))
    localization_activated = bool(result.get("tile_localization_activated"))
    visual_paths = _export_visual_artifacts(
        Image=Image,
        config={**config, "output_root": str(output_root), "write_empty_pred_mask": True},
        row=row,
        index=index,
        pred_mask=pred_mask,
        gt_mask=gt_mask,
        ignore_mask=ignore_mask,
        localization_activated=localization_activated,
    )
    return {
        "mode": mode,
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
        "localization_activated": localization_activated,
        "tile_localization_forced": bool(result.get("tile_localization_forced")),
        **visual_paths,
        "final_mask_source": result.get("final_mask_source"),
        "final_mask_area_pct": float(result.get("final_mask_area_pct", 0.0)),
        "raw_iou": metrics["raw_iou"],
        "raw_dice": metrics["raw_dice"],
        "valid_iou": metrics["valid_iou"],
        "valid_dice": metrics["valid_dice"],
        "latency_ms": float(latency_ms),
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
        "_result": result,
    }


def _aggregate_mode_profile(records: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    comparisons_by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in comparisons:
        comparisons_by_profile.setdefault(str(row["profile"]), []).append(row)
    for profile in sorted({str(row["profile"]) for row in records}):
        items = [row for row in records if row["profile"] == profile]
        real_items = [row for row in items if row["content_label"] == "real"]
        synthetic_items = [row for row in items if row["content_label"] == "synthetic"]
        tampered_items = [row for row in items if row["content_label"] == "tampered"]
        valid_ious = [float(row["valid_iou"]) for row in tampered_items if row.get("valid_iou") is not None]
        raw_ious = [float(row["raw_iou"]) for row in tampered_items if row.get("raw_iou") is not None]
        tampered_scores = [float(row["p_tampered"]) for row in tampered_items if row.get("p_tampered") is not None]
        comps = comparisons_by_profile.get(profile, [])
        out[profile] = {
            "sample_count": len(items),
            "tampered_recall": (
                sum(1 for row in tampered_items if row.get("pred_class") == "tampered") / len(tampered_items)
            ) if tampered_items else None,
            "localization_activation_recall": (
                sum(1 for row in tampered_items if row.get("localization_activated")) / len(tampered_items)
            ) if tampered_items else None,
            "tampered_valid_mean_iou": _mean(valid_ious),
            "tampered_valid_median_iou": _median(valid_ious),
            "tampered_raw_mean_iou": _mean(raw_ious),
            "mean_p_tampered_on_tampered": _mean(tampered_scores),
            "activation_flip_off_rate": (
                sum(1 for row in comps if row.get("activation_flip_off")) / len(comps)
            ) if comps else None,
            "class_flip_rate": (
                sum(1 for row in comps if row.get("pred_flip")) / len(comps)
            ) if comps else None,
            "non_tampered_high_mask_rate": (
                sum(1 for row in items if row["content_label"] != "tampered" and float(row.get("final_mask_area_pct") or 0.0) > NON_TAMPERED_HIGH_MASK_THRESHOLD_PCT)
                / len([row for row in items if row["content_label"] != "tampered"])
            ) if any(row["content_label"] != "tampered" for row in items) else None,
            "real_fpr": (
                sum(1 for row in real_items if row.get("pred_class") != "real") / len(real_items)
            ) if real_items else None,
            "synthetic_recall": (
                sum(1 for row in synthetic_items if row.get("pred_class") == "synthetic") / len(synthetic_items)
            ) if synthetic_items else None,
        }
    return out


def _mode_metrics(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_mode.setdefault(str(row["mode"]), []).append(row)
    metrics: dict[str, dict[str, Any]] = {}
    all_comparisons: list[dict[str, Any]] = []
    for mode, mode_records in by_mode.items():
        comps = join_clean_and_sns(mode_records)
        for comp in comps:
            comp["mode"] = mode
        all_comparisons.extend(comps)
        metrics[mode] = _aggregate_mode_profile(mode_records, comps)
    return metrics, all_comparisons


def _drop_metrics(per_profile: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for mode, profiles in per_profile.items():
        clean = profiles.get("clean", {})
        out[mode] = {}
        for profile, metrics in profiles.items():
            if profile == "clean":
                continue
            out[mode][profile] = {
                "tampered_recall_drop": _delta(clean.get("tampered_recall"), metrics.get("tampered_recall")),
                "localization_activation_recall_drop": _delta(clean.get("localization_activation_recall"), metrics.get("localization_activation_recall")),
                "valid_iou_drop": _delta(clean.get("tampered_valid_mean_iou"), metrics.get("tampered_valid_mean_iou")),
                "raw_iou_drop": _delta(clean.get("tampered_raw_mean_iou"), metrics.get("tampered_raw_mean_iou")),
            }
    return out


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _threshold_sweep(records: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    profiles = sorted({str(row["profile"]) for row in records})
    for threshold in thresholds:
        key = f"{threshold:.2f}"
        out[key] = {}
        for profile in profiles:
            items = [row for row in records if row["profile"] == profile]
            real_items = [row for row in items if row["content_label"] == "real"]
            tampered_items = [row for row in items if row["content_label"] == "tampered"]
            activated_tampered = [row for row in tampered_items if float(row.get("p_tampered") or 0.0) >= threshold]
            out[key][profile] = {
                "threshold": threshold,
                "localization_activation_recall": (len(activated_tampered) / len(tampered_items)) if tampered_items else None,
                "real_fpr": (
                    sum(1 for row in real_items if float(row.get("p_tampered") or 0.0) >= threshold) / len(real_items)
                ) if real_items else None,
                "tampered_valid_mean_iou_if_activated": _mean([float(row["valid_iou"]) for row in activated_tampered if row.get("valid_iou") is not None]),
            }
    return out


def _interpret(per_profile: dict[str, dict[str, Any]], threshold_sweep: dict[str, Any]) -> dict[str, Any]:
    normal_values = [float(metrics["tampered_valid_mean_iou"]) for metrics in per_profile.get("normal_gate", {}).values() if metrics.get("tampered_valid_mean_iou") is not None]
    oracle_values = [float(metrics["tampered_valid_mean_iou"]) for metrics in per_profile.get("oracle_tampered_gate", {}).values() if metrics.get("tampered_valid_mean_iou") is not None]
    normal_mean = _mean(normal_values)
    oracle_mean = _mean(oracle_values)
    if oracle_mean is not None and normal_mean is not None and oracle_mean > normal_mean + 0.05:
        failure_type = "class_activation_bottleneck"
    elif oracle_mean is not None and oracle_mean <= 0.05:
        failure_type = "mask_decoder_geometry_bottleneck"
    else:
        failure_type = "mixed_or_unclear"
    tradeoff = False
    clean_high = threshold_sweep.get("0.50", {}).get("clean", {})
    clean_low = threshold_sweep.get("0.01", {}).get("clean", {})
    if clean_high.get("localization_activation_recall") is not None and clean_low.get("localization_activation_recall") is not None:
        activation_gain = float(clean_low["localization_activation_recall"]) - float(clean_high["localization_activation_recall"])
        fpr_gain = float(clean_low.get("real_fpr") or 0.0) - float(clean_high.get("real_fpr") or 0.0)
        tradeoff = activation_gain > 0.0 and fpr_gain > 0.0
    return {
        "failure_type": failure_type,
        "normal_mean_valid_iou": normal_mean,
        "oracle_mean_valid_iou": oracle_mean,
        "threshold_tradeoff_detected": tradeoff,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def run_snsaug_v2_forced_oracle_gate(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_exists=True)
    output_root = _real(config["output_root"])
    if _inside_repo(output_root):
        raise SNSAugV2ForcedOracleGateError("output_root must be outside repository")
    output_root.mkdir(parents=True, exist_ok=True)
    rows, warnings = parse_fixed_pair_rows(load_meta_rows(config["meta_jsonl_path"]), list(config["profiles"]), config.get("max_samples"))
    bundle = load_best_bundle(config["best_bundle_path"])
    thresholds = [float(value) for value in config.get("thresholds", list(DEFAULT_THRESHOLDS))]
    records: list[dict[str, Any]] = []
    for mode_index, mode in enumerate(MODES):
        for row_index, row in enumerate(rows):
            records.append(_evaluate_row(bundle, config, row, row_index + mode_index * 100000, mode, output_root))
    per_profile, comparisons = _mode_metrics(records)
    drops = _drop_metrics(per_profile)
    sweep_source = [row for row in records if row["mode"] == "oracle_tampered_gate"]
    threshold_sweep = _threshold_sweep(sweep_source, thresholds)
    interpretation = _interpret(per_profile, threshold_sweep)
    worst = worst_samples([row for row in comparisons if row["mode"] == "oracle_tampered_gate"])
    visual_gallery = {
        "top_worst_cases": [
            {
                "base_id": row["base_id"],
                "profile": row["profile"],
                "content_label": row["content_label"],
                "sns_valid_iou": row.get("sns_valid_iou"),
            }
            for row in worst[: min(12, len(worst))]
        ]
    }
    artifact_manifest = {
        "marker": MARKER,
        "records_jsonl": str(output_root / "oracle_gate_eval_records.jsonl"),
        "comparisons_jsonl": str(output_root / "oracle_gate_eval_comparisons.jsonl"),
        "per_profile_metrics_json": str(output_root / "oracle_gate_per_profile_metrics.json"),
        "drop_metrics_json": str(output_root / "oracle_gate_drop_metrics.json"),
        "threshold_sweep_metrics_json": str(output_root / "threshold_sweep_metrics.json"),
        "activation_bottleneck_summary_json": str(output_root / "activation_bottleneck_summary.json"),
        "forced_localization_worst_samples_json": str(output_root / "forced_localization_worst_samples.json"),
        "visual_gallery_manifest_json": str(output_root / "visual_gallery_manifest.json"),
        "pred_masks_dir": str(output_root / "pred_masks"),
        "pred_red_overlays_dir": str(output_root / "pred_red_overlays"),
        "gt_red_overlays_dir": str(output_root / "gt_red_overlays"),
        "ignore_blue_overlays_dir": str(output_root / "ignore_blue_overlays"),
        "overlap_overlays_dir": str(output_root / "overlap_overlays"),
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    public_records = [{k: v for k, v in row.items() if not str(k).startswith("_")} for row in records]
    _write_jsonl(output_root / "oracle_gate_eval_records.jsonl", public_records)
    _write_jsonl(output_root / "oracle_gate_eval_comparisons.jsonl", comparisons)
    _write_json(output_root / "oracle_gate_per_profile_metrics.json", per_profile)
    _write_json(output_root / "oracle_gate_drop_metrics.json", drops)
    _write_json(output_root / "threshold_sweep_metrics.json", threshold_sweep)
    _write_json(output_root / "activation_bottleneck_summary.json", interpretation)
    _write_json(output_root / "forced_localization_worst_samples.json", worst)
    _write_json(output_root / "visual_gallery_manifest.json", visual_gallery)
    _write_json(output_root / "artifact_manifest.json", artifact_manifest)
    return {
        "marker": MARKER,
        "record_count": len(records),
        "comparison_count": len(comparisons),
        "warning_count": len(warnings),
        "warnings": warnings,
        "output_root": str(output_root),
        "failure_type": interpretation["failure_type"],
        "artifact_manifest_json": str(output_root / "artifact_manifest.json"),
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
