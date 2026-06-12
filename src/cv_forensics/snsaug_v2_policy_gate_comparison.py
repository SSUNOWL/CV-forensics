"""SNSAug v2 residual-vs-geometry policy gate comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MARKER = "SNSAUG_V2_POLICY_GATE_COMPARISON_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_POLICY_GATE_COMPARISON_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_policy_gate_comparison"
APPROVED_MODE = "approved_local_snsaug_v2_policy_gate_comparison"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_POLICY_GATE_COMPARISON"
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SELECTORS = [
    "fixed_original",
    "geometry_feature_gate",
    "residual_dct_feature_gate",
    "mixed_feature_gate",
    "oracle_best_policy_diagnostic",
]
DEFAULT_FOCUS_PROFILES = [
    "canvas_9x16_only",
    "resize_crop_pad",
    "zoom_crop",
    "recompression_light",
    "resize_jpeg",
    "screenshot_recapture_light",
    "news_meme_overlay",
    "platform_ui_same_size",
    "tiktok_like",
    "instagram_story_like",
    "youtube_shorts_like",
    "combined_sns_realistic",
]
GEOMETRY_FEATURES = ("crop_scale_proxy", "area_ratio_delta_abs", "aspect_ratio_delta_abs")
RESIDUAL_FEATURES = (
    "dct_total_energy_delta_abs",
    "dct_high_energy_delta_abs",
    "dct_low_energy_delta_abs",
    "dct_high_low_ratio_delta_abs",
    "histogram_l1",
    "highpass_energy_delta_abs",
    "srm_energy_delta_abs",
    "blockiness_delta_abs",
    "laplacian_variance_delta_abs",
    "sobel_edge_energy_delta_abs",
)


class SNSAugV2PolicyGateComparisonError(ValueError):
    """Raised when policy gate comparison validation fails."""


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
    errors = []
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


def load_snsaug_v2_policy_gate_comparison_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2PolicyGateComparisonError("policy gate comparison config root must be a JSON object")
    raw.setdefault("config_path", str(_real(path)))
    return raw


def validate_snsaug_v2_policy_gate_comparison_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    for field in ("schema_version", "config_kind", "execution_mode", "user_approval_text", "residual_records_path", "approved_input_roots", "approved_output_roots", "output_root", "no_training", "no_finetune", "no_network", "no_download"):
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
    input_roots = _roots(raw.get("approved_input_roots"))
    output_roots = _roots(raw.get("approved_output_roots"))
    if not input_roots:
        errors.append("approved_input_roots must be a non-empty list of absolute paths")
    if not output_roots:
        errors.append("approved_output_roots must be a non-empty list of absolute paths")
    for index, root in enumerate(input_roots):
        errors.extend(_abs_errors(root, f"approved_input_roots[{index}]"))
    for index, root in enumerate(output_roots):
        errors.extend(_abs_errors(root, f"approved_output_roots[{index}]"))
    for field in ("residual_records_path", "residual_feature_summary_path", "residual_feature_response_correlation_path"):
        if raw.get(field):
            errors.extend(_under_errors(raw.get(field), field, input_roots, require_exists=require_exists))
    output_root = raw.get("output_root")
    errors.extend(_abs_errors(output_root, "output_root"))
    if isinstance(output_root, str) and output_root.strip():
        if _inside_repo(output_root):
            errors.append("output_root must be outside repository")
        if any(_is_under(output_root, root) or _real(output_root) == _real(root) for root in input_roots):
            errors.append("output_root must not be under approved input roots")
        if output_roots and not any(_is_under(output_root, root) or _real(output_root) == _real(root) for root in output_roots):
            errors.append("output_root must be under approved output roots")
    selectors = raw.get("selectors", DEFAULT_SELECTORS)
    if not isinstance(selectors, list) or "fixed_original" not in selectors:
        errors.append("selectors must be a list including fixed_original")
    profiles = raw.get("focus_profiles", DEFAULT_FOCUS_PROFILES)
    if not isinstance(profiles, list) or not profiles:
        errors.append("focus_profiles must be a non-empty list")
    if raw.get("success_criteria", {}) is not None and not isinstance(raw.get("success_criteria", {}), dict):
        errors.append("success_criteria must be an object")
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_policy_gate_comparison_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2PolicyGateComparisonError("policy gate comparison config validation failed:\n" + "\n".join(errors))


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


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _num(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else default


def _pct(values: list[float], q: float) -> float:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return 0.0
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _mean(values: list[Any]) -> float | None:
    xs = [_num(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return sum(xs) / len(xs) if xs else None


def _score(row: dict[str, Any], features: tuple[str, ...], scales: dict[str, float]) -> float:
    vals = [_num(row.get(feature)) / (scales.get(feature) or 1.0) for feature in features]
    return sum(vals) / len(vals) if vals else 0.0


def _valid_iou(row: dict[str, Any], source: str) -> float | None:
    value = row.get(f"{source}_valid_iou")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _choose_source(row: dict[str, Any], selector: str, thresholds: dict[str, float]) -> str:
    if selector == "fixed_original":
        return "sns"
    if selector == "oracle_best_policy_diagnostic":
        label = row.get("content_label")
        clean_iou = _valid_iou(row, "clean") or 0.0
        sns_iou = _valid_iou(row, "sns") or 0.0
        clean_p = _num(row.get("p_tampered_clean"))
        sns_p = _num(row.get("p_tampered_sns"))
        if label == "tampered":
            return "clean" if (clean_iou, clean_p) >= (sns_iou, sns_p) else "sns"
        return "clean" if abs(clean_p) <= abs(sns_p) else "sns"
    geometry = _num(row.get("_geometry_gate_score")) >= thresholds.get("geometry", 0.0)
    residual = _num(row.get("_residual_gate_score")) >= thresholds.get("residual", 0.0)
    if selector == "geometry_feature_gate":
        return "clean" if geometry else "sns"
    if selector == "residual_dct_feature_gate":
        return "clean" if residual else "sns"
    if selector == "mixed_feature_gate":
        return "clean" if geometry or residual else "sns"
    return "sns"


def _selector_provenance(row: dict[str, Any], selector: str, source: str, thresholds: dict[str, float]) -> dict[str, Any]:
    geometry_score = _num(row.get("_geometry_gate_score"))
    residual_score = _num(row.get("_residual_gate_score"))
    geometry_pass = geometry_score >= thresholds.get("geometry", 0.0)
    residual_pass = residual_score >= thresholds.get("residual", 0.0)
    diagnostic = selector.startswith("oracle") or selector.endswith("diagnostic")
    if selector == "fixed_original":
        chosen_policy = "original"
        reason = "fixed_original_selector"
    elif selector == "oracle_best_policy_diagnostic":
        chosen_policy = "oracle_clean_policy" if source == "clean" else "original"
        reason = "oracle_best_clean_or_original_by_label_iou_and_p_tampered"
    elif selector == "geometry_feature_gate":
        chosen_policy = "geometry_feature_clean_policy" if source == "clean" else "original"
        reason = "geometry_score_passed_threshold" if source == "clean" else "geometry_score_below_threshold"
    elif selector == "residual_dct_feature_gate":
        chosen_policy = "residual_dct_clean_policy" if source == "clean" else "original"
        reason = "residual_score_passed_threshold" if source == "clean" else "residual_score_below_threshold"
    elif selector == "mixed_feature_gate":
        if source != "clean":
            chosen_policy = "original"
            reason = "geometry_and_residual_scores_below_threshold"
        elif geometry_pass and residual_pass:
            chosen_policy = "mixed_geometry_residual_clean_policy"
            reason = "geometry_and_residual_scores_passed_threshold"
        elif geometry_pass:
            chosen_policy = "geometry_feature_clean_policy"
            reason = "geometry_score_passed_threshold"
        else:
            chosen_policy = "residual_dct_clean_policy"
            reason = "residual_score_passed_threshold"
    else:
        chosen_policy = "original" if source == "sns" else f"{selector}_clean_policy"
        reason = "fallback_selector_mapping"
    return {
        "chosen_policy": chosen_policy,
        "selector_reason": reason,
        "diagnostic_only_selector": diagnostic,
        "geometry_score": geometry_score,
        "residual_score": residual_score,
        "selected_by_profile_family": row.get("profile_family"),
    }


def build_policy_gate_records(records: list[dict[str, Any]], selectors: list[str] | None = None) -> list[dict[str, Any]]:
    selectors = selectors or DEFAULT_SELECTORS
    scales = {feature: max(_pct([_num(row.get(feature)) for row in records], 0.75), 1e-9) for feature in GEOMETRY_FEATURES + RESIDUAL_FEATURES}
    enriched = []
    for row in records:
        item = dict(row)
        item["_geometry_gate_score"] = _score(item, GEOMETRY_FEATURES, scales)
        item["_residual_gate_score"] = _score(item, RESIDUAL_FEATURES, scales)
        enriched.append(item)
    thresholds = {
        "geometry": _pct([_num(row.get("_geometry_gate_score")) for row in enriched], 0.60),
        "residual": _pct([_num(row.get("_residual_gate_score")) for row in enriched], 0.60),
    }
    out = []
    for row in enriched:
        for selector in selectors:
            source = _choose_source(row, selector, thresholds)
            provenance = _selector_provenance(row, selector, source, thresholds)
            p_tampered = _num(row.get(f"p_tampered_{source}"))
            out.append(
                {
                    "marker": MARKER,
                    "selector": selector,
                    "selected_source": source,
                    **provenance,
                    "base_id": row.get("base_id"),
                    "profile": row.get("profile"),
                    "profile_family": row.get("profile_family"),
                    "content_label": row.get("content_label"),
                    "p_tampered": p_tampered,
                    "pred_tampered": p_tampered >= 0.5,
                    "valid_iou": _valid_iou(row, source),
                    "geometry_gate_score": row.get("_geometry_gate_score"),
                    "residual_gate_score": row.get("_residual_gate_score"),
                }
            )
    return out


def selector_profile_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    selectors = sorted({str(row.get("selector")) for row in rows})
    profiles = sorted({str(row.get("profile")) for row in rows})
    for selector in selectors:
        metrics[selector] = {}
        for profile in profiles:
            items = [row for row in rows if row.get("selector") == selector and row.get("profile") == profile]
            tampered = [row for row in items if row.get("content_label") == "tampered"]
            synthetic = [row for row in items if row.get("content_label") == "synthetic"]
            real = [row for row in items if row.get("content_label") == "real"]
            metrics[selector][profile] = {
                "row_count": len(items),
                "tampered_recall": _mean([1.0 if row.get("pred_tampered") else 0.0 for row in tampered]),
                "synthetic_recall": _mean([0.0 if row.get("pred_tampered") else 1.0 for row in synthetic]),
                "real_fpr": _mean([1.0 if row.get("pred_tampered") else 0.0 for row in real]),
                "tampered_valid_mean_iou": _mean([row.get("valid_iou") for row in tampered]),
                "mean_p_tampered": _mean([row.get("p_tampered") for row in items]),
            }
    return metrics


def decide_policy_gate(metrics: dict[str, dict[str, dict[str, Any]]], focus_profiles: list[str], selectors: list[str], criteria: dict[str, Any] | None = None) -> dict[str, Any]:
    criteria = criteria or {}
    min_good = int(criteria.get("min_good_profiles", 2))
    min_t_gain = float(criteria.get("min_tampered_recall_gain", 0.10))
    min_i_gain = float(criteria.get("min_valid_iou_gain", 0.05))
    min_gap = float(criteria.get("min_oracle_gap_closure", 0.40))
    max_syn_drop = float(criteria.get("max_mean_synthetic_recall_drop", 0.10))
    max_fpr_inc = float(criteria.get("max_mean_real_fpr_increase", 0.20))
    rows = []
    for selector in selectors:
        if selector == "fixed_original":
            continue
        good_profiles = []
        per_profile = {}
        for profile in focus_profiles:
            base = metrics.get("fixed_original", {}).get(profile, {})
            item = metrics.get(selector, {}).get(profile, {})
            oracle = metrics.get("oracle_best_policy_diagnostic", {}).get(profile, {})
            t_gain = None if base.get("tampered_recall") is None or item.get("tampered_recall") is None else item["tampered_recall"] - base["tampered_recall"]
            i_gain = None if base.get("tampered_valid_mean_iou") is None or item.get("tampered_valid_mean_iou") is None else item["tampered_valid_mean_iou"] - base["tampered_valid_mean_iou"]
            t_oracle = None if base.get("tampered_recall") is None or oracle.get("tampered_recall") is None else oracle["tampered_recall"] - base["tampered_recall"]
            i_oracle = None if base.get("tampered_valid_mean_iou") is None or oracle.get("tampered_valid_mean_iou") is None else oracle["tampered_valid_mean_iou"] - base["tampered_valid_mean_iou"]
            t_gap = None if not isinstance(t_oracle, (int, float)) or t_oracle <= 0 or not isinstance(t_gain, (int, float)) else max(0.0, min(1.0, t_gain / t_oracle))
            i_gap = None if not isinstance(i_oracle, (int, float)) or i_oracle <= 0 or not isinstance(i_gain, (int, float)) else max(0.0, min(1.0, i_gain / i_oracle))
            syn_change = None if base.get("synthetic_recall") is None or item.get("synthetic_recall") is None else item["synthetic_recall"] - base["synthetic_recall"]
            fpr_change = None if base.get("real_fpr") is None or item.get("real_fpr") is None else item["real_fpr"] - base["real_fpr"]
            is_good = ((t_gain is not None and t_gain >= min_t_gain) or (i_gain is not None and i_gain >= min_i_gain) or (i_gap is not None and i_gap >= min_gap)) and (syn_change is None or syn_change >= -max_syn_drop) and (fpr_change is None or fpr_change <= max_fpr_inc)
            if is_good:
                good_profiles.append(profile)
            per_profile[profile] = {"tampered_recall_gain": t_gain, "valid_iou_gain": i_gain, "tampered_recall_oracle_gap_closure": t_gap, "valid_iou_oracle_gap_closure": i_gap, "synthetic_recall_change": syn_change, "real_fpr_change": fpr_change, "is_good": is_good}
        row = {
            "selector": selector,
            "diagnostic_only": selector.startswith("oracle") or selector.endswith("diagnostic"),
            "good_profile_count": len(good_profiles),
            "good_profiles": good_profiles,
            "mean_tampered_recall_gain": _mean([x["tampered_recall_gain"] for x in per_profile.values()]),
            "mean_valid_iou_gain": _mean([x["valid_iou_gain"] for x in per_profile.values()]),
            "mean_tampered_gap_closure": _mean([x["tampered_recall_oracle_gap_closure"] for x in per_profile.values()]),
            "mean_valid_iou_gap_closure": _mean([x["valid_iou_oracle_gap_closure"] for x in per_profile.values()]),
            "mean_synthetic_recall_change": _mean([x["synthetic_recall_change"] for x in per_profile.values()]),
            "mean_real_fpr_change": _mean([x["real_fpr_change"] for x in per_profile.values()]),
            "per_profile": per_profile,
        }
        row["score"] = (row["mean_tampered_recall_gain"] or 0.0) + 2.0 * (row["mean_valid_iou_gain"] or 0.0) + 0.8 * (row["mean_valid_iou_gap_closure"] or 0.0) - max(0.0, row["mean_real_fpr_change"] or 0.0)
        rows.append(row)
    rows.sort(key=lambda row: row["score"], reverse=True)
    deployable = [row for row in rows if not row["diagnostic_only"] and row["good_profile_count"] >= min_good]
    diagnostic = [row for row in rows if row["diagnostic_only"] and row["good_profile_count"] >= min_good]
    if deployable:
        decision = "deployable_policy_gate_promising"
        recommendation = "Freeze the best deployable gate and run a final/full comparison."
    elif diagnostic:
        decision = "diagnostic_only_policy_gate"
        recommendation = "Recoverability exists, but deployable gates are insufficient; only train a residual branch if project scope requires it."
    else:
        decision = "policy_gate_not_sufficient"
        recommendation = "Stop model iteration and write the mixed geometry/degradation failure analysis."
    return {"decision": decision, "next_recommendation": recommendation, "selector_rows": rows, "deployable_candidates": deployable, "diagnostic_candidates": diagnostic}


def _paths(output_root: Path) -> dict[str, str]:
    return {
        "policy_gate_records": str(output_root / "policy_gate_records.jsonl"),
        "policy_gate_metrics": str(output_root / "policy_gate_metrics.json"),
        "policy_gate_oracle_gap_summary": str(output_root / "policy_gate_oracle_gap_summary.json"),
        "policy_gate_report": str(output_root / "policy_gate_report.md"),
        "policy_gate_comparison_report": str(output_root / "policy_gate_comparison_report.md"),
        "artifact_manifest": str(output_root / "artifact_manifest.json"),
    }


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def _report(summary: dict[str, Any], decision: dict[str, Any]) -> str:
    lines = [
        "# SNSAug V2 Policy Gate Comparison",
        "",
        MARKER,
        "",
        f"- Records: `{summary['record_count']}`",
        f"- Source records: `{summary['residual_records_path']}`",
        f"- Decision: `{decision['decision']}`",
        f"- Recommendation: `{decision['next_recommendation']}`",
        "",
        "## Selector Ranking",
        "",
        "| Selector | Diagnostic | Score | Good profiles | Mean tampered recall gain | Mean valid IoU gain | Mean valid IoU gap closure | Mean real FPR change |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in decision["selector_rows"]:
        lines.append(f"| {row['selector']} | {row['diagnostic_only']} | {_fmt(row['score'])} | {', '.join(row['good_profiles']) or '-'} | {_fmt(row['mean_tampered_recall_gain'])} | {_fmt(row['mean_valid_iou_gain'])} | {_fmt(row['mean_valid_iou_gap_closure'])} | {_fmt(row['mean_real_fpr_change'])} |")
    lines += ["", "## Interpretation", ""]
    if decision["decision"] == "deployable_policy_gate_promising":
        lines.append("A deployable residual/geometry gate improves multiple focus profiles and is worth freezing for final comparison.")
    elif decision["decision"] == "diagnostic_only_policy_gate":
        lines.append("The diagnostic oracle can recover performance, but deployable gates remain insufficient.")
    else:
        lines.append("Policy gating is not sufficient. Treat 0071 as evidence for a mixed geometry/degradation failure mode.")
    return "\n".join(lines) + "\n"


def run_snsaug_v2_policy_gate_comparison(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    assert_valid_config(config, require_exists=not dry_run)
    output_root = _real(config["output_root"])
    selectors = list(config.get("selectors") or DEFAULT_SELECTORS)
    focus_profiles = list(config.get("focus_profiles") or DEFAULT_FOCUS_PROFILES)
    paths = _paths(output_root)
    if dry_run:
        return {"marker": MARKER, "dry_run": True, "output_paths": paths}
    source = _load_jsonl(config["residual_records_path"])
    filtered = [row for row in source if row.get("profile") in set(focus_profiles)]
    if not filtered:
        raise SNSAugV2PolicyGateComparisonError("no residual records matched focus_profiles")
    records = build_policy_gate_records(filtered, selectors)
    metrics = selector_profile_metrics(records)
    decision = decide_policy_gate(metrics, focus_profiles, selectors, config.get("success_criteria") or {})
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "marker": MARKER,
        "residual_records_path": str(_real(config["residual_records_path"])),
        "output_root": str(output_root),
        "record_count": len(records),
        "source_record_count": len(source),
        "filtered_source_record_count": len(filtered),
        "selectors": selectors,
        "focus_profiles": focus_profiles,
        "decision": decision["decision"],
        "next_recommendation": decision["next_recommendation"],
        "output_paths": paths,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    _write_jsonl(Path(paths["policy_gate_records"]), records)
    _write_json(Path(paths["policy_gate_metrics"]), {"marker": MARKER, "metrics": metrics, "decision": decision})
    _write_json(Path(paths["policy_gate_oracle_gap_summary"]), {"marker": MARKER, "base_selector": "fixed_original", "oracle_selector": "oracle_best_policy_diagnostic", "decision": decision})
    report_text = _report(summary, decision)
    Path(paths["policy_gate_report"]).write_text(report_text, encoding="utf-8")
    Path(paths["policy_gate_comparison_report"]).write_text(report_text, encoding="utf-8")
    _write_json(Path(paths["artifact_manifest"]), summary)
    return summary
