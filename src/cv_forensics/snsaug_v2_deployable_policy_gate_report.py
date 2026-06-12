"""Freeze the deployable SNSAug v2 policy gate and write final report artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MARKER = "SNSAUG_V2_DEPLOYABLE_POLICY_GATE_REPORT_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_DEPLOYABLE_POLICY_GATE_REPORT_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_deployable_policy_gate_report"
APPROVED_MODE = "approved_local_snsaug_v2_deployable_policy_gate_report"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_DEPLOYABLE_POLICY_GATE_REPORT"
POLICY_GATE_MARKER = "SNSAUG_V2_POLICY_GATE_COMPARISON_OK"
REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_INPUT_FILES = (
    "policy_gate_records.jsonl",
    "policy_gate_metrics.json",
    "policy_gate_oracle_gap_summary.json",
    "policy_gate_report.md",
    "policy_gate_comparison_report.md",
    "artifact_manifest.json",
)
OUTPUT_FILES = (
    "deployable_policy_gate_spec.json",
    "final_policy_gate_records.jsonl",
    "final_policy_gate_metrics.json",
    "final_policy_gate_report.md",
    "final_policy_gate_notion_summary.md",
    "final_policy_gate_tables.tsv",
    "artifact_manifest.json",
)


class SNSAugV2DeployablePolicyGateReportError(ValueError):
    """Raised when deployable policy gate report validation or execution fails."""


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


def load_snsaug_v2_deployable_policy_gate_report_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2DeployablePolicyGateReportError("deployable policy gate report config root must be a JSON object")
    raw.setdefault("config_path", str(_real(path)))
    return raw


def validate_snsaug_v2_deployable_policy_gate_report_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    for field in (
        "schema_version",
        "config_kind",
        "execution_mode",
        "user_approval_text",
        "policy_gate_output_root",
        "approved_input_roots",
        "approved_output_roots",
        "output_root",
        "no_training",
        "no_finetune",
        "no_network",
        "no_download",
    ):
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
        errors.extend(_abs_errors(root, f"approved_input_roots[{index}]", require_exists=require_exists))
    for index, root in enumerate(output_roots):
        errors.extend(_abs_errors(root, f"approved_output_roots[{index}]", require_exists=require_exists))
    errors.extend(_under_errors(raw.get("policy_gate_output_root"), "policy_gate_output_root", input_roots, require_exists=require_exists))
    if require_exists and isinstance(raw.get("policy_gate_output_root"), str):
        for name in REQUIRED_INPUT_FILES:
            path = _real(raw["policy_gate_output_root"]) / name
            if not path.exists():
                errors.append(f"policy_gate_output_root missing {name}")
    output_root = raw.get("output_root")
    errors.extend(_abs_errors(output_root, "output_root"))
    if isinstance(output_root, str) and output_root.strip():
        if _inside_repo(output_root):
            errors.append("output_root must be outside repository")
        if any(_is_under(output_root, root) or _real(output_root) == _real(root) for root in input_roots):
            errors.append("output_root must not be under approved input roots")
        if output_roots and not any(_is_under(output_root, root) or _real(output_root) == _real(root) for root in output_roots):
            errors.append("output_root must be under approved output roots")
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_deployable_policy_gate_report_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2DeployablePolicyGateReportError("deployable policy gate report config validation failed:\n" + "\n".join(errors))


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


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SNSAugV2DeployablePolicyGateReportError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean(values: list[Any]) -> float | None:
    xs = [_num(value) for value in values]
    xs = [value for value in xs if value is not None]
    return sum(xs) / len(xs) if xs else None


def _output_paths(output_root: Path) -> dict[str, str]:
    return {
        "deployable_policy_gate_spec": str(output_root / "deployable_policy_gate_spec.json"),
        "final_policy_gate_records": str(output_root / "final_policy_gate_records.jsonl"),
        "final_policy_gate_metrics": str(output_root / "final_policy_gate_metrics.json"),
        "final_policy_gate_report": str(output_root / "final_policy_gate_report.md"),
        "final_policy_gate_notion_summary": str(output_root / "final_policy_gate_notion_summary.md"),
        "final_policy_gate_tables": str(output_root / "final_policy_gate_tables.tsv"),
        "artifact_manifest": str(output_root / "artifact_manifest.json"),
    }


def load_policy_gate_outputs(policy_gate_output_root: str | Path) -> dict[str, Any]:
    root = _real(policy_gate_output_root)
    return {
        "root": str(root),
        "records": _load_jsonl(root / "policy_gate_records.jsonl"),
        "metrics": _load_json(root / "policy_gate_metrics.json"),
        "oracle_gap": _load_json(root / "policy_gate_oracle_gap_summary.json"),
        "artifact": _load_json(root / "artifact_manifest.json"),
        "report_text": (root / "policy_gate_report.md").read_text(encoding="utf-8"),
        "comparison_report_text": (root / "policy_gate_comparison_report.md").read_text(encoding="utf-8"),
    }


def _is_oracle_policy(value: Any) -> bool:
    return "oracle" in str(value or "").lower()


def select_best_deployable_candidate(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    decision = metrics_payload.get("decision") if isinstance(metrics_payload.get("decision"), dict) else {}
    candidates = decision.get("deployable_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise SNSAugV2DeployablePolicyGateReportError("policy_gate_metrics.json has no deployable_candidates")
    clean = [candidate for candidate in candidates if isinstance(candidate, dict) and not candidate.get("diagnostic_only")]
    if not clean:
        raise SNSAugV2DeployablePolicyGateReportError("no non-diagnostic deployable candidate available")
    clean.sort(key=lambda row: _num(row.get("score")) if _num(row.get("score")) is not None else float("-inf"), reverse=True)
    return dict(clean[0])


def _chosen_policy_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        key = str(row.get("chosen_policy") or "")
        if key:
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _feature_gate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_geometry_score": _mean([row.get("geometry_score") for row in records]),
        "mean_residual_score": _mean([row.get("residual_score") for row in records]),
        "geometry_score_min": min((_num(row.get("geometry_score")) for row in records if _num(row.get("geometry_score")) is not None), default=None),
        "geometry_score_max": max((_num(row.get("geometry_score")) for row in records if _num(row.get("geometry_score")) is not None), default=None),
        "residual_score_min": min((_num(row.get("residual_score")) for row in records if _num(row.get("residual_score")) is not None), default=None),
        "residual_score_max": max((_num(row.get("residual_score")) for row in records if _num(row.get("residual_score")) is not None), default=None),
    }


def build_deployable_policy_spec(candidate: dict[str, Any], selected_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "marker": MARKER,
        "selected_selector": candidate.get("selector"),
        "selected_selector_score": candidate.get("score"),
        "good_profile_count": candidate.get("good_profile_count"),
        "good_profiles": candidate.get("good_profiles") or [],
        "mean_tampered_recall_gain": candidate.get("mean_tampered_recall_gain"),
        "mean_valid_iou_gain": candidate.get("mean_valid_iou_gain"),
        "mean_tampered_gap_closure": candidate.get("mean_tampered_gap_closure"),
        "mean_valid_iou_gap_closure": candidate.get("mean_valid_iou_gap_closure"),
        "mean_synthetic_recall_change": candidate.get("mean_synthetic_recall_change"),
        "mean_real_fpr_change": candidate.get("mean_real_fpr_change"),
        "chosen_policy_distribution": _chosen_policy_distribution(selected_records),
        "feature_gate_summary": _feature_gate_summary(selected_records),
        "diagnostic_only": False,
    }


def _pred_class(row: dict[str, Any]) -> str:
    if row.get("pred_class"):
        return str(row["pred_class"])
    if row.get("pred_tampered") is True:
        return "tampered"
    return "non_tampered"


def build_final_records(records: list[dict[str, Any]], selected_selector: str) -> list[dict[str, Any]]:
    selected = [row for row in records if row.get("selector") == selected_selector]
    if not selected:
        raise SNSAugV2DeployablePolicyGateReportError(f"no records found for selected selector {selected_selector}")
    if any(row.get("diagnostic_only_selector") for row in selected):
        raise SNSAugV2DeployablePolicyGateReportError("selected deployable records include diagnostic_only_selector=true")
    if any(not row.get("chosen_policy") for row in selected):
        raise SNSAugV2DeployablePolicyGateReportError("selected deployable records contain null chosen_policy")
    if any(_is_oracle_policy(row.get("chosen_policy")) for row in selected):
        raise SNSAugV2DeployablePolicyGateReportError("selected deployable records contain oracle-only chosen_policy")
    final: list[dict[str, Any]] = []
    for row in selected:
        p_tampered = row.get("p_tampered")
        final.append(
            {
                "marker": MARKER,
                "selector": row.get("selector"),
                "chosen_policy": row.get("chosen_policy"),
                "selector_reason": row.get("selector_reason"),
                "diagnostic_only_selector": False,
                "profile": row.get("profile"),
                "profile_family": row.get("profile_family"),
                "content_label": row.get("content_label"),
                "base_id": row.get("base_id"),
                "pred_class": _pred_class(row),
                "pred_tampered": row.get("pred_tampered"),
                "p_real": row.get("p_real"),
                "p_synthetic": row.get("p_synthetic"),
                "p_tampered": p_tampered,
                "valid_iou": row.get("valid_iou"),
                "geometry_score": row.get("geometry_score"),
                "residual_score": row.get("residual_score"),
                "selected_by_profile_family": row.get("selected_by_profile_family"),
            }
        )
    return final


def _selector_metrics(metrics_payload: dict[str, Any], selector: str) -> dict[str, Any]:
    metrics = metrics_payload.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    value = metrics.get(selector)
    return value if isinstance(value, dict) else {}


def _comparison_rows(selected_metrics: dict[str, Any], base_metrics: dict[str, Any], oracle_metrics: dict[str, Any]) -> dict[str, Any]:
    profiles = sorted(set(selected_metrics) | set(base_metrics) | set(oracle_metrics))
    out: dict[str, Any] = {}
    for profile in profiles:
        selected = selected_metrics.get(profile, {}) if isinstance(selected_metrics.get(profile), dict) else {}
        base = base_metrics.get(profile, {}) if isinstance(base_metrics.get(profile), dict) else {}
        oracle = oracle_metrics.get(profile, {}) if isinstance(oracle_metrics.get(profile), dict) else {}
        row = {"selected": selected, "fixed_original": base, "oracle_best_policy_diagnostic": oracle}
        for key in ("tampered_recall", "tampered_valid_mean_iou", "synthetic_recall", "real_fpr", "mean_p_tampered"):
            left = _num(selected.get(key))
            right = _num(base.get(key))
            oracle_value = _num(oracle.get(key))
            row[f"{key}_vs_fixed_original"] = None if left is None or right is None else left - right
            row[f"{key}_vs_oracle"] = None if left is None or oracle_value is None else left - oracle_value
        out[profile] = row
    return out


def _average_metrics(by_profile: dict[str, Any], profiles: list[str] | None = None) -> dict[str, Any]:
    selected_profiles = profiles or list(by_profile)
    rows = [by_profile[profile]["selected"] for profile in selected_profiles if isinstance(by_profile.get(profile), dict) and isinstance(by_profile[profile].get("selected"), dict)]
    return {
        "profile_count": len(rows),
        "tampered_recall": _mean([row.get("tampered_recall") for row in rows]),
        "tampered_valid_mean_iou": _mean([row.get("tampered_valid_mean_iou") for row in rows]),
        "synthetic_recall": _mean([row.get("synthetic_recall") for row in rows]),
        "real_fpr": _mean([row.get("real_fpr") for row in rows]),
        "mean_p_tampered": _mean([row.get("mean_p_tampered") for row in rows]),
    }


def build_final_metrics(metrics_payload: dict[str, Any], selected_selector: str, good_profiles: list[str]) -> dict[str, Any]:
    selected = _selector_metrics(metrics_payload, selected_selector)
    fixed = _selector_metrics(metrics_payload, "fixed_original")
    oracle = _selector_metrics(metrics_payload, "oracle_best_policy_diagnostic")
    comparisons = _comparison_rows(selected, fixed, oracle)
    return {
        "marker": MARKER,
        "selected_selector": selected_selector,
        "per_profile": selected,
        "comparison_against_fixed_original": {profile: row for profile, row in comparisons.items()},
        "comparison_against_oracle_best_policy_diagnostic": {profile: row for profile, row in comparisons.items() if row.get("oracle_best_policy_diagnostic")},
        "focus_profile_average": _average_metrics(comparisons, good_profiles),
        "all_profile_average": _average_metrics(comparisons),
    }


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def _render_report(spec: dict[str, Any], metrics: dict[str, Any]) -> str:
    unresolved = sorted(set((metrics.get("per_profile") or {}).keys()) - set(spec.get("good_profiles") or []))
    lines = [
        "# Final Deployable SNSAug Policy Gate",
        "",
        MARKER,
        "",
        f"- Selected selector: `{spec['selected_selector']}`",
        f"- Score: `{_fmt(spec.get('selected_selector_score'))}`",
        f"- Good profiles: `{spec.get('good_profile_count')}`",
        f"- Diagnostic only: `{spec.get('diagnostic_only')}`",
        "",
        "## Interpretation",
        "",
        "Naive SNSAug fine-tuning recovered tampered activation but over-predicted tampered on real and synthetic SNSAug samples, so it did not produce a stable operating point.",
        "",
        "Local nuisance masking was insufficient because ignore-mask overlay regions did not explain most of the recall and IoU collapse.",
        "",
        "Global geometry and degradation analysis showed that resize/crop, blockiness, residual, and DCT shifts dominated the SNSAug failure mode.",
        "",
        "The frozen deployable policy gate combines geometry and residual/DCT feature signals to choose a practical non-oracle preprocessing policy per sample.",
        "",
        "## Recovered Profiles",
        "",
    ]
    for profile in spec.get("good_profiles") or []:
        lines.append(f"- `{profile}`")
    lines.extend(["", "## Remaining Unresolved Profiles", ""])
    if unresolved:
        for profile in unresolved:
            lines.append(f"- `{profile}`")
    else:
        lines.append("- None in the selected metric table.")
    lines.extend([
        "",
        "## Why Deployable",
        "",
        "The selected gate is non-diagnostic, has `diagnostic_only=false`, and final exported records reject oracle-only chosen policies.",
        "",
        "## Final Metrics",
        "",
        "| Profile | Tampered Recall | Valid IoU | Synthetic Recall | Real FPR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for profile, row in sorted((metrics.get("per_profile") or {}).items()):
        lines.append(f"| {profile} | {_fmt(row.get('tampered_recall'))} | {_fmt(row.get('tampered_valid_mean_iou'))} | {_fmt(row.get('synthetic_recall'))} | {_fmt(row.get('real_fpr'))} |")
    return "\n".join(lines) + "\n"


def _render_notion_summary(spec: dict[str, Any], metrics: dict[str, Any]) -> str:
    avg = metrics.get("focus_profile_average") or {}
    return "\n".join(
        [
            "# Final SNSAug Recovery Summary",
            "",
            MARKER,
            "",
            f"Selected deployable gate: `{spec.get('selected_selector')}`",
            "",
            f"Good profiles: `{spec.get('good_profile_count')}`",
            f"Mean tampered recall gain: `{_fmt(spec.get('mean_tampered_recall_gain'))}`",
            f"Mean valid IoU gain: `{_fmt(spec.get('mean_valid_iou_gain'))}`",
            f"Focus average tampered recall: `{_fmt(avg.get('tampered_recall'))}`",
            f"Focus average valid IoU: `{_fmt(avg.get('tampered_valid_mean_iou'))}`",
            "",
            "Conclusion: freeze this non-oracle policy gate as the final practical SNSAug recovery policy.",
            "",
        ]
    )


def _render_tables_tsv(spec: dict[str, Any], metrics: dict[str, Any]) -> str:
    lines = ["section\tprofile\tmetric\tvalue"]
    for key, value in spec.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"spec\tselected\t{key}\t{value}")
    for profile, row in sorted((metrics.get("per_profile") or {}).items()):
        if isinstance(row, dict):
            for key in ("tampered_recall", "tampered_valid_mean_iou", "synthetic_recall", "real_fpr", "mean_p_tampered"):
                lines.append(f"per_profile\t{profile}\t{key}\t{row.get(key)}")
    return "\n".join(lines) + "\n"


def run_snsaug_v2_deployable_policy_gate_report(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    assert_valid_config(config, require_exists=not dry_run)
    output_root = _real(config["output_root"])
    paths = _output_paths(output_root)
    plan = {
        "marker": MARKER,
        "dry_run": dry_run,
        "policy_gate_output_root": str(_real(config["policy_gate_output_root"])),
        "output_root": str(output_root),
        "output_paths": paths,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    if dry_run:
        return plan
    payload = load_policy_gate_outputs(config["policy_gate_output_root"])
    if payload["artifact"].get("marker") != POLICY_GATE_MARKER:
        raise SNSAugV2DeployablePolicyGateReportError("input artifact marker is not a policy gate comparison artifact")
    candidate = select_best_deployable_candidate(payload["metrics"])
    selector = str(candidate.get("selector") or "")
    final_records = build_final_records(payload["records"], selector)
    spec = build_deployable_policy_spec(candidate, final_records)
    final_metrics = build_final_metrics(payload["metrics"], selector, list(spec.get("good_profiles") or []))
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(Path(paths["deployable_policy_gate_spec"]), spec)
    _write_jsonl(Path(paths["final_policy_gate_records"]), final_records)
    _write_json(Path(paths["final_policy_gate_metrics"]), final_metrics)
    _write_text(Path(paths["final_policy_gate_report"]), _render_report(spec, final_metrics))
    _write_text(Path(paths["final_policy_gate_notion_summary"]), _render_notion_summary(spec, final_metrics))
    _write_text(Path(paths["final_policy_gate_tables"]), _render_tables_tsv(spec, final_metrics))
    artifact = {
        **plan,
        "dry_run": False,
        "analysis_started": True,
        "training_started": False,
        "selected_selector": selector,
        "record_count": len(final_records),
        "source_record_count": len(payload["records"]),
        "input_artifact_marker": payload["artifact"].get("marker"),
        "output_paths": paths,
    }
    _write_json(Path(paths["artifact_manifest"]), artifact)
    return artifact


__all__ = [
    "APPROVAL_TEXT",
    "CONFIG_OK_MARKER",
    "MARKER",
    "SNSAugV2DeployablePolicyGateReportError",
    "build_deployable_policy_spec",
    "build_final_metrics",
    "build_final_records",
    "load_snsaug_v2_deployable_policy_gate_report_config",
    "run_snsaug_v2_deployable_policy_gate_report",
    "select_best_deployable_candidate",
    "validate_snsaug_v2_deployable_policy_gate_report_config",
]
