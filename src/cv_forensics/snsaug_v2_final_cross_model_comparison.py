"""Final cross-model comparison matrix for report-ready SNSAug artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MARKER = "SNSAUG_V2_0077_FINAL_CROSS_MODEL_COMPARISON_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_0077_FINAL_CROSS_MODEL_COMPARISON_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_final_cross_model_comparison"
APPROVED_MODE = "approved_local_snsaug_v2_final_cross_model_comparison"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_FINAL_CROSS_MODEL_COMPARISON"
REPO_ROOT = Path(__file__).resolve().parents[2]

MAIN_SYSTEMS = ("SIDA-7B", "mixed_feature_gate")
ABLATION_SYSTEMS = ("pre_sns_baseline", "failed_single_model_finetune", "mixed_feature_gate")
FAMILIES = ("type_a_local_overlay", "type_b_global_geometry_degradation", "strict_type_a_plus_type_b")
METRICS = ("synthetic_recall", "real_fpr", "tampered_recall", "valid_iou")
SIDA_EXTRA_METRICS = ("synthetic_tampered_fpr", "detected_only_iou", "ignore_capture")


class SNSAugV2FinalCrossModelComparisonError(ValueError):
    """Raised when final cross-model comparison config or execution fails."""


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


def _optional_abs_errors(value: Any, field: str, require_exists: bool = False) -> list[str]:
    if value in (None, ""):
        return []
    return _abs_errors(value, field, require_exists=require_exists)


def _under_errors(value: Any, field: str, roots: list[str], require_exists: bool = False) -> list[str]:
    errors = _abs_errors(value, field, require_exists=require_exists)
    if isinstance(value, str) and roots and not any(_is_under(value, root) or _real(value) == _real(root) for root in roots):
        errors.append(f"{field} must be under approved roots")
    return errors


def _optional_under_errors(value: Any, field: str, roots: list[str], require_exists: bool = False) -> list[str]:
    if value in (None, ""):
        return []
    return _under_errors(value, field, roots, require_exists=require_exists)


def load_snsaug_v2_final_cross_model_comparison_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2FinalCrossModelComparisonError("final cross-model comparison config root must be a JSON object")
    raw.setdefault("config_path", str(_real(path)))
    return raw


def validate_snsaug_v2_final_cross_model_comparison_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "user_approval_text",
        "unified_sida_gate_output_root",
        "mixed_gate_output_root",
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

    for field in ("unified_sida_gate_output_root", "mixed_gate_output_root"):
        errors.extend(_under_errors(raw.get(field), field, input_roots, require_exists=require_exists))
    for field in ("pre_sns_baseline_root", "failed_single_model_finetune_root"):
        errors.extend(_optional_under_errors(raw.get(field), field, input_roots, require_exists=require_exists))

    if require_exists and isinstance(raw.get("unified_sida_gate_output_root"), str):
        root = _real(raw["unified_sida_gate_output_root"])
        for name in (
            "unified_sida_gate_comparison_summary.json",
            "unified_sida_gate_profile_table.tsv",
            "unified_sida_gate_comparison_report.md",
            "artifact_manifest.json",
        ):
            if not (root / name).exists():
                errors.append(f"unified_sida_gate_output_root missing {name}")
    if require_exists and isinstance(raw.get("mixed_gate_output_root"), str):
        root = _real(raw["mixed_gate_output_root"])
        for name in ("final_policy_gate_metrics.json", "deployable_policy_gate_spec.json", "artifact_manifest.json"):
            if not (root / name).exists():
                errors.append(f"mixed_gate_output_root missing {name}")

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
    errors = validate_snsaug_v2_final_cross_model_comparison_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2FinalCrossModelComparisonError("final cross-model comparison config validation failed:\n" + "\n".join(errors))


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


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SNSAugV2FinalCrossModelComparisonError(f"{path} must contain a JSON object")
    return value


def _try_load_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            return _load_json(path)
    except (OSError, json.JSONDecodeError, SNSAugV2FinalCrossModelComparisonError):
        return None
    return None


def _paths(output_root: Path) -> dict[str, str]:
    return {
        "final_cross_model_comparison_matrix": str(output_root / "final_cross_model_comparison_matrix.json"),
        "final_cross_model_comparison_report": str(output_root / "final_cross_model_comparison_matrix.md"),
        "final_cross_model_comparison_table": str(output_root / "final_cross_model_comparison_table.tsv"),
        "final_claims_for_report": str(output_root / "final_claims_for_report.md"),
        "artifact_manifest": str(output_root / "artifact_manifest.json"),
    }


def _num(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    return None


def _fmt(value: Any) -> str:
    value = _num(value)
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metric_from(row: dict[str, Any], system_key: str, metric: str) -> Any:
    for key in (f"{system_key}_{metric}", metric):
        if key in row:
            return _num(row.get(key))
    metrics = row.get(system_key)
    if isinstance(metrics, dict) and metric in metrics:
        return _num(metrics.get(metric))
    return None


def _summary_family(summary: dict[str, Any], family: str) -> dict[str, Any]:
    for key in ("families", "family_summary", "summary"):
        block = summary.get(key)
        if isinstance(block, dict) and isinstance(block.get(family), dict):
            return dict(block[family])
    return {}


def _profile_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = summary.get("profiles")
    return profiles if isinstance(profiles, dict) else {}


def _aggregate_rows(rows: list[dict[str, Any]], system_key: str) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for metric in METRICS:
        values = [_metric_from(row, system_key, metric) for row in rows]
        numeric = [value for value in values if value is not None]
        aggregate[metric] = sum(numeric) / len(numeric) if numeric else None
    for metric in SIDA_EXTRA_METRICS:
        values = [_metric_from(row, system_key, metric) for row in rows]
        numeric = [value for value in values if value is not None]
        aggregate[metric] = sum(numeric) / len(numeric) if numeric else None
    return aggregate


def _family_metrics(summary: dict[str, Any], family: str, system_key: str) -> dict[str, Any]:
    direct = _summary_family(summary, family)
    row_source = direct if direct else {}
    if not row_source:
        profiles = _profile_rows(summary)
        rows = [
            row
            for row in profiles.values()
            if isinstance(row, dict) and row.get("profile_family") == family
        ]
        if family == "strict_type_a_plus_type_b":
            rows = [
                row
                for row in profiles.values()
                if isinstance(row, dict) and row.get("profile_family") in {"type_a_local_overlay", "type_b_global_geometry_degradation"}
            ]
        if rows:
            return _aggregate_rows(rows, system_key)
    metrics = {metric: _metric_from(row_source, system_key, metric) for metric in METRICS}
    if system_key == "sida":
        metrics.update({metric: _metric_from(row_source, system_key, metric) for metric in SIDA_EXTRA_METRICS})
    return metrics


def _model_entry(system: str, system_key: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": system,
        "role": "3-way label and mask diagnostic" if system == "SIDA-7B" else "deployable policy-gated recovery system",
        "drop_in_replacement_for_other_system": False,
        "family_metrics": {family: _family_metrics(summary, family, system_key) for family in FAMILIES},
    }


def _load_optional_context(root_value: Any, system: str) -> dict[str, Any]:
    empty = {
        "system": system,
        "loaded": False,
        "load_status": "not_configured",
        "metrics": {family: {metric: None for metric in METRICS} for family in FAMILIES},
    }
    if not isinstance(root_value, str) or not root_value.strip():
        return empty
    root = _real(root_value)
    candidates = (
        root / "final_policy_gate_metrics.json",
        root / "metrics.json",
        root / "summary.json",
        root / "artifact_manifest.json",
    )
    payload = None
    for candidate in candidates:
        payload = _try_load_json(candidate)
        if payload is not None:
            break
    if payload is None:
        empty["load_status"] = "unavailable_or_unparseable"
        return empty
    metrics = {family: _family_metrics(payload, family, system) for family in FAMILIES}
    return {"system": system, "loaded": True, "load_status": "loaded", "metrics": metrics}


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    unified_root = _real(config["unified_sida_gate_output_root"])
    gate_root = _real(config["mixed_gate_output_root"])
    unified_summary = _load_json(unified_root / "unified_sida_gate_comparison_summary.json")
    unified_manifest = _load_json(unified_root / "artifact_manifest.json")
    gate_metrics = _load_json(gate_root / "final_policy_gate_metrics.json")
    gate_spec = _load_json(gate_root / "deployable_policy_gate_spec.json")
    gate_manifest = _load_json(gate_root / "artifact_manifest.json")
    return {
        "unified_root": str(unified_root),
        "mixed_gate_root": str(gate_root),
        "unified_summary": unified_summary,
        "unified_manifest": unified_manifest,
        "gate_metrics": gate_metrics,
        "gate_spec": gate_spec,
        "gate_manifest": gate_manifest,
        "pre_sns_baseline": _load_optional_context(config.get("pre_sns_baseline_root"), "pre_sns_baseline"),
        "failed_single_model_finetune": _load_optional_context(config.get("failed_single_model_finetune_root"), "failed_single_model_finetune"),
    }


def build_matrix(inputs: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = inputs["unified_summary"]
    main_entries = [
        _model_entry("SIDA-7B", "sida", summary),
        _model_entry("mixed_feature_gate", "mixed_gate", summary),
    ]
    mixed_gate_context = {
        "system": "mixed_feature_gate",
        "loaded": True,
        "load_status": "loaded_from_main_comparison",
        "metrics": main_entries[1]["family_metrics"],
    }
    matrix = {
        "marker": MARKER,
        "source_roots": {
            "unified_sida_gate_output_root": inputs["unified_root"],
            "mixed_gate_output_root": inputs["mixed_gate_root"],
            "pre_sns_baseline_root": config.get("pre_sns_baseline_root"),
            "failed_single_model_finetune_root": config.get("failed_single_model_finetune_root"),
        },
        "main_table_systems": list(MAIN_SYSTEMS),
        "ablation_table_systems": list(ABLATION_SYSTEMS),
        "families": list(FAMILIES),
        "metrics": list(METRICS),
        "model_entries": main_entries,
        "ablation_entries": [
            inputs["pre_sns_baseline"],
            inputs["failed_single_model_finetune"],
            mixed_gate_context,
        ],
        "recommendations": {
            "main_table": "Use SIDA-7B and mixed_feature_gate as separate systems compared by synthetic preservation, real FPR, tampered recall, and tampered localization.",
            "ablation_table": "Use pre-SNS baseline and failed single-model fine-tune only as context for the failure path that motivated the gate.",
            "do_not_overclaim": [
                "Do not claim mixed_feature_gate is a drop-in replacement for SIDA.",
                "Do not claim SIDA and mixed_feature_gate emit identical outputs.",
                "Do not turn missing baseline/fine-tune metrics into numeric claims.",
            ],
        },
        "interpretation_guardrails": [
            "SIDA and mixed_feature_gate are not identical systems.",
            "SIDA emits 3-way labels and masks.",
            "mixed_feature_gate is a deployable policy-gated recovery system.",
            "The fair comparison is split across synthetic preservation, real FPR, tampered recall, and tampered localization.",
            "mixed_feature_gate is not a drop-in replacement for SIDA.",
        ],
        "safety_flags": {
            "no_training": True,
            "no_finetune": True,
            "no_network": True,
            "no_download": True,
        },
    }
    return matrix


def render_tsv(matrix: dict[str, Any]) -> str:
    columns = ["system", "family", "synthetic_recall", "real_fpr", "tampered_recall", "valid_iou", "synthetic_tampered_fpr", "detected_only_iou", "ignore_capture"]
    lines = ["\t".join(columns)]
    for entry in matrix["model_entries"]:
        system = entry["system"]
        for family in FAMILIES:
            metrics = entry["family_metrics"][family]
            lines.append("\t".join([system, family] + [_fmt(metrics.get(column)) for column in columns[2:]]))
    return "\n".join(lines) + "\n"


def _markdown_table(entries: list[dict[str, Any]], include_sida_extra: bool = False) -> list[str]:
    header = "| System | Family | Synthetic preservation | Real FPR | Tampered recall | Tampered localization |"
    divider = "| --- | --- | ---: | ---: | ---: | ---: |"
    lines = [header, divider]
    for entry in entries:
        for family in FAMILIES:
            metrics = entry["family_metrics"][family] if "family_metrics" in entry else entry["metrics"][family]
            lines.append(
                f"| {entry['system']} | {family} | {_fmt(metrics.get('synthetic_recall'))} | {_fmt(metrics.get('real_fpr'))} | {_fmt(metrics.get('tampered_recall'))} | {_fmt(metrics.get('valid_iou'))} |"
            )
    if include_sida_extra:
        lines += [
            "",
            "SIDA-specific mask context is retained in JSON/TSV when available: `synthetic_tampered_fpr`, `detected_only_iou`, and `ignore_capture`.",
        ]
    return lines


def render_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# Final Cross-model Comparison Matrix",
        "",
        MARKER,
        "",
        "## Recommendation",
        "",
        matrix["recommendations"]["main_table"],
        "",
        "## Main Table",
        "",
    ]
    lines.extend(_markdown_table(matrix["model_entries"], include_sida_extra=True))
    lines += [
        "",
        "## Ablation / Context Table",
        "",
        "Baseline and failed fine-tune rows explain the failure path and why the deployable gate was needed. They are not the primary claim.",
        "",
    ]
    lines.extend(_markdown_table(matrix["ablation_entries"]))
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    for item in matrix["interpretation_guardrails"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Do Not Overclaim",
        "",
    ]
    for item in matrix["recommendations"]["do_not_overclaim"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def render_claims(matrix: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Final Claims for Report",
            "",
            MARKER,
            "",
            "## Supported Claims",
            "",
            "- The report can compare SIDA-7B and mixed_feature_gate across synthetic preservation, real FPR, tampered recall, and tampered localization.",
            "- SIDA-7B should be described as a 3-way label and mask-producing diagnostic system.",
            "- mixed_feature_gate should be described as a deployable policy-gated recovery system.",
            "- The baseline and failed fine-tune rows are context for why policy gating was needed.",
            "",
            "## Unsupported Claims",
            "",
            "- mixed_feature_gate is not a drop-in replacement for SIDA.",
            "- SIDA-7B and mixed_feature_gate are not identical systems.",
            "- Missing or unparsable baseline/fine-tune metrics must remain NA/null.",
            "",
        ]
    )


def run_snsaug_v2_final_cross_model_comparison(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    assert_valid_config(config, require_exists=not dry_run)
    output_root = _real(config["output_root"])
    paths = _paths(output_root)
    plan = {
        "marker": MARKER,
        "dry_run": dry_run,
        "output_root": str(output_root),
        "output_paths": paths,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    if dry_run:
        return plan
    inputs = load_inputs(config)
    matrix = build_matrix(inputs, config)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(Path(paths["final_cross_model_comparison_matrix"]), matrix)
    _write_text(Path(paths["final_cross_model_comparison_report"]), render_markdown(matrix))
    _write_text(Path(paths["final_cross_model_comparison_table"]), render_tsv(matrix))
    _write_text(Path(paths["final_claims_for_report"]), render_claims(matrix))
    artifact = {
        **plan,
        "dry_run": False,
        "analysis_started": True,
        "training_started": False,
        "download_started": False,
        "network_used": False,
        "source_roots": matrix["source_roots"],
        "main_table_systems": matrix["main_table_systems"],
        "ablation_table_systems": matrix["ablation_table_systems"],
        "artifact_count": 5,
    }
    _write_json(Path(paths["artifact_manifest"]), artifact)
    return artifact


__all__ = [
    "APPROVAL_TEXT",
    "CONFIG_OK_MARKER",
    "MARKER",
    "SNSAugV2FinalCrossModelComparisonError",
    "build_matrix",
    "load_snsaug_v2_final_cross_model_comparison_config",
    "render_markdown",
    "render_tsv",
    "run_snsaug_v2_final_cross_model_comparison",
    "validate_snsaug_v2_final_cross_model_comparison_config",
]
