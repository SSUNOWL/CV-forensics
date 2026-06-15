"""Fill final cross-model ablation metrics from the final decision report."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MARKER = "SNSAUG_V2_0077A_FILL_FINAL_CROSS_MODEL_ABLATION_METRICS_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_0077A_FILL_FINAL_CROSS_MODEL_ABLATION_METRICS_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_fill_final_cross_model_ablation_metrics"
APPROVED_MODE = "approved_local_snsaug_v2_fill_final_cross_model_ablation_metrics"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_FILL_FINAL_CROSS_MODEL_ABLATION_METRICS"
REPO_ROOT = Path(__file__).resolve().parents[2]

TYPE_A_PROFILES = (
    "news_meme_overlay",
    "platform_ui_same_size",
    "tiktok_like",
    "instagram_story_like",
    "youtube_shorts_like",
    "combined_sns_realistic",
)
TYPE_B_PROFILES = (
    "canvas_9x16_only",
    "resize_crop_pad",
    "zoom_crop",
    "resize_jpeg",
    "screenshot_recapture_light",
    "recompression_light",
)
FAMILIES = ("type_a_local_overlay", "type_b_global_geometry_degradation", "strict_type_a_plus_type_b")
FAMILY_PROFILES = {
    "type_a_local_overlay": TYPE_A_PROFILES,
    "type_b_global_geometry_degradation": TYPE_B_PROFILES,
    "strict_type_a_plus_type_b": TYPE_A_PROFILES + TYPE_B_PROFILES,
}
OUTPUT_NAMES = {
    "matrix": "final_cross_model_comparison_matrix_filled.json",
    "report": "final_cross_model_comparison_matrix_filled.md",
    "table": "final_cross_model_comparison_table_filled.tsv",
    "claims": "final_claims_for_report_filled.md",
    "manifest": "artifact_manifest.json",
}


class SNSAugV2FillFinalCrossModelAblationMetricsError(ValueError):
    """Raised when 0077a config or execution fails."""


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


def load_snsaug_v2_fill_final_cross_model_ablation_metrics_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2FillFinalCrossModelAblationMetricsError("0077a config root must be a JSON object")
    raw.setdefault("config_path", str(_real(path)))
    return raw


def validate_snsaug_v2_fill_final_cross_model_ablation_metrics_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "user_approval_text",
        "comparison_search_root",
        "decision_report_search_root",
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
    for field in ("comparison_search_root", "decision_report_search_root"):
        errors.extend(_under_errors(raw.get(field), field, input_roots, require_exists=require_exists))
    for field in ("comparison_output_root", "decision_report_path"):
        errors.extend(_optional_under_errors(raw.get(field), field, input_roots, require_exists=require_exists))

    output_root = raw.get("output_root")
    errors.extend(_abs_errors(output_root, "output_root"))
    if isinstance(output_root, str) and output_root.strip():
        if _inside_repo(output_root):
            errors.append("output_root must be outside repository")
        if output_roots and not any(_is_under(output_root, root) or _real(output_root) == _real(root) for root in output_roots):
            errors.append("output_root must be under approved output roots")

    if require_exists:
        try:
            comparison_root = resolve_comparison_output_root(raw)
            decision_path = resolve_decision_report_path(raw)
            if not (comparison_root / "final_cross_model_comparison_matrix.json").exists():
                errors.append("comparison output root missing final_cross_model_comparison_matrix.json")
            if not decision_path.exists():
                errors.append("decision report path does not exist")
        except SNSAugV2FillFinalCrossModelAblationMetricsError as exc:
            errors.append(str(exc))
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_fill_final_cross_model_ablation_metrics_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2FillFinalCrossModelAblationMetricsError("0077a config validation failed:\n" + "\n".join(errors))


def _latest_dir(root: str | Path, pattern: str) -> Path:
    base = _real(root)
    matches = sorted(path for path in base.glob(pattern) if path.is_dir())
    if not matches:
        raise SNSAugV2FillFinalCrossModelAblationMetricsError(f"no directory matched {pattern} under {base}")
    return matches[-1]


def resolve_comparison_output_root(config: dict[str, Any]) -> Path:
    configured = config.get("comparison_output_root")
    if isinstance(configured, str) and configured.strip():
        return _real(configured)
    return _latest_dir(config["comparison_search_root"], "snsaug_v2_0077_final_cross_model_comparison_*")


def resolve_decision_report_path(config: dict[str, Any]) -> Path:
    configured = config.get("decision_report_path")
    if isinstance(configured, str) and configured.strip():
        return _real(configured)
    root = _latest_dir(config["decision_report_search_root"], "snsaug_v2_final_decision_report_*")
    return root / "final_decision_report.md"


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SNSAugV2FillFinalCrossModelAblationMetricsError(f"{path} must contain a JSON object")
    return value


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


def _num(value: str) -> float | None:
    text = value.strip()
    if not text or text.upper() == "NA":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def parse_final_decision_markdown_table(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        lowered = [cell.lower() for cell in cells]
        if lowered == ["model", "profile", "acc", "f1", "real_fpr", "syn_rec", "tamp_rec", "loc_act", "valid_iou", "high_mask"]:
            header = lowered
            continue
        if header is None or set(cells[0]) <= {"-"}:
            continue
        if len(cells) != len(header):
            continue
        item = dict(zip(header, cells))
        rows.append(
            {
                "model": item["model"],
                "profile": item["profile"],
                "three_way_accuracy": _num(item["acc"]),
                "macro_f1": _num(item["f1"]),
                "real_fpr": _num(item["real_fpr"]),
                "synthetic_recall": _num(item["syn_rec"]),
                "tampered_recall": _num(item["tamp_rec"]),
                "localization_activation": _num(item["loc_act"]),
                "valid_iou": _num(item["valid_iou"]),
                "high_mask": _num(item["high_mask"]),
            }
        )
    return rows


def _mean(values: list[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def aggregate_family_metrics(rows: list[dict[str, Any]], model: str, family: str) -> dict[str, Any]:
    profiles = set(FAMILY_PROFILES[family])
    selected = [row for row in rows if row.get("model") == model and row.get("profile") in profiles]
    return {
        "three_way_accuracy": _mean([row.get("three_way_accuracy") for row in selected]),
        "macro_f1": _mean([row.get("macro_f1") for row in selected]),
        "synthetic_recall": _mean([row.get("synthetic_recall") for row in selected]),
        "real_fpr": _mean([row.get("real_fpr") for row in selected]),
        "tampered_recall": _mean([row.get("tampered_recall") for row in selected]),
        "valid_iou": _mean([row.get("valid_iou") for row in selected]),
        "localization_activation": _mean([row.get("localization_activation") for row in selected]),
        "high_mask": _mean([row.get("high_mask") for row in selected]),
        "source_profile_count": len(selected),
        "source_profiles": [str(row["profile"]) for row in selected],
    }


def select_failed_model(rows: list[dict[str, Any]]) -> str | None:
    models = sorted({str(row.get("model")) for row in rows if row.get("model") and row.get("model") != "pre_sns_baseline"})
    preferences = (
        lambda name: "0064g" in name and "hybrid" in name,
        lambda name: "0064f" in name,
        lambda name: "0063b" in name,
        lambda name: "0060b" in name,
    )
    for predicate in preferences:
        for model in models:
            if predicate(model):
                return model
    if not models:
        return None
    scored: list[tuple[float, str]] = []
    for model in models:
        strict = aggregate_family_metrics(rows, model, "strict_type_a_plus_type_b")
        tampered = strict["tampered_recall"] or 0.0
        synthetic = strict["synthetic_recall"] or 0.0
        scored.append((tampered - synthetic, model))
    return sorted(scored, reverse=True)[0][1]


def _model_metrics(rows: list[dict[str, Any]], model: str | None) -> dict[str, Any]:
    if model is None:
        return {family: aggregate_family_metrics([], "", family) for family in FAMILIES}
    return {family: aggregate_family_metrics(rows, model, family) for family in FAMILIES}


def fill_matrix(matrix: dict[str, Any], decision_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
    filled = json.loads(json.dumps(matrix))
    failed_model = select_failed_model(decision_rows)
    baseline_metrics = _model_metrics(decision_rows, "pre_sns_baseline")
    failed_metrics = _model_metrics(decision_rows, failed_model)
    if isinstance(filled.get("models"), dict):
        models = filled["models"]
        baseline = models.setdefault("pre_sns_baseline", {})
        baseline["source_model"] = "pre_sns_baseline"
        baseline["fill_source"] = "final_decision_report.md"
        for family, metrics in baseline_metrics.items():
            baseline[family] = metrics
        failed = models.setdefault("failed_single_model_finetune", {})
        failed["source_model"] = failed_model
        failed["fill_source"] = "final_decision_report.md"
        for family, metrics in failed_metrics.items():
            failed[family] = metrics
    if isinstance(filled.get("ablation_entries"), list):
        for entry in filled["ablation_entries"]:
            if not isinstance(entry, dict):
                continue
            if entry.get("system") == "pre_sns_baseline":
                entry["loaded"] = True
                entry["load_status"] = "filled_from_final_decision_report"
                entry["source_model"] = "pre_sns_baseline"
                entry["metrics"] = baseline_metrics
            if entry.get("system") == "failed_single_model_finetune":
                entry["loaded"] = failed_model is not None
                entry["load_status"] = "filled_from_final_decision_report" if failed_model else "no_failed_model_found"
                entry["source_model"] = failed_model
                entry["metrics"] = failed_metrics
    filled["marker"] = MARKER
    filled["ablation_fill"] = {
        "source": "final_decision_report.md",
        "pre_sns_baseline_source_model": "pre_sns_baseline",
        "failed_single_model_source_model": failed_model,
        "family_groups": FAMILY_PROFILES,
    }
    return filled, failed_model


def _metric_block(model: dict[str, Any], family: str) -> dict[str, Any]:
    block = model.get(family)
    return block if isinstance(block, dict) else {}


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def render_tsv(matrix: dict[str, Any]) -> str:
    columns = ["model", "family", "acc", "f1", "real_fpr", "synthetic_recall", "tampered_recall", "localization_activation", "valid_iou", "high_mask", "source_profile_count"]
    lines = ["\t".join(columns)]
    models = matrix.get("models") if isinstance(matrix.get("models"), dict) else {}
    for model_name in ("pre_sns_baseline", "failed_single_model_finetune", "mixed_feature_gate"):
        model = models.get(model_name, {}) if isinstance(models, dict) else {}
        for family in FAMILIES:
            metrics = _metric_block(model, family)
            lines.append(
                "\t".join(
                    [
                        model_name,
                        family,
                        _fmt(metrics.get("three_way_accuracy")),
                        _fmt(metrics.get("macro_f1")),
                        _fmt(metrics.get("real_fpr")),
                        _fmt(metrics.get("synthetic_recall")),
                        _fmt(metrics.get("tampered_recall")),
                        _fmt(metrics.get("localization_activation")),
                        _fmt(metrics.get("valid_iou")),
                        _fmt(metrics.get("high_mask")),
                        _fmt(metrics.get("source_profile_count")),
                    ]
                )
            )
    return "\n".join(lines) + "\n"


def render_markdown(matrix: dict[str, Any]) -> str:
    failed_model = matrix.get("ablation_fill", {}).get("failed_single_model_source_model")
    lines = [
        "# Final Cross-model Comparison Matrix Filled",
        "",
        MARKER,
        "",
        "## Summary",
        "",
        "The main SIDA-7B vs mixed_feature_gate comparison remains unchanged. This filled artifact only adds final-decision-report ablation metrics for the pre-SNS baseline and the selected failed single-model fine-tune.",
        "",
        f"Selected failed fine-tune source model: `{failed_model or 'NA'}`",
        "",
        "## Ablation / Context Table",
        "",
        "| Model | Family | Acc | F1 | Real FPR | Synthetic recall | Tampered recall | Loc act | Valid IoU | High mask | Source profiles |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    models = matrix.get("models") if isinstance(matrix.get("models"), dict) else {}
    for model_name in ("pre_sns_baseline", "failed_single_model_finetune", "mixed_feature_gate"):
        model = models.get(model_name, {}) if isinstance(models, dict) else {}
        for family in FAMILIES:
            metrics = _metric_block(model, family)
            lines.append(
                f"| {model_name} | {family} | {_fmt(metrics.get('three_way_accuracy'))} | {_fmt(metrics.get('macro_f1'))} | {_fmt(metrics.get('real_fpr'))} | {_fmt(metrics.get('synthetic_recall'))} | {_fmt(metrics.get('tampered_recall'))} | {_fmt(metrics.get('localization_activation'))} | {_fmt(metrics.get('valid_iou'))} | {_fmt(metrics.get('high_mask'))} | {_fmt(metrics.get('source_profile_count'))} |"
            )
    lines += [
        "",
        "## Interpretation Guardrails",
        "",
        "- Baseline/fine-tune values are ablation context, not the primary SIDA-vs-gate claim.",
        "- Missing profile groups remain NA rather than being invented.",
        "- mixed_feature_gate is not a drop-in replacement for SIDA.",
        "",
    ]
    return "\n".join(lines)


def render_claims(matrix: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Final Claims for Report Filled",
            "",
            MARKER,
            "",
            "## Supported",
            "",
            "- The pre-SNS baseline and failed fine-tune ablation rows can be filled from the final decision report where profile rows exist.",
            "- The selected failed fine-tune source model should be reported explicitly.",
            "- Type groups with no source rows remain NA/null.",
            "",
            "## Not Supported",
            "",
            "- Do not treat ablation rows as the main SIDA-vs-gate result.",
            "- Do not claim missing Type B source rows were measured in the final decision report if they were absent.",
            "- Do not claim mixed_feature_gate is a drop-in replacement for SIDA.",
            "",
        ]
    )


def output_paths(output_root: Path) -> dict[str, str]:
    return {key: str(output_root / name) for key, name in OUTPUT_NAMES.items()}


def _resolved_output_root(value: str | Path) -> Path:
    text = str(value)
    if "YYYYMMDD_HHMMSS" in text:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        text = text.replace("YYYYMMDD_HHMMSS", stamp)
    return _real(text)


def run_snsaug_v2_fill_final_cross_model_ablation_metrics(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    assert_valid_config(config, require_exists=not dry_run)
    comparison_root = resolve_comparison_output_root(config)
    decision_path = resolve_decision_report_path(config)
    output_root = _resolved_output_root(config["output_root"])
    paths = output_paths(output_root)
    plan = {
        "marker": MARKER,
        "dry_run": dry_run,
        "comparison_output_root": str(comparison_root),
        "decision_report_path": str(decision_path),
        "output_root": str(output_root),
        "output_paths": paths,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    if dry_run:
        return plan
    matrix = _load_json(comparison_root / "final_cross_model_comparison_matrix.json")
    decision_rows = parse_final_decision_markdown_table(decision_path.read_text(encoding="utf-8"))
    filled, failed_model = fill_matrix(matrix, decision_rows)
    filled["source_roots"] = {
        **(filled.get("source_roots") if isinstance(filled.get("source_roots"), dict) else {}),
        "0077_final_cross_model_comparison": str(comparison_root),
        "final_decision_report": str(decision_path.parent),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(Path(paths["matrix"]), filled)
    _write_text(Path(paths["report"]), render_markdown(filled))
    _write_text(Path(paths["table"]), render_tsv(filled))
    _write_text(Path(paths["claims"]), render_claims(filled))
    manifest = {
        **plan,
        "dry_run": False,
        "analysis_started": True,
        "training_started": False,
        "download_started": False,
        "network_used": False,
        "selected_failed_model": failed_model,
        "parsed_decision_rows": len(decision_rows),
        "artifact_count": 5,
    }
    _write_json(Path(paths["manifest"]), manifest)
    return manifest


__all__ = [
    "APPROVAL_TEXT",
    "CONFIG_OK_MARKER",
    "MARKER",
    "SNSAugV2FillFinalCrossModelAblationMetricsError",
    "aggregate_family_metrics",
    "fill_matrix",
    "load_snsaug_v2_fill_final_cross_model_ablation_metrics_config",
    "parse_final_decision_markdown_table",
    "run_snsaug_v2_fill_final_cross_model_ablation_metrics",
    "select_failed_model",
    "validate_snsaug_v2_fill_final_cross_model_ablation_metrics_config",
]
