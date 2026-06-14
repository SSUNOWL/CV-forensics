"""Final SIDA addendum and SNS-aware future-work report."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MARKER = "SNSAUG_V2_FINAL_SIDA_ADDENDUM_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_FINAL_SIDA_ADDENDUM_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_final_sida_addendum"
APPROVED_MODE = "approved_local_snsaug_v2_final_sida_addendum"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_FINAL_SIDA_ADDENDUM"
SIDA_MARKER = "SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE_OK"
GATE_MARKER = "SNSAUG_V2_DEPLOYABLE_POLICY_GATE_REPORT_OK"
REPO_ROOT = Path(__file__).resolve().parents[2]

STRICT_PROFILE_FAMILIES = {
    "news_meme_overlay": "type_a_local_overlay",
    "platform_ui_same_size": "type_a_local_overlay",
    "tiktok_like": "type_a_local_overlay",
    "instagram_story_like": "type_a_local_overlay",
    "youtube_shorts_like": "type_a_local_overlay",
    "combined_sns_realistic": "type_a_local_overlay",
    "canvas_9x16_only": "type_b_global_geometry_degradation",
    "resize_crop_pad": "type_b_global_geometry_degradation",
    "zoom_crop": "type_b_global_geometry_degradation",
    "resize_jpeg": "type_b_global_geometry_degradation",
    "screenshot_recapture_light": "type_b_global_geometry_degradation",
    "recompression_light": "type_b_global_geometry_degradation",
}


class SNSAugV2FinalSIDAAddendumError(ValueError):
    """Raised when final SIDA addendum validation or execution fails."""


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


def load_snsaug_v2_final_sida_addendum_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2FinalSIDAAddendumError("final SIDA addendum config root must be a JSON object")
    raw.setdefault("config_path", str(_real(path)))
    return raw


def validate_snsaug_v2_final_sida_addendum_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    for field in (
        "schema_version",
        "config_kind",
        "execution_mode",
        "user_approval_text",
        "sida_output_root",
        "mixed_gate_output_root",
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
    for field in ("sida_output_root", "mixed_gate_output_root"):
        errors.extend(_under_errors(raw.get(field), field, input_roots, require_exists=require_exists))
    if require_exists:
        sida_root = _real(raw.get("sida_output_root", ""))
        gate_root = _real(raw.get("mixed_gate_output_root", ""))
        for name in (
            "sida7b_corrected_type_a_type_b_summary.json",
            "sida7b_clean_safe_vs_mixed_gate_comparison.json",
            "sida7b_raw_output_audit.json",
            "artifact_manifest.json",
        ):
            if not (sida_root / name).exists():
                errors.append(f"sida_output_root missing {name}")
        for name in ("final_policy_gate_metrics.json", "final_policy_gate_report.md", "artifact_manifest.json"):
            if not (gate_root / name).exists():
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
    errors = validate_snsaug_v2_final_sida_addendum_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2FinalSIDAAddendumError("final SIDA addendum config validation failed:\n" + "\n".join(errors))


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
        raise SNSAugV2FinalSIDAAddendumError(f"{path} must contain a JSON object")
    return value


def _paths(output_root: Path) -> dict[str, str]:
    return {
        "final_sida_addendum_report": str(output_root / "final_sida_addendum_report.md"),
        "final_sida_addendum_notion_summary": str(output_root / "final_sida_addendum_notion_summary.md"),
        "final_sida_vs_mixed_gate_table": str(output_root / "final_sida_vs_mixed_gate_table.tsv"),
        "future_work_snsaware_model": str(output_root / "future_work_snsaware_model.md"),
        "artifact_manifest": str(output_root / "artifact_manifest.json"),
    }


def load_inputs(sida_output_root: str | Path, mixed_gate_output_root: str | Path) -> dict[str, Any]:
    sida_root = _real(sida_output_root)
    gate_root = _real(mixed_gate_output_root)
    sida_artifact = _load_json(sida_root / "artifact_manifest.json")
    gate_artifact = _load_json(gate_root / "artifact_manifest.json")
    if sida_artifact.get("marker") != SIDA_MARKER:
        raise SNSAugV2FinalSIDAAddendumError("SIDA artifact marker is not a SIDA diagnostic baseline artifact")
    if gate_artifact.get("marker") != GATE_MARKER:
        raise SNSAugV2FinalSIDAAddendumError("mixed-gate artifact marker is not a deployable policy gate artifact")
    return {
        "sida_root": str(sida_root),
        "mixed_gate_root": str(gate_root),
        "sida_summary": _load_json(sida_root / "sida7b_corrected_type_a_type_b_summary.json"),
        "comparison": _load_json(sida_root / "sida7b_clean_safe_vs_mixed_gate_comparison.json"),
        "raw_audit": _load_json(sida_root / "sida7b_raw_output_audit.json"),
        "sida_artifact": sida_artifact,
        "gate_metrics": _load_json(gate_root / "final_policy_gate_metrics.json"),
        "gate_report_text": (gate_root / "final_policy_gate_report.md").read_text(encoding="utf-8"),
        "gate_artifact": gate_artifact,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metric(summary: dict[str, Any], key: str, metric: str) -> Any:
    item = summary.get("summary", {}).get(key, {})
    return item.get(metric) if isinstance(item, dict) else None


def build_comparison_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profiles = comparison.get("profiles") if isinstance(comparison.get("profiles"), dict) else {}
    for profile, row in sorted(profiles.items()):
        if profile == "clean" or profile not in STRICT_PROFILE_FAMILIES or not isinstance(row, dict):
            continue
        rows.append(
            {
                "profile": profile,
                "profile_family": STRICT_PROFILE_FAMILIES[profile],
                "sida_tampered_recall": row.get("sida_tampered_recall"),
                "sida_synthetic_recall": row.get("sida_synthetic_recall"),
                "sida_real_fpr": row.get("sida_real_fpr"),
                "sida_valid_iou": row.get("sida_valid_iou"),
                "mixed_gate_tampered_recall": row.get("mixed_gate_tampered_recall"),
                "mixed_gate_synthetic_recall": row.get("mixed_gate_synthetic_recall"),
                "mixed_gate_real_fpr": row.get("mixed_gate_real_fpr"),
                "mixed_gate_valid_iou": row.get("mixed_gate_valid_iou"),
                "tampered_recall_delta_sida_minus_gate": row.get("tampered_recall_delta_sida_minus_gate"),
                "valid_iou_delta_sida_minus_gate": row.get("valid_iou_delta_sida_minus_gate"),
            }
        )
    return rows


def render_tsv(rows: list[dict[str, Any]]) -> str:
    columns = [
        "profile",
        "profile_family",
        "sida_tampered_recall",
        "sida_synthetic_recall",
        "sida_real_fpr",
        "sida_valid_iou",
        "mixed_gate_tampered_recall",
        "mixed_gate_synthetic_recall",
        "mixed_gate_real_fpr",
        "mixed_gate_valid_iou",
        "tampered_recall_delta_sida_minus_gate",
        "valid_iou_delta_sida_minus_gate",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(_fmt(row.get(column)) for column in columns))
    return "\n".join(lines) + "\n"


def _future_work_text() -> str:
    return "\n".join(
        [
            "# Future Work: SNS-Aware Forensic Model",
            "",
            MARKER,
            "",
            "## SIDA Rerun",
            "",
            "Rerun with SIDA-13B or a SIDA-description configuration that reliably emits segmentation or mask outputs. This is needed before making any SIDA localization-performance claim.",
            "",
            "## VLM-Guided Nuisance Preprocessor",
            "",
            "Use a VLM-guided nuisance preprocessor to identify social UI overlays, captions, stickers, borders, screenshot frames, and other nuisance regions before lightweight forensic inference.",
            "",
            "## SNS-Aware Dual-Branch Model",
            "",
            "Build a dual-branch model with a local nuisance branch for overlay/UI artifacts and a geometry/degradation residual branch for resize, crop, JPEG, recapture, and recompression shifts.",
            "",
            "## Distillation",
            "",
            "Distill SIDA-like VLM judgments into the lightweight mixed gate and residual model so the deployable system gains explanation-aware supervision without requiring a large VLM at inference time.",
            "",
        ]
    )


def render_report(inputs: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    summary = inputs["sida_summary"]
    comparison = inputs["comparison"]
    raw = inputs["raw_audit"]
    findings = summary.get("decision_findings", [])
    type_a_t = _metric(summary, "type_a_local_overlay", "tampered_recall")
    type_b_t = _metric(summary, "type_b_global_geometry_degradation", "tampered_recall")
    type_a_s = _metric(summary, "type_a_local_overlay", "synthetic_recall")
    type_b_s = _metric(summary, "type_b_global_geometry_degradation", "synthetic_recall")
    strict_available = comparison.get("strict_comparison_available")
    clean_missing = comparison.get("missing_reference_only_profiles", [])
    lines = [
        "# Final SIDA Addendum and SNSAug Future Work",
        "",
        MARKER,
        "",
        "## What SIDA-7B Was Used For",
        "",
        "SIDA-7B was used as an offline SIDA-style VLM diagnostic on the SNSAug evaluation subset. Its role was to test whether a large social-media forensic model shows the same Type A local-overlay and Type B global geometry/degradation sensitivity seen in the lightweight SNSAug experiments.",
        "",
        "## What Was Actually Measured",
        "",
        "The cached run measured text-output classification behavior only: parsed `real`, `synthetic`, and `tampered` labels, per-type accuracy/recall/error rates, raw output term counts, and strict Type A/B comparison against the 0072 deployable `mixed_feature_gate`.",
        "",
        "## What Was Not Measured",
        "",
        "SIDA localization performance was not measured because cached mask paths are missing. Any `[SEG]` text without a usable mask is treated as unavailable mask extraction, not as a localization result.",
        "",
        "## Type A vs Type B SIDA Classification",
        "",
        f"- Type A local overlay tampered recall: `{_fmt(type_a_t)}`",
        f"- Type B global geometry/degradation tampered recall: `{_fmt(type_b_t)}`",
        f"- Type A synthetic recall: `{_fmt(type_a_s)}`",
        f"- Type B synthetic recall: `{_fmt(type_b_s)}`",
        f"- Raw synthetic phrase count: `{raw.get('raw_contains_synthetic_count', 0)}`",
        f"- Raw generated phrase count: `{raw.get('raw_contains_generated_count', 0)}`",
        "",
        "The cached text outputs show synthetic-recall collapse for this parser/output format because no synthetic/generated phrasing was present in the raw outputs. Type A local overlays have lower SIDA tampered recall than Type B global geometry/degradation in the corrected SIDA summary.",
        "",
        "## Comparison With 0072 mixed_feature_gate",
        "",
        f"- Strict Type A/B comparison available: `{strict_available}`",
        f"- Missing strict mixed-gate profiles: `{comparison.get('missing_mixed_gate_profiles', [])}`",
        f"- Clean reference-only missing profiles: `{clean_missing}`",
        "",
        "The strict comparison uses only the 12 Type A/B SNSAug perturbation profiles. `clean` is SIDA reference context and is excluded from strict mixed-gate availability checks.",
        "",
        "| Profile | Family | SIDA Tampered Recall | Mixed Gate Tampered Recall | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {row['profile_family']} | {_fmt(row.get('sida_tampered_recall'))} | {_fmt(row.get('mixed_gate_tampered_recall'))} | {_fmt(row.get('tampered_recall_delta_sida_minus_gate'))} |"
        )
    lines += [
        "",
        "## Decision Findings",
        "",
    ]
    for finding in findings:
        lines.append(f"- `{finding}`")
    lines += [
        "",
        "## Future Work",
        "",
        "- Rerun SIDA-13B or a SIDA-description setup for better mask output before localization claims.",
        "- Add a VLM-guided nuisance preprocessor for social overlays and UI regions.",
        "- Build an SNS-aware dual-branch model with a local nuisance branch plus a geometry/degradation residual branch.",
        "- Distill SIDA-like VLM behavior into the lightweight mixed gate / residual model.",
        "",
    ]
    return "\n".join(lines)


def render_notion_summary(inputs: dict[str, Any]) -> str:
    summary = inputs["sida_summary"]
    comparison = inputs["comparison"]
    return "\n".join(
        [
            "# Final SIDA Addendum Summary",
            "",
            MARKER,
            "",
            "0072 `mixed_feature_gate` remains the practical offline SNSAug recovery artifact.",
            "",
            "The SIDA cached run is a classification diagnostic only because masks are missing; no SIDA localization-performance claim is made.",
            "",
            f"Type A tampered recall: `{_fmt(_metric(summary, 'type_a_local_overlay', 'tampered_recall'))}`",
            f"Type B tampered recall: `{_fmt(_metric(summary, 'type_b_global_geometry_degradation', 'tampered_recall'))}`",
            "",
            "Synthetic recall collapsed to 0.0 for the cached SIDA text outputs because synthetic/generated phrases were absent from raw output.",
            "",
            f"Strict Type A/B mixed-gate comparison available: `{comparison.get('strict_comparison_available')}`",
            "Clean is reference-only for SIDA and excluded from strict mixed-gate comparison.",
            "",
        ]
    )


def run_snsaug_v2_final_sida_addendum(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    assert_valid_config(config, require_exists=not dry_run)
    output_root = _real(config["output_root"])
    paths = _paths(output_root)
    plan = {
        "marker": MARKER,
        "dry_run": dry_run,
        "sida_output_root": str(_real(config["sida_output_root"])),
        "mixed_gate_output_root": str(_real(config["mixed_gate_output_root"])),
        "output_root": str(output_root),
        "output_paths": paths,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }
    if dry_run:
        return plan
    inputs = load_inputs(config["sida_output_root"], config["mixed_gate_output_root"])
    rows = build_comparison_rows(inputs["comparison"])
    output_root.mkdir(parents=True, exist_ok=True)
    _write_text(Path(paths["final_sida_addendum_report"]), render_report(inputs, rows))
    _write_text(Path(paths["final_sida_addendum_notion_summary"]), render_notion_summary(inputs))
    _write_text(Path(paths["final_sida_vs_mixed_gate_table"]), render_tsv(rows))
    _write_text(Path(paths["future_work_snsaware_model"]), _future_work_text())
    artifact = {
        **plan,
        "dry_run": False,
        "analysis_started": True,
        "training_started": False,
        "download_started": False,
        "network_used": False,
        "decision_findings": inputs["sida_summary"].get("decision_findings", []),
        "strict_comparison_available": inputs["comparison"].get("strict_comparison_available"),
        "comparison_profile_count": len(rows),
        "clean_reference_only": "clean" in inputs["comparison"].get("reference_only_profiles", []),
        "input_sida_marker": inputs["sida_artifact"].get("marker"),
        "input_mixed_gate_marker": inputs["gate_artifact"].get("marker"),
    }
    _write_json(Path(paths["artifact_manifest"]), artifact)
    return artifact


__all__ = [
    "APPROVAL_TEXT",
    "CONFIG_OK_MARKER",
    "MARKER",
    "SNSAugV2FinalSIDAAddendumError",
    "build_comparison_rows",
    "load_snsaug_v2_final_sida_addendum_config",
    "render_report",
    "render_tsv",
    "run_snsaug_v2_final_sida_addendum",
    "validate_snsaug_v2_final_sida_addendum_config",
]
