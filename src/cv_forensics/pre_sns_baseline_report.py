"""Pre-SNS baseline report helpers."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER = "PRE_SNS_BASELINE_REPORT_OK"
CONFIG_OK_MARKER = "PRE_SNS_BASELINE_REPORT_CONFIG_OK"
RESULT_OK_MARKER = "PRE_SNS_BASELINE_REPORT_RESULT_OK"
APPROVAL_TEXT = "I_APPROVE_PRE_SNS_BASELINE_REPORT"
CONFIG_KINDS = {"example_symbolic", "approved_pre_sns_baseline_report", "approved_local_pre_sns_baseline_report"}
REMOTE_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
SECRET_KEY_RE = re.compile(r"(^|_)(api[_-]?key|secret|token|password|credential|private[_-]?key)($|_)", re.I)
SECRET_VALUE_RE = re.compile(r"(api[_-]?key|secret|password|token=|bearer |private[_-]?key)", re.I)
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}
RESULT_PATH_FIELDS = (
    "training_result_path",
    "inference_report_path",
    "evaluation_result_path",
    "scaled_training_result_path",
)
REQUIRED_FIELDS = (
    "schema_version",
    "config_kind",
    "execution_mode",
    "required_approval_text",
    "user_approval_text",
    *RESULT_PATH_FIELDS,
    "approved_report_root",
    "write_report",
    "no_download",
    "no_network",
    "no_training",
    "no_inference",
    "no_evaluation",
    "no_checkpoint_writes",
    "no_sns_augmentation",
    "result_scope",
)
REQUIRED_SECTIONS = (
    "Training summary",
    "Single-image report summary",
    "Evaluation metrics",
    "Scaled training summary",
    "Pre-SNS limitations",
    "Next SNS augmentation step",
)
EVALUATION_METRICS = (
    "accuracy_3way",
    "macro_f1_3way",
    "family_accuracy",
    "mask_iou_mean",
    "localization_activation_recall",
    "latency_ms_mean",
    "fps_estimate",
)


class BaselineReportError(ValueError):
    """Raised when a pre-SNS baseline report input is invalid."""


def _err(message: str) -> str:
    return f"- {message}"


def _real(path: str | Path) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def _is_under(path: str | Path, root: str | Path) -> bool:
    try:
        return os.path.commonpath([str(_real(path)), str(_real(root))]) == str(_real(root))
    except ValueError:
        return False


def _inside_repo(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT)


def _repo_outputs_or_checkpoints(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT / "outputs") or _is_under(path, REPO_ROOT / "checkpoints")


def _has_path_traversal(value: str) -> bool:
    return any(part == ".." for part in value.replace("\\", "/").split("/"))


def _contains_protected_part(value: str) -> bool:
    return any(part in PROTECTED_PARTS for part in value.replace("\\", "/").split("/") if part)


def _walk_safety(value: Any, path: str = "", allowed_abs_values: set[str] | None = None) -> list[str]:
    allowed_abs_values = allowed_abs_values or set()
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if SECRET_KEY_RE.search(key_text):
                errors.append(_err(f"{child_path}: secret-like key rejected"))
            errors.extend(_walk_safety(child, child_path, allowed_abs_values))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_walk_safety(child, f"{path}[{index}]", allowed_abs_values))
    elif isinstance(value, str):
        text = value.strip()
        if REMOTE_RE.search(text):
            errors.append(_err(f"{path}: URL or remote scheme rejected"))
        if WINDOWS_DRIVE_RE.search(text):
            errors.append(_err(f"{path}: Windows drive path rejected"))
        if text in allowed_abs_values:
            if _has_path_traversal(text):
                errors.append(_err(f"{path}: path traversal rejected"))
            if _repo_outputs_or_checkpoints(text):
                errors.append(_err(f"{path}: repository outputs/checkpoints path rejected"))
            if _contains_protected_part(text):
                errors.append(_err(f"{path}: protected path segment rejected"))
            if SECRET_VALUE_RE.search(text):
                errors.append(_err(f"{path}: secret-like value rejected"))
            return errors
        if text.startswith("/"):
            errors.append(_err(f"{path}: absolute path rejected"))
        if _has_path_traversal(text):
            errors.append(_err(f"{path}: path traversal rejected"))
        if _contains_protected_part(text):
            errors.append(_err(f"{path}: protected path segment rejected"))
        if SECRET_VALUE_RE.search(text):
            errors.append(_err(f"{path}: secret-like value rejected"))
    return errors


def load_report_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise BaselineReportError("report config root must be a JSON object")
    return raw


def parse_json_with_logs(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise BaselineReportError(f"{path} does not contain a JSON object")


def _validate_explicit_json_path(path_value: Any, field: str, require_exists: bool) -> list[str]:
    errors: list[str] = []
    if not isinstance(path_value, str) or not path_value.strip():
        return [_err(f"{field} must be a non-empty explicit local JSON path")]
    text = path_value.strip()
    if REMOTE_RE.search(text):
        errors.append(_err(f"{field} must not be a URL or remote scheme"))
    if WINDOWS_DRIVE_RE.search(text):
        errors.append(_err(f"{field} must not be a Windows drive path"))
    if not text.startswith("/"):
        errors.append(_err(f"{field} must be absolute in approved local mode"))
    if _has_path_traversal(text):
        errors.append(_err(f"{field} must not contain path traversal"))
    if _contains_protected_part(text):
        errors.append(_err(f"{field} must not contain protected path segments"))
    if _repo_outputs_or_checkpoints(text):
        errors.append(_err(f"{field} must not be inside repository outputs/checkpoints"))
    if require_exists and not os.path.isfile(text):
        errors.append(_err(f"{field} must exist as a file"))
    return errors


def validate_report_root(path_value: Any, require_parent: bool) -> list[str]:
    errors: list[str] = []
    if not isinstance(path_value, str) or not path_value.strip():
        return [_err("approved_report_root must be a non-empty local directory path")]
    text = path_value.strip()
    if REMOTE_RE.search(text):
        errors.append(_err("approved_report_root must not be a URL or remote scheme"))
    if WINDOWS_DRIVE_RE.search(text):
        errors.append(_err("approved_report_root must not be a Windows drive path"))
    if not text.startswith("/"):
        errors.append(_err("approved_report_root must be absolute in approved local mode"))
    if _has_path_traversal(text):
        errors.append(_err("approved_report_root must not contain path traversal"))
    if _contains_protected_part(text):
        errors.append(_err("approved_report_root must not contain protected path segments"))
    if text.startswith("/") and _inside_repo(text):
        errors.append(_err("approved_report_root must be outside the repository"))
    if _repo_outputs_or_checkpoints(text):
        errors.append(_err("approved_report_root must not be inside repository outputs/checkpoints"))
    if require_parent:
        parent = Path(text).parent
        current = parent
        while not current.exists() and current != current.parent:
            current = current.parent
        if not current.is_dir() or not os.access(current, os.W_OK):
            errors.append(_err("approved_report_root parent must be valid or creatable"))
    return errors


def validate_report_config(raw: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    kind = raw.get("config_kind")
    if kind not in CONFIG_KINDS:
        errors.append(_err("config_kind must be example_symbolic or approved_pre_sns_baseline_report"))
        errors.extend(_walk_safety(raw))
        return errors
    for flag in ("no_download", "no_network", "no_training", "no_inference", "no_evaluation", "no_checkpoint_writes", "no_sns_augmentation"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if raw.get("required_approval_text") != APPROVAL_TEXT:
        errors.append(_err("required_approval_text must document the exact approval phrase"))
    if raw.get("write_report") not in {True, False}:
        errors.append(_err("write_report must be boolean"))
    if raw.get("recursive_scan") is True or raw.get("recursive_directory_scan") is True:
        errors.append(_err("recursive scan flags are rejected"))
    if kind == "example_symbolic":
        if raw.get("execution_mode") != "example_only":
            errors.append(_err("execution_mode must be example_only"))
        if raw.get("user_approval_text") not in {"", APPROVAL_TEXT}:
            errors.append(_err("example user_approval_text must be empty or the documented approval phrase"))
        errors.extend(_walk_safety(raw))
        return errors
    if raw.get("execution_mode") != "approved_local_pre_sns_baseline_report":
        errors.append(_err("execution_mode must be approved_local_pre_sns_baseline_report"))
    if raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(_err("user_approval_text does not match required approval phrase"))
    allowed_abs = {raw[field] for field in (*RESULT_PATH_FIELDS, "approved_report_root") if isinstance(raw.get(field), str)}
    errors.extend(_walk_safety(raw, allowed_abs_values=allowed_abs))
    for field in RESULT_PATH_FIELDS:
        errors.extend(_validate_explicit_json_path(raw.get(field), field, require_exists=True))
    errors.extend(validate_report_root(raw.get("approved_report_root"), require_parent=True))
    return errors


def _dig(source: dict[str, Any], paths: tuple[tuple[str, ...], ...], default: Any = None) -> Any:
    for path in paths:
        current: Any = source
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return current
    return default


def sanitize_checkpoint_path(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {"basename": None, "root_category": "not_reported"}
    path = Path(value)
    category = "relative_or_symbolic"
    if path.is_absolute():
        if _is_under(path, Path.home() / "cvf_checkpoints"):
            category = "home_cvf_checkpoints"
        elif _inside_repo(path):
            category = "inside_repository"
        else:
            category = "outside_repository"
    return {"basename": path.name, "root_category": category}


def _numeric_or_none(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return value


def extract_training_summary(raw: dict[str, Any]) -> dict[str, Any]:
    checkpoint = _dig(raw, (("checkpoint_path",), ("artifact_manifest", "checkpoint_path"), ("artifacts", "checkpoint_path")))
    return {
        "samples": _dig(raw, (("samples_seen",), ("samples_evaluated",), ("max_samples",), ("metrics", "samples_seen"))),
        "steps": _dig(raw, (("steps_completed",), ("training_steps",), ("metrics", "steps_completed"))),
        "epochs": _dig(raw, (("epochs",), ("epochs_completed",), ("config", "epochs"))),
        "initial_loss": _numeric_or_none(_dig(raw, (("initial_total_loss",), ("initial_loss",), ("metrics", "initial_total_loss")))),
        "final_loss": _numeric_or_none(_dig(raw, (("final_total_loss",), ("final_loss",), ("metrics", "final_total_loss")))),
        "checkpoint": sanitize_checkpoint_path(checkpoint),
    }


def extract_inference_summary(raw: dict[str, Any]) -> dict[str, Any]:
    required = ("class", "family", "localization_head", "reason")
    missing = [field for field in required if field not in raw]
    if missing:
        raise BaselineReportError("inference report missing required fields: " + ", ".join(missing))
    return {
        "class": raw.get("class"),
        "family": raw.get("family"),
        "localization_head": raw.get("localization_head"),
        "mask_area_pct": raw.get("mask_area_pct"),
        "reason": raw.get("reason"),
        "tampered_score": raw.get("tampered_score"),
    }


def extract_evaluation_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in EVALUATION_METRICS if field not in raw]
    if missing:
        raise BaselineReportError("evaluation result missing required metrics: " + ", ".join(missing))
    return {field: raw.get(field) for field in EVALUATION_METRICS}


def build_summary(training: dict[str, Any], inference: dict[str, Any], evaluation: dict[str, Any], scaled: dict[str, Any], result_scope: str) -> dict[str, Any]:
    return {
        "marker": MARKER,
        "training_summary": extract_training_summary(training),
        "single_image_report_summary": extract_inference_summary(inference),
        "evaluation_metrics": extract_evaluation_metrics(evaluation),
        "scaled_training_summary": extract_training_summary(scaled),
        "required_sections": list(REQUIRED_SECTIONS),
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_inference": True,
        "no_evaluation": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
        "sns_augmentation_status": "not_applied",
        "result_scope": result_scope,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    train = summary["training_summary"]
    infer = summary["single_image_report_summary"]
    eval_metrics = summary["evaluation_metrics"]
    scaled = summary["scaled_training_summary"]
    return "\n".join(
        [
            "# Pre-SNS Baseline Report",
            "",
            MARKER,
            "",
            "## Training summary",
            f"- Samples: {train.get('samples')}",
            f"- Steps: {train.get('steps')}",
            f"- Epochs: {train.get('epochs')}",
            f"- Initial loss: {train.get('initial_loss')}",
            f"- Final loss: {train.get('final_loss')}",
            f"- Checkpoint: {train.get('checkpoint', {}).get('basename')} ({train.get('checkpoint', {}).get('root_category')})",
            "",
            "## Single-image report summary",
            f"- Class: {infer.get('class')}",
            f"- Family: {infer.get('family')}",
            f"- Localization: {infer.get('localization_head')}",
            f"- Mask area pct: {infer.get('mask_area_pct')}",
            f"- Reason: {infer.get('reason')}",
            "",
            "## Evaluation metrics",
            f"- 3-way accuracy: {eval_metrics.get('accuracy_3way')}",
            f"- Macro-F1: {eval_metrics.get('macro_f1_3way')}",
            f"- Family accuracy: {eval_metrics.get('family_accuracy')}",
            f"- Mask IoU mean: {eval_metrics.get('mask_iou_mean')}",
            f"- Localization activation recall: {eval_metrics.get('localization_activation_recall')}",
            f"- Latency ms mean: {eval_metrics.get('latency_ms_mean')}",
            f"- FPS estimate: {eval_metrics.get('fps_estimate')}",
            "",
            "## Scaled training summary",
            f"- Samples: {scaled.get('samples')}",
            f"- Steps: {scaled.get('steps')}",
            f"- Epochs: {scaled.get('epochs')}",
            f"- Initial loss: {scaled.get('initial_loss')}",
            f"- Final loss: {scaled.get('final_loss')}",
            "",
            "## Pre-SNS limitations",
            "SNS augmentation has not been applied. These results describe the pre-SNS baseline only and are not a final full-dataset claim unless separately approved.",
            "",
            "## Next SNS augmentation step",
            "After this baseline is reviewed, the next phase may add controlled SNS perturbation robustness evaluation.",
            "",
        ]
    )


def generate_baseline_report(raw: dict[str, Any]) -> dict[str, Any]:
    errors = validate_report_config(raw)
    if errors:
        raise BaselineReportError("baseline report config validation failed:\n" + "\n".join(errors))
    if raw.get("config_kind") == "example_symbolic":
        raise BaselineReportError("report generation requires an approved local config")
    loaded = {field: parse_json_with_logs(raw[field]) for field in RESULT_PATH_FIELDS}
    summary = build_summary(
        loaded["training_result_path"],
        loaded["inference_report_path"],
        loaded["evaluation_result_path"],
        loaded["scaled_training_result_path"],
        str(raw.get("result_scope", "pre-SNS baseline report")),
    )
    markdown = render_markdown(summary)
    summary["markdown"] = markdown
    summary["write_report"] = bool(raw.get("write_report"))
    if raw.get("write_report") is True:
        md_path, json_path = write_report_artifacts(raw["approved_report_root"], markdown, summary)
        summary["report_markdown_path"] = str(md_path)
        summary["report_summary_path"] = str(json_path)
    return _json_safe(summary)


def write_report_artifacts(report_root: str | Path, markdown: str, summary: dict[str, Any]) -> tuple[Path, Path]:
    errors = validate_report_root(str(report_root), require_parent=True)
    if errors:
        raise BaselineReportError("\n".join(errors))
    root = _real(report_root)
    root.mkdir(parents=True, exist_ok=True)
    markdown_path = root / "pre_sns_baseline_report.md"
    summary_path = root / "pre_sns_baseline_summary.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({key: value for key, value in summary.items() if key != "markdown"}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return markdown_path, summary_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value
