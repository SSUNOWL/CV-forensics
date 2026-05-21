"""Pre-SNS single-image inference report helpers."""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from .explanation_templates import generate_reason
from .model_output_schema import (
    EVIDENCE_BOUNDARY_DISCONTINUITY,
    EVIDENCE_GLOBAL_SYNTHETIC_ARTIFACT,
    EVIDENCE_LOCAL_MASK_ACTIVATION,
    EVIDENCE_PROVENANCE_FAMILY_SIGNAL,
    LOCALIZATION_ACTIVATED,
    LOCALIZATION_NOT_APPLICABLE,
    LOCALIZATION_SKIPPED_BELOW_THRESHOLD,
    PERTURBATION_NONE,
    SCHEMA_VERSION,
)
from .pre_sns_integrated_model import CLASS_LABELS, FAMILY_SMOKE_LABELS, build_tiny_integrated_model, schema_class_label

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER = "PRE_SNS_SINGLE_IMAGE_REPORT_OK"
CONFIG_OK_MARKER = "PRE_SNS_INFERENCE_REPORT_CONFIG_OK"
REPORT_VALIDATED_MARKER = "PRE_SNS_SINGLE_IMAGE_REPORT_VALIDATED_OK"
APPROVAL_TEXT = "I_APPROVE_PRE_SNS_SINGLE_IMAGE_REPORT"
APPROVED_CONFIG_KIND = "approved_pre_sns_single_image_report"
LEGACY_APPROVED_CONFIG_KINDS = {"approved_local_pre_sns_single_image_report"}
CONFIG_KINDS = {"example_symbolic", APPROVED_CONFIG_KIND, *LEGACY_APPROVED_CONFIG_KINDS}
REMOTE_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
SECRET_KEY_RE = re.compile(r"(^|_)(api[_-]?key|secret|token|password|credential|private[_-]?key)($|_)", re.I)
SECRET_VALUE_RE = re.compile(r"(api[_-]?key|secret|password|token=|bearer |private[_-]?key)", re.I)
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}
APPROVED_PATH_FIELDS = {"image_path", "checkpoint_path", "report_root"}
APPROVED_ROOT_FIELDS = {
    "approved_image_roots",
    "approved_checkpoint_roots",
    "approved_report_roots",
    "approved_local_roots",
}
REQUIRED_FIELDS = (
    "schema_version",
    "config_kind",
    "execution_mode",
    "no_download",
    "no_network",
    "no_training",
    "no_checkpoint_writes",
    "no_sns_augmentation",
    "checkpoint_path",
    "image_path",
    "report_root",
    "write_report",
    "device",
    "cuda_device_index",
    "max_image_size",
    "threshold_tau",
    "class_labels",
    "family_labels",
    "required_approval_text",
    "user_approval_text",
    "result_scope",
)


class InferenceReportError(ValueError):
    """Raised when a pre-SNS inference config or runtime input is invalid."""


def _err(message: str) -> str:
    return f"- {message}"


def _real(path: str | Path) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def _is_under(path: str | Path, root: str | Path) -> bool:
    path_real = _real(path)
    root_real = _real(root)
    try:
        return os.path.commonpath([str(path_real), str(root_real)]) == str(root_real)
    except ValueError:
        return False


def _has_path_traversal(value: str) -> bool:
    return any(part == ".." for part in value.replace("\\", "/").split("/"))


def _parts(value: str) -> list[str]:
    return [part for part in value.replace("\\", "/").split("/") if part]


def _contains_protected_part(value: str, allowed_parts: set[str] | None = None) -> bool:
    allowed_parts = allowed_parts or set()
    return any(part in PROTECTED_PARTS and part not in allowed_parts for part in _parts(value))


def _repo_outputs_or_checkpoints(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT / "outputs") or _is_under(path, REPO_ROOT / "checkpoints")


def _inside_repo(path: str | Path) -> bool:
    return _is_under(path, REPO_ROOT)


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
            if _contains_protected_part(text, {"checkpoints"}):
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


def _string_list(raw: dict[str, Any], field: str) -> tuple[list[str], list[str]]:
    value = raw.get(field)
    if value is None:
        return [], []
    if not isinstance(value, list):
        return [], [_err(f"{field} must be a list of absolute local roots")]
    roots: list[str] = []
    errors: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(_err(f"{field}[{index}] must be a non-empty string"))
        else:
            roots.append(item.strip())
    return roots, errors


def _validate_approved_root(root: str, field: str, allow_checkpoint_part: bool = False, report_root_policy: bool = False) -> list[str]:
    errors: list[str] = []
    if REMOTE_RE.search(root):
        errors.append(_err(f"{field} must not contain URL or remote roots"))
    if WINDOWS_DRIVE_RE.search(root):
        errors.append(_err(f"{field} must not contain Windows drive roots"))
    if not root.startswith("/"):
        errors.append(_err(f"{field} roots must be absolute local paths"))
    if _has_path_traversal(root):
        errors.append(_err(f"{field} roots must not contain path traversal"))
    allowed_parts = {"checkpoints"} if allow_checkpoint_part else set()
    if _contains_protected_part(root, allowed_parts):
        errors.append(_err(f"{field} roots must not contain protected path segments"))
    if SECRET_VALUE_RE.search(root):
        errors.append(_err(f"{field} roots must not contain secret-like values"))
    if report_root_policy and root.startswith("/"):
        if _inside_repo(root):
            errors.append(_err(f"{field} roots must be outside the repository"))
        if _repo_outputs_or_checkpoints(root):
            errors.append(_err(f"{field} roots must not be inside repository outputs/checkpoints"))
    return errors


def _approved_roots_for(raw: dict[str, Any], specific_field: str) -> tuple[list[str], list[str]]:
    specific_roots, specific_errors = _string_list(raw, specific_field)
    local_roots, local_errors = _string_list(raw, "approved_local_roots")
    roots = specific_roots if specific_roots else local_roots
    return roots, specific_errors + local_errors


def _is_under_any(path_value: str, roots: list[str]) -> bool:
    return any(_is_under(path_value, root) for root in roots)


def _nearest_existing_parent(path_value: str) -> Path:
    current = Path(path_value)
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def load_report_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise InferenceReportError("report config root must be a JSON object")
    return raw


def validate_explicit_file_path(
    path_value: Any,
    field: str,
    require_exists: bool,
    allowed_protected_parts: set[str] | None = None,
    approved_roots: list[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(path_value, str) or not path_value.strip():
        return [_err(f"{field} must be a non-empty local file path")]
    text = path_value.strip()
    if REMOTE_RE.search(text):
        errors.append(_err(f"{field} must not be a URL or remote scheme"))
    if WINDOWS_DRIVE_RE.search(text):
        errors.append(_err(f"{field} must not be a Windows drive path"))
    if not text.startswith("/"):
        errors.append(_err(f"{field} must be absolute in approved local mode"))
    if _has_path_traversal(text):
        errors.append(_err(f"{field} must not contain path traversal"))
    if _contains_protected_part(text, allowed_protected_parts):
        errors.append(_err(f"{field} must not contain protected path segments"))
    if field != "checkpoint_path" and _repo_outputs_or_checkpoints(text):
        errors.append(_err(f"{field} must not be inside repository outputs/checkpoints"))
    if approved_roots is not None and approved_roots and not _is_under_any(text, approved_roots):
        errors.append(_err(f"{field} must be under approved roots"))
    if require_exists and not os.path.isfile(text):
        errors.append(_err(f"{field} must exist as a file"))
    return errors


def validate_report_root(path_value: Any, require_exists: bool = False, approved_roots: list[str] | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(path_value, str) or not path_value.strip():
        return [_err("report_root must be a non-empty local directory path")]
    text = path_value.strip()
    if REMOTE_RE.search(text):
        errors.append(_err("report_root must not be a URL or remote scheme"))
    if WINDOWS_DRIVE_RE.search(text):
        errors.append(_err("report_root must not be a Windows drive path"))
    if not text.startswith("/"):
        errors.append(_err("report_root must be absolute in approved local mode"))
    if _has_path_traversal(text):
        errors.append(_err("report_root must not contain path traversal"))
    if _contains_protected_part(text):
        errors.append(_err("report_root must not contain protected path segments"))
    if text.startswith("/") and _inside_repo(text):
        errors.append(_err("report_root must be outside the repository"))
    if _repo_outputs_or_checkpoints(text):
        errors.append(_err("report_root must not be inside repository outputs/checkpoints"))
    if approved_roots is not None and approved_roots and not _is_under_any(text, approved_roots):
        errors.append(_err("report_root must be under approved roots"))
    if require_exists:
        if os.path.exists(text) and not os.path.isdir(text):
            errors.append(_err("report_root exists but is not a directory"))
        parent = os.path.dirname(text.rstrip(os.sep)) or os.sep
        if os.path.exists(parent) and not os.path.isdir(parent):
            errors.append(_err("report_root parent exists but is not a directory"))
        elif not os.path.exists(parent):
            ancestor = _nearest_existing_parent(parent)
            if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
                errors.append(_err("report_root parent must be valid or creatable"))
    return errors


def _validate_approved_roots(raw: dict[str, Any]) -> tuple[dict[str, list[str]], list[str]]:
    root_map: dict[str, list[str]] = {}
    errors: list[str] = []
    for field in APPROVED_ROOT_FIELDS:
        roots, list_errors = _string_list(raw, field)
        errors.extend(list_errors)
        root_map[field] = roots
        for root in roots:
            errors.extend(
                _validate_approved_root(
                    root,
                    field,
                    allow_checkpoint_part=field == "approved_checkpoint_roots",
                    report_root_policy=field == "approved_report_roots",
                )
            )
    return root_map, errors


def _validate_common(raw: dict[str, Any], errors: list[str]) -> None:
    for field in REQUIRED_FIELDS:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    for flag in ("no_download", "no_network", "no_training", "no_checkpoint_writes", "no_sns_augmentation"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    if raw.get("device") not in {"cpu", "cuda"}:
        errors.append(_err("device must be cpu or cuda"))
    if not isinstance(raw.get("cuda_device_index"), int) or isinstance(raw.get("cuda_device_index"), bool) or raw.get("cuda_device_index", 0) < 0:
        errors.append(_err("cuda_device_index must be a non-negative integer"))
    if not isinstance(raw.get("max_image_size"), int) or isinstance(raw.get("max_image_size"), bool) or not (1 <= raw.get("max_image_size", 0) <= 512):
        errors.append(_err("max_image_size must be an integer in [1, 512]"))
    tau = raw.get("threshold_tau")
    if not isinstance(tau, (int, float)) or isinstance(tau, bool) or not (0.0 <= float(tau) <= 1.0):
        errors.append(_err("threshold_tau must be a number in [0.0, 1.0]"))
    if raw.get("class_labels") != list(CLASS_LABELS):
        errors.append(_err("class_labels must match real, full_synthetic, tampered"))
    if raw.get("family_labels") != list(FAMILY_SMOKE_LABELS):
        errors.append(_err("family_labels must match supported pre-SNS family labels"))
    if raw.get("required_approval_text") != APPROVAL_TEXT:
        errors.append(_err("required_approval_text must document the exact approval phrase"))
    if raw.get("recursive_scan") is True or raw.get("recursive_directory_scan") is True:
        errors.append(_err("recursive scan flags are rejected"))


def validate_report_config(raw: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    kind = raw.get("config_kind")
    if kind not in CONFIG_KINDS:
        errors.append(_err("config_kind must be example_symbolic or approved_pre_sns_single_image_report"))
        errors.extend(_walk_safety(raw))
        return errors
    _validate_common(raw, errors)
    if kind == "example_symbolic":
        if raw.get("execution_mode") != "example_only":
            errors.append(_err("execution_mode must be example_only"))
        if raw.get("user_approval_text") not in {"", APPROVAL_TEXT}:
            errors.append(_err("example user_approval_text must be empty or the documented approval phrase"))
        errors.extend(_walk_safety(raw))
        return errors

    if raw.get("execution_mode") != "approved_local_single_image_report":
        errors.append(_err("execution_mode must be approved_local_single_image_report"))
    if raw.get("user_approval_text") != APPROVAL_TEXT:
        errors.append(_err("user_approval_text does not match required approval phrase"))
    root_map, root_errors = _validate_approved_roots(raw)
    errors.extend(root_errors)
    allowed_abs: set[str] = set()
    for field in APPROVED_PATH_FIELDS:
        value = raw.get(field)
        if isinstance(value, str):
            allowed_abs.add(value)
    for field in APPROVED_ROOT_FIELDS:
        allowed_abs.update(root_map.get(field, []))
    errors.extend(_walk_safety(raw, allowed_abs_values=allowed_abs))
    image_roots, image_root_errors = _approved_roots_for(raw, "approved_image_roots")
    checkpoint_roots, checkpoint_root_errors = _approved_roots_for(raw, "approved_checkpoint_roots")
    report_roots, report_root_errors = _approved_roots_for(raw, "approved_report_roots")
    errors.extend(image_root_errors)
    errors.extend(checkpoint_root_errors)
    errors.extend(report_root_errors)
    errors.extend(validate_explicit_file_path(raw.get("image_path"), "image_path", require_exists=True, approved_roots=image_roots))
    errors.extend(
        validate_explicit_file_path(
            raw.get("checkpoint_path"),
            "checkpoint_path",
            require_exists=True,
            allowed_protected_parts={"checkpoints"},
            approved_roots=checkpoint_roots,
        )
    )
    errors.extend(validate_report_root(raw.get("report_root"), require_exists=True, approved_roots=report_roots))
    if raw.get("write_report") is not False and raw.get("write_report") is not True:
        errors.append(_err("write_report must be boolean"))
    return errors


def _runtime_deps():
    try:
        import torch
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("torch and PIL are required for pre-SNS single-image inference") from exc
    return torch, Image


def select_device(torch: Any, raw: dict[str, Any]) -> str:
    requested = raw.get("device", "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda was requested but CUDA is unavailable")
        return f"cuda:{int(raw.get('cuda_device_index', 0))}"
    return "cpu"


def prepare_image_tensor(torch: Any, Image: Any, image_path: str, image_size: int, device: str):
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        raw = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        tensor = raw.reshape(image_size, image_size, 3).permute(2, 0, 1).float().div(255.0)
    return tensor.reshape(1, -1).to(device)


def load_trained_model(torch: Any, checkpoint_path: str, image_size: int, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("checkpoint must contain a JSON-like dictionary")
    checkpoint_image_size = int(checkpoint.get("image_size", image_size))
    model = build_tiny_integrated_model(torch, 3 * checkpoint_image_size * checkpoint_image_size, checkpoint_image_size * checkpoint_image_size)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint missing model_state_dict")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, checkpoint_image_size, checkpoint


def confidence_map(labels: tuple[str, ...] | list[str], probabilities: list[float]) -> dict[str, float]:
    return {label: float(probabilities[index]) for index, label in enumerate(labels)}


def localization_summary(tampered_score: float, threshold_tau: float, mask_probabilities: Any, torch: Any) -> tuple[str, float | None]:
    if tampered_score >= threshold_tau:
        area = float((mask_probabilities >= 0.5).float().mean().item() * 100.0)
        return LOCALIZATION_ACTIVATED, area
    return LOCALIZATION_SKIPPED_BELOW_THRESHOLD, None


def _evidence_for_report(class_label: str, family: str, localization_head: str) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    if class_label == "synthetic":
        evidence.append({"signal_id": EVIDENCE_GLOBAL_SYNTHETIC_ARTIFACT, "description": "pre-SNS class head synthetic signal"})
    if class_label == "tampered":
        evidence.append({"signal_id": EVIDENCE_BOUNDARY_DISCONTINUITY, "description": "pre-SNS class head tampered signal"})
    if localization_head == LOCALIZATION_ACTIVATED:
        evidence.append({"signal_id": EVIDENCE_LOCAL_MASK_ACTIVATION, "description": "pre-SNS localization head activation"})
    if family != "Real-or-N/A":
        evidence.append({"signal_id": EVIDENCE_PROVENANCE_FAMILY_SIGNAL, "description": "pre-SNS provenance family signal"})
    return evidence


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


def run_single_image_report(raw: dict[str, Any]) -> dict[str, Any]:
    errors = validate_report_config(raw)
    if errors:
        raise InferenceReportError("report config validation failed:\n" + "\n".join(errors))
    if raw.get("config_kind") not in {APPROVED_CONFIG_KIND, *LEGACY_APPROVED_CONFIG_KINDS}:
        raise InferenceReportError("single-image inference requires approved_pre_sns_single_image_report config")
    torch, Image = _runtime_deps()
    device = select_device(torch, raw)
    model, image_size, checkpoint = load_trained_model(torch, raw["checkpoint_path"], int(raw["max_image_size"]), device)
    image_tensor = prepare_image_tensor(torch, Image, raw["image_path"], image_size, device)
    started = time.perf_counter()
    with torch.no_grad():
        outputs = model(image_tensor)
        class_probs = torch.softmax(outputs["class_logits"][0], dim=0).detach().cpu().tolist()
        family_probs = torch.softmax(outputs["family_logits"][0], dim=0).detach().cpu().tolist()
        mask_probs = torch.sigmoid(outputs["localization_logits"][0]).detach().cpu()
    latency_ms = max((time.perf_counter() - started) * 1000.0, 0.000001)
    class_conf = confidence_map(CLASS_LABELS, class_probs)
    family_labels = tuple(checkpoint.get("family_labels", list(FAMILY_SMOKE_LABELS)))
    if len(family_labels) != len(family_probs):
        family_labels = FAMILY_SMOKE_LABELS
    family_conf = confidence_map(family_labels, family_probs)
    class_label = CLASS_LABELS[int(max(range(len(class_probs)), key=lambda idx: class_probs[idx]))]
    family = family_labels[int(max(range(len(family_probs)), key=lambda idx: family_probs[idx]))]
    tampered_score = float(class_conf["tampered"])
    threshold_tau = float(raw["threshold_tau"])
    if class_label == "tampered":
        localization_head, mask_area_pct = localization_summary(tampered_score, threshold_tau, mask_probs, torch)
    else:
        localization_head, mask_area_pct = LOCALIZATION_NOT_APPLICABLE, None
    evidence = _evidence_for_report(schema_class_label(class_label), family, localization_head)
    report = {
        "marker": MARKER,
        "schema_version": SCHEMA_VERSION,
        "class": class_label,
        "class_conf": class_conf,
        "family": family,
        "family_conf": family_conf,
        "tampered_score": tampered_score,
        "localization_head": localization_head,
        "mask_area_pct": mask_area_pct,
        "threshold_tau": threshold_tau,
        "evidence": evidence,
        "perturbations": [PERTURBATION_NONE],
        "reason": "",
        "latency_ms": float(latency_ms),
        "fps_estimate": float(1000.0 / latency_ms),
        "device": device,
        "image_path": raw["image_path"],
        "checkpoint_path": raw["checkpoint_path"],
        "write_report": bool(raw.get("write_report")),
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
        "result_scope": raw.get("result_scope", "single-image pre-SNS report; not a final performance claim"),
    }
    reason_input = dict(report)
    if class_label == "full_synthetic":
        reason_input["class"] = "synthetic"
        reason_input["class_conf"] = {
            "real": class_conf["real"],
            "synthetic": class_conf["full_synthetic"],
            "tampered": class_conf["tampered"],
        }
    report["reason"] = generate_reason(reason_input)
    if raw.get("write_report") is True:
        report_path = write_report_json(raw["report_root"], report)
        report["report_path"] = str(report_path)
    return _json_safe(report)


def write_report_json(report_root: str | Path, report: dict[str, Any]) -> Path:
    errors = validate_report_root(str(report_root), require_exists=False)
    if errors:
        raise InferenceReportError("\n".join(errors))
    root = _real(report_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "pre_sns_single_image_report.json"
    tmp = root / ".pre_sns_single_image_report.json.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)
    return path
