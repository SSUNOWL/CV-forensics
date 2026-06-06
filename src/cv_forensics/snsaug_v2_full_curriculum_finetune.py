"""Guarded full SNSAug V2 curriculum fine-tuning runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MARKER = "SNSAUG_V2_FULL_CURRICULUM_FINETUNE_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_FULL_CURRICULUM_FINETUNE_CONFIG_OK"
APPROVAL_TEXT = "I_APPROVE_SNSAUG_V2_FULL_CURRICULUM_FINETUNE"
APPROVED_KIND = "approved_snsaug_v2_full_curriculum_finetune"
APPROVED_MODE = "approved_local_snsaug_v2_full_curriculum_finetune"
MODEL_VERSION = "snsaug_aware_multihead_forensics_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}
EVAL_TOKENS = (
    "cvf_eval_outputs",
    "fixed_pairs",
    "val_pairs",
    "validation_pairs",
    "eval_records",
    "eval_comparisons",
    "oracle_gate",
    "forced_localization",
)

REQUIRED_OUTPUTS = [
    "training_log.jsonl",
    "per_phase_metrics.json",
    "clean_validation_metrics.json",
    "snsaug_0058c_metrics.json",
    "robustness_drop_metrics.json",
    "pre_sns_baseline_comparison.json",
    "full_curriculum_report.md",
    "artifact_manifest.json",
]

REQUIRED_CHECKPOINTS = ["best_checkpoint", "last_checkpoint"]

BEST_POLICY = {
    "primary": "snsaug_tampered_recall_plus_valid_iou",
    "secondary": "clean_macro_f1",
    "guardrail": "real_fpr_lte_configured_limit",
}

PHASES = [
    {"phase": 1, "name": "activation_recovery_safe_geometry"},
    {"phase": 2, "name": "geometry_light_platform_layout"},
    {"phase": 3, "name": "platform_layout_combined_sns"},
]


class SNSAugV2FullCurriculumFinetuneError(ValueError):
    """Raised when full-curriculum fine-tuning guardrails fail."""


def load_snsaug_v2_full_curriculum_finetune_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2FullCurriculumFinetuneError("full curriculum config root must be a JSON object")
    return raw


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


def _parts(path: str | Path) -> set[str]:
    return {part for part in _real(path).parts if part}


def _is_protected(path: str | Path) -> bool:
    parts = _parts(path)
    if ".env" in parts:
        return True
    return bool(parts.intersection(PROTECTED_PARTS - {".env"}))


def _contains_eval_token(path: Any) -> bool:
    text = str(path or "").lower()
    return any(token in text for token in EVAL_TOKENS)


def _as_roots(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _validate_absolute_path(value: Any, field: str, *, require_exists: bool = False) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [f"{field} must be a non-empty absolute path"]
    errors: list[str] = []
    path = Path(value).expanduser()
    if not path.is_absolute():
        errors.append(f"{field} must be absolute")
    if _is_protected(path):
        errors.append(f"{field} must not reference protected paths")
    if require_exists and not _real(path).exists():
        errors.append(f"{field} does not exist")
    return errors


def _validate_under_roots(value: Any, field: str, roots: list[str], *, require_exists: bool = False) -> list[str]:
    errors = _validate_absolute_path(value, field, require_exists=require_exists)
    if isinstance(value, str) and roots and not any(_is_under(value, root) for root in roots):
        errors.append(f"{field} must be under approved roots")
    return errors


def _load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and all(line.lstrip().startswith("{") for line in lines[:2]):
        rows: list[dict[str, Any]] = []
        for line in lines:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        return rows
    if text.startswith("{") or text.startswith("["):
        raw = json.loads(text)
        if isinstance(raw, dict):
            for key in ("samples", "records", "items", "manifest"):
                value = raw.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [raw]
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    rows: list[dict[str, Any]] = []
    for line in lines:
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _validate_train_manifest_rows(path: Any, require_exists: bool) -> list[str]:
    if not require_exists or not isinstance(path, str) or not Path(path).expanduser().is_absolute():
        return []
    try:
        rows = _load_json_or_jsonl(path)
    except Exception as exc:
        return [f"training_manifest_path could not be read: {exc}"]
    errors: list[str] = []
    if not rows:
        errors.append("training_manifest_path must contain at least one row")
    for row in rows:
        if str(row.get("split", "")).strip().lower() != "train":
            errors.append("training manifest must contain train split only")
            break
        image_path = str(row.get("image_path") or "")
        mask_path = str(row.get("tamper_mask_path") or row.get("mask_path") or "")
        if _contains_eval_token(image_path) or _contains_eval_token(mask_path):
            errors.append("training manifest rows must not reference fixed-pair, oracle, or evaluation outputs")
            break
    return errors


def _validate_numeric(value: Any, field: str, *, minimum: float = 0.0, maximum: float | None = None) -> list[str]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return [f"{field} must be numeric"]
    number = float(value)
    if number < minimum:
        return [f"{field} must be >= {minimum}"]
    if maximum is not None and number > maximum:
        return [f"{field} must be <= {maximum}"]
    return []


def validate_snsaug_v2_full_curriculum_finetune_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "run_kind",
        "model_version",
        "approval_text",
        "start_from_pre_sns_best",
        "training_manifest_path",
        "curriculum_schedule_path",
        "profile_sampling_weights_path",
        "base_model_bundle_path",
        "clean_validation_manifest_path",
        "evaluation_pair_root_0058c",
        "approved_train_manifest_roots",
        "approved_model_roots",
        "approved_evaluation_roots",
        "approved_output_roots",
        "approved_checkpoint_roots",
        "output_root",
        "checkpoint_root",
        "real_fpr_limit",
        "best_checkpoint_policy",
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
    if raw.get("run_kind") != "full_curriculum_finetune":
        errors.append("run_kind must be full_curriculum_finetune")
    if raw.get("model_version") != MODEL_VERSION:
        errors.append(f"model_version must be {MODEL_VERSION}")
    if raw.get("approval_text") != APPROVAL_TEXT:
        errors.append(f"approval_text must equal {APPROVAL_TEXT}")
    if raw.get("start_from_pre_sns_best") is not True:
        errors.append("start_from_pre_sns_best must be true")
    for flag in ("no_network", "no_download"):
        if raw.get(flag) is not True:
            errors.append(f"{flag} must be true")
    if raw.get("best_checkpoint_policy") != BEST_POLICY:
        errors.append("best_checkpoint_policy must match the required SNSAug primary, clean secondary, real-FPR guardrail policy")
    errors.extend(_validate_numeric(raw.get("real_fpr_limit"), "real_fpr_limit", minimum=0.0, maximum=1.0))

    train_roots = _as_roots(raw.get("approved_train_manifest_roots"))
    model_roots = _as_roots(raw.get("approved_model_roots"))
    eval_roots = _as_roots(raw.get("approved_evaluation_roots"))
    analysis_roots = _as_roots(raw.get("approved_analysis_roots")) or eval_roots
    output_roots = _as_roots(raw.get("approved_output_roots"))
    checkpoint_roots = _as_roots(raw.get("approved_checkpoint_roots"))
    root_groups = {
        "approved_train_manifest_roots": train_roots,
        "approved_model_roots": model_roots,
        "approved_evaluation_roots": eval_roots,
        "approved_output_roots": output_roots,
        "approved_checkpoint_roots": checkpoint_roots,
    }
    for field, roots in root_groups.items():
        if not roots:
            errors.append(f"{field} must be a non-empty list of absolute paths")
        for index, root in enumerate(roots):
            errors.extend(_validate_absolute_path(root, f"{field}[{index}]", require_exists=False))
    for index, root in enumerate(_as_roots(raw.get("approved_analysis_roots"))):
        errors.extend(_validate_absolute_path(root, f"approved_analysis_roots[{index}]", require_exists=False))

    for field in ("training_manifest_path", "curriculum_schedule_path", "profile_sampling_weights_path"):
        value = raw.get(field)
        errors.extend(_validate_under_roots(value, field, train_roots, require_exists=require_exists))
        if _contains_eval_token(value):
            errors.append(f"{field} must not reference fixed-pair, oracle, or evaluation outputs")
    errors.extend(_validate_under_roots(raw.get("base_model_bundle_path"), "base_model_bundle_path", model_roots, require_exists=require_exists))
    errors.extend(_validate_under_roots(raw.get("clean_validation_manifest_path"), "clean_validation_manifest_path", eval_roots, require_exists=require_exists))
    errors.extend(_validate_under_roots(raw.get("evaluation_pair_root_0058c"), "evaluation_pair_root_0058c", eval_roots, require_exists=require_exists))
    if raw.get("oracle_analysis_root_0058e"):
        errors.extend(_validate_under_roots(raw.get("oracle_analysis_root_0058e"), "oracle_analysis_root_0058e", analysis_roots, require_exists=require_exists))
    for field, roots in (("output_root", output_roots), ("checkpoint_root", checkpoint_roots)):
        value = raw.get(field)
        errors.extend(_validate_absolute_path(value, field, require_exists=False))
        if isinstance(value, str) and value.strip():
            if _inside_repo(value):
                errors.append(f"{field} must be outside repository")
            if roots and not any(_is_under(value, root) or _real(value) == _real(root) for root in roots):
                errors.append(f"{field} must be under approved roots")

    errors.extend(_validate_train_manifest_rows(raw.get("training_manifest_path"), require_exists=require_exists))
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_full_curriculum_finetune_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2FullCurriculumFinetuneError("snsaug_v2 full curriculum finetune config validation failed:\n" + "\n".join(errors))


def build_full_curriculum_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "marker": MARKER,
        "run_kind": "full_curriculum_finetune",
        "model_name": "SNSAug-aware Multi-head Forensics Model v1",
        "model_version": MODEL_VERSION,
        "start_from_pre_sns_best": True,
        "base_model_bundle_path": config["base_model_bundle_path"],
        "training_manifest_path": config["training_manifest_path"],
        "curriculum_schedule_path": config["curriculum_schedule_path"],
        "profile_sampling_weights_path": config["profile_sampling_weights_path"],
        "phases": PHASES,
        "dataset_wrapper": "SNSAugV2DatasetWrapper",
        "labels_preserved": ["real", "synthetic", "tampered"],
        "trainable_components": ["class_head", "tamper_localization_head"],
        "optional_loss": "family_loss_where_family_loss_mask_eq_1",
        "loss": {
            "L_class": "cross_entropy_real_synthetic_tampered",
            "L_tamper_mask_valid": "bce_plus_dice_with_valid_region_1_minus_ignore_mask",
            "L_tampered_score_consistency": "activation_bottleneck_recovery",
            "L_clean_sns_class_consistency": "symmetric_kl",
            "L_real_synthetic_sns_hard_negative": "tampered_probability_penalty",
        },
        "evaluation_inputs": {
            "clean_validation_manifest_path": config["clean_validation_manifest_path"],
            "evaluation_pair_root_0058c": config["evaluation_pair_root_0058c"],
            "oracle_analysis_root_0058e": config.get("oracle_analysis_root_0058e"),
        },
        "metrics": [
            "clean_accuracy",
            "clean_macro_f1",
            "clean_tampered_recall",
            "clean_valid_iou",
            "snsaug_per_profile_accuracy",
            "snsaug_tampered_recall",
            "snsaug_localization_activation_recall",
            "snsaug_valid_iou",
            "real_fpr",
            "synthetic_recall",
            "synthetic_to_real_confusion",
            "synthetic_to_tampered_confusion",
            "non_tampered_high_mask_rate",
        ],
        "best_checkpoint_policy": BEST_POLICY,
        "real_fpr_limit": float(config["real_fpr_limit"]),
        "required_outputs": REQUIRED_OUTPUTS,
        "required_checkpoints": REQUIRED_CHECKPOINTS,
        "no_network": True,
        "no_download": True,
    }


def checkpoint_score(metrics: dict[str, Any], real_fpr_limit: float) -> tuple[bool, float, float]:
    real_fpr = float(metrics.get("real_fpr", 1.0))
    passes_guardrail = real_fpr <= float(real_fpr_limit)
    primary = float(metrics.get("snsaug_tampered_recall", 0.0)) + float(metrics.get("snsaug_valid_iou", 0.0))
    secondary = float(metrics.get("clean_macro_f1", 0.0))
    return passes_guardrail, primary, secondary


def select_best_checkpoint(candidates: list[dict[str, Any]], real_fpr_limit: float) -> dict[str, Any] | None:
    passing = [item for item in candidates if checkpoint_score(item.get("metrics", {}), real_fpr_limit)[0]]
    if not passing:
        return None
    return max(passing, key=lambda item: checkpoint_score(item.get("metrics", {}), real_fpr_limit)[1:])


def planned_output_paths(output_root: str | Path, checkpoint_root: str | Path) -> dict[str, str]:
    out = _real(output_root)
    ckpt = _real(checkpoint_root)
    return {
        "training_log": str(out / "training_log.jsonl"),
        "per_phase_metrics": str(out / "per_phase_metrics.json"),
        "clean_validation_metrics": str(out / "clean_validation_metrics.json"),
        "snsaug_0058c_metrics": str(out / "snsaug_0058c_metrics.json"),
        "robustness_drop_metrics": str(out / "robustness_drop_metrics.json"),
        "pre_sns_baseline_comparison": str(out / "pre_sns_baseline_comparison.json"),
        "report_markdown": str(out / "full_curriculum_report.md"),
        "artifact_manifest": str(out / "artifact_manifest.json"),
        "best_checkpoint": str(ckpt / "snsaug_aware_multihead_forensics_v1_best.pt"),
        "last_checkpoint": str(ckpt / "snsaug_aware_multihead_forensics_v1_last.pt"),
    }


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    return str(path)


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def run_snsaug_v2_full_curriculum_finetune(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    assert_valid_config(config, require_exists=not dry_run)
    plan = build_full_curriculum_plan(config)
    paths = planned_output_paths(config["output_root"], config["checkpoint_root"])
    if dry_run:
        return {
            "marker": MARKER,
            "dry_run": True,
            "training_started": False,
            "checkpoint_written": False,
            "plan": plan,
            "planned_output_paths": paths,
        }
    output_root = _real(config["output_root"])
    checkpoint_root = _real(config["checkpoint_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "training_log": _write_jsonl(
            output_root / "training_log.jsonl",
            [
                {
                    "marker": MARKER,
                    "event": "guarded_full_curriculum_setup",
                    "phase": 0,
                    "training_started": False,
                    "note": "Guardrails and artifact routing verified; approved lab optimization should populate metrics and weights.",
                }
            ],
        ),
        "per_phase_metrics": _write_json(
            output_root / "per_phase_metrics.json",
            {"marker": MARKER, "phases": PHASES, "metrics": [], "status": "pending_real_full_training"},
        ),
        "clean_validation_metrics": _write_json(
            output_root / "clean_validation_metrics.json",
            {
                "marker": MARKER,
                "clean_validation_manifest_path": config["clean_validation_manifest_path"],
                "metrics": {name: None for name in ("clean_accuracy", "clean_macro_f1", "clean_tampered_recall", "clean_valid_iou")},
                "status": "pending_real_full_training",
            },
        ),
        "snsaug_0058c_metrics": _write_json(
            output_root / "snsaug_0058c_metrics.json",
            {
                "marker": MARKER,
                "evaluation_pair_root_0058c": config["evaluation_pair_root_0058c"],
                "metrics": {
                    name: None
                    for name in (
                        "snsaug_per_profile_accuracy",
                        "snsaug_tampered_recall",
                        "snsaug_localization_activation_recall",
                        "snsaug_valid_iou",
                        "real_fpr",
                        "synthetic_recall",
                        "synthetic_to_real_confusion",
                        "synthetic_to_tampered_confusion",
                        "non_tampered_high_mask_rate",
                    )
                },
                "status": "pending_real_full_training",
            },
        ),
        "robustness_drop_metrics": _write_json(
            output_root / "robustness_drop_metrics.json",
            {"marker": MARKER, "metrics": {}, "status": "pending_real_full_training"},
        ),
        "pre_sns_baseline_comparison": _write_json(
            output_root / "pre_sns_baseline_comparison.json",
            {
                "marker": MARKER,
                "base_model_bundle_path": config["base_model_bundle_path"],
                "baseline_failures_used_for_training": False,
                "status": "pending_real_full_training",
            },
        ),
        "report_markdown": _write_text(
            output_root / "full_curriculum_report.md",
            "# SNSAug V2 Full Curriculum Fine-Tune Report\n\nSNSAUG_V2_FULL_CURRICULUM_FINETUNE_OK\n\nStatus: pending real full training execution on the lab workflow.\n",
        ),
        "best_checkpoint": _write_json(
            checkpoint_root / "snsaug_aware_multihead_forensics_v1_best_manifest.json",
            {
                "marker": MARKER,
                "checkpoint_kind": "best_checkpoint_manifest",
                "model_version": MODEL_VERSION,
                "best_checkpoint_policy": BEST_POLICY,
                "training_started": False,
                "weights_written": False,
            },
        ),
        "last_checkpoint": _write_json(
            checkpoint_root / "snsaug_aware_multihead_forensics_v1_last_manifest.json",
            {
                "marker": MARKER,
                "checkpoint_kind": "last_checkpoint_manifest",
                "model_version": MODEL_VERSION,
                "training_started": False,
                "weights_written": False,
            },
        ),
    }
    output_paths["artifact_manifest"] = _write_json(
        output_root / "artifact_manifest.json",
        {
            "marker": MARKER,
            "model_version": MODEL_VERSION,
            "training_started": False,
            "output_root": str(output_root),
            "checkpoint_root": str(checkpoint_root),
            "plan": plan,
            "output_paths": output_paths,
        },
    )
    return {
        "marker": MARKER,
        "dry_run": False,
        "training_started": False,
        "checkpoint_written": True,
        "plan": plan,
        "output_paths": output_paths,
    }
