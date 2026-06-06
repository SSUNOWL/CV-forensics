"""Train-only SNSAug V2 curriculum manifest builder."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

MARKER = "SNSAUG_V2_TRAIN_CURRICULUM_MANIFEST_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_TRAIN_CURRICULUM_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_train_curriculum_manifest"
APPROVED_MODE = "approved_local_snsaug_v2_train_curriculum_manifest"

REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_CONTENT_LABELS = {"real", "synthetic", "tampered"}
REQUIRED_SAFETY_FLAGS = ("no_training", "no_finetune", "no_network", "no_download")
PROTECTED_PARTS = {".env", "secrets", "data", "datasets", "outputs", "checkpoints"}
EVAL_PATH_TOKENS = (
    "cvf_eval_outputs",
    "snsaug_v2_0058c_balanced_benchmark_eval",
    "snsaug_v2_0058d_balanced_eval_with_pred_redmask",
    "snsaug_v2_0058e_forced_localization_oracle_gate",
    "fixed_pairs",
    "val_pairs",
    "validation_pairs",
    "eval_records",
    "eval_comparisons",
)

PROFILE_GROUPS: dict[str, list[str]] = {
    "clean": [],
    "geometry_light": ["canvas_9x16_only", "resize_crop_pad", "zoom_crop"],
    "postprocess_light": ["recompression_light", "resize_jpeg"],
    "screenshot_light": ["screenshot_recapture_light"],
    "overlay_light": ["platform_ui_same_size", "news_meme_overlay"],
    "platform_layout": ["tiktok_like", "instagram_story_like", "youtube_shorts_like"],
    "combined": ["combined_sns_realistic"],
}

CURRICULUM_SCHEDULE: dict[str, dict[str, float]] = {
    "phase_1_activation_recovery_safe_geometry": {
        "clean": 0.45,
        "geometry_light": 0.25,
        "postprocess_light": 0.15,
        "overlay_light": 0.10,
        "screenshot_light": 0.05,
        "platform_layout": 0.00,
        "combined": 0.00,
    },
    "phase_2_geometry_light_platform_adaptation": {
        "clean": 0.35,
        "geometry_light": 0.25,
        "postprocess_light": 0.15,
        "screenshot_light": 0.10,
        "overlay_light": 0.05,
        "platform_layout": 0.10,
        "combined": 0.00,
    },
    "phase_3_platform_layout_combined_sns_adaptation": {
        "clean": 0.25,
        "geometry_light": 0.25,
        "postprocess_light": 0.15,
        "screenshot_light": 0.10,
        "overlay_light": 0.05,
        "platform_layout": 0.15,
        "combined": 0.05,
    },
}

SEVERITY_SCHEDULE: dict[str, dict[str, Any]] = {
    "clean": {"severity": "none", "profile_probability": 1.0, "notes": "No augmentation."},
    "geometry_light": {
        "severity": "light",
        "profile_probability": {"canvas_9x16_only": 0.40, "resize_crop_pad": 0.45, "zoom_crop": 0.15},
        "notes": "Low-probability zoom_crop and light geometry/canvas adaptation.",
    },
    "postprocess_light": {
        "severity": "light",
        "profile_probability": {"recompression_light": 0.60, "resize_jpeg": 0.40},
        "notes": "Light recompression and resize_jpeg early in the curriculum.",
    },
    "screenshot_light": {
        "severity": "light",
        "profile_probability": {"screenshot_recapture_light": 1.0},
        "notes": "Light screenshot recapture simulation.",
    },
    "overlay_light": {
        "severity": "light",
        "profile_probability": {"platform_ui_same_size": 0.50, "news_meme_overlay": 0.50},
        "notes": "Light nuisance overlays with ignore_mask excluded from tamper loss.",
    },
    "platform_layout": {
        "severity": "medium",
        "profile_probability": {"tiktok_like": 0.34, "instagram_story_like": 0.33, "youtube_shorts_like": 0.33},
        "notes": "Hard platform-layout profiles introduced after activation recovery.",
    },
    "combined": {
        "severity": "medium",
        "profile_probability": {"combined_sns_realistic": 1.0},
        "notes": "Combined realistic SNS shift for late curriculum only.",
    },
}


class SNSAugV2TrainCurriculumError(ValueError):
    """Raised when curriculum config, inputs, or guardrails fail."""


def load_snsaug_v2_train_curriculum_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2TrainCurriculumError("curriculum config root must be a JSON object")
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


def _path_parts(path: str | Path) -> set[str]:
    return {part for part in _real(path).parts if part}


def _has_protected_part(path: str | Path) -> bool:
    parts = _path_parts(path)
    if ".env" in parts:
        return True
    return bool(parts.intersection(PROTECTED_PARTS - {".env"}))


def _contains_eval_token(path: Any) -> bool:
    text = str(path or "").lower()
    return any(token in text for token in EVAL_PATH_TOKENS)


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _validate_absolute_path(value: Any, field: str, *, require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, str) or not value.strip():
        return [f"{field} must be a non-empty absolute path"]
    path = Path(value).expanduser()
    if not path.is_absolute():
        errors.append(f"{field} must be absolute")
    if _has_protected_part(path):
        errors.append(f"{field} must not reference protected paths")
    if require_exists and not _real(path).exists():
        errors.append(f"{field} does not exist")
    return errors


def _validate_under_roots(value: Any, field: str, roots: list[str], *, require_exists: bool = False) -> list[str]:
    errors = _validate_absolute_path(value, field, require_exists=require_exists)
    if isinstance(value, str) and roots and not any(_is_under(value, root) for root in roots):
        errors.append(f"{field} must be under approved_input_roots")
    return errors


def validate_curriculum_schedule(schedule: dict[str, dict[str, float]] | None = None) -> list[str]:
    errors: list[str] = []
    schedule = schedule or CURRICULUM_SCHEDULE
    expected_groups = set(PROFILE_GROUPS)
    for phase, weights in schedule.items():
        if set(weights) != expected_groups:
            errors.append(f"{phase} must contain exactly the required profile groups")
            continue
        total = sum(float(value) for value in weights.values())
        if abs(total - 1.0) > 1e-9:
            errors.append(f"{phase} profile group weights must sum to 1.0")
    return errors


def validate_snsaug_v2_train_curriculum_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "train_manifest_path",
        "approved_input_roots",
        "approved_output_roots",
        "output_root",
        "split_policy",
        *REQUIRED_SAFETY_FLAGS,
    )
    for field in required:
        if field not in raw:
            errors.append(f"{field} is required")
    if raw.get("config_kind") != APPROVED_KIND:
        errors.append(f"config_kind must be {APPROVED_KIND}")
    if raw.get("execution_mode") != APPROVED_MODE:
        errors.append(f"execution_mode must be {APPROVED_MODE}")
    if raw.get("split_policy") != "train_only":
        errors.append("split_policy must be train_only")
    for flag in REQUIRED_SAFETY_FLAGS:
        if raw.get(flag) is not True:
            errors.append(f"{flag} must be true")

    input_roots = _as_str_list(raw.get("approved_input_roots"))
    output_roots = _as_str_list(raw.get("approved_output_roots"))
    if not input_roots:
        errors.append("approved_input_roots must be a non-empty list of absolute paths")
    if not output_roots:
        errors.append("approved_output_roots must be a non-empty list of absolute paths")
    for index, root in enumerate(input_roots):
        errors.extend(_validate_absolute_path(root, f"approved_input_roots[{index}]", require_exists=require_exists))
    for index, root in enumerate(output_roots):
        errors.extend(_validate_absolute_path(root, f"approved_output_roots[{index}]", require_exists=False))

    train_manifest_path = raw.get("train_manifest_path")
    errors.extend(_validate_under_roots(train_manifest_path, "train_manifest_path", input_roots, require_exists=require_exists))
    if _contains_eval_token(train_manifest_path):
        errors.append("train_manifest_path must not point to validation pair roots or evaluation outputs")

    output_root = raw.get("output_root")
    errors.extend(_validate_absolute_path(output_root, "output_root", require_exists=False))
    if isinstance(output_root, str) and output_root.strip():
        if _inside_repo(output_root):
            errors.append("output_root must be outside repository")
        if input_roots and any(_is_under(output_root, root) for root in input_roots):
            errors.append("output_root must not be under approved_input_roots")
        if output_roots and not any(_is_under(output_root, root) or _real(output_root) == _real(root) for root in output_roots):
            errors.append("output_root must be under approved_output_roots")

    profile_groups = raw.get("profile_groups")
    if profile_groups is not None and profile_groups != PROFILE_GROUPS:
        errors.append("profile_groups must match the fixed SNSAug V2 curriculum profile groups")
    schedule = raw.get("curriculum_schedule")
    if schedule is not None and schedule != CURRICULUM_SCHEDULE:
        errors.append("curriculum_schedule must match the fixed three-phase SNSAug V2 curriculum")
    severity = raw.get("severity_schedule")
    if severity is not None and severity != SEVERITY_SCHEDULE:
        errors.append("severity_schedule must match the fixed SNSAug V2 severity schedule")
    errors.extend(validate_curriculum_schedule(CURRICULUM_SCHEDULE))
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_train_curriculum_config(config, require_exists=require_exists)
    if errors:
        raise SNSAugV2TrainCurriculumError("snsaug_v2 train curriculum config validation failed:\n" + "\n".join(errors))


def _load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
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
    for line in text.splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


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


def normalize_content_label(row: dict[str, Any]) -> str:
    raw = str(row.get("content_label") or row.get("label") or row.get("class_label") or "").strip().lower()
    aliases = {
        "fake": "synthetic",
        "ai": "synthetic",
        "ai_generated": "synthetic",
        "full_synthetic": "synthetic",
        "manipulated": "tampered",
        "partial": "tampered",
        "splice": "tampered",
        "authentic": "real",
    }
    return aliases.get(raw, raw)


def _base_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("base_id") or row.get("sample_id") or row.get("id") or f"train_{index:08d}")


def _source_dataset(row: dict[str, Any]) -> str:
    value = row.get("source_dataset") or row.get("dataset") or row.get("dataset_name") or "unknown"
    return str(value)


def _image_path(row: dict[str, Any]) -> str:
    value = row.get("image_path") or row.get("source_path") or row.get("path")
    return str(value or "")


def _tamper_mask_path(row: dict[str, Any]) -> str | None:
    value = row.get("tamper_mask_path") or row.get("mask_path") or row.get("mask")
    if isinstance(value, str) and value.strip():
        return value
    return None


def _family_label(row: dict[str, Any]) -> str | None:
    value = row.get("family_label") or row.get("architecture") or row.get("generator_family")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _reject_training_row(row: dict[str, Any]) -> str | None:
    split = str(row.get("split", "")).strip().lower()
    if split != "train":
        return f"non_train_split:{split or 'missing'}"
    image_path = _image_path(row)
    if not image_path:
        return "missing_image_path"
    if _contains_eval_token(image_path):
        return "evaluation_output_image_path"
    if _contains_eval_token(row.get("tamper_mask_path") or row.get("mask_path") or ""):
        return "evaluation_output_mask_path"
    return None


def _manifest_training_notes(label: str) -> list[str]:
    notes = [
        "train-only SNSAug V2 on-the-fly curriculum record",
        "label preserved after SNSAug",
        "ignore_mask excludes SNS nuisance pixels from valid-region localization loss",
        "supports clean/SNS class and tampered-score consistency",
    ]
    if label in {"real", "synthetic"}:
        notes.append("eligible as SNS hard negative for class-balanced sampling")
    if label == "tampered":
        notes.append("eligible for activation recovery and mask-robustness localization loss")
    return notes


def _class_balanced_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter(row["content_label"] for row in rows)
    labels = [label for label in ("real", "synthetic", "tampered") if counts.get(label, 0) > 0]
    if not labels:
        return {}
    total = float(sum(counts[label] for label in labels))
    return {label: total / (len(labels) * float(counts[label])) for label in labels}


def build_manifest_rows(input_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    rejected_counts: Counter[str] = Counter()
    for index, row in enumerate(input_rows):
        if not isinstance(row, dict):
            rejected_counts["non_object_row"] += 1
            continue
        rejection = _reject_training_row(row)
        if rejection is not None:
            rejected_counts[rejection] += 1
            continue
        label = normalize_content_label(row)
        if label not in VALID_CONTENT_LABELS:
            rejected_counts["unsupported_content_label"] += 1
            continue
        family = _family_label(row)
        record: dict[str, Any] = {
            "base_id": _base_id(row, index),
            "source_dataset": _source_dataset(row),
            "split": "train",
            "image_path": _image_path(row),
            "content_label": label,
            "family_loss_mask": 1 if family else 0,
            "allowed_profile_groups": list(PROFILE_GROUPS),
            "profile_sampling_weights_by_phase": CURRICULUM_SCHEDULE,
            "severity_schedule": SEVERITY_SCHEDULE,
            "sampling_weight": 1.0,
            "label_preserved": True,
            "training_notes": _manifest_training_notes(label),
        }
        mask_path = _tamper_mask_path(row)
        if mask_path is not None:
            record["tamper_mask_path"] = mask_path
        if family is not None:
            record["family_label"] = family
        accepted.append(record)

    weights = _class_balanced_weights(accepted)
    for record in accepted:
        record["sampling_weight"] = round(float(weights.get(record["content_label"], 1.0)), 8)
    accepted.sort(key=lambda item: (item["content_label"], item["source_dataset"], item["base_id"]))
    summary = {
        "input_record_count": len(input_rows),
        "accepted_train_record_count": len(accepted),
        "rejected_counts": dict(sorted(rejected_counts.items())),
    }
    return accepted, summary


def training_guardrails(output_root: str | Path) -> dict[str, Any]:
    return {
        "marker": MARKER,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
        "train_only": True,
        "output_root_outside_repository": not _inside_repo(output_root),
        "validation_samples_rejected": True,
        "evaluation_outputs_rejected_as_training_images": True,
        "checkpoint_writes": False,
        "generated_outputs_inside_repository": False,
    }


def training_intent_markdown() -> str:
    return """# SNSAug V2 Train Curriculum Intent And References

This manifest prepares train-only on-the-fly SNSAug V2 fine-tuning records. It does not train, fine-tune, download, or write checkpoints.

## Why This Training Is Needed

The pre-SNS model remains strong on clean samples, but SNSAug V2 diagnostics show p_tampered collapse under social-media transforms. Because localization is conditional on the tampered score, the gate turns off and the predicted redmask becomes empty or unavailable. Forced-localization diagnostics show that resize_crop_pad, screenshot_recapture_light, and zoom_crop can recover some IoU when localization is forced, while tiktok_like, instagram_story_like, youtube_shorts_like, and combined_sns_realistic still expose mask decoder weakness under platform layout.

The curriculum therefore starts with clean and light geometry/postprocess views for activation recovery, then introduces light platform adaptation, and finally adds harder platform-layout and combined SNS profiles.

## Train-Only Requirement

Only split=train rows are admitted. Validation and test fixed pairs are evaluation-only and must not become training examples, mining inputs, or hard-example replay. This protects the reported SNSAug V2 robustness numbers from leakage.

## Loss And Guardrail Intent

SNSAug labels are preserved: real stays real, synthetic stays synthetic, and tampered stays tampered. The wrapper is expected to generate SNS views on the fly, use ignore_mask as a valid-region exclusion for localization loss, and support clean/SNS class consistency and tampered-score consistency. SNS overlay pixels are nuisance regions, not tamper ground truth, so ignore_mask is excluded from tamper loss with valid_region = 1 - ignore_mask.

Real and synthetic rows remain useful as SNS hard negatives for class-balanced sampling, while tampered rows target both activation robustness and mask decoder robustness.

## References

- Community Forensics CVPR 2025
- SIDA CVPR 2025
- Ignoring the Decoy WACV 2026 Workshop
- Degradation-consistent paired training / clean-degraded consistency
- B-Free / bias-free paired detector training
"""


def build_snsaug_v2_train_curriculum_manifest(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_exists=True)
    output_root = _real(config["output_root"])
    if _inside_repo(output_root):
        raise SNSAugV2TrainCurriculumError("output_root must be outside repository")
    output_root.mkdir(parents=True, exist_ok=True)

    input_rows = _load_json_or_jsonl(config["train_manifest_path"])
    manifest_rows, input_summary = build_manifest_rows(input_rows)
    if not manifest_rows:
        raise SNSAugV2TrainCurriculumError("no valid train records remained after curriculum guardrails")

    manifest_path = _write_jsonl(output_root / "snsaug_v2_train_curriculum_manifest.jsonl", manifest_rows)
    schedule_path = _write_json(
        output_root / "snsaug_v2_curriculum_schedule.json",
        {"marker": MARKER, "curriculum_schedule": CURRICULUM_SCHEDULE, "phase_count": len(CURRICULUM_SCHEDULE)},
    )
    weights_path = _write_json(
        output_root / "snsaug_v2_profile_sampling_weights.json",
        {
            "marker": MARKER,
            "profile_groups": PROFILE_GROUPS,
            "profile_sampling_weights_by_phase": CURRICULUM_SCHEDULE,
            "severity_schedule": SEVERITY_SCHEDULE,
        },
    )
    class_counts = {label: sum(1 for row in manifest_rows if row["content_label"] == label) for label in ("real", "synthetic", "tampered")}
    class_balance_path = _write_json(
        output_root / "snsaug_v2_class_balance_summary.json",
        {
            "marker": MARKER,
            "record_total": len(manifest_rows),
            "counts_by_content_label": class_counts,
            "sampling_weight_policy": "inverse_frequency_over_present_classes",
            "input_summary": input_summary,
        },
    )
    guardrails_path = _write_json(output_root / "snsaug_v2_training_guardrails.json", training_guardrails(output_root))
    intent_path = _write_text(output_root / "snsaug_v2_training_intent_and_references.md", training_intent_markdown())
    output_paths = {
        "train_curriculum_manifest": manifest_path,
        "curriculum_schedule": schedule_path,
        "profile_sampling_weights": weights_path,
        "class_balance_summary": class_balance_path,
        "training_guardrails": guardrails_path,
        "training_intent_and_references": intent_path,
    }
    artifact_path = _write_json(
        output_root / "artifact_manifest.json",
        {
            "marker": MARKER,
            "artifact_kind": "snsaug_v2_train_curriculum_manifest",
            "output_root": str(output_root),
            "output_paths": output_paths,
            "record_count": len(manifest_rows),
            "guardrails": training_guardrails(output_root),
        },
    )
    output_paths["artifact_manifest"] = artifact_path
    return {"marker": MARKER, "record_count": len(manifest_rows), "output_paths": output_paths, "input_summary": input_summary}
