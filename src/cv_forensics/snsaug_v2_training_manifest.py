"""Train-only SNSAug V2 manifest builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pre_sns_v3_dual_scale_report import (
    _err,
    _inside_repo,
    _is_under,
    _real,
    _validate_absolute_path,
    _validate_under_roots,
)
from .pre_sns_v3_sns_robustness_eval import json_safe

MARKER = "SNSAUG_V2_TRAINING_MANIFEST_AND_WRAPPER_OK"
CONFIG_OK_MARKER = "SNSAUG_V2_TRAINING_MANIFEST_CONFIG_OK"
APPROVED_KIND = "approved_snsaug_v2_training_manifest"
APPROVED_MODE = "approved_local_snsaug_v2_training_manifest"

TAMPERED_GROUP_RATIOS = {
    "stable_correct_anchor": 0.40,
    "fragile_bundle": 0.35,
    "clean_fail": 0.15,
    "random_tampered_coverage": 0.10,
}
DEFAULT_CLASS_BALANCE = {
    "real": 0.275,
    "synthetic": 0.275,
    "tampered": 0.45,
}
DEFAULT_VIEW_RATIOS = {
    "clean_weight": 0.30,
    "basic_aug_weight": 0.30,
    "sns_aug_weight": 0.40,
}
DEFAULT_CURRICULUM = [
    {"start_epoch": 1, "end_epoch": 3, "clean": 0.50, "basic_aug": 0.30, "sns": 0.20, "severity": "light"},
    {"start_epoch": 4, "end_epoch": 8, "clean": 0.35, "basic_aug": 0.30, "sns": 0.35, "severity": "medium"},
    {"start_epoch": 9, "end_epoch": None, "clean": 0.25, "basic_aug": 0.25, "sns": 0.50, "severity": "strong"},
]


class SNSAugV2TrainingManifestError(ValueError):
    """Raised when snsaug_v2 training manifest inputs or guardrails fail."""


def load_snsaug_v2_training_manifest_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SNSAugV2TrainingManifestError("snsaug_v2 training manifest config root must be a JSON object")
    return raw


def _as_roots(value: Any) -> list[str]:
    return value if isinstance(value, list) and all(isinstance(item, str) for item in value) else []


def validate_snsaug_v2_training_manifest_config(raw: dict[str, Any], require_exists: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "config_kind",
        "execution_mode",
        "train_manifest_path",
        "train_mining_manifest_path",
        "approved_input_roots",
        "approved_output_roots",
        "output_root",
        "seed",
        "profiles",
        "no_training",
        "no_download",
        "no_network",
    )
    for field in required:
        if field not in raw:
            errors.append(_err(f"{field} is required"))
    if raw.get("config_kind") != APPROVED_KIND:
        errors.append(_err(f"config_kind must be {APPROVED_KIND}"))
    if raw.get("execution_mode") != APPROVED_MODE:
        errors.append(_err(f"execution_mode must be {APPROVED_MODE}"))
    for flag in ("no_training", "no_download", "no_network"):
        if raw.get(flag) is not True:
            errors.append(_err(f"{flag} must be true"))
    roots = _as_roots(raw.get("approved_input_roots"))
    if not roots:
        errors.append(_err("approved_input_roots must be a non-empty list of absolute paths"))
    for index, root in enumerate(roots):
        errors.extend(_validate_absolute_path(root, f"approved_input_roots[{index}]"))
    output_roots = _as_roots(raw.get("approved_output_roots"))
    if not output_roots:
        errors.append(_err("approved_output_roots must be a non-empty list of absolute paths"))
    for index, root in enumerate(output_roots):
        errors.extend(_validate_absolute_path(root, f"approved_output_roots[{index}]"))
    for field in ("train_manifest_path", "train_mining_manifest_path"):
        errors.extend(_validate_under_roots(raw.get(field), field, roots, require_file=require_exists))
    output_root = raw.get("output_root")
    errors.extend(_validate_absolute_path(output_root, "output_root", require_dir_parent=require_exists))
    if isinstance(output_root, str) and output_root.startswith("/"):
        if _inside_repo(_real(output_root)):
            errors.append(_err("output_root must be outside repository"))
        if any(_is_under(output_root, root) for root in roots):
            errors.append(_err("output_root must not be under approved input roots"))
        if output_roots and not any(_is_under(output_root, root) or str(_real(output_root)) == str(_real(root)) for root in output_roots):
            errors.append(_err("output_root must be under an approved output root"))
    if "seed" in raw:
        value = raw["seed"]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(_err("seed must be a positive integer"))
    profiles = raw.get("profiles")
    if not isinstance(profiles, list) or not profiles or not all(isinstance(item, str) for item in profiles):
        errors.append(_err("profiles must be a non-empty list of strings"))
    severity_schedule = raw.get("severity_schedule")
    if severity_schedule is not None and not isinstance(severity_schedule, list):
        errors.append(_err("severity_schedule must be a list when present"))
    return errors


def assert_valid_config(config: dict[str, Any], require_exists: bool = False) -> None:
    errors = validate_snsaug_v2_training_manifest_config(config, require_exists)
    if errors:
        raise SNSAugV2TrainingManifestError("snsaug_v2 training manifest config validation failed:\n" + "\n".join(errors))


def _load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        first_line = lines[0].lstrip()
        second_line = lines[1].lstrip()
        if first_line.startswith("{") and second_line.startswith("{"):
            rows = []
            for line in lines:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
            return rows
    if text.startswith("{") or text.startswith("["):
        raw = json.loads(text)
        items = raw.get("samples", raw) if isinstance(raw, dict) else raw
        return [item for item in items if isinstance(item, dict)]
    rows = []
    for line in text.splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    return str(path)


def curriculum_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    schedule = config.get("severity_schedule")
    if isinstance(schedule, list) and schedule:
        return schedule
    return list(DEFAULT_CURRICULUM)


def _resolve_group(mining_row: dict[str, Any]) -> str:
    groups = set(str(item) for item in mining_row.get("groups", []))
    if "stable_correct_anchor" in groups:
        return "stable_correct_anchor"
    if groups.intersection({"fragile_correct_to_fail", "confidence_fragile", "mask_iou_fragile"}):
        return "fragile_bundle"
    if "clean_fail" in groups:
        return "clean_fail"
    return "random_tampered_coverage"


def _recommended_profiles(label: str, mining_group: str, config: dict[str, Any]) -> list[str]:
    configured = [str(item) for item in config.get("profiles", [])]
    if label != "tampered":
        return configured[: min(3, len(configured))] if configured else ["clean", "jpeg_resize", "combined_sns_realistic"]
    if mining_group == "stable_correct_anchor":
        return [item for item in configured if item in {"clean", "jpeg_resize", "combined_sns_realistic", "tiktok_like"}] or configured
    if mining_group == "fragile_bundle":
        return [item for item in configured if item in {"combined_sns_realistic", "tiktok_like", "instagram_story_like", "youtube_shorts_like"}] or configured
    if mining_group == "clean_fail":
        return [item for item in configured if item in {"clean", "jpeg_resize", "annotation_sticker"}] or configured
    return configured or ["clean", "jpeg_resize", "combined_sns_realistic"]


def _max_severity(mining_group: str) -> str:
    if mining_group == "stable_correct_anchor":
        return "medium"
    if mining_group == "fragile_bundle":
        return "strong"
    if mining_group == "clean_fail":
        return "light"
    return "medium"


def _family_loss_allowed(row: dict[str, Any]) -> bool:
    family = row.get("family_label")
    return isinstance(family, str) and family.strip() != ""


def _class_label(row: dict[str, Any]) -> str:
    return str(row.get("content_label") or row.get("label") or row.get("class_label") or "").lower().replace("full_synthetic", "synthetic")


def build_snsaug_v2_training_manifest(config: dict[str, Any]) -> dict[str, Any]:
    assert_valid_config(config, require_exists=True)
    output_root = _real(config["output_root"])
    if _inside_repo(output_root):
        raise SNSAugV2TrainingManifestError("output_root must be outside repository")
    output_root.mkdir(parents=True, exist_ok=True)

    train_rows = [row for row in _load_json_or_jsonl(config["train_manifest_path"]) if isinstance(row, dict)]
    mining_rows = [row for row in _load_json_or_jsonl(config["train_mining_manifest_path"]) if isinstance(row, dict)]

    base_lookup = {str(row.get("base_id") or row.get("sample_id") or row.get("id")): row for row in train_rows if str(row.get("split", "train")) == "train"}
    dropped_non_train = sum(1 for row in train_rows if str(row.get("split", "train")) != "train")
    manifest_rows: list[dict[str, Any]] = []
    class_counts = {"real": 0, "synthetic": 0, "tampered": 0}
    group_counts = {key: 0 for key in TAMPERED_GROUP_RATIOS}

    for mining_row in mining_rows:
        base_id = str(mining_row.get("base_id") or "")
        source = base_lookup.get(base_id)
        if source is None:
            continue
        label = _class_label(source)
        if label not in class_counts:
            continue
        if str(source.get("split", "train")) != "train":
            continue
        if label == "tampered":
            mining_group = _resolve_group(mining_row)
        else:
            mining_group = "non_tampered_coverage"
        class_weight = DEFAULT_CLASS_BALANCE["tampered" if label == "tampered" else label]
        if label == "tampered":
            group_weight = TAMPERED_GROUP_RATIOS[mining_group]
            sampling_weight = class_weight * group_weight
            group_counts[mining_group] += 1
        else:
            sampling_weight = class_weight
        class_counts[label] += 1
        manifest_rows.append(
            {
                "base_id": base_id,
                "source_dataset": source.get("source_dataset"),
                "split": "train",
                "image_path": source.get("image_path") or source.get("source_path"),
                "tamper_mask_path": source.get("tamper_mask_path") or source.get("mask_path"),
                "content_label": label,
                "family_label": source.get("family_label"),
                "mining_group": mining_group,
                "recommended_profiles": _recommended_profiles(label, mining_group, config),
                "sampling_weight": float(sampling_weight),
                "clean_weight": float(config.get("clean_weight", DEFAULT_VIEW_RATIOS["clean_weight"])),
                "basic_aug_weight": float(config.get("basic_aug_weight", DEFAULT_VIEW_RATIOS["basic_aug_weight"])),
                "sns_aug_weight": float(config.get("sns_aug_weight", DEFAULT_VIEW_RATIOS["sns_aug_weight"])),
                "max_severity": _max_severity(mining_group),
                "allow_family_loss": _family_loss_allowed(source),
                "label_preserved": True,
            }
        )

    manifest_rows.sort(key=lambda row: (row["content_label"], row["base_id"]))
    manifest_path = _write_jsonl(output_root / "snsaug_v2_training_manifest.jsonl", manifest_rows)
    sampling_summary = {
        "marker": MARKER,
        "record_count": len(manifest_rows),
        "dropped_non_train_count": dropped_non_train,
        "tampered_group_ratios": TAMPERED_GROUP_RATIOS,
        "default_view_ratios": {
            "clean": float(config.get("clean_weight", DEFAULT_VIEW_RATIOS["clean_weight"])),
            "basic_aug": float(config.get("basic_aug_weight", DEFAULT_VIEW_RATIOS["basic_aug_weight"])),
            "sns_aug": float(config.get("sns_aug_weight", DEFAULT_VIEW_RATIOS["sns_aug_weight"])),
        },
        "curriculum": curriculum_schedule(config),
        "group_counts": group_counts,
    }
    class_balance_summary = {
        "marker": MARKER,
        "class_counts": class_counts,
        "target_class_balance": DEFAULT_CLASS_BALANCE,
    }
    sampling_path = _write_json(output_root / "snsaug_v2_sampling_summary.json", sampling_summary)
    class_balance_path = _write_json(output_root / "snsaug_v2_class_balance_summary.json", class_balance_summary)
    artifact = {
        "marker": MARKER,
        "output_paths": {
            "training_manifest": manifest_path,
            "sampling_summary": sampling_path,
            "class_balance_summary": class_balance_path,
        },
        "no_training": True,
        "no_download": True,
        "no_network": True,
    }
    artifact_path = _write_json(output_root / "artifact_manifest.json", artifact)
    return {
        "marker": MARKER,
        "record_count": len(manifest_rows),
        "output_paths": {
            "training_manifest": manifest_path,
            "sampling_summary": sampling_path,
            "class_balance_summary": class_balance_path,
            "artifact_manifest": artifact_path,
        },
    }
