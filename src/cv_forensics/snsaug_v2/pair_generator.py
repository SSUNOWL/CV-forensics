"""Deterministic clean/augmented pair generation for SNSAug V2."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from PIL import Image

from ..pre_sns_training_artifacts import validate_artifact_root
from .configs import SNSAugV2Config
from .sns_augmentor import SNSAugV2Augmentor
from .source_manifest_audit import audit_source_manifest, load_source_manifest_records
from .ui_renderers import draw_overlay_debug


def _safe_base_id(sample: dict[str, Any], index: int) -> str:
    raw = str(sample.get("base_id") or sample.get("sample_id") or sample.get("id") or f"sample_{index:06d}")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)


def _normalize_label(sample: dict[str, Any]) -> str:
    label = str(sample.get("content_label") or sample.get("label") or sample.get("class_label") or "").strip().lower()
    return "synthetic" if label == "full_synthetic" else label


CLASS_LABELS = ("real", "synthetic", "tampered")


class SNSAugV2PairGenerationError(ValueError):
    """Raised when fixed-pair generation cannot satisfy required benchmark sanity."""


def _valid_mask_path(sample: dict[str, Any]) -> bool:
    mask_path = sample.get("tamper_mask_path") or sample.get("mask_path") or sample.get("source_mask_path")
    return isinstance(mask_path, str) and bool(mask_path.strip()) and Path(mask_path).is_file()


def _select_samples(
    samples: list[dict[str, Any]],
    max_samples: int,
    samples_per_class: int | None,
    *,
    require_tampered_masks: bool = True,
    allow_missing_tampered: bool = False,
    allow_tampered_without_mask: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[str]]:
    warnings_out: list[str] = []
    class_counts = {label: 0 for label in CLASS_LABELS}
    tampered_base_count = 0
    tampered_with_valid_mask_count = 0
    tampered_missing_mask_count = 0
    examples_missing_mask: list[str] = []
    for sample in samples:
        label = _normalize_label(sample)
        if label not in class_counts:
            continue
        class_counts[label] += 1
        if label == "tampered":
            tampered_base_count += 1
            if _valid_mask_path(sample):
                tampered_with_valid_mask_count += 1
            else:
                tampered_missing_mask_count += 1
                if len(examples_missing_mask) < 5:
                    examples_missing_mask.append(str(sample.get("base_id") or sample.get("sample_id") or sample.get("id") or sample.get("image_path") or "unknown"))

    if samples_per_class is not None:
        if tampered_base_count == 0 and not allow_missing_tampered:
            raise SNSAugV2PairGenerationError("tampered class count is zero; use allow_missing_tampered only for explicit diagnostics")
        if require_tampered_masks and tampered_with_valid_mask_count == 0 and not allow_tampered_without_mask:
            raise SNSAugV2PairGenerationError("tampered_with_valid_mask_count is zero; use allow_tampered_without_mask only for explicit diagnostics")

    if samples_per_class is None:
        selected = samples[:max_samples]
    else:
        minimum_balanced_count = samples_per_class * len(CLASS_LABELS)
        if max_samples < minimum_balanced_count:
            raise SNSAugV2PairGenerationError(
                f"max_samples={max_samples} is smaller than the balanced requested count {minimum_balanced_count}"
            )
        buckets = {label: [] for label in CLASS_LABELS}
        for sample in samples:
            label = _normalize_label(sample)
            if label not in buckets:
                continue
            if label == "tampered" and require_tampered_masks and not allow_tampered_without_mask and not _valid_mask_path(sample):
                continue
            if len(buckets[label]) < samples_per_class:
                buckets[label].append(sample)
        selected = []
        for label in CLASS_LABELS:
            if len(buckets[label]) < samples_per_class:
                message = f"requested {samples_per_class} {label} samples but selected {len(buckets[label])}"
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                warnings_out.append(message)
            selected.extend(buckets[label])

    selected_per_class = {label: 0 for label in CLASS_LABELS}
    for sample in selected:
        label = _normalize_label(sample)
        if label in selected_per_class:
            selected_per_class[label] += 1
    class_balance_summary = {
        "requested_per_class": samples_per_class,
        "selected_per_class": selected_per_class,
        "available_per_class": class_counts,
    }
    mask_summary = {
        "tampered_base_count": tampered_base_count,
        "tampered_with_valid_mask_count": tampered_with_valid_mask_count,
        "tampered_missing_mask_count": tampered_missing_mask_count,
        "examples_missing_mask": examples_missing_mask,
    }
    return selected, class_balance_summary, mask_summary, warnings_out


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


class SNSAugV2PairGenerator:
    def __init__(
        self,
        *,
        input_manifest: str | Path | None = None,
        source_manifest_path: str | Path | None = None,
        output_root: str | Path,
        profiles: list[str],
        severity: str = "medium",
        seed: int = 0,
        split: str | None = None,
        max_samples: int = 100,
        samples_per_class: int | None = None,
        max_samples_per_class: int | None = None,
        output_size: int | None = None,
        font_path: str | None = None,
        include_debug_overlays: bool = True,
        require_tampered_masks: bool = True,
        allow_tampered_without_mask: bool = False,
        allow_missing_tampered: bool = False,
    ) -> None:
        self.input_manifest = Path(source_manifest_path or input_manifest or "")
        self.output_root = Path(output_root)
        self.profiles = list(profiles)
        self.severity = severity
        self.seed = int(seed)
        self.split = split
        self.max_samples = int(max_samples)
        self.samples_per_class = max_samples_per_class if max_samples_per_class is not None else samples_per_class
        self.output_size = output_size
        self.font_path = font_path
        self.include_debug_overlays = bool(include_debug_overlays)
        self.require_tampered_masks = bool(require_tampered_masks)
        self.allow_tampered_without_mask = bool(allow_tampered_without_mask)
        self.allow_missing_tampered = bool(allow_missing_tampered)

    def _save_image(self, image: Image.Image, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return str(path)

    def _load_clean_sample(self, sample: dict[str, Any]) -> tuple[Image.Image, Image.Image | None]:
        image_path = Path(str(sample["image_path"]))
        with Image.open(image_path) as img:
            clean_image = img.convert("RGB")
        mask_path_value = sample.get("tamper_mask_path") or sample.get("mask_path") or sample.get("source_mask_path")
        clean_mask = None
        if isinstance(mask_path_value, str) and mask_path_value.strip():
            with Image.open(mask_path_value) as mask_img:
                clean_mask = mask_img.convert("L")
        return clean_image, clean_mask

    def run(self) -> dict[str, Any]:
        output_root = validate_artifact_root(self.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        audit = audit_source_manifest(self.input_manifest, output_root=output_root, fail_fast=False)
        valid_rows = audit["valid_records"]
        selected, class_balance_summary, mask_summary, selection_warnings = _select_samples(
            valid_rows,
            self.max_samples,
            self.samples_per_class,
            require_tampered_masks=self.require_tampered_masks,
            allow_missing_tampered=self.allow_missing_tampered,
            allow_tampered_without_mask=self.allow_tampered_without_mask,
        )

        rows: list[dict[str, Any]] = []
        pair_index: list[dict[str, Any]] = []
        images_dir = output_root / "images"
        masks_dir = output_root / "tamper_masks"
        ignore_dir = output_root / "ignore_masks"
        debug_dir = output_root / "debug_overlays"

        for index, sample in enumerate(selected):
            base_id = _safe_base_id(sample, index)
            split = self.split or str(sample.get("split") or "train")
            clean_image, clean_mask = self._load_clean_sample(sample)
            label = _normalize_label(sample)
            source_path = str(sample["image_path"])
            source_mask_path = sample.get("tamper_mask_path")

            clean_out = images_dir / f"{base_id}__clean.png"
            clean_mask_out = masks_dir / f"{base_id}__clean.png"
            clean_ignore_out = ignore_dir / f"{base_id}__clean.png"
            clean_debug_out = debug_dir / f"{base_id}__clean.png"
            self._save_image(clean_image, clean_out)
            if clean_mask is not None:
                self._save_image(clean_mask, clean_mask_out)
            clean_ignore = Image.new("L", clean_image.size, 0)
            self._save_image(clean_ignore, clean_ignore_out)
            clean_debug = draw_overlay_debug(clean_image, [])
            self._save_image(clean_debug, clean_debug_out)
            clean_row = {
                "base_id": base_id,
                "source_dataset": sample.get("source_dataset"),
                "split": split,
                "source_path": source_path,
                "source_mask_path": source_mask_path,
                "content_label": label,
                "family_label": sample.get("family_label"),
                "view": "clean",
                "profile": "clean",
                "severity": self.severity,
                "seed": self.seed + index,
                "image_path": str(clean_out),
                "tamper_mask_path": str(clean_mask_out) if clean_mask is not None else None,
                "ignore_mask_path": str(clean_ignore_out),
                "debug_overlay_path": str(clean_debug_out),
                "aug_meta": {"label_preserved": True, "overlay_boxes": [], "geometric_transform_meta": {"type": "identity"}},
                "overlay_boxes": [],
                "label_preserved": True,
            }
            rows.append(clean_row)
            pair = {"base_id": base_id, "content_label": label, "views": [clean_row]}

            for profile_index, profile in enumerate(self.profiles):
                if profile == "clean":
                    continue
                cfg = SNSAugV2Config(
                    profile=profile,
                    severity=self.severity,
                    seed=self.seed + index * 100 + profile_index,
                    output_size=self.output_size,
                    font_path=self.font_path,
                )
                result = SNSAugV2Augmentor(cfg)(clean_image, clean_mask, label=label, base_id=base_id, seed=cfg.seed)
                image_out = images_dir / f"{base_id}__{profile}.png"
                mask_out = masks_dir / f"{base_id}__{profile}.png"
                ignore_out = ignore_dir / f"{base_id}__{profile}.png"
                debug_out = debug_dir / f"{base_id}__{profile}.png"
                self._save_image(result.image, image_out)
                if result.tamper_mask is not None:
                    self._save_image(result.tamper_mask, mask_out)
                self._save_image(result.ignore_mask, ignore_out)
                debug_image = draw_overlay_debug(result.image, result.meta.get("overlay_boxes", []))
                self._save_image(debug_image, debug_out)
                row = {
                    "base_id": base_id,
                    "source_dataset": sample.get("source_dataset"),
                    "split": split,
                    "source_path": source_path,
                    "source_mask_path": source_mask_path,
                    "content_label": label,
                    "family_label": sample.get("family_label"),
                    "view": "sns_aug",
                    "profile": profile,
                    "severity": self.severity,
                    "seed": cfg.seed,
                    "image_path": str(image_out),
                    "tamper_mask_path": str(mask_out) if result.tamper_mask is not None else None,
                    "ignore_mask_path": str(ignore_out),
                    "debug_overlay_path": str(debug_out),
                    "aug_meta": result.meta,
                    "overlay_boxes": result.meta.get("overlay_boxes", []),
                    "label_preserved": True,
                }
                rows.append(row)
                pair["views"].append(row)
            pair_index.append(pair)

        meta_path = _write_jsonl(output_root / "meta.jsonl", rows)
        pair_index_path = _write_json(output_root / "pair_index.json", {"pairs": pair_index})
        profile_count = 1 + len([profile for profile in self.profiles if profile != "clean"])
        rows_per_profile = {profile: sum(1 for row in rows if row.get("profile") == profile) for profile in sorted({str(row.get("profile")) for row in rows})}
        class_balance_summary.update(
            {
                "base_count": len(selected),
                "rows_per_profile": rows_per_profile,
                "profile_count": profile_count,
                "expected_record_count": len(selected) * profile_count,
                "actual_record_count": len(rows),
            }
        )
        class_balance_summary_path = _write_json(output_root / "class_balance_summary.json", class_balance_summary)
        mask_availability_summary_path = _write_json(output_root / "mask_availability_summary.json", mask_summary)
        artifact_manifest = {
            "source_manifest_path": str(self.input_manifest),
            "source_manifest_audit_summary": audit["audit_summary_path"],
            "meta_jsonl": meta_path,
            "pair_index_json": pair_index_path,
            "class_balance_summary_json": class_balance_summary_path,
            "mask_availability_summary_json": mask_availability_summary_path,
            "images_dir": str(images_dir),
            "tamper_masks_dir": str(masks_dir),
            "ignore_masks_dir": str(ignore_dir),
            "debug_overlays_dir": str(debug_dir),
            "sample_count": len(selected),
            "profiles": self.profiles,
            "severity": self.severity,
            "warnings": selection_warnings,
        }
        artifact_manifest_path = _write_json(output_root / "artifact_manifest.json", artifact_manifest)
        artifact_manifest["artifact_manifest_path"] = artifact_manifest_path
        _write_json(output_root / "artifact_manifest.json", artifact_manifest)
        return artifact_manifest
