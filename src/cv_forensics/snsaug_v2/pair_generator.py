"""Deterministic clean/augmented pair generation for SNSAug V2."""

from __future__ import annotations

import json
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


def _select_samples(samples: list[dict[str, Any]], max_samples: int, samples_per_class: int | None) -> list[dict[str, Any]]:
    if samples_per_class is None:
        return samples[:max_samples]
    buckets = {"real": [], "synthetic": [], "tampered": []}
    for sample in samples:
        label = _normalize_label(sample)
        if label in buckets and len(buckets[label]) < samples_per_class:
            buckets[label].append(sample)
    selected: list[dict[str, Any]] = []
    for label in ("real", "synthetic", "tampered"):
        selected.extend(buckets[label])
    return selected[:max_samples]


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
        selected = _select_samples(valid_rows, self.max_samples, self.samples_per_class)

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
        artifact_manifest = {
            "source_manifest_path": str(self.input_manifest),
            "source_manifest_audit_summary": audit["audit_summary_path"],
            "meta_jsonl": meta_path,
            "pair_index_json": pair_index_path,
            "images_dir": str(images_dir),
            "tamper_masks_dir": str(masks_dir),
            "ignore_masks_dir": str(ignore_dir),
            "debug_overlays_dir": str(debug_dir),
            "sample_count": len(selected),
            "profiles": self.profiles,
            "severity": self.severity,
        }
        artifact_manifest_path = _write_json(output_root / "artifact_manifest.json", artifact_manifest)
        artifact_manifest["artifact_manifest_path"] = artifact_manifest_path
        _write_json(output_root / "artifact_manifest.json", artifact_manifest)
        return artifact_manifest
