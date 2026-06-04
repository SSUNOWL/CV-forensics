"""Deterministic clean/augmented pair generator for SNSAug V2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .configs import SNSAugV2Config
from .sns_augmentor import SNSAugV2Augmentor


def _safe_base_id(sample: dict[str, Any], index: int) -> str:
    raw = str(sample.get("base_id") or sample.get("sample_id") or sample.get("id") or f"sample_{index:06d}")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)


def _select_samples(samples: list[dict[str, Any]], max_samples: int, samples_per_class: int | None) -> list[dict[str, Any]]:
    if samples_per_class is None:
        return samples[:max_samples]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        label = str(sample.get("content_label") or sample.get("label") or sample.get("class_label") or "")
        buckets.setdefault(label, []).append(sample)
    selected: list[dict[str, Any]] = []
    for label in sorted(buckets):
        selected.extend(buckets[label][:samples_per_class])
    return selected[:max_samples]


class SNSAugV2PairGenerator:
    def __init__(
        self,
        *,
        input_manifest: str | Path,
        output_root: str | Path,
        profiles: list[str],
        severity: str = "medium",
        seed: int = 0,
        split: str = "train",
        max_samples: int = 100,
        samples_per_class: int | None = None,
        output_size: int | None = None,
        font_path: str | None = None,
    ) -> None:
        self.input_manifest = Path(input_manifest)
        self.output_root = Path(output_root)
        self.profiles = list(profiles)
        self.severity = severity
        self.seed = int(seed)
        self.split = split
        self.max_samples = int(max_samples)
        self.samples_per_class = samples_per_class
        self.output_size = output_size
        self.font_path = font_path

    def _load_manifest(self) -> list[dict[str, Any]]:
        with open(self.input_manifest, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        samples = raw.get("samples", raw) if isinstance(raw, dict) else raw
        if not isinstance(samples, list):
            raise ValueError("pair generator manifest must be a list or contain samples")
        return [sample for sample in samples if isinstance(sample, dict)]

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _save_image(self, image: Image.Image, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return str(path)

    def run(self) -> dict[str, Any]:
        samples = _select_samples(self._load_manifest(), self.max_samples, self.samples_per_class)
        rows: list[dict[str, Any]] = []
        pair_index: list[dict[str, Any]] = []
        images_dir = self.output_root / "images"
        masks_dir = self.output_root / "tamper_masks"
        ignore_dir = self.output_root / "ignore_masks"
        for index, sample in enumerate(samples):
            base_id = _safe_base_id(sample, index)
            image_path = Path(str(sample["image_path"]))
            mask_path = Path(str(sample["mask_path"])) if sample.get("mask_path") else None
            with Image.open(image_path) as img:
                clean_image = img.convert("RGB")
            clean_mask = None
            if mask_path is not None:
                with Image.open(mask_path) as mask_img:
                    clean_mask = mask_img.convert("L")
            clean_out = images_dir / f"{base_id}__clean.png"
            clean_mask_out = masks_dir / f"{base_id}__clean.png"
            clean_ignore_out = ignore_dir / f"{base_id}__clean.png"
            self._save_image(clean_image, clean_out)
            if clean_mask is not None:
                self._save_image(clean_mask, clean_mask_out)
            self._save_image(Image.new("L", clean_image.size, 0), clean_ignore_out)
            clean_row = {
                "base_id": base_id,
                "source_dataset": sample.get("source_dataset"),
                "split": self.split,
                "source_path": str(image_path),
                "content_label": sample.get("content_label") or sample.get("label") or sample.get("class_label"),
                "family_label": sample.get("family_label"),
                "view": "clean",
                "profile": "clean",
                "severity": self.severity,
                "seed": self.seed + index,
                "image_path": str(clean_out),
                "tamper_mask_path": str(clean_mask_out) if clean_mask is not None else None,
                "ignore_mask_path": str(clean_ignore_out),
                "aug_meta": {"label_preserved": True},
                "overlay_boxes": [],
                "geometric_transform_meta": {"type": "identity"},
                "label_preserved": True,
            }
            rows.append(clean_row)
            pair = {"base_id": base_id, "views": [clean_row["image_path"]]}
            for profile_index, profile in enumerate(self.profiles):
                cfg = SNSAugV2Config(
                    profile=profile,
                    severity=self.severity,
                    seed=self.seed + index * 100 + profile_index,
                    output_size=self.output_size,
                    font_path=self.font_path,
                )
                result = SNSAugV2Augmentor(cfg)(clean_image, clean_mask, label=clean_row["content_label"], base_id=base_id, seed=cfg.seed)
                image_out = images_dir / f"{base_id}__{profile}.png"
                mask_out = masks_dir / f"{base_id}__{profile}.png"
                ignore_out = ignore_dir / f"{base_id}__{profile}.png"
                self._save_image(result.image, image_out)
                if result.tamper_mask is not None:
                    self._save_image(result.tamper_mask, mask_out)
                self._save_image(result.ignore_mask, ignore_out)
                row = {
                    "base_id": base_id,
                    "source_dataset": sample.get("source_dataset"),
                    "split": self.split,
                    "source_path": str(image_path),
                    "content_label": clean_row["content_label"],
                    "family_label": sample.get("family_label"),
                    "view": "sns_aug",
                    "profile": profile,
                    "severity": self.severity,
                    "seed": cfg.seed,
                    "image_path": str(image_out),
                    "tamper_mask_path": str(mask_out) if result.tamper_mask is not None else None,
                    "ignore_mask_path": str(ignore_out),
                    "aug_meta": result.meta,
                    "overlay_boxes": result.meta.get("overlay_boxes", []),
                    "geometric_transform_meta": result.meta.get("geometric_transform_meta"),
                    "label_preserved": True,
                }
                rows.append(row)
                pair["views"].append(row["image_path"])
            pair_index.append(pair)
        self._write_jsonl(self.output_root / "meta.jsonl", rows)
        self._write_json(self.output_root / "pair_index.json", {"pairs": pair_index})
        artifact_manifest = {
            "meta_jsonl": str(self.output_root / "meta.jsonl"),
            "pair_index_json": str(self.output_root / "pair_index.json"),
            "images_dir": str(images_dir),
            "tamper_masks_dir": str(masks_dir),
            "ignore_masks_dir": str(ignore_dir),
            "sample_count": len(samples),
            "profiles": self.profiles,
        }
        self._write_json(self.output_root / "artifact_manifest.json", artifact_manifest)
        return artifact_manifest
