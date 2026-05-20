#!/usr/bin/env python3
"""Standalone tests for the pre-SNS dataset manifest gate."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_manifest import normalize_manifest  # noqa: E402
from scripts.agent.validate_pre_sns_dataset_manifest import (  # noqa: E402
    APPROVAL_TEXT,
    load_config,
    validate_config,
)
from scripts.data.build_pre_sns_dataset_manifest import main as builder_main  # noqa: E402


EXAMPLE_CONFIG = REPO_ROOT / "configs" / "training" / "pre_sns_dataset_manifest.example.json"
TEMP_ROOT = Path.home() / ".codex" / "memories"


def temporary_local_root():
    return tempfile.TemporaryDirectory(dir=str(TEMP_ROOT))


def assert_pass(raw: dict) -> None:
    errors = validate_config(raw)
    assert not errors, "\n".join(errors)


def assert_fail(raw: dict, expected: str) -> None:
    errors = validate_config(raw)
    assert errors, "expected validation failure"
    joined = "\n".join(errors)
    assert expected in joined, joined


def make_approved(root: str, output_root: str) -> dict:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_local_pre_sns_manifest",
        "manifest_name": "temporary_pre_sns_manifest",
        "approved_real_data_access": True,
        "user_approval_text": APPROVAL_TEXT,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_outputs": True,
        "no_checkpoints": True,
        "no_sns_augmentation": True,
        "class_policy": {"labels": ["real", "full_synthetic", "tampered"]},
        "family_policy": {"labels": ["LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A"]},
        "source_dataset_policy": {"supported_sources": ["community_forensics_small", "sid_set"]},
        "approved_local_roots": [root],
        "approved_local_output_roots": [output_root],
        "manifest_output_path": os.path.join(output_root, "pre_sns_manifest.json"),
        "recursive_scan": False,
        "ordinary_prose": "This authoritative authentication policy is documentation only.",
        "samples": [
            {
                "source_dataset": "community_forensics_small",
                "sample_id": "cf_real_001",
                "image_path": os.path.join(root, "cf_real_001.png"),
                "class_label": "real",
                "family_label": "Real-or-N/A",
                "architecture": "camera_original",
                "model_name": "original",
                "subset": "tiny"
            },
            {
                "source_dataset": "community_forensics_small",
                "sample_id": "cf_syn_001",
                "image_path": os.path.join(root, "cf_syn_001.png"),
                "class_label": "full_synthetic",
                "family_label": "LatDiff",
                "architecture": "latent_diffusion",
                "model_name": "symbolic-latdiff",
                "subset": "tiny"
            },
            {
                "source_dataset": "sid_set",
                "sample_id": "sid_tampered_001",
                "image_path": os.path.join(root, "sid_tampered_001.png"),
                "mask_path": os.path.join(root, "sid_tampered_001_mask.png"),
                "class_label": "tampered",
                "label_id": 2,
                "split": "tiny",
                "img_id": "sid-t1"
            }
        ]
    }


def touch_sample_files(root: str) -> None:
    for name in (
        "cf_real_001.png",
        "cf_syn_001.png",
        "sid_tampered_001.png",
        "sid_tampered_001_mask.png",
    ):
        with open(os.path.join(root, name), "wb") as handle:
            handle.write(b"tiny fixture")


def as_sample_manifest(raw: dict) -> dict:
    changed = copy.deepcopy(raw)
    changed["sample_manifest"] = changed.pop("samples")
    return changed


def with_local_manifest_aliases(raw: dict) -> dict:
    changed = as_sample_manifest(raw)
    for sample in changed["sample_manifest"]:
        if sample["source_dataset"] == "community_forensics_small":
            sample["source_dataset"] = "Community Forensics-Small"
        elif sample["source_dataset"] == "sid_set":
            sample["source_dataset"] = "SID-Set"
        sample["label"] = sample.pop("class_label")
    return changed


def mutate(raw: dict, mutator) -> dict:
    changed = copy.deepcopy(raw)
    mutator(changed)
    return changed


def test_example_config_passes() -> None:
    assert_pass(load_config(EXAMPLE_CONFIG))


def test_approved_config_passes_and_normalizes() -> None:
    with temporary_local_root() as tmp:
        touch_sample_files(tmp)
        output_root = os.path.join(tmp, "manifest_out")
        raw = make_approved(tmp, output_root)
        assert_pass(raw)
        manifest = normalize_manifest(raw["samples"])
        assert manifest["summary"]["sample_count"] == 3
        assert manifest["summary"]["family_supervision_count"] == 2
        assert manifest["summary"]["localization_supervision_count"] == 1


def test_sample_manifest_absolute_paths_pass() -> None:
    with temporary_local_root() as tmp:
        touch_sample_files(tmp)
        output_root = os.path.join(tmp, "manifest_out")
        raw = with_local_manifest_aliases(make_approved(tmp, output_root))
        assert_pass(raw)


def test_rejections() -> None:
    with temporary_local_root() as tmp:
        touch_sample_files(tmp)
        output_root = os.path.join(tmp, "manifest_out")
        base = make_approved(tmp, output_root)
        outside = tempfile.TemporaryDirectory(dir=str(TEMP_ROOT))
        outside_path = outside.name
        with open(os.path.join(outside_path, "outside.png"), "wb") as handle:
            handle.write(b"outside")
        cases = [
            (lambda raw: raw.update({"user_approval_text": "no"}), "user_approval_text"),
            (lambda raw: raw["samples"][0].update({"image_path": "https://example.invalid/a.png"}), "URL"),
            (lambda raw: raw["samples"][0].update({"image_path": str(REPO_ROOT / "data" / "x.png")}), "protected"),
            (lambda raw: raw["samples"][0].update({"image_path": os.path.join(tmp, "..", "x.png")}), "path traversal"),
            (lambda raw: raw["samples"][0].update({"image_path": os.path.join(outside_path, "outside.png")}), "under approved roots"),
            (lambda raw: raw["samples"][2].update({"mask_path": os.path.join(outside_path, "outside.png")}), "under approved roots"),
            (lambda raw: raw["samples"][0].update({"image_path": tmp}), "must exist as a file"),
            (lambda raw: raw["samples"][1].update({"family_label": "ExactModel"}), "family_label"),
            (lambda raw: [sample.update({"class_label": "real"}) for sample in raw["samples"] if sample["class_label"] == "tampered"], "missing required class coverage: tampered"),
            (lambda raw: raw["samples"][2].update({"class_label": "real", "label_id": 0}), "non-tampered SID-Set sample must not include mask_path"),
            (lambda raw: raw["samples"][2].pop("mask_path"), "tampered SID-Set sample requires mask_path"),
            (lambda raw: raw.update({"manifest_output_path": str(REPO_ROOT / "outputs" / "manifest.json")}), "protected"),
            (lambda raw: raw.update({"no_download": False}), "no_download must be true"),
            (lambda raw: raw.update({"no_training": False}), "no_training must be true"),
            (lambda raw: raw.update({"no_sns_augmentation": False}), "no_sns_augmentation must be true"),
        ]
        for mutator, expected in cases:
            assert_fail(mutate(base, mutator), expected)
        outside.cleanup()


def test_builder_writes_only_approved_temp_manifest() -> None:
    with temporary_local_root() as tmp:
        touch_sample_files(tmp)
        output_root = os.path.join(tmp, "manifest_out")
        raw = make_approved(tmp, output_root)
        config_path = os.path.join(tmp, "approved.local.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(raw, handle)
        before_tmp = sorted(os.listdir(tmp))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = builder_main([config_path])
        assert exit_code == 0, stdout.getvalue()
        summary = json.loads(stdout.getvalue())
        assert summary["marker"] == "PRE_SNS_DATASET_MANIFEST_OK"
        assert os.path.exists(raw["manifest_output_path"])
        with open(raw["manifest_output_path"], "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        assert manifest["summary"]["sample_count"] == 3
        after_tmp = sorted(os.listdir(tmp))
        assert set(after_tmp) - set(before_tmp) == {"manifest_out"}


def main() -> int:
    tests = [
        test_example_config_passes,
        test_approved_config_passes_and_normalizes,
        test_sample_manifest_absolute_paths_pass,
        test_rejections,
        test_builder_writes_only_approved_temp_manifest,
    ]
    for test in tests:
        test()
    print("PRE_SNS_DATASET_MANIFEST_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
