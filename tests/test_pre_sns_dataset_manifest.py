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


def mutate(raw: dict, mutator) -> dict:
    changed = copy.deepcopy(raw)
    mutator(changed)
    return changed


def test_example_config_passes() -> None:
    assert_pass(load_config(EXAMPLE_CONFIG))


def test_approved_config_passes_and_normalizes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_root = os.path.join(tmp, "manifest_out")
        raw = make_approved(tmp, output_root)
        assert_pass(raw)
        manifest = normalize_manifest(raw["samples"])
        assert manifest["summary"]["sample_count"] == 3
        assert manifest["summary"]["family_supervision_count"] == 2
        assert manifest["summary"]["localization_supervision_count"] == 1


def test_rejections() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_root = os.path.join(tmp, "manifest_out")
        base = make_approved(tmp, output_root)
        cases = [
            (lambda raw: raw.update({"user_approval_text": "no"}), "user_approval_text"),
            (lambda raw: raw["samples"][0].update({"image_path": "https://example.invalid/a.png"}), "URL"),
            (lambda raw: raw["samples"][0].update({"image_path": str(REPO_ROOT / "data" / "x.png")}), "protected"),
            (lambda raw: raw["samples"][0].update({"image_path": os.path.join(tmp, "..", "x.png")}), "path traversal"),
            (lambda raw: raw["samples"][1].update({"family_label": "ExactModel"}), "family_label"),
            (lambda raw: [sample.update({"class_label": "real"}) for sample in raw["samples"] if sample["class_label"] == "tampered"], "missing required class coverage: tampered"),
            (lambda raw: raw["samples"][2].pop("mask_path"), "tampered SID-Set sample requires mask_path"),
            (lambda raw: raw.update({"manifest_output_path": str(REPO_ROOT / "outputs" / "manifest.json")}), "protected"),
            (lambda raw: raw.update({"no_download": False}), "no_download must be true"),
            (lambda raw: raw.update({"no_training": False}), "no_training must be true"),
            (lambda raw: raw.update({"no_sns_augmentation": False}), "no_sns_augmentation must be true"),
        ]
        for mutator, expected in cases:
            assert_fail(mutate(base, mutator), expected)


def test_builder_writes_only_approved_temp_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
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
        test_rejections,
        test_builder_writes_only_approved_temp_manifest,
    ]
    for test in tests:
        test()
    print("PRE_SNS_DATASET_MANIFEST_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
