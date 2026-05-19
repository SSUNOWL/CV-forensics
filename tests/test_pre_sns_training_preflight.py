#!/usr/bin/env python3
"""Standalone tests for the pre-SNS training preflight dry-run."""

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
from cv_forensics.pre_sns_preflight import summarize_manifest_readiness  # noqa: E402
from scripts.agent.validate_pre_sns_training_preflight_config import (  # noqa: E402
    APPROVAL_TEXT,
    load_config,
    validate_config,
)
from scripts.training.run_pre_sns_training_preflight import main as runner_main  # noqa: E402


EXAMPLE_CONFIG = REPO_ROOT / "configs" / "training" / "pre_sns_training_preflight.example.json"


def assert_pass(raw: dict) -> None:
    errors = validate_config(raw)
    assert not errors, "\n".join(errors)


def assert_fail(raw: dict, expected: str) -> None:
    errors = validate_config(raw)
    assert errors, "expected validation failure"
    joined = "\n".join(errors)
    assert expected in joined, joined


def base_samples(root: str) -> list[dict]:
    return [
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
        },
    ]


def make_manifest(root: str) -> dict:
    return normalize_manifest(base_samples(root))


def make_approved(root: str, manifest_path: str) -> dict:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_local_pre_sns_training_preflight",
        "preflight_name": "temporary_preflight",
        "approved_real_data_access": True,
        "user_approval_text": APPROVAL_TEXT,
        "no_download": True,
        "no_network": True,
        "no_outputs": True,
        "no_checkpoints": True,
        "no_real_training": True,
        "no_sns_augmentation": True,
        "approved_local_roots": [root],
        "manifest_path": manifest_path,
        "recursive_scan": False,
        "expected_manifest_schema": {"manifest_kind": "pre_sns_unified_dataset_manifest"},
        "tiny_preflight_limits": {
            "max_samples": 6,
            "max_image_size": 8,
            "cpu_only": True,
            "dry_run_optimizer_step": False
        },
        "loss_routing_policy": {
            "class_loss": "all samples",
            "family_loss": "family samples",
            "localization_loss": "tampered mask samples"
        },
        "ordinary_prose": "This authoritative authentication note is documentation only."
    }


def mutate(raw: dict, mutator) -> dict:
    changed = copy.deepcopy(raw)
    mutator(changed)
    return changed


def test_example_config_passes() -> None:
    assert_pass(load_config(EXAMPLE_CONFIG))


def test_approved_config_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = os.path.join(tmp, "manifest.json")
        assert_pass(make_approved(tmp, manifest_path))


def test_config_rejections() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = os.path.join(tmp, "manifest.json")
        base = make_approved(tmp, manifest_path)
        cases = [
            (lambda raw: raw.update({"user_approval_text": "no"}), "user_approval_text"),
            (lambda raw: raw.update({"manifest_path": "https://example.invalid/manifest.json"}), "URL"),
            (lambda raw: raw.update({"manifest_path": str(REPO_ROOT / "data" / "manifest.json")}), "protected"),
            (lambda raw: raw.update({"manifest_path": os.path.join(tmp, "..", "manifest.json")}), "path traversal"),
            (lambda raw: raw.update({"no_outputs": False}), "no_outputs must be true"),
            (lambda raw: raw.update({"no_checkpoints": False}), "no_checkpoints must be true"),
            (lambda raw: raw.update({"no_real_training": False}), "no_real_training must be true"),
            (lambda raw: raw["tiny_preflight_limits"].update({"cpu_only": False}), "cpu_only must be true"),
        ]
        for mutator, expected in cases:
            assert_fail(mutate(base, mutator), expected)


def test_preflight_summary_rejections() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest = make_manifest(tmp)
        assert summarize_manifest_readiness(manifest, 6, True)["ready"] is True
        missing_class = copy.deepcopy(manifest)
        missing_class["samples"] = [s for s in missing_class["samples"] if s["class_label"] != "tampered"]
        assert summarize_manifest_readiness(missing_class, 6, True)["ready"] is False
        missing_family = copy.deepcopy(manifest)
        for sample in missing_family["samples"]:
            if "family_label" in sample:
                sample.pop("family_label")
            sample["tasks_available"]["family"] = False
            sample["loss_routing"]["family_loss"] = False
        summary = summarize_manifest_readiness(missing_family, 6, True)
        assert summary["ready"] is False
        assert any(issue["field"] == "family_coverage" for issue in summary["issues"])
        missing_mask = copy.deepcopy(manifest)
        for sample in missing_mask["samples"]:
            if sample["class_label"] == "tampered":
                sample.pop("mask_path", None)
                sample["tasks_available"]["localization"] = False
                sample["loss_routing"]["localization_loss"] = False
        summary = summarize_manifest_readiness(missing_mask, 6, True)
        assert summary["ready"] is False
        assert any(issue["field"] == "tampered_mask_coverage" for issue in summary["issues"])


def _write_tiny_fixtures(root: str) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        raise RuntimeError("PIL unavailable")
    for name, color in {
        "cf_real_001.png": (20, 20, 20),
        "cf_syn_001.png": (170, 50, 50),
        "sid_tampered_001.png": (50, 160, 50),
    }.items():
        Image.new("RGB", (8, 8), color).save(os.path.join(root, name))
    mask = Image.new("L", (8, 8), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((2, 2, 5, 5), fill=255)
    mask.save(os.path.join(root, "sid_tampered_001_mask.png"))


def test_runner_if_optional_dependencies_available() -> None:
    try:
        import torch  # noqa: F401
        import PIL  # noqa: F401
    except Exception:
        print("SKIP: torch or PIL unavailable; runner execution skipped.")
        return
    with tempfile.TemporaryDirectory() as tmp:
        _write_tiny_fixtures(tmp)
        manifest_path = os.path.join(tmp, "manifest.json")
        config_path = os.path.join(tmp, "preflight.local.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(make_manifest(tmp), handle)
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(make_approved(tmp, manifest_path), handle)
        before = sorted(os.listdir(tmp))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = runner_main([config_path])
        after = sorted(os.listdir(tmp))
        assert exit_code == 0, stdout.getvalue()
        summary = json.loads(stdout.getvalue())
        assert summary["marker"] == "PRE_SNS_TRAINING_PREFLIGHT_OK"
        assert summary["total_loss_finite"] is True
        assert summary["optimizer_step_ran"] is False
        assert summary["readiness"]["ready"] is True
        assert before == after


def main() -> int:
    tests = [
        test_example_config_passes,
        test_approved_config_passes,
        test_config_rejections,
        test_preflight_summary_rejections,
        test_runner_if_optional_dependencies_available,
    ]
    for test in tests:
        test()
    print("PRE_SNS_TRAINING_PREFLIGHT_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
