#!/usr/bin/env python3
"""Standalone tests for the CF-Small multi-head tiny smoke validator."""

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

from scripts.agent.validate_cf_small_multihead_tiny_smoke_config import (  # noqa: E402
    APPROVAL_TEXT,
    load_config,
    validate_config,
)
from scripts.training.run_cf_small_multihead_tiny_smoke import main as runner_main  # noqa: E402


EXAMPLE_CONFIG = REPO_ROOT / "configs" / "training" / "cf_small_multihead_tiny_smoke.example.json"


def assert_pass(raw: dict) -> None:
    errors = validate_config(raw)
    assert not errors, "\n".join(errors)


def assert_fail(raw: dict, expected: str) -> None:
    errors = validate_config(raw)
    assert errors, "expected validation failure"
    joined = "\n".join(errors)
    assert expected in joined, joined


def make_approved(root: str) -> dict:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_local_multihead_smoke",
        "smoke_name": "temporary_cf_small_multihead_smoke",
        "dataset_name": "Community Forensics-Small",
        "stage": "non_sns_cf_small_multihead_provenance_tiny_smoke",
        "execution_mode": "approved_local_tiny_smoke",
        "approved_real_data_access": True,
        "user_approval_text": APPROVAL_TEXT,
        "no_download": True,
        "no_network": True,
        "no_outputs": True,
        "no_checkpoints": True,
        "no_sns_augmentation": True,
        "no_sns_perturbation_eval": True,
        "class_policy": {"labels": ["real", "synthetic"]},
        "family_policy": {
            "labels": ["LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A"],
            "real_sample_family_label": "Real-or-N/A",
            "preserve_metadata_when_available": ["model_name", "subset"]
        },
        "tiny_limits": {
            "max_samples": 4,
            "max_steps": 2,
            "max_epochs": 1,
            "max_image_size": 8,
            "cpu_only": True
        },
        "requested_run_limits": {
            "max_samples": 3,
            "max_steps": 1,
            "max_epochs": 1
        },
        "approved_local_roots": [root],
        "recursive_scan": False,
        "training_smoke_policy": {"family_loss_weight": 1.0},
        "ordinary_prose": "This authoritative authentication note is normal prose, not a credential.",
        "sample_manifest": [
            {
                "sample_id": "real_001",
                "image_path": os.path.join(root, "real_001.png"),
                "label": "real",
                "architecture": "camera_original",
                "family_label": "Real-or-N/A",
                "model_name": "original",
                "subset": "tiny"
            },
            {
                "sample_id": "synthetic_001",
                "image_path": os.path.join(root, "synthetic_001.png"),
                "label": "synthetic",
                "architecture": "latent_diffusion",
                "family_label": "LatDiff",
                "model_name": "symbolic-latdiff",
                "subset": "tiny"
            },
            {
                "sample_id": "synthetic_002",
                "image_path": os.path.join(root, "synthetic_002.png"),
                "label": "synthetic",
                "architecture": "gan",
                "family_label": "GAN",
                "model_name": "symbolic-gan",
                "subset": "tiny"
            }
        ]
    }


def with_mutation(raw: dict, mutator) -> dict:
    changed = copy.deepcopy(raw)
    mutator(changed)
    return changed


def test_example_config_passes() -> None:
    assert_pass(load_config(EXAMPLE_CONFIG))


def test_approved_local_config_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert_pass(make_approved(tmp))


def test_validation_rejections() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = make_approved(tmp)
        cases = [
            (lambda raw: raw.update({"user_approval_text": "no"}), "user_approval_text"),
            (lambda raw: raw["sample_manifest"][0].update({"image_path": "https://example.invalid/a.png"}), "URL"),
            (lambda raw: raw["sample_manifest"][0].update({"image_path": str(REPO_ROOT / "data" / "x.png")}), "protected"),
            (lambda raw: raw["sample_manifest"][0].update({"image_path": tmp + "/"}), "explicit file path"),
            (lambda raw: raw.update({"recursive_scan": True}), "recursive scan"),
            (lambda raw: raw["sample_manifest"][0].update({"label": "tampered"}), "label must be real or synthetic"),
            (lambda raw: raw["sample_manifest"][1].update({"family_label": "ExactModel"}), "family_label"),
            (lambda raw: raw["sample_manifest"][0].update({"family_label": "GAN"}), "real samples must use Real-or-N/A"),
            (lambda raw: raw["sample_manifest"][1].update({"family_label": "Real-or-N/A"}), "synthetic samples cannot use Real-or-N/A"),
            (lambda raw: raw["sample_manifest"][1].pop("architecture"), "architecture is required"),
            (lambda raw: [sample.update({"label": "synthetic", "family_label": "GAN"}) for sample in raw["sample_manifest"]], "both real and synthetic"),
            (lambda raw: (raw.update({"unknown_synthetic_family_allowed": True}), [sample.update({"family_label": "Real-or-N/A"}) for sample in raw["sample_manifest"] if sample["label"] == "synthetic"]), "at least one synthetic family"),
            (lambda raw: raw["tiny_limits"].update({"max_samples": 17}), "max_samples must be <="),
            (lambda raw: raw["tiny_limits"].update({"max_steps": 6}), "max_steps must be <="),
            (lambda raw: raw["tiny_limits"].update({"max_epochs": 3}), "max_epochs must be <="),
            (lambda raw: raw.update({"no_download": False}), "no_download must be true"),
            (lambda raw: raw.update({"no_network": False}), "no_network must be true"),
            (lambda raw: raw.update({"no_outputs": False}), "no_outputs must be true"),
            (lambda raw: raw.update({"no_checkpoints": False}), "no_checkpoints must be true"),
            (lambda raw: raw.update({"no_sns_augmentation": False}), "no_sns_augmentation must be true"),
            (lambda raw: raw.update({"no_sns_perturbation_eval": False}), "no_sns_perturbation_eval must be true"),
            (lambda raw: raw.update({"api_key": "value"}), "secret-like key"),
        ]
        for mutator, expected in cases:
            assert_fail(with_mutation(base, mutator), expected)


def test_unknown_synthetic_family_can_be_explicitly_allowed_when_other_family_exists() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw = make_approved(tmp)
        raw["unknown_synthetic_family_allowed"] = True
        raw["sample_manifest"][1]["family_label"] = "Real-or-N/A"
        assert_pass(raw)


def test_secret_like_values_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw = make_approved(tmp)
        raw["validation_notes"] = ["password=not_allowed"]
        assert_fail(raw, "secret-like value")


def _write_tiny_images(root: str) -> None:
    try:
        from PIL import Image
    except Exception:
        raise RuntimeError("PIL unavailable")
    colors = {
        "real_001.png": (20, 20, 20),
        "synthetic_001.png": (180, 50, 50),
        "synthetic_002.png": (50, 160, 50),
    }
    for name, color in colors.items():
        image = Image.new("RGB", (8, 8), color)
        image.save(os.path.join(root, name))


def test_runner_if_optional_dependencies_available() -> None:
    try:
        import torch  # noqa: F401
        import PIL  # noqa: F401
    except Exception:
        print("SKIP: torch or PIL unavailable; runner execution skipped.")
        return
    with tempfile.TemporaryDirectory() as tmp:
        _write_tiny_images(tmp)
        raw = make_approved(tmp)
        config_path = os.path.join(tmp, "approved.local.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(raw, handle)
        before = sorted(os.listdir(tmp))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = runner_main([config_path])
        after = sorted(os.listdir(tmp))
        assert exit_code == 0, stdout.getvalue()
        summary = json.loads(stdout.getvalue())
        assert summary["marker"] == "CF_SMALL_MULTIHEAD_TINY_SMOKE_OK"
        assert summary["class_loss_finite"] is True
        assert summary["family_loss_finite"] is True
        assert summary["total_loss_finite"] is True
        assert summary["no_outputs"] is True
        assert summary["no_checkpoints"] is True
        assert before == after


def main() -> int:
    tests = [
        test_example_config_passes,
        test_approved_local_config_passes,
        test_validation_rejections,
        test_unknown_synthetic_family_can_be_explicitly_allowed_when_other_family_exists,
        test_secret_like_values_are_rejected,
        test_runner_if_optional_dependencies_available,
    ]
    for test in tests:
        test()
    print("CF_SMALL_MULTIHEAD_TINY_SMOKE_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
