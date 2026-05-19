#!/usr/bin/env python3
"""Standalone tests for the pre-SNS integrated smoke."""

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

from cv_forensics.pre_sns_integrated_model import route_loss_availability  # noqa: E402
from scripts.agent.validate_pre_sns_integrated_smoke_config import (  # noqa: E402
    APPROVAL_TEXT,
    load_config,
    validate_config,
)
from scripts.training.run_pre_sns_integrated_smoke import main as runner_main  # noqa: E402


EXAMPLE_CONFIG = REPO_ROOT / "configs" / "training" / "pre_sns_integrated_smoke.example.json"


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
        "config_kind": "approved_local_pre_sns_integrated_smoke",
        "smoke_name": "temporary_pre_sns_integrated_smoke",
        "execution_mode": "approved_local_tiny_smoke",
        "approved_real_data_access": True,
        "user_approval_text": APPROVAL_TEXT,
        "no_download": True,
        "no_network": True,
        "no_outputs": True,
        "no_checkpoints": True,
        "no_sns_augmentation": True,
        "no_sns_perturbation_eval": True,
        "class_policy": {"labels": ["real", "full_synthetic", "tampered"]},
        "family_policy": {
            "labels": ["LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A"],
            "loss_applies_only_when_family_label_present": True
        },
        "localization_policy": {
            "loss_applies_only_to_tampered_samples_with_masks": True,
            "threshold_tau": 0.5
        },
        "loss_routing_policy": {
            "class_loss": "all samples",
            "family_loss": "only samples with family labels",
            "localization_loss": "only tampered samples with masks",
            "total_loss": "routed weighted sum"
        },
        "tiny_limits": {"max_samples": 4, "max_steps": 2, "max_epochs": 1, "max_image_size": 8, "cpu_only": True},
        "requested_run_limits": {"max_samples": 3, "max_steps": 1, "max_epochs": 1},
        "approved_local_roots": [root],
        "recursive_scan": False,
        "training_smoke_policy": {"family_loss_weight": 1.0, "localization_loss_weight": 1.0},
        "ordinary_prose": "This authoritative authentication policy is documented without credentials.",
        "sample_manifest": [
            {
                "sample_id": "real_001",
                "image_path": os.path.join(root, "real_001.png"),
                "class_label": "real",
                "family_label": "Real-or-N/A"
            },
            {
                "sample_id": "synthetic_001",
                "image_path": os.path.join(root, "synthetic_001.png"),
                "class_label": "full_synthetic",
                "family_label": "LatDiff"
            },
            {
                "sample_id": "tampered_001",
                "image_path": os.path.join(root, "tampered_001.png"),
                "mask_path": os.path.join(root, "tampered_001_mask.png"),
                "class_label": "tampered"
            }
        ]
    }


def mutate(raw: dict, mutator) -> dict:
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
            (lambda raw: raw["sample_manifest"][0].update({"image_path": str(REPO_ROOT / "outputs" / "x.png")}), "protected"),
            (lambda raw: raw["sample_manifest"][0].update({"image_path": tmp + "/"}), "explicit file path"),
            (lambda raw: raw.update({"recursive_scan": True}), "recursive scan"),
            (lambda raw: raw["sample_manifest"][0].update({"class_label": "synthetic"}), "class_label must be real"),
            (lambda raw: raw["sample_manifest"][1].update({"family_label": "UnknownModel"}), "family_label"),
            (lambda raw: raw["sample_manifest"][2].pop("mask_path"), "tampered samples must include mask_path"),
            (lambda raw: raw["sample_manifest"][0].update({"mask_path": os.path.join(tmp, "real_mask.png")}), "non-tampered"),
            (lambda raw: raw.update({"no_download": False}), "no_download must be true"),
            (lambda raw: raw.update({"no_network": False}), "no_network must be true"),
            (lambda raw: raw.update({"no_outputs": False}), "no_outputs must be true"),
            (lambda raw: raw.update({"no_checkpoints": False}), "no_checkpoints must be true"),
            (lambda raw: raw.update({"no_sns_augmentation": False}), "no_sns_augmentation must be true"),
            (lambda raw: raw.update({"no_sns_perturbation_eval": False}), "no_sns_perturbation_eval must be true"),
            (lambda raw: raw.update({"api_key": "value"}), "secret-like key"),
            (lambda raw: raw["sample_manifest"][0].update({"image_path": os.path.join(tmp, "..", "x.png")}), "path traversal"),
        ]
        for mutator, expected in cases:
            assert_fail(mutate(base, mutator), expected)


def test_conditional_loss_routing() -> None:
    routing = route_loss_availability(
        ["real", "full_synthetic", "tampered"],
        [0, 1, None],
        [False, False, True],
    )
    assert routing["class_loss"] is True
    assert routing["family_loss"] is True
    assert routing["localization_loss"] is True
    assert routing["family_indices"] == [0, 1]
    assert routing["localization_indices"] == [2]
    assert routing["localization_loss_applied_to_tampered_only"] is True


def _write_tiny_fixtures(root: str) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        raise RuntimeError("PIL unavailable")
    for name, color in {
        "real_001.png": (20, 20, 20),
        "synthetic_001.png": (170, 50, 50),
        "tampered_001.png": (50, 160, 50),
    }.items():
        Image.new("RGB", (8, 8), color).save(os.path.join(root, name))
    mask = Image.new("L", (8, 8), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((2, 2, 5, 5), fill=255)
    mask.save(os.path.join(root, "tampered_001_mask.png"))


def test_runner_if_optional_dependencies_available() -> None:
    try:
        import torch  # noqa: F401
        import PIL  # noqa: F401
    except Exception:
        print("SKIP: torch or PIL unavailable; runner execution skipped.")
        return
    with tempfile.TemporaryDirectory() as tmp:
        _write_tiny_fixtures(tmp)
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
        assert summary["marker"] == "PRE_SNS_INTEGRATED_SMOKE_OK"
        assert summary["class_loss_finite"] is True
        assert summary["family_loss_finite"] is True
        assert summary["localization_loss_finite"] is True
        assert summary["total_loss_finite"] is True
        assert summary["loss_routing"]["localization_loss_applied_to_tampered_only"] is True
        assert summary["schema_output"]["reason"]
        assert before == after


def main() -> int:
    tests = [
        test_example_config_passes,
        test_approved_local_config_passes,
        test_validation_rejections,
        test_conditional_loss_routing,
        test_runner_if_optional_dependencies_available,
    ]
    for test in tests:
        test()
    print("PRE_SNS_INTEGRATED_SMOKE_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
