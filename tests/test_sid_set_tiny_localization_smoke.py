#!/usr/bin/env python3
"""Standalone tests for the SID-Set tiny localization smoke validator."""

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

from scripts.agent.validate_sid_set_tiny_localization_smoke_config import (  # noqa: E402
    APPROVAL_TEXT,
    load_config,
    validate_config,
)
from scripts.training.run_sid_set_tiny_localization_smoke import main as runner_main  # noqa: E402


EXAMPLE_CONFIG = REPO_ROOT / "configs" / "training" / "sid_set_tiny_localization_smoke.example.json"


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
        "config_kind": "approved_local_sid_tiny_localization_smoke",
        "smoke_name": "temporary_sid_set_tiny_localization_smoke",
        "dataset_name": "SID-Set",
        "stage": "non_sns_sid_set_tiny_localization_smoke",
        "execution_mode": "approved_local_tiny_smoke",
        "approved_real_data_access": True,
        "user_approval_text": APPROVAL_TEXT,
        "no_download": True,
        "no_network": True,
        "no_outputs": True,
        "no_checkpoints": True,
        "no_sns_augmentation": True,
        "no_sns_perturbation_eval": True,
        "class_policy": {
            "labels": ["real", "full_synthetic", "tampered"],
            "label_id_map": {"0": "real", "1": "full_synthetic", "2": "tampered"}
        },
        "mask_policy": {
            "tampered_samples_require_masks": True,
            "real_samples_require_masks": False,
            "full_synthetic_samples_require_masks": False,
            "mask_path_allowed_only_for_tampered": True
        },
        "family_policy": {
            "sid_set_family_labels_required": False,
            "family_loss_applied_to_sid_set_samples": False
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
        "training_smoke_policy": {"mask_loss_weight": 1.0},
        "ordinary_prose": "This authoritative authentication note is normal prose, not a credential.",
        "sample_manifest": [
            {
                "sample_id": "real_001",
                "image_path": os.path.join(root, "real_001.png"),
                "label": "real",
                "label_id": 0,
                "split": "tiny",
                "img_id": "r1"
            },
            {
                "sample_id": "synthetic_001",
                "image_path": os.path.join(root, "synthetic_001.png"),
                "label": "full_synthetic",
                "label_id": 1,
                "split": "tiny",
                "img_id": "s1"
            },
            {
                "sample_id": "tampered_001",
                "image_path": os.path.join(root, "tampered_001.png"),
                "mask_path": os.path.join(root, "tampered_001_mask.png"),
                "label": "tampered",
                "label_id": 2,
                "split": "tiny",
                "img_id": "t1"
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


def test_non_tampered_samples_do_not_require_mask_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw = make_approved(tmp)
        assert "mask_path" not in raw["sample_manifest"][0]
        assert "mask_path" not in raw["sample_manifest"][1]
        assert_pass(raw)


def test_validation_rejections() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = make_approved(tmp)
        cases = [
            (lambda raw: raw.update({"user_approval_text": "no"}), "user_approval_text"),
            (lambda raw: raw["sample_manifest"][0].update({"image_path": "https://example.invalid/a.png"}), "URL"),
            (lambda raw: raw["sample_manifest"][0].update({"image_path": str(REPO_ROOT / "data" / "x.png")}), "protected"),
            (lambda raw: raw["sample_manifest"][0].update({"image_path": tmp + "/"}), "explicit file path"),
            (lambda raw: raw.update({"recursive_scan": True}), "recursive scan"),
            (lambda raw: raw["sample_manifest"][0].update({"label": "synthetic"}), "label must be real, full_synthetic, or tampered"),
            (lambda raw: raw["sample_manifest"][1].update({"label_id": 2}), "label_id mismatch"),
            (lambda raw: [sample.update({"label": "full_synthetic", "label_id": 1}) for sample in raw["sample_manifest"] if sample["label"] == "real"], "missing required class real"),
            (lambda raw: [sample.update({"label": "real", "label_id": 0}) for sample in raw["sample_manifest"] if sample["label"] == "full_synthetic"], "missing required class full_synthetic"),
            (lambda raw: [sample.update({"label": "real", "label_id": 0}) for sample in raw["sample_manifest"] if sample["label"] == "tampered"], "missing required class tampered"),
            (lambda raw: raw["sample_manifest"][2].pop("mask_path"), "tampered samples must include mask_path"),
            (lambda raw: raw["sample_manifest"][0].update({"mask_path": os.path.join(tmp, "real_mask.png")}), "non-tampered samples must not include mask_path"),
            (lambda raw: raw["tiny_limits"].update({"max_samples": 13}), "max_samples must be <="),
            (lambda raw: raw["tiny_limits"].update({"max_steps": 6}), "max_steps must be <="),
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
            assert_fail(with_mutation(base, mutator), expected)


def test_secret_like_values_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw = make_approved(tmp)
        raw["validation_notes"] = ["password=not_allowed"]
        assert_fail(raw, "secret-like value")


def _write_tiny_images_and_mask(root: str) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        raise RuntimeError("PIL unavailable")
    colors = {
        "real_001.png": (20, 20, 20),
        "synthetic_001.png": (170, 50, 50),
        "tampered_001.png": (50, 160, 50),
    }
    for name, color in colors.items():
        image = Image.new("RGB", (8, 8), color)
        image.save(os.path.join(root, name))
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
        _write_tiny_images_and_mask(tmp)
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
        assert summary["marker"] == "SID_SET_TINY_LOCALIZATION_SMOKE_OK"
        assert summary["class_loss_finite"] is True
        assert summary["mask_loss_finite"] is True
        assert summary["total_loss_finite"] is True
        assert summary["localization_loss_applied_to_tampered_only"] is True
        assert summary["no_outputs"] is True
        assert summary["no_checkpoints"] is True
        assert before == after


def main() -> int:
    tests = [
        test_example_config_passes,
        test_approved_local_config_passes,
        test_non_tampered_samples_do_not_require_mask_path,
        test_validation_rejections,
        test_secret_like_values_are_rejected,
        test_runner_if_optional_dependencies_available,
    ]
    for test in tests:
        test()
    print("SID_SET_TINY_LOCALIZATION_SMOKE_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
