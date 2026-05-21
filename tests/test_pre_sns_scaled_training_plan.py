#!/usr/bin/env python3
"""Standalone tests for scaled pre-SNS training plan helpers."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent.validate_pre_sns_scaled_training_plan import (  # noqa: E402
    APPROVAL_TEXT,
    OK_MARKER,
    load_config,
    validate_scaled_training_plan,
)
from scripts.training.prepare_pre_sns_scaled_train_config import build_scaled_config  # noqa: E402

EXAMPLE_CONFIG = REPO_ROOT / "configs" / "training" / "pre_sns_scaled_train.example.json"
TEMP_ROOT = Path.home() / ".codex" / "memories"


def temporary_root():
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=str(TEMP_ROOT))


def assert_pass(raw: dict) -> None:
    errors = validate_scaled_training_plan(raw)
    assert not errors, "\n".join(errors)


def assert_fail(raw: dict, expected: str) -> None:
    errors = validate_scaled_training_plan(raw)
    assert errors, "expected validation failure"
    joined = "\n".join(errors)
    assert expected in joined, joined


def mutate(raw: dict, key: str, value) -> dict:
    changed = copy.deepcopy(raw)
    changed[key] = value
    return changed


def make_base_config() -> dict:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_baseline_training",
        "execution_mode": "approved_local_training",
        "approved_training_run": True,
        "required_approval_text": APPROVAL_TEXT,
        "user_approval_text": APPROVAL_TEXT,
        "no_write_dry_run": False,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation": True,
        "seed": 23,
        "class_loss_weight": 1.0,
        "family_loss_weight": 1.0,
        "localization_loss_weight": 1.0,
    }


def make_approved(tmp: str, stamp: str = "20260521_120000") -> dict:
    manifest = Path(tmp) / "manifest.json"
    manifest.write_text('{"samples":[]}\n', encoding="utf-8")
    return build_scaled_config(
        make_base_config(),
        str(manifest),
        "scaled_pilot",
        8192,
        2,
        8,
        160,
        "cpu",
        stamp=stamp,
    )


def test_example_config_passes() -> None:
    assert_pass(load_config(EXAMPLE_CONFIG))


def test_approved_local_generated_config_passes() -> None:
    with temporary_root() as tmp:
        assert_pass(make_approved(tmp))


def test_excessive_max_samples_rejected() -> None:
    with temporary_root() as tmp:
        assert_fail(mutate(make_approved(tmp), "max_samples", 50001), "max_samples")


def test_excessive_epochs_rejected() -> None:
    with temporary_root() as tmp:
        assert_fail(mutate(make_approved(tmp), "epochs", 11), "epochs")


def test_excessive_batch_size_rejected() -> None:
    with temporary_root() as tmp:
        assert_fail(mutate(make_approved(tmp), "batch_size", 65), "batch_size")


def test_excessive_max_image_size_rejected() -> None:
    with temporary_root() as tmp:
        assert_fail(mutate(make_approved(tmp), "max_image_size", 513), "max_image_size")


def test_repo_outputs_checkpoints_rejected() -> None:
    with temporary_root() as tmp:
        raw = make_approved(tmp)
        raw["approved_run_root"] = str(REPO_ROOT / "outputs" / "scaled")
        assert_fail(raw, "approved_run_root")
        raw = make_approved(tmp)
        raw["approved_checkpoint_root"] = str(REPO_ROOT / "checkpoints" / "scaled")
        assert_fail(raw, "approved_checkpoint_root")


def test_no_download_false_rejected() -> None:
    with temporary_root() as tmp:
        assert_fail(mutate(make_approved(tmp), "no_download", False), "no_download")


def test_no_network_false_rejected() -> None:
    with temporary_root() as tmp:
        assert_fail(mutate(make_approved(tmp), "no_network", False), "no_network")


def test_no_sns_augmentation_false_rejected() -> None:
    with temporary_root() as tmp:
        assert_fail(mutate(make_approved(tmp), "no_sns_augmentation", False), "no_sns_augmentation")


def test_generator_produces_fresh_roots_and_does_not_train() -> None:
    with temporary_root() as tmp:
        first = make_approved(tmp, "20260521_120000")
        second = make_approved(tmp, "20260521_120001")
        assert first["approved_run_root"] != second["approved_run_root"]
        assert first["approved_checkpoint_root"] != second["approved_checkpoint_root"]
        assert first["scaled_training_policy"]["training_started"] is False
        assert first["scaled_training_policy"]["checkpoint_written"] is False
        assert first["marker"] == OK_MARKER
        assert not os.path.exists(first["approved_run_root"])
        assert not os.path.exists(first["approved_checkpoint_root"])


def test_generator_strips_stale_output_root_aliases() -> None:
    with temporary_root() as tmp:
        base = make_base_config()
        base["run_root"] = str(Path.home() / "cvf_runs" / "old")
        base["checkpoint_root"] = str(Path.home() / "cvf_checkpoints" / "old")
        base["approved_local_output_roots"] = [
            str(Path.home() / "cvf_runs"),
            str(Path.home() / "cvf_checkpoints"),
        ]
        manifest = Path(tmp) / "manifest.json"
        manifest.write_text('{"samples":[]}\n', encoding="utf-8")
        scaled = build_scaled_config(base, str(manifest), "scaled_pilot", 8192, 2, 8, 160, "cpu", stamp="20260521_120002")
        assert "run_root" not in scaled
        assert "checkpoint_root" not in scaled
        assert "approved_local_output_roots" not in scaled
        assert_pass(scaled)


def main() -> int:
    tests = [
        test_example_config_passes,
        test_approved_local_generated_config_passes,
        test_excessive_max_samples_rejected,
        test_excessive_epochs_rejected,
        test_excessive_batch_size_rejected,
        test_excessive_max_image_size_rejected,
        test_repo_outputs_checkpoints_rejected,
        test_no_download_false_rejected,
        test_no_network_false_rejected,
        test_no_sns_augmentation_false_rejected,
        test_generator_produces_fresh_roots_and_does_not_train,
        test_generator_strips_stale_output_root_aliases,
    ]
    for test in tests:
        test()
    print("PRE_SNS_SCALED_TRAINING_PLAN_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
