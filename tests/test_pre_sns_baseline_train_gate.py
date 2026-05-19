#!/usr/bin/env python3
"""Standalone tests for the guarded pre-SNS baseline training entrypoint."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.agent.validate_pre_sns_baseline_train_config import (  # noqa: E402
    APPROVAL_TEXT,
    load_config,
    validate_config,
)
from scripts.training.train_pre_sns_baseline import ENTRYPOINT_MARKER, main as runner_main  # noqa: E402


EXAMPLE_CONFIG = REPO_ROOT / "configs" / "training" / "pre_sns_baseline_train.example.json"
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


def mutate(raw: dict, mutator) -> dict:
    changed = copy.deepcopy(raw)
    mutator(changed)
    return changed


def make_manifest(root: str) -> dict:
    return {
        "schema_version": "1.0",
        "manifest_kind": "pre_sns_unified_dataset_manifest",
        "samples": [
            {
                "source_dataset": "community_forensics_small",
                "sample_id": "cf_real_001",
                "image_path": os.path.join(root, "cf_real_001.png"),
                "class_label": "real",
                "family_label": "Real-or-N/A",
                "architecture": "camera_original",
                "model_name": "original",
                "subset": "tiny",
                "tasks_available": {"class": True, "family": True, "localization": False},
                "loss_routing": {"class_loss": True, "family_loss": True, "localization_loss": False}
            },
            {
                "source_dataset": "sid_set",
                "sample_id": "sid_tampered_001",
                "image_path": os.path.join(root, "sid_tampered_001.png"),
                "mask_path": os.path.join(root, "sid_tampered_001_mask.png"),
                "class_label": "tampered",
                "label_id": 2,
                "tasks_available": {"class": True, "family": False, "localization": True},
                "loss_routing": {"class_loss": True, "family_loss": False, "localization_loss": True}
            }
        ]
    }


def make_approved(root: str, manifest_path: str) -> dict:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_baseline_training",
        "train_name": "temporary_pre_sns_baseline_training_gate",
        "approved_training_run": True,
        "user_approval_text": APPROVAL_TEXT,
        "no_write_dry_run": True,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation": True,
        "approved_local_roots": [root],
        "unified_manifest_path": manifest_path,
        "approved_run_root": os.path.join(root, "run-artifacts"),
        "approved_checkpoint_root": os.path.join(root, "weight-artifacts"),
        "device": "cpu",
        "batch_size": 2,
        "epochs": 1,
        "max_samples": 4,
        "seed": 23,
        "class_loss_weight": 1.0,
        "family_loss_weight": 1.0,
        "localization_loss_weight": 1.0,
        "recursive_scan": False,
        "ordinary_prose": "This authoritative authentication note is documentation only."
    }


def write_json(path: str, raw: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(raw, handle)


def test_example_config_passes() -> None:
    assert_pass(load_config(EXAMPLE_CONFIG))


def test_no_approval_exits_before_manifest_or_writes() -> None:
    raw = load_config(EXAMPLE_CONFIG)
    with temporary_local_root() as tmp:
        config_path = os.path.join(tmp, "example.json")
        write_json(config_path, raw)
        before = sorted(os.listdir(tmp))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = runner_main([config_path])
        after = sorted(os.listdir(tmp))
        assert exit_code == 0, stdout.getvalue()
        summary = json.loads(stdout.getvalue())
        assert summary["marker"] == ENTRYPOINT_MARKER
        assert summary["training_started"] is False
        assert "manifest loading" in summary["reason"]
        assert before == after


def test_validator_rejections() -> None:
    with temporary_local_root() as tmp:
        manifest_path = os.path.join(tmp, "manifest.json")
        base = make_approved(tmp, manifest_path)
        cases = [
            (lambda raw: raw.update({"user_approval_text": "bad"}), "user_approval_text"),
            (lambda raw: raw.update({"approved_training_run": False}), "approved_training_run must be true"),
            (lambda raw: raw.update({"unified_manifest_path": "https://example.invalid/manifest.json"}), "URL"),
            (lambda raw: raw.update({"approved_run_root": str(REPO_ROOT / "outputs" / "run")}), "protected"),
            (lambda raw: raw.update({"approved_checkpoint_root": str(REPO_ROOT / "checkpoints" / "run")}), "protected"),
            (lambda raw: raw.update({"recursive_scan": True}), "recursive scan"),
            (lambda raw: raw.update({"approved_run_root": os.path.join(tmp, "..", "run-artifacts")}), "path traversal"),
            (lambda raw: raw.update({"no_network": False}), "no_network must be true"),
            (lambda raw: raw.update({"no_download": False}), "no_download must be true"),
        ]
        for mutator, expected in cases:
            assert_fail(mutate(base, mutator), expected)


def test_no_write_dry_run_prints_marker_without_artifacts() -> None:
    try:
        import torch  # noqa: F401
        import PIL  # noqa: F401
    except Exception:
        print("SKIP: torch or PIL unavailable; model execution dependency check skipped.")
    with temporary_local_root() as tmp:
        manifest_path = os.path.join(tmp, "manifest.json")
        config_path = os.path.join(tmp, "train.local.json")
        write_json(manifest_path, make_manifest(tmp))
        write_json(config_path, make_approved(tmp, manifest_path))
        before = sorted(os.listdir(tmp))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = runner_main([config_path])
        after = sorted(os.listdir(tmp))
        assert exit_code == 0, stdout.getvalue()
        summary = json.loads(stdout.getvalue())
        assert summary["marker"] == ENTRYPOINT_MARKER
        assert summary["training_started"] is False
        assert summary["no_write_dry_run"] is True
        assert summary["manifest_sample_count"] == 2
        assert summary["image_paths_listed"] == 2
        assert summary["mask_paths_listed"] == 1
        assert before == after
        assert not os.path.exists(os.path.join(tmp, "run-artifacts"))
        assert not os.path.exists(os.path.join(tmp, "weight-artifacts"))


def main() -> int:
    tests = [
        test_example_config_passes,
        test_no_approval_exits_before_manifest_or_writes,
        test_validator_rejections,
        test_no_write_dry_run_prints_marker_without_artifacts,
    ]
    for test in tests:
        test()
    print("PRE_SNS_BASELINE_TRAIN_GATE_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
