"""Tests for CF-Small local tiny training smoke config and runner.

Runnable directly:
    python3 tests/test_cf_small_tiny_train_smoke.py

Uses only the Python standard library for the harness. Runner execution is
skipped when torch or PIL is unavailable.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path as _Path


_REPO_ROOT = next(
    (p for p in _Path(__file__).resolve().parents if (p / "scripts" / "agent").is_dir()),
    None,
)
if _REPO_ROOT is not None:
    _AGENT_ROOT = str(_REPO_ROOT / "scripts" / "agent")
    _TRAINING_ROOT = str(_REPO_ROOT / "scripts" / "training")
    for path in (_AGENT_ROOT, _TRAINING_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)

from validate_cf_small_tiny_train_smoke_config import check_local_data_readiness_config_safety, load_config, validate_config
from run_cf_small_tiny_train_smoke import main as runner_main


def _approved_raw(root: str) -> dict:
    return {
        "schema_version": "0.1.0",
        "config_kind": "approved_local_smoke",
        "smoke_name": "cf_small_local_tiny_training_smoke",
        "dataset_name": "Community Forensics-Small",
        "stage": "non_sns_cf_small_tiny_training_smoke",
        "approved_real_data_access": True,
        "user_approval_text": "I_APPROVE_LOCAL_NON_SNS_TINY_TRAINING_SMOKE",
        "no_download": True,
        "no_network": True,
        "no_outputs": True,
        "no_checkpoints": True,
        "no_sns_augmentation": True,
        "no_sns_perturbation_eval": True,
        "approved_local_roots": [root],
        "tiny_limits": {
            "max_samples": 4,
            "max_steps": 2,
            "max_epochs": 1,
            "max_image_size": 8,
            "cpu_only": True,
        },
        "max_samples": 2,
        "max_steps": 1,
        "max_epochs": 1,
        "class_policy": {
            "task": "binary real-vs-synthetic classification",
            "labels": ["real", "synthetic"],
        },
        "recursive_scan": False,
        "sample_manifest": [
            {
                "sample_id": "real_001",
                "image_path": os.path.join(root, "real_001.png"),
                "label": "real",
            },
            {
                "sample_id": "synthetic_001",
                "image_path": os.path.join(root, "synthetic_001.png"),
                "label": "synthetic",
            },
        ],
        "validation_notes": ["authoritative local smoke policy"],
    }


def _write_json(path: str, raw: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(raw, fh)


def _assert_rejected(raw: dict, expected: str = "") -> None:
    try:
        validate_config(raw)
        assert False, f"Expected rejection for {expected or raw!r}"
    except ValueError as exc:
        if expected:
            assert expected in str(exc), f"Expected {expected!r} in {exc!r}"


def test_tracked_example_config_passes_validation():
    if _REPO_ROOT is None:
        return
    load_config(str(_REPO_ROOT / "configs" / "training" / "cf_small_tiny_train_smoke.example.json"))


def test_approved_local_smoke_config_passes_validation():
    with tempfile.TemporaryDirectory() as td:
        validate_config(_approved_raw(td))


def test_approval_text_required():
    with tempfile.TemporaryDirectory() as td:
        raw = _approved_raw(td)
        raw["user_approval_text"] = "yes"
        _assert_rejected(raw, "user_approval_text")


def test_url_image_path_rejected():
    with tempfile.TemporaryDirectory() as td:
        raw = _approved_raw(td)
        raw["sample_manifest"][0]["image_path"] = "https://example.invalid/x.png"
        _assert_rejected(raw, "URL")


def test_protected_repo_path_rejected():
    with tempfile.TemporaryDirectory() as td:
        raw = _approved_raw(td)
        raw["approved_local_roots"] = [str(_REPO_ROOT or _Path.cwd())]
        raw["sample_manifest"][0]["image_path"] = str((_REPO_ROOT or _Path.cwd()) / "data" / "x.png")
        _assert_rejected(raw, "protected")


def test_absolute_path_outside_approved_roots_rejected():
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as other:
        raw = _approved_raw(td)
        raw["sample_manifest"][0]["image_path"] = os.path.join(other, "x.png")
        _assert_rejected(raw, "approved_local_roots")


def test_directory_only_sample_path_rejected():
    with tempfile.TemporaryDirectory() as td:
        raw = _approved_raw(td)
        raw["sample_manifest"][0]["image_path"] = td
        _assert_rejected(raw, "approved root directory")


def test_recursive_scan_flag_rejected():
    with tempfile.TemporaryDirectory() as td:
        raw = _approved_raw(td)
        raw["recursive_scan"] = True
        _assert_rejected(raw, "recursive")


def test_tiny_limit_caps_rejected():
    for key in ("max_samples", "max_steps", "max_epochs"):
        with tempfile.TemporaryDirectory() as td:
            raw = _approved_raw(td)
            raw[key] = raw["tiny_limits"][key] + 1
            _assert_rejected(raw, key)


def test_missing_class_rejected():
    with tempfile.TemporaryDirectory() as td:
        raw = _approved_raw(td)
        raw["sample_manifest"] = [raw["sample_manifest"][0]]
        _assert_rejected(raw, "both real and synthetic")
    with tempfile.TemporaryDirectory() as td:
        raw = _approved_raw(td)
        raw["sample_manifest"] = [raw["sample_manifest"][1]]
        _assert_rejected(raw, "both real and synthetic")


def test_invalid_label_rejected():
    with tempfile.TemporaryDirectory() as td:
        raw = _approved_raw(td)
        raw["sample_manifest"][0]["label"] = "tampered"
        _assert_rejected(raw, "real or synthetic")


def test_guardrail_flags_rejected():
    for flag in (
        "no_download",
        "no_network",
        "no_outputs",
        "no_checkpoints",
        "no_sns_augmentation",
        "no_sns_perturbation_eval",
    ):
        with tempfile.TemporaryDirectory() as td:
            raw = _approved_raw(td)
            raw[flag] = False
            _assert_rejected(raw, flag)


def test_secret_like_keys_and_values_rejected():
    with tempfile.TemporaryDirectory() as td:
        raw = _approved_raw(td)
        raw["api_key"] = "placeholder"
        _assert_rejected(raw, "secret-like key")
    with tempfile.TemporaryDirectory() as td:
        raw = _approved_raw(td)
        raw["validation_notes"] = ["bearer value"]
        _assert_rejected(raw, "secret-like values")


def test_auth_substring_prose_allowed():
    with tempfile.TemporaryDirectory() as td:
        raw = _approved_raw(td)
        raw["validation_notes"] = [
            "authoritative smoke note",
            "authentication policy is prose only",
        ]
        validate_config(raw)


def _deps_available() -> bool:
    try:
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        return False
    return True


def _write_tiny_images(root: str) -> None:
    from PIL import Image

    Image.new("RGB", (8, 8), color=(20, 20, 20)).save(os.path.join(root, "real_001.png"))
    Image.new("RGB", (8, 8), color=(220, 220, 220)).save(os.path.join(root, "synthetic_001.png"))


def test_training_runner_prints_marker_and_writes_no_artifacts():
    if not _deps_available():
        print("SKIP: torch or PIL unavailable; runner execution skipped.")
        return
    with tempfile.TemporaryDirectory() as td:
        _write_tiny_images(td)
        raw = _approved_raw(td)
        config_path = os.path.join(td, "cf_small_smoke.local.json")
        _write_json(config_path, raw)
        before = sorted(os.listdir(td))
        old_argv = sys.argv[:]
        stdout = io.StringIO()
        try:
            sys.argv = ["run_cf_small_tiny_train_smoke.py", config_path]
            with contextlib.redirect_stdout(stdout):
                runner_main()
        finally:
            sys.argv = old_argv
        output = stdout.getvalue()
        assert "CF_SMALL_TINY_TRAIN_SMOKE_OK" in output
        summary = json.loads(output)
        assert summary["marker"] == "CF_SMALL_TINY_TRAIN_SMOKE_OK"
        assert summary["samples_seen"] == 2
        assert summary["finite_loss"] is True
        assert summary["no_outputs"] is True
        assert summary["no_checkpoints"] is True
        after = sorted(os.listdir(td))
        assert before == after


if __name__ == "__main__":
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for name, test in tests:
        try:
            test()
            print(f"  PASS: {name}")
        except Exception as exc:
            failures.append(name)
            print(f"  FAIL: {name}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed.")
    if failures:
        print(f"Failed: {failures}")
        raise SystemExit(1)
    print("All tests passed.")
