#!/usr/bin/env python3
"""Standalone tests for pre-SNS single-image inference reports."""

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
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_inference_report import (  # noqa: E402
    APPROVAL_TEXT,
    MARKER,
    load_report_config,
    localization_summary,
    run_single_image_report,
    validate_report_config,
)
from cv_forensics.pre_sns_integrated_model import CLASS_LABELS, FAMILY_SMOKE_LABELS, build_tiny_integrated_model  # noqa: E402
from scripts.agent.validate_pre_sns_inference_report import parse_report_json, validate_report  # noqa: E402


EXAMPLE_CONFIG = REPO_ROOT / "configs" / "inference" / "pre_sns_report.example.json"
TEMP_ROOT = Path.home() / ".codex" / "memories"


def temporary_root():
    return tempfile.TemporaryDirectory(dir=str(TEMP_ROOT))


def assert_pass(raw: dict) -> None:
    errors = validate_report_config(raw)
    assert not errors, "\n".join(errors)


def assert_fail(raw: dict, expected: str) -> None:
    errors = validate_report_config(raw)
    assert errors, "expected validation failure"
    joined = "\n".join(errors)
    assert expected in joined, joined


def mutate(raw: dict, key: str, value) -> dict:
    changed = copy.deepcopy(raw)
    changed[key] = value
    return changed


def make_approved(root: str, image_path: str, checkpoint_path: str, report_root: str, write_report: bool = False) -> dict:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_single_image_report",
        "execution_mode": "approved_local_single_image_report",
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_checkpoint_writes": True,
        "no_sns_augmentation": True,
        "checkpoint_path": checkpoint_path,
        "image_path": image_path,
        "report_root": report_root,
        "write_report": write_report,
        "device": "cpu",
        "cuda_device_index": 0,
        "max_image_size": 8,
        "threshold_tau": 0.5,
        "class_labels": list(CLASS_LABELS),
        "family_labels": list(FAMILY_SMOKE_LABELS),
        "required_approval_text": APPROVAL_TEXT,
        "user_approval_text": APPROVAL_TEXT,
        "approved_local_roots": [root],
        "result_scope": "single-image pre-SNS report test fixture",
        "ordinary_prose": "This authoritative note is documentation only.",
    }


def make_valid_report() -> dict:
    family_conf = {label: 0.0 for label in FAMILY_SMOKE_LABELS}
    family_conf["Real-or-N/A"] = 1.0
    return {
        "marker": MARKER,
        "class": "real",
        "class_conf": {"real": 0.8, "full_synthetic": 0.1, "tampered": 0.1},
        "family": "Real-or-N/A",
        "family_conf": family_conf,
        "tampered_score": 0.1,
        "localization_head": "not_applicable",
        "mask_area_pct": None,
        "reason": "실제 이미지로 분류됨.",
        "latency_ms": 1.0,
        "fps_estimate": 1000.0,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }


def create_fake_image_and_checkpoint(root: str) -> tuple[str, str]:
    try:
        import torch
        from PIL import Image
    except Exception:
        return "", ""
    image_path = os.path.join(root, "single.png")
    checkpoint_path = os.path.join(root, "tiny.pt")
    Image.new("RGB", (8, 8), color=(120, 80, 40)).save(image_path)
    model = build_tiny_integrated_model(torch, 3 * 8 * 8, 8 * 8)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_labels": list(CLASS_LABELS),
            "family_labels": list(FAMILY_SMOKE_LABELS),
            "image_size": 8,
            "marker": "TEST_TINY_CHECKPOINT",
        },
        checkpoint_path,
    )
    return image_path, checkpoint_path


def test_example_config_validates() -> None:
    assert_pass(load_report_config(EXAMPLE_CONFIG))


def test_approved_local_config_kind_alias_validates() -> None:
    with temporary_root() as tmp:
        image_path = os.path.join(tmp, "image.png")
        checkpoint_path = os.path.join(tmp, "tiny.pt")
        Path(image_path).write_bytes(b"fake")
        Path(checkpoint_path).write_bytes(b"fake")
        raw = make_approved(tmp, image_path, checkpoint_path, os.path.join(tmp, "reports"))
        raw["config_kind"] = "approved_local_pre_sns_single_image_report"
        assert_pass(raw)


def test_approved_local_paths_under_approved_roots_validate() -> None:
    with temporary_root() as tmp:
        image_root = os.path.join(tmp, "images")
        checkpoint_root = os.path.join(tmp, "trained")
        report_root_base = os.path.join(tmp, "reports")
        os.mkdir(image_root)
        os.mkdir(checkpoint_root)
        os.mkdir(report_root_base)
        image_path = os.path.join(image_root, "image.png")
        checkpoint_path = os.path.join(checkpoint_root, "tiny.pt")
        Path(image_path).write_bytes(b"fake")
        Path(checkpoint_path).write_bytes(b"fake")
        raw = make_approved(tmp, image_path, checkpoint_path, os.path.join(report_root_base, "single"))
        raw["approved_image_roots"] = [image_root]
        raw["approved_checkpoint_roots"] = [checkpoint_root]
        raw["approved_report_roots"] = [report_root_base]
        del raw["approved_local_roots"]
        assert_pass(raw)


def test_approved_local_image_outside_approved_image_roots_rejected() -> None:
    with temporary_root() as tmp:
        image_root = os.path.join(tmp, "images")
        other_root = os.path.join(tmp, "other-images")
        os.mkdir(image_root)
        os.mkdir(other_root)
        image_path = os.path.join(other_root, "image.png")
        checkpoint_path = os.path.join(tmp, "tiny.pt")
        Path(image_path).write_bytes(b"fake")
        Path(checkpoint_path).write_bytes(b"fake")
        raw = make_approved(tmp, image_path, checkpoint_path, os.path.join(tmp, "reports"))
        raw["approved_image_roots"] = [image_root]
        assert_fail(raw, "image_path must be under approved roots")


def test_approved_local_checkpoint_outside_approved_checkpoint_roots_rejected() -> None:
    with temporary_root() as tmp:
        checkpoint_root = os.path.join(tmp, "trained")
        other_root = os.path.join(tmp, "other-trained")
        os.mkdir(checkpoint_root)
        os.mkdir(other_root)
        image_path = os.path.join(tmp, "image.png")
        checkpoint_path = os.path.join(other_root, "tiny.pt")
        Path(image_path).write_bytes(b"fake")
        Path(checkpoint_path).write_bytes(b"fake")
        raw = make_approved(tmp, image_path, checkpoint_path, os.path.join(tmp, "reports"))
        raw["approved_checkpoint_roots"] = [checkpoint_root]
        assert_fail(raw, "checkpoint_path must be under approved roots")


def test_approved_local_report_root_inside_repo_outputs_rejected() -> None:
    with temporary_root() as tmp:
        image_path = os.path.join(tmp, "image.png")
        checkpoint_path = os.path.join(tmp, "tiny.pt")
        Path(image_path).write_bytes(b"fake")
        Path(checkpoint_path).write_bytes(b"fake")
        raw = make_approved(tmp, image_path, checkpoint_path, str(REPO_ROOT / "outputs" / "report"))
        raw["approved_report_roots"] = [str(REPO_ROOT / "outputs")]
        assert_fail(raw, "report_root")


def test_approved_local_report_root_inside_repo_checkpoints_rejected() -> None:
    with temporary_root() as tmp:
        image_path = os.path.join(tmp, "image.png")
        checkpoint_path = os.path.join(tmp, "tiny.pt")
        Path(image_path).write_bytes(b"fake")
        Path(checkpoint_path).write_bytes(b"fake")
        raw = make_approved(tmp, image_path, checkpoint_path, str(REPO_ROOT / "checkpoints" / "report"))
        raw["approved_report_roots"] = [str(REPO_ROOT / "checkpoints")]
        assert_fail(raw, "report_root")


def test_example_symbolic_rejects_real_absolute_paths() -> None:
    raw = load_report_config(EXAMPLE_CONFIG)
    raw["image_path"] = "/tmp/pre_sns_example_image.png"
    assert_fail(raw, "absolute path rejected")


def test_unsafe_url_paths_rejected() -> None:
    raw = load_report_config(EXAMPLE_CONFIG)
    assert_fail(mutate(raw, "image_path", "https://example.invalid/image.png"), "URL")


def test_missing_approval_text_rejected() -> None:
    with temporary_root() as tmp:
        image_path = os.path.join(tmp, "image.png")
        checkpoint_path = os.path.join(tmp, "tiny.pt")
        Path(image_path).write_bytes(b"fake")
        Path(checkpoint_path).write_bytes(b"fake")
        raw = make_approved(tmp, image_path, checkpoint_path, os.path.join(tmp, "reports"))
        raw["user_approval_text"] = ""
        assert_fail(raw, "approval phrase")


def test_protected_paths_rejected() -> None:
    raw = load_report_config(EXAMPLE_CONFIG)
    assert_fail(mutate(raw, "image_path", "safe/data/image.png"), "protected")


def test_repo_outputs_checkpoints_report_root_rejected() -> None:
    with temporary_root() as tmp:
        image_path = os.path.join(tmp, "image.png")
        checkpoint_path = os.path.join(tmp, "tiny.pt")
        Path(image_path).write_bytes(b"fake")
        Path(checkpoint_path).write_bytes(b"fake")
        raw = make_approved(tmp, image_path, checkpoint_path, str(REPO_ROOT / "outputs" / "report"))
        assert_fail(raw, "report_root")
        raw = make_approved(tmp, image_path, checkpoint_path, str(REPO_ROOT / "checkpoints" / "report"))
        assert_fail(raw, "report_root")


def test_report_json_with_required_fields_accepted() -> None:
    errors = validate_report({}, make_valid_report())
    assert not errors, "\n".join(errors)


def test_missing_class_conf_rejected() -> None:
    report = make_valid_report()
    del report["class_conf"]
    assert "class_conf" in "\n".join(validate_report({}, report))


def test_missing_family_conf_rejected() -> None:
    report = make_valid_report()
    del report["family_conf"]
    assert "family_conf" in "\n".join(validate_report({}, report))


def test_missing_reason_rejected() -> None:
    report = make_valid_report()
    del report["reason"]
    assert "reason" in "\n".join(validate_report({}, report))


def test_threshold_tau_activates_localization() -> None:
    import torch

    probs = torch.ones(4)
    state, area = localization_summary(0.7, 0.5, probs, torch)
    assert state == "activated"
    assert area == 100.0


def test_threshold_tau_skips_localization() -> None:
    import torch

    probs = torch.ones(4)
    state, area = localization_summary(0.3, 0.5, probs, torch)
    assert state == "skipped_below_threshold"
    assert area is None


def test_parser_handles_leading_log_text() -> None:
    with temporary_root() as tmp:
        path = Path(tmp) / "report.log"
        payload = make_valid_report()
        path.write_text("log before json\n" + json.dumps(payload), encoding="utf-8")
        assert parse_report_json(path) == payload


def test_write_report_false_writes_nothing() -> None:
    with temporary_root() as tmp:
        image_path, checkpoint_path = create_fake_image_and_checkpoint(tmp)
        if not image_path:
            print("SKIP: torch or PIL unavailable; runtime inference write test skipped.")
            return
        report_root = os.path.join(tmp, "report-root")
        raw = make_approved(tmp, image_path, checkpoint_path, report_root, write_report=False)
        before_exists = os.path.exists(report_root)
        report = run_single_image_report(raw)
        after_exists = os.path.exists(report_root)
        assert report["marker"] == MARKER
        assert before_exists is False
        assert after_exists is False
        assert "report_path" not in report


def test_ordinary_authoritative_prose_not_rejected() -> None:
    with temporary_root() as tmp:
        image_path = os.path.join(tmp, "image.png")
        checkpoint_path = os.path.join(tmp, "tiny.pt")
        Path(image_path).write_bytes(b"fake")
        Path(checkpoint_path).write_bytes(b"fake")
        raw = make_approved(tmp, image_path, checkpoint_path, os.path.join(tmp, "reports"))
        raw["ordinary_prose"] = "An authoritative reviewer note is not a credential."
        errors = validate_report_config(raw)
        assert not errors, "\n".join(errors)


def main() -> int:
    tests = [
        test_example_config_validates,
        test_approved_local_config_kind_alias_validates,
        test_approved_local_paths_under_approved_roots_validate,
        test_approved_local_image_outside_approved_image_roots_rejected,
        test_approved_local_checkpoint_outside_approved_checkpoint_roots_rejected,
        test_approved_local_report_root_inside_repo_outputs_rejected,
        test_approved_local_report_root_inside_repo_checkpoints_rejected,
        test_example_symbolic_rejects_real_absolute_paths,
        test_unsafe_url_paths_rejected,
        test_missing_approval_text_rejected,
        test_protected_paths_rejected,
        test_repo_outputs_checkpoints_report_root_rejected,
        test_report_json_with_required_fields_accepted,
        test_missing_class_conf_rejected,
        test_missing_family_conf_rejected,
        test_missing_reason_rejected,
        test_threshold_tau_activates_localization,
        test_threshold_tau_skips_localization,
        test_parser_handles_leading_log_text,
        test_write_report_false_writes_nothing,
        test_ordinary_authoritative_prose_not_rejected,
    ]
    for test in tests:
        test()
    print("PRE_SNS_INFERENCE_REPORT_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
