#!/usr/bin/env python3
"""Plain Python tests for snsaug v2 finetune helpers."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TMP_PARENT = Path("/home/rlatjswo/.codex/memories")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2_finetune import (  # noqa: E402
    APPROVAL_TEXT,
    MARKER,
    build_validation_plan,
    compute_total_loss,
    curriculum_stage,
    masked_family_loss,
    run_snsaug_v2_finetune,
    validate_snsaug_v2_finetune_config,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def temp_root(prefix: str) -> Path:
    TMP_PARENT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(TMP_PARENT)))


def write_fixture_inputs(root: Path) -> tuple[Path, Path]:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    manifest_rows = [
        {"base_id": "train_ok", "split": "train", "image_path": str(inputs / "a.png"), "content_label": "tampered"},
        {"base_id": "val_bad", "split": "val", "image_path": str(inputs / "b.png"), "content_label": "tampered"},
    ]
    train_only_rows = [manifest_rows[0]]
    bundle_path = inputs / "bundle.json"
    bundle_path.write_text(json.dumps({"marker": "PRE_SNS_CURRENT_BEST_MODEL_BUNDLE_OK"}), encoding="utf-8")
    all_path = inputs / "all_manifest.jsonl"
    train_path = inputs / "train_manifest.jsonl"
    with open(all_path, "w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row) + "\n")
    with open(train_path, "w", encoding="utf-8") as handle:
        for row in train_only_rows:
            handle.write(json.dumps(row) + "\n")
    return all_path, train_path, bundle_path


def safe_config(root: Path, manifest_path: Path, bundle_path: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_finetune",
        "execution_mode": "approved_local_snsaug_v2_finetune",
        "training_manifest_path": str(manifest_path),
        "best_bundle_path": str(bundle_path),
        "approved_input_roots": [str(root / "inputs")],
        "approved_checkpoint_roots": [str(root / "inputs")],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "finetune"),
        "approval_text": APPROVAL_TEXT,
        "seed": 54,
        "epochs": 12,
        "batch_size": 2,
        "device": "cpu",
        "no_network": True,
        "no_download": True,
    }


def test_config_validator_rejects_val_test_training_samples() -> None:
    root = temp_root("cvf_snsaug_ft_validate_")
    all_path, train_path, bundle_path = write_fixture_inputs(root)
    cfg = safe_config(root, train_path, bundle_path)
    assert_equal(validate_snsaug_v2_finetune_config(cfg, require_exists=True), [], "train-only config")
    bad = safe_config(root, all_path, bundle_path)
    errors = validate_snsaug_v2_finetune_config(bad, require_exists=True)
    assert_true(bool(errors), "val/test rejected")


def test_loss_masking_with_ignore_mask() -> None:
    losses = compute_total_loss(
        class_loss=0.5,
        mask_pred=[0.9, 0.1],
        tamper_mask=[1.0, 1.0],
        ignore_mask=[0.0, 1.0],
        lambda_mask=1.0,
    )
    assert_true(float(losses["mask_loss"]) < 0.2, "ignore mask applied")


def test_family_loss_mask_out() -> None:
    value = masked_family_loss(
        [[0.8, 0.2], [0.2, 0.8]],
        [0, 1],
        [1.0, 0.0],
        reduction="mean",
    )
    assert_true(float(value) < 0.3, "family loss masked")


def test_curriculum_schedule() -> None:
    assert_equal(curriculum_stage(2)["severity"], "light", "early severity")
    assert_equal(curriculum_stage(5)["severity"], "medium", "mid severity")
    assert_equal(curriculum_stage(9)["severity"], "strong", "late severity")


def test_output_schema() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_snsaug_ft_run_")
    _all_path, train_path, bundle_path = write_fixture_inputs(root)
    summary = run_snsaug_v2_finetune(safe_config(root, train_path, bundle_path))
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "no repo writes")
    assert_equal(summary["marker"], MARKER, "summary marker")
    assert_true("validation_plan" in summary, "validation plan present")
    assert_true(Path(summary["output_paths"]["artifact_manifest"]).is_file(), "artifact manifest written")
    plan = build_validation_plan(safe_config(root, train_path, bundle_path))
    assert_true("accuracy_3way" in plan["reported_metrics"], "reported metric included")


def main() -> int:
    tests = [
        test_config_validator_rejects_val_test_training_samples,
        test_loss_masking_with_ignore_mask,
        test_family_loss_mask_out,
        test_curriculum_schedule,
        test_output_schema,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
