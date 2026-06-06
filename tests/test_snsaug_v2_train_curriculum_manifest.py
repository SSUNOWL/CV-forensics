#!/usr/bin/env python3
"""Plain Python tests for SNSAug V2 train curriculum manifest builder."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TMP_PARENT = Path("/home/rlatjswo/.codex/memories")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2_train_curriculum import (  # noqa: E402
    CURRICULUM_SCHEDULE,
    MARKER,
    PROFILE_GROUPS,
    SEVERITY_SCHEDULE,
    SNSAugV2TrainCurriculumError,
    build_snsaug_v2_train_curriculum_manifest,
    validate_curriculum_schedule,
    validate_snsaug_v2_train_curriculum_config,
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


def write_train_fixture(root: Path) -> Path:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "base_id": "real_train",
            "split": "train",
            "image_path": str(inputs / "real_train.png"),
            "content_label": "real",
            "source_dataset": "fixture",
        },
        {
            "base_id": "synth_train",
            "split": "train",
            "image_path": str(inputs / "synth_train.png"),
            "content_label": "full_synthetic",
            "family_label": "GAN",
            "source_dataset": "fixture",
        },
        {
            "base_id": "tampered_train",
            "split": "train",
            "image_path": str(inputs / "tampered_train.png"),
            "tamper_mask_path": str(inputs / "tampered_train_mask.png"),
            "content_label": "tampered",
            "source_dataset": "fixture",
        },
        {
            "base_id": "val_rejected",
            "split": "val",
            "image_path": str(inputs / "val_rejected.png"),
            "content_label": "tampered",
            "source_dataset": "fixture",
        },
    ]
    manifest_path = inputs / "train_manifest.json"
    manifest_path.write_text(json.dumps({"samples": rows}), encoding="utf-8")
    return manifest_path


def safe_config(root: Path, manifest_path: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_train_curriculum_manifest",
        "execution_mode": "approved_local_snsaug_v2_train_curriculum_manifest",
        "train_manifest_path": str(manifest_path),
        "approved_input_roots": [str(root / "inputs")],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "curriculum_manifest"),
        "split_policy": "train_only",
        "profile_groups": PROFILE_GROUPS,
        "curriculum_schedule": CURRICULUM_SCHEDULE,
        "severity_schedule": SEVERITY_SCHEDULE,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_train_only_split_enforcement_and_no_validation_leakage() -> None:
    root = temp_root("cvf_0059_train_only_")
    manifest_path = write_train_fixture(root)
    summary = build_snsaug_v2_train_curriculum_manifest(safe_config(root, manifest_path))
    rows = load_jsonl(Path(summary["output_paths"]["train_curriculum_manifest"]))
    assert_equal(len(rows), 3, "only train rows accepted")
    assert_true(all(row["split"] == "train" for row in rows), "all output rows are train")
    assert_true(all(row["base_id"] != "val_rejected" for row in rows), "validation row rejected")
    balance = json.loads(Path(summary["output_paths"]["class_balance_summary"]).read_text(encoding="utf-8"))
    assert_equal(balance["input_summary"]["rejected_counts"]["non_train_split:val"], 1, "val rejection counted")


def test_profile_weights_sum_to_one_per_phase_and_parse() -> None:
    assert_equal(validate_curriculum_schedule(), [], "default curriculum valid")
    for weights in CURRICULUM_SCHEDULE.values():
        assert_equal(set(weights), set(PROFILE_GROUPS), "phase contains exact groups")
        assert_true(abs(sum(weights.values()) - 1.0) < 1e-9, "phase weights sum to one")


def test_manifest_schema_and_family_loss_mask() -> None:
    root = temp_root("cvf_0059_schema_")
    manifest_path = write_train_fixture(root)
    summary = build_snsaug_v2_train_curriculum_manifest(safe_config(root, manifest_path))
    rows = load_jsonl(Path(summary["output_paths"]["train_curriculum_manifest"]))
    required = {
        "base_id",
        "source_dataset",
        "split",
        "image_path",
        "content_label",
        "family_loss_mask",
        "allowed_profile_groups",
        "profile_sampling_weights_by_phase",
        "severity_schedule",
        "sampling_weight",
        "label_preserved",
        "training_notes",
    }
    for row in rows:
        assert_true(required.issubset(row), f"schema complete for {row['base_id']}")
        assert_true(row["label_preserved"] is True, "label preserved true")
    by_id = {row["base_id"]: row for row in rows}
    assert_equal(by_id["synth_train"]["content_label"], "synthetic", "synthetic label normalized")
    assert_equal(by_id["synth_train"]["family_loss_mask"], 1, "family present contributes")
    assert_equal(by_id["tampered_train"]["family_loss_mask"], 0, "missing family masked")
    assert_true("tamper_mask_path" in by_id["tampered_train"], "tamper mask retained when available")


def test_class_balance_summary_and_sampling_weights() -> None:
    root = temp_root("cvf_0059_balance_")
    manifest_path = write_train_fixture(root)
    summary = build_snsaug_v2_train_curriculum_manifest(safe_config(root, manifest_path))
    balance = json.loads(Path(summary["output_paths"]["class_balance_summary"]).read_text(encoding="utf-8"))
    assert_equal(balance["record_total"], 3, "record total")
    assert_equal(balance["counts_by_content_label"], {"real": 1, "synthetic": 1, "tampered": 1}, "class counts")
    rows = load_jsonl(Path(summary["output_paths"]["train_curriculum_manifest"]))
    assert_true(all(float(row["sampling_weight"]) == 1.0 for row in rows), "balanced fixture weights are one")


def test_output_root_guardrails_and_config_rejections() -> None:
    root = temp_root("cvf_0059_guardrails_")
    manifest_path = write_train_fixture(root)
    cfg = safe_config(root, manifest_path)
    assert_equal(validate_snsaug_v2_train_curriculum_config(cfg, require_exists=True), [], "safe config passes")
    repo_output = dict(cfg)
    repo_output["output_root"] = str(REPO_ROOT / "curriculum_manifest")
    assert_true(validate_snsaug_v2_train_curriculum_config(repo_output, require_exists=False), "repo output rejected")
    non_train_policy = dict(cfg)
    non_train_policy["split_policy"] = "train_val"
    assert_true(validate_snsaug_v2_train_curriculum_config(non_train_policy, require_exists=False), "non-train policy rejected")
    missing_flag = dict(cfg)
    missing_flag["no_finetune"] = False
    assert_true(validate_snsaug_v2_train_curriculum_config(missing_flag, require_exists=False), "missing safety flag rejected")
    eval_manifest = dict(cfg)
    eval_manifest["train_manifest_path"] = str(root / "inputs" / "snsaug_v2_0058e_forced_localization_oracle_gate_records.jsonl")
    assert_true(validate_snsaug_v2_train_curriculum_config(eval_manifest, require_exists=False), "eval manifest path rejected")


def test_required_outputs_and_guardrails_are_written() -> None:
    root = temp_root("cvf_0059_outputs_")
    manifest_path = write_train_fixture(root)
    summary = build_snsaug_v2_train_curriculum_manifest(safe_config(root, manifest_path))
    expected = {
        "train_curriculum_manifest",
        "curriculum_schedule",
        "profile_sampling_weights",
        "class_balance_summary",
        "training_guardrails",
        "training_intent_and_references",
        "artifact_manifest",
    }
    assert_equal(set(summary["output_paths"]), expected, "all required outputs listed")
    for path in summary["output_paths"].values():
        assert_true(Path(path).exists(), f"{path} exists")
    guardrails = json.loads(Path(summary["output_paths"]["training_guardrails"]).read_text(encoding="utf-8"))
    for flag in (
        "no_training",
        "no_finetune",
        "no_network",
        "no_download",
        "train_only",
        "output_root_outside_repository",
        "validation_samples_rejected",
        "evaluation_outputs_rejected_as_training_images",
    ):
        assert_true(guardrails[flag] is True, f"{flag} guardrail true")
    intent = Path(summary["output_paths"]["training_intent_and_references"]).read_text(encoding="utf-8")
    assert_true("p_tampered collapse" in intent, "intent explains activation collapse")
    assert_true("Community Forensics CVPR 2025" in intent, "intent includes references")


def test_empty_after_guardrails_fails_before_writing_manifest() -> None:
    root = temp_root("cvf_0059_empty_")
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    manifest_path = inputs / "train_manifest.json"
    manifest_path.write_text(
        json.dumps({"samples": [{"base_id": "val_only", "split": "val", "image_path": str(inputs / "val.png"), "content_label": "real"}]}),
        encoding="utf-8",
    )
    try:
        build_snsaug_v2_train_curriculum_manifest(safe_config(root, manifest_path))
    except SNSAugV2TrainCurriculumError as exc:
        assert_true("no valid train records" in str(exc), "empty manifest error is explicit")
        return
    raise AssertionError("empty manifest should fail")


def main() -> int:
    tests = [
        test_train_only_split_enforcement_and_no_validation_leakage,
        test_profile_weights_sum_to_one_per_phase_and_parse,
        test_manifest_schema_and_family_loss_mask,
        test_class_balance_summary_and_sampling_weights,
        test_output_root_guardrails_and_config_rejections,
        test_required_outputs_and_guardrails_are_written,
        test_empty_after_guardrails_fails_before_writing_manifest,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
