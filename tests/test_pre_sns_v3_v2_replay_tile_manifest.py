#!/usr/bin/env python3
"""Plain Python tests for pre-SNS v2 replay tile-manifest building."""

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

from cv_forensics.pre_sns_v3_v2_replay_tile_manifest import (  # noqa: E402
    MARKER,
    build_replay_manifest,
    load_validation_case_ids,
    recommended_training_config,
    run_v2_replay_tile_manifest,
    validate_v2_replay_tile_manifest_config,
    weight_for_record,
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


def safe_config(root: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_pre_sns_v3_v2_replay_tile_manifest",
        "execution_mode": "approved_local_pre_sns_v3_v2_replay_tile_manifest",
        "base_tile_manifest_path": str(root / "inputs" / "tile_manifest.json"),
        "policy_still_failed_cases_path": str(root / "inputs" / "policy_still_failed_cases.json"),
        "policy_worse_cases_path": str(root / "inputs" / "policy_worse_cases.json"),
        "policy_non_tampered_high_mask_cases_path": None,
        "policy_fixed_cases_path": None,
        "policy_better_cases_path": None,
        "approved_input_roots": [str(root / "inputs")],
        "output_root": str(root / "reports" / "replay"),
        "max_records": 100,
        "no_download": True,
        "no_network": True,
        "no_training": True,
        "no_sns_augmentation": True,
    }


def write_inputs(root: Path) -> None:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_split": "train",
        "tile_size": 768,
        "records": [
            {"source_sample_id": "s1", "tile_class": "positive_tampered", "mining_bucket": "severe_iou_fail", "crop_box": [0, 0, 1, 1]},
            {"source_sample_id": "s2", "tile_class": "positive_tampered", "mining_bucket": "low_iou", "crop_box": [0, 0, 1, 1]},
            {"source_sample_id": "s3", "tile_class": "positive_tampered", "mining_bucket": "weak_iou", "crop_box": [0, 0, 1, 1]},
            {"source_sample_id": "s4", "tile_class": "positive_tampered", "failure_types": ["wrong_region"], "crop_box": [0, 0, 1, 1]},
            {"source_sample_id": "s5", "tile_class": "positive_tampered", "failure_types": ["undersegmented"], "crop_box": [0, 0, 1, 1]},
            {"source_sample_id": "s6", "tile_class": "hard_negative", "crop_box": [0, 0, 1, 1]},
            {"source_sample_id": "s7", "tile_class": "negative_real", "crop_box": [0, 0, 1, 1]},
            {"source_sample_id": "s8", "tile_class": "negative_synthetic", "crop_box": [0, 0, 1, 1]},
            {"source_sample_id": "s9", "tile_class": "positive_tampered", "mining_bucket": "good_iou", "crop_box": [0, 0, 1, 1]},
        ],
    }
    (inputs / "tile_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (inputs / "policy_still_failed_cases.json").write_text(json.dumps([{"sample_id": "s2"}]), encoding="utf-8")
    (inputs / "policy_worse_cases.json").write_text(json.dumps([{"sample_id": "s6"}]), encoding="utf-8")


def test_replay_weights() -> None:
    assert_equal(weight_for_record({"tile_class": "positive_tampered", "mining_bucket": "severe_iou_fail"}), 12, "severe weight")
    assert_equal(weight_for_record({"tile_class": "positive_tampered", "mining_bucket": "low_iou"}), 10, "low weight")
    assert_equal(weight_for_record({"tile_class": "positive_tampered", "mining_bucket": "weak_iou"}), 5, "weak weight")
    assert_equal(weight_for_record({"tile_class": "positive_tampered", "failure_types": ["wrong_region"]}), 12, "wrong region weight")
    assert_equal(weight_for_record({"tile_class": "positive_tampered", "failure_types": ["undersegmented"]}), 4, "underseg weight")
    assert_equal(weight_for_record({"tile_class": "hard_negative"}), 8, "hard negative weight")
    assert_equal(weight_for_record({"tile_class": "negative_real"}), 3, "negative real weight")
    assert_equal(weight_for_record({"tile_class": "positive_tampered", "mining_bucket": "good_iou"}), 1, "good iou weight")


def test_leakage_exclusion_and_manifest_schema() -> None:
    root = temp_root("cvf_v2_replay_manifest_")
    write_inputs(root)
    cfg = safe_config(root)
    excluded_ids = load_validation_case_ids(cfg)
    assert_equal(excluded_ids, {"s2", "s6"}, "excluded ids")
    base_manifest = json.loads((root / "inputs" / "tile_manifest.json").read_text(encoding="utf-8"))
    replay_manifest, replay_summary = build_replay_manifest(cfg, base_manifest, excluded_ids)
    assert_equal(replay_manifest["marker"], MARKER, "manifest marker")
    assert_true(all(record["source_sample_id"] not in excluded_ids for record in replay_manifest["records"]), "excluded ids removed")
    assert_true(replay_summary["replay_record_count"] > 0, "replay record count")
    assert_true("replay_weight" in replay_manifest["records"][0], "replay weight field")


def test_recommended_training_config_fields() -> None:
    root = temp_root("cvf_v2_replay_traincfg_")
    cfg = safe_config(root)
    replay_path = root / "reports" / "replay" / "replay_tile_manifest.json"
    out = recommended_training_config(cfg, replay_path)
    assert_equal(out["config_kind"], "approved_pre_sns_v3_tile_localizer_v2_training", "training kind")
    assert_equal(out["execution_mode"], "approved_local_pre_sns_v3_tile_localizer_v2_training", "training mode")
    assert_equal(out["tile_manifest_path"], str(replay_path), "manifest path")
    assert_equal(out["max_tiles_train"], 24000, "max tiles train")
    assert_equal(out["max_tiles_val"], 3000, "max tiles val")
    assert_equal(out["epochs"], 60, "epochs")
    assert_equal(out["learning_rate"], 3e-5, "lr")
    assert_equal(out["input_feature_mode"], "rgb_edge_residual", "feature mode")
    assert_true(out["progress_write_json"] is True and out["progress_write_jsonl"] is True, "progress outputs")


def test_validator_guardrails() -> None:
    root = temp_root("cvf_v2_replay_validator_")
    write_inputs(root)
    cfg = safe_config(root)
    assert_equal(validate_v2_replay_tile_manifest_config(cfg, require_exists=False), [], "safe config")
    repo_out = dict(cfg)
    repo_out["output_root"] = str(REPO_ROOT / "replay_out")
    assert_true(validate_v2_replay_tile_manifest_config(repo_out, require_exists=False), "repo output rejected")
    for key in ("no_network", "no_download", "no_training", "no_sns_augmentation"):
        unsafe = dict(cfg)
        unsafe[key] = False
        assert_true(validate_v2_replay_tile_manifest_config(unsafe, require_exists=False), f"{key} rejected")


def test_run_replay_builder_no_repo_writes() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_v2_replay_run_")
    write_inputs(root)
    cfg = safe_config(root)
    summary = run_v2_replay_tile_manifest(cfg)
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "no repo root writes")
    assert_equal(summary["marker"], MARKER, "summary marker")
    out = Path(cfg["output_root"])
    for name in (
        "replay_tile_manifest.json",
        "replay_tile_manifest_summary.json",
        "excluded_validation_case_ids.json",
        "recommended_v2_medium_train_config.json",
        "artifact_manifest.json",
    ):
        assert_true((out / name).is_file(), f"{name} exists")


def main() -> int:
    tests = [
        test_replay_weights,
        test_leakage_exclusion_and_manifest_schema,
        test_recommended_training_config_fields,
        test_validator_guardrails,
        test_run_replay_builder_no_repo_writes,
    ]
    for test in tests:
        test()
    print("PRE_SNS_V3_V2_REPLAY_TILE_MANIFEST_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
