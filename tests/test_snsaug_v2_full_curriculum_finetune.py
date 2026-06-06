#!/usr/bin/env python3
"""Plain Python tests for SNSAug V2 full-curriculum fine-tuning guardrails."""

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

from cv_forensics.snsaug_v2_full_curriculum_finetune import (  # noqa: E402
    APPROVAL_TEXT,
    BEST_POLICY,
    MARKER,
    MODEL_VERSION,
    build_full_curriculum_plan,
    checkpoint_score,
    run_snsaug_v2_full_curriculum_finetune,
    select_best_checkpoint,
    validate_snsaug_v2_full_curriculum_finetune_config,
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


def write_fixture(root: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    train_root = root / "train"
    eval_root = root / "eval"
    analysis_root = root / "analysis"
    model_root = root / "models"
    for path in (train_root, eval_root, analysis_root, model_root):
        path.mkdir(parents=True, exist_ok=True)
    manifest = train_root / "snsaug_v2_train_curriculum_manifest.jsonl"
    rows = [
        {"base_id": "real_train", "split": "train", "image_path": str(train_root / "real.png"), "content_label": "real"},
        {"base_id": "synth_train", "split": "train", "image_path": str(train_root / "synth.png"), "content_label": "synthetic"},
        {"base_id": "tampered_train", "split": "train", "image_path": str(train_root / "tampered.png"), "content_label": "tampered"},
    ]
    with open(manifest, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    schedule = train_root / "snsaug_v2_curriculum_schedule.json"
    weights = train_root / "snsaug_v2_profile_sampling_weights.json"
    schedule.write_text(json.dumps({"curriculum_schedule": {}}), encoding="utf-8")
    weights.write_text(json.dumps({"profile_sampling_weights": {}}), encoding="utf-8")
    bundle = model_root / "bundle.json"
    bundle.write_text(json.dumps({"marker": "fixture"}), encoding="utf-8")
    clean_val = eval_root / "clean_validation_manifest.jsonl"
    clean_val.write_text(json.dumps({"base_id": "val", "split": "val", "image_path": str(eval_root / "val.png")}) + "\n", encoding="utf-8")
    pair_root = eval_root / "snsaug_v2_0058c_fixed_pairs"
    pair_root.mkdir(parents=True, exist_ok=True)
    oracle_root = analysis_root / "snsaug_v2_0058e_oracle"
    oracle_root.mkdir(parents=True, exist_ok=True)
    return manifest, schedule, weights, bundle, clean_val, pair_root, oracle_root


def safe_config(root: Path) -> dict[str, object]:
    manifest, schedule, weights, bundle, clean_val, pair_root, oracle_root = write_fixture(root)
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_full_curriculum_finetune",
        "execution_mode": "approved_local_snsaug_v2_full_curriculum_finetune",
        "run_kind": "full_curriculum_finetune",
        "model_version": MODEL_VERSION,
        "approval_text": APPROVAL_TEXT,
        "start_from_pre_sns_best": True,
        "training_manifest_path": str(manifest),
        "curriculum_schedule_path": str(schedule),
        "profile_sampling_weights_path": str(weights),
        "base_model_bundle_path": str(bundle),
        "clean_validation_manifest_path": str(clean_val),
        "evaluation_pair_root_0058c": str(pair_root),
        "oracle_analysis_root_0058e": str(oracle_root),
        "approved_train_manifest_roots": [str(root / "train")],
        "approved_model_roots": [str(root / "models")],
        "approved_evaluation_roots": [str(root / "eval")],
        "approved_analysis_roots": [str(root / "analysis")],
        "approved_output_roots": [str(root / "runs")],
        "approved_checkpoint_roots": [str(root / "ckpts")],
        "output_root": str(root / "runs" / "full"),
        "checkpoint_root": str(root / "ckpts" / "full"),
        "real_fpr_limit": 0.05,
        "max_steps_per_phase": 3,
        "phase_1_max_steps": 2,
        "phase_2_max_steps": 2,
        "phase_3_max_steps": 2,
        "best_checkpoint_policy": BEST_POLICY,
        "no_network": True,
        "no_download": True,
    }


def test_approval_and_train_from_scratch_rejected() -> None:
    root = temp_root("cvf_0060b_approval_")
    cfg = safe_config(root)
    cfg["approval_text"] = ""
    assert_true(validate_snsaug_v2_full_curriculum_finetune_config(cfg, require_exists=True), "approval required")
    cfg = safe_config(root)
    cfg["start_from_pre_sns_best"] = False
    assert_true(validate_snsaug_v2_full_curriculum_finetune_config(cfg, require_exists=True), "scratch rejected")


def test_train_only_manifest_and_eval_training_input_rejected() -> None:
    root = temp_root("cvf_0060b_leakage_")
    cfg = safe_config(root)
    with open(str(cfg["training_manifest_path"]), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"base_id": "val_bad", "split": "val", "image_path": str(root / "train" / "val.png"), "content_label": "real"}) + "\n")
    errors = validate_snsaug_v2_full_curriculum_finetune_config(cfg, require_exists=True)
    assert_true(any("train split only" in error for error in errors), "val/test row rejected")
    cfg = safe_config(temp_root("cvf_0060b_evalpath_"))
    cfg["training_manifest_path"] = str(Path(cfg["approved_train_manifest_roots"][0]) / "fixed_pairs_manifest.jsonl")
    errors = validate_snsaug_v2_full_curriculum_finetune_config(cfg, require_exists=False)
    assert_true(any("fixed-pair" in error for error in errors), "fixed pair training input rejected")
    cfg = safe_config(temp_root("cvf_0060b_evalrow_"))
    with open(str(cfg["training_manifest_path"]), "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "base_id": "eval_image_bad",
                    "split": "train",
                    "image_path": str(Path(cfg["evaluation_pair_root_0058c"]) / "image.png"),
                    "content_label": "real",
                }
            )
            + "\n"
        )
    errors = validate_snsaug_v2_full_curriculum_finetune_config(cfg, require_exists=True)
    assert_true(any("evaluation roots" in error or "fixed-pair" in error or "validation" in error for error in errors), "eval image training input rejected")


def test_output_checkpoint_roots_and_required_policy_rejected() -> None:
    root = temp_root("cvf_0060b_roots_")
    cfg = safe_config(root)
    cfg["output_root"] = str(REPO_ROOT / "runs")
    assert_true(validate_snsaug_v2_full_curriculum_finetune_config(cfg, require_exists=False), "repo output rejected")
    cfg = safe_config(root)
    cfg["checkpoint_root"] = str(REPO_ROOT / "ckpts")
    assert_true(validate_snsaug_v2_full_curriculum_finetune_config(cfg, require_exists=False), "repo checkpoint rejected")
    cfg = safe_config(root)
    cfg["real_fpr_limit"] = 1.5
    assert_true(validate_snsaug_v2_full_curriculum_finetune_config(cfg, require_exists=False), "real fpr guardrail required")
    cfg = safe_config(root)
    cfg["best_checkpoint_policy"] = {"primary": "clean_only"}
    assert_true(validate_snsaug_v2_full_curriculum_finetune_config(cfg, require_exists=False), "best policy required")
    cfg = safe_config(root)
    cfg["max_steps_per_phase"] = 31
    assert_true(validate_snsaug_v2_full_curriculum_finetune_config(cfg, require_exists=False), "max steps per phase rejected")
    cfg = safe_config(root)
    cfg["phase_2_max_steps"] = 0
    assert_true(validate_snsaug_v2_full_curriculum_finetune_config(cfg, require_exists=False), "phase steps rejected")


def test_best_checkpoint_policy_and_real_fpr_guardrail() -> None:
    candidates = [
        {"id": "bad_fpr", "metrics": {"snsaug_tampered_recall": 0.9, "snsaug_valid_iou": 0.6, "clean_macro_f1": 0.9, "real_fpr": 0.20}},
        {"id": "ok_a", "metrics": {"snsaug_tampered_recall": 0.5, "snsaug_valid_iou": 0.2, "clean_macro_f1": 0.8, "real_fpr": 0.03}},
        {"id": "ok_b", "metrics": {"snsaug_tampered_recall": 0.5, "snsaug_valid_iou": 0.2, "clean_macro_f1": 0.9, "real_fpr": 0.03}},
    ]
    best = select_best_checkpoint(candidates, 0.05)
    assert_equal(best["id"], "ok_b", "secondary clean macro f1 tie-break")
    assert_true(checkpoint_score(candidates[0]["metrics"], 0.05)[0] is False, "real fpr guardrail rejects")


def test_dry_run_starts_no_training_and_lists_outputs() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_0060b_dryrun_")
    cfg = safe_config(root)
    summary = run_snsaug_v2_full_curriculum_finetune(cfg, dry_run=True)
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "no repo writes")
    assert_equal(summary["marker"], MARKER, "marker")
    assert_true(summary["training_started"] is False, "training not started")
    assert_true(summary["checkpoint_written"] is False, "checkpoint not written")
    paths = summary["planned_output_paths"]
    for key in ("training_log", "per_phase_metrics", "best_checkpoint", "last_checkpoint", "artifact_manifest"):
        assert_true(key in paths, f"{key} planned")
    assert_true(not Path(cfg["checkpoint_root"]).exists(), "checkpoint root not created")


def test_actual_tiny_training_writes_required_artifacts_and_checkpoints_outside_repo() -> None:
    root = temp_root("cvf_0060b_materialize_")
    cfg = safe_config(root)
    summary = run_snsaug_v2_full_curriculum_finetune(cfg, dry_run=False)
    assert_true(summary["training_started"] is True, "actual branch starts training")
    assert_true(summary["checkpoint_written"] is True, "checkpoints written")
    assert_true(str(summary["best_checkpoint_path"]).endswith(".pt"), "best checkpoint is .pt")
    assert_true(str(summary["last_checkpoint_path"]).endswith(".pt"), "last checkpoint is .pt")
    expected = {
        "training_log",
        "per_phase_metrics",
        "clean_validation_metrics",
        "snsaug_0058c_metrics",
        "robustness_drop_metrics",
        "pre_sns_baseline_comparison",
        "report_markdown",
        "best_checkpoint",
        "last_checkpoint",
        "artifact_manifest",
    }
    assert_equal(set(summary["output_paths"]), expected, "all required artifacts listed")
    for path in summary["output_paths"].values():
        path_obj = Path(path)
        assert_true(path_obj.exists(), f"{path} exists")
        assert_true(not str(path_obj).startswith(str(REPO_ROOT)), "artifact outside repo")
    assert_true(Path(summary["output_paths"]["best_checkpoint"]).name == "snsaug_aware_multihead_forensics_v1_best.pt", "best pt name")
    assert_true(Path(summary["output_paths"]["last_checkpoint"]).name == "snsaug_aware_multihead_forensics_v1_last.pt", "last pt name")
    artifact = json.loads(Path(summary["output_paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert_true(artifact["training_started"] is True, "artifact records training_started")
    assert_true(artifact["checkpoint_written"] is True, "artifact records checkpoint_written")
    assert_true(Path(artifact["best_checkpoint_path"]).exists(), "artifact best checkpoint exists")
    assert_true(Path(artifact["last_checkpoint_path"]).exists(), "artifact last checkpoint exists")
    phase_metrics = json.loads(Path(summary["output_paths"]["per_phase_metrics"]).read_text(encoding="utf-8"))
    assert_equal(len(phase_metrics["phases"]), 3, "three phase metrics")
    for phase in phase_metrics["phases"]:
        assert_true(phase["losses_finite"] is True, "phase losses finite")
        assert_true(float(phase["mean_losses"]["total_loss"]) >= 0.0, "total loss finite")
    clean_eval = json.loads(Path(summary["output_paths"]["clean_validation_metrics"]).read_text(encoding="utf-8"))
    sns_eval = json.loads(Path(summary["output_paths"]["snsaug_0058c_metrics"]).read_text(encoding="utf-8"))
    assert_true(clean_eval["eval_subset_only"] is True, "clean eval marked subset")
    assert_true("sample_count" in clean_eval, "clean sample count")
    assert_true(sns_eval["eval_subset_only"] is True, "sns eval marked subset")
    assert_true("sample_count" in sns_eval, "sns sample count")


def test_plan_schema_contains_phases_and_metrics() -> None:
    root = temp_root("cvf_0060b_plan_")
    plan = build_full_curriculum_plan(safe_config(root))
    assert_equal([phase["phase"] for phase in plan["phases"]], [1, 2, 3], "three phases")
    assert_true("SNSAugV2DatasetWrapper" == plan["dataset_wrapper"], "dataset wrapper named")
    assert_true("snsaug_tampered_recall" in plan["metrics"], "snsaug metric included")
    assert_equal(plan["best_checkpoint_policy"], BEST_POLICY, "best policy exposed")


def main() -> int:
    tests = [
        test_approval_and_train_from_scratch_rejected,
        test_train_only_manifest_and_eval_training_input_rejected,
        test_output_checkpoint_roots_and_required_policy_rejected,
        test_best_checkpoint_policy_and_real_fpr_guardrail,
        test_dry_run_starts_no_training_and_lists_outputs,
        test_actual_tiny_training_writes_required_artifacts_and_checkpoints_outside_repo,
        test_plan_schema_contains_phases_and_metrics,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
