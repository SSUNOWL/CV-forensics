#!/usr/bin/env python3
"""Plain Python tests for SNSAug V2 fine-tuning smoke infrastructure."""

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

from cv_forensics.snsaug_v2_finetune_runner import (  # noqa: E402
    APPROVAL_TEXT,
    MARKER,
    MODEL_VERSION,
    run_snsaug_v2_finetune_smoke,
    validate_snsaug_v2_finetune_smoke_config,
)
from cv_forensics.snsaug_v2_losses import (  # noqa: E402
    clean_sns_class_consistency_loss,
    compute_snsaug_v2_smoke_loss,
    hard_negative_tampered_loss,
    tampered_score_consistency_loss,
    valid_tamper_mask_loss,
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


def write_fixture(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    inputs = root / "inputs"
    eval_root = root / "eval" / "fixed_pairs"
    model_root = root / "models"
    inputs.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    rows = [
        {"base_id": "real_train", "split": "train", "image_path": str(inputs / "real.png"), "content_label": "real"},
        {"base_id": "synth_train", "split": "train", "image_path": str(inputs / "synthetic.png"), "content_label": "synthetic"},
        {"base_id": "tampered_train", "split": "train", "image_path": str(inputs / "tampered.png"), "content_label": "tampered"},
    ]
    manifest = inputs / "snsaug_v2_train_curriculum_manifest.jsonl"
    with open(manifest, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    schedule = inputs / "snsaug_v2_curriculum_schedule.json"
    weights = inputs / "snsaug_v2_profile_sampling_weights.json"
    schedule.write_text(json.dumps({"curriculum_schedule": {}}), encoding="utf-8")
    weights.write_text(json.dumps({"profile_groups": {}}), encoding="utf-8")
    bundle = model_root / "bundle.json"
    bundle.write_text(json.dumps({"marker": "fixture"}), encoding="utf-8")
    return manifest, schedule, weights, bundle, eval_root


def safe_config(root: Path) -> dict[str, object]:
    manifest, schedule, weights, bundle, eval_root = write_fixture(root)
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_finetune_smoke",
        "execution_mode": "approved_local_snsaug_v2_finetune_smoke",
        "run_kind": "smoke",
        "model_version": MODEL_VERSION,
        "approval_text": APPROVAL_TEXT,
        "training_manifest_path": str(manifest),
        "curriculum_schedule_path": str(schedule),
        "profile_sampling_weights_path": str(weights),
        "base_model_bundle_path": str(bundle),
        "evaluation_pair_root": str(eval_root),
        "approved_train_manifest_roots": [str(root / "inputs")],
        "approved_model_roots": [str(root / "models")],
        "approved_evaluation_roots": [str(root / "eval")],
        "approved_output_roots": [str(root / "runs")],
        "approved_checkpoint_roots": [str(root / "ckpts")],
        "output_root": str(root / "runs" / "smoke"),
        "checkpoint_root": str(root / "ckpts" / "smoke"),
        "max_steps": 5,
        "smoke_max_steps_limit": 300,
        "epochs": 1,
        "batch_size": 2,
        "samples_per_class": 2,
        "smoke_only": True,
        "no_full_training": True,
        "no_network": True,
        "no_download": True,
    }


def test_config_validator_requires_approval_text() -> None:
    root = temp_root("cvf_0060a_approval_")
    cfg = safe_config(root)
    cfg["approval_text"] = ""
    errors = validate_snsaug_v2_finetune_smoke_config(cfg, require_exists=True)
    assert_true(any("approval_text" in error for error in errors), "approval text required")


def test_config_validator_rejects_val_rows_and_pair_roots_as_training() -> None:
    root = temp_root("cvf_0060a_leakage_")
    cfg = safe_config(root)
    with open(str(cfg["training_manifest_path"]), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"base_id": "val_bad", "split": "val", "image_path": str(root / "inputs" / "val.png"), "content_label": "real"}) + "\n")
    errors = validate_snsaug_v2_finetune_smoke_config(cfg, require_exists=True)
    assert_true(any("train split only" in error for error in errors), "val row rejected")
    eval_train = safe_config(temp_root("cvf_0060a_evalpath_"))
    eval_train["training_manifest_path"] = str(Path(eval_train["approved_train_manifest_roots"][0]) / "fixed_pairs_manifest.jsonl")
    errors = validate_snsaug_v2_finetune_smoke_config(eval_train, require_exists=False)
    assert_true(any("evaluation roots" in error for error in errors), "pair root training input rejected")
    eval_row = safe_config(temp_root("cvf_0060a_evalrow_"))
    eval_root = Path(eval_row["approved_evaluation_roots"][0]) / "bench"
    eval_root.mkdir(parents=True, exist_ok=True)
    eval_row["evaluation_pair_root"] = str(eval_root)
    with open(str(eval_row["training_manifest_path"]), "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "base_id": "eval_image_bad",
                    "split": "train",
                    "image_path": str(eval_root / "image.png"),
                    "content_label": "real",
                }
            )
            + "\n"
        )
    errors = validate_snsaug_v2_finetune_smoke_config(eval_row, require_exists=True)
    assert_true(any("evaluation_pair_root" in error for error in errors), "eval pair image rejected")


def test_config_validator_rejects_repo_roots_and_smoke_limit_violations() -> None:
    root = temp_root("cvf_0060a_roots_")
    cfg = safe_config(root)
    cfg["output_root"] = str(REPO_ROOT / "smoke")
    assert_true(validate_snsaug_v2_finetune_smoke_config(cfg, require_exists=False), "repo output rejected")
    cfg = safe_config(root)
    cfg["checkpoint_root"] = str(REPO_ROOT / "ckpt")
    assert_true(validate_snsaug_v2_finetune_smoke_config(cfg, require_exists=False), "repo checkpoint rejected")
    cfg = safe_config(root)
    cfg["max_steps"] = 301
    assert_true(validate_snsaug_v2_finetune_smoke_config(cfg, require_exists=False), "max steps rejected")
    cfg = safe_config(root)
    cfg["smoke_only"] = False
    assert_true(validate_snsaug_v2_finetune_smoke_config(cfg, require_exists=False), "smoke_only rejected")
    cfg = safe_config(root)
    cfg["no_full_training"] = False
    assert_true(validate_snsaug_v2_finetune_smoke_config(cfg, require_exists=False), "no_full_training rejected")
    cfg = safe_config(root)
    cfg["epochs"] = 2
    assert_true(validate_snsaug_v2_finetune_smoke_config(cfg, require_exists=False), "epochs rejected")


def test_loss_functions_are_finite_and_ignore_mask_is_excluded() -> None:
    masked_loss = valid_tamper_mask_loss([0.9, 0.1], [1.0, 1.0], [0.0, 1.0])
    unmasked_loss = valid_tamper_mask_loss([0.9, 0.1], [1.0, 1.0], [0.0, 0.0])
    assert_true(float(masked_loss) < float(unmasked_loss), "ignore_mask excludes bad pixel")
    total = compute_snsaug_v2_smoke_loss(
        class_logits=[[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
        labels=["real", "tampered"],
        pred_mask=[0.1, 0.8],
        tamper_mask=[0.0, 1.0],
        ignore_mask=[0.0, 0.0],
        clean_logits=[[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]],
        sns_logits=[[0.0, 0.0, 2.0], [2.0, 0.0, 0.0]],
    )
    assert_true(float(total["total_loss"]) >= 0.0, "total loss finite")


def test_score_consistency_and_hardneg_penalize_expected_failures() -> None:
    collapse = tampered_score_consistency_loss([[0.0, 0.0, 4.0]], [[4.0, 0.0, 0.0]], ["tampered"])
    stable = tampered_score_consistency_loss([[0.0, 0.0, 4.0]], [[0.0, 0.0, 4.0]], ["tampered"])
    assert_true(float(collapse) > float(stable), "SNS tampered collapse penalized")
    hard_bad = hard_negative_tampered_loss([[0.0, 0.0, 4.0]], ["real"])
    hard_good = hard_negative_tampered_loss([[4.0, 0.0, 0.0]], ["real"])
    assert_true(float(hard_bad) > float(hard_good), "hard negative tampered probability penalized")
    consistency = clean_sns_class_consistency_loss([[4.0, 0.0, 0.0]], [[0.0, 0.0, 4.0]])
    assert_true(float(consistency) > 0.0, "class consistency detects distribution shift")


def test_runner_dry_run_writes_no_checkpoint_and_starts_no_training() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_0060a_dryrun_")
    cfg = safe_config(root)
    summary = run_snsaug_v2_finetune_smoke(cfg, dry_run=True)
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "no repo writes")
    assert_equal(summary["marker"], MARKER, "marker")
    assert_true(summary["training_started"] is False, "training not started")
    assert_true(summary["checkpoint_written"] is False, "checkpoint not written")
    assert_true(not Path(cfg["checkpoint_root"]).exists(), "checkpoint dir not created")


def test_runner_real_guarded_path_writes_required_artifacts_outside_repo() -> None:
    root = temp_root("cvf_0060a_realpath_")
    cfg = safe_config(root)
    summary = run_snsaug_v2_finetune_smoke(cfg, dry_run=False)
    assert_true(summary["training_started"] is True, "actual smoke training starts")
    assert_true(summary["checkpoint_written"] is True, "checkpoint written")
    expected = {
        "smoke_train_log",
        "smoke_train_summary",
        "loss_breakdown",
        "snsaug_sampling_summary",
        "smoke_eval_clean_summary",
        "smoke_eval_0058c_summary",
        "smoke_checkpoint",
        "artifact_manifest",
    }
    assert_equal(set(summary["output_paths"]), expected, "all required paths listed")
    for path in summary["output_paths"].values():
        path_obj = Path(path)
        assert_true(path_obj.exists(), f"{path} exists")
        assert_true(not str(path_obj).startswith(str(REPO_ROOT)), "artifact outside repo")
    artifact = json.loads(Path(summary["output_paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert_true(artifact["training_started"] is True, "artifact records training_started")
    assert_true(artifact["checkpoint_written"] is True, "artifact records checkpoint_written")
    loss_breakdown = json.loads(Path(summary["output_paths"]["loss_breakdown"]).read_text(encoding="utf-8"))
    assert_true(loss_breakdown["finite"] is True, "losses finite")
    assert_true(float(loss_breakdown["mean_losses"]["total_loss"]) >= 0.0, "mean total loss finite")
    clean_eval = json.loads(Path(summary["output_paths"]["smoke_eval_clean_summary"]).read_text(encoding="utf-8"))
    sns_eval = json.loads(Path(summary["output_paths"]["smoke_eval_0058c_summary"]).read_text(encoding="utf-8"))
    assert_true(clean_eval["eval_subset_only"] is True, "clean eval placeholder marked subset")
    assert_true(sns_eval["eval_subset_only"] is True, "sns eval placeholder marked subset")


def main() -> int:
    tests = [
        test_config_validator_requires_approval_text,
        test_config_validator_rejects_val_rows_and_pair_roots_as_training,
        test_config_validator_rejects_repo_roots_and_smoke_limit_violations,
        test_loss_functions_are_finite_and_ignore_mask_is_excluded,
        test_score_consistency_and_hardneg_penalize_expected_failures,
        test_runner_dry_run_writes_no_checkpoint_and_starts_no_training,
        test_runner_real_guarded_path_writes_required_artifacts_outside_repo,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
