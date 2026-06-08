#!/usr/bin/env python3
"""Plain Python tests for SNSAug V2 balanced hard-negative fine-tuning."""

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

from cv_forensics.snsaug_v2_balanced_hard_negative_finetune import (  # noqa: E402
    APPROVAL_TEXT,
    BEST_POLICY,
    MARKER,
    balanced_checkpoint_score,
    build_hard_negative_sampling_plan,
    load_snsaug_v2_balanced_hard_negative_finetune_config,
    loss_diagnostics,
    run_snsaug_v2_balanced_hard_negative_finetune,
    select_best_balanced_checkpoint,
    validate_real_checkpoint_payload,
    validate_snsaug_v2_balanced_hard_negative_finetune_config,
)
from cv_forensics.snsaug_v2_losses import (  # noqa: E402
    clean_sns_non_tampered_class_consistency_loss,
    non_tampered_tampered_suppression_loss,
    tampered_score_consistency_loss,
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


def write_fixture(root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    train_root = root / "train"
    eval_root = root / "eval"
    model_root = root / "models"
    for path in (train_root, eval_root, model_root):
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
    bundle.write_text(json.dumps({"long256_checkpoint_path": str(model_root / "long.pt")}), encoding="utf-8")
    clean_val = eval_root / "clean_validation_manifest.jsonl"
    clean_val.write_text(json.dumps({"base_id": "val", "split": "val", "image_path": str(eval_root / "val.png")}) + "\n", encoding="utf-8")
    pair_root = eval_root / "snsaug_v2_0058c_fixed_pairs"
    pair_root.mkdir(parents=True, exist_ok=True)
    return manifest, schedule, weights, bundle, clean_val, pair_root


def safe_config(root: Path) -> dict[str, object]:
    manifest, schedule, weights, bundle, clean_val, pair_root = write_fixture(root)
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_balanced_hard_negative_finetune",
        "execution_mode": "approved_local_snsaug_v2_balanced_hard_negative_finetune",
        "run_kind": "balanced_hard_negative_finetune",
        "model_version": "snsaug_aware_multihead_forensics_v1",
        "approval_text": "DRY_RUN_ONLY",
        "start_from_pre_sns_best": True,
        "no_training_from_scratch": True,
        "training_manifest_path": str(manifest),
        "curriculum_schedule_path": str(schedule),
        "profile_sampling_weights_path": str(weights),
        "base_model_bundle_path": str(bundle),
        "clean_validation_manifest_path": str(clean_val),
        "evaluation_pair_root_0058c": str(pair_root),
        "approved_train_manifest_roots": [str(root / "train")],
        "approved_model_roots": [str(root / "models")],
        "approved_evaluation_roots": [str(root / "eval")],
        "approved_output_roots": [str(root / "runs")],
        "approved_checkpoint_roots": [str(root / "ckpts")],
        "output_root": str(root / "runs" / "balanced"),
        "checkpoint_root": str(root / "ckpts" / "balanced"),
        "p_tampered_ceiling": 0.05,
        "lambda_hardneg": 2.0,
        "best_checkpoint_policy": BEST_POLICY,
        "hard_negative_sampling": {"enabled": True, "non_tampered_sns_to_tampered_sns_ratio": [2, 1]},
        "enable_real_training": False,
        "write_checkpoints": False,
        "no_network": True,
        "no_download": True,
    }


def test_example_config_validates() -> None:
    cfg = load_snsaug_v2_balanced_hard_negative_finetune_config(
        REPO_ROOT / "configs" / "training" / "snsaug_v2_balanced_hard_negative_finetune.example.json"
    )
    assert_equal(validate_snsaug_v2_balanced_hard_negative_finetune_config(cfg, require_exists=False), [], "example config validates")


def test_hardneg_loss_ceiling_semantics() -> None:
    high = non_tampered_tampered_suppression_loss([[0.02, 0.03, 0.95]], ["real"], ceiling=0.05)
    low = non_tampered_tampered_suppression_loss([[0.90, 0.05, 0.05]], ["synthetic"], ceiling=0.05)
    tampered = non_tampered_tampered_suppression_loss([[0.02, 0.03, 0.95]], ["tampered"], ceiling=0.05)
    assert_true(float(high) > 0.0, "hardneg loss positive above ceiling")
    assert_equal(float(low), 0.0, "hardneg loss zero at ceiling")
    assert_equal(float(tampered), 0.0, "hardneg ignores tampered labels")


def test_tampered_score_consistency_only_tampered() -> None:
    clean = [[0.0, 0.0, 3.0], [3.0, 0.0, 0.0]]
    sns = [[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]]
    losses = tampered_score_consistency_loss(clean, sns, ["tampered", "real"], reduction="none")
    assert_true(float(losses[0]) > 0.0, "tampered collapse penalized")
    assert_equal(float(losses[1]), 0.0, "real score recovery not applied")


def test_non_tampered_class_consistency_excludes_tampered_overactivation() -> None:
    clean = [[4.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    sns = [[4.0, 1.0, 4.0], [1.0, 4.0, 4.0], [4.0, 0.0, 0.0]]
    loss = clean_sns_non_tampered_class_consistency_loss(clean, sns, ["real", "synthetic", "tampered"])
    assert_equal(round(float(loss), 8), 0.0, "tampered mass is excluded from non-tampered consistency")


def test_loss_diagnostics_logs_required_values() -> None:
    diag = loss_diagnostics(
        [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        [[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [3.0, 0.0, 0.0]],
        ["real", "synthetic", "tampered"],
    )
    for key in (
        "mean_p_tampered_real_sns",
        "mean_p_tampered_synthetic_sns",
        "mean_p_tampered_tampered_sns",
        "hardneg_loss",
        "tampered_score_consistency_loss",
        "clean_sns_class_consistency_loss",
    ):
        assert_true(key in diag, f"{key} logged")


def test_sampling_plan_targets_two_to_one() -> None:
    cfg = safe_config(temp_root("cvf_0063_sampling_"))
    plan = build_hard_negative_sampling_plan(cfg)
    assert_equal(plan["non_tampered_sns_to_tampered_sns_ratio"], [2, 1], "2:1 ratio")
    assert_equal(round(plan["non_tampered_sns_fraction"], 4), 0.6667, "non-tampered fraction")
    assert_true(2 in plan["hard_sns_phases"] and 3 in plan["hard_sns_phases"], "hard phases listed")


def test_config_guardrails_reject_leakage_and_flags() -> None:
    root = temp_root("cvf_0063_guard_")
    cfg = safe_config(root)
    with open(str(cfg["training_manifest_path"]), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"base_id": "bad", "split": "val", "image_path": str(root / "train" / "bad.png")}) + "\n")
    errors = validate_snsaug_v2_balanced_hard_negative_finetune_config(cfg, require_exists=True)
    assert_true(any("train split only" in error for error in errors), "val row rejected")

    cfg = safe_config(temp_root("cvf_0063_evalrow_"))
    with open(str(cfg["training_manifest_path"]), "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "base_id": "eval_bad",
                    "split": "train",
                    "image_path": str(Path(cfg["evaluation_pair_root_0058c"]) / "image.png"),
                    "content_label": "real",
                }
            )
            + "\n"
        )
    errors = validate_snsaug_v2_balanced_hard_negative_finetune_config(cfg, require_exists=True)
    assert_true(any("evaluation roots" in error or "fixed-pair" in error for error in errors), "eval path rejected")

    cfg = safe_config(temp_root("cvf_0063_flags_"))
    cfg["no_network"] = False
    cfg["no_download"] = False
    errors = validate_snsaug_v2_balanced_hard_negative_finetune_config(cfg, require_exists=False)
    assert_true(any("no_network" in error for error in errors), "no_network required")
    assert_true(any("no_download" in error for error in errors), "no_download required")


def test_output_checkpoint_and_approval_guardrails() -> None:
    root = temp_root("cvf_0063_roots_")
    cfg = safe_config(root)
    cfg["output_root"] = str(REPO_ROOT / "runs")
    assert_true(validate_snsaug_v2_balanced_hard_negative_finetune_config(cfg, require_exists=False), "repo output rejected")
    cfg = safe_config(root)
    cfg["checkpoint_root"] = str(REPO_ROOT / "ckpts")
    assert_true(validate_snsaug_v2_balanced_hard_negative_finetune_config(cfg, require_exists=False), "repo checkpoint rejected")
    cfg = safe_config(root)
    cfg["enable_real_training"] = True
    cfg["approval_text"] = ""
    errors = validate_snsaug_v2_balanced_hard_negative_finetune_config(cfg, require_exists=False)
    assert_true(any(APPROVAL_TEXT in error for error in errors), "real training approval required")


def test_dry_run_starts_no_training_and_lists_required_outputs() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_0063_dry_")
    cfg = safe_config(root)
    summary = run_snsaug_v2_balanced_hard_negative_finetune(cfg, dry_run=True)
    after = set(os.listdir(REPO_ROOT))
    assert_equal(before, after, "no repo writes")
    assert_equal(summary["marker"], MARKER, "marker")
    assert_true(summary["training_started"] is False, "training not started")
    assert_true(summary["checkpoint_written"] is False, "checkpoint not written")
    for key in ("training_log", "loss_breakdown", "threshold_sweep_after_training", "best_checkpoint", "last_checkpoint"):
        assert_true(key in summary["planned_output_paths"], f"{key} planned")
    assert_true(not Path(cfg["checkpoint_root"]).exists(), "checkpoint root not created")


def test_balanced_score_and_guardrails() -> None:
    good = {
        "snsaug_tampered_recall": 0.70,
        "snsaug_valid_iou": 0.30,
        "synthetic_recall": 0.60,
        "real_fpr": 0.20,
        "non_tampered_high_mask_rate": 0.10,
        "clean_macro_f1": 0.80,
        "tampered_recall": 0.70,
    }
    bad = dict(good, real_fpr=0.40)
    passes, score = balanced_checkpoint_score(good)
    assert_true(passes is True, "good metrics pass")
    assert_equal(round(score, 4), 0.98, "balanced score formula")
    assert_true(balanced_checkpoint_score(bad)[0] is False, "real FPR guardrail")
    best = select_best_balanced_checkpoint([{"id": "bad", "metrics": bad}, {"id": "good", "metrics": good}])
    assert_equal(best["id"], "good", "best passing candidate selected")


def test_real_checkpoint_payload_validation() -> None:
    class TensorLike:
        def __init__(self, numel: int) -> None:
            self._numel = numel

        def numel(self) -> int:
            return self._numel

    good = {"checkpoint_format": "snsaug_v2_real_state_dict_v1", "model_state_dict": {"a": TensorLike(10), "b": TensorLike(20)}}
    stats = validate_real_checkpoint_payload(good)
    assert_true(stats["has_model_state_dict"] is True, "model_state_dict accepted")
    assert_equal(stats["tensor_total_numel"], 30, "tensor numel counted")
    try:
        validate_real_checkpoint_payload({"checkpoint_format": "snsaug_v2_real_state_dict_v1", "trainable_state": {"bias": [1.0]}})
    except Exception as exc:
        assert_true("trainable_state" in str(exc) or "model_state_dict" in str(exc), "proxy rejected")
    else:
        raise AssertionError("proxy-only checkpoint must be rejected")


def test_docs_marker_present() -> None:
    text = (REPO_ROOT / "docs" / "snsaug_v2_balanced_hard_negative_finetune.md").read_text(encoding="utf-8")
    assert_true(MARKER in text, "docs marker present")


def main() -> int:
    tests = [
        test_example_config_validates,
        test_hardneg_loss_ceiling_semantics,
        test_tampered_score_consistency_only_tampered,
        test_non_tampered_class_consistency_excludes_tampered_overactivation,
        test_loss_diagnostics_logs_required_values,
        test_sampling_plan_targets_two_to_one,
        test_config_guardrails_reject_leakage_and_flags,
        test_output_checkpoint_and_approval_guardrails,
        test_dry_run_starts_no_training_and_lists_required_outputs,
        test_balanced_score_and_guardrails,
        test_real_checkpoint_payload_validation,
        test_docs_marker_present,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
