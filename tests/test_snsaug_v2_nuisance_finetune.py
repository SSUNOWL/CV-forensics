#!/usr/bin/env python3
"""Plain Python tests for SNSAug V2 nuisance-mask fine-tuning."""

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

from cv_forensics.pre_sns_v3_model import build_pre_sns_v3_model  # noqa: E402
from cv_forensics.snsaug_v2_nuisance_finetune import (  # noqa: E402
    APPROVAL_TEXT,
    MARKER,
    collapse_guard_from_records,
    load_snsaug_v2_nuisance_finetune_config,
    run_snsaug_v2_nuisance_finetune,
    validate_real_checkpoint_payload,
    validate_snsaug_v2_nuisance_finetune_config,
)
from cv_forensics.snsaug_v2_nuisance_losses import (  # noqa: E402
    degradation_label_from_profile,
    non_tampered_mask_suppression_loss,
    sns_nuisance_mask_loss,
    sns_nuisance_mask_target,
    valid_tamper_mask_loss,
)
from cv_forensics.snsaug_v2_nuisance_model import apply_soft_nuisance_gate  # noqa: E402


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def temp_root(prefix: str) -> Path:
    TMP_PARENT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(TMP_PARENT)))


def write_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    train_root = root / "train"
    eval_root = root / "eval"
    model_root = root / "models"
    for path in (train_root, eval_root, model_root):
        path.mkdir(parents=True, exist_ok=True)
    manifest = train_root / "snsaug_v2_train_curriculum_manifest.jsonl"
    rows = [
        {"base_id": "real_train", "split": "train", "image_path": str(train_root / "real.png"), "content_label": "real", "profile": "tiktok_like_overlay_text"},
        {"base_id": "synth_train", "split": "train", "image_path": str(train_root / "synth.png"), "content_label": "synthetic", "profile": "jpeg_resize_light", "has_local_overlay": False},
        {"base_id": "tampered_train", "split": "train", "image_path": str(train_root / "tampered.png"), "content_label": "tampered", "profile": "combined_sns_realistic_sticker"},
    ]
    with open(manifest, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    import torch

    base_checkpoint = model_root / "pre_sns.pt"
    model = build_pre_sns_v3_model(torch, base_channels=2)
    torch.save(
        {
            "model_name": "pre_sns_v3",
            "image_size": 16,
            "base_channels": 2,
            "model_state_dict": model.state_dict(),
        },
        base_checkpoint,
    )
    bundle = model_root / "bundle.json"
    bundle.write_text(json.dumps({"long256_checkpoint_path": str(base_checkpoint)}), encoding="utf-8")
    clean_val = eval_root / "clean_validation_manifest.jsonl"
    clean_val.write_text(json.dumps({"base_id": "val", "split": "val", "image_path": str(eval_root / "val.png")}) + "\n", encoding="utf-8")
    pair_root = eval_root / "snsaug_v2_0058c_fixed_pairs"
    pair_root.mkdir(parents=True, exist_ok=True)
    return manifest, bundle, clean_val, pair_root


def safe_config(root: Path) -> dict[str, object]:
    manifest, bundle, clean_val, pair_root = write_fixture(root)
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_nuisance_mask_finetune",
        "execution_mode": "approved_local_snsaug_v2_nuisance_mask_finetune",
        "run_kind": "snsaug_v2_nuisance_mask_finetune",
        "model_version": "snsaug_aware_nuisance_multihead_forensics_v1",
        "approval_text": "DRY_RUN_ONLY",
        "start_from_pre_sns_best": True,
        "no_training_from_scratch": True,
        "training_manifest_path": str(manifest),
        "base_model_bundle_path": str(bundle),
        "clean_validation_manifest_path": str(clean_val),
        "evaluation_pair_root_0058c": str(pair_root),
        "approved_train_manifest_roots": [str(root / "train")],
        "approved_model_roots": [str(root / "models")],
        "approved_evaluation_roots": [str(root / "eval")],
        "approved_output_roots": [str(root / "runs")],
        "approved_checkpoint_roots": [str(root / "ckpts")],
        "output_root": str(root / "runs" / "nuisance"),
        "checkpoint_root": str(root / "ckpts" / "nuisance"),
        "lambda_sns_mask": 1.0,
        "lambda_degradation": 0.3,
        "lambda_hardneg": 1.0,
        "lambda_tamper_mask": 1.0,
        "lambda_gating_consistency": 0.5,
        "lambda_non_tampered_mask_suppression": 1.0,
        "lambda_mask_area_regularization": 0.1,
        "gating_alpha": 0.5,
        "phase_2_gating_alpha_max": 0.2,
        "phase_3_gating_alpha_max": 0.3,
        "p_tampered_ceiling": 0.05,
        "min_warm_start_loaded_numel_ratio": 0.1,
        "require_warm_start_loaded_numel_ratio_gte": 0.0,
        "allow_partial_warm_start": False,
        "allow_joint_tuning": False,
        "max_steps_per_phase": 1,
        "phase_1_max_steps": 1,
        "phase_2_max_steps": 1,
        "phase_3_max_steps": 1,
        "base_channels": 2,
        "image_size": 16,
        "enable_real_training": False,
        "write_checkpoints": False,
        "no_network": True,
        "no_download": True,
    }


def phase_counts(rows: list[dict[str, object]]) -> dict[int, int]:
    counts = {1: 0, 2: 0, 3: 0}
    for row in rows:
        counts[int(row["phase"])] += 1
    return counts


def test_example_config_validates() -> None:
    cfg = load_snsaug_v2_nuisance_finetune_config(REPO_ROOT / "configs" / "training" / "snsaug_v2_nuisance_finetune.example.json")
    assert_equal(validate_snsaug_v2_nuisance_finetune_config(cfg, require_exists=False), [], "example config validates")


def test_nuisance_mask_target_and_jpeg_only_profile() -> None:
    import torch

    ignore = torch.zeros((1, 1, 4, 4))
    ignore[:, :, :1, :] = 1.0
    target = sns_nuisance_mask_target(ignore, has_local_overlay=True)
    jpeg_target = sns_nuisance_mask_target(ignore, has_local_overlay=False)
    assert_equal(float(target.sum().item()), 4.0, "ignore mask becomes nuisance target")
    assert_equal(float(jpeg_target.sum().item()), 0.0, "jpeg-only has no local mask")
    labels = degradation_label_from_profile("jpeg_resize_light")
    assert_true(labels[0] == 1.0 and labels[1] == 1.0 and labels[9] == 1.0, "degradation labels set")


def test_mask_losses_and_gating() -> None:
    import torch

    pred = torch.full((1, 1, 4, 4), 0.1)
    tamper = torch.zeros((1, 1, 4, 4))
    tamper[:, :, :1, :] = 1.0
    ignore = torch.zeros((1, 1, 4, 4))
    ignore[:, :, :1, :] = 1.0
    ignored_loss = valid_tamper_mask_loss(pred, tamper, ignore)
    unignored_loss = valid_tamper_mask_loss(pred, tamper, torch.zeros_like(ignore))
    assert_true(float(ignored_loss) < float(unignored_loss), "tamper mask loss excludes ignore region")
    nuisance_loss = sns_nuisance_mask_loss(pred, ignore)
    assert_true(float(nuisance_loss) >= 0.0, "sns mask loss finite")
    features = torch.ones((1, 2, 4, 4))
    gated = apply_soft_nuisance_gate(features, ignore, alpha=0.5)
    assert_true(float(gated[:, :, :1, :].mean().item()) < float(gated[:, :, 1:, :].mean().item()), "gating reduces SNS areas")
    all_one = torch.ones((2, 1, 4, 4))
    suppression = non_tampered_mask_suppression_loss(all_one, torch.tensor([0, 1]))
    tampered_only = non_tampered_mask_suppression_loss(all_one, torch.tensor([2, 2]))
    assert_true(float(suppression) > 0.0, "non-tampered all-one masks penalized")
    assert_equal(float(tampered_only), 0.0, "tampered masks not suppressed by hard-negative mask loss")


def test_collapse_guard_detects_single_class_predictions() -> None:
    collapsed = collapse_guard_from_records([{"pred_class": "tampered"}, {"pred_class": "tampered"}, {"pred_class": "tampered"}])
    mixed = collapse_guard_from_records([{"pred_class": "real"}, {"pred_class": "synthetic"}, {"pred_class": "tampered"}])
    assert_true(collapsed["single_class_prediction_collapse"] is True, "single-class collapse detected")
    assert_true(mixed["single_class_prediction_collapse"] is False, "mixed predictions pass collapse guard")


def test_guardrails_reject_leakage_and_flags() -> None:
    root = temp_root("cvf_0064_guard_")
    cfg = safe_config(root)
    with open(str(cfg["training_manifest_path"]), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"base_id": "bad", "split": "val", "image_path": str(root / "train" / "bad.png"), "content_label": "real"}) + "\n")
    errors = validate_snsaug_v2_nuisance_finetune_config(cfg, require_exists=True)
    assert_true(any("train split only" in error for error in errors), "val row rejected")

    cfg = safe_config(temp_root("cvf_0064_evalrow_"))
    with open(str(cfg["training_manifest_path"]), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"base_id": "bad", "split": "train", "image_path": str(Path(cfg["evaluation_pair_root_0058c"]) / "bad.png"), "content_label": "real"}) + "\n")
    errors = validate_snsaug_v2_nuisance_finetune_config(cfg, require_exists=True)
    assert_true(any("evaluation roots" in error or "fixed-pair" in error for error in errors), "eval root rejected")

    cfg = safe_config(temp_root("cvf_0064_flags_"))
    cfg["no_network"] = False
    cfg["no_download"] = False
    errors = validate_snsaug_v2_nuisance_finetune_config(cfg, require_exists=False)
    assert_true(any("no_network" in error for error in errors), "no_network required")
    assert_true(any("no_download" in error for error in errors), "no_download required")


def test_dry_run_and_approval() -> None:
    before = set(os.listdir(REPO_ROOT))
    root = temp_root("cvf_0064_dry_")
    cfg = safe_config(root)
    summary = run_snsaug_v2_nuisance_finetune(cfg, dry_run=True)
    assert_equal(before, set(os.listdir(REPO_ROOT)), "no repo writes")
    assert_equal(summary["marker"], MARKER, "marker")
    assert_true(summary["training_started"] is False, "dry-run does not train")
    assert_true(summary["checkpoint_written"] is False, "dry-run writes no checkpoint")
    try:
        run_snsaug_v2_nuisance_finetune(cfg, dry_run=False)
    except Exception as exc:
        assert_true(APPROVAL_TEXT in str(exc), "approval required")
    else:
        raise AssertionError("real run without approval must fail")
    assert_true(not Path(cfg["output_root"]).exists(), "no output without approval")


def test_tiny_real_run_writes_checkpoints() -> None:
    root = temp_root("cvf_0064_real_")
    cfg = safe_config(root)
    cfg["approval_text"] = APPROVAL_TEXT
    cfg["require_warm_start_loaded_numel_ratio_gte"] = 0.90
    summary = run_snsaug_v2_nuisance_finetune(cfg, dry_run=False)
    assert_true(summary["training_started"] is True, "training started")
    assert_true(summary["checkpoint_written"] is True, "checkpoint written")
    paths = summary["output_paths"]
    for key in ("training_log", "warm_start_report", "loss_breakdown", "per_phase_metrics", "clean_validation_metrics", "snsaug_0058c_metrics", "threshold_sweep_after_training", "artifact_manifest", "best_checkpoint", "last_checkpoint"):
        path = Path(paths[key])
        assert_true(path.exists(), f"{key} exists")
        assert_true(not str(path).startswith(str(REPO_ROOT)), f"{key} outside repo")
    rows = [json.loads(line) for line in Path(paths["training_log"]).read_text(encoding="utf-8").splitlines()]
    assert_true(len(rows) >= 3, "1x3 training log rows")
    assert_equal(phase_counts(rows), {1: 1, 2: 1, 3: 1}, "1x3 phase counts")
    assert_equal(float(rows[0]["gating_alpha_effective"]), 0.0, "phase 1 gating disabled")
    for key in (
        "class_loss",
        "tamper_mask_loss",
        "sns_nuisance_mask_loss",
        "global_degradation_loss",
        "hardneg_loss",
        "clean_sns_class_consistency_loss",
        "non_tampered_mask_suppression_loss",
        "mask_area_regularization_loss",
        "total_loss",
    ):
        assert_true(key in rows[0] and float(rows[0][key]) >= 0.0, f"{key} logged")
    warm = json.loads(Path(paths["warm_start_report"]).read_text(encoding="utf-8"))
    assert_true(warm["warm_start_checkpoint_path"], "warm_start_checkpoint_path present")
    assert_true(warm["warm_start_checkpoint_exists"] is True, "warm-start source exists")
    assert_true(warm["loaded_numel"] > 0, "warm-start loaded weights")
    assert_true(warm["total_numel"] >= warm["loaded_numel"], "warm-start report total numel")
    assert_true(isinstance(warm["loaded_ratio"], float), "loaded_ratio finite")
    assert_equal(round(warm["loaded_ratio"], 8), round(warm["loaded_numel"] / warm["total_numel"], 8), "loaded ratio formula")
    assert_equal(warm["loaded_key_count"], len(warm["loaded_keys"]), "loaded key count")
    assert_true(warm["total_key_count"] >= warm["loaded_key_count"], "total key count")
    import torch

    artifact = json.loads(Path(paths["artifact_manifest"]).read_text(encoding="utf-8"))
    assert_true(artifact["warm_start_loaded_numel"] > 0, "artifact warm-start numel")
    assert_true("warm_start_summary" in artifact, "artifact warm_start_summary")
    assert_equal(artifact["warm_start_summary"]["warm_start_checkpoint_path"], warm["warm_start_checkpoint_path"], "summary path")
    assert_equal(artifact["warm_start_summary"]["loaded_ratio"], warm["loaded_ratio"], "summary ratio")
    assert_equal(artifact["warm_start_summary"]["missing_key_count"], len(warm["missing_keys"]), "summary missing count")
    assert_true(artifact["tensor_total_numel"] >= artifact["warm_start_loaded_numel"], "checkpoint tensor count comparable")
    for key in ("best_checkpoint", "last_checkpoint"):
        payload = torch.load(paths[key], map_location="cpu")
        assert_true("model_state_dict" in payload, f"{key} has model_state_dict")
        stats = validate_real_checkpoint_payload(payload)
        assert_true(stats["tensor_count"] > 0, f"{key} real tensors")
        assert_true(stats["tensor_total_numel"] >= warm["loaded_numel"], f"{key} tensor_total_numel comparable")
        assert_true(int(payload["global_step"]) >= 3, f"{key} global_step")
        assert_equal(int(payload["phase"]), 3, f"{key} final phase")


def test_phase_steps_2x3_are_honored() -> None:
    root = temp_root("cvf_0064d_2x3_")
    cfg = safe_config(root)
    cfg["approval_text"] = APPROVAL_TEXT
    cfg["phase_1_max_steps"] = 2
    cfg["phase_2_max_steps"] = 2
    cfg["phase_3_max_steps"] = 2
    summary = run_snsaug_v2_nuisance_finetune(cfg, dry_run=False)
    rows = [json.loads(line) for line in Path(summary["output_paths"]["training_log"]).read_text(encoding="utf-8").splitlines()]
    assert_true(len(rows) >= 6, "2x3 training log rows")
    assert_equal(phase_counts(rows), {1: 2, 2: 2, 3: 2}, "2x3 phase counts")
    artifact = json.loads(Path(summary["output_paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert_equal(artifact["phase_counts"], {"1": 2, "2": 2, "3": 2}, "artifact phase counts")
    import torch

    payload = torch.load(summary["output_paths"]["last_checkpoint"], map_location="cpu")
    assert_true("model_state_dict" in payload, "last checkpoint has model_state_dict")
    assert_true(int(payload["global_step"]) >= 6, "checkpoint global_step")
    assert_equal(payload["metrics"]["phase_counts"], {"1": 2, "2": 2, "3": 2}, "checkpoint phase counts")


def test_docs_marker_present() -> None:
    text = (REPO_ROOT / "docs" / "snsaug_v2_nuisance_finetune.md").read_text(encoding="utf-8")
    assert_true(MARKER in text, "docs marker present")


def main() -> int:
    tests = [
        test_example_config_validates,
        test_nuisance_mask_target_and_jpeg_only_profile,
        test_mask_losses_and_gating,
        test_collapse_guard_detects_single_class_predictions,
        test_guardrails_reject_leakage_and_flags,
        test_dry_run_and_approval,
        test_tiny_real_run_writes_checkpoints,
        test_phase_steps_2x3_are_honored,
        test_docs_marker_present,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
