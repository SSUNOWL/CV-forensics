#!/usr/bin/env python3
"""Plain Python tests for SNSAug V2 checkpoint comparison evaluation."""

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

from cv_forensics.snsaug_v2_checkpoint_comparison_eval import (  # noqa: E402
    APPROVAL_TEXT,
    MARKER,
    run_snsaug_v2_checkpoint_comparison_eval,
    validate_snsaug_v2_checkpoint_comparison_eval_config,
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
    pair_root = root / "eval" / "snsaug_v2_0058c_fixed_pairs"
    model_root = root / "models"
    ckpt_root = root / "ckpts"
    for path in (pair_root, model_root, ckpt_root):
        path.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in ("real", "synthetic", "tampered"):
        base_id = f"{label}_001"
        rows.append({"base_id": base_id, "view": "clean", "profile": "clean", "content_label": label})
        rows.append(
            {
                "base_id": base_id,
                "view": "sns_aug",
                "profile": "resize_crop_pad",
                "content_label": label,
                "pred_mask_path": str(pair_root / f"{base_id}_pred.png"),
                "pred_red_overlay_path": str(pair_root / f"{base_id}_red.png"),
                "gt_red_overlay_path": str(pair_root / f"{base_id}_gt.png"),
                "ignore_blue_overlay_path": str(pair_root / f"{base_id}_ignore.png"),
                "overlap_overlay_path": str(pair_root / f"{base_id}_overlap.png"),
            }
        )
    meta = pair_root / "meta.jsonl"
    with open(meta, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    bundle = model_root / "bundle.json"
    bundle.write_text(json.dumps({"marker": "baseline"}), encoding="utf-8")
    short = ckpt_root / "short.pt"
    medium = ckpt_root / "medium.pt"
    for path, class_bias in ((short, [0.0, 0.0, 0.05]), (medium, [0.0, 0.0, 0.15])):
        payload = {
            "marker": "dummy",
            "trainable_state": {
                "class_bias": class_bias,
                "mask_bias": 0.10,
            },
        }
        try:
            import torch
        except Exception:
            torch = None
        if torch is not None:
            torch.save(payload, path)
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")
    return pair_root, meta, bundle, short, medium


def safe_config(root: Path) -> dict[str, object]:
    pair_root, meta, bundle, short, medium = write_fixture(root)
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_checkpoint_comparison_eval",
        "execution_mode": "approved_local_snsaug_v2_checkpoint_comparison_eval",
        "user_approval_text": APPROVAL_TEXT,
        "pair_root": str(pair_root),
        "meta_jsonl_path": str(meta),
        "models": [
            {"model_id": "baseline", "model_kind": "pre_sns_bundle", "model_path": str(bundle)},
            {"model_id": "snsaug_30x3", "model_kind": "snsaug_finetuned_checkpoint", "model_path": str(short)},
            {"model_id": "snsaug_150x3", "model_kind": "snsaug_finetuned_checkpoint", "model_path": str(medium)},
        ],
        "visual_gallery_model_id": "snsaug_150x3",
        "approved_model_roots": [str(root / "models"), str(root / "ckpts")],
        "approved_pair_roots": [str(root / "eval")],
        "approved_output_roots": [str(root / "out")],
        "output_root": str(root / "out" / "comparison"),
        "eval_subset_only": True,
        "worst_sample_count": 2,
        "no_training": True,
        "no_finetune": True,
        "no_network": True,
        "no_download": True,
    }


def test_validator_rejects_training_eval_input_and_repo_output() -> None:
    root = temp_root("cvf_0061_guard_")
    cfg = safe_config(root)
    cfg["pair_root"] = str(root / "eval" / "train_manifest.jsonl")
    errors = validate_snsaug_v2_checkpoint_comparison_eval_config(cfg, require_exists=False)
    assert_true(any("training" in error for error in errors), "training pair root rejected")
    cfg = safe_config(root)
    cfg["output_root"] = str(REPO_ROOT / "eval_out")
    errors = validate_snsaug_v2_checkpoint_comparison_eval_config(cfg, require_exists=False)
    assert_true(any("outside repository" in error for error in errors), "repo output rejected")
    cfg = safe_config(root)
    cfg["models"][1]["pair_root"] = str(root / "eval" / "other_fixed_pairs")
    errors = validate_snsaug_v2_checkpoint_comparison_eval_config(cfg, require_exists=False)
    assert_true(any("same pair_root" in error for error in errors), "same pair root enforced")


def test_evaluator_outputs_metrics_and_no_training_flags() -> None:
    root = temp_root("cvf_0061_eval_")
    cfg = safe_config(root)
    errors = validate_snsaug_v2_checkpoint_comparison_eval_config(cfg, require_exists=True)
    assert_equal(errors, [], "fixture config validates")
    summary = run_snsaug_v2_checkpoint_comparison_eval(cfg)
    assert_equal(summary["marker"], MARKER, "marker")
    assert_true(summary["eval_subset_only"] is True, "subset flag")
    assert_true(summary["full_fixed_pair_evaluation_ran"] is False, "full flag false for subset")
    assert_true(summary["no_training"] is True, "no training")
    assert_true(summary["no_finetune"] is True, "no finetune")
    expected = {
        "model_eval_records",
        "model_eval_comparisons",
        "per_model_per_profile_metrics",
        "robustness_drop_by_model",
        "checkpoint_comparison_summary",
        "checkpoint_comparison_report",
        "worst_samples_by_model",
        "visual_gallery_manifest",
        "artifact_manifest",
    }
    assert_equal(set(summary["output_paths"]), expected, "all outputs written")
    for path in summary["output_paths"].values():
        assert_true(Path(path).exists(), f"{path} exists")
    metrics = json.loads(Path(summary["output_paths"]["per_model_per_profile_metrics"]).read_text(encoding="utf-8"))
    for model_id in ("baseline", "snsaug_30x3", "snsaug_150x3"):
        assert_true("clean" in metrics["metrics"][model_id], "clean profile metrics")
        assert_true("resize_crop_pad" in metrics["metrics"][model_id], "sns profile metrics")
        sns_metrics = metrics["metrics"][model_id]["resize_crop_pad"]
        assert_true(sns_metrics["synthetic_to_real_confusion"] is not None, "synthetic_to_real_confusion non-null")
        assert_true(sns_metrics["synthetic_to_tampered_confusion"] is not None, "synthetic_to_tampered_confusion non-null")
    artifact = json.loads(Path(summary["output_paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert_true(artifact["same_pair_root_for_all_models"] is True, "same pair root recorded")
    comparison = json.loads(Path(summary["output_paths"]["checkpoint_comparison_summary"]).read_text(encoding="utf-8"))
    names = {item["comparison"] for item in comparison["comparisons"]}
    assert_true("baseline_vs_snsaug_30x3" in names, "baseline vs 30x3 comparison")
    assert_true("baseline_vs_snsaug_150x3" in names, "baseline vs 150x3 comparison")
    assert_true("snsaug_30x3_vs_snsaug_150x3" in names, "30x3 vs 150x3 comparison")
    gallery = json.loads(Path(summary["output_paths"]["visual_gallery_manifest"]).read_text(encoding="utf-8"))
    assert_equal(gallery["model_id"], "snsaug_150x3", "150x3 gallery")


def main() -> int:
    tests = [
        test_validator_rejects_training_eval_input_and_repo_output,
        test_evaluator_outputs_metrics_and_no_training_flags,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
