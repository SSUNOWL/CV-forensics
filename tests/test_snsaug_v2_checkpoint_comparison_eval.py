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
    SNSAugV2CheckpointComparisonEvalError,
    run_snsaug_v2_checkpoint_comparison_eval,
    sanity_check_outputs,
    validate_snsaug_v2_checkpoint_comparison_eval_config,
)
from cv_forensics.pre_sns_v3_model import build_pre_sns_v3_model  # noqa: E402
from cv_forensics.pre_sns_v3_tile_localizer_v2_model import build_pre_sns_v3_tile_localizer_v2  # noqa: E402


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def temp_root(prefix: str) -> Path:
    TMP_PARENT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(TMP_PARENT)))


def _write_image_and_masks(pair_root: Path, base_id: str) -> tuple[str, str, str]:
    from PIL import Image

    image_path = pair_root / f"{base_id}.png"
    mask_path = pair_root / f"{base_id}_tamper.png"
    ignore_path = pair_root / f"{base_id}_ignore.png"
    image = Image.new("RGB", (16, 16), (80, 100, 130))
    mask = Image.new("L", (16, 16), 0)
    ignore = Image.new("L", (16, 16), 0)
    for y in range(4, 12):
        for x in range(4, 12):
            mask.putpixel((x, y), 255)
    for y in range(0, 4):
        for x in range(0, 4):
            ignore.putpixel((x, y), 255)
    image.save(image_path)
    mask.save(mask_path)
    ignore.save(ignore_path)
    return str(image_path), str(mask_path), str(ignore_path)


def _constant_v3_checkpoint(path: Path, class_bias: list[float], tamper_bias: list[float]) -> dict[str, object]:
    import torch

    model = build_pre_sns_v3_model(torch, base_channels=2)
    for param in model.parameters():
        param.data.zero_()
    model.class_head[2].bias.data.copy_(torch.tensor(class_bias))
    model.tamper_binary_head[2].bias.data.copy_(torch.tensor(tamper_bias))
    model.family_head[2].bias.data.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
    model.mask_head.bias.data.fill_(-6.0)
    payload = {
        "model_name": "pre_sns_v3",
        "image_size": 16,
        "base_channels": 2,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    return payload


def _tile_checkpoint(path: Path) -> None:
    import torch

    model = build_pre_sns_v3_tile_localizer_v2(torch, tile_size=16, base_channels=1, input_feature_mode="rgb_only")
    for param in model.parameters():
        param.data.zero_()
    model.mask_head.bias.data.fill_(-6.0)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tile_size": 16,
            "input_feature_mode": "rgb_only",
            "boundary_head": True,
            "confidence_head": True,
        },
        path,
    )


def write_fixture(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    pair_root = root / "eval" / "snsaug_v2_0058c_fixed_pairs"
    model_root = root / "models"
    ckpt_root = root / "ckpts"
    for path in (pair_root, model_root, ckpt_root):
        path.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in ("real", "synthetic", "tampered"):
        base_id = f"{label}_001"
        image_path, mask_path, ignore_path = _write_image_and_masks(pair_root, base_id)
        common = {
            "image_path": image_path,
            "tamper_mask_path": mask_path if label == "tampered" else None,
            "ignore_mask_path": ignore_path,
        }
        rows.append({"base_id": base_id, "view": "clean", "profile": "clean", "content_label": label, **common})
        rows.append(
            {
                "base_id": base_id,
                "view": "sns_aug",
                "profile": "resize_crop_pad",
                "content_label": label,
                **common,
            }
        )
    meta = pair_root / "meta.jsonl"
    with open(meta, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    long = model_root / "long256.pt"
    tile = model_root / "tile_v2.pt"
    base_payload = _constant_v3_checkpoint(long, [4.0, -1.0, -1.0], [4.0, -1.0])
    _tile_checkpoint(tile)
    bundle = model_root / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "long256_checkpoint_path": str(long),
                "tile_v2_checkpoint_path": str(tile),
                "policy_gated_report": {"mask_threshold": 0.45, "tile_size": 16, "tile_stride": 16},
            }
        ),
        encoding="utf-8",
    )
    short = ckpt_root / "short.pt"
    medium = ckpt_root / "medium.pt"
    for path, class_bias, tamper_bias in (
        (short, [-1.0, 4.0, -1.0], [4.0, -1.0]),
        (medium, [-1.0, -1.0, 4.0], [-1.0, 4.0]),
    ):
        payload = _constant_v3_checkpoint(ckpt_root / f"{path.stem}_source_long.pt", class_bias, tamper_bias)
        payload = {
            "marker": "dummy",
            "model_version": "snsaug_aware_multihead_forensics_v1",
            "base_model_bundle_metadata": json.loads(bundle.read_text(encoding="utf-8")),
            "long256_model_state_dict": payload["model_state_dict"],
        }
        import torch
        torch.save(payload, path)
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
            {"model_id": "pre_sns_baseline", "model_kind": "pre_sns_bundle", "model_path": str(bundle)},
            {"model_id": "snsaug_guarded_short_30x3", "model_kind": "snsaug_finetuned_checkpoint", "model_path": str(short)},
            {"model_id": "snsaug_medium_150x3", "model_kind": "snsaug_finetuned_checkpoint", "model_path": str(medium)},
        ],
        "visual_gallery_model_id": "snsaug_medium_150x3",
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
    for model_id in ("pre_sns_baseline", "snsaug_guarded_short_30x3", "snsaug_medium_150x3"):
        assert_true("clean" in metrics["metrics"][model_id], "clean profile metrics")
        assert_true("resize_crop_pad" in metrics["metrics"][model_id], "sns profile metrics")
        sns_metrics = metrics["metrics"][model_id]["resize_crop_pad"]
        assert_true(sns_metrics["synthetic_to_real_confusion"] is not None, "synthetic_to_real_confusion non-null")
        assert_true(sns_metrics["synthetic_to_tampered_confusion"] is not None, "synthetic_to_tampered_confusion non-null")
    records = [json.loads(line) for line in Path(summary["output_paths"]["model_eval_records"]).read_text(encoding="utf-8").splitlines()]
    assert_true(all("p_tampered" in row and "pred_class" in row for row in records), "records contain probabilities and predictions")
    baseline_preds = [row["pred_class"] for row in records if row["model_id"] == "pre_sns_baseline"]
    medium_preds = [row["pred_class"] for row in records if row["model_id"] == "snsaug_medium_150x3"]
    assert_true(baseline_preds != medium_preds, "dummy checkpoint changes predictions")
    artifact = json.loads(Path(summary["output_paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert_true(artifact["same_pair_root_for_all_models"] is True, "same pair root recorded")
    fine_infos = [item for item in artifact["model_checkpoint_info"] if item["model_kind"] == "snsaug_finetuned_checkpoint"]
    assert_true(all(item.get("checkpoint_sha256") for item in fine_infos), "fine checkpoints record sha256")
    comparison = json.loads(Path(summary["output_paths"]["checkpoint_comparison_summary"]).read_text(encoding="utf-8"))
    names = {item["comparison"] for item in comparison["comparisons"]}
    assert_true("pre_sns_baseline_vs_snsaug_guarded_short_30x3" in names, "baseline vs 30x3 comparison")
    assert_true("pre_sns_baseline_vs_snsaug_medium_150x3" in names, "baseline vs 150x3 comparison")
    assert_true("snsaug_guarded_short_30x3_vs_snsaug_medium_150x3" in names, "30x3 vs 150x3 comparison")
    gallery = json.loads(Path(summary["output_paths"]["visual_gallery_manifest"]).read_text(encoding="utf-8"))
    assert_equal(gallery["model_id"], "snsaug_medium_150x3", "150x3 gallery")


def test_invalid_checkpoint_path_fails() -> None:
    root = temp_root("cvf_0061_bad_ckpt_")
    cfg = safe_config(root)
    cfg["models"][1]["model_path"] = str(root / "ckpts" / "missing.pt")
    errors = validate_snsaug_v2_checkpoint_comparison_eval_config(cfg, require_exists=True)
    assert_true(any("does not exist" in error for error in errors), "missing checkpoint rejected")


def test_sanity_rejects_all_one_metrics() -> None:
    try:
        sanity_check_outputs(
            records=[{"model_id": "m", "p_tampered": 0.1, "pred_class": "real"}],
            per_profile={"m": {"clean": {"accuracy": 1.0, "sample_count": 1}}},
            summary={"comparisons": [{"comparison": "x"}]},
            model_infos=[],
        )
    except SNSAugV2CheckpointComparisonEvalError as exc:
        assert_true("all model/profile accuracy" in str(exc), "all-one metrics rejected")
    else:
        raise AssertionError("all-one metrics should fail sanity check")


def main() -> int:
    tests = [
        test_validator_rejects_training_eval_input_and_repo_output,
        test_evaluator_outputs_metrics_and_no_training_flags,
        test_invalid_checkpoint_path_fails,
        test_sanity_rejects_all_one_metrics,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
