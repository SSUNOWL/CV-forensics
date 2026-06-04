#!/usr/bin/env python3
"""Plain Python tests for snsaug v2 training manifest and wrapper."""

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

from cv_forensics.snsaug_v2_dataset_wrapper import (  # noqa: E402
    SNSAugV2DatasetWrapper,
    masked_localization_loss,
)
from cv_forensics.snsaug_v2_training_manifest import (  # noqa: E402
    DEFAULT_CLASS_BALANCE,
    MARKER,
    build_snsaug_v2_training_manifest,
    validate_snsaug_v2_training_manifest_config,
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
    from PIL import Image, ImageDraw

    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    samples = []
    mining_rows = []
    for sample_id, split, label, family in (
        ("train_anchor", "train", "tampered", ""),
        ("train_fragile", "train", "tampered", "LatDiff"),
        ("train_clean_fail", "train", "tampered", ""),
        ("train_real", "train", "real", ""),
        ("train_synth", "train", "synthetic", "GAN"),
        ("val_bad", "val", "tampered", ""),
    ):
        image_path = inputs / f"{sample_id}.png"
        image = Image.new("RGB", (20, 12), (60, 70, 80))
        ImageDraw.Draw(image).rectangle((5, 3, 12, 9), fill=(220, 30, 30))
        image.save(image_path)
        sample = {
            "base_id": sample_id,
            "sample_id": sample_id,
            "split": split,
            "image_path": str(image_path),
            "content_label": label,
            "source_dataset": "fixture",
            "family_label": family,
        }
        if label == "tampered":
            mask_path = inputs / f"{sample_id}_mask.png"
            mask = Image.new("L", (20, 12), 0)
            ImageDraw.Draw(mask).rectangle((5, 3, 12, 9), fill=255)
            mask.save(mask_path)
            sample["mask_path"] = str(mask_path)
        samples.append(sample)
    mining_rows.extend(
        [
            {"base_id": "train_anchor", "split": "train", "groups": ["stable_correct_anchor"]},
            {"base_id": "train_fragile", "split": "train", "groups": ["fragile_correct_to_fail"]},
            {"base_id": "train_clean_fail", "split": "train", "groups": ["clean_fail"]},
            {"base_id": "train_real", "split": "train", "groups": []},
            {"base_id": "train_synth", "split": "train", "groups": []},
            {"base_id": "val_bad", "split": "val", "groups": ["fragile_correct_to_fail"]},
        ]
    )
    manifest_path = inputs / "train_manifest.json"
    mining_path = inputs / "train_mining.jsonl"
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    with open(mining_path, "w", encoding="utf-8") as handle:
        for row in mining_rows:
            handle.write(json.dumps(row) + "\n")
    return manifest_path, mining_path


def safe_config(root: Path, manifest_path: Path, mining_path: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "config_kind": "approved_snsaug_v2_training_manifest",
        "execution_mode": "approved_local_snsaug_v2_training_manifest",
        "train_manifest_path": str(manifest_path),
        "train_mining_manifest_path": str(mining_path),
        "approved_input_roots": [str(root / "inputs")],
        "approved_output_roots": [str(root / "reports")],
        "output_root": str(root / "reports" / "snsaug_train_manifest"),
        "seed": 53,
        "profiles": ["clean", "jpeg_resize", "tiktok_like", "combined_sns_realistic"],
        "clean_weight": 0.3,
        "basic_aug_weight": 0.3,
        "sns_aug_weight": 0.4,
        "no_training": True,
        "no_download": True,
        "no_network": True,
    }


def test_training_manifest_excludes_val_test_split() -> None:
    root = temp_root("cvf_snsaug_train_manifest_")
    manifest_path, mining_path = write_fixture_inputs(root)
    summary = build_snsaug_v2_training_manifest(safe_config(root, manifest_path, mining_path))
    out_path = Path(summary["output_paths"]["training_manifest"])
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert_true(all(row["split"] == "train" for row in rows), "all rows train only")
    assert_true(all(row["base_id"] != "val_bad" for row in rows), "val excluded")


def test_group_sampling_weights_are_correct() -> None:
    root = temp_root("cvf_snsaug_train_weights_")
    manifest_path, mining_path = write_fixture_inputs(root)
    summary = build_snsaug_v2_training_manifest(safe_config(root, manifest_path, mining_path))
    rows = [json.loads(line) for line in Path(summary["output_paths"]["training_manifest"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["base_id"]: row for row in rows}
    assert_equal(round(by_id["train_anchor"]["sampling_weight"], 3), round(DEFAULT_CLASS_BALANCE["tampered"] * 0.40, 3), "stable weight")
    assert_equal(round(by_id["train_fragile"]["sampling_weight"], 3), round(DEFAULT_CLASS_BALANCE["tampered"] * 0.35, 3), "fragile weight")
    assert_equal(round(by_id["train_clean_fail"]["sampling_weight"], 3), round(DEFAULT_CLASS_BALANCE["tampered"] * 0.15, 3), "clean fail weight")


def test_dataset_wrapper_returns_zero_ignore_mask_for_clean() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (16, 10), (30, 40, 50))
    mask = Image.new("L", (16, 10), 0)
    ImageDraw.Draw(mask).rectangle((4, 2, 8, 6), fill=255)
    dataset = [{"base_id": "a", "image": image, "tamper_mask": mask, "content_label": "tampered", "recommended_profiles": ["clean"]}]
    wrapper = SNSAugV2DatasetWrapper(dataset, seed=3, clean_weight=1.0, basic_aug_weight=0.0, sns_aug_weight=0.0)
    wrapper.set_epoch(1)
    item = wrapper[0]
    assert_true(item["view"] == "clean", "clean selected early")
    assert_true(max(item["ignore_mask"].getdata()) == 0, "clean ignore zero")


def test_dataset_wrapper_returns_nonzero_ignore_mask_for_overlay() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (16, 10), (30, 40, 50))
    mask = Image.new("L", (16, 10), 0)
    ImageDraw.Draw(mask).rectangle((4, 2, 8, 6), fill=255)
    dataset = [{"base_id": "b", "image": image, "tamper_mask": mask, "content_label": "tampered", "recommended_profiles": ["tiktok_like"]}]
    wrapper = SNSAugV2DatasetWrapper(dataset, seed=5, clean_weight=0.0, basic_aug_weight=0.0, sns_aug_weight=1.0, sns_profiles=["tiktok_like"])
    wrapper.set_epoch(9)
    item = wrapper[0]
    assert_true(item["view"] == "sns_aug", "sns selected")
    assert_true(max(item["ignore_mask"].getdata()) > 0, "overlay ignore nonzero")


def test_labels_are_preserved() -> None:
    from PIL import Image

    dataset = [{"base_id": "c", "image": Image.new("RGB", (10, 10), (0, 0, 0)), "content_label": "real", "recommended_profiles": ["annotation_sticker"]}]
    wrapper = SNSAugV2DatasetWrapper(dataset, seed=7, clean_weight=0.0, basic_aug_weight=0.0, sns_aug_weight=1.0, sns_profiles=["annotation_sticker"])
    wrapper.set_epoch(9)
    item = wrapper[0]
    assert_equal(item["label"], "real", "label preserved")
    assert_true(item["aug_meta"]["label_preserved"], "meta label preserved")


def test_masked_localization_loss_ignores_ignore_mask_regions() -> None:
    value = masked_localization_loss(
        [0.9, 0.1],
        [1.0, 1.0],
        [0.0, 1.0],
        reduction="mean",
    )
    assert_true(float(value) < 0.2, "ignored region excluded from loss")


def test_family_loss_mask_works() -> None:
    from PIL import Image

    dataset = [
        {"base_id": "x", "image": Image.new("RGB", (8, 8), (0, 0, 0)), "content_label": "synthetic", "family_label": "GAN"},
        {"base_id": "y", "image": Image.new("RGB", (8, 8), (0, 0, 0)), "content_label": "tampered"},
    ]
    wrapper = SNSAugV2DatasetWrapper(dataset, seed=9)
    assert_equal(wrapper[0]["family_loss_mask"], 1, "family label contributes")
    assert_equal(wrapper[1]["family_loss_mask"], 0, "missing family label masked")


def test_validator_guardrails() -> None:
    root = temp_root("cvf_snsaug_train_validator_")
    manifest_path, mining_path = write_fixture_inputs(root)
    cfg = safe_config(root, manifest_path, mining_path)
    assert_equal(validate_snsaug_v2_training_manifest_config(cfg, require_exists=False), [], "safe config")
    unsafe = dict(cfg)
    unsafe["output_root"] = str(REPO_ROOT / "train_manifest")
    assert_true(validate_snsaug_v2_training_manifest_config(unsafe, require_exists=False), "repo output rejected")


def main() -> int:
    tests = [
        test_training_manifest_excludes_val_test_split,
        test_group_sampling_weights_are_correct,
        test_dataset_wrapper_returns_zero_ignore_mask_for_clean,
        test_dataset_wrapper_returns_nonzero_ignore_mask_for_overlay,
        test_labels_are_preserved,
        test_masked_localization_loss_ignores_ignore_mask_regions,
        test_family_loss_mask_works,
        test_validator_guardrails,
    ]
    for test in tests:
        test()
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
