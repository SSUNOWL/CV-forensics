#!/usr/bin/env python3
"""Plain Python tests for SNSAug V2 small benchmark class balance and masks."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TMP_PARENT = Path("/home/rlatjswo/.codex/memories")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2 import SNSAugV2PairGenerator  # noqa: E402
from cv_forensics.snsaug_v2.pair_generator import SNSAugV2PairGenerationError  # noqa: E402
from cv_forensics.snsaug_v2_fixed_pairs_eval import aggregate_per_profile, render_small_benchmark_summary  # noqa: E402


def _load_checker():
    path = REPO_ROOT / "scripts" / "snsaug_v2" / "check_snsaug_v2_pair_dataset.py"
    spec = importlib.util.spec_from_file_location("check_snsaug_v2_pair_dataset", path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def temp_root(prefix: str) -> Path:
    TMP_PARENT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(TMP_PARENT)))


def fixture_image_and_mask() -> tuple[Image.Image, Image.Image]:
    image = Image.new("RGB", (64, 48), (40, 50, 60))
    ImageDraw.Draw(image).rectangle((20, 12, 44, 34), fill=(220, 40, 40))
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((20, 12, 44, 34), fill=255)
    return image, mask


def write_manifest(root: Path, labels: list[str], *, tampered_mask: bool = True) -> Path:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    image, mask = fixture_image_and_mask()
    rows: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        image_path = inputs / f"{label}_{index}.png"
        image.save(image_path)
        row: dict[str, object] = {
            "base_id": f"{label}_{index}",
            "image_path": str(image_path),
            "content_label": label,
            "split": "validation",
            "source_dataset": "fixture",
        }
        if label == "tampered" and tampered_mask:
            mask_path = inputs / f"{label}_{index}_mask.png"
            mask.save(mask_path)
            row["tamper_mask_path"] = str(mask_path)
        rows.append(row)
    manifest_path = inputs / "manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")
    return manifest_path


def test_class_balanced_selection_and_summary_files() -> None:
    root = temp_root("cvf_balance_ok_")
    manifest_path = write_manifest(root, ["real", "real", "synthetic", "synthetic", "tampered", "tampered"])
    out_root = root / "pairs"
    SNSAugV2PairGenerator(
        source_manifest_path=manifest_path,
        output_root=out_root,
        profiles=["combined_sns_realistic"],
        severity="medium",
        seed=5,
        max_samples=99,
        max_samples_per_class=1,
    ).run()
    class_summary = json.loads((out_root / "class_balance_summary.json").read_text(encoding="utf-8"))
    mask_summary = json.loads((out_root / "mask_availability_summary.json").read_text(encoding="utf-8"))
    assert_equal(class_summary["requested_per_class"], 1, "requested per class")
    assert_equal(class_summary["selected_per_class"], {"real": 1, "synthetic": 1, "tampered": 1}, "selected per class")
    assert_equal(class_summary["expected_record_count"], class_summary["actual_record_count"], "expected rows")
    assert_equal(mask_summary["tampered_with_valid_mask_count"], 2, "tampered masks available")


def test_missing_tampered_class_fails_by_default() -> None:
    root = temp_root("cvf_balance_missing_tampered_")
    manifest_path = write_manifest(root, ["real", "synthetic"])
    try:
        SNSAugV2PairGenerator(
            source_manifest_path=manifest_path,
            output_root=root / "pairs",
            profiles=["combined_sns_realistic"],
            max_samples=99,
            max_samples_per_class=1,
        ).run()
    except SNSAugV2PairGenerationError as exc:
        assert_true("tampered class count is zero" in str(exc), "missing tampered error")
        return
    raise AssertionError("missing tampered class should fail")


def test_tampered_without_mask_fails_by_default() -> None:
    root = temp_root("cvf_balance_missing_mask_")
    manifest_path = write_manifest(root, ["real", "synthetic", "tampered"], tampered_mask=False)
    try:
        SNSAugV2PairGenerator(
            source_manifest_path=manifest_path,
            output_root=root / "pairs",
            profiles=["combined_sns_realistic"],
            max_samples=99,
            max_samples_per_class=1,
        ).run()
    except SNSAugV2PairGenerationError as exc:
        assert_true("tampered_with_valid_mask_count is zero" in str(exc), "missing mask error")
        return
    raise AssertionError("tampered without mask should fail")


def test_pair_dataset_sanity_checker_ok_and_duplicate_clean_failure() -> None:
    root = temp_root("cvf_balance_checker_")
    manifest_path = write_manifest(root, ["real", "synthetic", "tampered"])
    out_root = root / "pairs"
    SNSAugV2PairGenerator(
        source_manifest_path=manifest_path,
        output_root=out_root,
        profiles=["combined_sns_realistic"],
        max_samples=99,
        max_samples_per_class=1,
    ).run()
    checker = _load_checker()
    summary, errors, _warnings = checker.check_pair_dataset(out_root)
    assert_true(summary["ok"], "checker ok")
    assert_equal(errors, [], "no checker errors")
    rows = [json.loads(line) for line in (out_root / "meta.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    duplicate = dict(next(row for row in rows if row["view"] == "clean"))
    with open(out_root / "meta.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(duplicate, ensure_ascii=True) + "\n")
    _summary, errors, _warnings = checker.check_pair_dataset(out_root)
    assert_true(any("expected exactly one clean row" in error for error in errors), "duplicate clean detected")


def test_evaluator_denominator_reporting_and_na_rendering() -> None:
    records = [
        {
            "base_id": "real_1",
            "content_label": "real",
            "view": "clean",
            "profile": "clean",
            "pred_class": "real",
            "class_correct": True,
            "final_mask_area_pct": 0.0,
            "latency_ms": 10.0,
        }
    ]
    metrics = aggregate_per_profile(records)
    clean = metrics["clean"]
    assert_equal(clean["class_count_real"], 1, "real denominator")
    assert_equal(clean["class_count_synthetic"], 0, "synthetic denominator")
    assert_equal(clean["class_count_tampered"], 0, "tampered denominator")
    assert_equal(clean["tampered_mask_eval_count"], 0, "mask denominator")
    assert_equal(clean["localization_activation_denominator"], 0, "activation denominator")
    assert_equal(clean["synthetic_recall"], None, "empty synthetic recall is NA")
    assert_equal(clean["tampered_recall"], None, "empty tampered recall is NA")
    assert_equal(clean["localization_activation_recall"], None, "empty activation recall is NA")
    rendered = render_small_benchmark_summary(
        {"marker": "fixture", "record_count": 1, "comparison_count": 0, "warning_count": 0, "profiles": ["clean"]},
        metrics,
        {},
        {"main_collapse_cause": "unclear", "notes": []},
    )
    assert_true(" NA " in rendered or "| NA |" in rendered, "summary renders NA")


def main() -> int:
    tests = [
        test_class_balanced_selection_and_summary_files,
        test_missing_tampered_class_fails_by_default,
        test_tampered_without_mask_fails_by_default,
        test_pair_dataset_sanity_checker_ok_and_duplicate_clean_failure,
        test_evaluator_denominator_reporting_and_na_rendering,
    ]
    for test in tests:
        test()
    print("SNSAUG_V2_SMALL_BENCHMARK_CLASS_BALANCE_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
