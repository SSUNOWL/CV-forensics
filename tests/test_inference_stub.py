"""Tests for task 0008 lightweight inference stub.

Runnable directly:
    python3 tests/test_inference_stub.py

Pytest-compatible:
    pytest -q tests/test_inference_stub.py

Uses only the Python standard library. No third-party imports.
"""
from __future__ import annotations

# Allow direct execution from the repository root without installing the package.
from pathlib import Path as _Path
import sys as _sys

_REPO_ROOT = next(
    (p for p in _Path(__file__).resolve().parents if (p / "src" / "cv_forensics").is_dir()),
    None,
)
if _REPO_ROOT is not None:
    _SRC_ROOT = _REPO_ROOT / "src"
    if str(_SRC_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_SRC_ROOT))

import json
import os
import sys

# Ensure the project src is importable when run directly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cv_forensics.inference_stub import (
    REQUIRED_SCENARIOS,
    SCENARIO_LOW_CONFIDENCE,
    SCENARIO_REAL_CLEAN,
    SCENARIO_SYNTHETIC_LATDIFF,
    SCENARIO_TAMPERED_BELOW_THRESHOLD,
    SCENARIO_TAMPERED_LOCALIZED,
    FakeInput,
    check_fake_input_config_safety,
    run_fake_batch,
    run_fake_inference,
)
from cv_forensics.model_output_schema import (
    CLASS_LABEL_REAL,
    CLASS_LABEL_SYNTHETIC,
    CLASS_LABEL_TAMPERED,
    FAMILY_LATDIFF,
    LOCALIZATION_ACTIVATED,
    LOCALIZATION_NOT_APPLICABLE,
    LOCALIZATION_SKIPPED_BELOW_THRESHOLD,
    PERTURBATION_NONE,
    validate_forensic_output,
)

_CONFIG_PATH = os.path.join(
    _PROJECT_ROOT, "configs", "inference", "fake_inputs.example.json"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def _make_fake_input(
    scenario: str,
    class_hint: str,
    family_hint: str,
    tampered_score: float,
    threshold_tau: float = 0.5,
    mask_area_hint_pct=None,
    perturbations=None,
    evidence_hints=None,
) -> FakeInput:
    return FakeInput(
        input_id=f"test_{scenario}",
        scenario=scenario,
        class_hint=class_hint,
        family_hint=family_hint,
        tampered_score=tampered_score,
        threshold_tau=threshold_tau,
        mask_area_hint_pct=mask_area_hint_pct,
        perturbations=perturbations or [PERTURBATION_NONE],
        evidence_hints=evidence_hints or [],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_config_loads():
    cfg = _load_config()
    assert cfg["dry_run"] is True
    assert cfg["no_download"] is True
    assert cfg["no_training"] is True
    assert cfg["no_network"] is True
    assert isinstance(cfg["fake_inputs"], list)
    assert len(cfg["fake_inputs"]) > 0


def test_all_required_scenarios_present():
    cfg = _load_config()
    present = {item["scenario"] for item in cfg["fake_inputs"]}
    for scenario in REQUIRED_SCENARIOS:
        assert scenario in present, f"Required scenario missing: {scenario}"


def test_fake_input_from_dict():
    cfg = _load_config()
    for raw in cfg["fake_inputs"]:
        fi = FakeInput.from_dict(raw)
        assert isinstance(fi.input_id, str)
        assert isinstance(fi.scenario, str)
        assert isinstance(fi.tampered_score, float)
        assert isinstance(fi.threshold_tau, float)


def test_run_fake_inference_returns_validated_output():
    cfg = _load_config()
    for raw in cfg["fake_inputs"]:
        fi = FakeInput.from_dict(raw)
        output = run_fake_inference(fi)
        # Must not raise
        validate_forensic_output(output)
        assert "class" in output
        assert "family" in output
        assert "localization_head" in output
        assert "reason" in output
        assert "evidence" in output
        assert "perturbations" in output
        assert "tampered_score" in output
        assert "threshold_tau" in output


def test_real_clean_class_and_localization():
    fi = _make_fake_input(
        scenario=SCENARIO_REAL_CLEAN,
        class_hint=CLASS_LABEL_REAL,
        family_hint="Real-or-N/A",
        tampered_score=0.05,
    )
    output = run_fake_inference(fi)
    assert output["class"] == CLASS_LABEL_REAL
    # Localization must not be activated for real
    assert output["localization_head"] != LOCALIZATION_ACTIVATED


def test_synthetic_latdiff_class_and_family():
    fi = _make_fake_input(
        scenario=SCENARIO_SYNTHETIC_LATDIFF,
        class_hint=CLASS_LABEL_SYNTHETIC,
        family_hint=FAMILY_LATDIFF,
        tampered_score=0.10,
    )
    output = run_fake_inference(fi)
    assert output["class"] == CLASS_LABEL_SYNTHETIC
    # Family confidence distribution must contain LatDiff
    assert FAMILY_LATDIFF in output["family_conf"]
    # LatDiff must have reasonable confidence when hinted
    assert output["family_conf"][FAMILY_LATDIFF] > 0.5


def test_tampered_localized_activates_localization():
    fi = _make_fake_input(
        scenario=SCENARIO_TAMPERED_LOCALIZED,
        class_hint=CLASS_LABEL_TAMPERED,
        family_hint=FAMILY_LATDIFF,
        tampered_score=0.82,
        threshold_tau=0.5,
        mask_area_hint_pct=11.2,
    )
    output = run_fake_inference(fi)
    assert output["class"] == CLASS_LABEL_TAMPERED
    assert output["localization_head"] == LOCALIZATION_ACTIVATED
    assert output["mask_area_pct"] is not None
    assert float(output["mask_area_pct"]) > 0.0


def test_tampered_below_threshold_skips_localization():
    fi = _make_fake_input(
        scenario=SCENARIO_TAMPERED_BELOW_THRESHOLD,
        class_hint=CLASS_LABEL_TAMPERED,
        family_hint="GAN",
        tampered_score=0.35,
        threshold_tau=0.5,
    )
    output = run_fake_inference(fi)
    assert output["class"] == CLASS_LABEL_TAMPERED
    assert output["localization_head"] == LOCALIZATION_SKIPPED_BELOW_THRESHOLD
    assert output["mask_area_pct"] is None


def test_reasons_are_deterministic_and_non_empty():
    cfg = _load_config()
    for raw in cfg["fake_inputs"]:
        fi = FakeInput.from_dict(raw)
        output1 = run_fake_inference(fi)
        output2 = run_fake_inference(fi)
        reason = output1.get("reason", "")
        assert isinstance(reason, str) and reason.strip(), (
            f"Reason is empty for scenario {raw['scenario']!r}"
        )
        assert output1["reason"] == output2["reason"], (
            f"Reason is not deterministic for scenario {raw['scenario']!r}"
        )


def test_evidence_signal_ids_present():
    cfg = _load_config()
    all_signal_ids = set()
    for raw in cfg["fake_inputs"]:
        fi = FakeInput.from_dict(raw)
        output = run_fake_inference(fi)
        for ev in output.get("evidence", []):
            all_signal_ids.add(ev.get("signal_id", ""))
    assert len(all_signal_ids) > 0, "No evidence signals were produced across all scenarios"


def test_required_output_fields_present():
    required = [
        "schema_version", "class", "class_conf", "family", "family_conf",
        "localization_head", "mask_area_pct", "threshold_tau", "tampered_score",
        "evidence", "perturbations", "reason",
    ]
    fi = _make_fake_input(
        scenario=SCENARIO_REAL_CLEAN,
        class_hint=CLASS_LABEL_REAL,
        family_hint="Real-or-N/A",
        tampered_score=0.05,
    )
    output = run_fake_inference(fi)
    for field in required:
        assert field in output, f"Required field missing from output: {field!r}"


def test_run_fake_batch():
    cfg = _load_config()
    fake_inputs = [FakeInput.from_dict(raw) for raw in cfg["fake_inputs"]]
    results = run_fake_batch(fake_inputs)
    assert len(results) == len(fake_inputs)
    for output in results:
        validate_forensic_output(output)


def test_protected_path_in_config_is_not_present():
    cfg = _load_config()
    protected = ("data", "datasets", "outputs", "checkpoints", "secrets")
    for raw in cfg["fake_inputs"]:
        raw_str = json.dumps(raw)
        for component in protected:
            # Check that no path-like reference (with slash) contains a protected component
            import re
            # Find /component/ or /component at end or start of a path segment
            pattern = r'[/\\]' + component + r'([/\\]|")'
            matches = re.findall(pattern, raw_str)
            assert not matches, (
                f"Fake input {raw.get('input_id')} contains protected path component "
                f"{component!r}"
            )


def test_no_url_in_config():
    cfg = _load_config()
    cfg_str = json.dumps(cfg)
    for prefix in ("http://", "https://", "s3://", "gs://", "hf://"):
        assert prefix not in cfg_str, f"Config contains URL-like reference: {prefix}"


def test_no_output_file_written():
    import tempfile
    import os
    fi = _make_fake_input(
        scenario=SCENARIO_TAMPERED_LOCALIZED,
        class_hint=CLASS_LABEL_TAMPERED,
        family_hint=FAMILY_LATDIFF,
        tampered_score=0.75,
        threshold_tau=0.5,
        mask_area_hint_pct=8.0,
    )
    run_fake_inference(fi)
    # No output directory should have been created by inference
    assert not os.path.isdir(os.path.join(_PROJECT_ROOT, "outputs")), (
        "run_fake_inference must not create an outputs/ directory"
    )


# ---------------------------------------------------------------------------
# Config safety tests
# ---------------------------------------------------------------------------

def _base_raw() -> dict:
    """Minimal valid raw fake input record."""
    return {
        "input_id": "safety_test",
        "scenario": SCENARIO_REAL_CLEAN,
        "class_hint": CLASS_LABEL_REAL,
        "family_hint": "Real-or-N/A",
        "tampered_score": 0.05,
        "threshold_tau": 0.5,
        "mask_area_hint_pct": None,
        "perturbations": [PERTURBATION_NONE],
        "evidence_hints": [],
    }


def _assert_rejected(raw: dict, description: str) -> None:
    try:
        check_fake_input_config_safety(raw)
        assert False, f"Expected ValueError for {description}, but none was raised"
    except ValueError:
        pass


def _config_copy() -> dict:
    return json.loads(json.dumps(_load_config()))


def _assert_config_rejected(cfg: dict, description: str) -> None:
    try:
        check_fake_input_config_safety(cfg, context="fake_input_config")
        assert False, f"Expected ValueError for {description}, but none was raised"
    except ValueError:
        pass


def test_image_ref_data_rejected():
    raw = _base_raw()
    raw["image_ref"] = "data/foo.jpg"
    _assert_rejected(raw, "image_ref=data/foo.jpg")


def test_image_ref_dotslash_data_rejected():
    raw = _base_raw()
    raw["image_ref"] = "./data/foo.jpg"
    _assert_rejected(raw, "image_ref=./data/foo.jpg")


def test_mask_ref_outputs_rejected():
    raw = _base_raw()
    raw["mask_ref"] = "outputs/mask.png"
    _assert_rejected(raw, "mask_ref=outputs/mask.png")


def test_checkpoint_ref_checkpoints_rejected():
    raw = _base_raw()
    raw["checkpoint_ref"] = "checkpoints/model.pt"
    _assert_rejected(raw, "checkpoint_ref=checkpoints/model.pt")


def test_url_ref_https_rejected():
    raw = _base_raw()
    raw["url_ref"] = "https://example.com/image.jpg"
    _assert_rejected(raw, "url_ref=https://example.com/image.jpg")


def test_api_key_field_rejected():
    raw = _base_raw()
    raw["api_key"] = "abc123xyz"
    _assert_rejected(raw, "api_key field present")


def test_token_field_rejected():
    raw = _base_raw()
    raw["token"] = "xyztoken789"
    _assert_rejected(raw, "token field present")


def test_secret_like_value_rejected():
    raw = _base_raw()
    raw["extra_field"] = "my_secret_key_abc123"
    _assert_rejected(raw, "value containing 'secret'")


def test_abs_path_home_rejected():
    raw = _base_raw()
    raw["image_ref"] = "/home/example/file.jpg"
    _assert_rejected(raw, "image_ref=/home/example/file.jpg")


def test_windows_path_rejected():
    raw = _base_raw()
    raw["image_ref"] = "C:\\Users\\example\\file.jpg"
    _assert_rejected(raw, "image_ref=C:\\Users\\example\\file.jpg")


def test_example_config_passes_safety_check():
    cfg = _load_config()
    try:
        check_fake_input_config_safety(cfg, context="fake_input_config")
    except ValueError as exc:
        assert False, f"example config was unexpectedly rejected: {exc}"
    for raw in cfg["fake_inputs"]:
        try:
            check_fake_input_config_safety(raw, context=f"fake_input[{raw.get('input_id')}]")
        except ValueError as exc:
            assert False, (
                f"example config record {raw.get('input_id')!r} was unexpectedly rejected: {exc}"
            )


def test_top_level_dataset_ref_data_rejected():
    cfg = _config_copy()
    cfg["dataset_ref"] = "data/foo.jpg"
    _assert_config_rejected(cfg, "top-level dataset_ref=data/foo.jpg")


def test_top_level_dataset_ref_dotslash_data_rejected():
    cfg = _config_copy()
    cfg["dataset_ref"] = "./data/foo.jpg"
    _assert_config_rejected(cfg, "top-level dataset_ref=./data/foo.jpg")


def test_top_level_url_ref_https_rejected():
    cfg = _config_copy()
    cfg["url_ref"] = "https://example.com/image.jpg"
    _assert_config_rejected(cfg, "top-level url_ref=https://example.com/image.jpg")


def test_top_level_checkpoint_ref_rejected():
    cfg = _config_copy()
    cfg["checkpoint_ref"] = "checkpoints/model.pt"
    _assert_config_rejected(cfg, "top-level checkpoint_ref=checkpoints/model.pt")


def test_top_level_api_key_rejected():
    cfg = _config_copy()
    cfg["api_key"] = "abc"
    _assert_config_rejected(cfg, "top-level api_key")


def test_top_level_auth_key_rejected():
    cfg = _config_copy()
    cfg["auth"] = "abc"
    _assert_config_rejected(cfg, "top-level auth key")


def test_top_level_auth_value_rejected():
    cfg = _config_copy()
    cfg["note"] = "auth"
    _assert_config_rejected(cfg, "top-level note=auth")


def test_nested_fake_input_image_ref_rejected():
    cfg = _config_copy()
    cfg["fake_inputs"][0]["image_ref"] = "data/foo.jpg"
    _assert_config_rejected(cfg, "nested fake_inputs[0].image_ref=data/foo.jpg")


def test_nested_fake_input_mask_ref_rejected():
    cfg = _config_copy()
    cfg["fake_inputs"][0]["mask_ref"] = "outputs/mask.png"
    _assert_config_rejected(cfg, "nested fake_inputs[0].mask_ref=outputs/mask.png")


def test_nested_fake_input_checkpoint_ref_rejected():
    cfg = _config_copy()
    cfg["fake_inputs"][0]["checkpoint_ref"] = "checkpoints/model.pt"
    _assert_config_rejected(
        cfg,
        "nested fake_inputs[0].checkpoint_ref=checkpoints/model.pt",
    )


def test_nested_fake_input_url_ref_rejected():
    cfg = _config_copy()
    cfg["fake_inputs"][0]["url_ref"] = "https://example.com/image.jpg"
    _assert_config_rejected(
        cfg,
        "nested fake_inputs[0].url_ref=https://example.com/image.jpg",
    )


def test_authoritative_prose_is_not_secret_false_positive():
    cfg = _config_copy()
    cfg["description"] = "authoritative dry-run note"
    try:
        check_fake_input_config_safety(cfg, context="fake_input_config")
    except ValueError as exc:
        assert False, f"ordinary authoritative prose was unexpectedly rejected: {exc}"


def test_authentication_prose_is_not_secret_false_positive():
    cfg = _config_copy()
    cfg["description"] = "authentication policy is documented"
    try:
        check_fake_input_config_safety(cfg, context="fake_input_config")
    except ValueError as exc:
        assert False, f"ordinary authentication prose was unexpectedly rejected: {exc}"


# ---------------------------------------------------------------------------
# Direct runner
# ---------------------------------------------------------------------------

def _run_all() -> None:
    tests = [
        test_config_loads,
        test_all_required_scenarios_present,
        test_fake_input_from_dict,
        test_run_fake_inference_returns_validated_output,
        test_real_clean_class_and_localization,
        test_synthetic_latdiff_class_and_family,
        test_tampered_localized_activates_localization,
        test_tampered_below_threshold_skips_localization,
        test_reasons_are_deterministic_and_non_empty,
        test_evidence_signal_ids_present,
        test_required_output_fields_present,
        test_run_fake_batch,
        test_protected_path_in_config_is_not_present,
        test_no_url_in_config,
        test_no_output_file_written,
        # Config safety tests
        test_image_ref_data_rejected,
        test_image_ref_dotslash_data_rejected,
        test_mask_ref_outputs_rejected,
        test_checkpoint_ref_checkpoints_rejected,
        test_url_ref_https_rejected,
        test_api_key_field_rejected,
        test_token_field_rejected,
        test_secret_like_value_rejected,
        test_abs_path_home_rejected,
        test_windows_path_rejected,
        test_example_config_passes_safety_check,
        test_top_level_dataset_ref_data_rejected,
        test_top_level_dataset_ref_dotslash_data_rejected,
        test_top_level_url_ref_https_rejected,
        test_top_level_checkpoint_ref_rejected,
        test_top_level_api_key_rejected,
        test_top_level_auth_key_rejected,
        test_top_level_auth_value_rejected,
        test_nested_fake_input_image_ref_rejected,
        test_nested_fake_input_mask_ref_rejected,
        test_nested_fake_input_checkpoint_ref_rejected,
        test_nested_fake_input_url_ref_rejected,
        test_authoritative_prose_is_not_secret_false_positive,
        test_authentication_prose_is_not_secret_false_positive,
    ]
    passed = 0
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
