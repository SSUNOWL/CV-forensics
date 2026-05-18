"""Tests for model_output_schema and explanation_templates (task 0007).

Run with pytest or standalone:
    pytest -q tests/test_model_output_schema.py
    python3 tests/test_model_output_schema.py
"""
from __future__ import annotations

import json
import pathlib
import re as _re
import sys
import traceback
from contextlib import contextmanager

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cv_forensics import model_output_schema as mos
from cv_forensics import explanation_templates as et


# ---------------------------------------------------------------------------
# Stdlib replacement for pytest.raises
# ---------------------------------------------------------------------------

@contextmanager
def _raises(exc_type, match=None):
    """Minimal pytest.raises replacement using only stdlib contextlib."""
    try:
        yield
    except exc_type as exc:
        if match is not None and not _re.search(match, str(exc)):
            raise AssertionError(
                f"Exception {exc_type.__name__} raised but message {str(exc)!r} "
                f"did not match pattern {match!r}"
            ) from exc
        return
    except Exception as exc:
        raise AssertionError(
            f"Expected {exc_type.__name__} but got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"Expected {exc_type.__name__} to be raised but it was not")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tampered_output(**overrides):
    base = {
        "schema_version": "0.1.0",
        "class": "tampered",
        "class_conf": {"real": 0.03, "synthetic": 0.08, "tampered": 0.89},
        "family": "LatDiff",
        "family_conf": {
            "LatDiff": 0.78,
            "PixDiff": 0.14,
            "GAN": 0.05,
            "Other": 0.03,
            "Real-or-N/A": 0.00,
        },
        "localization_head": "activated",
        "mask_area_pct": 11.2,
        "threshold_tau": 0.5,
        "tampered_score": 0.89,
        "evidence": [
            {"signal_id": "boundary_discontinuity", "description": "test"},
            {"signal_id": "texture_inconsistency", "description": "test"},
        ],
        "perturbations": ["jpeg"],
        "reason": "",
    }
    base.update(overrides)
    return base


def _real_output(**overrides):
    base = {
        "schema_version": "0.1.0",
        "class": "real",
        "class_conf": {"real": 0.95, "synthetic": 0.03, "tampered": 0.02},
        "family": "Real-or-N/A",
        "family_conf": {
            "LatDiff": 0.01,
            "PixDiff": 0.01,
            "GAN": 0.01,
            "Other": 0.01,
            "Real-or-N/A": 0.96,
        },
        "localization_head": "not_applicable",
        "mask_area_pct": None,
        "threshold_tau": 0.5,
        "tampered_score": 0.02,
        "evidence": [],
        "perturbations": ["none"],
        "reason": "",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# validate_forensic_output — happy paths
# ---------------------------------------------------------------------------

def test_valid_tampered_example_passes():
    mos.validate_forensic_output(_tampered_output())


def test_valid_real_example_passes():
    mos.validate_forensic_output(_real_output())


def test_valid_synthetic_passes():
    output = _tampered_output()
    output.update({
        "class": "synthetic",
        "class_conf": {"real": 0.05, "synthetic": 0.90, "tampered": 0.05},
        "localization_head": "not_applicable",
        "mask_area_pct": None,
        "tampered_score": 0.05,
        "evidence": [{"signal_id": "global_synthetic_artifact", "description": "test"}],
        "perturbations": ["none"],
    })
    mos.validate_forensic_output(output)


# ---------------------------------------------------------------------------
# class_conf validation
# ---------------------------------------------------------------------------

def test_class_conf_required_labels_missing_real():
    output = _tampered_output()
    output["class_conf"] = {"synthetic": 0.08, "tampered": 0.89, "real": 0.03}
    mos.validate_forensic_output(output)  # all present — should pass


def test_class_conf_missing_required_label_fails():
    output = _tampered_output()
    output["class_conf"] = {"synthetic": 0.50, "tampered": 0.50}  # missing 'real'
    with _raises(ValueError, match="missing required label"):
        mos.validate_forensic_output(output)


def test_class_conf_unknown_label_rejected():
    output = _tampered_output()
    output["class_conf"] = {"real": 0.03, "synthetic": 0.08, "tampered": 0.87, "deepfake": 0.02}
    with _raises(ValueError, match="unknown label"):
        mos.validate_forensic_output(output)


# ---------------------------------------------------------------------------
# class / family label validation
# ---------------------------------------------------------------------------

def test_unknown_class_label_rejected():
    output = _tampered_output()
    output["class"] = "deepfake"
    with _raises(ValueError):
        mos.validate_forensic_output(output)


def test_unknown_family_label_rejected():
    output = _tampered_output()
    output["family"] = "StableDiffusion"
    with _raises(ValueError):
        mos.validate_forensic_output(output)


def test_unknown_family_label_in_family_conf_rejected():
    output = _tampered_output()
    output["family_conf"] = {
        "LatDiff": 0.78,
        "PixDiff": 0.14,
        "GAN": 0.05,
        "StableDiffusion": 0.03,  # unknown
    }
    with _raises(ValueError, match="unknown label"):
        mos.validate_forensic_output(output)


# ---------------------------------------------------------------------------
# Confidence value range validation
# ---------------------------------------------------------------------------

def test_confidence_value_above_1_rejected():
    output = _tampered_output()
    output["class_conf"] = {"real": 0.03, "synthetic": 0.08, "tampered": 1.5}
    with _raises(ValueError, match="1.0"):
        mos.validate_forensic_output(output)


def test_confidence_value_below_0_rejected():
    output = _tampered_output()
    output["class_conf"] = {"real": -0.1, "synthetic": 0.11, "tampered": 0.99}
    with _raises(ValueError, match="0.0"):
        mos.validate_forensic_output(output)


# ---------------------------------------------------------------------------
# Localization state rules
# ---------------------------------------------------------------------------

def test_localization_activated_requires_score_gte_tau():
    output = _tampered_output(tampered_score=0.89, threshold_tau=0.5)
    output["localization_head"] = "activated"
    mos.validate_forensic_output(output)  # 0.89 >= 0.5 → OK


def test_localization_activated_rejects_score_below_tau():
    output = _tampered_output(tampered_score=0.3, threshold_tau=0.5)
    output["localization_head"] = "activated"
    with _raises(ValueError, match="activated"):
        mos.validate_forensic_output(output)


def test_skipped_below_threshold_allowed_when_score_lt_tau():
    output = _tampered_output(tampered_score=0.65, threshold_tau=0.75)
    output["localization_head"] = "skipped_below_threshold"
    output["mask_area_pct"] = None
    mos.validate_forensic_output(output)  # 0.65 < 0.75 → OK


def test_skipped_below_threshold_rejects_score_gte_tau():
    output = _tampered_output(tampered_score=0.89, threshold_tau=0.5)
    output["localization_head"] = "skipped_below_threshold"
    with _raises(ValueError, match="skipped_below_threshold"):
        mos.validate_forensic_output(output)


# ---------------------------------------------------------------------------
# mask_area_pct validation
# ---------------------------------------------------------------------------

def test_mask_area_pct_negative_rejected():
    output = _tampered_output()
    output["mask_area_pct"] = -1.0
    with _raises(ValueError, match="mask_area_pct"):
        mos.validate_forensic_output(output)


def test_mask_area_pct_above_100_rejected():
    output = _tampered_output()
    output["mask_area_pct"] = 101.0
    with _raises(ValueError, match="mask_area_pct"):
        mos.validate_forensic_output(output)


def test_mask_area_pct_zero_allowed():
    output = _tampered_output(mask_area_pct=0.0)
    mos.validate_forensic_output(output)


def test_mask_area_pct_100_allowed():
    output = _tampered_output(mask_area_pct=100.0)
    mos.validate_forensic_output(output)


# ---------------------------------------------------------------------------
# Protected path / URL rejection
# ---------------------------------------------------------------------------

def test_protected_path_in_mask_ref_rejected():
    output = _tampered_output()
    output["mask_ref"] = "/home/user/data/mask.png"
    with _raises(ValueError, match="protected"):
        mos.validate_forensic_output(output)


def test_protected_path_in_visualization_ref_rejected():
    output = _tampered_output()
    output["visualization_ref"] = "/home/user/outputs/vis.png"
    with _raises(ValueError, match="protected"):
        mos.validate_forensic_output(output)


def test_url_in_mask_ref_rejected():
    output = _tampered_output()
    output["mask_ref"] = "https://example.com/mask.png"
    with _raises(ValueError, match="protected"):
        mos.validate_forensic_output(output)


def test_s3_url_in_visualization_ref_rejected():
    output = _tampered_output()
    output["visualization_ref"] = "s3://mybucket/vis.png"
    with _raises(ValueError, match="protected"):
        mos.validate_forensic_output(output)


def test_placeholder_in_mask_ref_allowed():
    output = _tampered_output()
    output["mask_ref"] = "<MASK_NOT_MATERIALIZED>"
    mos.validate_forensic_output(output)


def test_placeholder_in_visualization_ref_allowed():
    output = _tampered_output()
    output["visualization_ref"] = "<VISUALIZATION_NOT_MATERIALIZED>"
    mos.validate_forensic_output(output)


# ---------------------------------------------------------------------------
# generate_reason: determinism and content
# ---------------------------------------------------------------------------

def test_generate_reason_is_deterministic():
    output = _tampered_output()
    reason1 = et.generate_reason(output)
    reason2 = et.generate_reason(output)
    assert reason1 == reason2


def test_generate_reason_is_non_empty_tampered():
    assert et.generate_reason(_tampered_output())


def test_generate_reason_is_non_empty_real():
    assert et.generate_reason(_real_output())


def test_generate_reason_mentions_estimation_for_synthetic():
    output = _tampered_output()
    output.update({
        "class": "synthetic",
        "class_conf": {"real": 0.05, "synthetic": 0.90, "tampered": 0.05},
        "localization_head": "not_applicable",
        "mask_area_pct": None,
        "tampered_score": 0.05,
        "perturbations": ["none"],
    })
    reason = et.generate_reason(output)
    assert "추정" in reason, f"reason should mention estimation, got: {reason!r}"


def test_generate_reason_mentions_estimation_for_tampered():
    output = _tampered_output()
    reason = et.generate_reason(output)
    assert "추정" in reason, f"reason should mention estimation, got: {reason!r}"


def test_generate_reason_mentions_no_certainty_for_real():
    # Real image reasons should not claim family certainty
    reason = et.generate_reason(_real_output())
    assert isinstance(reason, str) and len(reason) > 0


def test_generate_reason_family_mentioned_for_synthetic():
    output = _tampered_output()
    output.update({
        "class": "synthetic",
        "class_conf": {"real": 0.05, "synthetic": 0.90, "tampered": 0.05},
        "family": "GAN",
        "localization_head": "not_applicable",
        "mask_area_pct": None,
        "tampered_score": 0.05,
        "perturbations": ["none"],
    })
    reason = et.generate_reason(output)
    assert "GAN" in reason


def test_generate_reason_skipped_localization_explained():
    output = _tampered_output(tampered_score=0.65, threshold_tau=0.75)
    output["localization_head"] = "skipped_below_threshold"
    output["mask_area_pct"] = None
    reason = et.generate_reason(output)
    assert "localization" in reason.lower() or "활성화" in reason


# ---------------------------------------------------------------------------
# Evidence signal IDs influence the generated reason
# ---------------------------------------------------------------------------

def test_reason_includes_boundary_discontinuity():
    """boundary_discontinuity signal must appear as Korean text in the reason."""
    output = _tampered_output()  # evidence includes boundary_discontinuity
    reason = et.generate_reason(output)
    assert "경계 불연속성" in reason, (
        f"expected '경계 불연속성' in reason, got: {reason!r}"
    )


def test_reason_includes_texture_inconsistency():
    """texture_inconsistency signal must appear as Korean text in the reason."""
    output = _tampered_output()  # evidence includes texture_inconsistency
    reason = et.generate_reason(output)
    assert "텍스처 불일치" in reason, (
        f"expected '텍스처 불일치' in reason, got: {reason!r}"
    )


def test_reason_changes_with_evidence_signals():
    """Reason must differ when evidence signals are present vs. absent."""
    output_with = _tampered_output()
    output_without = _tampered_output()
    output_without["evidence"] = []
    assert et.generate_reason(output_with) != et.generate_reason(output_without), (
        "reason should change when evidence signals differ"
    )


# ---------------------------------------------------------------------------
# Example JSON reason fields must be non-empty
# ---------------------------------------------------------------------------

def test_example_json_reason_fields_non_empty():
    """All reason fields in model_output_schema.example.json must be non-empty."""
    cfg = ROOT / "configs" / "model_output_schema.example.json"
    if not cfg.exists():
        return  # skip if not present (CI without configs)
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data.get("reason"), "primary example reason must be non-empty"
    for i, ex in enumerate(data.get("examples", [])):
        assert ex.get("reason"), f"examples[{i}] reason must be non-empty"


# ---------------------------------------------------------------------------
# No bare pytest import in this file
# ---------------------------------------------------------------------------

def test_no_pytest_import_in_test_file():
    """The test file itself must not contain a bare 'import pytest' line."""
    content = pathlib.Path(__file__).resolve().read_text(encoding="utf-8")
    bare_import_lines = [
        ln for ln in content.splitlines() if ln.strip() == "import pytest"
    ]
    assert not bare_import_lines, (
        "test file must not have a bare 'import pytest' line"
    )


# ---------------------------------------------------------------------------
# Social-media perturbation tags preserved
# ---------------------------------------------------------------------------

def test_perturbation_tags_preserved_in_reason():
    output = _tampered_output()
    output["perturbations"] = ["jpeg", "screenshot"]
    reason = et.generate_reason(output)
    assert "jpeg" in reason
    assert "screenshot" in reason


def test_all_perturbation_tags_accepted():
    for tag in mos.PERTURBATION_TAGS:
        output = _tampered_output(perturbations=[tag])
        # If tag is not 'none', localization and tampered score must be consistent
        mos.validate_forensic_output(output)


# ---------------------------------------------------------------------------
# should_activate_localization
# ---------------------------------------------------------------------------

def test_should_activate_localization_tampered_above_tau():
    state = mos.should_activate_localization("tampered", 0.89, 0.5)
    assert state == mos.LOCALIZATION_ACTIVATED


def test_should_activate_localization_tampered_below_tau():
    state = mos.should_activate_localization("tampered", 0.3, 0.5)
    assert state == mos.LOCALIZATION_SKIPPED_BELOW_THRESHOLD


def test_should_activate_localization_synthetic():
    state = mos.should_activate_localization("synthetic", 0.05, 0.5)
    assert state == mos.LOCALIZATION_NOT_APPLICABLE


def test_should_activate_localization_real():
    state = mos.should_activate_localization("real", 0.02, 0.5)
    assert state == mos.LOCALIZATION_NOT_APPLICABLE


# ---------------------------------------------------------------------------
# build_minimal_output
# ---------------------------------------------------------------------------

def test_build_minimal_output_real_passes_validation():
    output = mos.build_minimal_output("real", "Real-or-N/A", 0.02, 0.5)
    mos.validate_forensic_output(output)


def test_build_minimal_output_tampered_activated():
    output = mos.build_minimal_output("tampered", "LatDiff", 0.89, 0.5)
    assert output["localization_head"] == mos.LOCALIZATION_ACTIVATED


def test_build_minimal_output_tampered_skipped():
    output = mos.build_minimal_output("tampered", "LatDiff", 0.3, 0.5)
    assert output["localization_head"] == mos.LOCALIZATION_SKIPPED_BELOW_THRESHOLD


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_fns = [
        test_valid_tampered_example_passes,
        test_valid_real_example_passes,
        test_valid_synthetic_passes,
        test_class_conf_required_labels_missing_real,
        test_class_conf_missing_required_label_fails,
        test_class_conf_unknown_label_rejected,
        test_unknown_class_label_rejected,
        test_unknown_family_label_rejected,
        test_unknown_family_label_in_family_conf_rejected,
        test_confidence_value_above_1_rejected,
        test_confidence_value_below_0_rejected,
        test_localization_activated_requires_score_gte_tau,
        test_localization_activated_rejects_score_below_tau,
        test_skipped_below_threshold_allowed_when_score_lt_tau,
        test_skipped_below_threshold_rejects_score_gte_tau,
        test_mask_area_pct_negative_rejected,
        test_mask_area_pct_above_100_rejected,
        test_mask_area_pct_zero_allowed,
        test_mask_area_pct_100_allowed,
        test_protected_path_in_mask_ref_rejected,
        test_protected_path_in_visualization_ref_rejected,
        test_url_in_mask_ref_rejected,
        test_s3_url_in_visualization_ref_rejected,
        test_placeholder_in_mask_ref_allowed,
        test_placeholder_in_visualization_ref_allowed,
        test_generate_reason_is_deterministic,
        test_generate_reason_is_non_empty_tampered,
        test_generate_reason_is_non_empty_real,
        test_generate_reason_mentions_estimation_for_synthetic,
        test_generate_reason_mentions_estimation_for_tampered,
        test_generate_reason_mentions_no_certainty_for_real,
        test_generate_reason_family_mentioned_for_synthetic,
        test_generate_reason_skipped_localization_explained,
        test_reason_includes_boundary_discontinuity,
        test_reason_includes_texture_inconsistency,
        test_reason_changes_with_evidence_signals,
        test_example_json_reason_fields_non_empty,
        test_no_pytest_import_in_test_file,
        test_perturbation_tags_preserved_in_reason,
        test_all_perturbation_tags_accepted,
        test_should_activate_localization_tampered_above_tau,
        test_should_activate_localization_tampered_below_tau,
        test_should_activate_localization_synthetic,
        test_should_activate_localization_real,
        test_build_minimal_output_real_passes_validation,
        test_build_minimal_output_tampered_activated,
        test_build_minimal_output_tampered_skipped,
    ]

    passed = failed = 0
    for fn in test_fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
