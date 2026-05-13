"""Tests for task-0004 repo skeleton (uses only stdlib + pytest)."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

from cv_forensics.contracts import CLASS_LABELS, FAMILY_LABELS, OUTPUT_FIELDS
from cv_forensics.evidence import build_reason
from cv_forensics.outputs import ForensicsResult


# ---------------------------------------------------------------------------
# contracts
# ---------------------------------------------------------------------------

def test_class_labels_exact():
    assert set(CLASS_LABELS) == {"real", "synthetic", "tampered"}


def test_family_labels_exact():
    assert set(FAMILY_LABELS) == {"LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A"}


def test_output_fields_exact():
    expected = {
        "class",
        "class_conf",
        "family",
        "family_conf",
        "localization_head",
        "mask_area_pct",
        "reason",
    }
    assert expected == set(OUTPUT_FIELDS)


# ---------------------------------------------------------------------------
# ForensicsResult construction and to_dict
# ---------------------------------------------------------------------------

def _make_tampered() -> ForensicsResult:
    return ForensicsResult(
        class_label="tampered",
        class_conf={"real": 0.03, "synthetic": 0.08, "tampered": 0.89},
        family="LatDiff",
        family_conf={"LatDiff": 0.78, "PixDiff": 0.14, "GAN": 0.05, "Other": 0.03},
        localization_head="activated",
        mask_area_pct=11.2,
        reason="placeholder",
    )


def test_forensics_result_tampered_to_dict():
    d = _make_tampered().to_dict()
    assert d["class"] == "tampered"
    assert d["localization_head"] == "activated"
    assert abs(d["mask_area_pct"] - 11.2) < 1e-6


def test_forensics_result_real():
    r = ForensicsResult(
        class_label="real",
        class_conf={"real": 0.95, "synthetic": 0.03, "tampered": 0.02},
        family="Real-or-N/A",
        family_conf={"LatDiff": 0.01, "PixDiff": 0.01, "GAN": 0.01, "Other": 0.01},
        localization_head="skipped",
        mask_area_pct=None,
        reason="placeholder",
    )
    d = r.to_dict()
    assert d["class"] == "real"
    assert d["mask_area_pct"] is None
    assert d["localization_head"] == "skipped"


def test_forensics_result_synthetic():
    r = ForensicsResult(
        class_label="synthetic",
        class_conf={"real": 0.05, "synthetic": 0.90, "tampered": 0.05},
        family="GAN",
        family_conf={"LatDiff": 0.05, "PixDiff": 0.05, "GAN": 0.85, "Other": 0.05},
        localization_head="skipped",
        mask_area_pct=None,
        reason="placeholder",
    )
    assert r.to_dict()["family"] == "GAN"


def test_to_dict_keys_complete():
    d = _make_tampered().to_dict()
    expected_keys = {
        "class", "class_conf", "family", "family_conf",
        "localization_head", "mask_area_pct", "reason",
    }
    assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_class_label_raises():
    with pytest.raises(ValueError, match="class_label"):
        ForensicsResult(
            class_label="unknown",
            class_conf={},
            family="GAN",
            family_conf={},
            localization_head="skipped",
            mask_area_pct=None,
            reason="",
        )


def test_invalid_family_raises():
    with pytest.raises(ValueError, match="family"):
        ForensicsResult(
            class_label="real",
            class_conf={},
            family="StableDiffusion",
            family_conf={},
            localization_head="skipped",
            mask_area_pct=None,
            reason="",
        )


def test_invalid_localization_head_raises():
    with pytest.raises(ValueError, match="localization_head"):
        ForensicsResult(
            class_label="real",
            class_conf={},
            family="Real-or-N/A",
            family_conf={},
            localization_head="running",
            mask_area_pct=None,
            reason="",
        )


# ---------------------------------------------------------------------------
# build_reason
# ---------------------------------------------------------------------------

def test_build_reason_tampered_activated():
    r = build_reason("tampered", "LatDiff", "activated", 11.2)
    assert "tampered" in r.lower()
    assert "LatDiff" in r
    assert "11.2" in r


def test_build_reason_tampered_skipped():
    r = build_reason("tampered", "GAN", "skipped")
    assert "tampered" in r.lower()
    assert "GAN" in r


def test_build_reason_synthetic():
    r = build_reason("synthetic", "PixDiff", "skipped")
    assert "synthetic" in r.lower()
    assert "PixDiff" in r


def test_build_reason_real():
    r = build_reason("real", "Real-or-N/A", "skipped")
    assert "real" in r.lower()


def test_build_reason_returns_string():
    for cls in CLASS_LABELS:
        family = FAMILY_LABELS[0]
        result = build_reason(cls, family, "skipped")
        assert isinstance(result, str)
        assert len(result) > 0
