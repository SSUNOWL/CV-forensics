#!/usr/bin/env python3
"""Validate the task-0004 repo skeleton: required files exist and the package imports correctly."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"

REQUIRED_FILES = [
    "src/cv_forensics/__init__.py",
    "src/cv_forensics/contracts.py",
    "src/cv_forensics/outputs.py",
    "src/cv_forensics/evidence.py",
    "scripts/agent/validate_repo_skeleton.py",
    "tests/test_repo_skeleton.py",
    "docs/repo_structure.md",
]


def _check_files() -> list[str]:
    return [rel for rel in REQUIRED_FILES if not (ROOT / rel).exists()]


def _check_import() -> str | None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    try:
        from cv_forensics.contracts import CLASS_LABELS, FAMILY_LABELS, OUTPUT_FIELDS
        from cv_forensics.evidence import build_reason
        from cv_forensics.outputs import ForensicsResult

        assert set(CLASS_LABELS) == {"real", "synthetic", "tampered"}, "CLASS_LABELS mismatch"
        assert set(FAMILY_LABELS) == {
            "LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A"
        }, "FAMILY_LABELS mismatch"
        assert "class" in OUTPUT_FIELDS, "'class' missing from OUTPUT_FIELDS"
        assert "reason" in OUTPUT_FIELDS, "'reason' missing from OUTPUT_FIELDS"

        result = ForensicsResult(
            class_label="tampered",
            class_conf={"real": 0.03, "synthetic": 0.08, "tampered": 0.89},
            family="LatDiff",
            family_conf={"LatDiff": 0.78, "PixDiff": 0.14, "GAN": 0.05, "Other": 0.03},
            localization_head="activated",
            mask_area_pct=11.2,
            reason=build_reason("tampered", "LatDiff", "activated", 11.2),
        )
        d = result.to_dict()
        assert d["class"] == "tampered", "to_dict() 'class' wrong"
        assert d["localization_head"] == "activated", "to_dict() 'localization_head' wrong"
        assert d["mask_area_pct"] == 11.2, "to_dict() 'mask_area_pct' wrong"
        assert isinstance(d["reason"], str) and len(d["reason"]) > 0, "reason empty"

        real_result = ForensicsResult(
            class_label="real",
            class_conf={"real": 0.95, "synthetic": 0.03, "tampered": 0.02},
            family="Real-or-N/A",
            family_conf={"LatDiff": 0.01, "PixDiff": 0.01, "GAN": 0.01, "Other": 0.01},
            localization_head="skipped",
            mask_area_pct=None,
            reason=build_reason("real", "Real-or-N/A", "skipped"),
        )
        assert real_result.to_dict()["mask_area_pct"] is None, "mask_area_pct should be None for real"

        return None
    except Exception as exc:
        return str(exc)


def main() -> None:
    errors: list[str] = []

    missing = _check_files()
    for f in missing:
        errors.append(f"MISSING FILE: {f}")

    import_error = _check_import()
    if import_error:
        errors.append(f"IMPORT/LOGIC ERROR: {import_error}")

    if errors:
        for e in errors:
            print(f"FAIL  {e}", file=sys.stderr)
        sys.exit(1)

    print("OK  All repo skeleton checks passed.")


if __name__ == "__main__":
    main()
