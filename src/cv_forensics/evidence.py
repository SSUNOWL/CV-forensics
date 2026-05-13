from __future__ import annotations

from typing import Optional

from .contracts import LOCALIZATION_ACTIVATED


def build_reason(
    class_label: str,
    family: str,
    localization_head: str,
    mask_area_pct: Optional[float] = None,
) -> str:
    """Return a deterministic template-based explanation string.

    Combines class, family, and localization information as described in
    docs/project_brief.md Section 6.6.
    """
    if class_label == "tampered" and localization_head == LOCALIZATION_ACTIVATED:
        area_note = (
            f" ({mask_area_pct:.1f}% of image area)" if mask_area_pct is not None else ""
        )
        return (
            f"Image classified as tampered. "
            f"Suspicious region activated{area_note}. "
            f"Generator family estimated as {family}."
        )
    if class_label == "tampered":
        return (
            f"Image classified as tampered. "
            f"Localization head not activated. "
            f"Generator family estimated as {family}."
        )
    if class_label == "synthetic":
        return (
            f"Image classified as synthetic (AI-generated). "
            f"Generator family estimated as {family}."
        )
    if class_label == "real":
        return "Image classified as real. No manipulation or generation artifacts detected."
    return f"Image classified as {class_label}. Generator family estimated as {family}."
