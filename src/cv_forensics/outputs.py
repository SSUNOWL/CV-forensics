from __future__ import annotations

import dataclasses
from typing import Dict, Optional

from .contracts import (
    CLASS_LABELS,
    FAMILY_LABELS,
    LOCALIZATION_ACTIVATED,
    LOCALIZATION_SKIPPED,
)


@dataclasses.dataclass
class ForensicsResult:
    """Target output representation matching docs/project_brief.md Section 5."""

    class_label: str
    class_conf: Dict[str, float]
    family: str
    family_conf: Dict[str, float]
    localization_head: str
    mask_area_pct: Optional[float]
    reason: str

    def __post_init__(self) -> None:
        if self.class_label not in CLASS_LABELS:
            raise ValueError(
                f"class_label must be one of {CLASS_LABELS}, got {self.class_label!r}"
            )
        if self.family not in FAMILY_LABELS:
            raise ValueError(
                f"family must be one of {FAMILY_LABELS}, got {self.family!r}"
            )
        if self.localization_head not in (LOCALIZATION_ACTIVATED, LOCALIZATION_SKIPPED):
            raise ValueError(
                f"localization_head must be {LOCALIZATION_ACTIVATED!r} or "
                f"{LOCALIZATION_SKIPPED!r}, got {self.localization_head!r}"
            )

    def to_dict(self) -> Dict[str, object]:
        """Return a plain dict matching the target output format in project_brief.md."""
        return {
            "class": self.class_label,
            "class_conf": dict(self.class_conf),
            "family": self.family,
            "family_conf": dict(self.family_conf),
            "localization_head": self.localization_head,
            "mask_area_pct": self.mask_area_pct,
            "reason": self.reason,
        }
