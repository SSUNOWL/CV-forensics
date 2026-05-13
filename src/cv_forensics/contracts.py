from __future__ import annotations

from typing import Final, Tuple

CLASS_LABELS: Final[Tuple[str, ...]] = ("real", "synthetic", "tampered")

FAMILY_LABELS: Final[Tuple[str, ...]] = (
    "LatDiff",
    "PixDiff",
    "GAN",
    "Other",
    "Real-or-N/A",
)

OUTPUT_FIELDS: Final[Tuple[str, ...]] = (
    "class",
    "class_conf",
    "family",
    "family_conf",
    "localization_head",
    "mask_area_pct",
    "reason",
)

LOCALIZATION_ACTIVATED: Final[str] = "activated"
LOCALIZATION_SKIPPED: Final[str] = "skipped"
