from __future__ import annotations

import dataclasses
import re
from typing import Dict, Final, List, Optional, Tuple

SCHEMA_VERSION: Final[str] = "0.1.0"

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------
CLASS_LABEL_REAL: Final[str] = "real"
CLASS_LABEL_SYNTHETIC: Final[str] = "synthetic"
CLASS_LABEL_TAMPERED: Final[str] = "tampered"
CLASS_LABELS: Final[Tuple[str, ...]] = (
    CLASS_LABEL_REAL,
    CLASS_LABEL_SYNTHETIC,
    CLASS_LABEL_TAMPERED,
)

# ---------------------------------------------------------------------------
# Family labels (coarse generator-family provenance)
# ---------------------------------------------------------------------------
FAMILY_LATDIFF: Final[str] = "LatDiff"
FAMILY_PIXDIFF: Final[str] = "PixDiff"
FAMILY_GAN: Final[str] = "GAN"
FAMILY_OTHER: Final[str] = "Other"
FAMILY_REAL_OR_NA: Final[str] = "Real-or-N/A"
FAMILY_LABELS: Final[Tuple[str, ...]] = (
    FAMILY_LATDIFF,
    FAMILY_PIXDIFF,
    FAMILY_GAN,
    FAMILY_OTHER,
    FAMILY_REAL_OR_NA,
)

# ---------------------------------------------------------------------------
# Localization states
# ---------------------------------------------------------------------------
LOCALIZATION_NOT_APPLICABLE: Final[str] = "not_applicable"
LOCALIZATION_SKIPPED_BELOW_THRESHOLD: Final[str] = "skipped_below_threshold"
LOCALIZATION_ACTIVATED: Final[str] = "activated"
LOCALIZATION_UNAVAILABLE: Final[str] = "unavailable"
LOCALIZATION_STATES: Final[Tuple[str, ...]] = (
    LOCALIZATION_NOT_APPLICABLE,
    LOCALIZATION_SKIPPED_BELOW_THRESHOLD,
    LOCALIZATION_ACTIVATED,
    LOCALIZATION_UNAVAILABLE,
)

# ---------------------------------------------------------------------------
# Evidence signal IDs
# ---------------------------------------------------------------------------
EVIDENCE_GLOBAL_SYNTHETIC_ARTIFACT: Final[str] = "global_synthetic_artifact"
EVIDENCE_BOUNDARY_DISCONTINUITY: Final[str] = "boundary_discontinuity"
EVIDENCE_TEXTURE_INCONSISTENCY: Final[str] = "texture_inconsistency"
EVIDENCE_LOCAL_MASK_ACTIVATION: Final[str] = "local_mask_activation"
EVIDENCE_PROVENANCE_FAMILY_SIGNAL: Final[str] = "provenance_family_signal"
EVIDENCE_LOW_CONFIDENCE: Final[str] = "low_confidence"
EVIDENCE_SOCIAL_MEDIA_RECOMPRESSION: Final[str] = "social_media_recompression"
EVIDENCE_SCREENSHOT_PADDING: Final[str] = "screenshot_padding"
EVIDENCE_TEXT_OVERLAY_OCCLUSION: Final[str] = "text_overlay_occlusion"
EVIDENCE_STICKER_OVERLAY_OCCLUSION: Final[str] = "sticker_overlay_occlusion"
EVIDENCE_SIGNAL_IDS: Final[Tuple[str, ...]] = (
    EVIDENCE_GLOBAL_SYNTHETIC_ARTIFACT,
    EVIDENCE_BOUNDARY_DISCONTINUITY,
    EVIDENCE_TEXTURE_INCONSISTENCY,
    EVIDENCE_LOCAL_MASK_ACTIVATION,
    EVIDENCE_PROVENANCE_FAMILY_SIGNAL,
    EVIDENCE_LOW_CONFIDENCE,
    EVIDENCE_SOCIAL_MEDIA_RECOMPRESSION,
    EVIDENCE_SCREENSHOT_PADDING,
    EVIDENCE_TEXT_OVERLAY_OCCLUSION,
    EVIDENCE_STICKER_OVERLAY_OCCLUSION,
)

# ---------------------------------------------------------------------------
# Perturbation tags
# ---------------------------------------------------------------------------
PERTURBATION_NONE: Final[str] = "none"
PERTURBATION_JPEG: Final[str] = "jpeg"
PERTURBATION_RESIZE: Final[str] = "resize"
PERTURBATION_CROP: Final[str] = "crop"
PERTURBATION_ROTATION: Final[str] = "rotation"
PERTURBATION_PADDING: Final[str] = "padding"
PERTURBATION_SHEAR: Final[str] = "shear"
PERTURBATION_SCREENSHOT: Final[str] = "screenshot"
PERTURBATION_TEXT_OVERLAY: Final[str] = "text_overlay"
PERTURBATION_STICKER_OVERLAY: Final[str] = "sticker_overlay"
PERTURBATION_RECOMPRESSION_CHAIN: Final[str] = "recompression_chain"
PERTURBATION_TAGS: Final[Tuple[str, ...]] = (
    PERTURBATION_NONE,
    PERTURBATION_JPEG,
    PERTURBATION_RESIZE,
    PERTURBATION_CROP,
    PERTURBATION_ROTATION,
    PERTURBATION_PADDING,
    PERTURBATION_SHEAR,
    PERTURBATION_SCREENSHOT,
    PERTURBATION_TEXT_OVERLAY,
    PERTURBATION_STICKER_OVERLAY,
    PERTURBATION_RECOMPRESSION_CHAIN,
)

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------
CONFIDENCE_SUM_TOLERANCE: Final[float] = 0.02
LOW_CONFIDENCE_THRESHOLD: Final[float] = 0.5

# ---------------------------------------------------------------------------
# Protected path / reference detection
# ---------------------------------------------------------------------------
_PROTECTED_PATH_COMPONENT_NAMES: Final[frozenset] = frozenset(
    {"secrets", "data", "datasets", "outputs", "checkpoints"}
)
_PROTECTED_PATH_PREFIXES: Final[Tuple[str, ...]] = (
    "/home/",
    "/mnt/",
    "/root/",
    "/Users/",
)
_PROTECTED_URL_PREFIXES: Final[Tuple[str, ...]] = (
    "http://",
    "https://",
    "s3://",
    "gs://",
    "hf://",
)
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")


def _is_protected_reference(value: str) -> bool:
    """Return True if value looks like a protected path, URL, or local machine path."""
    if not value:
        return False
    # Allow placeholder strings (e.g. <MASK_NOT_MATERIALIZED>)
    if value.startswith("<"):
        return False
    lower = value.lower()
    # URL check
    for prefix in _PROTECTED_URL_PREFIXES:
        if lower.startswith(prefix):
            return True
    # Windows drive path (C:\ or C:/)
    if _WINDOWS_DRIVE_RE.match(value):
        return True
    # Absolute local path prefixes
    for prefix in _PROTECTED_PATH_PREFIXES:
        if value.startswith(prefix):
            return True
    # Path-component checks (only for strings that look like paths)
    if "/" in value or "\\" in value:
        parts = re.split(r"[/\\]", value)
        for part in parts:
            if part == ".env" or part.startswith(".env."):
                return True
            if part in _PROTECTED_PATH_COMPONENT_NAMES:
                return True
    # .env as a standalone value (no slashes)
    if value == ".env" or value.startswith(".env."):
        return True
    return False


# ---------------------------------------------------------------------------
# Dataclasses for schema structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ClassConf:
    """Confidence distribution over the three class labels."""
    real: float
    synthetic: float
    tampered: float

    def to_dict(self) -> Dict[str, float]:
        return {
            CLASS_LABEL_REAL: self.real,
            CLASS_LABEL_SYNTHETIC: self.synthetic,
            CLASS_LABEL_TAMPERED: self.tampered,
        }


@dataclasses.dataclass
class LocalizationSummary:
    """Localization head state and supporting scalars."""
    localization_head: str
    mask_area_pct: Optional[float]
    threshold_tau: float
    tampered_score: float


@dataclasses.dataclass
class EvidenceSignal:
    """A single tagged evidence signal from the model."""
    signal_id: str
    description: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {"signal_id": self.signal_id, "description": self.description}


@dataclasses.dataclass
class ForensicOutput:
    """Full forensic output matching the project target output contract."""
    schema_version: str
    class_label: str
    class_conf: Dict[str, float]
    family: str
    family_conf: Dict[str, float]
    localization_head: str
    mask_area_pct: Optional[float]
    threshold_tau: float
    tampered_score: float
    evidence: List[EvidenceSignal]
    perturbations: List[str]
    reason: str
    # Optional future fields
    mask_ref: Optional[str] = None
    visualization_ref: Optional[str] = None
    model_stage: Optional[str] = None
    family_policy: Optional[str] = None
    latency_ms: Optional[float] = None
    fps: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        d: Dict[str, object] = {
            "schema_version": self.schema_version,
            "class": self.class_label,
            "class_conf": dict(self.class_conf),
            "family": self.family,
            "family_conf": dict(self.family_conf),
            "localization_head": self.localization_head,
            "mask_area_pct": self.mask_area_pct,
            "threshold_tau": self.threshold_tau,
            "tampered_score": self.tampered_score,
            "evidence": [s.to_dict() for s in self.evidence],
            "perturbations": list(self.perturbations),
            "reason": self.reason,
        }
        for opt_field in (
            "mask_ref",
            "visualization_ref",
            "model_stage",
            "family_policy",
            "latency_ms",
            "fps",
        ):
            val = getattr(self, opt_field)
            if val is not None:
                d[opt_field] = val
        return d


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_class_label(label: str) -> None:
    if label not in CLASS_LABELS:
        raise ValueError(
            f"Invalid class label {label!r}. Must be one of {CLASS_LABELS}."
        )


def validate_family_label(label: str) -> None:
    if label not in FAMILY_LABELS:
        raise ValueError(
            f"Invalid family label {label!r}. Must be one of {FAMILY_LABELS}."
        )


def validate_confidence_map(
    conf: Dict[str, float],
    required_labels: Optional[Tuple[str, ...]] = None,
    known_labels: Optional[Tuple[str, ...]] = None,
    name: str = "confidence",
) -> None:
    """Validate a confidence distribution dictionary."""
    if required_labels:
        for label in required_labels:
            if label not in conf:
                raise ValueError(f"{name}: missing required label {label!r}")
    for label, value in conf.items():
        if known_labels is not None and label not in known_labels:
            raise ValueError(f"{name}: unknown label {label!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"{name}[{label!r}]: value must be numeric, got {type(value).__name__}"
            )
        fvalue = float(value)
        if not (0.0 <= fvalue <= 1.0):
            raise ValueError(
                f"{name}[{label!r}]: value must be in [0.0, 1.0], got {fvalue}"
            )
    if conf:
        total = sum(float(v) for v in conf.values())
        if abs(total - 1.0) > CONFIDENCE_SUM_TOLERANCE:
            raise ValueError(
                f"{name}: values must sum to approximately 1.0 "
                f"(tolerance ±{CONFIDENCE_SUM_TOLERANCE}), got {total:.4f}"
            )


def validate_localization_summary(
    localization_head: str,
    mask_area_pct: Optional[float],
    threshold_tau: float,
    tampered_score: float,
) -> None:
    """Validate localization state consistency."""
    if localization_head not in LOCALIZATION_STATES:
        raise ValueError(
            f"Invalid localization_head {localization_head!r}. "
            f"Must be one of {LOCALIZATION_STATES}."
        )
    if localization_head == LOCALIZATION_ACTIVATED:
        if float(tampered_score) < float(threshold_tau):
            raise ValueError(
                f"localization_head='activated' requires tampered_score >= threshold_tau "
                f"(got tampered_score={tampered_score}, threshold_tau={threshold_tau})"
            )
    if localization_head == LOCALIZATION_SKIPPED_BELOW_THRESHOLD:
        if float(tampered_score) >= float(threshold_tau):
            raise ValueError(
                f"localization_head='skipped_below_threshold' requires "
                f"tampered_score < threshold_tau "
                f"(got tampered_score={tampered_score}, threshold_tau={threshold_tau})"
            )
    if mask_area_pct is not None:
        if float(mask_area_pct) < 0.0:
            raise ValueError(
                f"mask_area_pct must be >= 0, got {mask_area_pct}"
            )
        if float(mask_area_pct) > 100.0:
            raise ValueError(
                f"mask_area_pct must be <= 100, got {mask_area_pct}"
            )


def validate_evidence_signals(signals: List[Dict]) -> None:
    """Validate a list of evidence signal dicts."""
    for i, signal in enumerate(signals):
        if not isinstance(signal, dict):
            raise TypeError(f"evidence[{i}]: expected dict, got {type(signal).__name__}")
        signal_id = signal.get("signal_id", "")
        if signal_id not in EVIDENCE_SIGNAL_IDS:
            raise ValueError(
                f"evidence[{i}]: unknown signal_id {signal_id!r}. "
                f"Must be one of {EVIDENCE_SIGNAL_IDS}."
            )


def validate_forensic_output(output: Dict[str, object]) -> None:
    """Validate a complete forensic output dictionary."""
    required_fields = [
        "schema_version",
        "class",
        "class_conf",
        "family",
        "family_conf",
        "localization_head",
        "mask_area_pct",
        "threshold_tau",
        "tampered_score",
        "evidence",
        "perturbations",
        "reason",
    ]
    for field in required_fields:
        if field not in output:
            raise ValueError(f"Missing required field: {field!r}")

    validate_class_label(str(output["class"]))
    validate_family_label(str(output["family"]))

    validate_confidence_map(
        output["class_conf"],
        required_labels=CLASS_LABELS,
        known_labels=CLASS_LABELS,
        name="class_conf",
    )
    validate_confidence_map(
        output["family_conf"],
        required_labels=None,
        known_labels=FAMILY_LABELS,
        name="family_conf",
    )

    validate_localization_summary(
        localization_head=str(output["localization_head"]),
        mask_area_pct=output["mask_area_pct"],
        threshold_tau=float(output["threshold_tau"]),
        tampered_score=float(output["tampered_score"]),
    )

    validate_evidence_signals(output["evidence"])

    for tag in output["perturbations"]:
        if tag not in PERTURBATION_TAGS:
            raise ValueError(
                f"Unknown perturbation tag {tag!r}. Must be one of {PERTURBATION_TAGS}."
            )

    if not isinstance(output["reason"], str):
        raise TypeError(f"reason must be a string, got {type(output['reason']).__name__}")

    # Check optional reference fields for protected paths
    for ref_field in ("mask_ref", "visualization_ref"):
        if ref_field in output and output[ref_field] is not None:
            ref_val = str(output[ref_field])
            if _is_protected_reference(ref_val):
                raise ValueError(
                    f"{ref_field}: value looks like a protected path or URL: {ref_val!r}"
                )


# ---------------------------------------------------------------------------
# Conditional localization helper
# ---------------------------------------------------------------------------

def should_activate_localization(
    class_label: str,
    tampered_score: float,
    threshold_tau: float,
) -> str:
    """Return the appropriate localization state for given inputs."""
    if class_label != CLASS_LABEL_TAMPERED:
        return LOCALIZATION_NOT_APPLICABLE
    if float(tampered_score) >= float(threshold_tau):
        return LOCALIZATION_ACTIVATED
    return LOCALIZATION_SKIPPED_BELOW_THRESHOLD


# ---------------------------------------------------------------------------
# Dry-run / fake output builder
# ---------------------------------------------------------------------------

def build_minimal_output(
    class_label: str = CLASS_LABEL_REAL,
    family: str = FAMILY_REAL_OR_NA,
    tampered_score: float = 0.0,
    threshold_tau: float = 0.5,
    class_conf: Optional[Dict[str, float]] = None,
    family_conf: Optional[Dict[str, float]] = None,
    perturbations: Optional[List[str]] = None,
    evidence: Optional[List[Dict]] = None,
) -> Dict[str, object]:
    """Build a minimal valid forensic output dict for dry-run or testing."""
    if class_conf is None:
        class_conf = {
            CLASS_LABEL_REAL: 1.0 if class_label == CLASS_LABEL_REAL else 0.0,
            CLASS_LABEL_SYNTHETIC: 1.0 if class_label == CLASS_LABEL_SYNTHETIC else 0.0,
            CLASS_LABEL_TAMPERED: 1.0 if class_label == CLASS_LABEL_TAMPERED else 0.0,
        }
    if family_conf is None:
        family_conf = {lbl: (1.0 if lbl == family else 0.0) for lbl in FAMILY_LABELS}
    if perturbations is None:
        perturbations = [PERTURBATION_NONE]
    if evidence is None:
        evidence = []

    localization_head = should_activate_localization(class_label, tampered_score, threshold_tau)
    mask_area_pct: Optional[float] = 0.0 if localization_head == LOCALIZATION_ACTIVATED else None

    return {
        "schema_version": SCHEMA_VERSION,
        "class": class_label,
        "class_conf": class_conf,
        "family": family,
        "family_conf": family_conf,
        "localization_head": localization_head,
        "mask_area_pct": mask_area_pct,
        "threshold_tau": threshold_tau,
        "tampered_score": tampered_score,
        "evidence": evidence,
        "perturbations": perturbations,
        "reason": "",
    }
