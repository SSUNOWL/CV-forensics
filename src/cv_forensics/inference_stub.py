"""Lightweight fake inference stub for task 0008.

Exercises the model output schema and template explanation pipeline from task 0007
using fully deterministic fake inputs. Does not perform real inference, load images,
access datasets, train models, or touch any protected path.
"""
from __future__ import annotations

import dataclasses
import re as _re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Config safety constants
# ---------------------------------------------------------------------------
_PROTECTED_DIRS = frozenset({"secrets", "data", "datasets", "outputs", "checkpoints"})
_PROTECTED_URL_PREFIXES = ("http://", "https://", "s3://", "gs://", "hf://")
_PROTECTED_ABS_PREFIXES = ("/home/", "/mnt/", "/root/", "/Users/")
_SECRET_KEYWORDS = frozenset({"token", "api_key", "password", "secret", "credential", "auth", "bearer"})
_SECRET_TOKEN_KEYWORDS = frozenset({"token", "password", "secret", "credential", "auth", "bearer"})


# ---------------------------------------------------------------------------
# Config safety helpers (public API: check_fake_input_config_safety)
# ---------------------------------------------------------------------------

def _is_unsafe_string(value: str) -> Optional[str]:
    """Return a description of the safety violation, or None if the value is safe."""
    if not value:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    lower = value.lower()
    for prefix in _PROTECTED_URL_PREFIXES:
        if stripped.lower().startswith(prefix):
            return f"URL-like reference: {value!r}"
    for prefix in _PROTECTED_ABS_PREFIXES:
        if stripped.startswith(prefix):
            return f"absolute machine path: {value!r}"
    # Windows drive paths: C:\ or C:/
    if len(stripped) >= 3 and stripped[0].isalpha() and stripped[1] == ":" and stripped[2] in ("\\", "/"):
        return f"Windows drive path: {value!r}"
    # .env files
    if stripped == ".env" or stripped.startswith(".env."):
        return f".env file reference: {value!r}"
    # Protected directory as a path segment
    parts = _re.split(r"[/\\]+", stripped)
    for part in parts:
        if part and part in _PROTECTED_DIRS:
            return f"protected directory {part!r} in path {value!r}"
    # Secret-looking string content. Tokenize to avoid false positives such as
    # "authoritative" merely containing "auth".
    tokens = _normalized_tokens(lower)
    for kw in _SECRET_TOKEN_KEYWORDS:
        if kw in tokens:
            return f"secret-looking value (contains token {kw!r}): {value!r}"
    if "api" in tokens and "key" in tokens:
        return f"secret-looking value (contains api key tokens): {value!r}"
    return None


def _normalized_tokens(value: str) -> List[str]:
    return [token for token in _re.split(r"[^a-z0-9]+", value.lower()) if token]


def _has_secret_key(key: str) -> bool:
    lower = key.lower()
    if lower in _SECRET_KEYWORDS:
        return True
    tokens = _normalized_tokens(lower)
    if any(token in _SECRET_TOKEN_KEYWORDS for token in tokens):
        return True
    return "api" in tokens and "key" in tokens


def _check_node(node: Any, path: str) -> None:
    """Recursively walk a config node and raise ValueError on the first unsafe finding."""
    if isinstance(node, dict):
        for k, v in node.items():
            child = f"{path}.{k}"
            if isinstance(k, str) and _has_secret_key(k):
                raise ValueError(f"{child}: secret-looking key name {k!r}")
            _check_node(v, child)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _check_node(item, f"{path}[{i}]")
    elif isinstance(node, str):
        reason = _is_unsafe_string(node)
        if reason:
            raise ValueError(f"{path}: {reason}")


def check_fake_input_config_safety(
    raw_input: Dict[str, Any], context: str = "fake_input"
) -> None:
    """Recursively validate a raw fake input config for unsafe references.

    Raises ValueError with an actionable message if any protected path, URL,
    secret-looking key, or secret-looking value is found.
    """
    _check_node(raw_input, context)

from .model_output_schema import (
    EVIDENCE_BOUNDARY_DISCONTINUITY,
    EVIDENCE_GLOBAL_SYNTHETIC_ARTIFACT,
    EVIDENCE_LOCAL_MASK_ACTIVATION,
    EVIDENCE_LOW_CONFIDENCE,
    EVIDENCE_PROVENANCE_FAMILY_SIGNAL,
    EVIDENCE_SCREENSHOT_PADDING,
    EVIDENCE_SOCIAL_MEDIA_RECOMPRESSION,
    EVIDENCE_STICKER_OVERLAY_OCCLUSION,
    EVIDENCE_TEXT_OVERLAY_OCCLUSION,
    EVIDENCE_TEXTURE_INCONSISTENCY,
    CLASS_LABEL_REAL,
    CLASS_LABEL_SYNTHETIC,
    CLASS_LABEL_TAMPERED,
    FAMILY_GAN,
    FAMILY_LATDIFF,
    FAMILY_OTHER,
    FAMILY_PIXDIFF,
    FAMILY_REAL_OR_NA,
    LOCALIZATION_ACTIVATED,
    LOCALIZATION_NOT_APPLICABLE,
    LOCALIZATION_SKIPPED_BELOW_THRESHOLD,
    PERTURBATION_JPEG,
    PERTURBATION_NONE,
    PERTURBATION_RECOMPRESSION_CHAIN,
    PERTURBATION_TEXT_OVERLAY,
    SCHEMA_VERSION,
    EvidenceSignal,
    ForensicOutput,
    validate_forensic_output,
)
from .explanation_templates import generate_reason

# ---------------------------------------------------------------------------
# Supported scenario names
# ---------------------------------------------------------------------------
SCENARIO_REAL_CLEAN = "real_clean"
SCENARIO_SYNTHETIC_LATDIFF = "synthetic_latdiff"
SCENARIO_TAMPERED_LOCALIZED = "tampered_localized"
SCENARIO_TAMPERED_BELOW_THRESHOLD = "tampered_below_threshold"
SCENARIO_LOW_CONFIDENCE = "low_confidence"

REQUIRED_SCENARIOS = (
    SCENARIO_REAL_CLEAN,
    SCENARIO_SYNTHETIC_LATDIFF,
    SCENARIO_TAMPERED_LOCALIZED,
    SCENARIO_TAMPERED_BELOW_THRESHOLD,
    SCENARIO_LOW_CONFIDENCE,
)


# ---------------------------------------------------------------------------
# Fake input record
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class FakeInput:
    input_id: str
    scenario: str
    class_hint: str
    family_hint: str
    tampered_score: float
    threshold_tau: float
    mask_area_hint_pct: Optional[float]
    perturbations: List[str]
    evidence_hints: List[str]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FakeInput":
        return FakeInput(
            input_id=str(d["input_id"]),
            scenario=str(d["scenario"]),
            class_hint=str(d["class_hint"]),
            family_hint=str(d["family_hint"]),
            tampered_score=float(d["tampered_score"]),
            threshold_tau=float(d["threshold_tau"]),
            mask_area_hint_pct=float(d["mask_area_hint_pct"]) if d.get("mask_area_hint_pct") is not None else None,
            perturbations=list(d.get("perturbations", [PERTURBATION_NONE])),
            evidence_hints=list(d.get("evidence_hints", [])),
        )


# ---------------------------------------------------------------------------
# Fake head functions — all deterministic, no real inference
# ---------------------------------------------------------------------------

def fake_backbone_summary(fake_input: FakeInput) -> Dict[str, Any]:
    """Return a symbolic backbone feature summary. No real computation."""
    return {
        "input_id": fake_input.input_id,
        "scenario": fake_input.scenario,
        "feature_dim": 512,
        "feature_norm": 1.0,
        "is_fake": True,
    }


def fake_classification_head(fake_input: FakeInput) -> Dict[str, float]:
    """Return a deterministic class_conf dict based on class_hint.

    Note: tampered_score is a separate localization-threshold signal, independent
    of class_conf["tampered"]. A low tampered_score means localization is skipped,
    but the classification can still output class=tampered.
    """
    hint = fake_input.class_hint
    if hint == CLASS_LABEL_REAL:
        return {CLASS_LABEL_REAL: 0.92, CLASS_LABEL_SYNTHETIC: 0.05, CLASS_LABEL_TAMPERED: 0.03}
    if hint == CLASS_LABEL_SYNTHETIC:
        return {CLASS_LABEL_REAL: 0.06, CLASS_LABEL_SYNTHETIC: 0.88, CLASS_LABEL_TAMPERED: 0.06}
    if hint == CLASS_LABEL_TAMPERED:
        # tampered always wins classification; tampered_score is an independent field
        return {CLASS_LABEL_REAL: 0.09, CLASS_LABEL_SYNTHETIC: 0.11, CLASS_LABEL_TAMPERED: 0.80}
    # low_confidence: spread evenly so no class dominates clearly
    return {CLASS_LABEL_REAL: 0.38, CLASS_LABEL_SYNTHETIC: 0.32, CLASS_LABEL_TAMPERED: 0.30}


def _family_conf_for(family_hint: str, class_hint: str) -> Dict[str, float]:
    """Build a deterministic family confidence dict."""
    if class_hint == CLASS_LABEL_REAL or family_hint == FAMILY_REAL_OR_NA:
        return {
            FAMILY_LATDIFF: 0.03,
            FAMILY_PIXDIFF: 0.02,
            FAMILY_GAN: 0.02,
            FAMILY_OTHER: 0.03,
            FAMILY_REAL_OR_NA: 0.90,
        }
    distributions: Dict[str, Dict[str, float]] = {
        FAMILY_LATDIFF: {
            FAMILY_LATDIFF: 0.80,
            FAMILY_PIXDIFF: 0.10,
            FAMILY_GAN: 0.05,
            FAMILY_OTHER: 0.03,
            FAMILY_REAL_OR_NA: 0.02,
        },
        FAMILY_PIXDIFF: {
            FAMILY_LATDIFF: 0.08,
            FAMILY_PIXDIFF: 0.78,
            FAMILY_GAN: 0.07,
            FAMILY_OTHER: 0.05,
            FAMILY_REAL_OR_NA: 0.02,
        },
        FAMILY_GAN: {
            FAMILY_LATDIFF: 0.05,
            FAMILY_PIXDIFF: 0.07,
            FAMILY_GAN: 0.80,
            FAMILY_OTHER: 0.06,
            FAMILY_REAL_OR_NA: 0.02,
        },
        FAMILY_OTHER: {
            FAMILY_LATDIFF: 0.10,
            FAMILY_PIXDIFF: 0.10,
            FAMILY_GAN: 0.10,
            FAMILY_OTHER: 0.68,
            FAMILY_REAL_OR_NA: 0.02,
        },
    }
    return distributions.get(family_hint, distributions[FAMILY_OTHER])


def fake_family_head(fake_input: FakeInput) -> Dict[str, float]:
    """Return a deterministic family_conf dict based on family_hint."""
    return _family_conf_for(fake_input.family_hint, fake_input.class_hint)


def fake_localization_head(
    fake_input: FakeInput,
    class_label: str,
) -> tuple[str, Optional[float]]:
    """Return (localization_head_state, mask_area_pct) based on class and tau logic."""
    if class_label != CLASS_LABEL_TAMPERED:
        return LOCALIZATION_NOT_APPLICABLE, None

    if float(fake_input.tampered_score) >= float(fake_input.threshold_tau):
        area = fake_input.mask_area_hint_pct if fake_input.mask_area_hint_pct is not None else 10.0
        return LOCALIZATION_ACTIVATED, float(area)

    return LOCALIZATION_SKIPPED_BELOW_THRESHOLD, None


def fake_evidence_aggregation(
    fake_input: FakeInput,
    class_label: str,
    family: str,
    localization_state: str,
) -> List[EvidenceSignal]:
    """Map class/family/localization/perturbation context into evidence signals."""
    signals: List[EvidenceSignal] = []

    # Directly requested hints from config
    hint_map = {
        EVIDENCE_BOUNDARY_DISCONTINUITY: EVIDENCE_BOUNDARY_DISCONTINUITY,
        EVIDENCE_GLOBAL_SYNTHETIC_ARTIFACT: EVIDENCE_GLOBAL_SYNTHETIC_ARTIFACT,
        EVIDENCE_LOCAL_MASK_ACTIVATION: EVIDENCE_LOCAL_MASK_ACTIVATION,
        EVIDENCE_LOW_CONFIDENCE: EVIDENCE_LOW_CONFIDENCE,
        EVIDENCE_PROVENANCE_FAMILY_SIGNAL: EVIDENCE_PROVENANCE_FAMILY_SIGNAL,
        EVIDENCE_SCREENSHOT_PADDING: EVIDENCE_SCREENSHOT_PADDING,
        EVIDENCE_SOCIAL_MEDIA_RECOMPRESSION: EVIDENCE_SOCIAL_MEDIA_RECOMPRESSION,
        EVIDENCE_STICKER_OVERLAY_OCCLUSION: EVIDENCE_STICKER_OVERLAY_OCCLUSION,
        EVIDENCE_TEXT_OVERLAY_OCCLUSION: EVIDENCE_TEXT_OVERLAY_OCCLUSION,
        EVIDENCE_TEXTURE_INCONSISTENCY: EVIDENCE_TEXTURE_INCONSISTENCY,
    }
    for hint in fake_input.evidence_hints:
        sid = hint_map.get(hint)
        if sid:
            signals.append(EvidenceSignal(signal_id=sid))

    # Auto-add signals based on class/localization context
    if class_label == CLASS_LABEL_SYNTHETIC:
        if not any(s.signal_id == EVIDENCE_GLOBAL_SYNTHETIC_ARTIFACT for s in signals):
            signals.append(EvidenceSignal(signal_id=EVIDENCE_GLOBAL_SYNTHETIC_ARTIFACT))
        if family not in (FAMILY_REAL_OR_NA, FAMILY_OTHER):
            if not any(s.signal_id == EVIDENCE_PROVENANCE_FAMILY_SIGNAL for s in signals):
                signals.append(EvidenceSignal(signal_id=EVIDENCE_PROVENANCE_FAMILY_SIGNAL))

    if localization_state == LOCALIZATION_ACTIVATED:
        if not any(s.signal_id == EVIDENCE_LOCAL_MASK_ACTIVATION for s in signals):
            signals.append(EvidenceSignal(signal_id=EVIDENCE_LOCAL_MASK_ACTIVATION))

    # Social-media perturbation signal
    sns_perturbs = {PERTURBATION_JPEG, PERTURBATION_RECOMPRESSION_CHAIN, PERTURBATION_TEXT_OVERLAY}
    if sns_perturbs & set(fake_input.perturbations):
        if PERTURBATION_RECOMPRESSION_CHAIN in fake_input.perturbations:
            if not any(s.signal_id == EVIDENCE_SOCIAL_MEDIA_RECOMPRESSION for s in signals):
                signals.append(EvidenceSignal(signal_id=EVIDENCE_SOCIAL_MEDIA_RECOMPRESSION))
        if PERTURBATION_TEXT_OVERLAY in fake_input.perturbations:
            if not any(s.signal_id == EVIDENCE_TEXT_OVERLAY_OCCLUSION for s in signals):
                signals.append(EvidenceSignal(signal_id=EVIDENCE_TEXT_OVERLAY_OCCLUSION))

    return signals


# ---------------------------------------------------------------------------
# Top-level fake inference entry points
# ---------------------------------------------------------------------------

def _resolve_class_label(class_conf: Dict[str, float]) -> str:
    """Return the class label with the highest confidence."""
    return max(class_conf, key=lambda k: class_conf[k])


def _resolve_family_label(family_conf: Dict[str, float]) -> str:
    """Return the family label with the highest confidence."""
    return max(family_conf, key=lambda k: family_conf[k])


def run_fake_inference(fake_input: FakeInput) -> Dict[str, Any]:
    """Run the full fake inference pipeline for a single FakeInput.

    Returns a validated forensic output dict. Raises ValueError if validation fails.
    """
    # Fake backbone (symbolic, not used in computation but mirrors the proposal flow)
    _backbone = fake_backbone_summary(fake_input)

    # Classification head
    class_conf = fake_classification_head(fake_input)
    class_label = _resolve_class_label(class_conf)

    # Family/provenance head
    family_conf = fake_family_head(fake_input)
    family = _resolve_family_label(family_conf)

    # Conditional localization
    localization_state, mask_area_pct = fake_localization_head(fake_input, class_label)

    # Evidence aggregation
    evidence_signals = fake_evidence_aggregation(
        fake_input, class_label, family, localization_state
    )

    # Assemble output dict
    output: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "class": class_label,
        "class_conf": class_conf,
        "family": family,
        "family_conf": family_conf,
        "localization_head": localization_state,
        "mask_area_pct": mask_area_pct,
        "threshold_tau": float(fake_input.threshold_tau),
        "tampered_score": float(fake_input.tampered_score),
        "evidence": [s.to_dict() for s in evidence_signals],
        "perturbations": list(fake_input.perturbations),
        "reason": "",
    }

    # Generate deterministic reason
    output["reason"] = generate_reason(output)

    # Validate via task 0007 schema validator
    validate_forensic_output(output)

    return output


def run_fake_batch(fake_inputs: List[FakeInput]) -> List[Dict[str, Any]]:
    """Run fake inference over a list of FakeInput records."""
    return [run_fake_inference(fi) for fi in fake_inputs]
