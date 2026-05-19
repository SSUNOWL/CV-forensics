"""Tiny pre-SNS integrated model helpers for smoke tests.

This module does not load images, inspect datasets, write outputs, or train a
real model. Torch is accepted as an explicit dependency argument so importing
the module remains lightweight for validators and documentation tooling.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .explanation_templates import generate_reason
from .model_output_schema import (
    EVIDENCE_BOUNDARY_DISCONTINUITY,
    EVIDENCE_GLOBAL_SYNTHETIC_ARTIFACT,
    EVIDENCE_LOCAL_MASK_ACTIVATION,
    EVIDENCE_PROVENANCE_FAMILY_SIGNAL,
    FAMILY_LABELS,
    LOCALIZATION_ACTIVATED,
    LOCALIZATION_NOT_APPLICABLE,
    LOCALIZATION_SKIPPED_BELOW_THRESHOLD,
    PERTURBATION_NONE,
    SCHEMA_VERSION,
    validate_forensic_output,
)

CLASS_LABELS: Tuple[str, ...] = ("real", "full_synthetic", "tampered")
SCHEMA_CLASS_LABELS: Tuple[str, ...] = ("real", "synthetic", "tampered")
FAMILY_SMOKE_LABELS: Tuple[str, ...] = FAMILY_LABELS
CLASS_TO_INDEX: Dict[str, int] = {label: index for index, label in enumerate(CLASS_LABELS)}
FAMILY_TO_INDEX: Dict[str, int] = {label: index for index, label in enumerate(FAMILY_SMOKE_LABELS)}


def schema_class_label(smoke_label: str) -> str:
    """Map smoke labels to the existing output schema label vocabulary."""
    if smoke_label == "full_synthetic":
        return "synthetic"
    return smoke_label


def build_tiny_integrated_model(torch: Any, input_dim: int, mask_dim: int):
    """Return a deterministic tiny shared-backbone multi-head torch module."""

    class TinyPreSnsIntegratedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 32),
                torch.nn.ReLU(),
            )
            self.class_head = torch.nn.Linear(32, len(CLASS_LABELS))
            self.family_head = torch.nn.Linear(32, len(FAMILY_SMOKE_LABELS))
            self.localization_head = torch.nn.Linear(32, mask_dim)

        def forward(self, inputs):
            features = self.backbone(inputs)
            class_logits = self.class_head(features)
            family_logits = self.family_head(features)
            localization_logits = self.localization_head(features)
            evidence_fields = {
                "feature_mean": features.mean(dim=1),
                "feature_abs_mean": features.abs().mean(dim=1),
            }
            return {
                "class_logits": class_logits,
                "family_logits": family_logits,
                "localization_logits": localization_logits,
                "evidence_fields": evidence_fields,
            }

    return TinyPreSnsIntegratedModel()


def family_loss_indices(family_targets: Sequence[Optional[int]]) -> List[int]:
    """Return sample indices that have supervised family labels."""
    return [index for index, target in enumerate(family_targets) if target is not None and int(target) >= 0]


def localization_loss_indices(class_labels: Sequence[str], has_mask: Sequence[bool]) -> List[int]:
    """Return sample indices where localization loss is allowed."""
    return [
        index
        for index, (label, mask_available) in enumerate(zip(class_labels, has_mask))
        if label == "tampered" and mask_available
    ]


def route_loss_availability(
    class_labels: Sequence[str],
    family_targets: Sequence[Optional[int]],
    has_mask: Sequence[bool],
) -> Dict[str, object]:
    """Describe which losses are available for a mixed tiny batch."""
    family_indices = family_loss_indices(family_targets)
    localization_indices = localization_loss_indices(class_labels, has_mask)
    return {
        "class_loss": True,
        "family_loss": bool(family_indices),
        "family_indices": family_indices,
        "localization_loss": bool(localization_indices),
        "localization_indices": localization_indices,
        "localization_loss_applied_to_tampered_only": all(class_labels[i] == "tampered" for i in localization_indices),
    }


def compute_integrated_losses(
    torch: Any,
    outputs: Dict[str, Any],
    class_targets: Any,
    family_targets: Sequence[Optional[int]],
    mask_targets: Any,
    class_labels: Sequence[str],
    has_mask: Sequence[bool],
    family_loss_weight: float = 1.0,
    localization_loss_weight: float = 1.0,
) -> Dict[str, Any]:
    """Compute routed tiny smoke losses without assuming every target exists."""
    routing = route_loss_availability(class_labels, family_targets, has_mask)
    class_loss = torch.nn.CrossEntropyLoss()(outputs["class_logits"], class_targets)

    family_indices = routing["family_indices"]
    if family_indices:
        index_tensor = torch.tensor(family_indices, dtype=torch.long)
        family_target_tensor = torch.tensor([int(family_targets[i]) for i in family_indices], dtype=torch.long)
        family_loss = torch.nn.CrossEntropyLoss()(
            outputs["family_logits"].index_select(0, index_tensor),
            family_target_tensor,
        )
    else:
        family_loss = outputs["family_logits"].sum() * 0.0

    localization_indices = routing["localization_indices"]
    if localization_indices:
        index_tensor = torch.tensor(localization_indices, dtype=torch.long)
        localization_loss = torch.nn.BCEWithLogitsLoss()(
            outputs["localization_logits"].index_select(0, index_tensor),
            mask_targets,
        )
    else:
        localization_loss = outputs["localization_logits"].sum() * 0.0

    total_loss = class_loss + family_loss_weight * family_loss + localization_loss_weight * localization_loss
    return {
        "class_loss": class_loss,
        "family_loss": family_loss,
        "localization_loss": localization_loss,
        "total_loss": total_loss,
        "routing": routing,
        "class_loss_finite": math.isfinite(float(class_loss.detach().item())),
        "family_loss_finite": math.isfinite(float(family_loss.detach().item())),
        "localization_loss_finite": math.isfinite(float(localization_loss.detach().item())),
        "total_loss_finite": math.isfinite(float(total_loss.detach().item())),
    }


def _confidence_map(labels: Iterable[str], probabilities: Sequence[float]) -> Dict[str, float]:
    return {label: float(probabilities[index]) for index, label in enumerate(labels)}


def build_schema_output_summary(
    torch: Any,
    outputs: Dict[str, Any],
    sample_index: int = 0,
    threshold_tau: float = 0.5,
    mask_area_pct: Optional[float] = None,
) -> Dict[str, object]:
    """Build and validate a schema-compatible output summary for one sample."""
    class_probs = torch.softmax(outputs["class_logits"][sample_index], dim=0).detach().cpu().tolist()
    family_probs = torch.softmax(outputs["family_logits"][sample_index], dim=0).detach().cpu().tolist()
    smoke_class = CLASS_LABELS[int(max(range(len(class_probs)), key=lambda idx: class_probs[idx]))]
    schema_class = schema_class_label(smoke_class)
    family = FAMILY_SMOKE_LABELS[int(max(range(len(family_probs)), key=lambda idx: family_probs[idx]))]
    tampered_score = float(class_probs[CLASS_TO_INDEX["tampered"]])

    if schema_class != "tampered":
        localization_head = LOCALIZATION_NOT_APPLICABLE
        area_pct = None
    elif tampered_score >= threshold_tau:
        localization_head = LOCALIZATION_ACTIVATED
        if mask_area_pct is None:
            mask_logits = outputs["localization_logits"][sample_index]
            area_pct = float((torch.sigmoid(mask_logits) >= 0.5).float().mean().item() * 100.0)
        else:
            area_pct = float(mask_area_pct)
    else:
        localization_head = LOCALIZATION_SKIPPED_BELOW_THRESHOLD
        area_pct = None

    class_conf = {
        "real": float(class_probs[CLASS_TO_INDEX["real"]]),
        "synthetic": float(class_probs[CLASS_TO_INDEX["full_synthetic"]]),
        "tampered": tampered_score,
    }
    family_conf = _confidence_map(FAMILY_SMOKE_LABELS, family_probs)
    evidence = []
    if schema_class == "synthetic":
        evidence.append({"signal_id": EVIDENCE_GLOBAL_SYNTHETIC_ARTIFACT, "description": "tiny smoke class signal"})
    if schema_class == "tampered":
        evidence.append({"signal_id": EVIDENCE_BOUNDARY_DISCONTINUITY, "description": "tiny smoke tampered signal"})
    if localization_head == LOCALIZATION_ACTIVATED:
        evidence.append({"signal_id": EVIDENCE_LOCAL_MASK_ACTIVATION, "description": "tiny smoke mask activation"})
    if family != "Real-or-N/A":
        evidence.append({"signal_id": EVIDENCE_PROVENANCE_FAMILY_SIGNAL, "description": "tiny smoke family signal"})

    summary: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "class": schema_class,
        "class_conf": class_conf,
        "family": family,
        "family_conf": family_conf,
        "localization_head": localization_head,
        "mask_area_pct": area_pct,
        "threshold_tau": float(threshold_tau),
        "tampered_score": tampered_score,
        "evidence": evidence,
        "perturbations": [PERTURBATION_NONE],
        "reason": "",
        "model_stage": "pre_sns_integrated_smoke",
        "family_policy": "family loss is conditional on available family labels",
    }
    summary["reason"] = generate_reason(summary)
    validate_forensic_output(summary)
    return summary
